import numpy as np
import tvm
from tvm import te

def imcflow_qconv2d(
    input : te.Tensor,
    filter : te.Tensor,
    strides, padding, dilation, data_layout="NCHW", kernel_layout="", out_dtype=None
):

  batch, in_channel, IH, IW = input.shape
  out_channel, _, KH, KW = filter.shape
  OH, OW = (IH - KH + 2 * padding[0]) // strides[0] + 1, (IW - KW + 2 * padding[1]) // strides[1] + 1

  # Pad input
  Apad = te.compute(
      (batch, in_channel, IH + 2 * padding[0], IW + 2 * padding[1]),
      lambda nn, cc, hh, ww: tvm.tir.if_then_else(
          tvm.tir.all(
            hh >= padding[0],
            (hh - padding[0]) < IH,
            ww >= padding[1], 
            (ww - padding[1]) < IW
          ),
          input[nn, cc, hh - padding[0], ww - padding[1]],
          tvm.tir.const(0, "uint8"),
      ),
      name="Apad",
  )

  # Create reduction variables
  rc = te.reduce_axis((0, in_channel), name="rc")
  ry = te.reduce_axis((0, KH), name="ry")
  rx = te.reduce_axis((0, KW), name="rx")

  # Compute the convolution
  # Cast to int16 for accumulation to prevent overflow
  B = te.compute(
      (batch, out_channel, OH, OW),
      lambda nn, ff, hh, ww: te.sum(
          Apad[nn, rc, hh * strides[0] + ry, ww * strides[1] + rx].astype("int16") *
          filter[ff, rc, ry, rx].astype("int16"),
          axis=[rc, ry, rx]
      ),
      name="B",
  )

  return B

def imcflow_psum_qunat_qconv2d(
    input : te.Tensor,
    filter : te.Tensor,
    strides, padding, dilation, 
    adcmode, vmode,
    data_layout="NCHW", kernel_layout="", out_dtype=None
):
  """
  Qconv2d with psum quantization.
  Input is bit serial.
  weight bit planes are mapped into a column of IMCU.
  In each column, accumulate bit product and quantize at the end of the column.
  Refer to imcflow simulator for more details.
  """
  batch, in_channel, IH, IW = input.shape
  out_channel, _, KH, KW = filter.shape
  OH, OW = (IH - KH + 2 * padding[0]) // strides[0] + 1, (IW - KW + 2 * padding[1]) // strides[1] + 1

  # Pad input
  Apad = te.compute(
      (batch, in_channel, IH + 2 * padding[0], IW + 2 * padding[1]),
      lambda nn, cc, hh, ww: tvm.tir.if_then_else(
          tvm.tir.all(
            hh >= padding[0],
            (hh - padding[0]) < IH,
            ww >= padding[1], 
            (ww - padding[1]) < IW
          ),
          input[nn, cc, hh - padding[0], ww - padding[1]],
          tvm.tir.const(0, "uint8"),
      ),
      name="Apad",
  )

  # Create reduction variables
  rc = te.reduce_axis((0, in_channel), name="rc")
  ry = te.reduce_axis((0, KH), name="ry")
  rx = te.reduce_axis((0, KW), name="rx")

  def get_bit(data, bit):
      return (data >> bit) & 1

  # Bit-wise convolution
  # Shape: [batch, out_channel, OH, OW, input_bit, weight_bit]
  BitConv = te.compute(
      (batch, out_channel, OH, OW, 4, 4),
      lambda nn, ff, hh, ww, bi, bw: te.sum(
          get_bit(Apad[nn, rc, hh * strides[0] + ry, ww * strides[1] + rx], bi).astype("int32") *
          get_bit(filter[ff, rc, ry, rx], bw).astype("int32"),
          axis=[rc, ry, rx]
      ),
      name="BitConv"
  )

  def psum_quantize_expr(data, adcmode, vmode):
      adc_divider = 2**(2 + adcmode - vmode)
      val = data.astype("float32")
      val = val / adc_divider
      val = tvm.tir.round(val + 0.01)
      val = tvm.tir.Max(tvm.tir.Min(val, 63.0), 0.0)
      val = val * adc_divider
      return val.astype("int16")

  # Quantize each bitplane result
  QuantizedBitConv = te.compute(
      (batch, out_channel, OH, OW, 4, 4),
      lambda nn, ff, hh, ww, bi, bw: psum_quantize_expr(BitConv[nn, ff, hh, ww, bi, bw], adcmode, vmode),
      name="QuantizedBitConv"
  )

  # Combine results
  rb_in = te.reduce_axis((0, 4), name="rb_in")
  rb_w = te.reduce_axis((0, 4), name="rb_w")

  def get_scale(bi, bw):
      # Weight bit 3 is sign bit (-8)
      w_scale = tvm.tir.if_then_else(bw == 3, -8, 1 << bw)
      in_scale = 1 << bi
      return w_scale * in_scale

  Output = te.compute(
      (batch, out_channel, OH, OW),
      lambda nn, ff, hh, ww: te.sum(
          QuantizedBitConv[nn, ff, hh, ww, rb_in, rb_w].astype("int32") * get_scale(rb_in, rb_w),
          axis=[rb_in, rb_w]
      ),
      name="Output"
  )

  if out_dtype is not None:
      Output = te.compute(Output.shape, lambda *i: Output(*i).astype(out_dtype), name="OutputCast")

  return Output