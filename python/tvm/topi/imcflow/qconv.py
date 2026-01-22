import numpy as np
import tvm
from tvm import te

def imcflow_qconv2d_no_psum_quant(
    input : te.Tensor,
    filter : te.Tensor,
    strides, padding, dilation, 
    adcmode=0, vmode=0,
    data_layout="NCHW", kernel_layout="", out_dtype=None
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

def imcflow_qconv2d(
    input : te.Tensor,
    filter : te.Tensor,
    strides, padding, dilation,
    adcmode, vmode, acc_mask,
    data_layout="NCHW", kernel_layout="", out_dtype=None
):
  """
  Qconv2d with psum quantization.
  Input is bit serial.
  weight bit planes are mapped into a column of IMCU.
  In each column, accumulate bit product and quantize at the end of the column.

  Args:
    acc_mask: Accumulation mask. Can be either:
              - An integer (compile-time constant): For each bit position b, if (acc_mask & (1 << b)) == 0,
                accumulation mode is enabled.
              - A te.Tensor scalar (runtime value): Same semantics but evaluated at runtime.
              When accumulation mode is enabled and the input bitplane has fewer than 8 ones,
              quantization is bypassed.

  Note: The current implementation uses arithmetic instead of if_then_else, which allows
        acc_mask to be a runtime parameter in the future. However, currently acc_mask is
        still expected to be a compile-time constant.

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

  # Count ones in each input bitplane for conditional quantization
  # Shape: [batch, OH, OW, input_bit]
  rc2 = te.reduce_axis((0, in_channel), name="rc2")
  ry2 = te.reduce_axis((0, KH), name="ry2")
  rx2 = te.reduce_axis((0, KW), name="rx2")
  InputBitCount = te.compute(
      (batch, OH, OW, 4),
      lambda nn, hh, ww, bi: te.sum(
          get_bit(Apad[nn, rc2, hh * strides[0] + ry2, ww * strides[1] + rx2], bi),
          axis=[rc2, ry2, rx2]
      ),
      name="InputBitCount"
  )

  def psum_quantize_expr(data, adcmode, vmode):
      adc_divider = 2**(2 + adcmode - vmode)
      val = data.astype("float32")
      val = val / adc_divider
      val = tvm.tir.round(val + 0.01)
      val = tvm.tir.Max(tvm.tir.Min(val, 63.0), 0.0)
      val = val * adc_divider
      return val.astype("int16")

  # Quantize each bitplane result with conditional quantization
  # If acc_mode is enabled (acc_mask & (1 << bi)) == 0) AND input bitplane is sparse (< 8 ones),
  # skip quantization
  #
  # Instead of if_then_else, use arithmetic to support runtime acc_mask:
  # skip_quant = (InputBitCount < 8) AND ((acc_mask & (1 << bi)) == 0)
  # result = skip_quant * no_quant + (1 - skip_quant) * quant
  def compute_quantized_bitconv(nn, ff, hh, ww, bi, bw):
      data_val = BitConv[nn, ff, hh, ww, bi, bw]

      # Compute skip_quant condition as integer (0 or 1)
      popcount_low = (InputBitCount[nn, hh, ww, bi] < 8).astype("int32")
      acc_mode_enabled = ((acc_mask & (1 << bi)) == 0).astype("int32")
      skip_quant = popcount_low * acc_mode_enabled

      # Compute both paths
      no_quant_val = data_val.astype("int16")
      quant_val = psum_quantize_expr(data_val, adcmode, vmode)

      # Select based on condition: skip_quant * no_quant + (1 - skip_quant) * quant
      return skip_quant * no_quant_val + (1 - skip_quant) * quant_val

  QuantizedBitConv = te.compute(
      (batch, out_channel, OH, OW, 4, 4),
      lambda nn, ff, hh, ww, bi, bw: compute_quantized_bitconv(nn, ff, hh, ww, bi, bw),
      name="QuantizedBitConv"
  )

  # Combine results
  # To match RTL/Python simulator behavior, we need to accumulate in int16 with wrapping
  # at each input bitplane iteration. The Python simulator does:
  #   post_result = np.zeros(64, dtype=np.int16)
  #   for b in range(4):
  #       bp_result = sum over weight bits of (pq_bitline * w_scale)
  #       post_result += bp_result * (2**b)  # wraps to int16 each iteration
  #
  # Step 1: Sum over weight bits for each input bit (in int32 to preserve precision)
  rb_w = te.reduce_axis((0, 4), name="rb_w")

  def get_w_scale(bw):
      # Weight bit 3 is sign bit (-8), others are 1, 2, 4
      return (1 << bw) - 16 * (bw // 3)

  # Shape: [batch, out_channel, OH, OW, 4 (input_bits)]
  WeightBitSum = te.compute(
      (batch, out_channel, OH, OW, 4),
      lambda nn, ff, hh, ww, bi: te.sum(
          QuantizedBitConv[nn, ff, hh, ww, bi, rb_w].astype("int32") * get_w_scale(rb_w),
          axis=[rb_w]
      ),
      name="WeightBitSum"
  )

  # Step 2: Multiply by input scale and cast to int16 (wrapping)
  # This matches: bp_result * (2**b) then truncate to int16
  ScaledBitContrib = te.compute(
      (batch, out_channel, OH, OW, 4),
      lambda nn, ff, hh, ww, bi: (WeightBitSum[nn, ff, hh, ww, bi] * (1 << bi)).astype("int16"),
      name="ScaledBitContrib"
  )

  # Step 3: Sum the 4 int16 contributions (int16 sum with wrapping)
  rb_in = te.reduce_axis((0, 4), name="rb_in")
  Output = te.compute(
      (batch, out_channel, OH, OW),
      lambda nn, ff, hh, ww: te.sum(
          ScaledBitContrib[nn, ff, hh, ww, rb_in],
          axis=[rb_in]
      ),
      name="Output"
  )

  if out_dtype is not None:
      Output = te.compute(Output.shape, lambda *i: Output(*i).astype(out_dtype), name="OutputCast")

  return Output