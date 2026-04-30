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
  Qconv2d with psum quantization, faithful to RTL atom-IC chunking.

  An IMCU column physically processes at most ``atom_IC = floor(256/(KH*KW))``
  input channels.  When ``in_channel`` exceeds ``atom_IC`` the IC reduction is
  split into ``num_atoms = ceil(in_channel/atom_IC)`` chunks; each chunk runs
  the full bit-plane / psum-quantize pipeline and produces an int16 partial
  sum.  The per-chunk int16 outputs are then accumulated in int32 and clipped
  to the int16 range at the very end — this mirrors what
  ``split_conv_to_atomic`` followed by saturating int16 add does at the relay
  level for the transformed reference path.  When ``in_channel <= atom_IC``
  (the post-split regime), ``num_atoms == 1`` and the cross-chunk reduction is
  a no-op, so the output is bit-exact with the pre-chunking implementation.

  Input is bit serial; weight bit planes are mapped into a column of IMCU.
  In each column, accumulate bit product and quantize at the end of the column.

  Args:
    acc_mask: Accumulation mask.  For each bit position b, if
              ``(acc_mask & (1 << b)) == 0``, accumulation mode is enabled and
              quantization is bypassed when the input bitplane has fewer than
              8 ones (counted *within* the atom chunk).  May be a compile-time
              int or a runtime te.Tensor scalar.

  Refer to imcflow simulator for more details.
  """
  batch, in_channel, IH, IW = input.shape
  out_channel, _, KH, KW = filter.shape
  OH = (IH - KH + 2 * padding[0]) // strides[0] + 1
  OW = (IW - KW + 2 * padding[1]) // strides[1] + 1

  # atom-IC chunking parameters (compile-time constants).
  in_channel_int = int(in_channel)
  KH_int = int(KH)
  KW_int = int(KW)
  atom_IC = 256 // (KH_int * KW_int)
  num_atoms = (in_channel_int + atom_IC - 1) // atom_IC
  padded_IC = num_atoms * atom_IC

  # Spatial padding (channel dim unchanged).
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

  # Channel-pad Apad and filter to padded_IC for clean chunk-aware indexing.
  # When in_channel == padded_IC the if_then_else always picks the data branch.
  ApadCh = te.compute(
      (batch, padded_IC, IH + 2 * padding[0], IW + 2 * padding[1]),
      lambda nn, cc, hh, ww: tvm.tir.if_then_else(
          cc < in_channel,
          Apad[nn, cc, hh, ww],
          tvm.tir.const(0, "uint8"),
      ),
      name="ApadCh",
  )
  FilterPad = te.compute(
      (out_channel, padded_IC, KH, KW),
      lambda ff, cc, ry_, rx_: tvm.tir.if_then_else(
          cc < in_channel,
          filter[ff, cc, ry_, rx_],
          tvm.tir.const(0, filter.dtype),
      ),
      name="FilterPad",
  )

  # Per-chunk reduction axes (rc reduces *within* a chunk of size atom_IC).
  rc = te.reduce_axis((0, atom_IC), name="rc")
  ry = te.reduce_axis((0, KH), name="ry")
  rx = te.reduce_axis((0, KW), name="rx")

  def get_bit(data, bit):
      return (data >> bit) & 1

  # Bit-wise convolution per atom chunk.
  # Shape: [batch, out_channel, OH, OW, num_atoms, input_bit, weight_bit]
  BitConv = te.compute(
      (batch, out_channel, OH, OW, num_atoms, 4, 4),
      lambda nn, ff, hh, ww, cc, bi, bw: te.sum(
          get_bit(ApadCh[nn, cc * atom_IC + rc,
                         hh * strides[0] + ry, ww * strides[1] + rx], bi).astype("int32") *
          get_bit(FilterPad[ff, cc * atom_IC + rc, ry, rx], bw).astype("int32"),
          axis=[rc, ry, rx]
      ),
      name="BitConv"
  )

  # Per-chunk popcount of input bitplane (for skip-quant decision).
  # Each IMCU column counts ones over its own atom slice.
  rc2 = te.reduce_axis((0, atom_IC), name="rc2")
  ry2 = te.reduce_axis((0, KH), name="ry2")
  rx2 = te.reduce_axis((0, KW), name="rx2")
  InputBitCount = te.compute(
      (batch, OH, OW, num_atoms, 4),
      lambda nn, hh, ww, cc, bi: te.sum(
          get_bit(ApadCh[nn, cc * atom_IC + rc2,
                         hh * strides[0] + ry2, ww * strides[1] + rx2], bi),
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

  # If acc_mode is enabled ((acc_mask & (1 << bi)) == 0) AND input bitplane is
  # sparse (< 8 ones within this atom chunk), skip quantization.
  #     skip_quant = (InputBitCount < 8) AND ((acc_mask & (1 << bi)) == 0)
  #     result = skip_quant * no_quant + (1 - skip_quant) * quant
  def compute_quantized_bitconv(nn, ff, hh, ww, cc, bi, bw):
      data_val = BitConv[nn, ff, hh, ww, cc, bi, bw]
      popcount_low = (InputBitCount[nn, hh, ww, cc, bi] < 8).astype("int32")
      acc_mode_enabled = ((acc_mask & (1 << bi)) == 0).astype("int32")
      skip_quant = popcount_low * acc_mode_enabled

      no_quant_val = data_val.astype("int16")
      quant_val = psum_quantize_expr(data_val, adcmode, vmode)

      return skip_quant * no_quant_val + (1 - skip_quant) * quant_val

  QuantizedBitConv = te.compute(
      (batch, out_channel, OH, OW, num_atoms, 4, 4),
      lambda nn, ff, hh, ww, cc, bi, bw: compute_quantized_bitconv(nn, ff, hh, ww, cc, bi, bw),
      name="QuantizedBitConv"
  )

  rb_w = te.reduce_axis((0, 4), name="rb_w")

  def get_w_scale(bw):
      # Weight bit 3 is sign bit (-8), others are 1, 2, 4.
      return (1 << bw) - 16 * (bw // 3)

  # Sum over weight bits for each input bit (int32 to preserve precision).
  WeightBitSum = te.compute(
      (batch, out_channel, OH, OW, num_atoms, 4),
      lambda nn, ff, hh, ww, cc, bi: te.sum(
          QuantizedBitConv[nn, ff, hh, ww, cc, bi, rb_w].astype("int32") * get_w_scale(rb_w),
          axis=[rb_w]
      ),
      name="WeightBitSum"
  )

  # Multiply by input scale and cast to int16 (wrapping), matching the
  # Python simulator's per-bitplane int16 accumulation.
  ScaledBitContrib = te.compute(
      (batch, out_channel, OH, OW, num_atoms, 4),
      lambda nn, ff, hh, ww, cc, bi: (WeightBitSum[nn, ff, hh, ww, cc, bi] * (1 << bi)).astype("int16"),
      name="ScaledBitContrib"
  )

  # Per-chunk int16 partial sum (sum over input bits, int16 wrap — same
  # semantics as the pre-chunking Output reduction).
  rb_in = te.reduce_axis((0, 4), name="rb_in")
  PerAtomOutput = te.compute(
      (batch, out_channel, OH, OW, num_atoms),
      lambda nn, ff, hh, ww, cc: te.sum(
          ScaledBitContrib[nn, ff, hh, ww, cc, rb_in],
          axis=[rb_in]
      ),
      name="PerAtomOutput"
  )

  # Cross-chunk saturating sum: int32 accumulate, single clip-to-int16 at
  # the end.  When num_atoms == 1 this reduces a single int16 cast to int32 —
  # the clip is a no-op and the final int16 cast is bit-exact with the old
  # implementation.
  rcc = te.reduce_axis((0, num_atoms), name="rcc")
  AtomSumI32 = te.compute(
      (batch, out_channel, OH, OW),
      lambda nn, ff, hh, ww: te.sum(
          PerAtomOutput[nn, ff, hh, ww, rcc].astype("int32"),
          axis=[rcc]
      ),
      name="AtomSumI32"
  )

  Output = te.compute(
      (batch, out_channel, OH, OW),
      lambda nn, ff, hh, ww: tvm.tir.Max(
          tvm.tir.Min(AtomSumI32[nn, ff, hh, ww], tvm.tir.const(32767, "int32")),
          tvm.tir.const(-32768, "int32")
      ).astype("int16"),
      name="Output"
  )

  if out_dtype is not None:
      Output = te.compute(Output.shape, lambda *i: Output(*i).astype(out_dtype), name="OutputCast")

  return Output


def imcflow_qdwconv2d(
    input : te.Tensor,
    filter : te.Tensor,
    strides, padding, dilation,
    adcmode=0, vmode=0,
    data_layout="NCHW", kernel_layout="", out_dtype=None
):
  """
  Depthwise convolution without psum quantization.

  Depthwise conv differs from standard conv:
  - Each input channel is convolved with its own filter (no cross-channel reduction)
  - Filter shape: (channels, 1, KH, KW)
  - Output channels = input channels

  Args:
    input: Input tensor of shape [N, C, H, W]
    filter: Filter tensor of shape [C, 1, KH, KW]
    strides: Stride values (tuple)
    padding: Padding values (tuple)
    dilation: Dilation values (tuple) - currently not used
    adcmode: ADC mode (not used in no_psum_quant version)
    vmode: Voltage mode (not used in no_psum_quant version)
  """
  batch, channels, IH, IW = input.shape
  _, _, KH, KW = filter.shape  # filter shape: (C, 1, KH, KW)
  OH = (IH - KH + 2 * padding[0]) // strides[0] + 1
  OW = (IW - KW + 2 * padding[1]) // strides[1] + 1

  # Pad input
  Apad = te.compute(
      (batch, channels, IH + 2 * padding[0], IW + 2 * padding[1]),
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

  # Create reduction variables (no channel reduction for depthwise)
  ry = te.reduce_axis((0, KH), name="ry")
  rx = te.reduce_axis((0, KW), name="rx")

  # Compute depthwise convolution
  # Each channel is convolved independently with its own filter
  B = te.compute(
      (batch, channels, OH, OW),
      lambda nn, cc, hh, ww: te.sum(
          Apad[nn, cc, hh * strides[0] + ry, ww * strides[1] + rx].astype("int16") *
          filter[cc, 0, ry, rx].astype("int16"),
          axis=[ry, rx]
      ),
      name="B",
  )

  return B
