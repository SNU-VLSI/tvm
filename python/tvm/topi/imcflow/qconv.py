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
          tvm.tir.const(0.0, "float32"),
      ),
      name="Apad",
  )

  # Create reduction variables
  rc = te.reduce_axis((0, in_channel), name="rc")
  ry = te.reduce_axis((0, KH), name="ry")
  rx = te.reduce_axis((0, KW), name="rx")

  # Compute the convolution
  B = te.compute(
      (batch, out_channel, OH, OW),
      lambda nn, ff, hh, ww: te.sum(
          Apad[nn, rc, hh * strides[0] + ry, ww * strides[1] + rx] * filter[ff, rc, ry, rx], axis=[rc, ry, rx]
      ),
      name="B",
  )

  return B