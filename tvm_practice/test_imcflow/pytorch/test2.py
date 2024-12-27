import torch
import tvm
import tvm.relay as relay
from typing import Sequence
from tvm.relay.qnn.op.qnn import imcflow_min_max_quantize, imcflow_nu_quantize
from tvm.relay.op.nn import imcflow_batch_norm
import numpy as np
import tvm.relay.op as _op
import tvm.relay.expr as _expr
from tvm.relay import pretty_print
from tvm.contrib.relay_viz import RelayVisualizer, DotPlotter, DotVizParser
from tvm.relay.build_module import bind_params_by_name
from tvm.relay import transform
from tvm.relay.backend.contrib.imcflow import transform as imcflow_transform
from tvm.relay.op.contrib import imcflow
from tvm.contrib.imcflow import ImcflowDeviceConfig

import math

@torch.library.custom_op("imcflow::min_max_quant", mutates_args=())
def min_max_quant(pic: torch.Tensor, min:int, max:int) -> torch.Tensor:
    return pic.clone()

@torch.library.custom_op("imcflow::linear_quant", mutates_args=())
def linear_quant(x:torch.Tensor, scale:float, zero_point:int) -> torch.Tensor:
  return x.clone()

def make_min_max(input, input_types):
  MinNDArray = tvm.runtime.ndarray.array(np.array(input[1], dtype=np.float32))
  MaxNDArray = tvm.runtime.ndarray.array(np.array(input[2], dtype=np.float32))
  return imcflow_min_max_quantize(input[0], tvm.relay.Constant(MinNDArray), tvm.relay.Constant(MaxNDArray), 1, "float32")

def make_quantize(input, input_types):
  scale = tvm.relay.Constant(tvm.runtime.ndarray.array(np.array(input[1], dtype=np.float32)))
  bias = tvm.relay.Constant(tvm.runtime.ndarray.array(np.array(input[2], dtype=np.int32)))
  return tvm.relay.qnn.op.quantize(input[0], scale, bias, 1, "float32")

def printModel(mod, param_dict, mod_name):
  RelayVisualizer(
    relay_mod = mod,
    relay_param = param_dict,
    plotter = DotPlotter(),
    parser = DotVizParser(),
  ).render(f"results/{mod_name}")

  with open(f"results/{mod_name}.txt", "w") as f:
    f.write(pretty_print(mod))

if __name__ == "__main__":
  x1 = relay.var("input", shape=(1, 64, 32, 32), dtype="float32")
  weight1 = relay.var("weight1", shape=(64, 64, 3, 3), dtype="float32")
  y = relay.nn.conv2d(
      x1,
      weight1,
      channels=64,
      kernel_size=(3, 3),
      padding=(1, 1),
  )

  bias = relay.var("bias1", shape=(64,), dtype="float32")
  y = relay.nn.bias_add(y, bias)

  fused_scale = relay.var("fused_scale", shape=(64,), dtype="float32")
  fused_bias = relay.var("fused_bias", shape=(64,), dtype="float32")
  y = imcflow_batch_norm(y, fused_scale, fused_bias, 1)[0]

  quant_scale = relay.var("quant_scale", shape=(64,), dtype="float32")
  quant_zero_point = relay.var("quant_zero_point", shape=(64,), dtype="int32")
  y = tvm.relay.qnn.quantize(y, quant_scale, quant_zero_point, out_dtype="int8", axis=1)

  y = relay.op.transform.imcflow_packing(y, [math.ceil(64*32*32/2)*2])
  y = relay.op.transform.imcflow_fake_tensor(y, [64, 32, 32], dtype="int8")

  mod = tvm.IRModule.from_expr(y)
  mod = relay.transform.InferType()(mod)

  # origin
  printModel(mod, {}, "origin")