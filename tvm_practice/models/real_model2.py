import pytest
import itertools
import numpy as np
import sys
import subprocess
import math
import collections
import os

from tvm.relay.backend import te_compiler
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.op.contrib import imcflow
import tvm.testing
from tvm.contrib import utils, graph_executor
from tvm import runtime as tvm_runtime

from tvm.relay.qnn.op.qnn import imcflow_min_max_quantize, imcflow_nu_quantize
from tvm.relay.op.nn import imcflow_batch_norm, imcflow_qconv2d

def getModel():
  # input_ = relay.var("input", shape=(1, 28, 56, 56))
  # FIXME: smaller input size for testing purpose
  input_ = relay.var("input", shape=(1, 28, 16, 16))

  y = relay.nn.conv2d(
      input_,
      relay.var("weight1", shape=(28, 28, 3, 3)),
      channels=28,
      kernel_size=(3, 3),
      padding=(1, 1),
  )

  y = relay.nn.bias_add(y, relay.var("bias1", shape=(28,)))
  y = imcflow_batch_norm(y, relay.var("fused_scale1", shape=(28,)), relay.var("fused_bias1", shape=(28,)))[0]
  y = relay.nn.relu(y)

  y = relay.qnn.simulated_quantize(y, relay.var("quant_scale", shape=(28,)), relay.var("quant_zp", shape=(28,), dtype="int32"), axis=1)

  y = imcflow_qconv2d(
    y,
    relay.var("weight2_0", shape=(64,28,3,3)),
    channels=64,
    kernel_size=(3, 3),
    padding=(1, 1),
  )

  y = imcflow_batch_norm(y, relay.var("fused_scale2", shape=(64,)), relay.var("fused_bias2", shape=(64,)))[0]
  y = relay.nn.relu(y)
  y = imcflow_min_max_quantize(y, relay.const(0.0, "float32"), relay.const(1.0, "float32"), 1, "float32")

  param_dict = {
    "quant_scale": np.random.rand(28).astype("float32"),
    "quant_zp": np.random.randint(0, 255, 28).astype("int"),
    "weight1": np.random.rand(28, 28, 3, 3).astype("float32"),
    "bias1": np.random.rand(28).astype("float32"),
    "fused_scale1": np.random.rand(28).astype("float32"),
    "fused_bias1": np.random.rand(28).astype("float32"),
    "weight2_0": np.random.rand(64,28,3,3).astype("float32"),
    "fused_scale2": np.random.rand(64).astype("float32"),
    "fused_bias2": np.random.rand(64).astype("float32"),
  }

  out = tvm.IRModule.from_expr(y)

  return out, param_dict

def getOneConvModel():
  N, IC, IH, IW = 2, 32, 4, 4
  OC, KH, KW = 128, 3, 3
  padding = (0, 0)
  stride = (1, 1)
  OH, OW = (IH - KH + 2 * padding[0]) // stride[0] + 1, (IW - KW + 2 * padding[1]) // stride[1] + 1

  atom_IC = math.floor(256/(KH*KW))
  atom_OC = 64
  ic_gnum = math.ceil(IC/atom_IC)
  oc_gnum = math.ceil(OC/atom_OC)

  input = relay.var("input", shape=(N,ic_gnum,IH,IW,4,8), dtype="int32")
  y = imcflow_qconv2d(
    input,
    relay.var("weight", shape=(oc_gnum,ic_gnum,256,8), dtype="int32"),
    channels=OC,
    in_channels=IC,
    kernel_size=(KH, KW),
    padding=padding,
    strides=stride,
    out_dtype="int16"
  )

  weight_numpy = np.random.rand(oc_gnum,ic_gnum,256,8).astype("int32")
  print(weight_numpy.dtype)
  param_dict = {
    "weight": weight_numpy
  }

  out = tvm.IRModule.from_expr(y)

  return out, param_dict

def getOneReluModel():
  # input_ = relay.var("input", shape=(1, 28, 4, 4))
  input_ = relay.var("input", shape=(1,28,4,4), dtype="int16")
  y = relay.nn.pad(input_, pad_width=((0, 0),(0, 0),(1, 1),(1,1)), pad_value=0)
  y = relay.nn.relu(y)

  param_dict = {
    # "quant_scale": np.random.rand(28).astype("float32"),
    # "quant_zp": np.random.randint(0, 255, 28).astype("int"),
    # "weight": np.random.rand(28,28,3,3).astype("float32")
  }


  out = tvm.IRModule.from_expr(y)

  return out, param_dict