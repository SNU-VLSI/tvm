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

from tvm.relay.backend.contrib.imcflow.acim_util import ConfigData
from tvm.relay.qnn.op.qnn import imcflow_min_max_quantize, imcflow_nu_quantize
from tvm.relay.op.nn import imcflow_batch_norm, imcflow_qconv2d

def get_height(H, KH, padding, stride):
    pad_h = padding
    out_h = (H + 2 * pad_h - KH) // stride + 1
    return out_h

def get_width(W, KW, padding, stride):
    pad_w = padding
    out_w = (W + 2 * pad_w - KW) // stride + 1
    return out_w

def getOneReluModel():
  N, C, H, W = 1, 28, 4, 4
  input_ = relay.var("input", shape=(N,C,H,W), dtype="int16")
  y = relay.nn.relu(input_)

  param_dict = { }

  out = tvm.IRModule.from_expr(y)

  return out, param_dict

def getOneConvModel():
  N, IC, H, W = 1, 28, 4, 4
  OC = 64
  KH, KW = 3, 3
  stride, padding = 1, 1
  # input = relay.var("conv_input", shape=(N,math.ceil(IC/256),H,W,4,8), dtype="int32")
  input = relay.var("conv_input", shape=(N,IC,H,W), dtype="uint8")

  y = imcflow_qconv2d(
    input,
    relay.var("conv_weight", shape=(OC,IC,KH,KW), dtype="int8"),
    ConfigData((N, IC, H, W), (OC, IC, KH, KW), padding=padding, stride=stride).get_as_const_tensor(),
    in_channels=IC,
    channels=OC,
    kernel_size=(KH, KW),
    padding=(padding, padding),
    out_dtype="int16"
  )

  param_dict = {"conv_weight": np.ones((OC,IC,KH,KW), dtype="int8")}

  out = tvm.IRModule.from_expr(y)

  return out, param_dict

def getResidualModel():
  N, IC, H, W = 1, 16, 1, 1
  y = relay.var("input", shape=(N,IC,H,W), dtype="uint8")
  y = imcflow_qconv2d(
    y,
    relay.var("weight2_1", shape=(16,16,3,3), dtype="int8"),
    ConfigData((N, IC, H, W), (16,16,3,3), padding=1, stride=1).get_as_const_tensor(),
    in_channels=16,
    channels=16,
    kernel_size=(3, 3),
    padding=(1, 1),
    out_dtype="int16"
  )
  residual = y
  IC, H, W = (16, get_height(H, 3, 1, 1), get_width(W, 3, 1, 1))

  y = imcflow_min_max_quantize(y, relay.var("quant_min_2", shape=(), dtype="int16"), relay.var("quant_max_2", shape=(), dtype="int16"), axis=1, out_dtype="uint8", channel=16)
  y = imcflow_qconv2d(
    y,
    relay.var("weight2_2", shape=(16,16,3,3), dtype="int8"),
    ConfigData((N, IC, H, W), (16,16,3,3), padding=1, stride=1).get_as_const_tensor(),
    in_channels=16,
    channels=16,
    kernel_size=(3, 3),
    padding=(1, 1),
    out_dtype="int16"
  )
  IC, H, W = (16, get_height(H, 3, 1, 1), get_width(W, 3, 1, 1))

  y = y + residual

  param_dict = {
    "weight2_1"  : np.ones((16,16,3,3), dtype="int8"),
    "weight2_2"  : np.ones((16,16,3,3), dtype="int8"),
    "quant_min_2": np.array(-128, dtype="int16"),
    "quant_max_2": np.array(127, dtype="int16"),
  }

  out = tvm.IRModule.from_expr(y)
  return out, param_dict