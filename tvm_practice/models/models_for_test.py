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