import itertools
import numpy as np
import sys
import subprocess
import math
import collections
import os
import argparse

from tvm.relay.backend import te_compiler
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing.temp_op_attr import TempOpAttr
import tvm.testing
from tvm.contrib import utils
from tvm import runtime as tvm_runtime
from tvm.contrib import graph_executor

from tvm.relay.backend import Executor, Runtime
from tvm.relay import pretty_print
from tvm.relay.op.nn.nn import imcflow_batch_norm, imcflow_qconv2d
from tvm.relay.qnn.op.qnn import imcflow_min_max_quantize, imcflow_nu_quantize

from tvm.relay.backend.contrib.imcflow import transform as imcflow_transform
from tvm.relay.op.contrib import imcflow
from tvm.contrib.imcflow import ImcflowDeviceConfig

from models import *

np.random.seed(0)

def get_graph(IC, IH, IW, OC, KH, KW):
    x1 = relay.var("x1", shape=(1, IC, IH, IW))
    bias = relay.var("bias", shape=(OC,))
    weight = relay.var("weight", shape=(OC, IC, KH, KW))
    y = relay.nn.conv2d(
        x1,
        weight,
        channels=OC,
        kernel_size=(KH, KW),
        padding=(1, 1),
    )
    y = relay.nn.bias_add(y, bias)
    y = relay.nn.relu(y)
    y = relay.nn.global_max_pool2d(y)
    dic = {
        "x1": (1, IC, IH, IW),
        "weight": (OC, IC, KH, KW),
        "bias": (OC,),
    }
    param_lst = ["weight", "bias"]
    out = tvm.IRModule.from_expr(y)
    return out, dic, param_lst

def test_batchnorm():
  IC = 64
  IH, IW = 32, 32

  data = relay.var("data", shape=(1, IC, IH, IW), dtype="int16")
  fused_scale = relay.var("fused_scale", shape=(IC,), dtype="int16")
  fused_bias = relay.var("fused_bias", shape=(IC,), dtype="int16")
  y = imcflow_batch_norm(data, fused_scale, fused_bias, 1)
  out = tvm.IRModule.from_expr(y[0])
  out = relay.transform.InferType()(out)
  print(out)

if __name__ == "__main__":
  tvm.testing.main()

def test_min_max_quant():
  IC = 64
  IH, IW = 32, 32

  data = relay.var("data", shape=(1, IC, IH, IW), dtype="int16")
  # min = relay.var("min", type_annotation=relay.TensorType([1], "int16"))
  # max = relay.var("max", type_annotation=relay.TensorType([1], "int16"))
  min = relay.var("min", shape=(), dtype="int16")
  max = relay.var("max", shape=(), dtype="int16")
  y = imcflow_min_max_quantize(data, min, max, 1, "int4")
  out = tvm.IRModule.from_expr(y)
  out = relay.transform.InferType()(out)
  print(out)

def test_nu_quant():
  IC = 64
  IH, IW = 32, 32

  data = relay.var("data", shape=(1, IC, IH, IW), dtype="int16")
  threshold = relay.var("threshold", shape=(16,), dtype="int16")
  y = imcflow_nu_quantize(data, threshold, 1, "int4")
  out = tvm.IRModule.from_expr(y)
  out = relay.transform.InferType()(out)
  print(out)

def test_qconv():
  IC = 64
  OC =64
  IH, IW = 32, 32

  data = relay.var("data", shape=(1, IC, IH, IW), dtype="int4")
  weight = relay.var("weight", shape=(OC, IC, 3, 3), dtype="int4")
  y = imcflow_qconv2d(data, weight)
  out = tvm.IRModule.from_expr(y)
  out = relay.transform.InferType()(out)
  print(out)

if __name__ == "__main__":
  tvm.testing.main()