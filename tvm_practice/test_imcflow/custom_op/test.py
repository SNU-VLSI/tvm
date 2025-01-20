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
from tvm.relay.op.nn.nn import imcflow_batch_norm
from tvm.relay.op.transform import imcflow_packing_test
from tvm.relay.op.transform import imcflow_unpacking_test
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
  func = relay.Function([data, fused_scale, fused_bias], y[0])
  
  input_data = np.ones((1, IC, IH, IW), dtype="int16")
  fused_scale_data = np.ones((IC,), dtype="int16")
  fused_bias_data = np.ones((IC,), dtype="int16")

  params = {"fused_scale": fused_scale_data, "fused_bias": fused_bias_data}

  target = "llvm"
  ctx = tvm.cpu(0)

  mod = tvm.IRModule.from_expr(func)
  mod = transform.InferType()(mod)
  print(mod)
  graph, lib, params = relay.build(mod, target=target, params=params)
  mod = graph_executor.create(graph, lib, device=ctx)
  mod.set_input(**params)
  mod.set_input(data=input_data)
  mod.run()

  res = mod.get_output(0).asnumpy()
  ref_res = np.full((1, IC, IH, IW), 2, dtype="int16")

  tvm.testing.assert_allclose(res, ref_res, atol=1e-5, rtol=1e-5)

  out = tvm.IRModule.from_expr(y[0])
  out = relay.transform.InferType()(out)
  print(out)

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

def test_imcflow_packing_1():
  IC = 7
  IH, IW = 1, 1

  data = relay.var("data", shape=(1, IC, IH, IW), dtype="float32")
  input_data = np.array([[[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[-1.0]], [[-2.0]], [[-1.0]]]], dtype="float32")
  print(input_data)
  newshape = relay.const(np.array([4], dtype="int32"), dtype="int32")
  y = imcflow_packing_test(data, newshape)
  func = relay.Function([data], y)

  target = "llvm"
  ctx = tvm.cpu(0)

  mod = tvm.IRModule.from_expr(func)
  mod = transform.InferType()(mod)
  print(mod)
  graph, lib, params = relay.build(mod, target=target)
  mod = graph_executor.create(graph, lib, device=ctx)
  mod.set_input(data=input_data)
  mod.run()

  res = mod.get_output(0).asnumpy()
  """Todo"""
  ref_res = np.array([33, 67, -17, 15], dtype="int8")

  tvm.testing.assert_allclose(res, ref_res, atol=1e-5, rtol=1e-5)

  out = tvm.IRModule.from_expr(y)
  out = relay.transform.InferType()(out)
  print(out)

def test_imcflow_packing_2():
  IC = 2
  IH, IW = 2, 2

  data = relay.var("data", shape=(1, IC, IH, IW), dtype="float32")
  input_data = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[-1.0, -2.0], [-1.0, 1.0]]]], dtype="float32")
  print(input_data)
  newshape = relay.const(np.array([4], dtype="int32"), dtype="int32")
  y = imcflow_packing_test(data, newshape)
  func = relay.Function([data], y)

  target = "llvm"
  ctx = tvm.cpu(0)

  mod = tvm.IRModule.from_expr(func)
  mod = transform.InferType()(mod)
  print(mod)
  graph, lib, params = relay.build(mod, target=target)
  mod = graph_executor.create(graph, lib, device=ctx)
  mod.set_input(data=input_data)
  mod.run()

  res = mod.get_output(0).asnumpy()
  """Todo"""
  ref_res = np.array([33, 67, -17, 31], dtype="int8")

  tvm.testing.assert_allclose(res, ref_res, atol=1e-5, rtol=1e-5)

  out = tvm.IRModule.from_expr(y)
  out = relay.transform.InferType()(out)
  print(out)

def test_imcflow_unpacking_1():
  IC = 7
  IH, IW = 1, 1

  data = relay.var("data", shape=(4,), dtype="int8")
  input_data = np.array([33, 67, -17, 15], dtype="int8")
  print(input_data)
  newshape = relay.const(np.array([1, IC, IH, IW], dtype="int32"), dtype="int32")
  y = imcflow_unpacking_test(data, newshape)
  func = relay.Function([data], y)

  target = "llvm"
  ctx = tvm.cpu(0)

  mod = tvm.IRModule.from_expr(func)
  mod = transform.InferType()(mod)
  print(mod)
  graph, lib, params = relay.build(mod, target=target)
  mod = graph_executor.create(graph, lib, device=ctx)
  mod.set_input(data=input_data)
  mod.run()

  res = mod.get_output(0).asnumpy()
  """Todo"""
  ref_res = np.array([[[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[-1.0]], [[-2.0]], [[-1.0]]]], dtype="float32")

  tvm.testing.assert_allclose(res, ref_res, atol=1e-5, rtol=1e-5)

  out = tvm.IRModule.from_expr(y)
  out = relay.transform.InferType()(out)
  print(out)

def test_imcflow_unpacking_2():
  IC = 2
  IH, IW = 2, 2

  data = relay.var("data", shape=(4,), dtype="int8")
  input_data = np.array([33, 67, -17, 31], dtype="int8")
  print(input_data)
  newshape = relay.const(np.array([1, IC, IH, IW], dtype="int32"), dtype="int32")
  y = imcflow_unpacking_test(data, newshape)
  func = relay.Function([data], y)

  target = "llvm"
  ctx = tvm.cpu(0)

  mod = tvm.IRModule.from_expr(func)
  mod = transform.InferType()(mod)
  print(mod)
  graph, lib, params = relay.build(mod, target=target)
  mod = graph_executor.create(graph, lib, device=ctx)
  mod.set_input(data=input_data)
  mod.run()

  res = mod.get_output(0).asnumpy()
  """Todo"""
  ref_res = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[-1.0, -2.0], [-1.0, 1.0]]]], dtype="float32")

  tvm.testing.assert_allclose(res, ref_res, atol=1e-5, rtol=1e-5)

  out = tvm.IRModule.from_expr(y)
  out = relay.transform.InferType()(out)
  print(out)

if __name__ == "__main__":
  tvm.testing.main()