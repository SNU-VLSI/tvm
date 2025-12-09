import itertools
import numpy as np
import sys
import subprocess
import math
import collections
import os
import argparse
import torch

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
from tvm.relay.op.nn.nn import imcflow_batch_norm
from tvm.relay.op.transform import imcflow_packing
from tvm.relay.op.transform import imcflow_unpacking
from tvm.relay.qnn.op.qnn import imcflow_min_max_quantize, imcflow_nu_quantize

from tvm.relay.backend.contrib.imcflow import transform as imcflow_transform
from tvm.relay.op.contrib import imcflow
from tvm.contrib.imcflow import ImcflowDeviceConfig
from tvm.relay.backend.contrib.imcflow.acim_util import ConfigData

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
  func = relay.Function([data, fused_scale, fused_bias], y)
  
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
  print(res)
  ref_res = np.full((1, IC, IH, IW), 2, dtype="int16")

  tvm.testing.assert_allclose(res, ref_res, atol=1e-5, rtol=1e-5)

  out = tvm.IRModule.from_expr(y)
  out = relay.transform.InferType()(out)
  print(out)

def test_min_max_quant():
  IC = 64
  IH, IW = 32, 32

  # data = relay.var("data", shape=(4, 2), dtype="float32")
  # min = relay.var("min", type_annotation=relay.TensorType([1], "int16"))
  # max = relay.var("max", type_annotation=relay.TensorType([1], "int16"))
  data = relay.var("data", shape=(1, IC, IH, IW), dtype="int16")
  min = relay.var("min", shape=(), dtype="int16")
  max = relay.var("max", shape=(), dtype="int16")
  y = imcflow_min_max_quantize(data, min, max, 1, "uint8", "int16")
  func = relay.Function([data, min, max], y)
  
  target = "llvm"
  ctx = tvm.cpu(0)

  # input_data = np.array([[1.2, 0.1], [0.7, 1.7], [-1.1, 5.1], [6.7, -4.6]], dtype="int16")
  # input_data = np.random.rand(1, IC, IH, IW).astype("int16")
  input_data = np.random.randint(0, 16, size=(1, IC, IH, IW)).astype("int16")
  min_data = np.int16(0)
  max_data = np.int16(15)

  params = {"min": min_data, "max": max_data}

  mod = tvm.IRModule.from_expr(func)
  mod = transform.InferType()(mod)
  print(mod)
  graph, lib, params = relay.build(mod, target=target, params=params)
  mod = graph_executor.create(graph, lib, device=ctx)
  mod.set_input(**params)
  mod.set_input(data=input_data)
  mod.run()

  res = mod.get_output(0).asnumpy()
  print(res)

  # Compute reference result using Python implementation for min_max_quantize_hw
  def min_max_quantize_hw_ref(data, min_val, max_val, bits):
    """Reference implementation of HW min-max quantization."""
    thresholds = []
    for i in range(2**bits - 1):
      offset = (i + 1) * (max_val - min_val)
      norm_offset = np.floor(offset / (2**bits))
      thresholds.append(min_val + norm_offset)
    
    return _quantize_with_thresholds_ref(data, thresholds)
  
  def _quantize_with_thresholds_ref(data, thresholds):
    """Reference implementation of threshold-based quantization."""
    data_int32 = data.astype(np.int32)
    result = np.zeros_like(data_int32, dtype=np.int32)
    updated = np.zeros_like(data, dtype=bool)
    
    for idx, thres in enumerate(thresholds):
      is_leq = (data_int32 <= thres)
      result[is_leq & ~updated] = idx
      updated = updated | is_leq
    
    result[~updated] = len(thresholds)
    return result.astype(np.uint16)
  
  bits = 4
  ref_res = min_max_quantize_hw_ref(input_data, min_data, max_data, bits)

  tvm.testing.assert_allclose(res, ref_res, atol=1e-5, rtol=1e-5)

  out = tvm.IRModule.from_expr(y)
  out = relay.transform.InferType()(out)
  print(out)

def test_nu_quant():
  IC = 64
  IH, IW = 32, 32

  data = relay.var("data", shape=(5, 2), dtype="float32")
  threshold = relay.var("threshold", shape=(15,), dtype="int16")
  y = imcflow_nu_quantize(data, threshold, 1, "float32", "int16")
  func = relay.Function([data, threshold], y)
  
  target = "llvm"
  ctx = tvm.cpu(0)

  input_data = np.array([[11.2, 24.1], [1.7, 0.7], [-5.1, 5.1], [7.7, -2.6], [-13.1, 12.1]], dtype="float32")
  threshold_data = np.array([-10, -7, -5, -1, 0, 2, 4, 5, 6, 10, 11, 16, 17, 18, 19], dtype="int16")

  params = {"threshold": threshold_data}

  mod = tvm.IRModule.from_expr(func)
  mod = transform.InferType()(mod)
  print(mod)
  graph, lib, params = relay.build(mod, target=target, params=params)
  mod = graph_executor.create(graph, lib, device=ctx)
  mod.set_input(**params)
  mod.set_input(data=input_data)
  mod.run()

  res = mod.get_output(0).asnumpy()
  print(res)
  ref_res = np.array([[10.0, 15.0], [5.0, 4.0], [2.0, 7.0], [9.0, 3.0], [0.0, 11.0]], dtype="float32")

  tvm.testing.assert_allclose(res, ref_res, atol=1e-5, rtol=1e-5)

  out = tvm.IRModule.from_expr(y)
  out = relay.transform.InferType()(out)
  print(out)

def test_imcflow_packing_1():
  IC = 7
  IH, IW = 1, 1

  data = relay.var("data", shape=(1, IC, IH, IW), dtype="float32")
  input_data = np.array([[[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[-1.0]], [[-2.0]], [[-1.0]]]], dtype="float32")
  newshape = relay.const(np.array([4], dtype="int32"), dtype="int32")
  y = imcflow_packing(data, newshape, "int8")
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
  newshape = relay.const(np.array([4], dtype="int32"), dtype="int32")
  y = imcflow_packing(data, newshape, "int8")
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
  newshape = relay.const(np.array([1, IC, IH, IW], dtype="int32"), dtype="int32")
  y = imcflow_unpacking(data, newshape, "float32")
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
  newshape = relay.const(np.array([1, IC, IH, IW], dtype="int32"), dtype="int32")
  y = imcflow_unpacking(data, newshape, "float32")
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

def test_imcflow_qconv2d():
  N, H, W = 2, 16, 16
  IC, OC = 32, 32
  kH, kW = 3, 3
  padding, stride = 1, 1

  input_ = relay.var("input", shape=(N, IC, H, W), dtype="uint8")
  weight_ = relay.var("weight", shape=(OC, IC, kH, kW), dtype="int8")
  config_ = ConfigData((N, IC, H, W), (OC, IC, kH, kW), padding=padding, stride=stride).get_as_const_tensor()

  y = imcflow_qconv2d(
    input_,
    weight_,
    config_,
    channels=OC,
    in_channels=IC,
    kernel_size=(kH, kW),
    padding=(padding, padding),
  )

  func = relay.Function([input_, weight_], y)
  mod = tvm.IRModule.from_expr(func)
  mod = transform.InferType()(mod)

  print(mod)

  target = "llvm"
  ctx = tvm.cpu(0)

  with tvm.transform.PassContext(opt_level=0):
    graph, lib, params = relay.build(mod, target=target)
  mod = graph_executor.create(graph, lib, device=ctx)
  # Load constant parameters (config)
  if params:
    mod.load_params(tvm.runtime.save_param_dict(params))


  input_data = np.random.randint(0, 16, size=(2, 32, 16, 16)).astype("uint8")
  weight_data = np.random.randint(-8, 7, size=(32, 32, 3, 3)).astype("int8")

  mod.set_input(input=input_data)
  mod.set_input(weight=weight_data)
  mod.run()

  res = mod.get_output(0).asnumpy()
  ref_res = torch.functional.F.conv2d(
    input=torch.tensor(input_data, dtype=torch.float32),
    weight=torch.tensor(weight_data, dtype=torch.float32),
    padding=(padding, padding),
    stride=(stride, stride)
  ).numpy()

  tvm.testing.assert_allclose(res, ref_res, atol=1e-5, rtol=1e-5)


def test_packing_unpacking():
  IC = 2
  IH, IW = 2, 2

  data = relay.var("data", shape=(1, IC, IH, IW), dtype="float32")
  input_data = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[-1.0, -2.0], [-1.0, 1.0]]]], dtype="float32")
  newshape = relay.const(np.array([4], dtype="int32"), dtype="int32")
  y = imcflow_packing(data, newshape, "int8")
  y = imcflow_unpacking(y, relay.const(np.array([1, IC, IH, IW], dtype="int32"), dtype="int32"), "float32")
  func = relay.Function([data], y)
  mod = tvm.IRModule.from_expr(func)
  mod = transform.InferType()(mod)
  print(mod)

  target = "llvm"
  ctx = tvm.cpu(0)
  graph, lib, params = relay.build(mod, target=target)
  mod = graph_executor.create(graph, lib, device=ctx)
  mod.set_input(data=input_data)
  mod.run()

  res = mod.get_output(0).asnumpy()
  ref_res = np.array([[[[1.0, 2.0], [3.0, 4.0]], [[-1.0, -2.0], [-1.0, 1.0]]]], dtype="float32")

  tvm.testing.assert_allclose(res, ref_res, atol=1e-5, rtol=1e-5)

  out = tvm.IRModule.from_expr(y)
  out = relay.transform.InferType()(out)
  print(out)
  


if __name__ == "__main__":
  tvm.testing.main()