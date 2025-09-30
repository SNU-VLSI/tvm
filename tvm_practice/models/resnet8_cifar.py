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

def getModel_(input_shape):
  input = relay.var("input", shape=input_shape, dtype="float32")

  y = relay.nn.conv2d(
      input,
      relay.var("weight1", shape=(16, 3, 3, 3), dtype="float32"),
      in_channels=3,
      channels=16,
      kernel_size=(3, 3),
      padding=(1, 1),
  )

  y = relay.nn.batch_norm(y, 
                          relay.var("bn_gamma", shape=(16,), dtype="float32"), relay.var("bn_beta", shape=(16,), dtype="float32"), 
                          relay.var("bn_moving_mean", shape=(16,), dtype="float32"), relay.var("bn_moving_var", shape=(16,), dtype="float32"))[0]
  
  y = y * relay.var("x_f_1", shape=(1,), dtype="float32")
  y = relay.cast(y, dtype="int16")

  # basic block 1
  residual = y
  y = imcflow_min_max_quantize(y, relay.var("quant_min_1", shape=(), dtype="int16"), relay.var("quant_max_1", shape=(), dtype="int16"), axis=1, out_dtype="int4")
  y = imcflow_qconv2d(
    y,
    relay.var("weight2_1", shape=(16,16,3,3), dtype="int8"),
    in_channels=16,
    channels=16,
    kernel_size=(3, 3),
    padding=(1, 1),
    out_dtype="int16"
  )
  y = imcflow_batch_norm(y, relay.var("fused_scale1", shape=(16,), dtype="int16"), relay.var("fused_bias1", shape=(16,), dtype="int16"))[0]
  y = imcflow_min_max_quantize(y, relay.var("quant_min_2", shape=(), dtype="int16"), relay.var("quant_max_2", shape=(), dtype="int16"), axis=1, out_dtype="int4")
  y = imcflow_qconv2d(
    y,
    relay.var("weight2_2", shape=(16,16,3,3), dtype="int8"),
    in_channels=16,
    channels=16,
    kernel_size=(3, 3),
    padding=(1, 1),
    out_dtype="int16"
  )
  y = imcflow_batch_norm(y, relay.var("fused_scale2", shape=(16,), dtype="int16"), relay.var("fused_bias2", shape=(16,), dtype="int16"))[0]
  y = y + residual * relay.var("y_f_1", shape=(1,), dtype="int16")

  # basic block 2
  residual = y
  y = imcflow_min_max_quantize(y, relay.var("quant_min_3", shape=(), dtype="int16"), relay.var("quant_max_3", shape=(), dtype="int16"), axis=1, out_dtype="int4")
  y = imcflow_qconv2d(
    y,
    relay.var("weight3_1", shape=(32,16,3,3), dtype="int8"),
    in_channels=16,
    channels=32,
    kernel_size=(3, 3),
    padding=(1, 1),
    strides=(2,2),
    out_dtype="int16"
  )
  y = imcflow_batch_norm(y, relay.var("fused_scale3", shape=(32,), dtype="int16"), relay.var("fused_bias3", shape=(32,), dtype="int16"))[0]
  y = imcflow_min_max_quantize(y, relay.var("quant_min_4", shape=(), dtype="int16"), relay.var("quant_max_4", shape=(), dtype="int16"), axis=1, out_dtype="int4")
  y = imcflow_qconv2d(
    y,
    relay.var("weight3_2", shape=(32,32,3,3), dtype="int8"),
    in_channels=32,
    channels=32,
    kernel_size=(3, 3),
    padding=(1, 1),
    out_dtype="int16"
  )
  y = imcflow_batch_norm(y, relay.var("fused_scale4", shape=(32,), dtype="int16"), relay.var("fused_bias4", shape=(32,), dtype="int16"))[0]

  y_residual = imcflow_min_max_quantize(residual, relay.var("quant_min_4_2", shape=(), dtype="int16"), relay.var("quant_max_4_2", shape=(), dtype="int16"), axis=1, out_dtype="int4")
  y_residual = imcflow_qconv2d(
    y_residual,
    relay.var("weight3_0", shape=(32,16,1,1), dtype="int8"),
    in_channels=16,
    channels=32,
    kernel_size=(1, 1),
    strides=(2,2),
    out_dtype="int16"
  )
  y_residual = relay.var("bn_out_f_1", shape=(32,1,1), dtype="int16") * y_residual + relay.var("bn_out_f_0", shape=(32,1,1), dtype="int16")
  y = y + y_residual

  # basic block 3
  residual = y
  y = imcflow_min_max_quantize(y, relay.var("quant_min_5", shape=(), dtype="int16"), relay.var("quant_max_5", shape=(), dtype="int16"), axis=1, out_dtype="int4")
  y = imcflow_qconv2d(
    y,
    relay.var("weight4_1", shape=(64,32,3,3), dtype="int8"),
    in_channels=32,
    channels=64,
    kernel_size=(3, 3),
    padding=(1, 1),
    strides=(2,2),
    out_dtype="int16"
  )
  y = imcflow_batch_norm(y, relay.var("fused_scale5", shape=(64,), dtype="int16"), relay.var("fused_bias5", shape=(64,), dtype="int16"))[0]
  y = imcflow_min_max_quantize(y, relay.var("quant_min_6", shape=(), dtype="int16"), relay.var("quant_max_6", shape=(), dtype="int16"), axis=1, out_dtype="int4")
  y = imcflow_qconv2d(
    y,
    relay.var("weight4_2", shape=(64,64,3,3), dtype="int8"),
    in_channels=64,
    channels=64,
    kernel_size=(3, 3),
    padding=(1, 1),
    out_dtype="int16"
  )
  y = imcflow_batch_norm(y, relay.var("fused_scale6", shape=(64,), dtype="int16"), relay.var("fused_bias6", shape=(64,), dtype="int16"))[0]

  y_residual = imcflow_min_max_quantize(residual, relay.var("quant_min_6_2", shape=(), dtype="int16"), relay.var("quant_max_6_2", shape=(), dtype="int16"), axis=1, out_dtype="int4")
  y_residual = imcflow_qconv2d(
    y_residual,
    relay.var("weight4_0", shape=(64,32,1,1), dtype="int8"),
    in_channels=32,
    channels=64,
    kernel_size=(1, 1),
    strides=(2,2),
    out_dtype="int16"
  )
  y_residual = relay.var("bn_out_f_3", shape=(64,1,1), dtype="int16") * y_residual + relay.var("bn_out_f_2", shape=(64,1,1), dtype="int16")

  y = y + y_residual

  # post process
  y = relay.cast(y,dtype="float32") / relay.var("post_f", shape=(1,), dtype="float32")
  y = relay.nn.relu(y)
  y = relay.nn.adaptive_avg_pool2d(y, output_size=(1,1))
  y = relay.nn.batch_flatten(y) 
  y = relay.nn.dense(y, relay.var("dense_weight", shape=(10, 64), dtype="float32"))
  y = relay.nn.bias_add(y, relay.var("dense_bias", shape=(10,), dtype="float32"))

  # Collect parameter vars from the graph (exclude the input var)
  free_vars = relay.analysis.free_vars(y)
  var_info = {}
  for v in free_vars:
    if v is input:
      continue
    name = v.name_hint
    # Deduplicate by name in case of separately-constructed Vars with the same name
    if name in var_info:
      continue
    ttype = v.type_annotation
    if isinstance(ttype, relay.ty.TensorType):
      # Convert TVM shape (IntImm / PrimExpr) to Python ints when possible
      shape = []
      for dim in ttype.shape:
        try:
          shape.append(int(dim))
        except Exception:
          # Fallback if dynamic: leave as-is
          shape.append(dim)
      var_info[name] = {"shape": tuple(shape), "dtype": ttype.dtype}
    else:
      # If no TensorType annotation, skip or set defaults
      continue

  out = tvm.IRModule.from_expr(y)

  return out, var_info

def getModel():
  out, var_dict = getModel_([1, 3, 28, 28])

  def _rand_tensor(dtype: str, shape):
    # Handle common dtypes with appropriate ranges
    if dtype in ("float32", "float16", "float64"):
      return np.random.uniform(-1, 1, shape).astype(dtype)
    if dtype.startswith("int"):
      # Parse bit width if available (e.g., int4, int8, int16, int32)
      try:
        bits = int(dtype.replace("int", ""))
      except Exception:
        bits = 32
      if bits == 4:
        # No native int4 in numpy; store in int8 within valid int4 range
        return np.random.randint(-8, 8, size=shape, dtype=np.int8)
      if bits == 8:
        return np.random.randint(-128, 128, size=shape, dtype=np.int8)
      if bits == 16:
        return np.random.randint(-32768, 32768, size=shape, dtype=np.int16)
      if bits == 32:
        return np.random.randint(-2**31, 2**31, size=shape, dtype=np.int32)
      if bits == 64:
        return np.random.randint(-2**63, 2**63 - 1, size=shape, dtype=np.int64)
      # Fallback: use int32
      return np.random.randint(-2**31, 2**31, size=shape, dtype=np.int32)
    if dtype.startswith("uint"):
      try:
        bits = int(dtype.replace("uint", ""))
      except Exception:
        bits = 32
      if bits == 4:
        return np.random.randint(0, 16, size=shape, dtype=np.uint8)
      if bits == 8:
        return np.random.randint(0, 256, size=shape, dtype=np.uint8)
      if bits == 16:
        return np.random.randint(0, 2**16, size=shape, dtype=np.uint16)
      if bits == 32:
        return np.random.randint(0, 2**32, size=shape, dtype=np.uint32)
      if bits == 64:
        # numpy uint64 randint high is exclusive and must be <= 2**64-1
        return np.random.randint(0, np.iinfo(np.uint64).max, size=shape, dtype=np.uint64)
      return np.random.randint(0, 2**32, size=shape, dtype=np.uint32)
    # Default float32 if unrecognized
    return np.random.uniform(-1, 1, shape).astype("float32")

  params_dict = {}
  # Sort by name for determinism
  for name in sorted(var_dict.keys()):
    info = var_dict[name]
    params_dict[name] = _rand_tensor(info["dtype"], info["shape"])

  return out, params_dict