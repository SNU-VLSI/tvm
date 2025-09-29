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
  y = imcflow_min_max_quantize(y, relay.var("quant_min_1", shape=(1,), dtype="int16"), relay.var("quant_max_1", shape=(1,), dtype="int16"), axis=1, out_dtype="int4")
  y = imcflow_qconv2d(
    y,
    relay.var("weight2_1", shape=(16,16,3,3), dtype="int4"),
    in_channels=16,
    channels=16,
    kernel_size=(3, 3),
    padding=(1, 1),
    out_dtype="int16"
  )
  y = imcflow_batch_norm(y, relay.var("fused_scale1", shape=(16,), dtype="int16"), relay.var("fused_bias1", shape=(16,), dtype="int16"))[0]
  y = imcflow_min_max_quantize(y, relay.var("quant_min_2", shape=(1,), dtype="int16"), relay.var("quant_max_2", shape=(1,), dtype="int16"), axis=1, out_dtype="int4")
  y = imcflow_qconv2d(
    y,
    relay.var("weight2_2", shape=(16,16,3,3), dtype="int4"),
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
  y = imcflow_min_max_quantize(y, relay.var("quant_min_3", shape=(1,), dtype="int16"), relay.var("quant_max_3", shape=(1,), dtype="int16"), axis=1, out_dtype="int4")
  y = imcflow_qconv2d(
    y,
    relay.var("weight3_1", shape=(32,16,3,3), dtype="int4"),
    in_channels=16,
    channels=32,
    kernel_size=(3, 3),
    padding=(1, 1),
    strides=(2,2),
    out_dtype="int16"
  )
  y = imcflow_batch_norm(y, relay.var("fused_scale3", shape=(32,), dtype="int16"), relay.var("fused_bias3", shape=(32,), dtype="int16"))[0]
  y = imcflow_min_max_quantize(y, relay.var("quant_min_4", shape=(1,), dtype="int16"), relay.var("quant_max_4", shape=(1,), dtype="int16"), axis=1, out_dtype="int4")
  y = imcflow_qconv2d(
    y,
    relay.var("weight3_2", shape=(32,32,3,3), dtype="int4"),
    in_channels=32,
    channels=32,
    kernel_size=(3, 3),
    padding=(1, 1),
    out_dtype="int16"
  )
  y = imcflow_batch_norm(y, relay.var("fused_scale4", shape=(32,), dtype="int16"), relay.var("fused_bias4", shape=(32,), dtype="int16"))[0]

  y_residual = imcflow_min_max_quantize(residual, relay.var("quant_min_4", shape=(1,), dtype="int16"), relay.var("quant_max_4", shape=(1,), dtype="int16"), axis=1, out_dtype="int4")
  y_residual = imcflow_qconv2d(
    y_residual,
    relay.var("weight3_0", shape=(32,16,1,1), dtype="int4"),
    in_channels=16,
    channels=32,
    kernel_size=(1, 1),
    strides=(2,2),
    out_dtype="int16"
  )
  y_residual = relay.var("bn_out_f_1", shape=(32,), dtype="int16") * y_residual + relay.var("bn_out_f_0", shape=(32,), dtype="int16")
  y = y + y_residual

  # basic block 3
  residual = y
  y = imcflow_min_max_quantize(y, relay.var("quant_min_5", shape=(1,), dtype="int16"), relay.var("quant_max_5", shape=(1,), dtype="int16"), axis=1, out_dtype="int4")
  y = imcflow_qconv2d(
    y,
    relay.var("weight4_1", shape=(64,32,3,3), dtype="int4"),
    in_channels=32,
    channels=64,
    kernel_size=(3, 3),
    padding=(1, 1),
    strides=(2,2),
    out_dtype="int16"
  )
  y = imcflow_batch_norm(y, relay.var("fused_scale5", shape=(64,), dtype="int16"), relay.var("fused_bias5", shape=(64,), dtype="int16"))[0]
  y = imcflow_min_max_quantize(y, relay.var("quant_min_6", shape=(1,), dtype="int16"), relay.var("quant_max_6", shape=(1,), dtype="int16"), axis=1, out_dtype="int4")
  y = imcflow_qconv2d(
    y,
    relay.var("weight4_2", shape=(64,64,3,3), dtype="int4"),
    in_channels=64,
    channels=64,
    kernel_size=(3, 3),
    padding=(1, 1),
    out_dtype="int16"
  )
  y = imcflow_batch_norm(y, relay.var("fused_scale6", shape=(64,), dtype="int16"), relay.var("fused_bias6", shape=(64,), dtype="int16"))[0]

  y_residual = imcflow_min_max_quantize(residual, relay.var("quant_min_6", shape=(1,), dtype="int16"), relay.var("quant_max_6", shape=(1,), dtype="int16"), axis=1, out_dtype="int4")
  y_residual = imcflow_qconv2d(
    y_residual,
    relay.var("weight4_0", shape=(64,32,1,1), dtype="int4"),
    in_channels=32,
    channels=64,
    kernel_size=(1, 1),
    strides=(2,2),
    out_dtype="int16"
  )
  y_residual = relay.var("bn_out_f_3", shape=(64,), dtype="int16") * y_residual + relay.var("bn_out_f_2", shape=(64,), dtype="int16")

  y = y + y_residual

  # post process
  y = relay.cast(y,dtype="float32") / relay.var("post_f", shape=(1,), dtype="float32")
  y = relay.nn.relu(y)
  y = relay.nn.adaptive_avg_pool2d(y, output_size=(1,1))
  y = relay.nn.batch_flatten(y) 
  y = relay.nn.dense(y, relay.var("dense_weight", shape=(10, 64), dtype="float32"))
  y = relay.nn.bias_add(y, relay.var("dense_bias", shape=(10,), dtype="float32"))

  var_dict = {
    "weight1": (16, 3, 3, 3),
    "bn_gamma": (16,),
    "bn_beta": (16,),
    "bn_moving_mean": (16,),
    "bn_moving_var": (16,),

    "x_f_1": (1,),

    "weight2_1": (16,16,3,3),
    "fused_scale1": (16,),
    "fused_bias1": (16,),
    "weight2_2": (16,16,3,3),
    "fused_scale2": (16,),
    "fused_bias2": (16,),
    "quant_min_1": (16,),
    "quant_max_1": (16,),
    "quant_min_2": (16,),
    "quant_max_2": (16,),
    "y_f_1": (1,),

    "weight3_1": (32,16,3,3),
    "fused_scale3": (32,),
    "fused_bias3": (32,),
    "weight3_2": (32,32,3,3),
    "fused_scale4": (32,),
    "fused_bias4": (32,),
    "weight3_0": (32,16,1,1),
    "quant_min_3": (16,),
    "quant_max_3": (16,),
    "quant_min_4": (32,),
    "quant_max_4": (32,),
    "bn_out_f_1": (32,),
    "bn_out_f_0": (32,),
    "y_f_2": (1,),

    "weight4_1": (64,32,3,3),
    "fused_scale5": (64,),
    "fused_bias5": (64,),
    "weight4_2": (64,64,3,3),
    "fused_scale6": (64,),
    "fused_bias6": (64,),
    "weight4_0": (64,32,1,1),
    "quant_min_5": (32,),
    "quant_max_5": (32,),
    "quant_min_6": (64,),
    "quant_max_6": (64,),
    "bn_out_f_3": (64,),
    "bn_out_f_2": (64,),
    "y_f_3": (1,),

    "post_f": (1,),

    "dense_weight": (10, 64),
    "dense_bias": (10,),
  }

  out = tvm.IRModule.from_expr(y)

  return out, var_dict

def getModel():
  out, var_dict = getModel_([1, 3, 28, 28])
  params_dict = {}
  for k, v in var_dict.items():
    params_dict[k] = np.random.uniform(-1, 1, v).astype("float32")
  return out, params_dict