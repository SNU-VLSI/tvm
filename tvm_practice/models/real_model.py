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
  input_ = relay.var("input", shape=(1, 32, 56, 56))

  y = relay.nn.conv2d(
      input_,
      relay.var("weight1", shape=(32, 32, 3, 3)),
      channels=32,
      kernel_size=(3, 3),
      padding=(1, 1),
  )

  y = relay.nn.bias_add(y, relay.var("bias1", shape=(32,)))
  y = imcflow_batch_norm(y, relay.var("fused_scale1", shape=(32,)), relay.var("fused_bias1", shape=(32,)))[0]
  y = relay.nn.relu(y)

  y = relay.qnn.simulated_quantize(y, relay.var("quant_scale", shape=(32,)), relay.var("quant_zp", shape=(32,), dtype="int32"), axis=1)

  y = relay.op.split(y, [4,], axis=1)

  y1 = imcflow_qconv2d(
    y[1],
    relay.var("weight2_0", shape=(64,28,3,3)),
    channels=64,
    kernel_size=(3, 3),
    padding=(1, 1),
  )

  y2 = imcflow_qconv2d(
    y[0],
    relay.var("weight2_1", shape=(64,4,3,3)),
    channels=64,
    kernel_size=(3, 3),
    padding=(1, 1),
  )

  y3 = y1+y2
  y3 = imcflow_batch_norm(y3, relay.var("fused_scale2", shape=(64,)), relay.var("fused_bias2", shape=(64,)))[0]
  y3 = relay.nn.relu(y3)
  y3 = imcflow_min_max_quantize(y3, relay.const(0.0, "float32"), relay.const(1.0, "float32"), 1, "float32")

  y1 = imcflow_qconv2d(
    y[1],
    relay.var("weight3_0", shape=(64,28,3,3)),
    channels=64,
    kernel_size=(3, 3),
    padding=(1, 1),
  )

  y2 = imcflow_qconv2d(
    y[0],
    relay.var("weight3_1", shape=(64,4,3,3)),
    channels=64,
    kernel_size=(3, 3),
    padding=(1, 1),
  )

  y4 = y1+y2
  y4 = imcflow_batch_norm(y4, relay.var("fused_scale3", shape=(64,)), relay.var("fused_bias3", shape=(64,)))[0]
  y4 = relay.nn.relu(y4)
  y4 = imcflow_min_max_quantize(y4, relay.const(0.0, "float32"), relay.const(1.0, "float32"), 1, "float32")

  y = relay.concatenate([y3, y4], axis=1)

  y = relay.op.split(y, [28, 56, 84, 112], axis=1)
  
  y1 = imcflow_qconv2d(
    y[0],
    relay.var("weight4_0", shape=(64,28,3,3)),
    channels=64,
    kernel_size=(3, 3),
    padding=(1, 1),
  )
  y2 = imcflow_qconv2d(
    y[1],
    relay.var("weight4_1", shape=(64,28,3,3)),
    channels=64,
    kernel_size=(3, 3),
    padding=(1, 1),
  )
  y3 = imcflow_qconv2d(
    y[2],
    relay.var("weight4_2", shape=(64,28,3,3)),
    channels=64,
    kernel_size=(3, 3),
    padding=(1, 1),
  )
  y4 = imcflow_qconv2d(
    y[3],
    relay.var("weight4_3", shape=(64,28,3,3)),
    channels=64,
    kernel_size=(3, 3),
    padding=(1, 1),
  )
  y5 = imcflow_qconv2d(
    y[4],
    relay.var("weight4_4", shape=(64,16,3,3)),
    channels=64,
    kernel_size=(3, 3),
    padding=(1, 1),
  )

  y_1 = y1/(relay.const(1.0, "float32")) + y2/(relay.const(1.0, "float32")) + y3/(relay.const(1.0, "float32")) + y4/(relay.const(1.0, "float32")) + y5/(relay.const(1.0, "float32"))
  y_1 = imcflow_batch_norm(y_1, relay.var("fused_scale4", shape=(64,)), relay.var("fused_bias4", shape=(64,)))[0]
  y_1 = relay.nn.relu(y_1)
  y_1 = imcflow_min_max_quantize(y_1, relay.const(0.0, "float32"), relay.const(1.0, "float32"), 1, "float32")

  y1 = imcflow_qconv2d(
    y[0],
    relay.var("weight5_0", shape=(64,28,3,3)),
    channels=64,
    kernel_size=(3, 3),
    padding=(1, 1),
  )
  y2 = imcflow_qconv2d(
    y[1],
    relay.var("weight5_1", shape=(64,28,3,3)),
    channels=64,
    kernel_size=(3, 3),
    padding=(1, 1),
  )
  y3 = imcflow_qconv2d(
    y[2],
    relay.var("weight5_2", shape=(64,28,3,3)),
    channels=64,
    kernel_size=(3, 3),
    padding=(1, 1),
  )
  y4 = imcflow_qconv2d(
    y[3],
    relay.var("weight5_3", shape=(64,28,3,3)),
    channels=64,
    kernel_size=(3, 3),
    padding=(1, 1),
  )
  y5 = imcflow_qconv2d(
    y[4],
    relay.var("weight5_4", shape=(64,16,3,3)),
    channels=64,
    kernel_size=(3, 3),
    padding=(1, 1),
  )

  y_2 = y1/(relay.const(1.0, "float32")) + y2/(relay.const(1.0, "float32")) + y3/(relay.const(1.0, "float32")) + y4/(relay.const(1.0, "float32")) + y5/(relay.const(1.0, "float32"))
  y_2 = imcflow_batch_norm(y_2, relay.var("fused_scale5", shape=(64,)), relay.var("fused_bias5", shape=(64,)))[0]
  y_2 = relay.nn.relu(y_2)
  y_2 = imcflow_min_max_quantize(y_2, relay.const(0.0, "float32"), relay.const(1.0, "float32"), 1, "float32")

  y = relay.concatenate([y_1, y_2], axis=1)

  param_dict = {
    "quant_scale": np.random.rand(32).astype("float32"),
    "quant_zp": np.random.randint(0, 255, 32).astype("int"),
    "weight1": np.random.rand(32, 32, 3, 3).astype("float32"),
    "bias1": np.random.rand(32).astype("float32"),
    "fused_scale1": np.random.rand(32).astype("float32"),
    "fused_bias1": np.random.rand(32).astype("float32"),
    "weight2_0": np.random.rand(64,28,3,3).astype("float32"),
    "weight2_1": np.random.rand(64,4,3,3).astype("float32"),
    "fused_scale2": np.random.rand(64).astype("float32"),
    "fused_bias2": np.random.rand(64).astype("float32"),
    "weight3_0": np.random.rand(64,28,3,3).astype("float32"),
    "weight3_1": np.random.rand(64,4,3,3).astype("float32"),
    "fused_scale3": np.random.rand(64).astype("float32"),
    "fused_bias3": np.random.rand(64).astype("float32"),
    "weight4_0": np.random.rand(64,28,3,3).astype("float32"),
    "weight4_1": np.random.rand(64,28,3,3).astype("float32"),
    "weight4_2": np.random.rand(64,28,3,3).astype("float32"),
    "weight4_3": np.random.rand(64,28,3,3).astype("float32"),
    "weight4_4": np.random.rand(64,16,3,3).astype("float32"),
    "fused_scale4": np.random.rand(64).astype("float32"),
    "fused_bias4": np.random.rand(64).astype("float32"),
    "weight5_0": np.random.rand(64,28,3,3).astype("float32"),
    "weight5_1": np.random.rand(64,28,3,3).astype("float32"),
    "weight5_2": np.random.rand(64,28,3,3).astype("float32"),
    "weight5_3": np.random.rand(64,28,3,3).astype("float32"),
    "weight5_4": np.random.rand(64,16,3,3).astype("float32"),
    "fused_scale5": np.random.rand(64).astype("float32"),
    "fused_bias5": np.random.rand(64).astype("float32"),
  }

  out = tvm.IRModule.from_expr(y)

  return out, param_dict