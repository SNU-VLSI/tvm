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

def get_model1():
  input_ = relay.var("input", shape=(1, 32, 16, 16))

  y = relay.nn.conv2d(
      input_,
      relay.var("weight1", shape=(65, 32, 3, 3)),
      channels=65,
      in_channels=32,
      kernel_size=(3, 3),
      padding=(1, 1),
  )

  y = relay.nn.conv2d(
      y,
      relay.var("weight2", shape=(65, 65, 3, 3)),
      channels=65,
      in_channels=65,
      kernel_size=(3, 3),
      padding=(1, 1),
  )

  y = relay.nn.bias_add(y, relay.var("bias2", shape=(65,)))
  y = relay.nn.batch_norm(y, relay.var("gamma2", shape=(65,), dtype="float32"),
                             relay.var("beta2", shape=(65,), dtype="float32"),
                             relay.var("moving_mean2", shape=(65,), dtype="float32"),
                             relay.var("moving_var2", shape=(65,), dtype="float32"))[0]
  y = relay.nn.relu(y)

  param_dict = {
    "weight1": np.random.rand(65, 32, 3, 3).astype("float32"),
    "weight2": np.random.rand(65, 65, 3, 3).astype("float32"),
    "bias2": np.random.rand(65).astype("float32"),
    "gamma2": np.random.rand(65).astype("float32"),
    "beta2": np.random.rand(65).astype("float32"),
    "moving_mean2": np.random.rand(65).astype("float32"),
    "moving_var2": np.random.rand(65).astype("float32"),
  }

  out = tvm.IRModule.from_expr(y)

  return out, param_dict