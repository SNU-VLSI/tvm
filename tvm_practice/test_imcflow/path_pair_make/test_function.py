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

from tvm.relay.backend.contrib.imcflow import transform as imcflow_transform
from tvm.relay.op.contrib import imcflow
from tvm.contrib.relay_viz import RelayVisualizer, DotPlotter, DotVizParser

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

    input_shape = {"x1": (1, IC, IH, IW)}
    param_shape = {"weight": (OC, IC, KH, KW), "bias": (OC,)}
    out = tvm.IRModule.from_expr(y)
    return out, input_shape, param_shape

def RunTest():

  mod, input_shape, param_shape = get_graph(3, 32, 32, 64, 3, 3)
  dtype="float32"
  param_dict = {k: np.random.uniform(-1, 1, v).astype(dtype) for k, v in param_shape.items()}
  input_dict = {k: np.random.uniform(-1, 1, v).astype(dtype) for k, v in input_shape.items()}

  TestName="test_func"
  def printModel(mod, param_dict, mod_name):
    RelayVisualizer(
      relay_mod = mod,
      relay_param = param_dict,
      plotter = DotPlotter(),
      parser = DotVizParser(),
    ).render(f"{TestName}/{mod_name}")

    with open(f"{TestName}/{mod_name}.txt", "w") as f:
      f.write(pretty_print(mod))

  os.makedirs(TestName, exist_ok=True)

  # first phase
  mod["main"] = bind_params_by_name(mod["main"], param_dict)
  printModel(mod, param_dict, "before_model")

  mod["main"].body.args[0] = relay.nn.leaky_relu(mod["main"].body.args[0].args[0])
  printModel(mod, param_dict, "after_model")

def test():
  RunTest()

if __name__ == "__main__":
  tvm.testing.main()