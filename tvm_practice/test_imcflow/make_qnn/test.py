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
from tvm.contrib.imcflow import ImcflowDeviceConfig

from models import *

np.random.seed(0)

def get_graph(IC, IH, IW, OC, KH, KW):
    x1 = relay.var("input", shape=(1, IC, IH, IW), dtype="int8")
    weight = relay.var("weight", shape=(OC, IC/2, KH, KW), dtype="int8")
    y = relay.nn.conv2d(
        x1,
        weight,
        channels=OC,
        kernel_size=(KH, KW),
        padding=(1, 1),
    )
    out = tvm.IRModule.from_expr(y)
    return out

def buildAndRun(name, mod, data_dict, param_dict):

  dev = tvm.cpu()
  Executor_ = Executor("graph")
  Runtime_  = Runtime("cpp") 

  # build
  with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
    te_compiler.get().clear()
    mod = relay.build(mod, target="c", params=param_dict, executor=Executor_, runtime=Runtime_)


  mod.export_library(f"{name}.so")
  lib = tvm.runtime.load_module(f"{name}.so")
  
  gmod = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
  gmod.set_input(**data_dict)
  gmod.run()

  return gmod.get_output(0)

def RunTestModel(name):
  from tvm.contrib.relay_viz import RelayVisualizer, DotPlotter, DotVizParser

  TestName=name
  def printModel(mod, param_dict, mod_name):
    RelayVisualizer(
      relay_mod = mod,
      relay_param = param_dict,
      plotter = DotPlotter(),
      parser = DotVizParser(),
    ).render(f"{TestName}/{mod_name}")

    with open(f"{TestName}/{mod_name}.txt", "w") as f:
      f.write(pretty_print(mod))

  print(f"Running Test {name}")
  os.makedirs(TestName, exist_ok=True)

  # first phase
  irmod = get_graph(16, 16, 16, 16, 3, 3)
  param_dict = {
    "weight" : np.random.randint(-8, 8, (16, 8, 3, 3)).astype(np.int8),
  }

  # init
  eval_mod = irmod
  eval_param_dict = param_dict
  eval_data_dict = {"input" : np.random.randint(-8, 8, (1, 16, 16, 8)).astype(np.int8)}
  eval_mod['main'] = bind_params_by_name(eval_mod['main'], eval_param_dict)
  print(eval_mod.astext(show_meta_data=True))
  printModel(eval_mod, eval_param_dict, "before_model")
  transform.InferType()(eval_mod)

  # buildAndRun(name, eval_mod, eval_data_dict, eval_param_dict)

def test_main():
  RunTestModel("make_qnn")

if __name__ == "__main__":
  tvm.testing.main()