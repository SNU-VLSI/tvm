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
from tvm.relay.op.contrib import dnnl
import tvm.testing
from tvm.contrib import utils
from tvm import runtime as tvm_runtime
from tvm.contrib import graph_executor

from tvm.relay.backend import Executor, Runtime
from tvm.relay import pretty_print
from tvm.relay.backend.contrib.imcflow import transform as imcflow_transform

from models import *

np.random.seed(0)

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
  print(f"Running Test {name}")
  if name == "resnet":
    irmod, param_dict, shape_dict = resnet.getTestModel()
  elif name == "mobilenet":
    irmod, param_dict, shape_dict = mobilenet.getTestModel()
  elif name == "deep_autoencoder":
    irmod, param_dict, shape_dict = deep_autoencoder.getTestModel()
  elif name == "ds_cnn":
    irmod, param_dict, shape_dict = ds_cnn.getTestModel()
  elif name == "wide_model":
    irmod, param_dict, shape_dict = wide_model.getTestModel()
  else:
    raise ValueError("Model not found")

  os.makedirs(TestName, exist_ok=True)
  with open(f"{TestName}/before_model.txt", "w") as f:
    f.write(pretty_print(irmod))
  
  transforms = tvm.transform.Sequential([imcflow_transform.DenseToConv()])
  split_mod = transforms(irmod)

  with open(f"{TestName}/model.txt", "w") as f:
    f.write(pretty_print(split_mod))

  RelayVisualizer(
    relay_mod = split_mod,
    relay_param = param_dict,
    plotter = DotPlotter(),
    parser = DotVizParser(),
  ).render(f"{TestName}/model")

  dtype="float32"
  input_dict = {
      k: np.random.uniform(-1, 1, v).astype(dtype)
      for k, v in shape_dict.items()
  }

  Data1 = buildAndRun(TestName+"_ref", irmod, input_dict, param_dict)
  Data2 = buildAndRun(TestName+"_evl", split_mod, input_dict, param_dict)
  tvm.testing.assert_allclose(Data1.numpy(), Data2.numpy(), rtol=1e-3, atol=1e-3)

  os.system("rm " + f"{TestName}_ref.so " + f"{TestName}_evl.so")
  print("Good")

def test_resnet():
  RunTestModel("resnet")

def test_mobilenet():
  RunTestModel("mobilenet")

def test_deep_autoencoder():
  RunTestModel("deep_autoencoder")

def test_ds_cnn():
  RunTestModel("ds_cnn")

def test_wide_model():
  RunTestModel("wide_model")

if __name__ == "__main__":
  tvm.testing.main()