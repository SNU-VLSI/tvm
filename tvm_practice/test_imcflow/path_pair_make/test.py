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

def getCustomTestModel():
  x1 = relay.var("x1", shape=(1, 4, 4, 4))
  x2 = relay.split(x1, 2, 1)
  x3 = x2[0]
  x4 = x2[1]
  y = relay.add(x3, x4)
  out = tvm.IRModule.from_expr(y)
  return out, {}, {"x1" : (1, 4, 4, 4)}

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

def RunTest(Shapes):
  print("Running Test")
  print(Shapes)
  irmod, dic, param_list = get_graph(**Shapes)
  CustomPass = imcflow_transform.ConvSplitToAtom()
  split_mod = CustomPass(irmod)

  dtype="float32"
  param_dict = {x: np.random.uniform(-1, 1, dic[x]).astype(dtype) for x in param_list}
  input_dict = {
      k: np.random.uniform(-1, 1, v).astype(dtype)
      for k, v in dic.items()
      if k not in param_list
  }

  TestName="_".join([f"{k}_{v}" for k,v in Shapes.items()])

  Data1 = buildAndRun(TestName+"_ref", irmod, input_dict, param_dict)
  Data2 = buildAndRun(TestName+"_evl", split_mod, input_dict, param_dict)
  tvm.testing.assert_allclose(Data1.numpy(), Data2.numpy(), rtol=1e-3, atol=1e-3)

  os.system("rm " + f"{TestName}_ref.so " + f"{TestName}_evl.so")
  print("Good")

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
  elif name == "small_model":
    irmod, param_dict, shape_dict = small_model.getTestModel()
  elif name == "custom":
    irmod, param_dict, shape_dict = getCustomTestModel()
  else:
    raise ValueError("Model not found")

  os.makedirs(TestName, exist_ok=True)

  # first phase
  printModel(irmod, param_dict, "before_model")

  # init
  eval_mod = irmod
  eval_param_dict = param_dict

  # imcflow specific first phase
  ConvSplitToAtom = imcflow_transform.ConvSplitToAtom(eval_param_dict)
  transforms = tvm.transform.Sequential([ConvSplitToAtom])
  eval_mod = transforms(eval_mod)
  eval_param_dict = ConvSplitToAtom.NewParamDict
  printModel(eval_mod, eval_param_dict, "after_split")

  # bind params
  eval_mod["main"] = bind_params_by_name(eval_mod["main"], eval_param_dict)
  printModel(eval_mod, eval_param_dict, "after_bind")

  # byoc pass
  eval_mod = transform.MergeComposite(imcflow.pattern_table())(eval_mod)
  printModel(eval_mod, eval_param_dict, "after_merge")

  SplitConcatRegions = imcflow_transform.getSplitConcatDepsRegions(eval_mod["main"])
  eval_mod = imcflow.ImcflowAnnotationPass(SplitConcatRegions)(eval_mod)
  eval_mod = transform.MergeCompilerRegions()(eval_mod)
  eval_mod = transform.PartitionGraph()(eval_mod)
  printModel(eval_mod, eval_param_dict, "after_split_concat_partition")

  AnnotGenerator = imcflow_transform.AnnotGenerator()
  AnnotGenerator(eval_mod)
  # print(AnnotGenerator.RegionList)
  eval_mod = imcflow.ImcflowAnnotationPass(AnnotGenerator.RegionList)(eval_mod)
  printModel(eval_mod, eval_param_dict, "after_annot")

  eval_mod = transform.MergeCompilerRegions()(eval_mod)
  printModel(eval_mod, eval_param_dict, "after_merge_region")

  eval_mod = imcflow.ImcflowCleanRegionTag()(eval_mod)
  printModel(eval_mod, eval_param_dict, "after_clean_region")

  eval_mod = transform.PartitionGraph()(eval_mod)
  printModel(eval_mod, eval_param_dict, "after_partition_graph")

  eval_mod = imcflow.flattenSubgraphs(eval_mod)
  printModel(eval_mod, eval_param_dict, "after_flatten")

  eval_mod = imcflow.prune_imcflow_subgraphs(eval_mod)
  imcflow_transform.constructUsefulMappings(eval_mod)
  imcflow_transform.constructCustomIDInFunc(eval_mod)
  printModel(eval_mod, eval_param_dict, "after_prune_model")

  imcflow_transform.NodeMapper()(eval_mod)
  imcflow_transform.constructTensorEdgeList(eval_mod)
  imcflow_transform.constructActiveIMCEDict(eval_mod)

  print("Active IMCE list")
  print(ImcflowDeviceConfig().ActiveIMCEPerFunc)

  print("HW MAP")
  print(ImcflowDeviceConfig().HWNodeMap)

  print("CustomID TO Name")
  print(imcflow.CustomIDToName())

  print("Tensor Edge List")
  for key, paths in ImcflowDeviceConfig().TensorEdgeListDict.items():
    print(key)
    for path in paths:
      print(path)
  
  imcflow_transform.constructTensorIDToTensorEdgeDict()
  print("Tensor ID to Tensor Edge")
  for key, paths in ImcflowDeviceConfig().TensorIDtoEdge.items():
    print(f"{key} : {paths}")
  
  imcflow_transform.constructNoCPathDict(eval_mod)
  print("NoC Paths")
  for key, paths in ImcflowDeviceConfig().NoCPaths.items():
    print(key)
    for k, v in paths.items():
      print(k, v)

  MemoryCalculator = imcflow_transform.MemoryCalculator()(eval_mod)
  PolicyTableGenerator = imcflow_transform.PolicyTableGenerator(ImcflowDeviceConfig().NoCPaths, MemoryCalculator.SizeDict)(eval_mod)

def test_1x1_small():
  Shapes = { "IC": 257, "IH": 16, "IW": 16, "OC": 65, "KH": 1, "KW": 1}
  RunTest(Shapes)

def test_1x1_big():
  Shapes = { "IC": 513, "IH": 16, "IW": 16, "OC": 129, "KH": 1, "KW": 1}
  RunTest(Shapes)

def test_3x3_small():
  Shapes = { "IC": 29, "IH": 16, "IW": 16, "OC": 65, "KH": 3, "KW": 3}
  RunTest(Shapes)

def test_3x3_big():
  Shapes = { "IC": 57, "IH": 16, "IW": 16, "OC": 129, "KH": 3, "KW": 3}
  RunTest(Shapes)

def test_5x5_small():
  Shapes = { "IC": 11, "IH": 16, "IW": 16, "OC": 65, "KH": 5, "KW": 5}
  RunTest(Shapes)

def test_5x5_big():
  Shapes = { "IC": 21, "IH": 16, "IW": 16, "OC": 129, "KH": 5, "KW": 5}
  RunTest(Shapes)

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

def test_small_model():
  RunTestModel("small_model")

def test_custom_model():
  RunTestModel("custom")

if __name__ == "__main__":
  tvm.testing.main()