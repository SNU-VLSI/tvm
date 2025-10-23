import pytest
import torch
import tvm
import tvm.testing
from typing import Sequence
from tvm.relay.qnn.op.qnn import imcflow_min_max_quantize, imcflow_nu_quantize
import numpy as np
import tvm.relay.op as _op
import tvm.relay.expr as _expr
from tvm.relay import pretty_print
from tvm.contrib.relay_viz import RelayVisualizer, DotPlotter, DotVizParser
from tvm.relay.build_module import bind_params_by_name
from tvm.relay import transform
from tvm.relay.backend.contrib.imcflow import transform as imcflow_transform
from tvm.relay.op.contrib import imcflow
from tvm.contrib.imcflow import ImcflowDeviceConfig
from copy import deepcopy

from models.real_model import getModel
from models import real_model2

@torch.library.custom_op("imcflow::min_max_quant", mutates_args=())
def min_max_quant(pic: torch.Tensor, min:int, max:int) -> torch.Tensor:
    return pic.clone()

@torch.library.custom_op("imcflow::linear_quant", mutates_args=())
def linear_quant(x:torch.Tensor, scale:float, zero_point:int) -> torch.Tensor:
  return x.clone()

def make_min_max(input, input_types):
  MinNDArray = tvm.runtime.ndarray.array(np.array(input[1], dtype=np.float32))
  MaxNDArray = tvm.runtime.ndarray.array(np.array(input[2], dtype=np.float32))
  return imcflow_min_max_quantize(input[0], tvm.relay.Constant(MinNDArray), tvm.relay.Constant(MaxNDArray), 1, "float32")

def make_quantize(input, input_types):
  scale = tvm.relay.Constant(tvm.runtime.ndarray.array(np.array(input[1], dtype=np.float32)))
  bias = tvm.relay.Constant(tvm.runtime.ndarray.array(np.array(input[2], dtype=np.int32)))
  return tvm.relay.qnn.op.quantize(input[0], scale, bias, 1, "float32")

def runModel(test_name, mod, param_dict):
  input_data = np.random.rand(1, 32, 16, 16).astype(np.float32)
  mod = deepcopy(mod)
  mod = imcflow_transform.clearCompilerTag(mod)
  printModel(test_name, mod, param_dict, "clear_compiler_tag")

  tvm.relay.backend.te_compiler.get().clear()
  device = tvm.cpu(0)
  with tvm.transform.PassContext(opt_level=0):
      ref_lib = tvm.relay.build(mod, target="llvm", params=param_dict)
  ref_rt_mod = tvm.contrib.graph_executor.GraphModule(ref_lib["default"](device))

  ref_rt_mod.set_input("input", input_data)
  ref_rt_mod.run()
  out = ref_rt_mod.get_output(0)
  ref_result = out.numpy()
  print(ref_result)

def printModel(result_dir, mod, param_dict, mod_name):
  RelayVisualizer(
    relay_mod = mod,
    relay_param = param_dict,
    plotter = DotPlotter(),
    parser = DotVizParser(),
  ).render(f"{result_dir}/{mod_name}")

  with open(f"{result_dir}/{mod_name}.txt", "w") as f:
    f.write(pretty_print(mod))

def run_test(test_name, mod, param_dict):
  eval_mod, eval_param_dict = mod, param_dict
  ImcflowDeviceConfig().clear()

  # origin
  printModel(test_name, eval_mod, eval_param_dict, "origin")

  # bind param
  eval_mod["main"] = bind_params_by_name(eval_mod["main"], eval_param_dict)
  eval_mod = transform.InferType()(eval_mod)
  printModel(test_name, eval_mod, eval_param_dict, "after_bind")

  eval_mod = transform.MergeComposite(imcflow.pattern_table())(eval_mod)
  printModel(test_name, eval_mod, eval_param_dict, "after_merge")

  SplitConcatRegions = imcflow_transform.getSplitConcatDepsRegions(eval_mod["main"])
  eval_mod = imcflow.ImcflowAnnotationPass(SplitConcatRegions)(eval_mod)
  eval_mod = transform.MergeCompilerRegions()(eval_mod)
  eval_mod = transform.PartitionGraph()(eval_mod)
  printModel(test_name, eval_mod, eval_param_dict, "after_split_concat_partition")

  runModel(test_name, eval_mod, eval_param_dict)

  AnnotGenerator = imcflow_transform.AnnotGenerator()
  AnnotGenerator(eval_mod)
  # print(AnnotGenerator.RegionList)
  eval_mod = imcflow.ImcflowAnnotationPass(AnnotGenerator.RegionList)(eval_mod)
  printModel(test_name, eval_mod, eval_param_dict, "after_annot")

  eval_mod = transform.MergeCompilerRegions()(eval_mod)
  printModel(test_name, eval_mod, eval_param_dict, "after_merge_region")

  eval_mod = imcflow.ImcflowCleanRegionTag()(eval_mod)
  printModel(test_name, eval_mod, eval_param_dict, "after_clean_region")

  eval_mod = transform.PartitionGraph()(eval_mod)
  printModel(test_name, eval_mod, eval_param_dict, "after_partition_graph")

  eval_mod = imcflow.flattenSubgraphs(eval_mod)
  printModel(test_name, eval_mod, eval_param_dict, "after_flatten")

  eval_mod = imcflow.prune_imcflow_subgraphs(eval_mod)
  printModel(test_name, eval_mod, eval_param_dict, "after_prune_model")

  eval_mod = imcflow_transform.PackingInserter()(eval_mod)
  printModel(test_name, eval_mod, eval_param_dict, "after_packing")

  imcflow_transform.constructUsefulMappings(eval_mod)
  imcflow_transform.constructCustomIDInFunc(eval_mod)
  printModel(test_name, eval_mod, eval_param_dict, "with_custom_id")

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

  MemoryAllocator = imcflow_transform.MemoryAllocator()(eval_mod)
  PolicyTableGenerator = imcflow_transform.PolicyTableGenerator(ImcflowDeviceConfig().NoCPaths)(eval_mod)

def test_big():
  mod, param_dict = getModel()
  run_test("big", mod, param_dict)

def test_small():
  mod, param_dict = real_model2.getModel()
  run_test("small", mod, param_dict)

def test_ref_graph():
  param1 = tvm.relay.var("param1", shape=(1, 3, 16, 16))
  param2 = tvm.relay.var("param2", shape=(1, 3, 16, 16))

  f1 = tvm.relay.Function([param1, param2], param1+param2)
  f1 = f1.with_attr("Primitive", 1)
  # f2 = tvm.relay.Function([data1, data2], data1-data2)
  # y = tvm.relay.Call(f1, [data1, data2]) + tvm.relay.Call(f2, [data1, data2])

  data1 = tvm.relay.var("data1", shape=(1, 3, 16, 16))
  data2 = tvm.relay.var("data2", shape=(1, 3, 16, 16))
  y = tvm.relay.Call(f1, [data1, data2])
  mod = tvm.IRModule.from_expr(y)

  printModel("ref", mod, {}, "ref_func")

  tvm.relay.backend.te_compiler.get().clear()
  device = tvm.cpu(0)
  with tvm.transform.PassContext(opt_level=0):
      ref_lib = tvm.relay.build(mod, target="llvm", params={})
  ref_rt_mod = tvm.contrib.graph_executor.GraphModule(ref_lib["default"](device))

  data1 = np.random.rand(1, 3, 16, 16).astype(np.float32)
  data2 = np.random.rand(1, 3, 16, 16).astype(np.float32)
  ref_rt_mod.set_input("data1", data1)
  ref_rt_mod.set_input("data2", data2)
  ref_rt_mod.run()
  out = ref_rt_mod.get_output(0)
  ref_result = out.numpy()
  print(ref_result)

if __name__ == "__main__":
  tvm.testing.main()