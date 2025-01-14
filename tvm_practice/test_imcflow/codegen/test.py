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
from tvm.relay.backend.contrib.imcflow import codegen as imcflow_codegen
from tvm.relay.op.contrib import imcflow
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig

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
  DevConfig().clear()

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
  print(DevConfig().ActiveIMCEPerFunc)

  print("HW MAP")
  print(DevConfig().HWNodeMap)

  print("CustomID TO Name")
  print(imcflow.CustomIDToName())

  print("Tensor Edge List")
  for key, paths in DevConfig().TensorEdgeListDict.items():
    print(key)
    for path in paths:
      print(path)

  imcflow_transform.constructTensorIDToTensorEdgeDict()
  print("Tensor ID to Tensor Edge")
  for key, paths in DevConfig().TensorIDtoEdge.items():
    print(f"{key} : {paths}")

  imcflow_transform.constructNoCPathDict(eval_mod)
  print("NoC Paths")
  for key, paths in DevConfig().NoCPaths.items():
    print(key)
    for k, v in paths.items():
      print(k, v)

  MemoryAllocator = imcflow_transform.MemoryAllocator()(eval_mod)
  PolicyTableGenerator = imcflow_transform.PolicyTableGenerator(DevConfig().NoCPaths)(eval_mod)

  # get the config
  config = DevConfig()

  print(f"mem_layout: {config.MemLayout}")
  print(f"nodemap: {config.HWNodeMap}")
  print(f"edgeinfo: {config.TensorEdgetoInfo}")
  print(f"idtoedge: {config.TensorIDtoEdge}")

  CodegenSuite = imcflow_codegen.CodegenSuite()
  CodegenSuite(eval_mod)

def test_big():
  mod, param_dict = getModel()
  run_test("big", mod, param_dict)

def test_small():
  mod, param_dict = real_model2.getModel()
  run_test("small", mod, param_dict)

if __name__ == "__main__":
  tvm.testing.main()