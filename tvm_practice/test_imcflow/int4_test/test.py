import tvm
import numpy as np
import pathlib
from tvm.micro import export_model_library_format
from tvm.micro.testing import get_target
from tvm.contrib.utils import tempdir
import tvm.testing
from tvm.relay import pretty_print
from tvm.contrib.relay_viz import RelayVisualizer, DotPlotter, DotVizParser
from tvm.relay.build_module import bind_params_by_name
from tvm.relay import transform
from tvm.relay.backend.contrib.imcflow import transform as imcflow_transform
from tvm.relay.backend.contrib.imcflow import codegen as imcflow_codegen
from tvm.relay.op.contrib import imcflow
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from tvm.relay.backend import Executor, Runtime
from tvm.contrib.imcflow import DataBlock
from tvm import relay
import os

from models import real_model, real_model2
from models import small_model

def printModel(result_dir, mod, param_dict, mod_name):
  RelayVisualizer(
      relay_mod=mod,
      relay_param=param_dict,
      plotter=DotPlotter(),
      parser=DotVizParser(),
  ).render(f"{result_dir}/{mod_name}")

  with open(f"{result_dir}/{mod_name}.txt", "w") as f:
    f.write(pretty_print(mod))

def run_test_evl(test_name, mod, param_dict):
  """Generate IMCFLOW evaluation results (original function renamed)"""
  print(f"\n{'='*60}")
  print(f"GENERATING EVALUATION RESULTS FOR: {test_name}")
  print(f"{'='*60}")

  eval_dir = f"{test_name}_evl"
  os.makedirs(eval_dir, exist_ok=True)

  eval_mod, eval_param_dict = mod, param_dict
  DevConfig().clear()

  # origin
  printModel(eval_dir, eval_mod, eval_param_dict, "origin")

  # bind param
  eval_mod["main"] = bind_params_by_name(eval_mod["main"], eval_param_dict)
  eval_mod = transform.InferType()(eval_mod)
  printModel(eval_dir, eval_mod, eval_param_dict, "after_bind")

  eval_mod = transform.MergeComposite(imcflow.pattern_table())(eval_mod)
  printModel(eval_dir, eval_mod, eval_param_dict, "after_merge")

  SplitConcatRegions = imcflow_transform.getSplitConcatDepsRegions(
      eval_mod["main"])
  eval_mod = imcflow.ImcflowAnnotationPass(SplitConcatRegions)(eval_mod)
  eval_mod = transform.MergeCompilerRegions()(eval_mod)
  eval_mod = transform.PartitionGraph()(eval_mod)
  printModel(eval_dir, eval_mod, eval_param_dict,
             "after_split_concat_partition")

  AnnotGenerator = imcflow_transform.AnnotGenerator()
  AnnotGenerator(eval_mod)
  # print(AnnotGenerator.RegionList)
  eval_mod = imcflow.ImcflowAnnotationPass(AnnotGenerator.RegionList)(eval_mod)
  printModel(eval_dir, eval_mod, eval_param_dict, "after_annot")

  eval_mod = transform.MergeCompilerRegions()(eval_mod)
  printModel(eval_dir, eval_mod, eval_param_dict, "after_merge_region")

  eval_mod = imcflow.ImcflowCleanRegionTag()(eval_mod)
  printModel(eval_dir, eval_mod, eval_param_dict, "after_clean_region")

  eval_mod = transform.PartitionGraph()(eval_mod)
  printModel(eval_dir, eval_mod, eval_param_dict, "after_partition_graph")

  eval_mod = imcflow.flattenSubgraphs(eval_mod)
  printModel(eval_dir, eval_mod, eval_param_dict, "after_flatten")

  eval_mod = imcflow.prune_imcflow_subgraphs(eval_mod)
  printModel(eval_dir, eval_mod, eval_param_dict, "after_prune_model")

  # eval_mod = imcflow_transform.PackingInserter()(eval_mod)
  # printModel(eval_dir, eval_mod, eval_param_dict, "after_packing")

  imcflow_transform.constructUsefulMappings(eval_mod)
  imcflow_transform.constructCustomIDInFunc(eval_mod)
  printModel(eval_dir, eval_mod, eval_param_dict, "with_custom_id")

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

  imcflow_transform.MemoryAllocator()(eval_mod)
  imcflow_transform.PolicyTableGenerator(DevConfig().NoCPaths)(eval_mod)

  # get the config
  config = DevConfig()

  print(f"nodemap: {config.HWNodeMap}")
  print(f"edgeinfo: {config.TensorEdgetoInfo}")
  print(f"idtoedge: {config.TensorIDtoEdge}")

  CodegenSuite = imcflow_codegen.CodegenSuite(f"{eval_dir}/build")
  CodegenSuite(eval_mod)

  print(f"mem_layout: {config.MemLayout}")
  print(f"Evaluation generation completed for {test_name}")

if __name__ == "__main__":
  # input_ = relay.var("input", shape=(1, 28, 4, 4))
  input_ = relay.var("input", shape=(1,28,4,4), dtype="int4")
  y = relay.nn.pad(input_, pad_width=((0, 0),(0, 0),(1, 1),(1,1)), pad_value=0)
  y = relay.nn.relu(y)

  param_dict = {
    # "quant_scale": np.random.rand(28).astype("float32"),
    # "quant_zp": np.random.randint(0, 255, 28).astype("int"),
    # "weight": np.random.rand(28,28,3,3).astype("float32")
  }

  out = tvm.IRModule.from_expr(y)

  run_test_evl("test", out, param_dict)