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

def generate_graph_executor(ref_mod, param_dict, dir_name):
  executor_cfg = Executor("graph")
  runtime_cfg = Runtime("crt", {"system-lib": True})
  print("\n" + "="*40)
  print("GENERATING GRAPH EXECUTOR")
  print("="*40)

  with tvm.transform.PassContext(opt_level=0, config={"tir.disable_vectorize": True}):
    module = tvm.relay.build(
      ref_mod,
      target="c",
      params=param_dict,
      executor=executor_cfg,
      runtime=runtime_cfg,
    )

  script_dir = os.path.dirname(os.path.realpath(__file__))
  tar_name = f"lib_graph_system-lib.tar"
  tar_path = os.path.join(script_dir, dir_name, tar_name)
  export_model_library_format(module, tar_path)
  return module, tar_path


def run_test_ref(test_name, mod, param_dict):
  """Generate reference TVM compilation results"""
  print(f"\n{'='*60}")
  print(f"GENERATING REFERENCE RESULTS FOR: {test_name}")
  print(f"{'='*60}")

  ref_dir = f"{test_name}_ref"
  os.makedirs(ref_dir, exist_ok=True)

  # Save original model
  printModel(ref_dir, mod, param_dict, "origin")

  # Create a new IRModule for reference (TVM IRModule doesn't have copy method)
  ref_mod = tvm.IRModule({"main": mod["main"]})

  # Check if model contains IMCFLOW operations - if so, skip reference generation
  model_str = pretty_print(ref_mod)
  if "imcflow" in model_str.lower():
    print("⚠️  WARNING: Model contains IMCFLOW operations!")
    print("Cannot generate standard TVM reference for model with IMCFLOW-specific operations.")
    print("Skipping reference generation...")

    # Create a placeholder file to indicate this
    with open(f"{ref_dir}/SKIPPED_IMCFLOW_MODEL.txt", "w") as f:
      f.write("Reference generation skipped.\n")
      f.write(
          "Model contains IMCFLOW-specific operations that cannot be compiled with standard TVM.\n")
      f.write("To generate a reference, use a model without IMCFLOW operations.\n")

    print(f"Created placeholder: {ref_dir}/SKIPPED_IMCFLOW_MODEL.txt")
    return ref_mod

  # Proceed with reference generation for clean models
  ref_mod["main"] = bind_params_by_name(ref_mod["main"], param_dict)
  ref_mod = transform.InferType()(ref_mod)
  printModel(ref_dir, ref_mod, param_dict, "after_bind")

  # Apply only standard TVM optimizations (no IMCFLOW)
  with tvm.transform.PassContext(opt_level=3):
    ref_mod = transform.FoldConstant()(ref_mod)
    ref_mod = transform.SimplifyInference()(ref_mod)
    ref_mod = transform.FoldScaleAxis()(ref_mod)
    ref_mod = transform.SimplifyExpr()(ref_mod)
    ref_mod = transform.FoldConstant()(ref_mod)

  printModel(ref_dir, ref_mod, param_dict, "after_std_optimization")

  generate_graph_executor(ref_mod, param_dict, ref_dir)

  # Save final model state
  printModel(ref_dir, ref_mod, param_dict, "final_ref_model")

  print(f"Reference generation completed for {test_name}")
  return ref_mod


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

  eval_mod = imcflow_transform.PackingInserter()(eval_mod)
  printModel(eval_dir, eval_mod, eval_param_dict, "after_packing")

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
  print(f"policy_table: {config.PolicyTableDict}")

  CodegenSuite = imcflow_codegen.CodegenSuite(f"{eval_dir}/build")
  CodegenSuite(eval_mod)

  print(f"mem_layout: {config.MemLayout}")
  print(f"Evaluation generation completed for {test_name}")

  generate_graph_executor(eval_mod, eval_param_dict, eval_dir)


def test_big_ref():
  """Generate only reference for big model"""
  assert False, "Big model reference is not supported yet"


def test_small_ref():
  """Generate only reference for small model"""
  mod, param_dict, _ = small_model.getTestModel()
  run_test_ref("small", mod, param_dict)


def test_big_evl():
  """Generate only evaluation for big model"""
  mod, param_dict = real_model.getModel()
  run_test_evl("big", mod, param_dict)


def test_small_evl():
  """Generate only evaluation for small model"""
  mod, param_dict = real_model2.getModel()
  run_test_evl("small", mod, param_dict)

def test_one_conv_evl():
  mod, param_dict = real_model2.getOneConvModel()
  run_test_evl("one_conv", mod, param_dict)

def test_one_relu_evl():
  """Generate evaluation for relu model"""
  mod, param_dict = real_model2.getOneReluModel()
  run_test_evl("one_relu", mod, param_dict)

if __name__ == "__main__":
  tvm.testing.main()
