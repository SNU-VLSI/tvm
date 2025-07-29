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
# Add imports for reference TVM compilation
from tvm.relay.backend import Executor, Runtime
from tvm.contrib import graph_executor
from tvm.relay.backend import te_compiler
import os

from models import real_model, real_model2
from models import small_model

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

def buildAndRun_ref(name, mod, data_dict, param_dict):
  """Generate reference TVM results without IMCFLOW processing"""
  print(f"=== Generating REFERENCE for {name} ===")

  dev = tvm.cpu()
  Executor_ = Executor("graph")
  Runtime_ = Runtime("cpp")

  # Build with standard TVM C backend (NO IMCFLOW)
  with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
    te_compiler.get().clear()
    lib = tvm.relay.build(mod, target="c", params=param_dict, executor=Executor_, runtime=Runtime_)

  lib.export_library(f"{name}_ref.so")
  lib_loaded = tvm.runtime.load_module(f"{name}_ref.so")

  gmod = graph_executor.GraphModule(lib_loaded["default"](dev))
  gmod.set_input(**data_dict)
  gmod.run()

  result = gmod.get_output(0)
  print(f"Reference result shape: {result.shape}")
  print(f"Reference result sample: {result.numpy().flatten()[:10]}")
  return result

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
      f.write("Model contains IMCFLOW-specific operations that cannot be compiled with standard TVM.\n")
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

  # Generate standard TVM C code
  with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
    te_compiler.get().clear()
    lib = tvm.relay.build(ref_mod, target="c", params=param_dict)

    # Extract and save the generated C source code
    try:
      # Get the underlying library module and extract C source
      underlying_lib = lib.get_lib()
      c_source = underlying_lib.get_source("c")
      with open(f"{ref_dir}/{test_name}_generated.c", "w") as f:
        f.write(c_source)
      print(f"Generated C source saved: {ref_dir}/{test_name}_generated.c")
    except Exception as e:
      print(f"Could not extract C source: {e}")

  lib.export_library(f"{ref_dir}/{test_name}_reference.so")
  print(f"Generated reference library: {ref_dir}/{test_name}_reference.so")

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

  SplitConcatRegions = imcflow_transform.getSplitConcatDepsRegions(eval_mod["main"])
  eval_mod = imcflow.ImcflowAnnotationPass(SplitConcatRegions)(eval_mod)
  eval_mod = transform.MergeCompilerRegions()(eval_mod)
  eval_mod = transform.PartitionGraph()(eval_mod)
  printModel(eval_dir, eval_mod, eval_param_dict, "after_split_concat_partition")

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

  # print(f"nodemap: {config.HWNodeMap}")
  # print(f"edgeinfo: {config.TensorEdgetoInfo}")
  # print(f"idtoedge: {config.TensorIDtoEdge}")

  CodegenSuite = imcflow_codegen.CodegenSuite(f"{eval_dir}/build")
  CodegenSuite(eval_mod)

  print(f"mem_layout: {config.MemLayout}")
  print(f"Evaluation generation completed for {test_name}")

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

if __name__ == "__main__":
  tvm.testing.main()