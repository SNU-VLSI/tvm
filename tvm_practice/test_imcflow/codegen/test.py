import pytest
import torch
import tvm
from tvm.micro import export_model_library_format
import tarfile
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

def printModel(result_dir, mod, param_dict, mod_name):
  RelayVisualizer(
      relay_mod=mod,
      relay_param=param_dict,
      plotter=DotPlotter(),
      parser=DotVizParser(),
  ).render(f"{result_dir}/{mod_name}")

  with open(f"{result_dir}/{mod_name}.txt", "w") as f:
    f.write(pretty_print(mod))

def generate_graph_executor(test_name, mod, param_dict, ref_dir):
  print("\n" + "="*40)
  print("GENERATING GRAPH EXECUTOR")
  print("="*40)
  with tvm.transform.PassContext(opt_level=0, config={"tir.disable_vectorize": True}):
    mod = tvm.relay.build(mod, target="c", params=param_dict, executor=Executor("graph"), runtime=Runtime("cpp"))
    mod.export_library(f"{ref_dir}/{test_name}.so")


def generate_aot_c_code(test_name, mod, param_dict, ref_dir):
  print("\n" + "="*40)
  print("GENERATING AOT C CODE")
  print("="*40)
  try:
    with tvm.transform.PassContext(opt_level=0, config={
      "tir.disable_vectorize": True
      }):
      te_compiler.get().clear()

      # Configure AOT executor with C interface for microcontroller deployment
      # executor = Executor("aot", {
      #     "interface-api": "c",
      #     "unpacked-api": True,
      #     "workspace-byte-alignment": 8,
      #     "link-params": False
      # })

      executor = Executor("aot")
      runtime = Runtime("cpp", {"system-lib": False})

      print(f"Building with AOT configuration for {test_name}")

      # Create external kernel module for tvmgen_default_imcflow_main_4
      # Read the external kernel source from file
      external_kernel_path = "tvmgen_default_imcflow_main_4.cc"
      try:
        with open(external_kernel_path, "r") as f:
          external_kernel_source = f.read()
        print(f"✅ Loaded external kernel from: {external_kernel_path}")
      except FileNotFoundError:
        print(f"⚠️  Warning: External kernel file {external_kernel_path} not found")
        print("Proceeding without external kernel - this may cause compilation errors")
        external_kernel_source = None

      # Create external kernel module if source is available
      if external_kernel_source:
        try:
          # Create external kernel module using the proper TVM API
          # Based on the test file, we use CSourceModuleCreate
          external_kernel_module = tvm.runtime._ffi_api.CSourceModuleCreate(
              external_kernel_source,
              "cc",
              [],  # func_names - empty list for default
              []   # const_vars - empty list for default
          )
          print("✅ External kernel module created using CSourceModuleCreate")

          # Add external module to the IRModule
          mod_attrs = dict(mod.attrs) if mod.attrs else {}
          external_mods = mod_attrs.get("external_mods", [])
          external_mods.append(external_kernel_module)
          mod = mod.with_attr("external_mods", external_mods)
          print("✅ External kernel module attached to IRModule")

        except Exception as e:
          print(f"❌ Failed to create external kernel module: {e}")
          print("Proceeding without external kernel")

      # Build with AOT configuration
      aot_lib = tvm.relay.build(mod, target="c", params=param_dict,
                                executor=executor, runtime=runtime)

      print(f"Compiling with AOT configuration for {test_name}")

      # Extract and save the AOT C source code
      try:
        aot_underlying_lib = aot_lib.get_lib()
        aot_c_source = aot_underlying_lib.get_source("c")
        with open(f"{ref_dir}/{test_name}_aot_generated.c", "w") as f:
          f.write(aot_c_source)
        print(f"✅ AOT C source saved: {ref_dir}/{test_name}_aot_generated.c")

      except Exception as e:
        print(f"❌ Could not extract AOT C source: {e}")

      # Export AOT library
      mlf_tar_path = f"{ref_dir}/{test_name}_aot.tar"
      export_model_library_format(aot_lib, mlf_tar_path)
      with tarfile.open(mlf_tar_path, "r") as tar:
        tar.extractall(f"{ref_dir}/mlf_extracted")
      print(f"✅ Generated AOT library: {mlf_tar_path}")

      # Copy external kernel files to output directory
      if external_kernel_source:
        try:
          # Copy the .cc file
          with open(f"{ref_dir}/tvmgen_default_imcflow_main_4.cc", "w") as f:
            f.write(external_kernel_source)
          print(f"✅ External kernel saved: {ref_dir}/tvmgen_default_imcflow_main_4.cc")

          # Copy the .h file if it exists
          header_path = "tvmgen_default_imcflow_main_4.h"
          try:
            with open(header_path, "r") as f:
              header_content = f.read()
            with open(f"{ref_dir}/tvmgen_default_imcflow_main_4.h", "w") as f:
              f.write(header_content)
            print(f"✅ External kernel header saved: {ref_dir}/tvmgen_default_imcflow_main_4.h")
          except FileNotFoundError:
            print(f"⚠️  Header file {header_path} not found - skipping")

        except Exception as e:
          print(f"❌ Failed to copy external kernel files: {e}")

  except Exception as e:
    print(f"❌ AOT generation failed: {e}")
    print("This might be due to model complexity or unsupported operations in AOT mode")


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

  generate_graph_executor(test_name, ref_mod, param_dict, ref_dir)
  generate_aot_c_code(test_name, ref_mod, param_dict, ref_dir)

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

  # print(f"nodemap: {config.HWNodeMap}")
  # print(f"edgeinfo: {config.TensorEdgetoInfo}")
  # print(f"idtoedge: {config.TensorIDtoEdge}")

  CodegenSuite = imcflow_codegen.CodegenSuite(f"{eval_dir}/build")
  CodegenSuite(eval_mod)

  print(f"mem_layout: {config.MemLayout}")
  print(f"Evaluation generation completed for {test_name}")

  generate_graph_executor(test_name, eval_mod, eval_param_dict, eval_dir)
  generate_aot_c_code(test_name, eval_mod, eval_param_dict, eval_dir)


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
