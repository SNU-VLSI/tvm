# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
IMCFlow Compiler Driver

This module provides the main entry points for compiling models for IMCFlow hardware.
It includes:
- transform_model_for_imcflow: Transform a Relay module for IMCFlow deployment
- run_imcflow_codegen: Generate hardware deployment code
- generate_graph_executor: Build the graph executor for hardware deployment
"""

import os
import pprint
import numpy as np

import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.backend import Executor, Runtime
from tvm.relay.backend.contrib.imcflow import transform as imcflow_transform
from tvm.relay.backend.contrib.imcflow import codegen as imcflow_codegen
from tvm.relay.op.contrib import imcflow
from tvm.relay.op.contrib.imcflow import HashToCustomID
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from tvm.micro import export_model_library_format
from tvm.contrib.relay_viz import RelayVisualizer, DotPlotter, DotVizParser
from tvm.relay.backend.contrib.imcflow.joint_pnr_ilp import (
    run_joint_pnr_and_update_config,
    build_policy_tables_from_pnr_result,
)
from tvm.relay.backend.contrib.imcflow.policy_table_builder import PolicyTableBuilder

def printModel(result_dir, mod, param_dict, mod_name):
  RelayVisualizer(
      relay_mod=mod,
      relay_param=param_dict,
      plotter=DotPlotter(),
      parser=DotVizParser(),
  ).render(f"{result_dir}/{mod_name}")

  with open(f"{result_dir}/{mod_name}.txt", "w") as f:
    # f.write(pretty_print(mod))
    f.write(mod.astext(show_meta_data=True))

def transform_model_for_imcflow(mod, param_dict, output_dir, save_intermediate=True):
    """
    Transform a Relay module for IMCFlow hardware deployment.

    This function applies the full transformation pipeline with the new order:
    1. Bind parameters
    2. Partition IMCFlow subgraph
    3. Merge composites for partition (for simpler converge detection)
    4. Partition into rounds (with merged composites for accurate cost estimation)
    5. Unmerge composites (inline back to original ops)
    6. Split conv to atomic ops
    7. Merge composite ops (final merge)
    8. Concat distributor
    9-26. Flatten, prune, annotate, mappings, NoC, memory, etc.

    Args:
        mod: TVM IRModule to transform
        param_dict: Dictionary of model parameters
        output_dir: Directory to save intermediate results
        save_intermediate: Whether to save intermediate model states

    Returns:
        Tuple of (transformed_mod, param_dict)
    """
    DevConfig().clear()

    def _print(name):
        if save_intermediate:
            printModel(output_dir, mod, param_dict, name)

    # Step 0: Original model
    _print("0_origin")

    # Step 1: Bind parameters
    if save_intermediate:
        # Save param_dict as npz (numpy arrays)
        param_np = {k: v.numpy() if hasattr(v, 'numpy') else np.array(v)
                    for k, v in param_dict.items()}
        np.savez(f"{output_dir}/params_before_bind.npz", **param_np)

        # Also save a summary txt for quick reference
        with open(f"{output_dir}/params_before_bind_info.txt", "w") as f:
            for k, v in param_dict.items():
                f.write(f"{k}: shape={v.shape}, dtype={v.dtype}\n")

    mod["main"] = bind_params_by_name(mod["main"], param_dict)
    mod = transform.InferType()(mod)
    _print("1_after_bind")

    # Step 2: First level imcflow graph partition
    mod = imcflow_transform.partitionImcflowSubGraph(mod)
    _print("2_after_L1_partition")

    # Step 3: Merge composites for partition (simpler converge detection, accurate cost)
    mod = imcflow_transform.merge_composite_for_partition(mod)
    _print("3_after_merge_for_partition")

    # Step 4: Partition into rounds (with merged composites)
    mod = imcflow_transform.partitionRound(mod)
    _print("4_after_partition_round")

    # Step 5: Flatten top-level functions
    mod = imcflow.flattenImcflowTopFuncs(mod)
    _print("5_after_flatten")

    # Step 5: Unmerge composites (inline back to original ops for split_conv_to_atomic)
    mod = imcflow_transform.unmerge_composite(mod)
    _print("6_after_unmerge_composite")

    # Step 6: Split imcflow function conv to atomic ops
    mod, param_dict = imcflow_transform.split_conv_to_atomic(mod, param_dict)
    _print("8_after_atom_split")

    # Step 7: Merge composite OPs (final merge)
    mod = imcflow_transform.merge_composite_ops(mod)
    _print("8_after_final_merge")

    # Step 8: Concat distributor
    mod = imcflow_transform.ConcatDistributor(max_inputs=4).run(mod)
    _print("9_after_concat_distributor")

    # Step 10: Prune subgraphs
    # mod = imcflow.prune_imcflow_subgraphs(mod)
    _print("10_after_prune_model_disabled")

    # Step 11: Annotate custom IDs
    mod = imcflow_transform.annotateCustomId(mod)
    _print("11_after_annotate_custom_id")
    imcflow_transform.constructUsefulMappings(mod)

    if save_intermediate:
        with open(f"{output_dir}/custom_id_to_name.txt", "w") as f:
            pprint.pprint(imcflow.CustomIDToName(), stream=f)
        with open(f"{output_dir}/node_to_custom_id.txt", "w") as f:
            pprint.pprint(HashToCustomID(), stream=f)

    _print("12_with_custom_id")

    # Step 13: Legalize layout
    mod, ttype_map = imcflow_transform.legalizeImcflowLayout(mod)
    _print("13_after_legalize_layout")

    # Step 14: Re-annotate custom IDs
    mod = imcflow_transform.annotateCustomId(mod)
    _print("14_after_reannotate_custom_id")

    # Step 15: Construct mappings
    imcflow_transform.constructUsefulMappings(mod)
    imcflow_transform.constructCustomIDInFunc(mod)
    imcflow_transform.constructImcflowFuncMap(mod)

    if save_intermediate:
        with open(f"{output_dir}/custom_id_to_name.txt", "w") as f:
            pprint.pprint(imcflow.CustomIDToName(), stream=f)
        with open(f"{output_dir}/node_to_custom_id.txt", "w") as f:
            pprint.pprint(HashToCustomID(), stream=f)
        with open(f"{output_dir}/func_map.txt", "w") as f:
            pprint.pprint(DevConfig().ImcflowFuncMap, stream=f)

    _print("15_with_mappings")

    # ========================================================================
    # NEW PIPELINE: Joint PnR ILP Integration
    # ========================================================================

    # Step 16: Joint PnR ILP (mapping + routing simultaneously)
    print("[Joint PnR] Running Joint Place & Route ILP...")
    pnr_results = run_joint_pnr_and_update_config(mod)
    if save_intermediate:
        with open(f"{output_dir}/hw_node_map.txt", "w") as f:
            pprint.pprint(DevConfig().HWNodeMap, stream=f)
        with open(f"{output_dir}/joint_pnr_results.txt", "w") as f:
            for func_name, result in pnr_results.items():
                print(f"Function: {func_name}", file=f)
                print(f"  Success: {result.success}", file=f)
                print(f"  Status: {result.solver_status}", file=f)
                print(f"  Max congestion: {result.max_congestion}", file=f)
                print(f"  Total hops: {result.total_hops}", file=f)
                print(f"  Mappings: {len(result.mapping)}", file=f)
                print(f"  Routes: {len(result.routes)}", file=f)
                print("", file=f)

    # Step 17: Construct tensor edge list (uses HWNodeMap from Joint PnR)
    imcflow_transform.constructTensorEdgeList(mod)
    if save_intermediate:
        with open(f"{output_dir}/tensor_edge_list.txt", "w") as f:
            for key, paths in DevConfig().TensorEdgeListDict.items():
                print(key, file=f)
                for path in paths:
                    print(path, file=f)

    # Step 18: Tensor ID to edge
    imcflow_transform.constructTensorIDToTensorEdgeDict()
    if save_intermediate:
        with open(f"{output_dir}/tensor_id_to_edge.txt", "w") as f:
            for key, paths in DevConfig().TensorIDtoEdge.items():
                print(f"{key} : {paths}", file=f)

    # Step 19: Active IMCE
    imcflow_transform.constructActiveIMCEDict(mod)
    if save_intermediate:
        with open(f"{output_dir}/active_imce_list.txt", "w") as f:
            pprint.pprint(DevConfig().ActiveIMCEPerFunc, stream=f)

    # Step 20: constructNoCPathDict - SKIPPED (Joint ILP provides routes directly)
    # Note: We still need NoCPaths for PolicyTableBuilder, so construct it
    # but the actual routing came from Joint ILP
    imcflow_transform.constructNoCPathDict(mod)
    if save_intermediate:
        with open(f"{output_dir}/noc_paths.txt", "w") as f:
            for key, paths in DevConfig().NoCPaths.items():
                print(key, file=f)
                for k, v in paths.items():
                    print(k, v, file=f)

    # Step 21: Memory allocation (tensor memory)
    imcflow_transform.MemoryAllocator().run(mod, ttype_map)
    if save_intermediate:
        with open(f"{output_dir}/mem_layout.txt", "w") as f:
            pprint.pprint(DevConfig().MemLayout, stream=f)

    # Step 22: Policy table generation using PolicyTableBuilder
    # Note: For now, we still use the existing routing pipeline since
    # the Joint ILP routes need more integration work
    imcflow_transform.PolicyTableGenerator(DevConfig().NoCPaths).run(mod)
    if save_intermediate:
        with open(f"{output_dir}/policy_table.txt", "w") as f:
            f.write(DevConfig().format_policy_table())

    # Step 23: NoC visualizations
    imcflow_transform.generateNoCVisualizations(mod, output_dir + "/noc_visualizations")

    # Step 24: FIFO conflict monitoring
    fifo_monitor = imcflow_transform.FIFOConflictMonitor()
    fifo_monitor.run(mod)
    fifo_monitor.print_conflict_summary()
    fifo_monitor.export_conflict_table(f"{output_dir}/fifo_conflict_table.txt")

    # Step 25: Deadlock detection
    deadlock_detector = imcflow_transform.NoCDeadlockDetector()
    deadlock_detector.run(mod)
    deadlock_detector.print_deadlock_summary()
    deadlock_detector.export_deadlock_table(f"{output_dir}/noc_deadlock_table.txt")

    # Step 26: Export final config
    config = DevConfig()

    def _dump(title, dict_data):
        with open(f"{output_dir}/final_imcflow_config_{title}.txt", "w") as f:
            print(f"----------------------- {title} ------------------------", file=f)
            for key, value in dict_data.items():
                pprint.pprint(f"{key} : {value}", stream=f)

    if save_intermediate:
        _dump("HWNodeMap", config.HWNodeMap)
        _dump("TensorEdgetoInfo", config.TensorEdgetoInfo)
        _dump("TensorIDtoEdge", config.TensorIDtoEdge)
        _dump("PolicyTableDict", config.PolicyTableDict)
        _dump("memory_layout", config.MemLayout)

    # Save DevConfig state for rebuild_modified_cpp support
    devconfig_state_path = os.path.join(output_dir, "devconfig_state.pkl")
    config.save_state(devconfig_state_path)

    return mod, param_dict


def run_imcflow_codegen(mod, output_dir, save_intermediate=True, rebuild_modified_cpp=False):
    """
    Run IMCFLOW codegen to generate hardware deployment code.

    Args:
        mod: Transformed TVM IRModule
        output_dir: Directory to save generated code
        rebuild_modified_cpp: If True, just rebuild modified C++ files before simulation
        save_intermediate: If True, save intermediate files during codegen
    """
    config = DevConfig()

    CodegenSuite = imcflow_codegen.CodegenSuite(
        output_dir, mod, host_isa=DevConfig().HOST_ISA, rebuild_modified_cpp=rebuild_modified_cpp
    )
    CodegenSuite(mod)
    print(f"mem_layout: {config.MemLayout}")
    if save_intermediate:
        with open(f"{output_dir}/mem_layout.txt", "w") as f:
            pprint.pprint(DevConfig().MemLayout, stream=f)

    imcflow_transform.constructDataBlockDict(mod, update_compiled_blocks_only=rebuild_modified_cpp)
    print(f"data_blocks: {config.DataBlocks}")

    # Update the DevConfig's DataBlocks state after codegen
    devconfig_state_path = os.path.join(output_dir, "devconfig_state.pkl")
    config.update_datablocks_state(devconfig_state_path)


def generate_graph_executor(mod, param_dict, output_dir):
    """
    Generate graph executor for hardware deployment.

    Args:
        mod: Transformed TVM IRModule
        param_dict: Dictionary of model parameters
        output_dir: Directory to save the generated library

    Returns:
        Tuple of (module, tar_path)
    """
    executor_cfg = Executor("graph")
    runtime_cfg = Runtime("crt", {"system-lib": True})

    print("\n" + "=" * 40)
    print("GENERATING GRAPH EXECUTOR")
    print("=" * 40)

    with tvm.transform.PassContext(opt_level=0, config={"tir.disable_vectorize": True}):
        module = tvm.relay.build(
            mod,
            target="c",
            params=param_dict,
            executor=executor_cfg,
            runtime=runtime_cfg,
        )

    tar_name = "lib_graph_system-lib.tar"
    tar_path = os.path.join(output_dir, tar_name)
    export_model_library_format(module, tar_path)

    return module, tar_path


def compile_for_imcflow(mod, param_dict, output_dir, skip_codegen=False):
    """
    Full compilation pipeline for IMCFlow hardware.

    This is the main entry point for compiling a model for IMCFlow deployment.

    Args:
        mod: TVM IRModule to compile
        param_dict: Dictionary of model parameters
        output_dir: Directory to save all outputs
        skip_codegen: If True, skip codegen and graph executor generation

    Returns:
        Tuple of (transformed_mod, param_dict, tar_path)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Transform the model
    print("\n--- Transforming Model ---")
    mod, param_dict = transform_model_for_imcflow(mod, param_dict, output_dir)

    if skip_codegen:
        return mod, param_dict, None

    # Step 2: Run codegen
    print("\n--- Running IMCFlow Codegen ---")
    run_imcflow_codegen(mod, output_dir, save_intermediate=True)

    # Step 3: Generate graph executor
    print("\n--- Generating Graph Executor ---")
    _, tar_path = generate_graph_executor(mod, param_dict, output_dir)

    return mod, param_dict, tar_path

def rebuild_imcflow_cpp_only(mod, param_dict, output_dir):
    """
    Rebuild modified IMCFlow C++ files only.

    This function is used when only C++ files have been modified and need to be rebuilt
    without going through the full transformation and codegen pipeline.

    Args:
        output_dir: Directory where the model and C++ files are located
    """

    print("\n--- Rebuilding Modified IMCFlow C++ Files Only ---")
    run_imcflow_codegen(mod, output_dir, save_intermediate=True, rebuild_modified_cpp=True)

    print("\n--- Generating Graph Executor ---")
    _, tar_path = generate_graph_executor(mod, param_dict, output_dir)

    return mod, param_dict, tar_path