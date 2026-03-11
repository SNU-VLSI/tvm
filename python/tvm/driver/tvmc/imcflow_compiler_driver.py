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
import copy
import pickle
import pprint
import numpy as np
import sys

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
    construct_noc_paths_from_pnr_results,
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
    os.environ['IMCFLOW_EVAL_DIR'] = output_dir

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
    imcflow_transform.constructSplitInfo(mod)

    if save_intermediate:
        with open(f"{output_dir}/custom_id_to_name.txt", "w") as f:
            pprint.pprint(imcflow.CustomIDToName(), stream=f)
        with open(f"{output_dir}/node_to_custom_id.txt", "w") as f:
            pprint.pprint(HashToCustomID(), stream=f)
        with open(f"{output_dir}/func_map.txt", "w") as f:
            pprint.pprint(DevConfig().ImcflowFuncMap, stream=f)
        with open(f"{output_dir}/split_info.txt", "w") as f:
            pprint.pprint(DevConfig().SplitInfo, stream=f)

    _print("15_with_mappings")

    # ========================================================================
    # NEW PIPELINE: Joint PnR ILP Integration
    # ========================================================================

    # Step 16a: Construct tensor edge list FIRST (provides input for Joint PnR)
    # Note: TensorEdgeList is constructed from logical edges, does NOT require HWNodeMap
    print("[TensorEdgeList] Constructing tensor edge list...")
    imcflow_transform.constructTensorEdgeList(mod)
    if save_intermediate:
        with open(f"{output_dir}/tensor_edge_list.txt", "w") as f:
            for key, paths in DevConfig().TensorEdgeListDict.items():
                print(key, file=f)
                for path in paths:
                    print(path, file=f)

    # Step 16b: Tensor ID to edge mapping
    imcflow_transform.constructTensorIDToTensorEdgeDict()
    if save_intermediate:
        with open(f"{output_dir}/tensor_id_to_edge.txt", "w") as f:
            for key, paths in DevConfig().TensorIDtoEdge.items():
                print(f"{key} : {paths}", file=f)

    # Step 17: Joint PnR ILP (mapping + routing simultaneously)
    print("[Joint PnR] Running Joint Place & Route ILP...")
    pnr_results = run_joint_pnr_and_update_config(mod, DevConfig().TensorEdgeListDict)
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

    # Step 18: Active IMCE (requires HWNodeMap from Joint PnR)
    imcflow_transform.constructActiveIMCEDict(mod)
    if save_intermediate:
        with open(f"{output_dir}/active_imce_list.txt", "w") as f:
            pprint.pprint(DevConfig().ActiveIMCEPerFunc, stream=f)

    # Step 19: NoCPathDict from Joint PnR results (replaces legacy constructNoCPathDict)
    # Uses Joint PnR mapping instead of getInnerNodeID() lookup (works with composite nodes)
    construct_noc_paths_from_pnr_results(pnr_results, DevConfig().TensorEdgeListDict)
    if save_intermediate:
        with open(f"{output_dir}/noc_paths.txt", "w") as f:
            for key, paths in DevConfig().NoCPaths.items():
                print(key, file=f)
                for k, v in paths.items():
                    print(k, v, file=f)

    # Step 20: Memory allocation (tensor memory)
    imcflow_transform.MemoryAllocator().run(mod, ttype_map)
    if save_intermediate:
        with open(f"{output_dir}/mem_layout.txt", "w") as f:
            pprint.pprint(DevConfig().MemLayout, stream=f)

    # Step 21: Policy table generation
    # Uses Joint ILP routes via PolicyTableBuilder (MCF router removed)
    # noc_paths contains both TensorEdge paths and instruction paths (NodeID keys)
    for func_name, pnr_result in pnr_results.items():
        if not pnr_result.success:
            raise RuntimeError(f"Joint PnR failed for {func_name}: {pnr_result.solver_status}")
        noc_paths = DevConfig().NoCPaths.get(func_name, {})
        build_policy_tables_from_pnr_result(
            pnr_result,
            func_name,
            noc_paths,
        )
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


def run_imcflow_codegen(mod, output_dir, save_intermediate=True, rebuild_cpp_only=False, stop_at_codegen=False):
    """
    Run IMCFLOW codegen to generate hardware deployment code.

    Args:
        mod: Transformed TVM IRModule
        output_dir: Directory to save generated code
        save_intermediate: If True, save intermediate files during codegen
        rebuild_cpp_only: If True, just rebuild modified C++ files before simulation
        stop_at_codegen: If True, return early after codegen (don't build data blocks)

    Returns:
        dict: Status information. Returns {"status": "codegen_complete"} if stop_at_codegen=True.
    """
    config = DevConfig()

    CodegenSuite = imcflow_codegen.CodegenSuite(
        output_dir, mod, host_isa=DevConfig().HOST_ISA, rebuild_modified_cpp=rebuild_cpp_only
    )
    CodegenSuite(mod)
    print(f"mem_layout: {config.MemLayout}")
    if save_intermediate:
        with open(f"{output_dir}/mem_layout.txt", "w") as f:
            pprint.pprint(DevConfig().MemLayout, stream=f)

    if stop_at_codegen:
        print("\n⏭️  Codegen only mode: returning after codegen")
        return {"status": "codegen_complete"}

    imcflow_transform.constructDataBlockDict(mod, update_compiled_blocks_only=rebuild_cpp_only)
    print(f"data_blocks: {config.DataBlocks}")

    # Update the DevConfig's DataBlocks state after codegen
    devconfig_state_path = os.path.join(output_dir, "devconfig_state.pkl")
    config.update_datablocks_state(devconfig_state_path)

    return {"status": "complete"}


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
    run_imcflow_codegen(mod, output_dir, save_intermediate=True, stop_at_codegen=False)

    # Step 3: Generate graph executor
    print("\n--- Generating Graph Executor ---")
    _, tar_path = generate_graph_executor(mod, param_dict, output_dir)

    return mod, param_dict, tar_path


def rebuild_imcflow_cpp_only(mod, param_dict, output_dir, stop_at_codegen=False):
    """
    Rebuild modified IMCFlow C++ files only.

    This function is used when only C++ files have been modified and need to be rebuilt
    without going through the full transformation and codegen pipeline.

    Args:
        mod: TVM IRModule
        param_dict: Dictionary of model parameters
        output_dir: Directory where the model and C++ files are located
        stop_at_codegen: If True, stop after codegen and don't generate graph executor

    Returns:
        Tuple of (mod, param_dict, tar_path). tar_path is None if stop_at_codegen=True.
    """
    print("\n--- Rebuilding Modified IMCFlow C++ Files Only ---")
    result = run_imcflow_codegen(mod, output_dir, save_intermediate=True, rebuild_cpp_only=True, stop_at_codegen=stop_at_codegen)

    if stop_at_codegen:
        return mod, param_dict, None

    print("\n--- Generating Graph Executor ---")
    _, tar_path = generate_graph_executor(mod, param_dict, output_dir)

    return mod, param_dict, tar_path


# ========================================================================
# Single-QConv-Per-Round Compilation Mode
# ========================================================================

def extract_qconv_placements(mod):
    """
    Extract placement info for all qconv nodes after Phase 1 (normal compilation).

    Walks all imcflow functions, finds qconv call nodes (bare or inside composites),
    assigns topological indices, and maps them to their PnR placement via HWNodeMap.

    Returns:
        Dict[int, tuple] mapping topo_index -> (row, col_in_imce) placement coord
    """
    from tvm.relay.op.contrib.imcflow import HashToCustomID as _HashToCustomID
    from tvm.relay.backend.contrib.imcflow.transform import get_qconv_topo_indices

    config = DevConfig()
    hw_node_map = config.HWNodeMap

    call_to_idx, idx_to_call = get_qconv_topo_indices(mod)

    # Collect placed qconvs in topo order, then re-index from 0.
    # Phase 1's final module may have extra qconvs from round partition structure,
    # but only qconvs with HWNodeMap entries are actual placed qconvs.
    # Re-indexing from 0 ensures indices match Phase 2's post-split_conv_to_atomic indices.
    placed_list = []  # list of (row, col_in_imce) in topo order
    hash_to_id = _HashToCustomID()

    for topo_idx in sorted(call_to_idx.values()):
        call = idx_to_call[topo_idx]
        custom_id = hash_to_id.get(int(hash(call)))

        if custom_id is not None and custom_id in hw_node_map:
            node_id = hw_node_map[custom_id]
            # NodeID is an Enum; to_coord() returns (row, col) where col=0 is INODE, col=1-4 is IMCE
            row, col = node_id.to_coord()
            col_in_imce = col - 1  # IMCE col index (0-based)
            placed_list.append((row, col_in_imce))
            print(f"[extract_qconv_placements] orig_topo_idx={topo_idx} -> custom_id={custom_id} -> ({row}, {col_in_imce})")

    # Re-index placed qconvs from 0
    placements = {i: coord for i, coord in enumerate(placed_list)}
    print(f"[extract_qconv_placements] Extracted {len(placements)} qconv placements (re-indexed from 0)")
    return placements


def transform_model_single_qconv(mod, param_dict, output_dir, placements, top_output_dir=None, save_intermediate=True):
    """
    Phase 2 pipeline: Transform model so each imcflow function has exactly one atomic qconv.
    All other ops (dwconv, bias_add, relu, etc.) stay on CPU.

    Args:
        mod: TVM IRModule
        param_dict: Dictionary of model parameters
        output_dir: Directory to save intermediate results
        placements: Dict[int, tuple] from extract_qconv_placements (topo_idx -> (row, col_in_imce))
        top_output_dir: Top-level output directory for files that downstream tools expect
                        at the model's eval dir (e.g., devconfig_state.pkl). Defaults to output_dir.
        save_intermediate: Whether to save intermediate states
    """
    from tvm.relay.backend.contrib.imcflow.joint_pnr_ilp import Coord
    from tvm.relay.backend.contrib.imcflow.transform import (
        get_qconv_topo_indices,
        dissolve_imcflow_functions,
        partitionImcflowSubGraph_qconv_only,
    )

    # Clear all singleton state from Phase 1
    DevConfig().clear()
    HashToCustomID().clear()
    imcflow.CustomIDToName().clear()
    imcflow.CustomIDToNode().clear()
    imcflow.CustomIDInFunc().clear()

    os.environ['IMCFLOW_EVAL_DIR'] = output_dir

    def _print(name):
        if save_intermediate:
            printModel(output_dir, mod, param_dict, name)

    _print("0_origin_p2")

    # Step 1: Bind parameters
    mod["main"] = bind_params_by_name(mod["main"], param_dict)
    mod = transform.InferType()(mod)
    _print("1_after_bind_p2")

    # Step 2: L1 partition (all supported ops -> imcflow, needed for split_conv_to_atomic)
    mod = imcflow_transform.partitionImcflowSubGraph(mod)
    _print("2_after_L1_partition_p2")

    # Steps 3-4: SKIP merge_composite_for_partition and partitionRound

    # Step 7: split_conv_to_atomic (must run BEFORE flatten so qconvs are inside
    # imcflow-named functions; split_conv_to_atomic only visits functions whose
    # global_symbol contains "imcflow", and flattenImcflowTopFuncs inlines those
    # functions back into @main, leaving no imcflow-named functions for the splitter
    # to find).
    mod, param_dict = imcflow_transform.split_conv_to_atomic(mod, param_dict)
    _print("7_after_atom_split_p2")

    # Step 5: Flatten (inline imcflow functions back into @main after splitting)
    mod = imcflow.flattenImcflowTopFuncs(mod)
    _print("5_after_flatten_p2")

    # Save qconv topo indices BEFORE dissolving (for placement matching)
    call_to_idx_pre, _ = get_qconv_topo_indices(mod)
    if save_intermediate:
        with open(f"{output_dir}/qconv_topo_indices_pre_dissolve.txt", "w") as f:
            for call, idx in call_to_idx_pre.items():
                f.write(f"topo_idx={idx}\n")

    # Step 9: Dissolve imcflow functions (inline back to main)
    mod = dissolve_imcflow_functions(mod)
    _print("9_after_dissolve_p2")

    # Step 10: Re-partition with only qconv supported
    mod = partitionImcflowSubGraph_qconv_only(mod)
    _print("10_after_qconv_only_partition_p2")

    # Step 11: Annotate custom IDs
    mod = imcflow_transform.annotateCustomId(mod)
    imcflow_transform.constructUsefulMappings(mod)

    if save_intermediate:
        with open(f"{output_dir}/custom_id_to_name_p2.txt", "w") as f:
            pprint.pprint(imcflow.CustomIDToName(), stream=f)
        with open(f"{output_dir}/node_to_custom_id_p2.txt", "w") as f:
            pprint.pprint(HashToCustomID(), stream=f)

    _print("12_with_custom_id_p2")

    # Step 13: Legalize layout
    mod, ttype_map = imcflow_transform.legalizeImcflowLayout(mod)
    _print("13_after_legalize_layout_p2")

    # Step 14: Re-annotate custom IDs
    mod = imcflow_transform.annotateCustomId(mod)
    _print("14_after_reannotate_custom_id_p2")

    # Step 15: Construct mappings
    imcflow_transform.constructUsefulMappings(mod)
    imcflow_transform.constructCustomIDInFunc(mod)
    imcflow_transform.constructImcflowFuncMap(mod)
    imcflow_transform.constructSplitInfo(mod)

    if save_intermediate:
        with open(f"{output_dir}/custom_id_to_name_p2.txt", "w") as f:
            pprint.pprint(imcflow.CustomIDToName(), stream=f)
        with open(f"{output_dir}/node_to_custom_id_p2.txt", "w") as f:
            pprint.pprint(HashToCustomID(), stream=f)
        with open(f"{output_dir}/func_map_p2.txt", "w") as f:
            pprint.pprint(DevConfig().ImcflowFuncMap, stream=f)
        with open(f"{output_dir}/split_info_p2.txt", "w") as f:
            pprint.pprint(DevConfig().SplitInfo, stream=f)

    _print("15_with_mappings_p2")

    # ========================================================================
    # Build pre-assigned placements for PnR
    # ========================================================================
    # Walk each imcflow function directly to find its qconv and extract custom_id from attrs.
    # Since each Phase 2 imcflow function has exactly one qconv, this is straightforward.
    # Collect (func_name, custom_id) pairs in topo order matching get_qconv_topo_indices.
    from tvm.relay.backend.contrib.imcflow.transform import isImcflowFunc

    class _QConvCustomIdFinder(relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.qconv_custom_ids = []

        def visit_call(self, call):
            super().visit_call(call)
            if isinstance(call.op, tvm.ir.Op) and call.op.name == "nn.imcflow_qconv":
                if call.attrs and hasattr(call.attrs, "custom_id"):
                    self.qconv_custom_ids.append(int(call.attrs["custom_id"]))

    # Collect qconv custom_ids per function in sorted order (matching get_qconv_topo_indices order)
    qconv_entries = []  # list of (func_name, custom_id) in topo order
    for gv in sorted(mod.functions.keys(), key=lambda g: g.name_hint):
        func = mod[gv]
        if isinstance(func, relay.Function) and isImcflowFunc(func, mod):
            finder = _QConvCustomIdFinder()
            finder.visit(func)
            for cid in finder.qconv_custom_ids:
                qconv_entries.append((gv.name_hint, cid))

    preassigned = {}  # func_name -> {graph_node_id -> Coord}
    for topo_idx, (func_name, custom_id) in enumerate(qconv_entries):
        if topo_idx in placements:
            row, col_in_imce = placements[topo_idx]
            coord = Coord(row, col_in_imce + 1)  # PnR uses col 1-4 for IMCE (col 0 = INODE)
            if func_name not in preassigned:
                preassigned[func_name] = {}
            preassigned[func_name][custom_id] = coord
            print(f"[single_qconv] topo_idx={topo_idx} -> func={func_name}, "
                  f"custom_id={custom_id} -> ({row}, {col_in_imce})")

    if save_intermediate:
        with open(f"{output_dir}/preassigned_placements_p2.txt", "w") as f:
            pprint.pprint(preassigned, stream=f)

    # ========================================================================
    # Joint PnR ILP with pre-assigned placements
    # ========================================================================
    print("[TensorEdgeList] Constructing tensor edge list...")
    imcflow_transform.constructTensorEdgeList(mod)
    if save_intermediate:
        with open(f"{output_dir}/tensor_edge_list_p2.txt", "w") as f:
            for key, paths in DevConfig().TensorEdgeListDict.items():
                print(key, file=f)
                for path in paths:
                    print(path, file=f)

    imcflow_transform.constructTensorIDToTensorEdgeDict()

    print("[Joint PnR] Running Joint Place & Route ILP with pre-assigned placements...")
    pnr_results = run_joint_pnr_and_update_config(
        mod, DevConfig().TensorEdgeListDict, preassigned_placements=preassigned
    )
    if save_intermediate:
        with open(f"{output_dir}/hw_node_map_p2.txt", "w") as f:
            pprint.pprint(DevConfig().HWNodeMap, stream=f)
        with open(f"{output_dir}/joint_pnr_results_p2.txt", "w") as f:
            for func_name, result in pnr_results.items():
                print(f"Function: {func_name}", file=f)
                print(f"  Success: {result.success}", file=f)
                print(f"  Status: {result.solver_status}", file=f)
                print(f"  Max congestion: {result.max_congestion}", file=f)
                print(f"  Total hops: {result.total_hops}", file=f)
                print("", file=f)

    # Step 18: Active IMCE
    imcflow_transform.constructActiveIMCEDict(mod)

    # Step 19: NoCPaths
    construct_noc_paths_from_pnr_results(pnr_results, DevConfig().TensorEdgeListDict)
    if save_intermediate:
        with open(f"{output_dir}/noc_paths_p2.txt", "w") as f:
            for key, paths in DevConfig().NoCPaths.items():
                print(key, file=f)
                for k, v in paths.items():
                    print(k, v, file=f)

    # Step 20: Memory allocation
    imcflow_transform.MemoryAllocator().run(mod, ttype_map)

    # Step 21: Policy table generation
    for func_name, pnr_result in pnr_results.items():
        if not pnr_result.success:
            raise RuntimeError(f"Joint PnR failed for {func_name}: {pnr_result.solver_status}")
        noc_paths = DevConfig().NoCPaths.get(func_name, {})
        build_policy_tables_from_pnr_result(pnr_result, func_name, noc_paths)
    if save_intermediate:
        with open(f"{output_dir}/policy_table_p2.txt", "w") as f:
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
    if save_intermediate:
        def _dump(title, dict_data):
            with open(f"{output_dir}/final_imcflow_config_{title}.txt", "w") as f:
                print(f"----------------------- {title} ------------------------", file=f)
                for key, value in dict_data.items():
                    pprint.pprint(f"{key} : {value}", stream=f)

        _dump("HWNodeMap", config.HWNodeMap)
        _dump("TensorEdgetoInfo", config.TensorEdgetoInfo)
        _dump("TensorIDtoEdge", config.TensorIDtoEdge)
        _dump("PolicyTableDict", config.PolicyTableDict)
        _dump("memory_layout", config.MemLayout)

    # Save devconfig_state.pkl to top-level dir (cmake/codegen expects it there)
    _top_dir = top_output_dir if top_output_dir else output_dir
    devconfig_state_path = os.path.join(_top_dir, "devconfig_state.pkl")
    config.save_state(devconfig_state_path)

    return mod, param_dict


def compile_for_imcflow_single_qconv(mod, param_dict, output_dir, skip_codegen=False):
    """
    Two-phase compilation: one atomic qconv per round.

    Phase 1: Run normal compilation to extract placement info for qconv nodes.
    Phase 2: Re-run with single-qconv-per-round partitioning using Phase 1 placements.

    Args:
        mod: TVM IRModule to compile
        param_dict: Dictionary of model parameters
        output_dir: Directory to save all outputs
        skip_codegen: If True, skip codegen and graph executor generation

    Returns:
        Tuple of (transformed_mod, param_dict, tar_path)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save original module for Phase 2 (Phase 1 mutates mod in-place via C++ objects)
    # copy.deepcopy doesn't truly deep copy TVM C++ IR, so we serialize instead
    mod_text = mod.astext(show_meta_data=True)
    param_dict_orig = {k: v.numpy() if hasattr(v, 'numpy') else np.array(v)
                       for k, v in param_dict.items()}

    # ========================================================================
    # Phase 1: Normal compilation to get placement info
    # ========================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Normal compilation for placement extraction")
    print("=" * 60)
    phase1_dir = os.path.join(output_dir, "phase1_placement")
    os.makedirs(phase1_dir, exist_ok=True)

    mod_p1, _ = transform_model_for_imcflow(mod, param_dict, phase1_dir)
    placements = extract_qconv_placements(mod_p1)

    # Save placements
    placement_path = os.path.join(output_dir, "qconv_placements.pkl")
    with open(placement_path, "wb") as f:
        pickle.dump(placements, f)
    with open(os.path.join(output_dir, "qconv_placements.txt"), "w") as f:
        pprint.pprint(placements, stream=f)
    print(f"[Phase 1] Saved {len(placements)} qconv placements to {placement_path}")

    # ========================================================================
    # Phase 2: Single-qconv-per-round compilation
    # ========================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Single-qconv-per-round compilation")
    print("=" * 60)
    phase2_dir = os.path.join(output_dir, "phase2_single_qconv")
    os.makedirs(phase2_dir, exist_ok=True)

    # Reconstruct module from saved text for Phase 2
    mod_p2_input = tvm.relay.fromtext(mod_text)
    param_dict_p2_input = {k: tvm.nd.array(v) for k, v in param_dict_orig.items()}

    mod_p2, param_dict_p2 = transform_model_single_qconv(
        mod_p2_input, param_dict_p2_input, phase2_dir, placements,
        top_output_dir=output_dir
    )

    if skip_codegen:
        return mod_p2, param_dict_p2, None

    # Codegen and graph executor output go to top-level output_dir
    # (cmake expects lib_graph_system-lib.tar and build/ in the model's eval dir, not phase2 subdir)
    print("\n--- Running IMCFlow Codegen (Phase 2) ---")
    DevConfig().single_qconv = True
    run_imcflow_codegen(mod_p2, output_dir, save_intermediate=True, stop_at_codegen=False)

    # Apply saturating arithmetic to CPU-side int16 ops (add, subtract, multiply)
    # In single-qconv mode, ops like residual add/multiply run on CPU in @main.
    # Without saturation, int16 overflow wraps instead of clamping to [-32768, 32767].
    from tvm.relay.backend.contrib.imcflow.cpu_run import apply_saturating_arithmetic
    mod_p2 = apply_saturating_arithmetic(mod_p2)
    print("[Phase 2] Applied saturating arithmetic to CPU-side int16 ops")

    # Graph executor
    print("\n--- Generating Graph Executor (Phase 2) ---")
    _, tar_path = generate_graph_executor(mod_p2, param_dict_p2, output_dir)

    return mod_p2, param_dict_p2, tar_path