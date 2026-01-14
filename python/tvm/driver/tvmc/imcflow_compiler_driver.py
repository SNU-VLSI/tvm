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


def printModel(dir, mod, param_dict, name):
    """Save model to file for debugging."""
    os.makedirs(dir, exist_ok=True)
    with open(f"{dir}/{name}.relay", "w") as f:
        f.write(relay.pretty_print(mod))


def transform_model_for_imcflow(mod, param_dict, output_dir, save_intermediate=True):
    """
    Transform a Relay module for IMCFlow hardware deployment.

    This function applies the full transformation pipeline:
    1. Bind parameters
    2. Partition IMCFlow subgraph
    3. Merge composites for partition (optional, for simpler converge detection)
    4. Partition into rounds
    5. Split conv to atomic ops
    6. Merge composite ops
    7. Handle split/concat dependencies
    8. Apply various hardware mappings and optimizations

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
    mod["main"] = bind_params_by_name(mod["main"], param_dict)
    mod = transform.InferType()(mod)
    _print("1_after_bind")

    # Step 2: First level imcflow graph partition
    mod = imcflow_transform.partitionImcflowSubGraph(mod)
    _print("2_after_L1_partition")

    # Step 3: Split imcflow function conv to atomic ops
    mod, param_dict = imcflow_transform.split_conv_to_atomic(mod, param_dict)
    _print("2_after_atom_split")

    # Step 4: Merge composite OPs
    mod = imcflow_transform.merge_composite_ops(mod)
    _print("3_after_merge")

    # Step 5: Make split and concat super node
    mod = imcflow_transform.makeSplitConcatDepsRegions(mod)
    _print("4_after_split_concat_partition")

    # Step 6: Concat distributor
    mod = imcflow_transform.ConcatDistributor(max_inputs=4).run(mod)
    _print("4.5_after_concat_distributor")

    # Step 7: Partition into rounds
    mod = imcflow_transform.partitionRound(mod)
    _print("5_after_annot")

    # Step 8: Flatten top-level functions
    mod = imcflow.flattenImcflowTopFuncs(mod)
    _print("6_after_flatten")

    # Step 9: Prune subgraphs
    mod = imcflow.prune_imcflow_subgraphs(mod)
    _print("7_after_prune_model")

    # Step 10: Annotate custom IDs
    mod = imcflow_transform.annotateCustomId(mod)
    _print("7.5_after_annotate_custom_id")
    imcflow_transform.constructUsefulMappings(mod)

    if save_intermediate:
        with open(f"{output_dir}/custom_id_to_name.txt", "w") as f:
            pprint.pprint(imcflow.CustomIDToName(), stream=f)
        with open(f"{output_dir}/node_to_custom_id.txt", "w") as f:
            pprint.pprint(HashToCustomID(), stream=f)

    _print("7.6_with_custom_id")

    # Step 11: Legalize layout
    mod, ttype_map = imcflow_transform.legalizeImcflowLayout(mod)
    _print("7.7_after_mark_in_out")

    # Step 12: Re-annotate custom IDs
    mod = imcflow_transform.annotateCustomId(mod)
    _print("8.5_after_annotate_custom_id")

    # Step 13: Construct mappings
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

    _print("9_with_custom_id")

    # Step 14: Node mapping
    imcflow_transform.NodeMapper().run(mod)
    if save_intermediate:
        with open(f"{output_dir}/hw_node_map.txt", "w") as f:
            pprint.pprint(DevConfig().HWNodeMap, stream=f)

    # Step 15: Construct tensor edge list
    imcflow_transform.constructTensorEdgeList(mod)
    if save_intermediate:
        with open(f"{output_dir}/tensor_edge_list.txt", "w") as f:
            for key, paths in DevConfig().TensorEdgeListDict.items():
                print(key, file=f)
                for path in paths:
                    print(path, file=f)

    # Step 16: Active IMCE
    imcflow_transform.constructActiveIMCEDict(mod)
    if save_intermediate:
        with open(f"{output_dir}/active_imce_list.txt", "w") as f:
            pprint.pprint(DevConfig().ActiveIMCEPerFunc, stream=f)

    # Step 17: Tensor ID to edge
    imcflow_transform.constructTensorIDToTensorEdgeDict()
    if save_intermediate:
        with open(f"{output_dir}/tensor_id_to_edge.txt", "w") as f:
            for key, paths in DevConfig().TensorIDtoEdge.items():
                print(f"{key} : {paths}", file=f)

    # Step 18: NoC paths
    imcflow_transform.constructNoCPathDict(mod)
    if save_intermediate:
        with open(f"{output_dir}/noc_paths.txt", "w") as f:
            for key, paths in DevConfig().NoCPaths.items():
                print(key, file=f)
                for k, v in paths.items():
                    print(k, v, file=f)

    # Step 19: Memory allocation
    imcflow_transform.MemoryAllocator().run(mod, ttype_map)
    if save_intermediate:
        with open(f"{output_dir}/mem_layout.txt", "w") as f:
            pprint.pprint(DevConfig().MemLayout, stream=f)

    # Step 20: Policy table generation
    imcflow_transform.PolicyTableGenerator(DevConfig().NoCPaths).run(mod)
    if save_intermediate:
        with open(f"{output_dir}/policy_table.txt", "w") as f:
            f.write(DevConfig().format_policy_table())

    # Step 21: NoC visualizations
    imcflow_transform.generateNoCVisualizations(mod, output_dir + "/noc_visualizations")

    # Step 22: FIFO conflict monitoring
    fifo_monitor = imcflow_transform.FIFOConflictMonitor()
    fifo_monitor.run(mod)
    fifo_monitor.print_conflict_summary()
    fifo_monitor.export_conflict_table(f"{output_dir}/fifo_conflict_table.txt")

    # Step 23: Deadlock detection
    deadlock_detector = imcflow_transform.NoCDeadlockDetector()
    deadlock_detector.run(mod)
    deadlock_detector.print_deadlock_summary()
    deadlock_detector.export_deadlock_table(f"{output_dir}/noc_deadlock_table.txt")

    # Step 24: Export final config
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

    return mod, param_dict


def run_imcflow_codegen(mod, output_dir):
    """
    Run IMCFLOW codegen to generate hardware deployment code.

    Args:
        mod: Transformed TVM IRModule
        output_dir: Directory to save generated code
    """
    config = DevConfig()

    CodegenSuite = imcflow_codegen.CodegenSuite(
        output_dir, mod, host_isa=DevConfig().HOST_ISA
    )
    CodegenSuite(mod)
    print(f"mem_layout: {config.MemLayout}")

    imcflow_transform.constructDataBlockDict(mod)
    print(f"data_blocks: {config.DataBlocks}")


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
    run_imcflow_codegen(mod, output_dir)

    # Step 3: Generate graph executor
    print("\n--- Generating Graph Executor ---")
    _, tar_path = generate_graph_executor(mod, param_dict, output_dir)

    return mod, param_dict, tar_path
