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
IMCFlow Compiler Driver v2 — single-phase, column-disable support.

Differences from v1 (imcflow_compiler_driver.py):
  - Single phase (no Phase-1/Phase-2 split)
  - Random IMCU assignment instead of ILP-optimised placement
  - Column-disable support: noisy IMCU columns are skipped
  - One atomic qconv per imcflow function (same as v1 single_qconv)
"""

import os
import random
import pickle
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
from tvm.relay.op.contrib.imcflow import HashToCustomID, CustomIDToName, CustomIDToNode, CustomIDInFunc
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig, NodeID
from tvm.micro import export_model_library_format
from tvm.contrib.relay_viz import RelayVisualizer, DotPlotter, DotVizParser
from tvm.relay.backend.contrib.imcflow.joint_pnr_ilp import (
    Coord,
    run_joint_pnr_and_update_config,
    build_policy_tables_from_pnr_result,
    construct_noc_paths_from_pnr_results,
)
from tvm.relay.backend.contrib.imcflow.transform import (
    dissolve_imcflow_functions,
    partitionImcflowSubGraph_qconv_only,
    apply_column_disable_shape_pass,
)
from tvm.relay.backend.contrib.imcflow.transform_utils import isImcflowFunc


def _print_model(result_dir, mod, param_dict, mod_name):
    # The .txt dump is cheap and always produced.
    with open(f"{result_dir}/{mod_name}.txt", "w") as f:
        f.write(mod.astext(show_meta_data=True))

    # Skip the graphviz PDF render when the graph is too large — dot can hang
    # for many minutes on dense graphs (e.g. column-disable with effective_oc=1
    # explodes a single conv into hundreds of sub-functions). Set
    # IMCFLOW_SKIP_RELAY_VIZ=1 to always skip; otherwise auto-skip past the
    # threshold.
    skip_viz = os.getenv("IMCFLOW_SKIP_RELAY_VIZ", "0") == "1"
    num_funcs = len(mod.functions)
    if skip_viz or num_funcs > 50:
        if not skip_viz:
            print(f"[v2] Skipping graphviz render for {mod_name} "
                  f"({num_funcs} functions > 50; set IMCFLOW_SKIP_RELAY_VIZ=1 to silence)")
        return
    RelayVisualizer(
        relay_mod=mod,
        relay_param=param_dict,
        plotter=DotPlotter(),
        parser=DotVizParser(),
    ).render(f"{result_dir}/{mod_name}")


# ========================================================================
# Random IMCE assignment helpers
# ========================================================================

def _build_random_func_to_imce(mod, seed=None):
    """Assign each imcflow function a random IMCE.

    The candidate IMCE pool is restricted to ``DevConfig().get_active_imce_ids()``,
    which honors the column-disable JSON: when the JSON only mentions a subset of
    cores, that subset becomes the active pool. Multiple functions may collide on
    the same IMCE; the per-function PnR runs serialize execution.

    Returns:
        func_to_imce: dict  func_name (str) -> imce_linear_id (int)
    """
    rng = random.Random(seed)
    active_ids = DevConfig().get_active_imce_ids()
    print(f"[v2] Active IMCE set: {active_ids}")
    func_to_imce = {}
    for gv in sorted(mod.functions.keys(), key=lambda g: g.name_hint):
        func = mod[gv]
        if isinstance(func, relay.Function) and isImcflowFunc(func, mod):
            func_to_imce[gv.name_hint] = rng.choice(active_ids)
    return func_to_imce


def _build_preassigned_from_func_to_imce(mod, func_to_imce):
    """Convert func_to_imce into the preassigned_placements dict expected by
    run_joint_pnr_and_update_config.

    Walks each imcflow function after re-annotation (so custom_ids are final)
    and maps them to Coord via func_to_imce.

    Returns:
        preassigned: dict  func_name -> {custom_id (int) -> Coord}
    """
    class _QConvCustomIdFinder(relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.qconv_custom_ids = []

        def visit_call(self, call):
            super().visit_call(call)
            if isinstance(call.op, tvm.ir.Op) and call.op.name == "nn.imcflow_qconv":
                if call.attrs and hasattr(call.attrs, "custom_id"):
                    self.qconv_custom_ids.append(int(call.attrs["custom_id"]))

    preassigned = {}
    for gv in sorted(mod.functions.keys(), key=lambda g: g.name_hint):
        func = mod[gv]
        if not (isinstance(func, relay.Function) and isImcflowFunc(func, mod)):
            continue
        func_name = gv.name_hint
        if func_name not in func_to_imce:
            continue
        imce_linear = func_to_imce[func_name]
        row = imce_linear // DevConfig.IMCE_W_NUM
        col_in_imce = imce_linear % DevConfig.IMCE_W_NUM
        coord = Coord(row, col_in_imce + 1)  # PnR col: 0=INODE, 1-4=IMCE

        finder = _QConvCustomIdFinder()
        finder.visit(func)
        for cid in finder.qconv_custom_ids:
            preassigned.setdefault(func_name, {})[cid] = coord
            print(f"[v2] preassigned func={func_name} custom_id={cid} -> "
                  f"IMCE({row},{col_in_imce}) linear={imce_linear}")

    return preassigned


# ========================================================================
# Main entry point
# ========================================================================

def compile_for_imcflow_v2(
    mod,
    param_dict,
    output_dir,
    column_disable_config_path=None,
    num_disable_columns=8,
    skip_codegen=False,
    save_intermediate=True,
    random_seed=None,
):
    """Single-phase compilation for IMCFlow with column-disable support.

    Args:
        mod: TVM IRModule
        param_dict: model parameters
        output_dir: directory for all outputs
        column_disable_config_path: JSON file with per-IMCE disabled column indices.
            If None, no column-disable is applied (effective_oc=64).
        num_disable_columns: number of disabled columns per IMCE (default 8).
        skip_codegen: if True, skip codegen + graph executor generation.
        save_intermediate: if True, dump intermediate relay at each step.
        random_seed: optional seed for reproducible IMCE assignment.

    Returns:
        (mod, param_dict, tar_path)
    """
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 0. Configuration
    # ------------------------------------------------------------------
    config = DevConfig()
    config.clear()
    HashToCustomID().clear()
    CustomIDToName().clear()
    CustomIDToNode().clear()
    CustomIDInFunc().clear()

    os.environ["IMCFLOW_EVAL_DIR"] = output_dir

    column_disable_enabled = column_disable_config_path is not None
    if column_disable_enabled:
        config.load_column_disable_config(column_disable_config_path, num_disable_columns)
    effective_oc = config.get_effective_oc() if column_disable_enabled else 64
    print(f"[v2] effective_oc={effective_oc}  column_disable={column_disable_enabled}")

    def _save(name):
        if save_intermediate:
            _print_model(output_dir, mod, param_dict, name)

    # ------------------------------------------------------------------
    # 1. Bind parameters
    # ------------------------------------------------------------------
    mod["main"] = bind_params_by_name(mod["main"], param_dict)
    mod = transform.InferType()(mod)
    _save("01_after_bind")

    # ------------------------------------------------------------------
    # 2. L1 partition (all supported ops -> imcflow)
    # ------------------------------------------------------------------
    mod = imcflow_transform.partitionImcflowSubGraph(mod)
    _save("02_after_L1_partition")

    # ------------------------------------------------------------------
    # 3. Split conv to atomic with effective_oc
    # ------------------------------------------------------------------
    mod, param_dict = imcflow_transform.split_conv_to_atomic(mod, param_dict, effective_oc)
    _save("03_after_atom_split")

    # ------------------------------------------------------------------
    # 4. Flatten imcflow functions back into @main
    # ------------------------------------------------------------------
    mod = imcflow.flattenImcflowTopFuncs(mod)
    _save("04_after_flatten")

    # ------------------------------------------------------------------
    # 5. Dissolve (remove all imcflow partitioning)
    # ------------------------------------------------------------------
    mod = dissolve_imcflow_functions(mod)
    _save("05_after_dissolve")

    # ------------------------------------------------------------------
    # 6. qconv-only partition (one qconv per imcflow function)
    # ------------------------------------------------------------------
    mod = partitionImcflowSubGraph_qconv_only(mod)
    _save("06_after_qconv_only_partition")

    # ------------------------------------------------------------------
    # 7. Annotate custom IDs + construct useful mappings
    # ------------------------------------------------------------------
    mod = imcflow_transform.annotateCustomId(mod)
    imcflow_transform.constructUsefulMappings(mod)
    _save("07_after_annotate")

    # ------------------------------------------------------------------
    # 8. Random IMCE assignment
    # ------------------------------------------------------------------
    func_to_imce = _build_random_func_to_imce(mod, seed=random_seed)
    if save_intermediate:
        with open(f"{output_dir}/func_to_imce.txt", "w") as f:
            pprint.pprint(func_to_imce, stream=f)

    # ------------------------------------------------------------------
    # 9. Column disable shape pass (if enabled)
    # ------------------------------------------------------------------
    func_column_info = {}
    if column_disable_enabled:
        mod, func_column_info = apply_column_disable_shape_pass(mod, func_to_imce)
        _save("09_after_column_disable")
        if save_intermediate:
            with open(f"{output_dir}/func_column_info.txt", "w") as f:
                pprint.pprint(func_column_info, stream=f)

    # ------------------------------------------------------------------
    # 10. Layout legalization
    # ------------------------------------------------------------------
    mod, ttype_map = imcflow_transform.legalizeImcflowLayout(mod)
    _save("10_after_layout_legalize")

    # ------------------------------------------------------------------
    # 11. Re-annotate + construct all mappings (custom_ids changed by layout)
    # ------------------------------------------------------------------
    mod = imcflow_transform.annotateCustomId(mod)
    imcflow_transform.constructUsefulMappings(mod)
    imcflow_transform.constructCustomIDInFunc(mod)
    imcflow_transform.constructImcflowFuncMap(mod)
    imcflow_transform.constructSplitInfo(mod)
    _save("11_after_reannotate")

    if save_intermediate:
        with open(f"{output_dir}/custom_id_to_name.txt", "w") as f:
            pprint.pprint(CustomIDToName(), stream=f)
        with open(f"{output_dir}/func_map.txt", "w") as f:
            pprint.pprint(config.ImcflowFuncMap, stream=f)

    # ------------------------------------------------------------------
    # 12. Build preassigned placements from func_to_imce (uses new custom_ids)
    # ------------------------------------------------------------------
    preassigned = _build_preassigned_from_func_to_imce(mod, func_to_imce)
    if save_intermediate:
        with open(f"{output_dir}/preassigned_placements.txt", "w") as f:
            pprint.pprint(preassigned, stream=f)

    # ------------------------------------------------------------------
    # 13. Tensor edge list + Joint PnR with preassigned placements
    # ------------------------------------------------------------------
    print("[v2] Constructing tensor edge list...")
    imcflow_transform.constructTensorEdgeList(mod)
    imcflow_transform.constructTensorIDToTensorEdgeDict()

    if save_intermediate:
        with open(f"{output_dir}/tensor_edge_list.txt", "w") as f:
            for key, paths in config.TensorEdgeListDict.items():
                print(key, file=f)
                for path in paths:
                    print(path, file=f)

    print("[v2] Running Joint PnR with random preassigned placements...")
    pnr_results = run_joint_pnr_and_update_config(
        mod, config.TensorEdgeListDict, preassigned_placements=preassigned
    )
    if save_intermediate:
        with open(f"{output_dir}/hw_node_map.txt", "w") as f:
            pprint.pprint(config.HWNodeMap, stream=f)

    # ------------------------------------------------------------------
    # 14. Post-PnR infrastructure
    # ------------------------------------------------------------------
    imcflow_transform.constructActiveIMCEDict(mod)

    construct_noc_paths_from_pnr_results(pnr_results, config.TensorEdgeListDict)
    if save_intermediate:
        with open(f"{output_dir}/noc_paths.txt", "w") as f:
            for key, paths in config.NoCPaths.items():
                print(key, file=f)
                for k, v in paths.items():
                    print(k, v, file=f)

    imcflow_transform.MemoryAllocator().run(mod, ttype_map)

    for func_name, pnr_result in pnr_results.items():
        if not pnr_result.success:
            raise RuntimeError(f"Joint PnR failed for {func_name}: {pnr_result.solver_status}")
        noc_paths = config.NoCPaths.get(func_name, {})
        build_policy_tables_from_pnr_result(pnr_result, func_name, noc_paths)
    if save_intermediate:
        with open(f"{output_dir}/policy_table.txt", "w") as f:
            f.write(config.format_policy_table())

    # NoC visualizations
    imcflow_transform.generateNoCVisualizations(mod, output_dir + "/noc_visualizations")

    # FIFO conflict monitoring
    fifo_monitor = imcflow_transform.FIFOConflictMonitor()
    fifo_monitor.run(mod)
    fifo_monitor.print_conflict_summary()
    fifo_monitor.export_conflict_table(f"{output_dir}/fifo_conflict_table.txt")

    # Deadlock detection
    deadlock_detector = imcflow_transform.NoCDeadlockDetector()
    deadlock_detector.run(mod)
    deadlock_detector.print_deadlock_summary()
    deadlock_detector.export_deadlock_table(f"{output_dir}/noc_deadlock_table.txt")

    # Save devconfig state
    devconfig_state_path = os.path.join(output_dir, "devconfig_state.pkl")
    config.save_state(devconfig_state_path)

    if save_intermediate:
        with open(f"{output_dir}/final_config_HWNodeMap.txt", "w") as f:
            pprint.pprint(config.HWNodeMap, stream=f)
        with open(f"{output_dir}/final_config_MemLayout.txt", "w") as f:
            pprint.pprint(config.MemLayout, stream=f)

    _save("14_final_transformed")

    # ------------------------------------------------------------------
    # 15. Codegen + graph executor
    # ------------------------------------------------------------------
    if skip_codegen:
        return mod, param_dict, None

    print("\n--- Running IMCFlow Codegen (v2) ---")
    config.single_qconv = True

    codegen_suite = imcflow_codegen.CodegenSuite(
        output_dir, mod, host_isa=DevConfig.HOST_ISA, rebuild_modified_cpp=False
    )
    codegen_suite(mod)
    imcflow_transform.constructDataBlockDict(mod, update_compiled_blocks_only=False)
    config.update_datablocks_state(devconfig_state_path)

    # Saturating arithmetic for CPU-side int16 ops
    from tvm.relay.backend.contrib.imcflow.cpu_run import apply_saturating_arithmetic
    mod = apply_saturating_arithmetic(mod)
    print("[v2] Applied saturating arithmetic to CPU-side int16 ops")

    # Graph executor
    print("\n--- Generating Graph Executor (v2) ---")
    executor_cfg = Executor("graph")
    runtime_cfg = Runtime("crt", {"system-lib": True})
    with tvm.transform.PassContext(opt_level=0, config={"tir.disable_vectorize": True}):
        module = tvm.relay.build(
            mod, target="c", params=param_dict,
            executor=executor_cfg, runtime=runtime_cfg,
        )
    tar_name = "lib_graph_system-lib.tar"
    tar_path = os.path.join(output_dir, tar_name)
    export_model_library_format(module, tar_path)
    print(f"[v2] Graph executor saved to {tar_path}")

    return mod, param_dict, tar_path
