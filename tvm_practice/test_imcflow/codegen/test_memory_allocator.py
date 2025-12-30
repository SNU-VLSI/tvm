"""
Test script for MemoryAllocator with small graphs.

This script runs all necessary passes before MemoryAllocator and allows
debugging the allocator with minimal graphs.

Usage:
    python test_memory_allocator.py [model_name]

Available models:
    - one_conv_quant (default): Single quantized convolution
    - conv_bn_quant: Conv + BN + Quant
    - residual_model: Simple residual block (good for testing multi-input)
"""
import sys
import os
import pprint
import copy

import tvm
import numpy as np
from tvm.relay.build_module import bind_params_by_name
from tvm.relay import transform
from tvm.relay.backend.contrib.imcflow import transform as imcflow_transform
from tvm.relay.op.contrib import imcflow
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from tvm.relay.op.contrib.imcflow import HashToCustomID

# Add parent directories to path for model imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.dirname(os.path.dirname(script_dir)))  # tvm_practice directory
from models import models_for_test


# Small model registry for testing
SMALL_MODELS = {
    "one_conv": lambda:models_for_test.getOneConvModel(H=32, W=32),
    "one_conv_quant": models_for_test.getOneConvQuantModel,
    "conv_bn_quant": models_for_test.getConvBNQuantModel,
    "residual_model": lambda: models_for_test.getResidualModel(False),
    "residual_rnd_model": lambda: models_for_test.getResidualModel(True),
    "conv_quant_conv": models_for_test.getConvQuantConvModel,
    "big_conv_quant_conv": models_for_test.getBigConvQuantConvModel,
    "multi_io": lambda: models_for_test.getMultiInputOutputModel(height=8, width=8),
    "multi_io_large": lambda: models_for_test.getMultiInputOutputModel(height=32, width=32),
}


def run_passes_until_memory_allocator(mod, param_dict, output_dir=None, verbose=True):
    """
    Run all passes required before MemoryAllocator.

    Args:
        mod: TVM relay module
        param_dict: Model parameters
        output_dir: Directory to save intermediate results (optional)
        verbose: Print progress messages

    Returns:
        mod: Transformed module ready for MemoryAllocator
        param_dict: Transformed parameters
        ttype_map: Type map from layout legalization
    """
    DevConfig().clear()

    def log(msg):
        if verbose:
            print(f"  [PASS] {msg}")

    def save_model(mod, param_dict, name):
        if output_dir:
            from tvm.contrib.relay_viz import RelayVisualizer, DotPlotter, DotVizParser
            RelayVisualizer(
                relay_mod=mod,
                relay_param=param_dict,
                plotter=DotPlotter(),
                parser=DotVizParser(),
            ).render(f"{output_dir}/{name}")
            with open(f"{output_dir}/{name}.txt", "w") as f:
                f.write(mod.astext(show_meta_data=True))

    log("Binding parameters")
    mod["main"] = bind_params_by_name(mod["main"], param_dict)
    mod = transform.InferType()(mod)
    save_model(mod, param_dict, "1_after_bind")

    log("L1 partition")
    mod = imcflow_transform.partitionImcflowSubGraph(mod)
    save_model(mod, param_dict, "2_after_L1_partition")

    log("Split conv to atomic ops")
    mod, param_dict = imcflow_transform.split_conv_to_atomic(mod, param_dict)
    save_model(mod, param_dict, "2_after_atom_split")

    log("Merge composite ops")
    mod = imcflow_transform.merge_composite_ops(mod)
    save_model(mod, param_dict, "3_after_merge")

    log("Make split/concat super node")
    mod = imcflow_transform.makeSplitConcatDepsRegions(mod)
    save_model(mod, param_dict, "4_after_split_concat_partition")

    log("Concat distributor")
    mod = imcflow_transform.ConcatDistributor(max_inputs=4).run(mod)
    save_model(mod, param_dict, "4.5_after_concat_distributor")

    log("Partition round")
    mod = imcflow_transform.partitionRound(mod)
    save_model(mod, param_dict, "5_after_annot")

    log("Flatten imcflow top funcs")
    mod = imcflow.flattenImcflowTopFuncs(mod)
    save_model(mod, param_dict, "6_after_flatten")

    log("Prune imcflow subgraphs")
    mod = imcflow.prune_imcflow_subgraphs(mod)
    save_model(mod, param_dict, "7_after_prune_model")

    log("Annotate custom ID (first)")
    mod = imcflow_transform.annotateCustomId(mod)
    save_model(mod, param_dict, "7.5_after_annotate_custom_id")
    imcflow_transform.constructUsefulMappings(mod)

    log("Legalize imcflow layout")
    mod, ttype_map = imcflow_transform.legalizeImcflowLayout(mod)
    save_model(mod, param_dict, "7.7_after_mark_in_out")

    log("Annotate custom ID (second)")
    mod = imcflow_transform.annotateCustomId(mod)
    save_model(mod, param_dict, "8.5_after_annotate_custom_id")

    log("Construct mappings")
    imcflow_transform.constructUsefulMappings(mod)
    imcflow_transform.constructCustomIDInFunc(mod)
    imcflow_transform.constructImcflowFuncMap(mod)
    save_model(mod, param_dict, "9_with_custom_id")

    log("Node mapper")
    imcflow_transform.NodeMapper().run(mod)

    log("Construct tensor edge list")
    imcflow_transform.constructTensorEdgeList(mod)

    log("Construct active IMCE dict")
    imcflow_transform.constructActiveIMCEDict(mod)

    log("Construct tensor ID to edge dict")
    imcflow_transform.constructTensorIDToTensorEdgeDict()

    log("Construct NoC path dict")
    imcflow_transform.constructNoCPathDict(mod)

    return mod, param_dict, ttype_map


def run_memory_allocator(mod, ttype_map, verbose=True):
    """
    Run the MemoryAllocator pass.

    Args:
        mod: Transformed module (after all prerequisite passes)
        ttype_map: Type map from layout legalization
        verbose: Print debug info

    Returns:
        None (modifies DevConfig() in place)
    """
    if verbose:
        print("\n" + "="*60)
        print("RUNNING MEMORY ALLOCATOR")
        print("="*60)

    # Enable debug output
    os.environ["IMCFLOW_DEBUG"] = "1"

    allocator = imcflow_transform.MemoryAllocator()
    allocator.run(mod, ttype_map)

    if verbose:
        print("\n" + "-"*40)
        print("Memory Layout Results:")
        print("-"*40)
        for func_name, layout in DevConfig().MemLayout.items():
            print(f"\nFunction: {func_name}")
            for region_name, region in layout.items():
                print(f"  {region_name}:")
                # Try different attribute names for allocated blocks
                blocks = getattr(region, 'allocated_blocks', None) or \
                         getattr(region, 'blocks', None) or \
                         getattr(region, '_blocks', [])
                if blocks:
                    for block in blocks:
                        print(f"    {block}")
                else:
                    print(f"    (no blocks or unknown attribute)")


def print_tensor_edge_info(verbose=True):
    """Print TensorEdgeInfo with tiling information"""
    if not verbose:
        return

    print("\n" + "-"*40)
    print("Tensor Edge Info (with tiling):")
    print("-"*40)

    for edge, info in DevConfig().TensorEdgetoInfo.items():
        print(f"\nEdge: {edge.simple_name()}")
        if hasattr(info, 'block_tiling_info') and info.block_tiling_info:
            ti = info.block_tiling_info
            print(f"  height_base_coords: {ti.height_base_coords}")
            print(f"  height_sizes: {ti.height_sizes}")
            print(f"  pkt_cnts: {ti.pkt_cnts}")
            print(f"  c_input_var_offsets: {ti.c_input_var_offsets}")
            print(f"  c_input_var_sizes: {ti.c_input_var_sizes}")
        elif hasattr(info, 'height_base_coords'):
            print(f"  height_base_coords: {info.get_height_base_coords()}")
            print(f"  height_sizes: {info.get_height_sizes()}")
            print(f"  pkt_cnts: {info.get_pkt_cnts()}")


def test_memory_allocator(model_name="one_conv_quant", output_dir=None):
    """
    Main test function for MemoryAllocator.

    Args:
        model_name: Name of the model to test
        output_dir: Directory to save outputs (default: test_mem_alloc_{model_name})
    """
    print("="*60)
    print(f"Testing MemoryAllocator with model: {model_name}")
    print("="*60)

    if model_name not in SMALL_MODELS:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Available models: {list(SMALL_MODELS.keys())}")
        return

    # Setup output directory
    if output_dir is None:
        output_dir = f"test_mem_alloc_{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Get model
    print(f"\n[1] Loading model: {model_name}")
    mod, param_dict = SMALL_MODELS[model_name]()

    # Run passes
    print(f"\n[2] Running prerequisite passes...")
    mod, param_dict, ttype_map = run_passes_until_memory_allocator(
        mod, param_dict, output_dir=output_dir, verbose=True
    )

    # Print intermediate state
    print(f"\n[3] Intermediate state before MemoryAllocator:")
    print(f"  Functions in module: {list(mod.functions.keys())}")
    print(f"  ImcflowFuncMap: {list(DevConfig().ImcflowFuncMap.keys())}")
    print(f"  TensorEdgeListDict keys: {list(DevConfig().TensorEdgeListDict.keys())}")

    # Run memory allocator
    print(f"\n[4] Running MemoryAllocator...")
    try:
        run_memory_allocator(mod, ttype_map, verbose=True)
        print("\n✅ MemoryAllocator completed successfully!")
    except Exception as e:
        print(f"\n❌ MemoryAllocator failed with error:")
        import traceback
        traceback.print_exc()
        return

    # Print tensor edge info
    print_tensor_edge_info(verbose=True)

    # Save results
    print(f"\n[5] Saving results to {output_dir}/")
    with open(f"{output_dir}/mem_layout.txt", "w") as f:
        pprint.pprint(DevConfig().MemLayout, stream=f)

    with open(f"{output_dir}/tensor_edge_info.txt", "w") as f:
        for edge, info in DevConfig().TensorEdgetoInfo.items():
            print(f"\nEdge: {edge.simple_name()}", file=f)
            print(f"  Info: {info}", file=f)
            if hasattr(info, 'block_tiling_info') and info.block_tiling_info:
                ti = info.block_tiling_info
                print(f"  Tiling Info:", file=f)
                print(f"    height_base_coords: {ti.height_base_coords}", file=f)
                print(f"    height_sizes: {ti.height_sizes}", file=f)
                print(f"    pkt_cnts: {ti.pkt_cnts}", file=f)
                print(f"    c_input_var_offsets: {ti.c_input_var_offsets}", file=f)
                print(f"    c_input_var_sizes: {ti.c_input_var_sizes}", file=f)

    print(f"\n✅ Test completed. Results saved to {output_dir}/")


if __name__ == "__main__":
    # Parse command line arguments
    model_name = sys.argv[1] if len(sys.argv) > 1 else "one_conv_quant"

    if model_name == "--help" or model_name == "-h":
        print(__doc__)
        print("\nAvailable models:")
        for name in SMALL_MODELS:
            print(f"  - {name}")
        sys.exit(0)

    if model_name == "--list":
        print("Available models:")
        for name in SMALL_MODELS:
            print(f"  - {name}")
        sys.exit(0)

    test_memory_allocator(model_name)
