#!/usr/bin/env python3
"""
Compare TVM debug executor intermediate tensors with PyTorch debug model outputs.

Maps TVM graph nodes (from debug_executor_output_tensors.pkl) to corresponding
PyTorch layer outputs (from ResNet8IMCFlow debug model) and identifies where
results diverge.

Usage:
  cd /root/project/tvm/tvm_practice/test_imcflow/codegen
  python unittests/compare_tvm_pytorch_intermediates.py

Prerequisites:
  - debug_executor_output_tensors.pkl must exist in eval_dir/resnet8_subset31_pretrained_orig_evl.baremetal/
  - The same model_input.npy used for the TVM run
"""

import sys
import os
import pickle
import numpy as np

# Add CIM root and resnet8 directory for PyTorch model imports
sys.path.insert(0, '/root/project/CIM')
sys.path.insert(0, '/root/project/CIM/deploy/image_classification/resnet8')

import torch
import torch.nn.functional as F
import copy

from deploy.image_classification.resnet8.debug_utils import (
    create_debug_models, compare_tensors
)


# ============================================================================
# Configuration
# ============================================================================
CODEGEN_DIR = '/root/project/tvm/tvm_practice/test_imcflow/codegen'
EVAL_DIR = os.path.join(CODEGEN_DIR, 'eval_dir/resnet8_subset31_pretrained_orig_evl.baremetal')
PKL_PATH = os.path.join(EVAL_DIR, 'debug_executor_output_tensors.pkl')
INPUT_PATH = os.path.join(EVAL_DIR, 'test_inputs/model_input.npy')
CHECKPOINT_PATH = '/root/project/CIM/trained_models/image_classification/NAT/prange_full_psum_duplication_1/greedy_ch_split/2026-Feb-12-20-38-13/imcflow/2026-Feb-26-21-34-16/checkpoint.pth.tar'


def load_tvm_intermediates(pkl_path):
    """Load TVM debug executor output tensors."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_tvm_tensor(tvm_data, topo_index, output_num=0):
    """Get a specific tensor from TVM data by topo index."""
    for key, arr in tvm_data.items():
        parts = key.split('____')
        t_idx = int(parts[1].split(':')[1])
        o_idx = int(parts[2].split(':')[1])
        if t_idx == topo_index and o_idx == output_num:
            return arr
    return None


def get_tvm_tensor_by_name(tvm_data, name_substring):
    """Get tensor(s) matching a name substring."""
    matches = []
    for key, arr in tvm_data.items():
        node_name = key.split('____')[0]
        if name_substring in node_name:
            matches.append((key, arr))
    return matches


def run_pytorch_debug(checkpoint_path, input_tensor):
    """Run PyTorch IMCFlow debug model and return intermediate tensors."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    adjust_factors = checkpoint['adjust_factors']

    # The imcflow checkpoint stores the original training checkpoint inside
    # 'original_checkpoint', which has the 'module.' prefixed keys expected
    # by ResNet8/ResNet8IMCFlow constructors.
    original_checkpoint = checkpoint['original_checkpoint']

    # Create debug model using original checkpoint (has 'module.' keys)
    # Use PsumConv (bit-serial + ADC psum quantization) to match TVM imcflow_qconv
    psum_config = {
        'arraySize': 256, 'wbits': 4, 'abits': 4,
        'pbits': 6, 'prange': 1, 'cbits': 1,
    }
    _, _, _, debug_model_int16 = create_debug_models(
        original_checkpoint, adjust_factors, psum_config=psum_config
    )
    debug_model_int16.eval()

    with torch.no_grad():
        output, debug_info = debug_model_int16(input_tensor)

    return output, debug_info, adjust_factors


def tensor_stats(arr, name=""):
    """Get statistics for a tensor (numpy or torch)."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr_f = arr.astype(np.float64)
    return {
        'name': name,
        'shape': arr.shape,
        'dtype': str(arr.dtype),
        'min': arr_f.min(),
        'max': arr_f.max(),
        'mean': arr_f.mean(),
        'std': arr_f.std(),
    }


def compare_arrays(tvm_arr, pt_arr, name="", tolerance=1.0):
    """Compare TVM and PyTorch arrays, return comparison stats."""
    if isinstance(pt_arr, torch.Tensor):
        pt_arr = pt_arr.detach().cpu().numpy()

    # Ensure same dtype for comparison
    tvm_f = tvm_arr.astype(np.float64)
    pt_f = pt_arr.astype(np.float64)

    # Handle shape mismatches
    if tvm_arr.shape != pt_arr.shape:
        return {
            'name': name,
            'match': False,
            'error': f'Shape mismatch: TVM {tvm_arr.shape} vs PyTorch {pt_arr.shape}',
            'tvm_stats': tensor_stats(tvm_arr, f'{name}_tvm'),
            'pt_stats': tensor_stats(pt_arr, f'{name}_pt'),
        }

    diff = tvm_f - pt_f
    abs_diff = np.abs(diff)

    exact_match = np.array_equal(tvm_arr, pt_arr)
    max_abs_err = abs_diff.max()
    mean_abs_err = abs_diff.mean()
    mse = np.mean(diff ** 2)

    # Relative error (avoid division by zero)
    denom = (np.abs(tvm_f) + np.abs(pt_f)) / 2.0 + 1e-8
    rel_err = abs_diff / denom
    mean_rel_err = rel_err.mean()

    within_tol = np.mean(abs_diff <= tolerance) * 100.0
    n_mismatch = np.sum(abs_diff > 0)

    return {
        'name': name,
        'match': exact_match,
        'max_abs_err': max_abs_err,
        'mean_abs_err': mean_abs_err,
        'mse': mse,
        'mean_rel_err': mean_rel_err,
        'within_tolerance': within_tol,
        'n_mismatch': int(n_mismatch),
        'total_elements': int(tvm_arr.size),
        'tvm_stats': tensor_stats(tvm_arr, f'{name}_tvm'),
        'pt_stats': tensor_stats(pt_arr, f'{name}_pt'),
    }


def print_comparison(result, verbose=True):
    """Pretty-print a comparison result."""
    name = result['name']
    if 'error' in result:
        print(f"  {name}: ERROR - {result['error']}")
        print(f"    TVM: shape={result['tvm_stats']['shape']} dtype={result['tvm_stats']['dtype']} "
              f"range=[{result['tvm_stats']['min']:.4f}, {result['tvm_stats']['max']:.4f}]")
        print(f"    PT:  shape={result['pt_stats']['shape']} dtype={result['pt_stats']['dtype']} "
              f"range=[{result['pt_stats']['min']:.4f}, {result['pt_stats']['max']:.4f}]")
        return

    status = "EXACT MATCH" if result['match'] else "MISMATCH"
    icon = "OK" if result['match'] else "!!"

    print(f"  [{icon}] {name}: {status}")
    if verbose or not result['match']:
        ts, ps = result['tvm_stats'], result['pt_stats']
        print(f"    TVM: shape={ts['shape']} dtype={ts['dtype']} "
              f"range=[{ts['min']:.4f}, {ts['max']:.4f}] mean={ts['mean']:.4f} std={ts['std']:.4f}")
        print(f"    PT:  shape={ps['shape']} dtype={ps['dtype']} "
              f"range=[{ps['min']:.4f}, {ps['max']:.4f}] mean={ps['mean']:.4f} std={ps['std']:.4f}")
        if not result['match']:
            print(f"    max_abs_err={result['max_abs_err']:.4f} "
                  f"mean_abs_err={result['mean_abs_err']:.4f} "
                  f"MSE={result['mse']:.4f}")
            print(f"    mean_rel_err={result['mean_rel_err']:.6f} "
                  f"within_tol(1.0)={result['within_tolerance']:.1f}% "
                  f"mismatches={result['n_mismatch']}/{result['total_elements']}")


def main():
    print("=" * 80)
    print("TVM vs PyTorch Intermediate Tensor Comparison")
    print("=" * 80)

    # Load TVM intermediates
    print(f"\nLoading TVM debug executor pkl: {PKL_PATH}")
    tvm_data = load_tvm_intermediates(PKL_PATH)
    print(f"  Loaded {len(tvm_data)} entries")

    # Load input
    print(f"Loading input: {INPUT_PATH}")
    model_input_np = np.load(INPUT_PATH)
    model_input_torch = torch.from_numpy(model_input_np).float()
    print(f"  Input shape: {model_input_np.shape}, range: [{model_input_np.min():.4f}, {model_input_np.max():.4f}]")

    # Run PyTorch debug model
    print(f"\nRunning PyTorch IMCFlow debug model...")
    pt_output, pt_debug, adjust_factors = run_pytorch_debug(CHECKPOINT_PATH, model_input_torch)
    print(f"  PyTorch output shape: {pt_output.shape}")

    print(f"\n  adjust_factors:")
    for k, v in sorted(adjust_factors.items()):
        print(f"    {k}: {v}")

    # =========================================================================
    # Layer-by-layer comparison
    # =========================================================================
    # New topo index mapping (after conv1+bn1 fuse to multiply+add):
    #   0: model_input
    #   2: conv2d (conv1)
    #   4: multiply (fused scale)
    #   6: add (fused bias) = conv1+bn1 output
    #   7: clip
    #   8: cast int16
    #  13: min_max_quantize (layer1 act1)
    #  16: qconv (layer1 conv1)
    #  19: fused_batch_norm (layer1 bn1)
    #  22: min_max_quantize_1 (layer1 act2)
    #  25: qconv_1 (layer1 conv2)
    #  28: fused_batch_norm_1 (layer1 bn2)
    #  29: relu (residual path)
    #  47: cast_3_1 (layer1 output after residual add + clip + cast)
    #  50: min_max_quantize_2 (layer2 act1)
    #  53: qconv_1_1 (layer2 conv1, stride=2)
    #  56: fused_batch_norm_1_1 (layer2 bn1)
    #  59: min_max_quantize_1_1 (layer2 act2)
    #  71: cast_5 (layer2 conv2 after split accum + clip + cast)
    #  74: fused_batch_norm_1_2 (layer2 bn2)
    #  85: fused_batch_norm_1_3 (layer2 downsample bn)
    #  93: cast_5_1 (layer2 output after residual add)
    # 111: min_max_quantize_1_2 (layer3 act1)
    # 123: cast_8 (layer3 conv1 after split accum)
    # 126: fused_batch_norm_2 (layer3 bn1)
    # 129: min_max_quantize_3 (layer3 act2)
    # 147: cast_8_1 (layer3 conv2 after 3-way split)
    # 150: fused_batch_norm_2_1 (layer3 bn2)
    # 161: fused_batch_norm_2_2 (layer3 downsample bn)
    # 169: cast_8_2 (layer3 output after residual add)
    # 187: layout_transform_13 -> NCHW (after relu, dequant)
    # 188: adaptive_avg_pool2d
    # 189: reshape
    # 191: dense
    # 193: bias_add (final output)

    comparisons = []

    # --- 1. Input ---
    print("\n" + "=" * 80)
    print("SECTION 1: Initial Conv + BN(fused) + Quantize (before imcflow regions)")
    print("=" * 80)

    tvm_input = get_tvm_tensor(tvm_data, 0)
    r = compare_arrays(tvm_input, model_input_np, name="model_input")
    print_comparison(r)
    comparisons.append(r)

    # Conv1 output (TVM topo 2)
    tvm_conv1 = get_tvm_tensor(tvm_data, 2)
    pt_conv1 = pt_debug['after_conv1']
    r = compare_arrays(tvm_conv1, pt_conv1, name="conv1_output")
    print_comparison(r)
    comparisons.append(r)

    # Fused conv1+bn1 output (TVM topo 6 = after multiply + add)
    tvm_bn1 = get_tvm_tensor(tvm_data, 6)
    pt_bn1 = pt_debug['after_bn1']
    r = compare_arrays(tvm_bn1, pt_bn1, name="conv1+bn1_fused_output")
    print_comparison(r)
    comparisons.append(r)

    # After quantize to int16 (TVM topo 8 = cast to int16)
    tvm_int16 = get_tvm_tensor(tvm_data, 8)
    layer1_debug = pt_debug['layer1']
    pt_int16_input = layer1_debug['input_quantized']
    r = compare_arrays(tvm_int16, pt_int16_input, name="quantized_to_int16 (input to layer1)")
    print_comparison(r)
    comparisons.append(r)

    # --- 2. Layer 1 (Basic Block 1: 16->16, no downsample) ---
    print("\n" + "=" * 80)
    print("SECTION 2: Layer 1 (BasicBlock 16->16, no downsample)")
    print("=" * 80)

    # Act1 output (min_max_quantize -> uint8)
    tvm_act1 = get_tvm_tensor(tvm_data, 13)
    pt_act1 = layer1_debug['after_act1']
    print(f"\n  layer1.act1 (min_max_quantize):")
    print(f"    TVM: shape={tvm_act1.shape} dtype={tvm_act1.dtype} range=[{tvm_act1.min()}, {tvm_act1.max()}]")
    print(f"    PT:  shape={pt_act1.shape} dtype={pt_act1.detach().numpy().dtype} range=[{pt_act1.min().item()}, {pt_act1.max().item()}]")

    # Conv1 output (qconv)
    tvm_qconv1 = get_tvm_tensor(tvm_data, 16)
    pt_conv1_l1 = layer1_debug['after_conv1']
    r = compare_arrays(tvm_qconv1, pt_conv1_l1, name="layer1.conv1 (qconv)")
    print_comparison(r)
    comparisons.append(r)

    # BN1 output (fused_batch_norm)
    tvm_bn1_l1 = get_tvm_tensor(tvm_data, 19)
    pt_bn1_l1 = layer1_debug['after_bn1']
    r = compare_arrays(tvm_bn1_l1, pt_bn1_l1, name="layer1.bn1 (fused_batch_norm)")
    print_comparison(r)
    comparisons.append(r)

    # Act2 (second quantize)
    tvm_act2 = get_tvm_tensor(tvm_data, 22)
    pt_act2 = layer1_debug['after_act2']
    print(f"\n  layer1.act2 (min_max_quantize):")
    print(f"    TVM: shape={tvm_act2.shape} dtype={tvm_act2.dtype} range=[{tvm_act2.min()}, {tvm_act2.max()}]")
    print(f"    PT:  shape={pt_act2.shape} dtype={pt_act2.detach().numpy().dtype} range=[{pt_act2.min().item()}, {pt_act2.max().item()}]")

    # Conv2 output (qconv)
    tvm_qconv2 = get_tvm_tensor(tvm_data, 25)
    pt_conv2_l1 = layer1_debug['after_conv2']
    r = compare_arrays(tvm_qconv2, pt_conv2_l1, name="layer1.conv2 (qconv)")
    print_comparison(r)
    comparisons.append(r)

    # BN2 output
    tvm_bn2_l1 = get_tvm_tensor(tvm_data, 28)
    pt_bn2_l1 = layer1_debug['after_bn2']
    r = compare_arrays(tvm_bn2_l1, pt_bn2_l1, name="layer1.bn2 (fused_batch_norm)")
    print_comparison(r)
    comparisons.append(r)

    # Residual + output
    # TVM topo 47: cast_3_1 = final output after residual add + clip + cast
    tvm_block1_out = get_tvm_tensor(tvm_data, 47)
    pt_block1_out = layer1_debug['output']
    r = compare_arrays(tvm_block1_out, pt_block1_out, name="layer1 output (after residual add)")
    print_comparison(r)
    comparisons.append(r)

    # --- 3. Layer 2 (Basic Block 2: 16->32, with downsample, stride=2) ---
    print("\n" + "=" * 80)
    print("SECTION 3: Layer 2 (BasicBlock 16->32, downsample stride=2)")
    print("=" * 80)

    layer2_debug = pt_debug['layer2']

    # Act1 (quantize)
    tvm_l2_act1 = get_tvm_tensor(tvm_data, 50)
    pt_l2_act1 = layer2_debug['after_act1']
    print(f"\n  layer2.act1 (min_max_quantize):")
    print(f"    TVM: shape={tvm_l2_act1.shape} dtype={tvm_l2_act1.dtype} range=[{tvm_l2_act1.min()}, {tvm_l2_act1.max()}]")
    print(f"    PT:  shape={pt_l2_act1.shape} dtype={pt_l2_act1.detach().numpy().dtype} range=[{pt_l2_act1.min().item()}, {pt_l2_act1.max().item()}]")

    # Conv1 (stride=2, 16->32)
    tvm_l2_conv1 = get_tvm_tensor(tvm_data, 53)
    pt_l2_conv1 = layer2_debug['after_conv1']
    r = compare_arrays(tvm_l2_conv1, pt_l2_conv1, name="layer2.conv1 (qconv stride=2)")
    print_comparison(r)
    comparisons.append(r)

    # BN1
    tvm_l2_bn1 = get_tvm_tensor(tvm_data, 56)
    pt_l2_bn1 = layer2_debug['after_bn1']
    r = compare_arrays(tvm_l2_bn1, pt_l2_bn1, name="layer2.bn1 (fused_batch_norm)")
    print_comparison(r)
    comparisons.append(r)

    # Conv2 (split into 28+4 channels, accumulated)
    # TVM topo 71 = cast_5 = final conv2 output after split accumulation + clip + cast
    tvm_l2_conv2 = get_tvm_tensor(tvm_data, 71)
    pt_l2_conv2 = layer2_debug['after_conv2']
    r = compare_arrays(tvm_l2_conv2, pt_l2_conv2, name="layer2.conv2 (qconv split 28+4)")
    print_comparison(r)
    comparisons.append(r)

    # BN2
    tvm_l2_bn2 = get_tvm_tensor(tvm_data, 74)
    pt_l2_bn2 = layer2_debug['after_bn2']
    r = compare_arrays(tvm_l2_bn2, pt_l2_bn2, name="layer2.bn2 (fused_batch_norm)")
    print_comparison(r)
    comparisons.append(r)

    # Downsample output
    # TVM topo 85 = fused_batch_norm_1_3 = downsample path bn output
    tvm_l2_ds = get_tvm_tensor(tvm_data, 85)
    pt_l2_ds = layer2_debug['downsample_output']
    r = compare_arrays(tvm_l2_ds, pt_l2_ds, name="layer2.downsample output")
    print_comparison(r)
    comparisons.append(r)

    # Block output (after residual add)
    # TVM topo 93 = cast_5_1 = layer2 output after residual add + clip + cast
    tvm_l2_out = get_tvm_tensor(tvm_data, 93)
    pt_l2_out = layer2_debug['output']
    r = compare_arrays(tvm_l2_out, pt_l2_out, name="layer2 output (after residual add)")
    print_comparison(r)
    comparisons.append(r)

    # --- 4. Layer 3 (Basic Block 3: 32->64, with downsample, stride=2) ---
    print("\n" + "=" * 80)
    print("SECTION 4: Layer 3 (BasicBlock 32->64, downsample stride=2)")
    print("=" * 80)

    layer3_debug = pt_debug['layer3']

    # Act1 (quantize)
    tvm_l3_act1 = get_tvm_tensor(tvm_data, 111)
    pt_l3_act1 = layer3_debug['after_act1']
    print(f"\n  layer3.act1 (min_max_quantize):")
    print(f"    TVM: shape={tvm_l3_act1.shape} dtype={tvm_l3_act1.dtype} range=[{tvm_l3_act1.min()}, {tvm_l3_act1.max()}]")
    print(f"    PT:  shape={pt_l3_act1.shape} dtype={pt_l3_act1.detach().numpy().dtype} range=[{pt_l3_act1.min().item()}, {pt_l3_act1.max().item()}]")

    # Conv1 (split 28+4, stride=2, 32->64)
    # TVM topo 123 = cast_8 = conv1 output after split accumulation
    tvm_l3_conv1 = get_tvm_tensor(tvm_data, 123)
    pt_l3_conv1 = layer3_debug['after_conv1']
    r = compare_arrays(tvm_l3_conv1, pt_l3_conv1, name="layer3.conv1 (qconv split 28+4, stride=2)")
    print_comparison(r)
    comparisons.append(r)

    # BN1
    tvm_l3_bn1 = get_tvm_tensor(tvm_data, 126)
    pt_l3_bn1 = layer3_debug['after_bn1']
    r = compare_arrays(tvm_l3_bn1, pt_l3_bn1, name="layer3.bn1 (fused_batch_norm)")
    print_comparison(r)
    comparisons.append(r)

    # Conv2 (split 28+28+8, 64->64)
    # TVM topo 147 = cast_8_1 = conv2 output after 3-way split accumulation
    tvm_l3_conv2 = get_tvm_tensor(tvm_data, 147)
    pt_l3_conv2 = layer3_debug['after_conv2']
    r = compare_arrays(tvm_l3_conv2, pt_l3_conv2, name="layer3.conv2 (qconv 3-way split)")
    print_comparison(r)
    comparisons.append(r)

    # BN2
    tvm_l3_bn2 = get_tvm_tensor(tvm_data, 150)
    pt_l3_bn2 = layer3_debug['after_bn2']
    r = compare_arrays(tvm_l3_bn2, pt_l3_bn2, name="layer3.bn2 (fused_batch_norm)")
    print_comparison(r)
    comparisons.append(r)

    # Downsample output
    # TVM topo 161 = fused_batch_norm_2_2 = downsample path bn output
    tvm_l3_ds = get_tvm_tensor(tvm_data, 161)
    pt_l3_ds = layer3_debug['downsample_output']
    r = compare_arrays(tvm_l3_ds, pt_l3_ds, name="layer3.downsample output")
    print_comparison(r)
    comparisons.append(r)

    # Block output (after residual add)
    # TVM topo 169 = cast_8_2 = layer3 output after residual add
    # Actually check: topo 180 = cast_8_3 might be after layout transforms for residual
    tvm_l3_out = get_tvm_tensor(tvm_data, 180)
    pt_l3_out = layer3_debug['output']
    r = compare_arrays(tvm_l3_out, pt_l3_out, name="layer3 output (after residual add)")
    print_comparison(r)
    comparisons.append(r)

    # --- 5. Post-processing: dequant, relu, avgpool, dense ---
    print("\n" + "=" * 80)
    print("SECTION 5: Post-processing (dequant, relu, avgpool, dense)")
    print("=" * 80)

    # After relu: TVM topo 187 (layout transformed to NCHW)
    tvm_after_relu = get_tvm_tensor(tvm_data, 187)
    pt_after_relu = pt_debug['after_relu']
    r = compare_arrays(tvm_after_relu, pt_after_relu, name="after_relu")
    print_comparison(r)
    comparisons.append(r)

    # After avgpool
    tvm_avgpool = get_tvm_tensor(tvm_data, 188)
    pt_avgpool = pt_debug['after_avgpool']
    r = compare_arrays(tvm_avgpool, pt_avgpool, name="after_avgpool")
    print_comparison(r)
    comparisons.append(r)

    # After flatten (view)
    tvm_flatten = get_tvm_tensor(tvm_data, 189)
    pt_view = pt_debug['after_view']
    r = compare_arrays(tvm_flatten, pt_view, name="after_flatten/view")
    print_comparison(r)
    comparisons.append(r)

    # Final output (dense + bias_add)
    tvm_output = get_tvm_tensor(tvm_data, 193)
    pt_final = pt_debug['output']
    r = compare_arrays(tvm_output, pt_final, name="final_output (dense + bias)")
    print_comparison(r)
    comparisons.append(r)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    n_exact = sum(1 for r in comparisons if r.get('match', False))
    n_total = len(comparisons)
    n_mismatch = n_total - n_exact

    print(f"\nTotal comparison points: {n_total}")
    print(f"Exact matches: {n_exact}")
    print(f"Mismatches: {n_mismatch}")

    if n_mismatch > 0:
        print(f"\nMISMATCHED layers (in order of pipeline):")
        first_mismatch = None
        for r in comparisons:
            if not r.get('match', False) and 'error' not in r:
                name = r['name']
                max_err = r.get('max_abs_err', 'N/A')
                mean_err = r.get('mean_abs_err', 'N/A')
                print(f"  {name}: max_err={max_err:.4f}, mean_err={mean_err:.4f}")
                if first_mismatch is None:
                    first_mismatch = r

        if first_mismatch:
            print(f"\n>>> FIRST DIVERGENCE POINT: {first_mismatch['name']}")
            print(f"    This is likely where the bug originates.")

    # Print final output comparison
    print(f"\nFinal outputs:")
    tvm_final = get_tvm_tensor(tvm_data, 193)
    pt_final_np = pt_debug['output'].detach().cpu().numpy()
    print(f"  TVM:     {tvm_final.ravel()}")
    print(f"  PyTorch: {pt_final_np.ravel()}")

    # Argmax comparison
    tvm_pred = np.argmax(tvm_final)
    pt_pred = np.argmax(pt_final_np)
    print(f"\n  TVM predicted class:     {tvm_pred}")
    print(f"  PyTorch predicted class: {pt_pred}")
    print(f"  Match: {'YES' if tvm_pred == pt_pred else 'NO'}")


if __name__ == '__main__':
    main()
