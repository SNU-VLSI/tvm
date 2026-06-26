#!/usr/bin/env python3
"""
Compare TVM DS-CNN (KWS) intermediate tensors against the PyTorch DSCNNIMCFlow
INT16 reference, bit-for-bit, at the four pointwise (PsumConv) op boundaries.

This is the KWS counterpart of compare_tvm_pytorch_intermediates.py (ResNet8).
Unlike the ResNet8 script, this one does NOT run PyTorch locally: the PyTorch
INT16 intermediates are produced on berlin1 (see CIM/handoff.md §5) and handed
over as a pickle. This script:

  1. loads that PyTorch pkl,
  2. extracts the FP MFCC model_input for a chosen sample index,
  3. re-runs the TVM CPU debug executor on that EXACT input (reusing the
     already-compiled transformed_cpu_model.pkl — no codegen, no gem5),
  4. maps TVM's pointwise imcflow_qconv outputs to the PyTorch
     ('Int16', idx, 'blockN.block_int16.pw', 'output') tensors and compares
     them bit-for-bit (integer int16 equality).

The only on-array MVM in DS-CNN is the pointwise 1x1 conv, so these four
boundaries are the primary bit-exact verification target. Depthwise runs
off-array (use_imcu=0) and is excluded from the int16 contract here.

Usage:
  cd /root/project/tvm/tvm_practice/test_imcflow/codegen
  source ~/.zshrc
  python unittests/compare_dscnn_tvm_pytorch_intermediates.py \
      --pytorch-pkl /path/to/debug_kws_dscnn.pkl \
      --sample 0

Prerequisites:
  - A prior TVM run of ds_cnn_full_pretrained that produced
    eval_dir/ds_cnn_full_pretrained_evl.baremetal/transformed_cpu_model.pkl
    (any `--ref-models transformed` run does this). The same CKPT must be used
    on both sides (kws_old_noise) so the weights match.
  - The PyTorch pkl from berlin1 (CIM/handoff.md §5).
"""

import sys
import os
import argparse
import pickle
import numpy as np

CODEGEN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODEGEN_DIR)

DEFAULT_EVAL_DIR = os.path.join(
    CODEGEN_DIR, "eval_dir", "ds_cnn_full_pretrained_evl.baremetal"
)

# TVM transformed-graph topo indices of the four pointwise imcflow_qconv outputs,
# in block order. Verified against debug_executor_output_tensors_transformed.pkl
# (node names tvmgen_default_fused_nn_imcflow_qconv[_1.._3], all (1,64,24,5) int16).
TVM_PW_QCONV_TOPO = {
    1: 43,   # block1.block_int16.pw  -> nn_imcflow_qconv
    2: 70,   # block2.block_int16.pw  -> nn_imcflow_qconv_1
    3: 97,   # block3.block_int16.pw  -> nn_imcflow_qconv_2
    4: 124,  # block4.block_int16.pw  -> nn_imcflow_qconv_3
}

# PyTorch (berlin1) hooked module names for the four pointwise PsumConv modules
# (CIM/handoff.md §5). The pkl key is ('Int16', idx, <module>, 'output').
PT_PW_MODULE = {
    n: f"block{n}.block_int16.pw" for n in (1, 2, 3, 4)
}

INPUT_KEY_FMT = ("Input", "model_input")  # ('Input', idx, 'model_input')


# ---------------------------------------------------------------------------
# PyTorch pkl access
# ---------------------------------------------------------------------------
def load_pytorch_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _to_numpy(x):
    # PyTorch tensors stored in the pkl may be torch.Tensor or already numpy.
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def pt_get(store, key, what):
    """Fetch store[key]; on miss, print nearby keys to help diagnose format drift."""
    if key in store:
        return _to_numpy(store[key])
    # Diagnostic: show the keys that share the leading tag so a format change in
    # the berlin1 dump is obvious rather than a bare KeyError.
    tag = key[0] if isinstance(key, tuple) and key else None
    near = [k for k in store.keys() if isinstance(k, tuple) and k and k[0] == tag]
    raise KeyError(
        f"PyTorch pkl missing {what} key {key!r}.\n"
        f"  Keys with tag {tag!r} ({len(near)}): "
        + ", ".join(repr(k) for k in near[:12])
        + ("..." if len(near) > 12 else "")
    )


def pt_sample_indices(store):
    """All sample indices present under the ('Input', idx, 'model_input') tag."""
    idxs = set()
    for k in store.keys():
        if isinstance(k, tuple) and len(k) == 3 and k[0] == "Input" and k[2] == "model_input":
            idxs.add(k[1])
    return sorted(idxs)


# ---------------------------------------------------------------------------
# TVM side: re-run CPU debug executor on a supplied input
# ---------------------------------------------------------------------------
def regenerate_tvm_transformed_pkl(eval_dir, model_input_np):
    """Run the TVM CPU debug executor on model_input_np, writing
    debug_executor_output_tensors_transformed.pkl into eval_dir. Returns the
    loaded dict."""
    from test import load_transformed_model, _build_and_run_on_cpu

    cpu_mod, cpu_params = load_transformed_model(
        eval_dir, pkl_name="transformed_cpu_model.pkl"
    )
    model_input_np = np.ascontiguousarray(model_input_np.astype("float32"))
    _build_and_run_on_cpu(
        cpu_mod, cpu_params, {"model_input": model_input_np}, eval_dir, "transformed"
    )
    pkl_path = os.path.join(eval_dir, "debug_executor_output_tensors_transformed.pkl")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def get_tvm_tensor(tvm_data, topo_index, output_num=0):
    """Get a tensor from the TVM debug pkl by topo index (keys:
    'name____topo-index:N____output-num:M')."""
    for key, arr in tvm_data.items():
        parts = key.split("____")
        t_idx = int(parts[1].split(":")[1])
        o_idx = int(parts[2].split(":")[1])
        if t_idx == topo_index and o_idx == output_num:
            return np.asarray(arr)
    return None


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def compare_bit_exact(tvm_arr, pt_arr, name):
    pt_arr = _to_numpy(pt_arr)
    res = {"name": name, "tvm_shape": tvm_arr.shape, "tvm_dtype": str(tvm_arr.dtype),
           "pt_shape": pt_arr.shape, "pt_dtype": str(pt_arr.dtype)}
    if tvm_arr.shape != pt_arr.shape:
        res.update(match=False, error=f"shape mismatch TVM {tvm_arr.shape} vs PT {pt_arr.shape}")
        return res
    exact = np.array_equal(tvm_arr.astype(np.int64), pt_arr.astype(np.int64))
    diff = np.abs(tvm_arr.astype(np.int64) - pt_arr.astype(np.int64))
    res.update(match=bool(exact),
               max_abs_err=int(diff.max()),
               n_mismatch=int((diff > 0).sum()),
               total=int(tvm_arr.size))
    return res


def print_result(r):
    if r.get("error"):
        print(f"  [!!] {r['name']}: ERROR — {r['error']}")
        print(f"       TVM {r['tvm_shape']} {r['tvm_dtype']} | PT {r['pt_shape']} {r['pt_dtype']}")
        return
    icon = "OK" if r["match"] else "!!"
    status = "EXACT MATCH" if r["match"] else "MISMATCH"
    print(f"  [{icon}] {r['name']}: {status}  ({r['tvm_dtype']} {tuple(r['tvm_shape'])})")
    if not r["match"]:
        print(f"       max_abs_err={r['max_abs_err']}  mismatches={r['n_mismatch']}/{r['total']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pytorch-pkl", required=True,
                    help="Path to berlin1 debug_kws_dscnn.pkl")
    ap.add_argument("--sample", type=int, default=0,
                    help="Sample index within the PyTorch pkl (default 0)")
    ap.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR,
                    help="TVM eval dir holding transformed_cpu_model.pkl")
    ap.add_argument("--input-tol", type=float, default=0.0,
                    help="Max allowed abs diff when checking the FP input is "
                         "identical on both sides (default 0 = bit-exact float)")
    args = ap.parse_args()

    print("=" * 80)
    print("DS-CNN KWS: TVM vs PyTorch (berlin1) bit-exact comparison")
    print("=" * 80)

    print(f"\nLoading PyTorch pkl: {args.pytorch_pkl}")
    store = load_pytorch_pkl(args.pytorch_pkl)
    avail = pt_sample_indices(store)
    print(f"  sample indices available: {avail}")
    if args.sample not in avail:
        print(f"  ERROR: --sample {args.sample} not in pkl. Pick one of {avail}.")
        sys.exit(2)
    idx = args.sample

    # 1. Extract the FP MFCC input for this sample.
    pt_input = pt_get(store, ("Input", idx, "model_input"), "input")
    pt_input = _to_numpy(pt_input).astype("float32")
    print(f"\nSample {idx}: model_input shape={pt_input.shape} "
          f"range=[{pt_input.min():.4f}, {pt_input.max():.4f}]")

    # 2. Re-run TVM CPU debug executor on that exact input.
    print(f"\nRe-running TVM CPU debug executor on sample {idx}'s input...")
    print(f"  eval_dir: {args.eval_dir}")
    tvm_data = regenerate_tvm_transformed_pkl(args.eval_dir, pt_input)
    print(f"  TVM debug pkl entries: {len(tvm_data)}")

    # 2b. Input-consistency check: TVM topo-0 input must equal the PyTorch input.
    tvm_input = get_tvm_tensor(tvm_data, 0)
    if tvm_input is None:
        print("  WARNING: could not find TVM topo-0 input for consistency check.")
    else:
        in_diff = np.abs(tvm_input.astype(np.float64) - pt_input.astype(np.float64)).max()
        ok = in_diff <= args.input_tol
        print(f"  input consistency: max|TVM-PT| = {in_diff:.6g} "
              f"({'OK' if ok else 'MISMATCH'})")
        if not ok:
            print("  ERROR: TVM and PyTorch did not run on the same input; "
                  "downstream comparison is meaningless. Aborting.")
            sys.exit(3)

    # 3. Compare the four pointwise qconv outputs bit-for-bit.
    print("\n" + "=" * 80)
    print("Pointwise PsumConv outputs (the on-array MVM bit-exact targets)")
    print("=" * 80)
    results = []
    for n in (1, 2, 3, 4):
        topo = TVM_PW_QCONV_TOPO[n]
        tvm_arr = get_tvm_tensor(tvm_data, topo)
        if tvm_arr is None:
            print(f"  [!!] block{n}.pw: TVM topo {topo} not found in debug pkl")
            results.append({"name": f"block{n}.pw", "match": False,
                            "error": f"TVM topo {topo} missing"})
            continue
        pt_key = ("Int16", idx, PT_PW_MODULE[n], "output")
        pt_arr = pt_get(store, pt_key, f"block{n}.pw output")
        r = compare_bit_exact(tvm_arr, pt_arr, f"block{n}.pw (topo {topo})")
        print_result(r)
        results.append(r)

    # 4. Summary.
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    n_exact = sum(1 for r in results if r.get("match"))
    print(f"  sample {idx}: {n_exact}/{len(results)} pointwise convs bit-exact")
    first_bad = next((r for r in results if not r.get("match")), None)
    if first_bad is None:
        print("  ✅ All pointwise PsumConv outputs match PyTorch bit-for-bit.")
        sys.exit(0)
    else:
        print(f"  ❌ First divergence: {first_bad['name']} "
              f"({first_bad.get('error', 'mismatch')})")
        sys.exit(1)


if __name__ == "__main__":
    main()
