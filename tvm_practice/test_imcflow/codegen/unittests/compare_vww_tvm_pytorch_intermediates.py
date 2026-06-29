#!/usr/bin/env python3
"""
Compare TVM VWW MobileNetV1 intermediate tensors against the PyTorch
MobileNetV1VWWIMCFlow INT16 reference, bit-for-bit, at the 13 pointwise
(PsumConv) op boundaries.

VWW counterpart of compare_dscnn_tvm_pytorch_intermediates.py (KWS DS-CNN). Like
that one, it does NOT run PyTorch locally: the PyTorch INT16 intermediates are
dumped on berlin1 (deploy/inference.py --model vww, DEBUG=1 --sample_logits ...)
and handed over as a pickle. This script:

  1. loads that PyTorch pkl,
  2. extracts the FP model_input (1,3,96,96) for a chosen sample index,
  3. re-runs the TVM CPU debug executor on that EXACT input (reusing the
     already-compiled transformed_cpu_model.pkl — no codegen, no gem5),
  4. maps TVM's pointwise conv outputs to the PyTorch
     ('Int16', idx, 'blockN.block_int16.pw', 'output') tensors and compares
     them bit-for-bit (integer int16 equality).

The only on-array MVM in VWW MobileNet is the pointwise 1x1 conv, so these 13
boundaries are the primary bit-exact verification target. Depthwise runs
off-array (use_imcu=0) and is excluded from the int16 contract.

NOTE on OC-split: blocks with pointwise OC>64 (blocks 6..13) get the qconv
OC-split into multiple atomic convs that are concatenated. The TVM tensor that
matches PyTorch's full-OC blockN.pw output is the post-concat int16 (1, OC, H, W)
tensor — the data input to that block's pointwise batch_norm. This script locates
it exactly by walking the GraphExecutor JSON: it finds the 26 imcflow batch_norm
nodes (13 blocks x {bn_dw, bn_pw}) in dataflow order, takes every 2nd (bn_pw),
and reads the unique graph-node feeding each via its inputs[0] edge. No shape/
name heuristic — robust to OC-split (where several same-shape concatenates exist).

Usage:
  cd /root/project/tvm/tvm_practice/test_imcflow/codegen
  source ~/.zshrc
  python unittests/compare_vww_tvm_pytorch_intermediates.py \
      --pytorch-pkl /path/to/debug_vww_mobilenet.pkl \
      --sample 0

Prerequisites:
  - A prior TVM run of vww_full_pretrained that produced
    eval_dir/vww_full_pretrained_evl.baremetal/transformed_cpu_model.pkl
    (any `--ref-models transformed` run does this). The same CKPT must be used
    on both sides so the weights match.
  - The PyTorch pkl from berlin1.
"""

import sys
import os
import argparse
import pickle
import numpy as np

CODEGEN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODEGEN_DIR)

DEFAULT_EVAL_DIR = os.path.join(
    CODEGEN_DIR, "eval_dir", "vww_full_pretrained_evl.baremetal"
)

NUM_BLOCKS = 13

# PyTorch (berlin1) hooked module names for the 13 pointwise PsumConv modules.
# The pkl key is ('Int16', idx, <module>, 'output'). The VWW deploy model names
# blocks 0-based ('blocks.{i}.block_int16.pw'), so relay block n (1..13) maps to
# PyTorch blocks.{n-1}.
PT_PW_MODULE = {n: f"blocks.{n - 1}.block_int16.pw" for n in range(1, NUM_BLOCKS + 1)}


# ---------------------------------------------------------------------------
# PyTorch pkl access
# ---------------------------------------------------------------------------
def load_pytorch_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def pt_get(store, key, what):
    if key in store:
        return _to_numpy(store[key])
    tag = key[0] if isinstance(key, tuple) and key else None
    near = [k for k in store.keys() if isinstance(k, tuple) and k and k[0] == tag]
    raise KeyError(
        f"PyTorch pkl missing {what} key {key!r}.\n"
        f"  Keys with tag {tag!r} ({len(near)}): "
        + ", ".join(repr(k) for k in near[:12])
        + ("..." if len(near) > 12 else "")
    )


def pt_sample_indices(store):
    idxs = set()
    for k in store.keys():
        if isinstance(k, tuple) and len(k) == 3 and k[0] == "Input" and k[2] == "model_input":
            idxs.add(k[1])
    return sorted(idxs)


# ---------------------------------------------------------------------------
# TVM side: build the CPU graph, run the debug executor, and use the graph JSON
# to address each block's pointwise output node exactly.
# ---------------------------------------------------------------------------
def build_tvm_debug(eval_dir, model_input_np):
    """Build the transformed CPU model, run the debug executor on model_input_np,
    and return (graph_json_dict, debug_tensors) where debug_tensors maps
    graph-node-name -> output ndarray (output 0).

    We build here (rather than calling test._build_and_run_on_cpu) so we also keep
    the GraphExecutor JSON, which lets us map a relay node to its exact debug-pkl
    graph-node name (the debug key 'name' is the deduplicated graph node name, not
    the fused primfunc name).
    """
    import json
    import tvm
    from tvm import relay
    from tvm.relay.backend import Executor, Runtime
    from tvm.contrib.debugger import debug_executor
    from test import load_transformed_model

    cpu_mod, cpu_params = load_transformed_model(eval_dir, pkl_name="transformed_cpu_model.pkl")
    model_input_np = np.ascontiguousarray(model_input_np.astype("float32"))

    executor_ = Executor("graph")
    runtime_ = Runtime("crt", {"system-lib": True})
    with tvm.transform.PassContext(opt_level=0, config={"tir.disable_vectorize": True}):
        graph, lib, params = tvm.relay.build(
            cpu_mod, target="llvm", params=cpu_params, executor=executor_, runtime=runtime_)

    ctx = tvm.cpu(0)
    ex = debug_executor.create(graph, lib, device=ctx)
    if params:
        ex.load_params(tvm.runtime.save_param_dict(params))
    ex.set_input("model_input", model_input_np)
    ex.run()

    # debug tensors keyed by 'name____topo-index:N____output-num:M'; collapse to
    # graph-node-name -> output-0 ndarray (these pw nodes are single-output).
    debug = {}
    for k, v in ex.debug_datum.get_output_tensors().items():
        name = k.split("____")[0]
        onum = int(k.split("____")[2].split(":")[1])
        if onum == 0:
            debug[name] = v.asnumpy()

    return json.loads(graph), debug


def _graph_node_meta(graph_json):
    """Return list of per-node dicts: {name, func_name, op, inputs:[node_idx,...]}."""
    meta = []
    for node in graph_json["nodes"]:
        attrs = node.get("attrs", {}) or {}
        meta.append({
            "name": node["name"],
            "func_name": attrs.get("func_name", ""),
            "op": node.get("op", ""),
            "inputs": [e[0] for e in node.get("inputs", [])],
        })
    return meta


def find_block_pw_tensors(graph_json, debug, pt_store, idx):
    """Map each block's pointwise-conv output to its exact TVM debug tensor.

    Walk the GraphExecutor JSON (same dataflow order as relay). The pw output of
    block b is the data-input (inputs[0]) of that block's pointwise
    imcflow.fused_batch_norm node. There are 26 batch_norm graph nodes (13 blocks
    x {bn_dw, bn_pw}) in relay order; the pw ones are every 2nd (indices 1,3,5,...).
    Each node has a unique graph-node name, so the debug tensor is addressed
    exactly with no shape/cursor heuristic.

    Returns dict block_idx -> (node_name, tvm_ndarray) or None.
    """
    meta = _graph_node_meta(graph_json)
    bn_idxs = [i for i, m in enumerate(meta) if "imcflow_fused_batch_norm" in m["func_name"]]
    pw_bn_idxs = bn_idxs[1::2]  # bn_pw per block, in block order
    result = {}
    for n in range(1, NUM_BLOCKS + 1):
        if n - 1 >= len(pw_bn_idxs):
            result[n] = None
            continue
        bn_i = pw_bn_idxs[n - 1]
        pw_node_i = meta[bn_i]["inputs"][0]
        node_name = meta[pw_node_i]["name"]
        arr = debug.get(node_name)
        result[n] = (node_name, np.asarray(arr)) if arr is not None else None
    return result


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def compare_bit_exact(tvm_arr, pt_arr, name):
    pt_arr = _to_numpy(pt_arr)
    res = {"name": name, "tvm_shape": tvm_arr.shape, "tvm_dtype": str(tvm_arr.dtype),
           "pt_shape": pt_arr.shape, "pt_dtype": str(pt_arr.dtype)}
    if tuple(tvm_arr.shape) != tuple(pt_arr.shape):
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
                    help="Path to berlin1 VWW debug pkl")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR,
                    help="TVM eval dir holding transformed_cpu_model.pkl")
    ap.add_argument("--input-tol", type=float, default=0.0,
                    help="Max allowed abs diff when checking the FP input matches "
                         "on both sides (default 0 = bit-exact float)")
    args = ap.parse_args()

    print("=" * 80)
    print("VWW MobileNetV1: TVM vs PyTorch (berlin1) bit-exact comparison")
    print("=" * 80)

    print(f"\nLoading PyTorch pkl: {args.pytorch_pkl}")
    store = load_pytorch_pkl(args.pytorch_pkl)
    avail = pt_sample_indices(store)
    print(f"  sample indices available: {avail}")
    if args.sample not in avail:
        print(f"  ERROR: --sample {args.sample} not in pkl. Pick one of {avail}.")
        sys.exit(2)
    idx = args.sample

    # 1. Extract the FP input for this sample.
    pt_input = _to_numpy(pt_get(store, ("Input", idx, "model_input"), "input")).astype("float32")
    print(f"\nSample {idx}: model_input shape={pt_input.shape} "
          f"range=[{pt_input.min():.4f}, {pt_input.max():.4f}]")

    # 2. Build the TVM CPU graph + run debug executor on that exact input.
    print(f"\nBuilding TVM CPU graph + debug executor on sample {idx}'s input...")
    print(f"  eval_dir: {args.eval_dir}")
    graph_json, debug = build_tvm_debug(args.eval_dir, pt_input)
    print(f"  TVM debug graph nodes: {len(graph_json['nodes'])}, debug tensors: {len(debug)}")

    # 2b. Input-consistency check: the input node 'model_input' tensor must equal PT.
    tvm_input = debug.get("model_input")
    if tvm_input is None:
        print("  WARNING: could not find TVM 'model_input' node for consistency check.")
    else:
        in_diff = np.abs(tvm_input.astype(np.float64) - pt_input.astype(np.float64)).max()
        ok = in_diff <= args.input_tol
        print(f"  input consistency: max|TVM-PT| = {in_diff:.6g} ({'OK' if ok else 'MISMATCH'})")
        if not ok:
            print("  ERROR: TVM and PyTorch did not run on the same input; aborting.")
            sys.exit(3)

    # 3. Locate + compare the 13 pointwise conv outputs bit-for-bit.
    print("\n" + "=" * 80)
    print("Pointwise PsumConv outputs (the on-array MVM bit-exact targets)")
    print("=" * 80)
    located = find_block_pw_tensors(graph_json, debug, store, idx)
    results = []
    for n in range(1, NUM_BLOCKS + 1):
        loc = located.get(n)
        if loc is None:
            print(f"  [!!] block{n}.pw: pw-output graph node not found")
            results.append({"name": f"block{n}.pw", "match": False,
                            "error": "TVM node not located"})
            continue
        name, tvm_arr = loc
        pt_arr = pt_get(store, ("Int16", idx, PT_PW_MODULE[n], "output"), f"block{n}.pw output")
        r = compare_bit_exact(tvm_arr, pt_arr, f"block{n}.pw ({name})")
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
