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
matches PyTorch's full-OC blockN.pw output is therefore the post-concat int16
(1, OC, H, W) tensor (i.e. the input to that block's pw batch_norm), NOT an
individual atomic qconv. This script locates the comparison tensor by matching
the PyTorch output's exact (1, OC, H, W) int16 shape against TVM nodes in
topological order, which is robust to whether a block was OC-split.

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
# The pkl key is ('Int16', idx, <module>, 'output').
PT_PW_MODULE = {n: f"block{n}.block_int16.pw" for n in range(1, NUM_BLOCKS + 1)}


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


def _parse_key(key):
    """Return (name, topo_index, output_num) for a TVM debug pkl key."""
    parts = key.split("____")
    name = parts[0]
    t_idx = int(parts[1].split(":")[1])
    o_idx = int(parts[2].split(":")[1])
    return name, t_idx, o_idx


def get_tvm_tensor(tvm_data, topo_index, output_num=0):
    for key, arr in tvm_data.items():
        _, t_idx, o_idx = _parse_key(key)
        if t_idx == topo_index and o_idx == output_num:
            return np.asarray(arr)
    return None


def tvm_nodes_sorted(tvm_data):
    """All (topo_index, name, ndarray) sorted by topo index."""
    rows = []
    for key, arr in tvm_data.items():
        name, t_idx, o_idx = _parse_key(key)
        rows.append((t_idx, o_idx, name, np.asarray(arr)))
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def _pw_output_node_types(eval_dir):
    """Walk the transformed @main graph and return, in block order, the relay op
    name of the data-input to each block's POINTWISE batch_norm (= the pw conv
    result = PyTorch blockN.pw output).

    The transformed graph has, per block, two imcflow.fused_batch_norm calls:
    bn_dw then bn_pw. So the bn_pw of block b is the (2b)-th imcflow batch_norm.
    Its data input is the pw output node, whose op differs per block depending on
    whether the pointwise conv was OC-split:
      - no split:  strided_slice / layout_transform (deblocked qconv output)
      - OC-split:  concatenate
    Returning the op-name SEQUENCE lets us match the right debug-executor node by
    op type in topological order, which is robust where a fixed shape/name is not.
    """
    from test import load_transformed_model
    from tvm import relay
    import tvm

    mod, _ = load_transformed_model(eval_dir, pkl_name="transformed_cpu_model.pkl")

    def opname(e):
        if isinstance(e, relay.Call):
            if isinstance(e.op, relay.GlobalVar):
                return e.op.name_hint
            if hasattr(e.op, "name"):
                return e.op.name
            return type(e.op).__name__
        return type(e).__name__

    bn_inputs = []  # (bn_opname, input_opname) in topo order

    class V(relay.ExprVisitor):
        def visit_call(self, c):
            super().visit_call(c)
            nm = opname(c)
            # imcflow.fused_batch_norm is an Op (name 'imcflow.fused_batch_norm'),
            # not a GlobalVar. Exclude the FP stem 'nn.batch_norm'.
            if "fused_batch_norm" in nm:
                bn_inputs.append(opname(c.args[0]))

    V().visit(mod["main"])
    # bn_inputs are the imcflow bn data-inputs in order: [bn_dw1, bn_pw1, bn_dw2, ...]
    # The pointwise ones are the odd positions (0-based: 1,3,5,...).
    pw_input_ops = [bn_inputs[2 * b + 1] for b in range(NUM_BLOCKS)]
    return pw_input_ops


def find_block_pw_tensors(tvm_data, pt_store, idx, eval_dir):
    """Locate the TVM debug tensor matching each PyTorch blockN.pw output.

    Uses the transformed graph to learn, per block, the op type that produces the
    pw output (bn_pw's data input), then matches debug-executor nodes of that op
    type in topological order while ALSO requiring the PyTorch output's exact
    (1, OC, H, W) int16 shape. The combination (correct op type + correct shape +
    topo cursor) pins the right node even where blocks share a shape.

    Returns dict block_idx -> (topo_index, name, tvm_ndarray) or None.
    """
    pw_ops = _pw_output_node_types(eval_dir)
    rows = tvm_nodes_sorted(tvm_data)
    result = {}
    cursor = 0
    for n in range(1, NUM_BLOCKS + 1):
        pt = _to_numpy(pt_get(pt_store, ("Int16", idx, PT_PW_MODULE[n], "output"),
                              f"block{n}.pw output"))
        want_shape = tuple(int(s) for s in pt.shape)
        want_op = pw_ops[n - 1]  # relay op name, e.g. 'strided_slice'/'layout_transform'/'concatenate'
        found = None
        for j in range(cursor, len(rows)):
            t_idx, o_idx, name, arr = rows[j]
            # debug node 'name' is the fused TIR func name; map relay op -> its substring
            if arr.dtype == np.int16 and tuple(arr.shape) == want_shape \
               and want_op.split(".")[-1] in name:
                found = (t_idx, name, arr)
                cursor = j + 1
                break
        result[n] = found
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

    # 2. Re-run TVM CPU debug executor on that exact input.
    print(f"\nRe-running TVM CPU debug executor on sample {idx}'s input...")
    print(f"  eval_dir: {args.eval_dir}")
    tvm_data = regenerate_tvm_transformed_pkl(args.eval_dir, pt_input)
    print(f"  TVM debug pkl entries: {len(tvm_data)}")

    # 2b. Input-consistency check.
    tvm_input = get_tvm_tensor(tvm_data, 0)
    if tvm_input is None:
        print("  WARNING: could not find TVM topo-0 input for consistency check.")
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
    located = find_block_pw_tensors(tvm_data, store, idx, args.eval_dir)
    results = []
    for n in range(1, NUM_BLOCKS + 1):
        loc = located.get(n)
        if loc is None:
            print(f"  [!!] block{n}.pw: no matching TVM int16 NCHW tensor found")
            results.append({"name": f"block{n}.pw", "match": False,
                            "error": "TVM tensor not located"})
            continue
        topo, name, tvm_arr = loc
        pt_arr = pt_get(store, ("Int16", idx, PT_PW_MODULE[n], "output"), f"block{n}.pw output")
        r = compare_bit_exact(tvm_arr, pt_arr, f"block{n}.pw (topo {topo} {name})")
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
