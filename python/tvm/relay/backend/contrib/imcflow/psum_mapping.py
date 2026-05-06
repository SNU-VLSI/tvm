# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
"""Psum tensor -> IMCU/column mapping dump utility.

For a qconv with input  [N, IC, IH, IW] and output [N, OC, OH, OW], the
partial-sum tensor is conceptually [N, OC, IC, OH, OW]. Each element
psum[n, oc, ic, oh, ow] is computed on:

    - some IMCU (linear id 0..IMCE_NUM-1, or (row, col_in_imce))
    - some IMCU column (0..63)

The mapping is invariant under N, OH, OW (those dims are time-multiplexed
on the same IMCU), so only the (oc, ic) projection is meaningful. We dump
this projection as int arrays indexed by the *original* conv's (oc, ic).

Atomic-split bookkeeping
------------------------
``split_conv_to_atomic`` slices a large conv along OC (block_size = 64 or
the column-disable effective_oc) and IC (block_size = floor(256/(KH*KW))
for a regular conv, 32 for depthwise). Each (oc_block, ic_block) becomes
its own imcflow function with exactly one qconv. The split is recorded in
the weight Var name as ``{orig}_oc{oc_id}_ic{ic_id}``; we parse that back
to identify the original conv and the (oc, ic) subrange.

Public entry points
-------------------
- ``dump_psum_mapping(mod, output_path, func_column_info=None)``: in-memory
  dump, called from the v1 / v2 compiler driver right after PnR.
- ``dump_psum_mapping_offline(eval_dir, output_path)``: load
  ``transformed_model.pkl`` + ``devconfig_state.pkl`` from a finished build
  and produce the same npz.
"""

import os
import re
import pickle
import pprint
from typing import Optional, Dict, List, Tuple

import numpy as np

import tvm
from tvm import relay


_WEIGHT_OC_IC_RE = re.compile(r"^(.*?)_oc(\d+)_ic(\d+)$")


def _parse_orig_conv(weight_name: str) -> Tuple[str, int, int]:
    """Parse a split-weight var name back into (orig_name, oc_id, ic_id).

    Returns (weight_name, -1, -1) if the pattern does not match (e.g. when
    the weight was a Constant rather than a Var, so the split helper never
    appended ``_oc{}_ic{}``).
    """
    m = _WEIGHT_OC_IC_RE.match(weight_name)
    if not m:
        return weight_name, -1, -1
    return m.group(1), int(m.group(2)), int(m.group(3))


def _weight_const_hash(weight_const) -> Optional[str]:
    """Mirror of Spliter._weight_hash in transform.split_conv_to_atomic.

    Returns None if ``weight_const`` is not a relay.Constant.
    """
    import hashlib
    if not isinstance(weight_const, relay.Constant):
        return None
    data = weight_const.data
    arr = data.asnumpy() if hasattr(data, "asnumpy") \
        else (data.numpy() if hasattr(data, "numpy") else data)
    h = hashlib.sha1()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def _walk_qconv_in_func(func):
    """Return (qconv_call, weight_var_name_or_const_idx) for the single
    qconv inside ``func``.

    ``single_qconv`` / v2 mode guarantees one atomic qconv per imcflow
    function. Returns (None, None) if no qconv is found.
    """
    found = [None]

    class _F(relay.ExprVisitor):
        def visit_call(self, call):
            if found[0] is None and isinstance(call.op, tvm.ir.Op):
                if call.op.name in ("nn.imcflow_qconv", "nn.imcflow_qdwconv"):
                    found[0] = call
            super().visit_call(call)

    _F().visit(func.body)
    qconv = found[0]
    if qconv is None:
        return None, None

    weight = qconv.args[1]
    if isinstance(weight, relay.Var):
        weight_name = weight.name_hint
    elif isinstance(weight, relay.Constant):
        weight_name = "<const>"
    else:
        weight_name = f"<{type(weight).__name__}>"
    return qconv, weight_name


def _qconv_custom_id(qconv_call) -> Optional[int]:
    if qconv_call.attrs is None:
        return None
    if not hasattr(qconv_call.attrs, "custom_id"):
        return None
    return int(qconv_call.attrs["custom_id"])


def _imce_coord_from_node_id(node_id, imce_w_num: int) -> Optional[Tuple[int, int, int]]:
    """Convert a NodeID enum to (row, col_in_imce, imce_linear_id).

    NodeID.to_coord() returns (row, col_pnr) where col_pnr=0 is the INODE
    column and col_pnr=1..IMCE_W_NUM are IMCE columns. Returns None if the
    node is an INODE (qconv should never land on an INODE — this guards
    against malformed maps).
    """
    row, col_pnr = node_id.to_coord()
    if col_pnr == 0:
        return None
    col_in_imce = col_pnr - 1
    imce_linear = row * imce_w_num + col_in_imce
    return row, col_in_imce, imce_linear


def _is_imcflow_func(func, mod) -> bool:
    """Mirror of imcflow.transform_utils.isImcflowFunc but tolerant of
    pickled modules where the helper isn't on the import path."""
    try:
        from tvm.relay.backend.contrib.imcflow.transform_utils import isImcflowFunc
        return isImcflowFunc(func, mod)
    except Exception:
        # Fall back to the surface check used everywhere else: the func name
        # carries the "imcflow" tag and the body contains a qconv.
        if not isinstance(func, relay.Function):
            return False
        return _walk_qconv_in_func(func)[0] is not None


def _build_records(
    mod,
    hw_node_map: Dict,
    imce_w_num: int,
    func_column_info: Optional[Dict] = None,
    column_disable_map: Optional[Dict] = None,
    atomic_split_info: Optional[Dict] = None,
    atomic_split_info_by_func: Optional[Dict] = None,
) -> List[dict]:
    """Walk ``mod`` and produce one record per imcflow function (atomic qconv).

    Each record has the keys consumed by ``_records_to_npz``.

    The split-time metadata in ``atomic_split_info`` (``DevConfig().AtomicSplitInfo``)
    is the primary source for (orig_conv, oc_id, ic_id, oc_block, ic_block,
    total_oc, total_ic). For backwards-compat with earlier builds that did not
    populate it, we fall back to weight-Var-name parsing.
    """
    records = []
    for gv in sorted(mod.functions.keys(), key=lambda g: g.name_hint):
        func = mod[gv]
        if not isinstance(func, relay.Function):
            continue
        if "imcflow" not in gv.name_hint:
            continue
        qconv, weight_name = _walk_qconv_in_func(func)
        if qconv is None:
            continue

        custom_id = _qconv_custom_id(qconv)
        if custom_id is None:
            print(f"[psum_mapping] {gv.name_hint}: qconv has no custom_id, skipping")
            continue

        node_id = hw_node_map.get(custom_id)
        if node_id is None:
            print(f"[psum_mapping] {gv.name_hint}: custom_id={custom_id} "
                  f"not in HWNodeMap, skipping")
            continue
        coord = _imce_coord_from_node_id(node_id, imce_w_num)
        if coord is None:
            print(f"[psum_mapping] {gv.name_hint}: custom_id={custom_id} "
                  f"placed on INODE ({node_id}), skipping")
            continue
        row, col_in_imce, imce_linear = coord

        attrs = qconv.attrs
        oc_size = int(attrs.channels)
        ic_size = int(attrs.in_channels)
        kh, kw = (int(k) for k in attrs.kernel_size)
        op_name = qconv.op.name
        is_depthwise = (op_name == "nn.imcflow_qdwconv")

        # Primary path: function-name keyed lookup. This dict is populated
        # before layout legalization and survives all subsequent rewrites,
        # so it's the most reliable source.
        split_meta = None
        if atomic_split_info_by_func is not None:
            split_meta = atomic_split_info_by_func.get(gv.name_hint)
        # Secondary: hash the (post-legalization) weight Constant. This only
        # matches if AtomicSplitInfo was recorded after layout legalization,
        # but is harmless to attempt and useful for tests / debugging.
        if split_meta is None and atomic_split_info:
            wkey = _weight_const_hash(qconv.args[1])
            if wkey is not None:
                split_meta = atomic_split_info.get(wkey)

        if split_meta is not None:
            orig_conv = split_meta["orig_conv_name"]
            oc_id = split_meta["oc_id"]
            ic_id = split_meta["ic_id"]
            oc_block = split_meta["oc_block"]
            ic_block = split_meta["ic_block"]
            total_oc = split_meta["total_oc"]
            total_ic = split_meta["total_ic"]
            orig_conv_id = split_meta["orig_conv_id"]
        else:
            # Fallback: parse the weight Var name (only works when the weight
            # is still a Var carrying the `_oc{}_ic{}` suffix from
            # split_conv_to_atomic, which is rare post-bind_params).
            orig_conv, oc_id, ic_id = _parse_orig_conv(weight_name)
            oc_block = oc_size if oc_id < 0 else -1
            ic_block = ic_size if ic_id < 0 else -1
            total_oc = -1
            total_ic = -1
            orig_conv_id = -1

        if func_column_info and gv.name_hint in func_column_info:
            _oc_actual, valid_cols = func_column_info[gv.name_hint]
            valid_cols = list(valid_cols[:oc_size])
        elif column_disable_map and imce_linear in column_disable_map and column_disable_map[imce_linear]:
            disabled = set(column_disable_map[imce_linear])
            valid_cols = [c for c in range(64) if c not in disabled][:oc_size]
        else:
            valid_cols = list(range(oc_size))

        records.append({
            "func_name": gv.name_hint,
            "qconv_op": op_name,
            "is_depthwise": is_depthwise,
            "custom_id": custom_id,
            "imce_row": row,
            "imce_col_in_imce": col_in_imce,
            "imce_linear_id": imce_linear,
            "oc_size": oc_size,
            "ic_size": ic_size,
            "kernel_h": kh,
            "kernel_w": kw,
            "oc_id": oc_id,
            "ic_id": ic_id,
            "oc_block": oc_block,
            "ic_block": ic_block,
            "total_oc": total_oc,
            "total_ic": total_ic,
            "orig_conv": orig_conv,
            "orig_conv_id": orig_conv_id,
            "weight_var": weight_name,
            "valid_cols": valid_cols,
        })

    return records


def _records_to_npz(records: List[dict], output_path: str, metadata: Optional[dict] = None):
    """Flatten per-function records to numpy arrays and save as npz.

    Layout
    ------
    Per-function flat arrays (length = number of atomic qconvs):
      func_names[i]           : str
      qconv_op[i]              : str  ("nn.imcflow_qconv" or "nn.imcflow_qdwconv")
      is_depthwise[i]          : bool
      custom_id[i]             : int32
      imce_row[i]              : int32
      imce_col_in_imce[i]      : int32  (0..IMCE_W_NUM-1)
      imce_linear_id[i]        : int32  (0..IMCE_NUM-1)
      oc_size[i], ic_size[i]   : int32
      kernel_h[i], kernel_w[i] : int32
      oc_id[i], ic_id[i]       : int32 (-1 if unknown)
      orig_conv[i]             : str
      weight_var[i]            : str
      valid_cols/<func_name>   : int32 array of length oc_size

    Per-original-conv 2-D mappings (one set per distinct orig_conv name):
      conv/<orig>/imce_linear  : int32 [OC, IC] (-1 = no atomic qconv)
      conv/<orig>/imce_row     : int32 [OC, IC]
      conv/<orig>/imce_col     : int32 [OC, IC]
      conv/<orig>/column       : int32 [OC, IC] (-1 if unknown)

    The 2-D arrays let you broadcast to [N, OC, IC, OH, OW] by tiling
    (the mapping is invariant under N, OH, OW).
    """
    if not records:
        print(f"[psum_mapping] no atomic qconvs found, writing empty npz to {output_path}")

    out: Dict[str, np.ndarray] = {}

    str_keys = {"func_name", "qconv_op", "orig_conv", "weight_var"}
    int_keys = {
        "custom_id", "imce_row", "imce_col_in_imce", "imce_linear_id",
        "oc_size", "ic_size", "kernel_h", "kernel_w", "oc_id", "ic_id",
        "oc_block", "ic_block", "total_oc", "total_ic", "orig_conv_id",
    }
    bool_keys = {"is_depthwise"}

    for k in sorted(str_keys | int_keys | bool_keys):
        values = [r[k] for r in records]
        out_key = "func_names" if k == "func_name" else k
        if k in str_keys:
            out[out_key] = np.array(values, dtype=object)
        elif k in bool_keys:
            out[out_key] = np.array(values, dtype=bool)
        else:
            out[out_key] = np.array(values, dtype=np.int32)

    for r in records:
        out[f"valid_cols/{r['func_name']}"] = np.array(r["valid_cols"], dtype=np.int32)

    # Build per-original-conv [OC, IC] tables. We size each table to cover
    # every (oc_id, ic_id) sub-block that appears, computing the block's
    # global (oc, ic) range from oc_id*oc_size_block + offset.
    # When oc_id/ic_id are -1 (unknown — Constant weight), we group those
    # records under a synthetic name and emit a 1-D record list instead.
    by_conv: Dict[str, List[dict]] = {}
    for r in records:
        by_conv.setdefault(r["orig_conv"], []).append(r)

    for orig, rs in by_conv.items():
        # Skip groups where split metadata is missing entirely (older builds
        # produced before AtomicSplitInfo was added). The per-function flat
        # arrays still carry everything the user needs.
        if any(r["total_oc"] <= 0 or r["total_ic"] <= 0
               or r["oc_id"] < 0 or r["ic_id"] < 0 for r in rs):
            print(f"[psum_mapping] orig_conv={orig!r}: split metadata missing "
                  f"for some records; skipping 2-D conv/<orig>/* tables")
            continue

        # All records in this group share the same original conv, so total_oc /
        # total_ic / *_block are identical across rs. Take from any.
        total_oc = rs[0]["total_oc"]
        total_ic = rs[0]["total_ic"]
        oc_block = rs[0]["oc_block"]
        ic_block = rs[0]["ic_block"]

        imce_lin = -np.ones((total_oc, total_ic), dtype=np.int32)
        imce_row_arr = -np.ones((total_oc, total_ic), dtype=np.int32)
        imce_col_arr = -np.ones((total_oc, total_ic), dtype=np.int32)
        column = -np.ones((total_oc, total_ic), dtype=np.int32)

        for r in rs:
            oc_lo = r["oc_id"] * oc_block
            oc_hi = oc_lo + r["oc_size"]
            # Depthwise: each atomic qconv covers one OC-block in IC=1; the
            # full IC dimension in the original conv is IC=OC, so map the
            # block's IC slice to [oc_lo:oc_hi] as well.
            if r["is_depthwise"]:
                ic_lo, ic_hi = oc_lo, oc_hi
            else:
                ic_lo = r["ic_id"] * ic_block
                ic_hi = ic_lo + r["ic_size"]
            imce_lin[oc_lo:oc_hi, ic_lo:ic_hi] = r["imce_linear_id"]
            imce_row_arr[oc_lo:oc_hi, ic_lo:ic_hi] = r["imce_row"]
            imce_col_arr[oc_lo:oc_hi, ic_lo:ic_hi] = r["imce_col_in_imce"]
            valid = r["valid_cols"]
            for oc_local in range(r["oc_size"]):
                col_id = valid[oc_local] if oc_local < len(valid) else -1
                column[oc_lo + oc_local, ic_lo:ic_hi] = col_id

        prefix = f"conv/{orig}"
        out[f"{prefix}/imce_linear"] = imce_lin
        out[f"{prefix}/imce_row"] = imce_row_arr
        out[f"{prefix}/imce_col"] = imce_col_arr
        out[f"{prefix}/column"] = column

    if metadata:
        out["metadata"] = np.array([repr(metadata)], dtype=object)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez(output_path, **out)
    print(f"[psum_mapping] wrote {len(records)} atomic qconv record(s) to {output_path}")


def _load_func_column_info_txt(path: str) -> Optional[Dict]:
    """Parse a pprint-formatted func_column_info dict file.

    The file is written via ``pprint.pprint(dict, stream=f)`` and contains
    only literals (str keys, tuple of (int, list[int]) values), so
    ``ast.literal_eval`` recovers it safely. Returns None if the file does
    not exist.
    """
    import ast
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        body = f.read()
    try:
        return ast.literal_eval(body)
    except Exception as e:
        print(f"[psum_mapping] failed to parse {path}: {e}")
        return None


def dump_psum_mapping(
    mod,
    output_path: str,
    func_column_info: Optional[Dict] = None,
):
    """In-memory dump. Call after Joint PnR has populated DevConfig.HWNodeMap.

    Args:
        mod: the post-PnR IRModule containing one atomic qconv per imcflow
            function (single_qconv mode or v2).
        output_path: target ``.npz`` path.
        func_column_info: optional dict ``{func_name: (oc_actual, valid_cols)}``
            from the column-disable shape pass; if None, the function falls
            back to ``DevConfig().get_valid_columns(imce_linear_id)``.
    """
    from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
    config = DevConfig()
    metadata = {
        "imce_h_num": config.IMCE_H_NUM,
        "imce_w_num": config.IMCE_W_NUM,
        "imce_num": config.IMCE_NUM,
        "single_qconv": getattr(config, "single_qconv", False),
    }
    records = _build_records(
        mod,
        hw_node_map=config.HWNodeMap,
        imce_w_num=config.IMCE_W_NUM,
        func_column_info=func_column_info,
        column_disable_map=config.ColumnDisableMap,
        atomic_split_info=getattr(config, "AtomicSplitInfo", None),
        atomic_split_info_by_func=getattr(config, "AtomicSplitInfoByFunc", None),
    )
    _records_to_npz(records, output_path, metadata=metadata)


def dump_psum_mapping_offline(
    eval_dir: str,
    output_path: Optional[str] = None,
    transformed_pkl: str = "transformed_model.pkl",
    devconfig_pkl: str = "devconfig_state.pkl",
    func_column_info_txt: str = "func_column_info.txt",
):
    """Reconstruct the npz from a finished build directory.

    Loads ``{eval_dir}/transformed_model.pkl`` (mod) and
    ``{eval_dir}/devconfig_state.pkl`` (HWNodeMap), then optionally pulls
    column-disable info from ``func_column_info.txt`` (v2 only).

    Args:
        eval_dir: path to a model's eval dir (e.g.
            ``eval_dir/<model>_evl.linux/``).
        output_path: where to write the npz. Defaults to
            ``{eval_dir}/psum_imcu_column_map.npz``.

    Returns:
        The output_path actually written.
    """
    if output_path is None:
        output_path = os.path.join(eval_dir, "psum_imcu_column_map.npz")

    pkl_path = os.path.join(eval_dir, transformed_pkl)
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"transformed model not found at {pkl_path}; "
            f"re-run the pipeline through the TRANSFORM stage first")
    with open(pkl_path, "rb") as f:
        save_data = pickle.load(f)
    mod = save_data["mod"]

    devconfig_path = os.path.join(eval_dir, devconfig_pkl)
    if not os.path.exists(devconfig_path):
        raise FileNotFoundError(f"devconfig state not found at {devconfig_path}")
    from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
    config = DevConfig()
    config.load_state(devconfig_path)

    # func_column_info isn't part of devconfig_state.pkl; pull it from the
    # txt sidecar when present (v2 + column-disable).
    func_column_info = _load_func_column_info_txt(
        os.path.join(eval_dir, func_column_info_txt))

    dump_psum_mapping(mod, output_path, func_column_info=func_column_info)
    return output_path
