"""Analyze weight3_0 ref-psum usage and CSV noise rows.

This script is the first stage of the low-level weight3_0 noise workflow:

  1. Read real ResNet dump inputs for weight3_0.
  2. Recompute per-(abit,wbit) raw psum, ADC code, and acc-mode skip mask.
  3. Aggregate how often each CSV lookup row is used.
  4. Emit candidate tuples for synthetic chip sampling.

The aggregation key is intentionally close to the hardware/CSV lookup unit:
  (core_h, core_w, valid_col, pseudo_ch, oc_local, abit, wbit,
   raw_psum, adc_code, skip)

Usage:
    python scripts/analyze_weight3_0_ref_psum.py --n-samples 200
    python scripts/analyze_weight3_0_ref_psum.py --noise-csv B2_noise_matrix_per_ch_concat.csv
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

import diagnose_noise_per_qconv as diag


TARGET_ORIG = "weight3_0"
TARGET_FUNC = "tvmgen_default_imcflow_main_15"
DEFAULT_OUT_DIR = os.path.join(diag.CODEGEN, "debugging/noise_lowlevel/weight3_0")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--sample-start", type=int, default=0)
    ap.add_argument("--noise-csv", type=str, default=diag.DEFAULT_CSV_NAME)
    ap.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--acc-mask", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=20,
                    help="Number of non-skip candidate tuples to write.")
    ap.add_argument("--min-per-wbit", type=int, default=3,
                    help="Try to include at least this many candidates per wbit.")
    ap.add_argument("--prefer-low-adc", type=int, default=8,
                    help="Prioritize adc_code <= this value when selecting candidates.")
    return ap.parse_args()


def find_weight3_0_atomic(atomics):
    matches = [a for a in atomics if a["orig_conv"] == TARGET_ORIG]
    if not matches:
        raise RuntimeError(f"No atomic found for {TARGET_ORIG}")
    exact = [a for a in matches if a["func"] == TARGET_FUNC]
    return exact[0] if exact else matches[0]


def load_weight3_0_weight():
    ckpt = torch.load(diag.CKPT_PATH, map_location="cpu", weights_only=False)
    key = diag.CONV_PARAMS[TARGET_ORIG][3]
    return ckpt["state_dict"][key].cpu().numpy().astype(np.int32)


def find_qconv_output(sample_dir, func):
    for fname in os.listdir(sample_dir):
        if fname.endswith(f"_{func}.npy"):
            return os.path.join(sample_dir, fname)
    return None


def conv1x1_raw_psums(input_slice, weight_tile, stride):
    """Return raw popcounts and per-bit psums for weight3_0.

    input_slice: (1, IC, H, W) uint8
    weight_tile: (OC, IC, 1, 1) signed int4 in int32

    Returns:
      raw_psum_all: (ABITS, WBITS, OC, OH, OW) int16
      popcount_all: (ABITS, OH, OW) int16
    """
    x = input_slice[0]
    w = weight_tile[:, :, 0, 0]
    _, h, w_in = x.shape
    oh = (h - 1) // stride + 1
    ow = (w_in - 1) // stride + 1
    x_sp = x[:, 0:h:stride, 0:w_in:stride]

    raw = np.zeros((diag.ABITS, diag.WBITS, weight_tile.shape[0], oh, ow), dtype=np.int16)
    pop = np.zeros((diag.ABITS, oh, ow), dtype=np.int16)
    w_uint = w & 0xF
    for abit in range(diag.ABITS):
        xb = ((x_sp >> abit) & 1).astype(np.int16)
        pop[abit] = xb.sum(axis=0)
        xb_flat = xb.reshape(xb.shape[0], -1).astype(np.int32)
        for wbit in range(diag.WBITS):
            wb = ((w_uint >> wbit) & 1).astype(np.int32)
            ps = wb @ xb_flat
            raw[abit, wbit] = ps.reshape(weight_tile.shape[0], oh, ow).astype(np.int16)
    return raw, pop


def adc_from_raw(raw_psum):
    return np.clip(np.round(raw_psum.astype(np.float64) / diag.PSTEP + 0.01),
                   0, diag.NUM_LEVELS - 1).astype(np.int16)


def add_csv_stats(row, csv_data):
    pseudo_ch = int(row["pseudo_ch"])
    wbit = int(row["wbit"])
    adc_code = int(row["adc_code"])
    csv_row = wbit * csv_data["n_refs"] + adc_code
    scale = diag.PSTEP * diag.W_SCALE[wbit] * (1 << int(row["abit"]))
    probs = csv_data["probs"][pseudo_ch, csv_row]
    diff_bins = csv_data["diff_bins"]
    nonzero = probs > 1e-12
    row["csv_diff_mean"] = float(csv_data["E"][pseudo_ch, csv_row])
    row["csv_diff_std"] = float(np.sqrt(csv_data["Var"][pseudo_ch, csv_row]))
    row["csv_diff_min"] = float(csv_data["diff_min"][pseudo_ch, csv_row])
    row["csv_diff_max"] = float(csv_data["diff_max"][pseudo_ch, csv_row])
    row["output_scale"] = float(scale)
    row["csv_out_mean"] = float(row["csv_diff_mean"] * scale)
    row["csv_out_std"] = float(row["csv_diff_std"] * abs(scale))
    if nonzero.any():
        row["csv_support_n"] = int(nonzero.sum())
        row["csv_prob_mass"] = float(probs.sum())
        row["csv_mode_diff"] = float(diff_bins[np.argmax(probs)])
    else:
        row["csv_support_n"] = 0
        row["csv_prob_mass"] = 0.0
        row["csv_mode_diff"] = float("nan")
    return row


def candidate_histogram(row, csv_data):
    pseudo_ch = int(row["pseudo_ch"])
    wbit = int(row["wbit"])
    adc_code = int(row["adc_code"])
    csv_row = wbit * csv_data["n_refs"] + adc_code
    probs = csv_data["probs"][pseudo_ch, csv_row]
    nz = probs > 1e-12
    return {
        "diff_bins": [float(x) for x in csv_data["diff_bins"][nz]],
        "probs": [float(x) for x in probs[nz]],
    }


def select_candidates(df, csv_data, top_k, min_per_wbit, prefer_low_adc):
    pool = df[(df["skip"] == False) & (df["count_non_skip"] > 0)].copy()  # noqa: E712
    if pool.empty:
        return []

    pool["low_adc_rank"] = (pool["adc_code"] > prefer_low_adc).astype(np.int64)
    pool = pool.sort_values(
        ["low_adc_rank", "count_non_skip", "count_total"],
        ascending=[True, False, False],
    )

    chosen_idx = []
    for wbit in range(diag.WBITS):
        sub = pool[pool["wbit"] == wbit].head(min_per_wbit)
        chosen_idx.extend(sub.index.tolist())

    for idx in pool.index:
        if len(chosen_idx) >= top_k:
            break
        if idx not in chosen_idx:
            chosen_idx.append(idx)

    candidates = []
    for cand_id, (_, row) in enumerate(pool.loc[chosen_idx[:top_k]].iterrows()):
        d = row.to_dict()
        d["candidate_id"] = cand_id
        d["orig_conv"] = TARGET_ORIG
        d["func"] = TARGET_FUNC
        d["wpattern"] = f"{1 << int(d['wbit']):04b}"
        d["target_popcount"] = int(max(int(d["raw_psum"]), 8))
        d["csv_hist"] = candidate_histogram(d, csv_data)
        candidates.append(_json_sanitize(d))
    return candidates


def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = diag.resolve_noise_csv_path(args.noise_csv)

    print("=" * 100)
    print("  analyze_weight3_0_ref_psum")
    print("=" * 100)
    print(f"  samples   : {args.sample_start} .. {args.sample_start + args.n_samples - 1}")
    print(f"  noise csv : {csv_path}")
    print(f"  out dir   : {args.out_dir}")

    atomics = diag.load_atomic_info(diag.NPZ_PATH)
    atomic = find_weight3_0_atomic(atomics)
    pseudo_map, _, _ = diag.load_pseudo_ch_map(diag.LAYOUT_JSON)
    atomic["pseudo_chs"] = np.array(
        [pseudo_map[(atomic["imce_h"], atomic["imce_w_1based"], int(c))]
         for c in atomic["valid_cols"]],
        dtype=np.int64,
    )
    csv_data = diag.load_noise_csv(csv_path)
    weights = load_weight3_0_weight()

    kh, stride, padding, _ = diag.CONV_PARAMS[TARGET_ORIG]
    if kh != 1 or padding != 0:
        raise RuntimeError(f"{TARGET_ORIG} expected 1x1 pad0, got kh={kh}, padding={padding}")

    agg = defaultdict(lambda: {"count_total": 0, "count_non_skip": 0, "count_skip": 0})
    sample_rows = []

    sample_range = range(args.sample_start, args.sample_start + args.n_samples)
    for s_idx in sample_range:
        sample_dir = os.path.join(diag.FPGA_DIR, f"sample_{s_idx}")
        if not os.path.isdir(sample_dir):
            print(f"  [skip] sample_{s_idx}: missing {sample_dir}")
            continue
        qconv_to_input = diag.build_qconv_to_input_map(sample_dir)
        input_fname = qconv_to_input.get(atomic["func"])
        out_path = find_qconv_output(sample_dir, atomic["func"])
        if input_fname is None or out_path is None:
            print(f"  [skip] sample_{s_idx}: missing input/output for {atomic['func']}")
            continue

        qin = np.load(os.path.join(sample_dir, input_fname))
        ic_lo = atomic["ic_id"] * atomic["ic_block"]
        ic_hi = ic_lo + atomic["ic_size"]
        oc_lo = atomic["oc_id"] * atomic["oc_block"]
        oc_hi = oc_lo + atomic["oc_size"]
        input_slice = qin[:, ic_lo:ic_hi, :, :]
        weight_tile = weights[oc_lo:oc_hi, ic_lo:ic_hi, :, :]

        raw_psum, popcount = conv1x1_raw_psums(input_slice, weight_tile, stride)
        adc_codes = adc_from_raw(raw_psum)

        for abit in range(diag.ABITS):
            acc_mode = (args.acc_mask & (1 << abit)) == 0
            skip_2d = (popcount[abit] < 8) if acc_mode else np.zeros_like(popcount[abit], dtype=bool)
            for wbit in range(diag.WBITS):
                for oc_local, (valid_col, pseudo_ch) in enumerate(zip(atomic["valid_cols"], atomic["pseudo_chs"])):
                    rp = raw_psum[abit, wbit, oc_local]
                    ac = adc_codes[abit, wbit, oc_local]
                    for raw_val, adc_val, skip_val in zip(rp.ravel(), ac.ravel(), skip_2d.ravel()):
                        key = (
                            int(atomic["imce_h"]),
                            int(atomic["imce_w_1based"]),
                            int(valid_col),
                            int(pseudo_ch),
                            int(oc_local),
                            int(abit),
                            int(wbit),
                            int(raw_val),
                            int(adc_val),
                            bool(skip_val),
                        )
                        agg[key]["count_total"] += 1
                        if skip_val:
                            agg[key]["count_skip"] += 1
                        else:
                            agg[key]["count_non_skip"] += 1

        sample_rows.append({
            "sample": s_idx,
            "popcount_min": int(popcount.min()),
            "popcount_max": int(popcount.max()),
            "skip_frac": float(((popcount < 8).sum()) / popcount.size),
        })

    rows = []
    for key, counts in agg.items():
        (core_h, core_w, valid_col, pseudo_ch, oc_local,
         abit, wbit, raw_psum, adc_code, skip) = key
        row = {
            "orig_conv": TARGET_ORIG,
            "func": atomic["func"],
            "core_h": core_h,
            "core_w": core_w,
            "valid_col": valid_col,
            "pseudo_ch": pseudo_ch,
            "oc_local": oc_local,
            "abit": abit,
            "wbit": wbit,
            "raw_psum": raw_psum,
            "adc_code": adc_code,
            "skip": skip,
            **counts,
        }
        rows.append(add_csv_stats(row, csv_data))

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["count_total", "count_non_skip", "abit", "wbit", "pseudo_ch"],
            ascending=[False, False, True, True, True],
        )
    by_tuple_path = os.path.join(args.out_dir, "weight3_0_ref_psum_by_tuple.csv")
    df.to_csv(by_tuple_path, index=False)

    if df.empty:
        by_adc = pd.DataFrame()
    else:
        by_adc = (df.groupby(["abit", "wbit", "adc_code", "skip"], as_index=False)
                    .agg(count_total=("count_total", "sum"),
                         count_non_skip=("count_non_skip", "sum"),
                         count_skip=("count_skip", "sum"),
                         csv_out_mean_mean=("csv_out_mean", "mean"),
                         csv_out_std_mean=("csv_out_std", "mean")))
    by_adc_path = os.path.join(args.out_dir, "weight3_0_ref_psum_by_adc.csv")
    by_adc.to_csv(by_adc_path, index=False)

    candidates = select_candidates(df, csv_data, args.top_k, args.min_per_wbit, args.prefer_low_adc)
    candidate_path = os.path.join(args.out_dir, "weight3_0_candidate_tuples.json")
    with open(candidate_path, "w") as f:
        json.dump({
            "metadata": {
                "orig_conv": TARGET_ORIG,
                "func": atomic["func"],
                "noise_csv": csv_path,
                "sample_start": args.sample_start,
                "n_samples": args.n_samples,
                "acc_mask": args.acc_mask,
                "ic_size": int(atomic["ic_size"]),
                "oc_size": int(atomic["oc_size"]),
                "core_h": int(atomic["imce_h"]),
                "core_w": int(atomic["imce_w_1based"]),
                "valid_cols": [int(x) for x in atomic["valid_cols"]],
                "samples": sample_rows,
            },
            "candidates": candidates,
        }, f, indent=2, sort_keys=True)

    print(f"  wrote {by_tuple_path} ({len(df)} rows)")
    print(f"  wrote {by_adc_path} ({len(by_adc)} rows)")
    print(f"  wrote {candidate_path} ({len(candidates)} candidates)")
    if not df.empty:
        non_skip = int(df["count_non_skip"].sum())
        total = int(df["count_total"].sum())
        print(f"  non-skip coverage: {non_skip}/{total} ({non_skip / max(total, 1) * 100:.2f}%)")
        print("\nTop non-skip rows:")
        cols = ["abit", "wbit", "raw_psum", "adc_code", "valid_col", "pseudo_ch",
                "count_non_skip", "csv_diff_mean", "csv_diff_std", "output_scale"]
        print(df[df["count_non_skip"] > 0].head(10)[cols].to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
