"""Build a concat per-pseudo-channel noise CSV from TargetPsumNoisePlanner pkl.

This converts the focused target-psum planner output into the same wire format
as CIM/noise/noise_df/B2_out/N32/B2_noise_matrix_per_ch_concat.csv:

  rows    : 0001_0.0 ... 1000_126.0 (4 wbits x 64 ADC references)
  columns : two-level MultiIndex (diff_bin, pseudo_channel)

The count/probability construction intentionally reuses CIM's
noise_csv_util.py normalization logic, then applies the existing
concat_per_core.json pseudo-channel layout so downstream code can use
--noise-csv without changing its lookup path.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CODEGEN = Path("/root/project/tvm/tvm_practice/test_imcflow/codegen")
CIM_NOISE = Path("/root/project/CIM/noise")
IMCFLOW = Path("/root/project/imcflow")
DEFAULT_RESULT_PKL = (
    IMCFLOW / "xilinx/measurement/workspace/TargetPsumNoisePlanner/results/result.pkl"
)
DEFAULT_LAYOUT_JSON = (
    CIM_NOISE / "noise_df/B2_out/N32/concat_per_core.json"
)
DEFAULT_OUTPUT = (
    CODEGEN / "debugging/noise_lowlevel/target_psum_csv/"
    "B2_target_psum_noise_matrix_per_ch_concat.csv"
)

N_CH = 64
WPATTERNS = ("0001", "0010", "0100", "1000")


def add_import_paths():
    paths = [
        CIM_NOISE,
        IMCFLOW / "xilinx/measurement",
        IMCFLOW / "pmap/ISA_sim/multi_core",
        IMCFLOW / "pmap/ISA_sim/multi_core/test",
        IMCFLOW / "pmap/compiler/src/python",
        IMCFLOW / "pmap/include",
    ]
    for p in reversed(paths):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result-pkl", default=str(DEFAULT_RESULT_PKL))
    ap.add_argument("--layout-json", default=str(DEFAULT_LAYOUT_JSON),
                    help="Existing concat_per_core.json defining pseudo channels.")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--metadata-out", default=None,
                    help="Default: <output>.meta.json")
    ap.add_argument("--round-bins", action="store_true", default=True,
                    help="Round normalized diff/ref bins. Enabled by default.")
    ap.add_argument("--no-round-bins", dest="round_bins", action="store_false")
    ap.add_argument("--fill-missing", choices=["zero-delta", "uniform"], default="zero-delta",
                    help="Distribution for unmeasured ADC refs.")
    ap.add_argument("--support-pad", type=int, default=0,
                    help="Expand non-zero support by +/- this many diff bins with tiny mass.")
    ap.add_argument("--pad-alpha", type=float, default=1e-6,
                    help="Probability floor used with --support-pad before renormalization.")
    return ap.parse_args()


def load_result_df(path):
    from df_util import convert_to_numeric

    df = pd.read_pickle(path)
    df = convert_to_numeric(df)
    required = {"res", "ref", "wpattern", "h_id", "w_id", "target_ref_psum"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def full_row_index():
    refs = [float(v) for v in range(0, 128, 2)]
    return [f"{wp}_{ref:.1f}" for wp in WPATTERNS for ref in refs]


def full_diff_columns(per_pseudo, support_pad):
    cols = {0.0}
    for ch_df in per_pseudo.values():
        if ch_df is not None and not ch_df.empty:
            cols.update(float(c) for c in ch_df.columns)
    if support_pad > 0:
        padded = set(cols)
        for c in cols:
            center = int(round(c))
            for d in range(center - support_pad, center + support_pad + 1):
                padded.add(float(d))
        cols = padded
    return sorted(cols)


def apply_support_pad(aligned, support_pad, pad_alpha):
    if support_pad <= 0:
        return aligned
    values = aligned.to_numpy(dtype=np.float64, copy=True)
    cols = np.asarray(aligned.columns, dtype=np.float64)
    nonzero = values > 0
    for row in range(values.shape[0]):
        active_cols = cols[nonzero[row]]
        for c in active_cols:
            lo = c - support_pad
            hi = c + support_pad
            values[row, (cols >= lo) & (cols <= hi)] += pad_alpha
    return pd.DataFrame(values, index=aligned.index, columns=aligned.columns)


def missing_distribution(index, columns, mode):
    df = pd.DataFrame(0.0, index=index, columns=columns)
    if mode == "zero-delta":
        if 0.0 not in df.columns:
            raise ValueError("0.0 diff column is required for zero-delta fill")
        df.loc[:, 0.0] = 1.0
    elif mode == "uniform":
        df.loc[:, :] = 1.0 / len(columns)
    else:
        raise ValueError(mode)
    return df


def save_concat_full_grid(per_pseudo, n_pseudo, output_path, fill_missing, support_pad, pad_alpha):
    index = full_row_index()
    columns = full_diff_columns(per_pseudo, support_pad)
    aligned_dfs = []
    missing_rows = 0

    for pseudo_ch in range(n_pseudo):
        ch_df = per_pseudo.get(pseudo_ch)
        if ch_df is not None and not ch_df.empty:
            ch_df = ch_df.copy()
            ch_df.columns = ch_df.columns.astype(float)
            aligned = ch_df.reindex(index=index, columns=columns, fill_value=0.0)
            empty = aligned.sum(axis=1) <= 0
            missing_rows += int(empty.sum())
            if empty.any():
                aligned.loc[empty, :] = missing_distribution(
                    aligned.index[empty], columns, fill_missing
                ).to_numpy()
            aligned = apply_support_pad(aligned, support_pad, pad_alpha)
        else:
            missing_rows += len(index)
            aligned = missing_distribution(index, columns, fill_missing)

        vals = aligned.to_numpy(dtype=np.float64)
        row_sums = vals.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        aligned = pd.DataFrame(vals / row_sums, index=index, columns=columns)
        aligned.columns = pd.MultiIndex.from_arrays(
            [aligned.columns.astype(float), [pseudo_ch] * len(aligned.columns)],
            names=["diff_bin", "channel"],
        )
        aligned.index.name = "wpattern_ref"
        aligned_dfs.append(aligned)

    merged = pd.concat(aligned_dfs, axis=1)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path)
    return {
        "rows": int(merged.shape[0]),
        "cols": int(merged.shape[1]),
        "n_pseudo_ch": int(n_pseudo),
        "n_diff_bins": int(len(columns)),
        "diff_min": float(min(columns)),
        "diff_max": float(max(columns)),
        "filled_missing_rows": int(missing_rows),
    }


def build_per_pseudo(cache, layout):
    from noise_csv_util import _counts_to_prob

    pseudo_map = layout["pseudo_ch_to_orig"]
    n_pseudo = int(layout["n_pseudo_ch"])
    per_pseudo = {}
    missing = []
    for pseudo_str, entry in pseudo_map.items():
        pseudo = int(pseudo_str)
        h_str, w_str = str(entry["core"]).split("_")
        core = (int(h_str), int(w_str))
        orig_ch = int(entry["orig_ch"])
        cdf = cache.get(core, {}).get(orig_ch)
        if cdf is None or cdf.empty:
            per_pseudo[pseudo] = pd.DataFrame(index=pd.Index([], name="wpattern_ref"))
            missing.append((pseudo, core, orig_ch))
        else:
            per_pseudo[pseudo] = _counts_to_prob(cdf)
    if sorted(per_pseudo) != list(range(n_pseudo)):
        missing_ids = sorted(set(range(n_pseudo)) - set(per_pseudo))
        raise ValueError(f"layout pseudo channels are not contiguous; missing={missing_ids[:10]}")
    return per_pseudo, missing


def main():
    args = parse_args()
    add_import_paths()

    from noise_csv_util import build_per_core_cache

    result_pkl = os.path.abspath(args.result_pkl)
    layout_json = os.path.abspath(args.layout_json)
    output = os.path.abspath(args.output)
    metadata_out = args.metadata_out or (output + ".meta.json")

    print(f"Loading pkl: {result_pkl}", flush=True)
    df = load_result_df(result_pkl)
    print(f"Loaded {len(df)} rows; cores={df[['h_id', 'w_id']].drop_duplicates().shape[0]}", flush=True)

    print("Building CIM noise_csv_util per-core cache ...", flush=True)
    cache = build_per_core_cache(
        df,
        normalize_adc_step=True,
        normalize_wpattern=True,
        round_bins=args.round_bins,
    )

    with open(layout_json) as f:
        layout = json.load(f)
    per_pseudo, missing = build_per_pseudo(cache, layout)

    print(f"Saving concat CSV: {output}", flush=True)
    stats = save_concat_full_grid(
        per_pseudo,
        int(layout["n_pseudo_ch"]),
        output,
        fill_missing=args.fill_missing,
        support_pad=args.support_pad,
        pad_alpha=args.pad_alpha,
    )

    metadata = {
        "source_result_pkl": result_pkl,
        "layout_json": layout_json,
        "round_bins": bool(args.round_bins),
        "fill_missing": args.fill_missing,
        "support_pad": int(args.support_pad),
        "pad_alpha": float(args.pad_alpha),
        "input_rows": int(len(df)),
        "target_ref_psum": sorted(map(int, df["target_ref_psum"].unique())),
        "wpattern": sorted(str(v).zfill(4) for v in df["wpattern"].astype(str).unique()),
        "missing_pseudo_entries": [
            {"pseudo_ch": int(p), "core": f"{c[0]}_{c[1]}", "orig_ch": int(ch)}
            for p, c, ch in missing
        ],
        **stats,
    }
    with open(metadata_out, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {metadata_out}", flush=True)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
