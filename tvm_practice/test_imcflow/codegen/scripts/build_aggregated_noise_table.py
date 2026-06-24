#!/usr/bin/env python3
"""Build aggregated noise table from chip dumps.

For each output element we have:
  - pseudo_ch: which CIM column produced it
  - clean_ref: noise-free int16 psum (aggregated across all bitplanes)
  - obs_noise: chip_output - clean_ref (int16 level)

We build:
  P(obs_noise | pseudo_ch, clean_ref_bin)

This is a direct observation-based noise model — no per-bitplane decomposition,
no reliance on the original CSV.

Usage:
  python scripts/build_aggregated_noise_table.py \\
      --dump-dir debugging/fpga/uqat_tmp02_refine_ndis32_0_78/run_00 \\
      --samples 0-199 \\
      --noise-dir /root/project/CIM/noise/noise_df/B2_out/N32 \\
      --output aggregated_noise_table.npz

  # Multiple runs (data from all runs is combined):
  python scripts/build_aggregated_noise_table.py \\
      --dump-dir debugging/fpga/uqat_tmp02_refine_ndis32_0_78/run_0{0..9} \\
      --samples 0-99 \\
      --layout-json /path/to/concat_per_core.json \\
      --npz-path eval_dir/.../psum_imcu_column_map.npz \\
      --output aggregated_noise_table_multi.npz
"""

import os, sys, argparse, json
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

CODEGEN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CIM_DIR = '/root/project/CIM'
sys.path.insert(0, CIM_DIR)
sys.path.insert(0, os.path.join(CODEGEN, 'scripts'))

from diagnose_noise_per_qconv import (
    signed_int16, load_atomic_info, load_pseudo_ch_map,
    build_qconv_to_input_map, noise_free_qconv,
    PSTEP, NUM_LEVELS, WBITS, ABITS, W_SCALE, CONV_PARAMS,
)

DEFAULT_NOISE_DIR = '/root/project/CIM/noise/noise_df/B2_out/N32'
DEFAULT_NPZ_PATH = os.path.join(
    CODEGEN, 'eval_dir/resnet8_subset31_pretrained_orig_evl.linux/psum_imcu_column_map.npz')


def parse_acc_mask(value):
    text = str(value).strip()
    if text.startswith('AccMask.'):
        text = text.split('.', 1)[1]
    if text.startswith('BM_'):
        text = text[3:]
    try:
        mask = int(text, 2) if len(text) == 4 and set(text) <= {'0', '1'} else int(text, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'invalid acc mask: {value!r}') from exc
    if not 0 <= mask < 16:
        raise argparse.ArgumentTypeError(f'acc mask out of range [0, 16): {value!r}')
    return mask


def parse_sample_range(s):
    result = []
    for part in s.split(','):
        if '-' in part:
            lo, hi = part.split('-')
            result.extend(range(int(lo), int(hi) + 1))
        else:
            result.append(int(part))
    return result


def iter_existing_sample_dirs(dump_dirs, sample_range):
    for dump_dir in dump_dirs:
        for s_idx in sample_range:
            sample_dir = os.path.join(dump_dir, f'sample_{s_idx}')
            if os.path.isdir(sample_dir):
                yield dump_dir, s_idx, sample_dir


def load_weights(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict']
    weights = {}
    for orig, (kh, st, pad, key) in CONV_PARAMS.items():
        w = sd[key].cpu().numpy().astype(np.int32)
        weights[orig] = w
    return weights


class AggregatedNoiseAccumulator:
    """Accumulate (pseudo_ch, clean_ref_bin) -> obs noise statistics."""

    def __init__(self, n_pseudo, ref_bin_edges, noise_bin_edges):
        self.n_pseudo = n_pseudo
        self.ref_bin_edges = ref_bin_edges
        self.noise_bin_edges = noise_bin_edges
        self.n_ref_bins = len(ref_bin_edges) - 1
        self.n_noise_bins = len(noise_bin_edges) - 1
        self.ref_bin_centers = (ref_bin_edges[:-1] + ref_bin_edges[1:]) / 2
        self.noise_bin_centers = (noise_bin_edges[:-1] + noise_bin_edges[1:]) / 2

        # Histogram: (n_pseudo, n_ref_bins, n_noise_bins)
        self.hist = np.zeros((n_pseudo, self.n_ref_bins, self.n_noise_bins), dtype=np.float64)
        # Raw stats for E/Var without binning artifacts
        self.obs_sum = np.zeros((n_pseudo, self.n_ref_bins), dtype=np.float64)
        self.obs_sum2 = np.zeros((n_pseudo, self.n_ref_bins), dtype=np.float64)
        self.count = np.zeros((n_pseudo, self.n_ref_bins), dtype=np.int64)

    def add_batch(self, pseudo_chs, clean_ref, obs_noise):
        """Add observations.

        Args:
            pseudo_chs: (OC,) int
            clean_ref: (OC, OH, OW) int32 — noise-free aggregated psum
            obs_noise: (OC, OH, OW) int32 — observed noise (chip - clean)
        """
        OC = len(pseudo_chs)
        for oc_i in range(OC):
            pch = pseudo_chs[oc_i]
            r = clean_ref[oc_i].ravel().astype(np.float64)
            n = obs_noise[oc_i].ravel().astype(np.float64)

            # Bin clean_ref
            ref_idx = np.digitize(r, self.ref_bin_edges) - 1
            ref_idx = np.clip(ref_idx, 0, self.n_ref_bins - 1)

            # Bin obs_noise
            noise_idx = np.digitize(n, self.noise_bin_edges) - 1
            noise_idx = np.clip(noise_idx, 0, self.n_noise_bins - 1)

            # Scatter-add into histogram
            np.add.at(self.hist[pch], (ref_idx, noise_idx), 1.0)
            np.add.at(self.obs_sum[pch], (ref_idx,), n)
            np.add.at(self.obs_sum2[pch], (ref_idx,), n * n)
            np.add.at(self.count[pch], (ref_idx,), 1)

    def results(self, min_count=10, smoothing=0.0):
        """Compute probability table and statistics."""
        # Normalize histograms to probabilities
        probs = self.hist.copy()
        for pch in range(self.n_pseudo):
            for rb in range(self.n_ref_bins):
                total = probs[pch, rb].sum()
                if total < min_count:
                    # Not enough data: delta at noise=0
                    zero_idx = np.argmin(np.abs(self.noise_bin_centers))
                    probs[pch, rb] = 0.0
                    probs[pch, rb, zero_idx] = 1.0
                else:
                    if smoothing > 0:
                        probs[pch, rb] += smoothing
                    probs[pch, rb] /= probs[pch, rb].sum()

        # E and Var from raw sums
        E = np.zeros((self.n_pseudo, self.n_ref_bins))
        Var = np.zeros((self.n_pseudo, self.n_ref_bins))
        valid = self.count > 0
        E[valid] = self.obs_sum[valid] / self.count[valid]
        Var[valid] = self.obs_sum2[valid] / self.count[valid] - E[valid] ** 2
        Var = np.maximum(Var, 0.0)

        return {
            'probs': probs,                     # (n_pseudo, n_ref_bins, n_noise_bins)
            'ref_bin_edges': self.ref_bin_edges,
            'ref_bin_centers': self.ref_bin_centers,
            'noise_bin_edges': self.noise_bin_edges,
            'noise_bin_centers': self.noise_bin_centers,
            'E': E,                              # (n_pseudo, n_ref_bins) mean obs noise
            'Var': Var,                           # (n_pseudo, n_ref_bins) variance
            'Std': np.sqrt(Var),
            'count': self.count,                  # (n_pseudo, n_ref_bins)
        }


class RefOnlyCsvAccumulator:
    """Accumulate exact P(obs_noise | pseudo_ch, clean_ref) for CIM training CSVs."""

    def __init__(self, n_pseudo):
        self.n_pseudo = int(n_pseudo)
        self.per_ch = [Counter() for _ in range(self.n_pseudo)]
        self.aggregate = Counter()
        self.total = 0

    @staticmethod
    def _canonical_int_array(values):
        values = np.asarray(values)
        if not np.issubdtype(values.dtype, np.integer):
            rounded = np.rint(values)
            if not np.allclose(values, rounded, rtol=0.0, atol=0.0):
                raise ValueError("ref/noise CSV output requires integer-valued observations")
            values = rounded
        return values.astype(np.int64, copy=False)

    def add_batch(self, pseudo_chs, clean_ref, obs_noise):
        clean_ref = self._canonical_int_array(clean_ref)
        obs_noise = self._canonical_int_array(obs_noise)
        for oc_i, pch in enumerate(pseudo_chs):
            pch = int(pch)
            refs = clean_ref[oc_i].reshape(-1)
            diffs = obs_noise[oc_i].reshape(-1)
            pairs = np.stack([refs, diffs], axis=1)
            unique, counts = np.unique(pairs, axis=0, return_counts=True)
            for (ref, diff), count in zip(unique, counts):
                key = (int(ref), int(diff))
                count = int(count)
                self.per_ch[pch][key] += count
                self.aggregate[key] += count
                self.total += count

    def to_dataframe(self, min_count=1, fallback="aggregate"):
        if self.total <= 0:
            raise ValueError("no observations were accumulated for CSV output")
        if fallback not in ("aggregate", "zero"):
            raise ValueError(f"unknown fallback policy: {fallback}")

        refs = sorted({ref for counter in self.per_ch for ref, _ in counter.keys()})
        diffs = sorted({diff for counter in self.per_ch for _, diff in counter.keys()})
        if 0 not in diffs:
            diffs.append(0)
            diffs = sorted(diffs)

        columns = pd.MultiIndex.from_product(
            [list(map(float, diffs)), list(range(self.n_pseudo))],
            names=["diff_bin", "channel"],
        )
        out = pd.DataFrame(0.0, index=pd.Index(list(map(float, refs)), name="ref"), columns=columns)

        by_ch_ref = []
        for counter in self.per_ch:
            ref_map = {}
            for (ref, diff), count in counter.items():
                ref_map.setdefault(ref, {})[diff] = count
            by_ch_ref.append(ref_map)
        aggregate_by_ref = {}
        for (ref, diff), count in self.aggregate.items():
            aggregate_by_ref.setdefault(ref, {})[diff] = count

        zero_idx = float(0)
        for pch, ref_map in enumerate(by_ch_ref):
            for ref in refs:
                ref_counts = ref_map.get(ref, {})
                total = sum(ref_counts.values())
                if total < min_count:
                    if fallback == "aggregate":
                        ref_counts = aggregate_by_ref.get(ref, {})
                        total = sum(ref_counts.values())
                    if total <= 0:
                        out.loc[float(ref), (zero_idx, pch)] = 1.0
                        continue
                for diff, count in ref_counts.items():
                    out.loc[float(ref), (float(diff), pch)] = float(count) / float(total)
        return out

    def summary(self):
        nonempty = sum(1 for c in self.per_ch if c)
        return {
            "n_pseudo": self.n_pseudo,
            "nonempty_pseudo_channels": nonempty,
            "total_observations": int(self.total),
            "ref_count": len({ref for c in self.per_ch for ref, _ in c.keys()}),
            "diff_bin_count": len({diff for c in self.per_ch for _, diff in c.keys()}),
        }


def write_ref_only_csv(acc, csv_path, layout_json, layout_output=None, metadata_output=None, min_count=1):
    """Write CIM-compatible ref-only per-channel CSV and matching layout JSON."""
    df = acc.to_dataframe(min_count=min_count)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    df.to_csv(csv_path)
    print(f"Ref-only CSV saved: {csv_path} ({df.shape[0]} refs x {df.shape[1]} columns)")

    if layout_output is None:
        layout_output = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "concat_per_core.json")
    with open(layout_json) as f:
        layout = json.load(f)
    os.makedirs(os.path.dirname(os.path.abspath(layout_output)), exist_ok=True)
    with open(layout_output, "w") as f:
        json.dump(layout, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Layout JSON saved: {layout_output}")

    if metadata_output is None:
        metadata_output = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "chip_ref_noise_metadata.json")
    metadata = acc.summary()
    metadata.update({
        "csv_path": csv_path,
        "layout_json": layout_output,
        "table_format": "ref",
        "granularity": "output",
        "noise_unit": "output",
        "source": "build_aggregated_noise_table.py chip debug dumps",
    })
    with open(metadata_output, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"CSV metadata saved: {metadata_output}")


def main():
    parser = argparse.ArgumentParser(description='Build aggregated noise table')
    parser.add_argument('--dump-dir', required=True, nargs='+')
    parser.add_argument('--samples', default='0-199')
    parser.add_argument('--output', default='aggregated_noise_table.npz')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--n-ref-bins', type=int, default=200,
                        help='Number of clean_ref bins')
    parser.add_argument('--ref-range', type=float, nargs=2, default=None,
                        help='Clean ref range (default: auto)')
    parser.add_argument('--n-noise-bins', type=int, default=200,
                        help='Number of obs noise bins')
    parser.add_argument('--noise-range', type=float, nargs=2, default=None,
                        help='Obs noise range (default: auto)')
    parser.add_argument('--smoothing', type=float, default=0.0)
    parser.add_argument('--min-count', type=int, default=10)
    parser.add_argument('--csv-output', default=None,
                        help='Optional CIM-compatible ref-only per-channel CSV output path')
    parser.add_argument('--csv-layout-output', default=None,
                        help='Optional concat_per_core.json output path for --csv-output')
    parser.add_argument('--csv-metadata-output', default=None,
                        help='Optional metadata JSON output path for --csv-output')
    parser.add_argument('--csv-min-count', type=int, default=1,
                        help='Minimum observations for a (pseudo_ch, ref) CSV row before fallback')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--acc-mask', type=parse_acc_mask, default=0,
                        help='4-bit acc mask used by the compiled chip model')
    parser.add_argument('--noise-dir', default=DEFAULT_NOISE_DIR,
                        help='Directory containing concat_per_core.json '
                             f'(default: {DEFAULT_NOISE_DIR})')
    parser.add_argument('--layout-json', default=None,
                        help='Path to concat_per_core.json. If omitted, uses '
                             '--noise-dir/concat_per_core.json')
    parser.add_argument('--npz-path', default=DEFAULT_NPZ_PATH,
                        help='Path to psum_imcu_column_map.npz '
                             f'(default: {DEFAULT_NPZ_PATH})')
    args = parser.parse_args()

    dump_dirs = [os.path.join(CODEGEN, d) if not os.path.isabs(d) else d
                 for d in args.dump_dir]
    noise_dir = args.noise_dir if os.path.isabs(args.noise_dir) else os.path.join(CODEGEN, args.noise_dir)
    layout_json = args.layout_json or os.path.join(noise_dir, 'concat_per_core.json')
    if not os.path.isabs(layout_json):
        layout_json = os.path.join(CODEGEN, layout_json)
    npz_path = args.npz_path if os.path.isabs(args.npz_path) else os.path.join(CODEGEN, args.npz_path)

    sample_range = parse_sample_range(args.samples)
    print(f"Dump dirs ({len(dump_dirs)}):")
    for d in dump_dirs:
        print(f"  {d}")
    print(f"Samples: {sample_range[0]}..{sample_range[-1]} ({len(sample_range)})")
    print(f"Noise dir: {noise_dir}")
    print(f"Layout JSON: {layout_json}")
    print(f"NPZ path: {npz_path}")

    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        # Walk up from first dump_dir to find ckpt alias (skip run_XX levels)
        alias_dir = dump_dirs[0].rstrip('/')
        while os.path.basename(alias_dir).startswith('run_'):
            alias_dir = os.path.dirname(alias_dir)
        alias = os.path.basename(alias_dir)
        from checkpoints import resolve as ckpt_resolve
        ckpt_path, _ = ckpt_resolve('B2', 'half', alias)
    print(f"Checkpoint: {ckpt_path}")

    print("Loading metadata...")
    atomics = load_atomic_info(npz_path)
    pseudo_map, n_per_core, n_pseudo = load_pseudo_ch_map(layout_json)
    weights = load_weights(ckpt_path)

    for a in atomics:
        h = a['imce_h']
        w = a['imce_w_1based']
        a['pseudo_chs'] = np.array(
            [pseudo_map[(h, w, int(c))] for c in a['valid_cols']],
            dtype=np.int64,
        )

    print(f"  Atomics: {len(atomics)}, pseudo_chs: {n_pseudo}")

    # Auto-determine ranges from a subset
    if args.ref_range is None or args.noise_range is None:
        print("Scanning subset for range estimation...")
        ref_samples, noise_samples = [], []
        n_scanned_samples = 0
        for dump_dir, s_idx, sample_dir in iter_existing_sample_dirs(dump_dirs, sample_range):
            if n_scanned_samples >= 5:
                break
            n_scanned_samples += 1
            qconv_to_input = build_qconv_to_input_map(sample_dir)
            for a in atomics:  # All atomics to catch worst-case channels
                orig = a['orig_conv']
                kh, st, pad, _ = CONV_PARAMS[orig]
                input_fname = qconv_to_input.get(a['func'])
                if input_fname is None:
                    continue
                out_npy = None
                for f in os.listdir(sample_dir):
                    if f.endswith(f'_{a["func"]}.npy'):
                        out_npy = os.path.join(sample_dir, f)
                        break
                if out_npy is None:
                    continue
                try:
                    qin = np.load(os.path.join(sample_dir, input_fname), allow_pickle=True)
                    dump = np.load(out_npy, allow_pickle=True)
                except Exception:
                    continue
                ic_lo = a['ic_id'] * a['ic_block']
                ic_hi = ic_lo + a['ic_size']
                w_full = weights[orig]
                oc_lo = a['oc_id'] * a['oc_block']
                oc_hi = oc_lo + a['oc_size']
                w_tile = w_full[oc_lo:oc_hi, ic_lo:ic_hi, :, :]
                clean_out, _, _ = noise_free_qconv(
                    qin[:, ic_lo:ic_hi, :, :], w_tile,
                    kernel_h=kh, stride=st, padding=pad,
                    acc_mask=args.acc_mask, device=args.device)
                clean_sq = clean_out.squeeze(0)
                dump_sq = dump[0, 0]
                dump_sel = dump_sq[:, :, a['valid_cols']].transpose(2, 0, 1).astype(np.int32)
                obs = signed_int16(dump_sel - clean_sq)
                ref_samples.append(clean_sq.ravel())
                noise_samples.append(obs.ravel())
        if not ref_samples:
            checked = []
            for dump_dir in dump_dirs:
                preview = [os.path.join(dump_dir, f'sample_{s}') for s in sample_range[:5]]
                checked.extend(preview)
            raise RuntimeError(
                "No usable qconv dump pairs found while estimating ranges. "
                "This is not caused by the noise CSV row format; "
                "build_aggregated_noise_table.py does not read the noise CSV. "
                "Check that --dump-dir points to the directory containing sample_<N> "
                f"subdirectories. First checked paths: {checked[:5]}"
            )

        all_ref = np.concatenate(ref_samples)
        all_noise = np.concatenate(noise_samples)

    if args.ref_range is None:
        ref_lo, ref_hi = float(all_ref.min()), float(all_ref.max())
        margin = (ref_hi - ref_lo) * 0.02
        ref_lo, ref_hi = ref_lo - margin, ref_hi + margin
    else:
        ref_lo, ref_hi = args.ref_range

    if args.noise_range is None:
        n_abs_max = float(np.abs(all_noise).max())
        noise_lo, noise_hi = -n_abs_max * 1.02, n_abs_max * 1.02
    else:
        noise_lo, noise_hi = args.noise_range

    print(f"  Ref range: [{ref_lo:.0f}, {ref_hi:.0f}] ({args.n_ref_bins} bins)")
    print(f"  Noise range: [{noise_lo:.0f}, {noise_hi:.0f}] ({args.n_noise_bins} bins)")

    ref_bin_edges = np.linspace(ref_lo, ref_hi, args.n_ref_bins + 1)
    noise_bin_edges = np.linspace(noise_lo, noise_hi, args.n_noise_bins + 1)
    acc = AggregatedNoiseAccumulator(n_pseudo, ref_bin_edges, noise_bin_edges)
    csv_acc = RefOnlyCsvAccumulator(n_pseudo) if args.csv_output else None

    # Main pass
    print(f"\nProcessing {len(dump_dirs)} dirs × {len(sample_range)} samples × {len(atomics)} atomics...")
    n_done = 0
    for dd_idx, dump_dir in enumerate(dump_dirs):
      print(f"\n--- Dir {dd_idx+1}/{len(dump_dirs)}: {os.path.basename(dump_dir)} ---")
      for s_idx in sample_range:
        sample_dir = os.path.join(dump_dir, f'sample_{s_idx}')
        if not os.path.isdir(sample_dir):
            continue
        qconv_to_input = build_qconv_to_input_map(sample_dir)

        for a in atomics:
            orig = a['orig_conv']
            kh, st, pad, _ = CONV_PARAMS[orig]
            input_fname = qconv_to_input.get(a['func'])
            if input_fname is None:
                continue
            out_npy = None
            for f in os.listdir(sample_dir):
                if f.endswith(f'_{a["func"]}.npy'):
                    out_npy = os.path.join(sample_dir, f)
                    break
            if out_npy is None:
                continue

            try:
                qin = np.load(os.path.join(sample_dir, input_fname), allow_pickle=True)
                dump = np.load(out_npy, allow_pickle=True)
            except Exception as e:
                print(f"  [skip] sample_{s_idx} {a['func']}: {e}")
                continue

            ic_lo = a['ic_id'] * a['ic_block']
            ic_hi = ic_lo + a['ic_size']
            w_full = weights[orig]
            oc_lo = a['oc_id'] * a['oc_block']
            oc_hi = oc_lo + a['oc_size']
            w_tile = w_full[oc_lo:oc_hi, ic_lo:ic_hi, :, :]

            clean_out, _, _ = noise_free_qconv(
                qin[:, ic_lo:ic_hi, :, :], w_tile,
                kernel_h=kh, stride=st, padding=pad,
                acc_mask=args.acc_mask, device=args.device)
            clean_sq = clean_out.squeeze(0)

            dump_sq = dump[0, 0]
            dump_sel = dump_sq[:, :, a['valid_cols']].transpose(2, 0, 1).astype(np.int32)
            obs = signed_int16(dump_sel - clean_sq)

            acc.add_batch(a['pseudo_chs'], clean_sq, obs)
            if csv_acc is not None:
                csv_acc.add_batch(a['pseudo_chs'], clean_sq, obs)

            n_done += 1
            if n_done % 500 == 0:
                print(f"  {n_done} atomic-sample pairs...")

    print(f"  Total: {n_done} pairs")

    results = acc.results(min_count=args.min_count, smoothing=args.smoothing)

    # Save full table (compressed, ~4MB for 200×200 bins)
    # For relative output, save to parent of first dump_dir (or dump_dir itself if single)
    if not os.path.isabs(args.output):
        base = dump_dirs[0].rstrip('/')
        if len(dump_dirs) > 1 and os.path.basename(base).startswith('run_'):
            base = os.path.dirname(base)
        output_path = os.path.join(base, args.output)
    else:
        output_path = args.output
    np.savez_compressed(output_path, **results)
    print(f"\nTable saved: {output_path} "
          f"({os.path.getsize(output_path)/1e6:.1f} MB)")

    if csv_acc is not None:
        write_ref_only_csv(
            csv_acc,
            args.csv_output,
            layout_json,
            layout_output=args.csv_layout_output,
            metadata_output=args.csv_metadata_output,
            min_count=args.csv_min_count,
        )

    # Summary
    print(f"\n{'='*70}")
    print(f"  Summary")
    print(f"{'='*70}")
    has_data = results['count'] > 0
    total_cells = n_pseudo * acc.n_ref_bins
    print(f"  Ref bins: {acc.n_ref_bins}, Noise bins: {acc.n_noise_bins}")
    print(f"  Coverage: {has_data.sum()}/{total_cells} ({100*has_data.sum()/total_cells:.1f}%)")
    print(f"  Total observations: {results['count'].sum():,}")
    print(f"  Median obs/cell (where >0): {np.median(results['count'][has_data]):.0f}")

    # Per-pseudo_ch summary
    print(f"\n  Per pseudo_ch noise stats (top 10 by |E| magnitude):")
    pch_E_mean = np.array([
        results['E'][p][has_data[p]].mean() if has_data[p].any() else 0.0
        for p in range(n_pseudo)
    ])
    top = np.argsort(np.abs(pch_E_mean))[-10:][::-1]
    for p in top:
        mask_p = has_data[p]
        if not mask_p.any():
            continue
        e = results['E'][p][mask_p]
        s = results['Std'][p][mask_p]
        cnt = results['count'][p][mask_p].sum()
        print(f"    pch {p:3d}: E_mean={e.mean():>+8.1f}  Std_mean={s.mean():>7.1f}  "
              f"n_bins={mask_p.sum():>3d}  total_obs={cnt:>10,}")


if __name__ == '__main__':
    main()
