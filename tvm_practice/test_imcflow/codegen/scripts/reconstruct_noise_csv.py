#!/usr/bin/env python3
"""Reconstruct noise CSV from chip dump observations.

Uses actual ResNet inference chip dumps to build per-bitplane noise distribution:
  P(diff | pseudo_ch, wpattern, ref)

Approach: iterative residual attribution
  Round 0: direct — attribute total int16 obs noise to each bitplane by dividing
           by the bitplane scale. Cross-plane noise averages out over many samples.
  Round k>0: subtract predicted noise from other 15 planes (using round k-1 CSV),
             attribute residual to target plane.

Usage:
  python scripts/reconstruct_noise_csv.py \\
      --dump-dir debugging/fpga/uqat_tmp02_refine_ndis32_0_78 \\
      --samples 0-199 \\
      --output-csv reconstructed_noise.csv \\
      [--iterations 2]
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import defaultdict

CODEGEN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CIM_DIR = '/root/project/CIM'
sys.path.insert(0, CIM_DIR)
sys.path.insert(0, os.path.join(CODEGEN, 'scripts'))

from diagnose_noise_per_qconv import (
    signed_int16, load_atomic_info, load_pseudo_ch_map,
    load_noise_csv, build_qconv_to_input_map, noise_free_qconv,
    compute_predicted_stats,
    PSTEP, NUM_LEVELS, WBITS, ABITS, W_SCALE, CONV_PARAMS,
)

NOISE_DIR = '/root/project/CIM/noise/noise_df/B2_out/N32'
NOISE_CSV = os.path.join(NOISE_DIR, 'B2_noise_matrix_per_ch_concat__alpha0.5_w5_T2.0.csv')
LAYOUT_JSON = os.path.join(NOISE_DIR, 'concat_per_core.json')
NPZ_PATH = os.path.join(CODEGEN, 'eval_dir/resnet8_subset31_pretrained_orig_evl.linux/psum_imcu_column_map.npz')

WPATTERN_STRINGS = ['0001', '0010', '0100', '1000']


def parse_sample_range(s):
    result = []
    for part in s.split(','):
        if '-' in part:
            lo, hi = part.split('-')
            result.extend(range(int(lo), int(hi) + 1))
        else:
            result.append(int(part))
    return result


def load_weights(ckpt_path):
    """Load weights keyed by orig_conv, same as diagnose script."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict']
    weights = {}
    for orig, (kh, st, pad, key) in CONV_PARAMS.items():
        w = sd[key].cpu().numpy().astype(np.int32)
        weights[orig] = w
    return weights


def predict_per_plane(adc_codes, skip_mask, pseudo_chs, csv_data):
    """Vectorized per-(abit,wbit) predicted E[noise] at int16 level.

    Returns: (ABITS, WBITS, OC, OH, OW) float64 — predicted noise contribution
    """
    A, W, OC, OH, OW = adc_codes.shape
    n_refs = csv_data['n_refs']
    E = csv_data['E']  # (C, R)

    wbit_axis = np.arange(W).reshape(1, W, 1, 1, 1)
    rows = wbit_axis * n_refs + adc_codes.astype(np.int64)

    ch_idx = pseudo_chs.reshape(1, 1, OC, 1, 1).astype(np.int64)
    ch_b = np.broadcast_to(ch_idx, (A, W, OC, OH, OW))
    rows_b = np.broadcast_to(rows, (A, W, OC, OH, OW))
    E_lk = E[ch_b, rows_b]  # (A, W, OC, OH, OW)

    # Apply skip mask
    skip_b = np.broadcast_to(skip_mask[:, None, None, :, :], (A, W, OC, OH, OW))
    E_lk = np.where(skip_b, 0.0, E_lk)

    # Scale to int16 level
    abit_arr = np.arange(A).reshape(A, 1, 1, 1, 1)
    w_scale = np.array(W_SCALE, dtype=np.float64).reshape(1, W, 1, 1, 1)
    scale = PSTEP * (1 << abit_arr).astype(np.float64) * w_scale  # (A, W, 1, 1, 1)

    return E_lk * scale  # (A, W, OC, OH, OW)


class HistogramAccumulator:
    """Accumulates histograms for (pseudo_ch, wbit*n_refs+ref) in fixed bins."""

    def __init__(self, n_pseudo, n_rows, bin_lo, bin_hi):
        self.bin_lo = bin_lo
        self.bin_hi = bin_hi
        self.n_bins = bin_hi - bin_lo + 1
        self.hist = np.zeros((n_pseudo, n_rows, self.n_bins), dtype=np.float64)
        self.count = np.zeros((n_pseudo, n_rows), dtype=np.int64)
        # Also accumulate raw sum/sum2 for E/Var computation without binning artifacts
        self.raw_sum = np.zeros((n_pseudo, n_rows), dtype=np.float64)
        self.raw_sum2 = np.zeros((n_pseudo, n_rows), dtype=np.float64)

    def add_batch(self, pseudo_chs, wbit, codes, diffs, skip_mask_a):
        """Add a batch of observations.

        Args:
            pseudo_chs: (OC,) int array
            wbit: int
            codes: (OC, OH, OW) int16 — ADC codes (= ref index)
            diffs: (OC, OH, OW) float64 — normalized noise values
            skip_mask_a: (OH, OW) bool — True where skipped
        """
        OC, OH, OW = codes.shape
        n_refs = NUM_LEVELS
        valid = ~skip_mask_a  # (OH, OW)

        for oc_i in range(OC):
            pch = pseudo_chs[oc_i]
            c = codes[oc_i][valid]   # flat
            d = diffs[oc_i][valid]   # flat

            if c.size == 0:
                continue

            rows = wbit * n_refs + c.astype(np.int64)
            d_rounded = np.clip(np.round(d).astype(int) - self.bin_lo, 0, self.n_bins - 1)

            # Use np.add.at for scatter-add into histograms
            # We need joint (row, bin) indexing
            np.add.at(self.hist[pch], (rows, d_rounded), 1.0)
            np.add.at(self.count[pch], (rows,), 1)
            np.add.at(self.raw_sum[pch], (rows,), d)
            np.add.at(self.raw_sum2[pch], (rows,), d * d)


def collect_and_accumulate(dump_dir, sample_range, atomics, weights,
                           csv_data, acc, iteration=0, prev_csv_data=None,
                           device='cpu'):
    """Process all chip dumps, accumulating per-bitplane noise into histograms."""
    n_done = 0
    n_skipped = 0

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
                n_skipped += 1
                continue
            in_path = os.path.join(sample_dir, input_fname)
            out_npy = None
            for f in os.listdir(sample_dir):
                if f.endswith(f'_{a["func"]}.npy'):
                    out_npy = os.path.join(sample_dir, f)
                    break
            if out_npy is None:
                n_skipped += 1
                continue

            qin = np.load(in_path)
            dump = np.load(out_npy)

            ic_lo = a['ic_id'] * a['ic_block']
            ic_hi = ic_lo + a['ic_size']
            input_slice = qin[:, ic_lo:ic_hi, :, :]
            w_full = weights[orig]
            oc_lo = a['oc_id'] * a['oc_block']
            oc_hi = oc_lo + a['oc_size']
            w_tile = w_full[oc_lo:oc_hi, ic_lo:ic_hi, :, :]

            clean_out, adc_codes, skip_mask = noise_free_qconv(
                input_slice, w_tile, kernel_h=kh, stride=st, padding=pad,
                device=device,
            )
            clean_sq = clean_out.squeeze(0)

            dump_sq = dump[0, 0]
            dump_sel = dump_sq[:, :, a['valid_cols']].transpose(2, 0, 1).astype(np.int32)
            obs_total = signed_int16(dump_sel - clean_sq).astype(np.float64)

            OC, OH, OW = obs_total.shape
            pseudo_chs = a['pseudo_chs']

            # For iteration > 0: compute per-plane predicted noise from prev CSV
            if iteration > 0 and prev_csv_data is not None:
                pred_per_plane = predict_per_plane(
                    adc_codes, skip_mask, pseudo_chs, prev_csv_data
                )  # (A, W, OC, OH, OW)
            else:
                pred_per_plane = None

            for abit in range(ABITS):
                skip_a = skip_mask[abit]
                for wbit in range(WBITS):
                    scale = PSTEP * W_SCALE[wbit] * (1 << abit)
                    if scale == 0:
                        continue
                    codes = adc_codes[abit, wbit]  # (OC, OH, OW)

                    if pred_per_plane is not None:
                        # Subtract other 15 planes' predicted contribution
                        other_sum = pred_per_plane.sum(axis=(0, 1)) - pred_per_plane[abit, wbit]
                        residual = obs_total - other_sum
                        diff = residual / scale
                    else:
                        diff = obs_total / scale

                    acc.add_batch(pseudo_chs, wbit, codes, diff, skip_a)

            n_done += 1
            if n_done % 200 == 0:
                print(f'  [iter {iteration}] {n_done} atomic-sample pairs...')

    print(f'  [iter {iteration}] done: {n_done} pairs processed, {n_skipped} skipped')


def build_csv_and_data(acc, n_pseudo, smoothing_alpha=0.5, min_count=5):
    """Convert accumulated histograms into CSV DataFrame and csv_data dict."""
    n_refs = NUM_LEVELS
    n_rows = WBITS * n_refs
    n_bins = acc.n_bins
    diff_bins = np.arange(acc.bin_lo, acc.bin_hi + 1, dtype=float)

    probs = np.zeros((n_pseudo, n_rows, n_bins), dtype=np.float64)
    zero_idx = -acc.bin_lo  # index of diff=0

    for pch in range(n_pseudo):
        for row in range(n_rows):
            h = acc.hist[pch, row]
            total = h.sum()
            if total < min_count:
                probs[pch, row, zero_idx] = 1.0
            else:
                if smoothing_alpha > 0:
                    h = h + smoothing_alpha
                probs[pch, row] = h / h.sum()

    # Coverage stats
    has_data = acc.count > 0
    total_cells = n_pseudo * n_rows
    covered = has_data.sum()
    median_count = np.median(acc.count[has_data]) if has_data.any() else 0
    print(f"  Coverage: {covered}/{total_cells} ({100*covered/total_cells:.1f}%)")
    print(f"  Median obs/cell (where >0): {median_count:.0f}")

    # Compute E and Var from raw sums (more accurate than from binned histograms)
    E_raw = np.zeros((n_pseudo, n_rows))
    Var_raw = np.zeros((n_pseudo, n_rows))
    valid = acc.count > 0
    E_raw[valid] = acc.raw_sum[valid] / acc.count[valid]
    Var_raw[valid] = acc.raw_sum2[valid] / acc.count[valid] - E_raw[valid] ** 2
    Var_raw = np.maximum(Var_raw, 0.0)

    # Also compute E/Var from histogram (for CSV compatibility)
    E_hist = np.sum(probs * diff_bins[None, None, :], axis=2)
    Var_hist = np.sum(probs * diff_bins[None, None, :] ** 2, axis=2) - E_hist ** 2
    Var_hist = np.maximum(Var_hist, 0.0)

    # Build DataFrame
    col_tuples = []
    for pch in range(n_pseudo):
        for db in diff_bins:
            col_tuples.append((str(db), str(pch)))
    columns = pd.MultiIndex.from_tuples(col_tuples)

    row_labels = []
    for wbit in range(WBITS):
        wp_str = WPATTERN_STRINGS[wbit]
        for ref in range(n_refs):
            ref_val = ref * PSTEP
            row_labels.append(f'{wp_str}_{ref_val}')

    data = np.zeros((n_rows, n_pseudo * n_bins))
    for pch in range(n_pseudo):
        for row in range(n_rows):
            data[row, pch * n_bins: (pch + 1) * n_bins] = probs[pch, row]

    df = pd.DataFrame(data, index=row_labels, columns=columns)
    df.index.name = 'wpattern_ref'

    # diff_min/diff_max per (C, R)
    diff_min = np.full((n_pseudo, n_rows), 0.0)
    diff_max = np.full((n_pseudo, n_rows), 0.0)
    mode_noise = np.full((n_pseudo, n_rows), 0.0)
    for pch in range(n_pseudo):
        for row in range(n_rows):
            p = probs[pch, row]
            nz = np.nonzero(p > 1e-12)[0]
            if len(nz) > 0:
                diff_min[pch, row] = diff_bins[nz[0]]
                diff_max[pch, row] = diff_bins[nz[-1]]
                mode_noise[pch, row] = diff_bins[nz[np.argmax(p[nz])]]

    csv_data = {
        'probs': probs,
        'diff_bins': diff_bins,
        'n_refs': n_refs,
        'n_wpatterns': WBITS,
        'refs': np.arange(n_refs) * PSTEP,
        'wpatterns_order': WPATTERN_STRINGS,
        'E': E_raw,       # use raw (non-binned) E for accuracy
        'Var': Var_raw,
        'E_hist': E_hist,  # histogram-based E for comparison
        'diff_min': diff_min,
        'diff_max': diff_max,
        'mode_noise': mode_noise,
    }

    return df, csv_data


def main():
    parser = argparse.ArgumentParser(description='Reconstruct noise CSV from chip dumps')
    parser.add_argument('--dump-dir', required=True)
    parser.add_argument('--samples', default='0-199')
    parser.add_argument('--output-csv', default='reconstructed_noise.csv')
    parser.add_argument('--iterations', type=int, default=0,
                        help='0=direct only, >0=iterative refinement')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--smoothing', type=float, default=0.5)
    parser.add_argument('--min-count', type=int, default=5)
    parser.add_argument('--bin-range', type=int, nargs=2, default=[-25, 15],
                        help='Diff bin range (default: -25 15, similar to original CSV)')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    dump_dir = (os.path.join(CODEGEN, args.dump_dir)
                if not os.path.isabs(args.dump_dir) else args.dump_dir)
    sample_range = parse_sample_range(args.samples)
    print(f"Dump dir: {dump_dir}")
    print(f"Samples: {sample_range[0]}..{sample_range[-1]} ({len(sample_range)})")

    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        alias = os.path.basename(dump_dir.rstrip('/'))
        from checkpoints import resolve as ckpt_resolve
        ckpt_path, alias = ckpt_resolve('B2', 'half', alias)
    print(f"Checkpoint: {ckpt_path}")

    print("Loading metadata...")
    atomics = load_atomic_info(NPZ_PATH)
    pseudo_map, n_per_core, n_pseudo = load_pseudo_ch_map(LAYOUT_JSON)
    orig_csv_data = load_noise_csv(NOISE_CSV)
    weights = load_weights(ckpt_path)

    for a in atomics:
        h = a['imce_h']
        w = a['imce_w_1based']
        a['pseudo_chs'] = np.array(
            [pseudo_map[(h, w, int(c))] for c in a['valid_cols']],
            dtype=np.int64,
        )

    print(f"  Atomics: {len(atomics)}, pseudo_chs: {n_pseudo}")
    print(f"  Original CSV diff_bins: {orig_csv_data['diff_bins']}")
    print(f"  Reconstruction bin range: [{args.bin_range[0]}, {args.bin_range[1]}]")

    # Iteration strategy:
    #   iter 0: use ORIGINAL CSV to predict other 15 planes, attribute residual
    #           to target plane. This gives clean single-pass reconstruction.
    #   iter k>0: use iter (k-1) reconstructed CSV for the 15-plane subtraction.
    #   "direct" mode (--iterations -1): skip residual, just divide obs by scale.
    current_csv_data = orig_csv_data  # start from original CSV
    n_iters = max(args.iterations, 0) + 1
    for iteration in range(n_iters):
        use_residual = args.iterations >= 0  # -1 = direct mode
        label = 'residual' if use_residual else 'direct'
        print(f"\n{'='*70}")
        print(f"  Iteration {iteration} ({label})")
        print(f"{'='*70}")

        acc = HistogramAccumulator(
            n_pseudo, WBITS * NUM_LEVELS,
            bin_lo=args.bin_range[0], bin_hi=args.bin_range[1],
        )

        collect_and_accumulate(
            dump_dir, sample_range, atomics, weights,
            csv_data=orig_csv_data,
            acc=acc,
            iteration=1 if use_residual else 0,  # 1 = use prev_csv_data for subtraction
            prev_csv_data=current_csv_data if use_residual else None,
            device=args.device,
        )

        df, current_csv_data = build_csv_and_data(
            acc, n_pseudo,
            smoothing_alpha=args.smoothing,
            min_count=args.min_count,
        )

        if iteration < n_iters - 1:
            iter_path = args.output_csv.replace('.csv', f'_iter{iteration}.csv')
            df.to_csv(iter_path)
            print(f"  Saved intermediate: {iter_path}")

    df.to_csv(args.output_csv)
    print(f"\nFinal CSV saved: {args.output_csv}")

    counts_path = args.output_csv.replace('.csv', '_counts.npy')
    np.save(counts_path, acc.count)
    print(f"Counts saved: {counts_path}")

    # Also save csv_data as npz for easy loading
    npz_path = args.output_csv.replace('.csv', '_data.npz')
    np.savez(npz_path,
             probs=current_csv_data['probs'],
             diff_bins=current_csv_data['diff_bins'],
             E=current_csv_data['E'],
             Var=current_csv_data['Var'],
             counts=acc.count)
    print(f"Data npz saved: {npz_path}")

    # Comparison with original
    print(f"\n{'='*70}")
    print(f"  Comparison: original vs reconstructed")
    print(f"{'='*70}")
    orig_E = orig_csv_data['E']
    new_E = current_csv_data['E']
    mask = acc.count >= args.min_count
    if mask.any():
        diff = new_E[mask] - orig_E[mask]
        print(f"  E[noise] diff (where count >= {args.min_count}):")
        print(f"    mean: {diff.mean():.4f}, std: {diff.std():.4f}")
        print(f"    |diff| mean: {np.abs(diff).mean():.4f}")
        corr = np.corrcoef(orig_E[mask], new_E[mask])[0, 1]
        print(f"    correlation: {corr:.4f}")

    # Per-pseudo_ch comparison
    print(f"\n  Per pseudo_ch E[noise] comparison (top 10 biggest shifts):")
    pch_orig_mean = np.array([orig_E[p, mask[p]].mean() if mask[p].any() else 0
                              for p in range(n_pseudo)])
    pch_new_mean = np.array([new_E[p, mask[p]].mean() if mask[p].any() else 0
                             for p in range(n_pseudo)])
    pch_shift = pch_new_mean - pch_orig_mean
    top_shift = np.argsort(np.abs(pch_shift))[-10:][::-1]
    for p in top_shift:
        n_obs = acc.count[p].sum()
        print(f"    pch {p:3d}: orig={pch_orig_mean[p]:+.3f}  new={pch_new_mean[p]:+.3f}  "
              f"shift={pch_shift[p]:+.3f}  total_obs={n_obs}")


if __name__ == '__main__':
    main()
