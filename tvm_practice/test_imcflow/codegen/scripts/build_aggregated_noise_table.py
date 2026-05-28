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
      --dump-dir debugging/fpga/uqat_tmp02_refine_ndis32_0_78 \\
      --samples 0-199 \\
      --output aggregated_noise_table.npz
"""

import os, sys, argparse
import numpy as np
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

NOISE_DIR = '/root/project/CIM/noise/noise_df/B2_out/N32'
LAYOUT_JSON = os.path.join(NOISE_DIR, 'concat_per_core.json')
NPZ_PATH = os.path.join(CODEGEN, 'eval_dir/resnet8_subset31_pretrained_orig_evl.linux/psum_imcu_column_map.npz')


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


def main():
    parser = argparse.ArgumentParser(description='Build aggregated noise table')
    parser.add_argument('--dump-dir', required=True)
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
        ckpt_path, _ = ckpt_resolve('B2', 'half', alias)
    print(f"Checkpoint: {ckpt_path}")

    print("Loading metadata...")
    atomics = load_atomic_info(NPZ_PATH)
    pseudo_map, n_per_core, n_pseudo = load_pseudo_ch_map(LAYOUT_JSON)
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
        for s_idx in sample_range[:20]:
            sample_dir = os.path.join(dump_dir, f'sample_{s_idx}')
            if not os.path.isdir(sample_dir):
                continue
            qconv_to_input = build_qconv_to_input_map(sample_dir)
            for a in atomics[:6]:
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
                qin = np.load(os.path.join(sample_dir, input_fname))
                dump = np.load(out_npy)
                ic_lo = a['ic_id'] * a['ic_block']
                ic_hi = ic_lo + a['ic_size']
                w_full = weights[orig]
                oc_lo = a['oc_id'] * a['oc_block']
                oc_hi = oc_lo + a['oc_size']
                w_tile = w_full[oc_lo:oc_hi, ic_lo:ic_hi, :, :]
                clean_out, _, _ = noise_free_qconv(
                    qin[:, ic_lo:ic_hi, :, :], w_tile,
                    kernel_h=kh, stride=st, padding=pad, device=args.device)
                clean_sq = clean_out.squeeze(0)
                dump_sq = dump[0, 0]
                dump_sel = dump_sq[:, :, a['valid_cols']].transpose(2, 0, 1).astype(np.int32)
                obs = signed_int16(dump_sel - clean_sq)
                ref_samples.append(clean_sq.ravel())
                noise_samples.append(obs.ravel())

        all_ref = np.concatenate(ref_samples)
        all_noise = np.concatenate(noise_samples)

    if args.ref_range is None:
        r1, r99 = np.percentile(all_ref, [0.2, 99.8])
        margin = (r99 - r1) * 0.05
        ref_lo, ref_hi = r1 - margin, r99 + margin
    else:
        ref_lo, ref_hi = args.ref_range

    if args.noise_range is None:
        n99 = np.percentile(np.abs(all_noise), 99.5)
        noise_lo, noise_hi = -n99, n99
    else:
        noise_lo, noise_hi = args.noise_range

    print(f"  Ref range: [{ref_lo:.0f}, {ref_hi:.0f}] ({args.n_ref_bins} bins)")
    print(f"  Noise range: [{noise_lo:.0f}, {noise_hi:.0f}] ({args.n_noise_bins} bins)")

    ref_bin_edges = np.linspace(ref_lo, ref_hi, args.n_ref_bins + 1)
    noise_bin_edges = np.linspace(noise_lo, noise_hi, args.n_noise_bins + 1)
    acc = AggregatedNoiseAccumulator(n_pseudo, ref_bin_edges, noise_bin_edges)

    # Main pass
    print(f"\nProcessing {len(sample_range)} samples × {len(atomics)} atomics...")
    n_done = 0
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

            qin = np.load(os.path.join(sample_dir, input_fname))
            dump = np.load(out_npy)

            ic_lo = a['ic_id'] * a['ic_block']
            ic_hi = ic_lo + a['ic_size']
            w_full = weights[orig]
            oc_lo = a['oc_id'] * a['oc_block']
            oc_hi = oc_lo + a['oc_size']
            w_tile = w_full[oc_lo:oc_hi, ic_lo:ic_hi, :, :]

            clean_out, _, _ = noise_free_qconv(
                qin[:, ic_lo:ic_hi, :, :], w_tile,
                kernel_h=kh, stride=st, padding=pad, device=args.device)
            clean_sq = clean_out.squeeze(0)

            dump_sq = dump[0, 0]
            dump_sel = dump_sq[:, :, a['valid_cols']].transpose(2, 0, 1).astype(np.int32)
            obs = signed_int16(dump_sel - clean_sq)

            acc.add_batch(a['pseudo_chs'], clean_sq, obs)

            n_done += 1
            if n_done % 500 == 0:
                print(f"  {n_done} atomic-sample pairs...")

    print(f"  Total: {n_done} pairs")

    results = acc.results(min_count=args.min_count, smoothing=args.smoothing)

    # Save
    output_path = (os.path.join(dump_dir, args.output)
                   if not os.path.isabs(args.output) else args.output)
    np.savez(output_path, **results)
    print(f"\nAggregated noise table saved: {output_path}")

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
