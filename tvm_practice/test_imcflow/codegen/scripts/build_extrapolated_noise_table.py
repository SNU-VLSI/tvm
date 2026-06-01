#!/usr/bin/env python3
"""Build extrapolated noise table from an observed noise table.

Takes an existing aggregated noise table (NPZ) and fills uncovered ref bins
using trend-based extrapolation:

  1. For each pseudo_ch with sufficient observed bins (>= min_fit_bins):
     - Fit weighted linear regression: E(ref) = a*ref + b, Std(ref) = c*ref + d
     - For bins with count < reliable_count: replace with Gaussian N(E, Std^2)
       where E/Std come from the fit (or observed values for well-sampled bins)
     - Bins with count >= reliable_count: keep original observed distribution

  2. For channels with zero or insufficient coverage:
     - Use a global median E/Std profile (pooled from well-covered channels)

  3. Clamp Std to a minimum floor to prevent degenerate distributions.

Usage:
  python scripts/build_extrapolated_noise_table.py \\
      --input .../aggregated_noise_table__multi5.npz \\
      --output .../noise_table_extrapolated.npz

  # Bins with count < 200 get replaced with smooth Gaussian:
  python scripts/build_extrapolated_noise_table.py \\
      --input .../aggregated_noise_table__multi5.npz \\
      --output .../noise_table_extrapolated.npz \\
      --reliable-count 200
"""

import os, sys, argparse
import numpy as np
from scipy.stats import norm


def make_gaussian_probs(noise_centers, E, Std, std_floor=20.0):
    """Create a discretized Gaussian distribution on the noise bin grid.

    Uses CDF differences for accurate bin probabilities (no aliasing artifacts).
    """
    Std = max(Std, std_floor)
    K = len(noise_centers)
    bin_width = noise_centers[1] - noise_centers[0] if K > 1 else 1.0
    half_bw = bin_width / 2.0

    edges_lo = noise_centers - half_bw
    edges_hi = noise_centers + half_bw
    probs = norm.cdf(edges_hi, loc=E, scale=Std) - norm.cdf(edges_lo, loc=E, scale=Std)

    total = probs.sum()
    if total > 0:
        probs /= total
    else:
        idx = np.argmin(np.abs(noise_centers - E))
        probs[idx] = 1.0

    return probs


def extrapolate_noise_table(input_path, min_count=10, min_fit_bins=10,
                            reliable_count=200, std_floor=20.0, fit_degree=1):
    """Build extrapolated noise table from an observed one.

    Bins with count >= reliable_count keep their original distribution.
    Bins with count < reliable_count (including unobserved) get a smooth
    Gaussian with E/Std from the linear fit or global fallback.

    Returns dict with same keys as input NPZ but with filled distributions.
    """
    data = np.load(input_path)
    probs = data['probs'].copy()          # (C, R, K)
    ref_centers = data['ref_bin_centers']  # (R,)
    noise_centers = data['noise_bin_centers']  # (K,)
    E_orig = data['E'].copy()             # (C, R)
    Std_orig = data['Std'].copy()         # (C, R)
    count = data['count']                 # (C, R)

    C, R, K = probs.shape
    # For fitting: use bins with enough data to contribute to the trend
    fit_eligible = count >= min_count  # (C, R) bool
    # For keeping original distribution: need high confidence
    reliable = count >= reliable_count  # (C, R) bool

    E_extrap = E_orig.copy()
    Std_extrap = Std_orig.copy()
    probs_extrap = probs.copy()
    extrapolated_mask = np.zeros((C, R), dtype=bool)

    # Collect per-channel linear fits for E and Std
    fit_E = {}   # pch -> polynomial coefficients
    fit_Std = {}
    # Per-channel Std floor: prevent extrapolated Std from going below observed minimum
    pch_std_floor = np.full(C, std_floor)

    # Global fallback: pooled E/Std profile from well-covered channels
    global_E_sum = np.zeros(R, dtype=np.float64)
    global_Std_sum = np.zeros(R, dtype=np.float64)
    global_count = np.zeros(R, dtype=np.float64)

    n_fitted = 0
    n_fallback_global = 0

    for pch in range(C):
        valid = fit_eligible[pch]
        n_valid = valid.sum()

        if n_valid > 0:
            # Use weighted 25th percentile of observed Std as channel-specific floor
            pch_std_floor[pch] = max(np.percentile(Std_orig[pch][valid], 25), std_floor)

        if n_valid >= min_fit_bins:
            refs = ref_centers[valid]
            weights = np.sqrt(count[pch][valid].astype(np.float64))

            try:
                c_e = np.polyfit(refs, E_orig[pch][valid], fit_degree, w=weights)
                c_s = np.polyfit(refs, Std_orig[pch][valid], fit_degree, w=weights)
                fit_E[pch] = c_e
                fit_Std[pch] = c_s
                n_fitted += 1
            except (np.linalg.LinAlgError, ValueError):
                fit_E[pch] = None
                fit_Std[pch] = None
        else:
            fit_E[pch] = None
            fit_Std[pch] = None

        # Accumulate into global profile
        if n_valid > 0:
            global_E_sum[valid] += E_orig[pch][valid]
            global_Std_sum[valid] += Std_orig[pch][valid]
            global_count[valid] += 1.0

    # Global mean profile
    global_valid = global_count > 0
    global_E_profile = np.zeros(R)
    global_Std_profile = np.full(R, std_floor)
    global_E_profile[global_valid] = global_E_sum[global_valid] / global_count[global_valid]
    global_Std_profile[global_valid] = global_Std_sum[global_valid] / global_count[global_valid]

    # Fit global profile for extrapolation into completely empty ref bins
    gv = np.where(global_valid)[0]
    if len(gv) >= 2:
        gc_e = np.polyfit(ref_centers[gv], global_E_profile[gv], 1)
        gc_s = np.polyfit(ref_centers[gv], global_Std_profile[gv], 1)
        for r in range(R):
            if not global_valid[r]:
                global_E_profile[r] = np.polyval(gc_e, ref_centers[r])
                global_Std_profile[r] = max(np.polyval(gc_s, ref_centers[r]), std_floor)

    print(f"Channels with fit: {n_fitted}/{C}")
    print(f"Global profile: {global_valid.sum()}/{R} ref bins covered")
    print(f"Reliable bins (count >= {reliable_count}): {reliable.sum()}/{C*R} "
          f"({100*reliable.sum()/(C*R):.1f}%)")

    # Fill each channel
    for pch in range(C):
        fe = fit_eligible[pch]
        n_fe = fe.sum()
        vi = np.where(fe)[0]

        if n_fe == 0:
            # Completely empty channel — use global profile + Gaussian
            n_fallback_global += 1
            E_extrap[pch] = global_E_profile
            Std_extrap[pch] = np.maximum(global_Std_profile, std_floor)
            extrapolated_mask[pch] = True

            for r in range(R):
                probs_extrap[pch, r] = make_gaussian_probs(
                    noise_centers, E_extrap[pch, r], Std_extrap[pch, r],
                    std_floor=std_floor)
            continue

        # Channel with some coverage
        c_e = fit_E.get(pch)
        c_s = fit_Std.get(pch)

        for r in range(R):
            if reliable[pch, r]:
                continue  # High-confidence observed bin — keep original

            extrapolated_mask[pch, r] = True

            # Determine E and Std for this bin
            sf = pch_std_floor[pch]  # channel-specific Std floor
            if c_e is not None:
                E_extrap[pch, r] = np.polyval(c_e, ref_centers[r])
                Std_extrap[pch, r] = max(np.polyval(c_s, ref_centers[r]), sf)
            elif fe[r]:
                # Has some data but not enough for fit — use observed E/Std
                E_extrap[pch, r] = E_orig[pch, r]
                Std_extrap[pch, r] = max(Std_orig[pch, r], sf)
            else:
                # No data at all — use nearest observed
                nearest = vi[np.argmin(np.abs(ref_centers[vi] - ref_centers[r]))]
                E_extrap[pch, r] = E_orig[pch, nearest]
                Std_extrap[pch, r] = max(Std_orig[pch, nearest], sf)

            # Replace with smooth Gaussian
            probs_extrap[pch, r] = make_gaussian_probs(
                noise_centers, E_extrap[pch, r], Std_extrap[pch, r],
                std_floor=sf)

    print(f"Channels using global fallback: {n_fallback_global}")
    print(f"Extrapolated/smoothed bins: {extrapolated_mask.sum()}/{C*R} "
          f"({100*extrapolated_mask.sum()/(C*R):.1f}%)")

    # Recompute E, Std from distributions for consistency
    E_final = np.zeros((C, R), dtype=np.float64)
    Std_final = np.zeros((C, R), dtype=np.float64)
    for pch in range(C):
        for r in range(R):
            p = probs_extrap[pch, r]
            e = (p * noise_centers).sum()
            e2 = (p * noise_centers**2).sum()
            E_final[pch, r] = e
            Std_final[pch, r] = max(np.sqrt(max(e2 - e**2, 0.0)), 0.0)

    # For reliable bins, keep original E/Std (more accurate than re-binned)
    E_final[reliable] = E_orig[reliable]
    Std_final[reliable] = Std_orig[reliable]

    # Build count array: synthetic count for non-reliable bins
    count_out = count.copy()
    count_out[extrapolated_mask & ~reliable] = min_count

    return {
        'probs': probs_extrap,
        'ref_bin_edges': data['ref_bin_edges'],
        'ref_bin_centers': ref_centers,
        'noise_bin_edges': data['noise_bin_edges'],
        'noise_bin_centers': noise_centers,
        'E': E_final,
        'Var': Std_final**2,
        'Std': Std_final,
        'count': count_out,
        'extrapolated_mask': extrapolated_mask,
        'original_count': count,
    }


def main():
    parser = argparse.ArgumentParser(description='Build extrapolated noise table')
    parser.add_argument('--input', required=True, help='Input NPZ noise table')
    parser.add_argument('--output', required=True, help='Output NPZ path')
    parser.add_argument('--min-count', type=int, default=10,
                        help='Minimum observation count to include bin in linear fit')
    parser.add_argument('--min-fit-bins', type=int, default=10,
                        help='Minimum observed bins per channel for linear fit')
    parser.add_argument('--reliable-count', type=int, default=200,
                        help='Minimum count to keep original distribution (below: replace with Gaussian)')
    parser.add_argument('--std-floor', type=float, default=20.0,
                        help='Minimum Std to prevent degenerate distributions')
    parser.add_argument('--fit-degree', type=int, default=1,
                        help='Polynomial degree for E/Std fit (default: 1=linear)')
    args = parser.parse_args()

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Reliable count threshold: {args.reliable_count}")

    results = extrapolate_noise_table(
        args.input,
        min_count=args.min_count,
        min_fit_bins=args.min_fit_bins,
        reliable_count=args.reliable_count,
        std_floor=args.std_floor,
        fit_degree=args.fit_degree,
    )

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    np.savez_compressed(args.output, **results)
    sz = os.path.getsize(args.output) / 1e6
    print(f"\nSaved: {args.output} ({sz:.1f} MB)")

    # Summary
    C = results['probs'].shape[0]
    R = results['probs'].shape[1]
    orig_count = results['original_count']
    extrap = results['extrapolated_mask']
    obs = orig_count >= args.min_count
    rel = orig_count >= args.reliable_count

    print(f"\n{'='*60}")
    print(f"  Extrapolated Noise Table Summary")
    print(f"{'='*60}")
    print(f"  Reliable bins (kept original): {rel.sum()}/{C*R} ({100*rel.sum()/(C*R):.1f}%)")
    print(f"  Low-count observed (smoothed): {(obs & ~rel).sum()}/{C*R} ({100*(obs & ~rel).sum()/(C*R):.1f}%)")
    print(f"  Unobserved (extrapolated):     {(~obs).sum()}/{C*R} ({100*(~obs).sum()/(C*R):.1f}%)")
    print(f"  Total Gaussian-replaced:       {extrap.sum()}/{C*R} ({100*extrap.sum()/(C*R):.1f}%)")

    E_rel_mean = np.nanmean(np.where(rel, results['E'], np.nan))
    E_ext_mean = np.nanmean(np.where(extrap, results['E'], np.nan))
    print(f"  Mean E (reliable bins):     {E_rel_mean:+.1f}")
    print(f"  Mean E (Gaussian bins):     {E_ext_mean:+.1f}")


if __name__ == '__main__':
    main()
