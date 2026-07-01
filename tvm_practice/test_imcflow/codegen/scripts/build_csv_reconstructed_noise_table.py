#!/usr/bin/env python3
"""Build a CSV-reconstructed noise table from chip debug dumps.

Uses the same dump-loading pipeline as build_aggregated_noise_table.py, but
instead of using observed chip noise (chip_output - clean_ref), it:
  1. Runs noise_free_qconv to get per-bitplane ADC codes (adc_codes_all)
  2. Samples noise from the CSV distribution. Legacy wpattern_ref CSVs are
     sampled per (abit, wbit, pseudo_ch, adc_code). Ref-only CSVs can be
     sampled either per input bitplane or once at the output clean_ref.
  3. Accumulates sampled noise when using a bitplane mode.
  4. Bins (clean_ref, reconstructed_noise) into the same histogram format

This allows direct comparison: does the per-bitplane CSV noise model reproduce
the observed accumulated noise?

Usage:
  python scripts/build_csv_reconstructed_noise_table.py \\
      --dump-dir debugging/fpga/uqat_cycle4b_repro/run_00 \\
                  debugging/fpga/uqat_cycle4b_repro/run_01 \\
      --samples 0-199 \\
      --n-mc-trials 50 \\
      --output csv_reconstructed_noise_table.npz

  # Use the same ref/noise range as an existing NPZ for direct comparison:
  python scripts/build_csv_reconstructed_noise_table.py \\
      --dump-dir debugging/fpga/uqat_cycle4b_repro/run_0{0..4} \\
      --samples 0-199 \\
      --match-npz /path/to/aggregated_noise_table__multi5.npz \\
      --output csv_reconstructed.npz
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import torch

CODEGEN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CIM_DIR = '/root/project/CIM'
sys.path.insert(0, CIM_DIR)
sys.path.insert(0, os.path.join(CODEGEN, 'scripts'))

from diagnose_noise_per_qconv import (
    signed_int16, load_atomic_info, load_pseudo_ch_map,
    noise_free_qconv, load_noise_csv,
    PSTEP, NUM_LEVELS, WBITS, ABITS, W_SCALE, CONV_PARAMS,
    resolve_model_profile, get_conv_params, get_default_npz_path,
)
from build_aggregated_noise_table import (
    AggregatedNoiseAccumulator, load_weights, parse_sample_range,
    iter_existing_sample_dirs, iter_func_dump_candidates,
    iter_prior_quant_candidates,
)

DEFAULT_NOISE_DIR = '/root/project/CIM/noise/noise_df/B2_out/N32'
DEFAULT_LAYOUT_JSON = os.path.join(DEFAULT_NOISE_DIR, 'concat_per_core.json')
DEFAULT_NPZ_PATH = os.path.join(CODEGEN, 'eval_dir/resnet8_subset31_pretrained_orig_evl.linux/psum_imcu_column_map.npz')
DEFAULT_CSV = os.path.join(CIM_DIR,
    'noise/noise_df/B2_out_refine_fixed_full_v1_partial/N32/B2_noise_matrix_per_ch_concat.csv')


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


def load_matching_reconstruction_inputs(sample_dir, atomic, weights, conv_params, acc_mask, device):
    """Return clean qconv output and ADC-code tensors for a compatible dump pair.

    Preserved debug dump directories can contain stale nodes from previous
    compiles.  Match each qconv output candidate with the nearest prior
    min_max_quantize input candidate and only accept the pair when the input
    channel range is compatible and the recomputed clean output shape matches
    the selected dump shape.
    """
    orig = atomic["orig_conv"]
    kh, st, pad, _ = conv_params[orig]
    ic_lo = atomic["ic_id"] * atomic["ic_block"]
    ic_hi = ic_lo + atomic["ic_size"]
    w_full = weights[orig]
    oc_lo = atomic["oc_id"] * atomic["oc_block"]
    oc_hi = oc_lo + atomic["oc_size"]
    w_tile = w_full[oc_lo:oc_hi, ic_lo:ic_hi, :, :]
    notes = []

    dump_candidates, dump_notes = iter_func_dump_candidates(
        sample_dir, atomic["func"], atomic["valid_cols"])
    notes.extend(dump_notes)
    for out_node_id, out_fname, dump_sel in dump_candidates:
        for _, input_fname in iter_prior_quant_candidates(sample_dir, out_node_id):
            try:
                qin = np.load(os.path.join(sample_dir, input_fname), allow_pickle=True)
            except Exception as exc:
                notes.append(f"{input_fname}: input load failed: {exc}")
                continue
            if qin.ndim != 4 or qin.shape[1] < ic_hi:
                notes.append(
                    f"{input_fname}: input shape {tuple(qin.shape)} incompatible "
                    f"with channels [{ic_lo}, {ic_hi})"
                )
                continue
            try:
                clean_out, adc_codes_all, skip_mask_all = noise_free_qconv(
                    qin[:, ic_lo:ic_hi, :, :], w_tile,
                    kernel_h=kh, stride=st, padding=pad,
                    acc_mask=acc_mask, device=device)
            except Exception as exc:
                notes.append(f"{input_fname} -> {out_fname}: clean qconv failed: {exc}")
                continue
            clean_sq = clean_out.squeeze(0)
            if tuple(clean_sq.shape) == tuple(dump_sel.shape):
                return clean_sq, dump_sel, adc_codes_all, skip_mask_all, notes
            notes.append(
                f"{input_fname} -> {out_fname}: clean={tuple(clean_sq.shape)} "
                f"dump={tuple(dump_sel.shape)}"
            )

    return None, None, None, None, notes


def _detect_csv_format(csv_path):
    first = str(pd.read_csv(csv_path, header=[0, 1], index_col=0, nrows=1).index[0])
    if '_' in first:
        return 'wpattern_ref'
    try:
        float(first)
    except ValueError as exc:
        raise ValueError(
            f"Cannot detect noise CSV format from first row index {first!r}"
        ) from exc
    return 'ref'


def _parse_noise_columns(raw, label):
    diff_bins_lv = raw.columns.get_level_values(0).astype(float).to_numpy()
    channels_lv = raw.columns.get_level_values(1).astype(int).to_numpy()
    uniq_channels = sorted(set(channels_lv.tolist()))
    if uniq_channels != list(range(len(uniq_channels))):
        raise ValueError(f"{label} CSV channels must be contiguous 0..C-1; got {uniq_channels}")
    mask0 = channels_lv == 0
    diff_bins = diff_bins_lv[mask0]
    for ch in uniq_channels[1:]:
        m = channels_lv == ch
        if not (diff_bins_lv[m] == diff_bins).all():
            raise ValueError(f"{label} channel {ch} has different diff_bin order from channel 0")
    return diff_bins, channels_lv, uniq_channels


def load_ref_noise_csv(csv_path):
    raw = pd.read_csv(csv_path, header=[0, 1], index_col=0)
    refs = np.asarray([float(idx) for idx in raw.index], dtype=np.float64)
    order = np.argsort(refs)
    raw = raw.iloc[order]
    refs = refs[order]

    diff_bins, channels_lv, uniq_channels = _parse_noise_columns(raw, 'ref')
    C = len(uniq_channels)
    R = raw.shape[0]
    K = len(diff_bins)
    probs = np.zeros((C, R, K), dtype=np.float64)
    raw_np = raw.to_numpy(dtype=np.float64)
    for ch in uniq_channels:
        probs[ch] = raw_np[:, channels_lv == ch]

    row_sum = probs.sum(axis=-1, keepdims=True)
    if np.any(row_sum <= 0):
        bad = np.argwhere(row_sum.squeeze(-1) <= 0)[:10].tolist()
        raise ValueError(f"ref CSV has zero-probability rows at {bad}")
    probs = probs / row_sum

    return {
        'table_format': 'ref',
        'probs': probs,
        'diff_bins': diff_bins,
        'refs': refs,
        'n_refs': R,
        'n_wpatterns': 1,
    }


def load_noise_csv_auto(csv_path, table_format='auto'):
    detected = _detect_csv_format(csv_path)
    if table_format == 'auto':
        table_format = detected
    if table_format != detected:
        raise ValueError(
            f"Requested --noise-table-format={table_format}, but CSV looks like {detected}"
        )
    if table_format == 'ref':
        return load_ref_noise_csv(csv_path)
    data = load_noise_csv(csv_path)
    data['table_format'] = 'wpattern_ref'
    return data


def resolve_csv_path(csv_path, noise_dir):
    if os.path.isabs(csv_path) or os.path.dirname(csv_path):
        return csv_path
    return os.path.join(noise_dir, csv_path)


def _build_alias_tables(probs_csv, diff_bins):
    """Precompute alias tables for all (pch, row) distributions.

    Returns:
        alias_prob: (C, R, K) float64 — alias method probability table
        alias_idx:  (C, R, K) int64   — alias method redirect index
        valid:      (C, R) bool       — whether the distribution has any mass
    """
    C, R, K = probs_csv.shape
    alias_prob = np.zeros((C, R, K), dtype=np.float64)
    alias_idx = np.zeros((C, R, K), dtype=np.int64)
    valid = np.zeros((C, R), dtype=bool)

    for pch in range(C):
        for row in range(R):
            p = probs_csv[pch, row]
            p_sum = p.sum()
            if p_sum <= 0:
                continue
            valid[pch, row] = True
            q = p / p_sum * K  # scaled probabilities

            small, large = [], []
            for i in range(K):
                if q[i] < 1.0:
                    small.append(i)
                else:
                    large.append(i)

            prob = np.ones(K, dtype=np.float64)
            idx = np.arange(K, dtype=np.int64)

            while small and large:
                s = small.pop()
                l = large.pop()
                prob[s] = q[s]
                idx[s] = l
                q[l] = q[l] + q[s] - 1.0
                if q[l] < 1.0:
                    small.append(l)
                else:
                    large.append(l)

            alias_prob[pch, row] = prob
            alias_idx[pch, row] = idx

    return alias_prob, alias_idx, valid


def _alias_sample(alias_prob, alias_idx, diff_bins, rng, size):
    """Sample from a single alias table.

    alias_prob: (K,), alias_idx: (K,), diff_bins: (K,)
    Returns: (size,) float64 array of sampled diff_bin values.
    """
    K = len(diff_bins)
    col = rng.integers(0, K, size=size)
    u = rng.random(size=size)
    use_alias = u >= alias_prob[col]
    chosen = np.where(use_alias, alias_idx[col], col)
    return diff_bins[chosen]


def sample_csv_noise(adc_codes_all, skip_mask_all, pseudo_chs, csv_data, rng,
                     n_trials=1, alias_tables=None, clean_ref=None,
                     ref_reconstruction_granularity='input_bitplane'):
    """Sample reconstructed noise from CSV distributions using actual per-bitplane ADC codes.

    All n_trials are batched together for vectorized sampling.

    Args:
        adc_codes_all: (ABITS, WBITS, OC, OH, OW) int16 — ADC output per bitplane
        skip_mask_all: (ABITS, OH, OW) bool — True where chip skips ADC+noise
        pseudo_chs: (OC,) int — pseudo channel index per output channel
        csv_data: dict from load_noise_csv
        rng: numpy random Generator
        n_trials: number of MC noise samples per output element
        alias_tables: optional precomputed (alias_prob, alias_idx, valid) tuple

    Returns:
        recon_noise: (n_trials, OC, OH, OW) int32 — reconstructed noise at int16 output level
    """
    if csv_data.get('table_format') == 'ref':
        if ref_reconstruction_granularity == 'input_bitplane':
            return sample_ref_csv_noise(
                adc_codes_all, skip_mask_all, pseudo_chs, csv_data, rng,
                n_trials=n_trials, alias_tables=alias_tables)
        if ref_reconstruction_granularity == 'output':
            if clean_ref is None:
                raise ValueError(
                    "clean_ref is required when "
                    "ref_reconstruction_granularity='output'")
            return sample_ref_csv_noise_output(
                clean_ref, pseudo_chs, csv_data, rng,
                n_trials=n_trials, alias_tables=alias_tables)
        raise ValueError(
            "ref_reconstruction_granularity must be 'input_bitplane' or 'output'")

    A, W, OC, OH, OW = adc_codes_all.shape
    diff_bins = csv_data['diff_bins']  # (K,)
    n_refs = csv_data['n_refs']
    K = len(diff_bins)
    N = OH * OW  # spatial elements per channel

    if alias_tables is None:
        alias_tables = _build_alias_tables(csv_data['probs'], diff_bins)
    alias_prob, alias_idx, alias_valid = alias_tables

    adc_codes = adc_codes_all.astype(np.int64)

    # Batch all trials: sample (n_trials, OC, OH, OW) noise per (abit, wbit)
    recon_noise = np.zeros((n_trials, OC, OH, OW), dtype=np.float64)

    for abit in range(A):
        abit_scale = 1 << abit
        skip_2d = skip_mask_all[abit]  # (OH, OW)

        wbit_noise_sum = np.zeros((n_trials, OC, OH, OW), dtype=np.float64)

        for wbit in range(W):
            w_scale = W_SCALE[wbit]
            codes = adc_codes[abit, wbit]  # (OC, OH, OW)

            # noise_adc: (n_trials, OC, N) flat spatial
            noise_adc_flat = np.zeros((n_trials, OC, N), dtype=np.float64)

            for oc_i in range(OC):
                pch = pseudo_chs[oc_i]
                codes_flat = codes[oc_i].ravel()  # (N,)
                rows = wbit * n_refs + codes_flat

                unique_rows, inverse = np.unique(rows, return_inverse=True)
                for ui, row in enumerate(unique_rows):
                    if not alias_valid[pch, row]:
                        continue
                    elem_mask = inverse == ui  # (N,) bool
                    n_elem = elem_mask.sum()
                    total_samples = n_trials * n_elem
                    sampled = _alias_sample(
                        alias_prob[pch, row], alias_idx[pch, row],
                        diff_bins, rng, total_samples)
                    noise_adc_flat[:, oc_i, elem_mask] = sampled.reshape(n_trials, n_elem)

            noise_adc = noise_adc_flat.reshape(n_trials, OC, OH, OW)

            wbit_noise_sum += noise_adc * PSTEP * w_scale

        # Apply skip mask
        wbit_noise_sum[:, :, skip_2d] = 0.0

        shifted = (wbit_noise_sum * abit_scale).astype(np.int64)
        recon_noise += shifted

    return recon_noise.astype(np.int32)


def _nearest_ref_indices(values, refs):
    vals = np.asarray(values, dtype=np.float64)
    hi = np.searchsorted(refs, vals, side='left')
    hi = np.clip(hi, 0, refs.size - 1)
    lo = np.clip(hi - 1, 0, refs.size - 1)
    return np.where(np.abs(vals - refs[hi]) < np.abs(vals - refs[lo]), hi, lo).astype(np.int64)


def sample_ref_csv_noise(adc_codes_all, skip_mask_all, pseudo_chs, csv_data, rng,
                         n_trials=1, alias_tables=None):
    """Sample reconstructed noise from signed-ref/input-bitplane CSV.

    CSV rows are signed refs. For each activation bitplane:
      signed_ref = sum_wbit(adc_code * PSTEP * W_SCALE[wbit])
      noise      = sample P(noise | pseudo_ch, nearest signed_ref)
      output_noise += noise * (1 << abit)
    """
    A, W, OC, OH, OW = adc_codes_all.shape
    refs = csv_data['refs']
    diff_bins = csv_data['diff_bins']
    N = OH * OW

    if alias_tables is None:
        alias_tables = _build_alias_tables(csv_data['probs'], diff_bins)
    alias_prob, alias_idx, alias_valid = alias_tables

    adc_codes = adc_codes_all.astype(np.float64)
    w_scale = np.asarray(W_SCALE, dtype=np.float64).reshape(1, W, 1, 1, 1)
    signed_refs = (adc_codes * PSTEP * w_scale).sum(axis=1)  # (A, OC, OH, OW)

    recon_noise = np.zeros((n_trials, OC, OH, OW), dtype=np.float64)
    for abit in range(A):
        abit_scale = 1 << abit
        skip_2d = skip_mask_all[abit]
        bit_noise_flat = np.zeros((n_trials, OC, N), dtype=np.float64)

        for oc_i in range(OC):
            pch = pseudo_chs[oc_i]
            rows = _nearest_ref_indices(signed_refs[abit, oc_i].ravel(), refs)
            unique_rows, inverse = np.unique(rows, return_inverse=True)
            for ui, row in enumerate(unique_rows):
                if not alias_valid[pch, row]:
                    continue
                elem_mask = inverse == ui
                n_elem = int(elem_mask.sum())
                total_samples = n_trials * n_elem
                sampled = _alias_sample(
                    alias_prob[pch, row], alias_idx[pch, row],
                    diff_bins, rng, total_samples)
                bit_noise_flat[:, oc_i, elem_mask] = sampled.reshape(n_trials, n_elem)

        bit_noise = bit_noise_flat.reshape(n_trials, OC, OH, OW)
        bit_noise[:, :, skip_2d] = 0.0
        recon_noise += (bit_noise * abit_scale).astype(np.int64)

    return recon_noise.astype(np.int32)


def sample_ref_csv_noise_output(clean_ref, pseudo_chs, csv_data, rng,
                                n_trials=1, alias_tables=None):
    """Sample reconstructed noise from an output-level signed-ref CSV.

    CSV rows are final clean_ref values and diff_bins are already int16 output
    noise values. This mode samples once per output element:
      noise = sample P(noise | pseudo_ch, nearest clean_ref)
    """
    OC, OH, OW = clean_ref.shape
    refs = csv_data['refs']
    diff_bins = csv_data['diff_bins']
    N = OH * OW

    if alias_tables is None:
        alias_tables = _build_alias_tables(csv_data['probs'], diff_bins)
    alias_prob, alias_idx, alias_valid = alias_tables

    recon_noise_flat = np.zeros((n_trials, OC, N), dtype=np.float64)
    clean_ref = clean_ref.astype(np.float64)
    for oc_i in range(OC):
        pch = pseudo_chs[oc_i]
        rows = _nearest_ref_indices(clean_ref[oc_i].ravel(), refs)
        unique_rows, inverse = np.unique(rows, return_inverse=True)
        for ui, row in enumerate(unique_rows):
            if not alias_valid[pch, row]:
                continue
            elem_mask = inverse == ui
            n_elem = int(elem_mask.sum())
            total_samples = n_trials * n_elem
            sampled = _alias_sample(
                alias_prob[pch, row], alias_idx[pch, row],
                diff_bins, rng, total_samples)
            recon_noise_flat[:, oc_i, elem_mask] = sampled.reshape(n_trials, n_elem)

    return recon_noise_flat.reshape(n_trials, OC, OH, OW).astype(np.int32)


def main():
    parser = argparse.ArgumentParser(description='Build CSV-reconstructed noise table')
    parser.add_argument('--dump-dir', required=True, nargs='+')
    parser.add_argument('--samples', default='0-199')
    parser.add_argument('--output', default='csv_reconstructed_noise_table.npz')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--csv', default=DEFAULT_CSV, help='Per-channel noise CSV path')
    parser.add_argument('--noise-table-format', default='auto',
                        choices=['auto', 'wpattern_ref', 'ref'],
                        help='CSV format. auto detects from row labels.')
    parser.add_argument('--ref-reconstruction-granularity', default='input_bitplane',
                        choices=['input_bitplane', 'output'],
                        help='How to reconstruct ref-format CSVs. Use output for '
                             'chip-derived CSVs where rows are final clean_ref and '
                             'diff_bins are already output-level noise.')
    parser.add_argument('--noise-dir', default=DEFAULT_NOISE_DIR,
                        help='Directory containing concat_per_core.json')
    parser.add_argument('--layout-json', default=None,
                        help='Path to concat_per_core.json. Defaults to --noise-dir/concat_per_core.json')
    parser.add_argument('--npz-path', default=None,
                        help='Path to psum_imcu_column_map.npz. Defaults to selected model profile.')
    parser.add_argument('--model-profile', default=os.environ.get('NOISE_MODEL_PROFILE', None),
                        help='Noise diagnostics profile: resnet8 or kws_dscnn '
                             '(default: NOISE_MODEL_PROFILE or resnet8)')
    parser.add_argument('--n-mc-trials', type=int, default=50,
                        help='Number of MC noise samples per output element (default: 50)')
    parser.add_argument('--n-ref-bins', type=int, default=200)
    parser.add_argument('--ref-range', type=float, nargs=2, default=None)
    parser.add_argument('--n-noise-bins', type=int, default=200)
    parser.add_argument('--noise-range', type=float, nargs=2, default=None)
    parser.add_argument('--match-npz', default=None,
                        help='Match ref/noise bin range from an existing NPZ file')
    parser.add_argument('--smoothing', type=float, default=0.0)
    parser.add_argument('--min-count', type=int, default=10)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--acc-mask', type=parse_acc_mask, default=0,
                        help='4-bit acc mask used by the compiled chip model')
    args = parser.parse_args()
    model_profile = resolve_model_profile(args.model_profile)
    conv_params = get_conv_params(model_profile)

    dump_dirs = [os.path.join(CODEGEN, d) if not os.path.isabs(d) else d
                 for d in args.dump_dir]
    noise_dir = args.noise_dir if os.path.isabs(args.noise_dir) else os.path.join(CODEGEN, args.noise_dir)
    layout_json = args.layout_json or os.path.join(noise_dir, 'concat_per_core.json')
    if not os.path.isabs(layout_json):
        layout_json = os.path.join(CODEGEN, layout_json)
    npz_path = args.npz_path if args.npz_path else get_default_npz_path(model_profile)
    npz_path = npz_path if os.path.isabs(npz_path) else os.path.join(CODEGEN, npz_path)
    csv_path = resolve_csv_path(args.csv, noise_dir)
    sample_range = parse_sample_range(args.samples)
    rng = np.random.default_rng(args.seed)

    print(f"Dump dirs ({len(dump_dirs)}):")
    for d in dump_dirs:
        print(f"  {d}")
    print(f"Samples: {sample_range[0]}..{sample_range[-1]} ({len(sample_range)})")
    print(f"MC trials per element: {args.n_mc_trials}")
    print(f"CSV: {csv_path}")
    print(f"CSV format: {args.noise_table_format}")
    print(f"Ref reconstruction granularity: {args.ref_reconstruction_granularity}")
    print(f"Noise dir: {noise_dir}")
    print(f"Layout JSON: {layout_json}")
    print(f"NPZ path: {npz_path}")
    print(f"Model profile: {model_profile}")

    # Resolve checkpoint
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
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
    weights = load_weights(ckpt_path, conv_params)
    csv_data = load_noise_csv_auto(csv_path, args.noise_table_format)

    print(f"  CSV: probs shape {csv_data['probs'].shape}, "
          f"format={csv_data['table_format']}, "
          f"diff_bins: [{csv_data['diff_bins'][0]:.0f}..{csv_data['diff_bins'][-1]:.0f}]")

    print("Building alias tables for fast sampling...")
    alias_tables = _build_alias_tables(csv_data['probs'], csv_data['diff_bins'])
    print("  Alias tables ready.")

    for a in atomics:
        h = a['imce_h']
        w = a['imce_w_1based']
        a['pseudo_chs'] = np.array(
            [pseudo_map[(h, w, int(c))] for c in a['valid_cols']],
            dtype=np.int64,
        )

    print(f"  Atomics: {len(atomics)}, pseudo_chs: {n_pseudo}")

    # Determine bin ranges
    if args.match_npz:
        print(f"Matching bin range from: {args.match_npz}")
        ref_npz = np.load(args.match_npz)
        ref_edges = ref_npz['ref_bin_edges']
        noise_edges = ref_npz['noise_bin_edges']
        ref_lo, ref_hi = float(ref_edges[0]), float(ref_edges[-1])
        noise_lo, noise_hi = float(noise_edges[0]), float(noise_edges[-1])
        args.n_ref_bins = len(ref_edges) - 1
        args.n_noise_bins = len(noise_edges) - 1
    elif args.ref_range and args.noise_range:
        ref_lo, ref_hi = args.ref_range
        noise_lo, noise_hi = args.noise_range
    else:
        # Auto-detect from subset
        print("Scanning subset for range estimation...")
        ref_samples, noise_samples = [], []
        n_scanned_samples = 0
        for dump_dir, s_idx, sample_dir in iter_existing_sample_dirs(dump_dirs, sample_range):
            if n_scanned_samples >= 5:
                break
            n_scanned_samples += 1
            for a in atomics:
                clean_sq, dump_sel, _, _, _ = load_matching_reconstruction_inputs(
                    sample_dir, a, weights, conv_params, args.acc_mask, args.device)
                if dump_sel is None:
                    continue
                obs = signed_int16(dump_sel - clean_sq)
                ref_samples.append(clean_sq.ravel())
                noise_samples.append(obs.ravel())
        if not ref_samples:
            checked = []
            for dump_dir in dump_dirs:
                checked.extend(os.path.join(dump_dir, f'sample_{s}') for s in sample_range[:5])
            raise RuntimeError(
                "No usable qconv dump pairs found while estimating ranges. "
                "Check that --dump-dir points to the directory containing sample_<N> "
                f"subdirectories. First checked paths: {checked[:5]}"
            )

        all_ref = np.concatenate(ref_samples)
        all_noise = np.concatenate(noise_samples)
        ref_lo, ref_hi = float(all_ref.min()), float(all_ref.max())
        margin = (ref_hi - ref_lo) * 0.02
        ref_lo -= margin; ref_hi += margin
        n_abs_max = float(np.abs(all_noise).max())
        noise_lo, noise_hi = -n_abs_max * 1.02, n_abs_max * 1.02

    print(f"  Ref range: [{ref_lo:.0f}, {ref_hi:.0f}] ({args.n_ref_bins} bins)")
    print(f"  Noise range: [{noise_lo:.0f}, {noise_hi:.0f}] ({args.n_noise_bins} bins)")

    ref_bin_edges = np.linspace(ref_lo, ref_hi, args.n_ref_bins + 1)
    noise_bin_edges = np.linspace(noise_lo, noise_hi, args.n_noise_bins + 1)
    acc = AggregatedNoiseAccumulator(n_pseudo, ref_bin_edges, noise_bin_edges)

    # Main pass
    n_trials = args.n_mc_trials
    print(f"\nProcessing {len(dump_dirs)} dirs x {len(sample_range)} samples "
          f"x {len(atomics)} atomics x {n_trials} MC trials...")
    n_done = 0
    for dd_idx, dump_dir in enumerate(dump_dirs):
        print(f"\n--- Dir {dd_idx+1}/{len(dump_dirs)}: {os.path.basename(dump_dir)} ---")
        sample_items = [(s_idx, sample_dir)
                        for dd, s_idx, sample_dir in iter_existing_sample_dirs([dump_dir], sample_range)]
        for s_idx, sample_dir in sample_items:
            for a in atomics:
                clean_sq, _, adc_codes_all, skip_mask_all, candidates = (
                    load_matching_reconstruction_inputs(
                        sample_dir, a, weights, conv_params, args.acc_mask, args.device)
                )
                if clean_sq is None:
                    preview = "; ".join(candidates[:3]) if candidates else "no candidates"
                    print(
                        f"  [skip] sample_{s_idx} {a['func']}: no compatible input/dump pair "
                        f"({preview})"
                    )
                    continue

                # Sample noise from CSV using the requested table semantics.
                recon_noise = sample_csv_noise(
                    adc_codes_all, skip_mask_all, a['pseudo_chs'],
                    csv_data, rng, n_trials=n_trials, alias_tables=alias_tables,
                    clean_ref=clean_sq,
                    ref_reconstruction_granularity=args.ref_reconstruction_granularity)

                # Add each trial as an observation
                for trial in range(n_trials):
                    acc.add_batch(a['pseudo_chs'], clean_sq, recon_noise[trial])

                n_done += 1
                if n_done % 100 == 0:
                    print(f"  {n_done} atomic-sample pairs "
                          f"({n_done * n_trials:,} total observations)...")

    print(f"  Total: {n_done} pairs ({n_done * n_trials:,} observations)")

    results = acc.results(min_count=args.min_count, smoothing=args.smoothing)

    # Save
    if not os.path.isabs(args.output):
        base = dump_dirs[0].rstrip('/')
        if len(dump_dirs) > 1 and os.path.basename(base).startswith('run_'):
            base = os.path.dirname(base)
        output_path = os.path.join(base, args.output)
    else:
        output_path = args.output
    np.savez_compressed(output_path, **results)
    print(f"\nTable saved: {output_path} ({os.path.getsize(output_path)/1e6:.1f} MB)")

    # Summary
    print(f"\n{'='*70}")
    print(f"  CSV-Reconstructed Noise Table Summary")
    print(f"{'='*70}")
    has_data = results['count'] > 0
    total_cells = n_pseudo * args.n_ref_bins
    print(f"  Ref bins: {args.n_ref_bins}, Noise bins: {args.n_noise_bins}")
    print(f"  Coverage: {has_data.sum()}/{total_cells} ({100*has_data.sum()/total_cells:.1f}%)")
    print(f"  Total observations: {results['count'].sum():,}")
    print(f"  MC trials per element: {n_trials}")

    pch_E_mean = np.array([
        results['E'][p][has_data[p]].mean() if has_data[p].any() else 0.0
        for p in range(n_pseudo)
    ])
    top = np.argsort(np.abs(pch_E_mean))[-10:][::-1]
    print(f"\n  Top 10 pseudo_chs by |E| magnitude:")
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
