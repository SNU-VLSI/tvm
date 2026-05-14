"""Per-qconv chip-noise diagnostic.

For each atomic qconv on each sample:
  observed_noise = chip_dump - noise_free(chip_dump_input, weight_slice)
For each output element, look up the CSV-predicted noise distribution per
(pseudo_channel, wbit_pos, adc_code) tuple — using the *noise-free* intermediate
adc codes — and compute predicted E[noise] / Var[noise]. Compare observed to
predicted in aggregate.

Inputs come from:
  - chip dumps under codegen/debuggig/fpga/sample_<N>/
  - psum_imcu_column_map.npz under eval_dir/<model>_evl.linux/
  - concat_per_core.json + noise CSV under /root/project/CIM/noise/noise_df/B2_out/N32/
  - imcflow checkpoint pointed to by resnet8_subset_models.py:317

Usage:
    python scripts/diagnose_noise_per_qconv.py --n-samples 10
    python scripts/diagnose_noise_per_qconv.py --n-samples 200 --output-dir runs/full
    python scripts/diagnose_noise_per_qconv.py --n-samples 5 --layers weight4_2
"""
import argparse
import json
import os
import pickle
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


CODEGEN = '/root/project/tvm/tvm_practice/test_imcflow/codegen'
FPGA_DIR = os.path.join(CODEGEN, 'debuggig/fpga')
PYSIM_DIR = os.path.join(CODEGEN, 'eval_dir/resnet8_subset31_pretrained_orig_evl.baremetal/test_outputs/py_runner')
NPZ_PATH = os.path.join(CODEGEN, 'eval_dir/resnet8_subset31_pretrained_orig_evl.linux/psum_imcu_column_map.npz')
NOISE_DIR = '/root/project/CIM/noise/noise_df/B2_out/N32'
CSV_PATH = os.path.join(NOISE_DIR, 'B2_noise_matrix_per_ch_concat.csv')
LAYOUT_JSON = os.path.join(NOISE_DIR, 'concat_per_core.json')
CKPT_PATH = ('/root/project/CIM/trained_models/image_classification/'
             'NAT_PER_CH/prange_half_psum_duplication_1/B2/'
             '2026-May-13-10-11-49/imcflow/2026-May-14-16-04-08/checkpoint.pth.tar')

# Hardware constants (HALF mode = prange=2)
ARRAY_SIZE = 256
PRANGE = 2
PBITS = 6
WBITS = 4
ABITS = 4
PBOUND = ARRAY_SIZE / 2 / PRANGE       # 64
PSTEP = 2 * PBOUND / (2 ** PBITS)      # 2.0
NUM_LEVELS = 2 ** PBITS                # 64
W_SCALE = (1, 2, 4, -8)                # per wbit

# Per-orig-conv (kH, stride, pad) and checkpoint key
CONV_PARAMS = {
    'weight2_1': (3, 1, 1, 'layer1.block_int16.conv1.weight'),
    'weight2_2': (3, 1, 1, 'layer1.block_int16.conv2.weight'),
    'weight3_1': (3, 2, 1, 'layer2.block_int16.conv1.weight'),
    'weight3_2': (3, 1, 1, 'layer2.block_int16.conv2.weight'),
    'weight3_0': (1, 2, 0, 'layer2.block_int16.downsample.1.weight'),  # downsample 1x1
    'weight4_1': (3, 2, 1, 'layer3.block_int16.conv1.weight'),
    'weight4_2': (3, 1, 1, 'layer3.block_int16.conv2.weight'),
    'weight4_0': (1, 2, 0, 'layer3.block_int16.downsample.1.weight'),
}


# ---------------------------------------------------------------- helpers --

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n-samples', type=int, default=10,
                    help='Number of samples to process starting from --sample-start (default: 10)')
    ap.add_argument('--sample-start', type=int, default=0,
                    help='First sample index (default: 0)')
    ap.add_argument('--layers', type=str, default=None,
                    help='Comma-separated orig_conv names to restrict analysis to (default: all)')
    ap.add_argument('--output-dir', type=str, default=None,
                    help='If set, save per-atomic raw observed/predicted arrays as npz under this dir')
    ap.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                    help='Torch device for noise-free conv compute')
    ap.add_argument('--verbose', action='store_true', help='Per-atomic per-sample print')
    ap.add_argument('--compare', type=str, default='chip', choices=['chip', 'pysim', 'both'],
                    help='Which dump to compare against noise-free reference. "pysim" uses py_runner '
                         'dumps under eval_dir/.../py_runner/sample_<N>/. With pysim greedy mode, '
                         'observed should exactly equal predicted_mode_sum (sanity check on the '
                         'CSV-lookup pipeline). Default: chip')
    ap.add_argument('--acc-mask', type=int, default=0,
                    help='acc_mask config (default 0 = all abits popcount<8 eligible to skip ADC+noise). '
                         'Must match the config used to compile/run the dump source.')
    return ap.parse_args()


def load_atomic_info(npz_path):
    """Return list of dicts, one per atomic, with all fields needed downstream."""
    z = np.load(npz_path, allow_pickle=True)
    n = len(z['func_names'])
    atomics = []
    for i in range(n):
        func = str(z['func_names'][i])
        orig = str(z['orig_conv'][i])
        atomics.append({
            'i': i,
            'func': func,
            'orig_conv': orig,
            'imce_h': int(z['imce_row'][i]),
            'imce_w_0based': int(z['imce_col_in_imce'][i]),
            'imce_w_1based': int(z['imce_col_in_imce'][i]) + 1,  # CSV core key uses 1-based w
            'ic_id': int(z['ic_id'][i]),
            'ic_size': int(z['ic_size'][i]),
            'ic_block': int(z['ic_block'][i]),
            'oc_id': int(z['oc_id'][i]),
            'oc_size': int(z['oc_size'][i]),
            'oc_block': int(z['oc_block'][i]),
            'total_ic': int(z['total_ic'][i]),
            'total_oc': int(z['total_oc'][i]),
            'kernel_h': int(z['kernel_h'][i]),
            'valid_cols': z[f'valid_cols/{func}'].astype(np.int64),
        })
    return atomics


def build_qconv_to_input_map(sample_dir):
    """For each imcflow_main_X file, find the most recent fused_qnn_imcflow_min_max_quantize before it."""
    files = sorted(os.listdir(sample_dir))
    qconv_to_input = {}
    last_quant = None
    for fname in files:
        if not fname.endswith('.npy'):
            continue
        m = re.match(r'(\d+)_(.+)\.npy', fname)
        if not m:
            continue
        name = m.group(2)
        if 'imcflow_min_max_quantize' in name and 'imcflow_main' not in name:
            last_quant = fname
        elif name.startswith('tvmgen_default_imcflow_main_'):
            qconv_to_input[name] = last_quant
    return qconv_to_input


def load_pseudo_ch_map(layout_json):
    """Build (core_h, core_w_1based, orig_ch_0..63) -> pseudo_ch_0..511 map."""
    with open(layout_json) as f:
        d = json.load(f)
    m = d['pseudo_ch_to_orig']
    inv = {}
    for pseudo_str, entry in m.items():
        pseudo = int(pseudo_str)
        core = entry['core']  # 'h_w' (w is 1-based)
        orig_ch = int(entry['orig_ch'])
        h_str, w_str = core.split('_')
        inv[(int(h_str), int(w_str), orig_ch)] = pseudo
    n_per_core = int(d['n_per_core'])
    n_pseudo = int(d['n_pseudo_ch'])
    return inv, n_per_core, n_pseudo


def load_noise_csv(csv_path):
    """Load CSV → tensor of shape (C, R, K) probabilities,
    plus diff_bins (K floats), wpatterns_order (list of 4 strings),
    refs (R/n_wpatterns floats), n_rows_per_ch = R."""
    raw = pd.read_csv(csv_path, header=[0, 1], index_col=0)
    diff_bins_lv = raw.columns.get_level_values(0).astype(float).to_numpy()
    channels_lv = raw.columns.get_level_values(1).astype(int).to_numpy()
    C = int(channels_lv.max()) + 1
    mask0 = channels_lv == 0
    diff_bins = diff_bins_lv[mask0]
    K = int(mask0.sum())
    R = raw.shape[0]
    probs = np.zeros((C, R, K), dtype=np.float64)
    for ch in range(C):
        m = channels_lv == ch
        probs[ch] = raw.loc[:, m].to_numpy()

    # Parse row index labels: "wpattern_ref" e.g. "0001_0.0", "1000_126.0"
    index_pairs = [str(idx).split('_') for idx in raw.index]
    wpattern_ref_pairs = [(p[0], float(p[1])) for p in index_pairs]
    wpatterns_order = []
    for p, _ in wpattern_ref_pairs:
        if p not in wpatterns_order:
            wpatterns_order.append(p)
    refs_unique = sorted({r for _, r in wpattern_ref_pairs})
    # Validate row layout matches noise.py convention: row = wbit_pos * n_refs + ref_idx
    n_refs = len(refs_unique)
    n_wpatterns = len(wpatterns_order)
    assert R == n_wpatterns * n_refs, f'R={R} != {n_wpatterns}*{n_refs}'
    assert n_wpatterns == WBITS, f'expected {WBITS} wpatterns, got {n_wpatterns}: {wpatterns_order}'
    # Verify ordering is (wpattern_0001, ref_0..n_refs-1), (wpattern_0010, ...), ...
    for wbit, wp in enumerate(wpatterns_order):
        for ridx, ref in enumerate(refs_unique):
            row = wbit * n_refs + ridx
            wp_actual, ref_actual = wpattern_ref_pairs[row]
            assert wp_actual == wp and abs(ref_actual - ref) < 1e-9, \
                f'row {row}: expected ({wp}, {ref}), got ({wp_actual}, {ref_actual})'

    # diff_bins are noise level integers (e.g., -20..10 skipping -15)
    # Precompute mean and variance per row (over diff_bins, weighted by probs)
    db = diff_bins.reshape(1, 1, K)  # (1, 1, K)
    E = (probs * db).sum(axis=-1)               # (C, R)
    E2 = (probs * (db ** 2)).sum(axis=-1)
    Var = np.maximum(E2 - E ** 2, 0.0)
    # min/max diff_bin where prob > 0 per (C, R)
    EPS = 1e-12
    has = probs > EPS
    # to broadcast min/max safely, use masked arrays
    db_grid = np.broadcast_to(diff_bins, (C, R, K))
    db_masked_max = np.where(has, db_grid, -np.inf).max(axis=-1)
    db_masked_min = np.where(has, db_grid, +np.inf).min(axis=-1)

    # Mode noise: argmax-prob diff_bin per (C, R). Matches noise.py's
    # `noise_data['mode_noise'] = noise_levels[mode_idx]` used by the greedy
    # sampling path (line 736-738, 914). For greedy-mode py_runner this is the
    # deterministic noise value injected per (channel, wbit, adc_code).
    mode_idx = np.argmax(probs, axis=-1)            # (C, R)
    mode_noise = diff_bins[mode_idx]                # (C, R)

    return {
        'probs': probs,        # (C, R, K) full distribution
        'diff_bins': diff_bins,
        'n_refs': n_refs,
        'n_wpatterns': n_wpatterns,
        'refs': np.array(refs_unique, dtype=np.float64),
        'wpatterns_order': wpatterns_order,
        'E': E,                # (C, R)
        'Var': Var,            # (C, R)
        'diff_min': db_masked_min,  # (C, R) min noise level with non-zero prob
        'diff_max': db_masked_max,  # (C, R) max noise level with non-zero prob
        'mode_noise': mode_noise,   # (C, R) argmax diff_bin for greedy
    }


# ------------------------------------------------------------ noise-free --

def noise_free_qconv(input_uint8, weight_int, kernel_h, stride, padding, acc_mask=0, device='cpu'):
    """Chip-ISA-faithful bit-serial conv WITHOUT noise injection.

    Matches PsumConv._forward_hw (CIM/deploy/deploy_modules.py:280-452):
      - bit-serial loop over (abit, wbit)
      - per-(abit,wbit) ADC quantize: round(psum/PSTEP + 0.01).clamp(0, 63) * PSTEP
      - popcount<8 short-circuit: when (acc_mask & (1<<abit)) == 0 AND spatial
        popcount of input bit-plane is < 8, skip ADC quantization and use the
        raw conv result (= integer count) directly. py_runner also skips noise
        injection in this branch (deploy_modules.py:405-414), so for predicted
        noise the contribution from those skipped elements is 0.
      - per-abit int16 wrap of (weight_bit_sum << abit), final int16 wrap.

    Returns:
      out_int16     : np.ndarray shape (1, OC, OH, OW) int32 with int16-wrapped values
      adc_codes_all : np.ndarray shape (ABITS, WBITS, OC, OH, OW) int16 — ADC output
                      code (0..63) per (abit, wbit) per output element. Needed for
                      predicted-noise lookup downstream.
      skip_mask_all : np.ndarray shape (ABITS, OH, OW) bool — True where the chip
                      skipped ADC+noise (popcount<8 and acc_mask bit unset).
    """
    x = torch.from_numpy(input_uint8.astype(np.int64)).to(device)
    w = torch.from_numpy(np.asarray(weight_int)).to(device).to(torch.int64)
    OC, IC, kH, kW = w.shape
    assert kH == kernel_h, f'kernel mismatch: got {kH}, expected {kernel_h}'

    # Activation bit-decompose -> (ABITS, 1, IC, H, W) float32 {0,1}
    act_bp = torch.stack([((x >> abit) & 1).to(torch.float32) for abit in range(ABITS)])

    # Weight 2's complement bit-decompose. weight values are signed in [-8, 7];
    # convert to unsigned 4-bit pattern with high bit = sign.
    w_uint = (w & 0xF)
    w_bp = torch.stack([((w_uint >> wbit) & 1).to(torch.float32) for wbit in range(WBITS)])

    # Ones kernel for spatial popcount over (IC, kH, kW)
    ones_kernel = torch.ones((1, IC, kH, kW), dtype=torch.float32, device=device)

    OH = (x.shape[-2] + 2 * padding - kH) // stride + 1
    OW = (x.shape[-1] + 2 * padding - kW) // stride + 1
    adc_codes_all = torch.zeros((ABITS, WBITS, OC, OH, OW), dtype=torch.int16, device=device)
    skip_mask_all = torch.zeros((ABITS, OH, OW), dtype=torch.bool, device=device)

    out_int32 = torch.zeros((1, OC, OH, OW), dtype=torch.int64, device=device)
    for abit in range(ABITS):
        # popcount<8 short-circuit eligibility: bit unset in acc_mask
        acc_mode = (acc_mask & (1 << abit)) == 0
        if acc_mode:
            popcount = F.conv2d(act_bp[abit], ones_kernel, bias=None, stride=stride, padding=padding)
            skip_mask = popcount < 8   # (1, 1, OH, OW) bool
            skip_mask_all[abit] = skip_mask[0, 0]
        else:
            skip_mask = None

        wbs = torch.zeros((1, OC, OH, OW), dtype=torch.float64, device=device)
        for wbit in range(WBITS):
            psum = F.conv2d(act_bp[abit], w_bp[wbit], bias=None, stride=stride, padding=padding)
            # ADC: round(psum/pstep + 0.01).clamp(0, num_levels-1)
            adc = torch.clamp(torch.round(psum / PSTEP + 0.01), 0, NUM_LEVELS - 1)
            adc_codes_all[abit, wbit] = adc.squeeze(0).to(torch.int16)
            psum_adc = adc * PSTEP
            if skip_mask is not None:
                # popcount<8 branch: use raw conv integer count, no ADC quantize
                psum_final = torch.where(skip_mask, psum, psum_adc)
            else:
                psum_final = psum_adc
            wbs = wbs + psum_final.to(torch.float64) * W_SCALE[wbit]
        # int16 wrap of (wbs << abit)
        shifted = (wbs.to(torch.int64) * (1 << abit))
        wrapped = ((shifted + 32768) & 0xFFFF) - 32768
        out_int32 = out_int32 + wrapped
    out_int16 = ((out_int32 + 32768) & 0xFFFF) - 32768

    return (out_int16.cpu().numpy().astype(np.int32),
            adc_codes_all.cpu().numpy().astype(np.int16),
            skip_mask_all.cpu().numpy())


# ----------------------------------------------------------- predicted ---

def compute_predicted_stats(adc_codes_all, skip_mask_all, pseudo_chs, csv_data):
    """Given the (ABITS,WBITS,OC,OH,OW) ADC codes, the (ABITS,OH,OW) skip mask,
    and the (OC,) pseudo channel indices, compute predicted noise stats per
    output element.

    Per-(abit,wbit) noise contribution at the int16 output level:
        noise_contrib = diff_bin * PSTEP * w_scale[wbit] * (1 << abit)
    where PSTEP comes from `inject_noise`'s `pp_noisy = (adc_code + diff_bin) *
    pp_scale` formula (noise.py:1297). Missing this PSTEP factor underestimates
    predicted std by a factor of PSTEP.

    Skipped elements (popcount<8 with acc_mask bit unset) contribute 0 since
    py_runner / chip skip noise injection there.
    """
    A, W, OC, OH, OW = adc_codes_all.shape
    n_refs = csv_data['n_refs']
    E = csv_data['E']             # (C, R) where R = WBITS * n_refs
    Var = csv_data['Var']
    diff_min = csv_data['diff_min']
    diff_max = csv_data['diff_max']
    mode_noise = csv_data['mode_noise']

    # Build row indices: row = wbit * n_refs + adc_code, then index into E[pseudo_ch, row]
    # Shape after broadcast: (A, W, OC, OH, OW)
    wbit_axis = np.arange(W).reshape(1, W, 1, 1, 1)
    rows = wbit_axis * n_refs + adc_codes_all.astype(np.int64)  # (1, W, ...) broadcast

    # Expand pseudo_chs to (OC, OH, OW) by broadcasting OH/OW
    pseudo_chs_b = pseudo_chs.reshape(OC, 1, 1).astype(np.int64)  # (OC, 1, 1)

    # Need indices (A, W, OC, OH, OW) for both pseudo and row
    ch_idx = np.broadcast_to(pseudo_chs_b, (OC, OH, OW))  # (OC, OH, OW)
    rows_b = np.broadcast_to(rows, (A, W, OC, OH, OW))
    ch_b = np.broadcast_to(ch_idx[None, None], (A, W, OC, OH, OW))
    E_lk = E[ch_b, rows_b]            # (A, W, OC, OH, OW)
    Var_lk = Var[ch_b, rows_b]
    dmin_lk = diff_min[ch_b, rows_b]
    dmax_lk = diff_max[ch_b, rows_b]
    mode_lk = mode_noise[ch_b, rows_b]

    # Apply popcount<8 skip: at (abit, oh, ow) skipped, noise contribution = 0.
    # skip_mask_all shape (A, OH, OW) -> broadcast to (A, W, OC, OH, OW)
    skip_b = np.broadcast_to(skip_mask_all[:, None, None, :, :], (A, W, OC, OH, OW))
    E_lk = np.where(skip_b, 0.0, E_lk)
    Var_lk = np.where(skip_b, 0.0, Var_lk)
    dmin_lk = np.where(skip_b, 0.0, dmin_lk)
    dmax_lk = np.where(skip_b, 0.0, dmax_lk)
    mode_lk = np.where(skip_b, 0.0, mode_lk)

    # Per-(abit,wbit) noise scale at int16 output level:
    #   noise_contrib = diff_bin * PSTEP * w_scale[wbit] * (1<<abit)
    abit_arr = np.arange(A).reshape(A, 1, 1, 1, 1).astype(np.int64)
    w_scale = np.array(W_SCALE, dtype=np.int64).reshape(1, W, 1, 1, 1)
    scale = PSTEP * (1 << abit_arr) * w_scale.astype(np.float64)  # (A, W, 1, 1, 1)

    E_total = (E_lk * scale).sum(axis=(0, 1))     # (OC, OH, OW)
    Var_total = (Var_lk * (scale ** 2)).sum(axis=(0, 1))
    mode_total = (mode_lk * scale).sum(axis=(0, 1))

    # Predicted noise range bounds (worst-case sum of per-contribution extremes,
    # monotonic in signed scale).
    scale_pos = scale >= 0
    contrib_max = np.where(scale_pos, dmax_lk * scale, dmin_lk * scale)
    contrib_min = np.where(scale_pos, dmin_lk * scale, dmax_lk * scale)
    range_max = contrib_max.sum(axis=(0, 1))
    range_min = contrib_min.sum(axis=(0, 1))

    return E_total, Var_total, range_min, range_max, mode_total


# ----------------------------------------------------------- per-source --

def process_dump_source(label, dump_dir, atomics, weights, csv_data, sample_range,
                        args, layers_filter):
    """Process all (sample, atomic) pairs for one dump source (chip or pysim).
    Returns per-atomic stats dict."""
    stats = defaultdict(lambda: {
        'obs': [], 'pred_E': [], 'pred_Var': [], 'pred_mode': [],
        'pred_range_min': [], 'pred_range_max': [],
        'in_range': [], 'skip_mask': [],
    })
    n_atomics_per_sample = sum(1 for a in atomics
                                if not layers_filter or a['orig_conv'] in layers_filter)
    n_done = 0
    for s_idx in sample_range:
        sample_dir = os.path.join(dump_dir, f'sample_{s_idx}')
        if not os.path.isdir(sample_dir):
            print(f'  [{label}] [skip] sample_{s_idx}: dir missing')
            continue
        qconv_to_input = build_qconv_to_input_map(sample_dir)

        for a in atomics:
            orig = a['orig_conv']
            if layers_filter and orig not in layers_filter:
                continue
            kh, st, pd, _ = CONV_PARAMS[orig]
            input_fname = qconv_to_input.get(a['func'])
            if input_fname is None:
                print(f'  [{label}] [skip] {a["func"]}: no input dump found in sample_{s_idx}')
                continue
            in_path = os.path.join(sample_dir, input_fname)
            qin = np.load(in_path)   # (1, total_ic, H, W) uint8
            out_npy = None
            for f in os.listdir(sample_dir):
                if f.endswith(f'_{a["func"]}.npy'):
                    out_npy = os.path.join(sample_dir, f)
                    break
            if out_npy is None:
                print(f'  [{label}] [skip] {a["func"]}: no output dump found in sample_{s_idx}')
                continue
            dump = np.load(out_npy)   # (1, 1, OH, OW, 64) int16

            ic_lo = a['ic_id'] * a['ic_block']
            ic_hi = ic_lo + a['ic_size']
            input_slice = qin[:, ic_lo:ic_hi, :, :]
            w_full = weights[orig]
            oc_lo = a['oc_id'] * a['oc_block']
            oc_hi = oc_lo + a['oc_size']
            w_tile = w_full[oc_lo:oc_hi, ic_lo:ic_hi, :, :]

            clean_out, adc_codes, skip_mask = noise_free_qconv(
                input_slice, w_tile, kernel_h=kh, stride=st, padding=pd,
                acc_mask=args.acc_mask, device=args.device,
            )
            clean_sq = clean_out.squeeze(0)  # (oc_size, OH, OW)

            dump_sq = dump[0, 0]
            dump_sel = dump_sq[:, :, a['valid_cols']].transpose(2, 0, 1).astype(np.int32)
            observed = dump_sel - clean_sq

            E_pred, Var_pred, range_min, range_max, mode_pred = compute_predicted_stats(
                adc_codes, skip_mask, a['pseudo_chs'], csv_data,
            )
            in_range = (observed >= range_min) & (observed <= range_max)

            st_d = stats[a['func']]
            st_d['obs'].append(observed)
            st_d['pred_E'].append(E_pred)
            st_d['pred_Var'].append(Var_pred)
            st_d['pred_mode'].append(mode_pred)
            st_d['pred_range_min'].append(range_min)
            st_d['pred_range_max'].append(range_max)
            st_d['in_range'].append(in_range)
            st_d['skip_mask'].append(skip_mask)

            n_done += 1
            if args.verbose:
                z = (observed - E_pred) / np.sqrt(np.maximum(Var_pred, 1.0))
                mode_match = (observed == np.round(mode_pred).astype(np.int32)).mean() * 100
                print(f'  [{label}] s{s_idx:3d} {a["func"]:35s} core({a["imce_h"]},{a["imce_w_1based"]}) '
                      f'oc{a["oc_id"]}({a["oc_size"]:>2d})/ic{a["ic_id"]}({a["ic_size"]:>2d}) '
                      f'obs[mean={observed.mean():>7.1f} std={observed.std():>6.1f}] '
                      f'pred[E={E_pred.mean():>7.1f} sd={np.sqrt(Var_pred).mean():>5.1f}] '
                      f'mode%={mode_match:>5.1f} inrange%={in_range.mean()*100:>5.1f} |z|>3%={(np.abs(z)>3).mean()*100:>5.1f}')

    print(f'  [{label}] processed {n_done}/{n_atomics_per_sample * len(list(sample_range))} (sample, atomic) pairs')
    return stats


def print_summaries(label, per_atomic_stats, atomics, layers_filter):
    """Print per-atomic and per-orig_conv summary tables for one dump source."""
    VAR_FLOOR = 1.0

    print('\n' + '=' * 100)
    print(f'  [{label}] Per-atomic summary')
    print('=' * 100)
    print(f'{"func":35s} {"orig":11s} {"core":6s} {"n_elem":>9s} '
          f'{"obs_mean":>8s} {"obs_std":>8s} '
          f'{"pred_E":>8s} {"pred_sd":>7s} '
          f'{"mode%":>6s} {"inrng%":>7s} {"|z|>3%":>7s}')

    layer_agg = defaultdict(lambda: {'obs': [], 'in_range_total': 0, 'n_total': 0,
                                      'zsq_total': 0.0, 'zlarge_total': 0,
                                      'obs_minus_pred_total': 0.0,
                                      'exact_match_total': 0,
                                      'skip_total': 0})
    for a in atomics:
        if layers_filter and a['orig_conv'] not in layers_filter:
            continue
        sd = per_atomic_stats.get(a['func'])
        if sd is None or not sd['obs']:
            continue
        obs = np.concatenate([o.ravel() for o in sd['obs']])
        E = np.concatenate([e.ravel() for e in sd['pred_E']])
        Var = np.concatenate([v.ravel() for v in sd['pred_Var']])
        mode = np.concatenate([m.ravel() for m in sd['pred_mode']])
        ir = np.concatenate([r.ravel() for r in sd['in_range']])
        sk = np.concatenate([s.ravel() for s in sd['skip_mask']])  # (n_samples * ABITS * OH * OW)

        z = (obs - E) / np.sqrt(np.maximum(Var, VAR_FLOOR))
        n_large = int((np.abs(z) > 3).sum())
        exact_match = int((obs == np.round(mode).astype(np.int32)).sum())

        print(f'{a["func"]:35s} {a["orig_conv"]:11s} ({a["imce_h"]},{a["imce_w_1based"]})   '
              f'{obs.size:>9d} '
              f'{obs.mean():>8.2f} {obs.std():>8.2f} '
              f'{E.mean():>8.2f} {np.sqrt(Var).mean():>7.2f} '
              f'{exact_match/obs.size*100:>5.2f}% '
              f'{ir.mean()*100:>6.2f}% {n_large/obs.size*100:>6.2f}%')

        L = a['orig_conv']
        layer_agg[L]['obs'].append(obs)
        layer_agg[L]['in_range_total'] += int(ir.sum())
        layer_agg[L]['n_total'] += obs.size
        layer_agg[L]['zsq_total'] += float((z ** 2).sum())
        layer_agg[L]['zlarge_total'] += n_large
        layer_agg[L]['obs_minus_pred_total'] += float((obs - E).sum())
        layer_agg[L]['exact_match_total'] += exact_match
        layer_agg[L]['skip_total'] += int(sk.sum())

    print('\n' + '=' * 100)
    print(f'  [{label}] Per-orig-conv summary')
    print('=' * 100)
    print(f'{"orig_conv":12s} {"n_elem":>10s} {"obs_mean":>8s} {"obs_std":>8s} '
          f'{"bias":>8s} {"mode%":>6s} {"inrng%":>7s} {"|z|>3%":>7s} {"chi2/n":>9s}')
    for L in sorted(layer_agg.keys()):
        a = layer_agg[L]
        obs_all = np.concatenate(a['obs'])
        chi2 = a['zsq_total'] / max(a['n_total'], 1)
        bias = a['obs_minus_pred_total'] / max(a['n_total'], 1)
        print(f'{L:12s} {a["n_total"]:>10d} {obs_all.mean():>8.2f} {obs_all.std():>8.2f} '
              f'{bias:>8.2f} '
              f'{a["exact_match_total"]/a["n_total"]*100:>5.2f}% '
              f'{a["in_range_total"]/a["n_total"]*100:>6.2f}% '
              f'{a["zlarge_total"]/a["n_total"]*100:>6.2f}% {chi2:>9.2f}')


# ----------------------------------------------------------- main loop ---

def main():
    args = parse_args()
    layers_filter = set(args.layers.split(',')) if args.layers else None

    print('=' * 100)
    print('  diagnose_noise_per_qconv')
    print('=' * 100)
    print(f'  samples       : {args.sample_start} .. {args.sample_start + args.n_samples - 1}')
    print(f'  layers filter : {sorted(layers_filter) if layers_filter else "<all>"}')
    print(f'  device        : {args.device}')
    print(f'  compare       : {args.compare}')
    print(f'  acc_mask      : {args.acc_mask}  (0 = all abits popcount<8 eligible)')

    print('\nLoading atomic info, layout, noise CSV, checkpoint...')
    atomics = load_atomic_info(NPZ_PATH)
    pseudo_map, n_per_core, n_pseudo = load_pseudo_ch_map(LAYOUT_JSON)
    csv_data = load_noise_csv(CSV_PATH)
    print(f'  atomics       : {len(atomics)}')
    print(f'  pseudo chs    : {n_pseudo} ({n_per_core}/core × 16 cores)')
    print(f'  csv shape     : probs {csv_data["probs"].shape}, refs={csv_data["n_refs"]}, wpatterns={csv_data["n_wpatterns"]}')

    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    sd_ckpt = ckpt['state_dict']
    weights = {}
    for orig, (kh, st, pd, key) in CONV_PARAMS.items():
        w = sd_ckpt[key].cpu().numpy().astype(np.int32)
        weights[orig] = w

    for a in atomics:
        h = a['imce_h']; w = a['imce_w_1based']
        a['pseudo_chs'] = np.array(
            [pseudo_map[(h, w, int(c))] for c in a['valid_cols']],
            dtype=np.int64,
        )

    sources = []
    if args.compare in ('chip', 'both'):
        sources.append(('chip', FPGA_DIR))
    if args.compare in ('pysim', 'both'):
        sources.append(('pysim', PYSIM_DIR))

    sample_range = list(range(args.sample_start, args.sample_start + args.n_samples))
    all_stats = {}
    for label, dump_dir in sources:
        print(f'\nProcessing dump source [{label}] under {dump_dir}')
        if not os.path.isdir(dump_dir):
            print(f'  [{label}] [WARN] directory missing — skipping')
            continue
        all_stats[label] = process_dump_source(
            label, dump_dir, atomics, weights, csv_data, sample_range, args, layers_filter,
        )

    for label, _ in sources:
        if label in all_stats:
            print_summaries(label, all_stats[label], atomics, layers_filter)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        for label, stats in all_stats.items():
            for a in atomics:
                if layers_filter and a['orig_conv'] not in layers_filter:
                    continue
                sd = stats.get(a['func'])
                if sd is None or not sd['obs']:
                    continue
                out_path = os.path.join(args.output_dir, f'{label}__{a["func"]}.npz')
                np.savez_compressed(
                    out_path,
                    observed=np.stack(sd['obs']),
                    pred_E=np.stack(sd['pred_E']),
                    pred_Var=np.stack(sd['pred_Var']),
                    pred_mode=np.stack(sd['pred_mode']),
                    pred_range_min=np.stack(sd['pred_range_min']),
                    pred_range_max=np.stack(sd['pred_range_max']),
                    in_range=np.stack(sd['in_range']),
                    skip_mask=np.stack(sd['skip_mask']),
                    pseudo_chs=a['pseudo_chs'],
                    valid_cols=a['valid_cols'],
                    imce_h=a['imce_h'], imce_w_1based=a['imce_w_1based'],
                    orig_conv=a['orig_conv'],
                    sample_start=args.sample_start, n_samples=args.n_samples,
                )
        print(f'\nRaw arrays saved under {args.output_dir}/')

    print('\nDone.')


if __name__ == '__main__':
    sys.exit(main())
