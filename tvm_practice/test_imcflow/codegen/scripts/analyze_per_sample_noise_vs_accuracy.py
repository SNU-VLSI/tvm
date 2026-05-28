#!/usr/bin/env python3
"""Per-sample analysis: chip accuracy vs noise magnitude.

For two checkpoints, parse eval result files to get per-sample correct/wrong,
then compute per-sample noise stats from debug dumps (chip - noise_free_qconv),
and compare noise profiles of correct vs wrong samples.

Usage:
    python scripts/analyze_per_sample_noise_vs_accuracy.py
"""

import argparse
import os
import re
import sys
import numpy as np

CODEGEN = '/root/project/tvm/tvm_practice/test_imcflow/codegen'
sys.path.insert(0, os.path.join(CODEGEN, 'scripts'))

from diagnose_noise_per_qconv import (
    load_atomic_info, build_qconv_to_input_map, noise_free_qconv,
    CONV_PARAMS, NPZ_PATH,
)


def parse_eval_results(path):
    """Parse eval result file -> list of (sample_idx, predicted_label, scores)."""
    samples = []
    with open(path) as f:
        for line in f:
            m = re.match(r'\[Sample (\d+)\] Scores: \[(.+)\]', line.strip())
            if m:
                idx = int(m.group(1))
                scores = [float(x) for x in m.group(2).split(',')]
                pred = int(np.argmax(scores))
                samples.append((idx, pred, scores))
    return samples


def load_checkpoint_weights(ckpt_path):
    """Load conv weights from checkpoint using CONV_PARAMS key mapping."""
    import torch
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict']
    weights = {}
    for orig, (kh, st, pd, key) in CONV_PARAMS.items():
        weights[orig] = sd[key].cpu().numpy().astype(np.int32)
    return weights


def compute_per_sample_noise(sample_dir, atomics, weights):
    """For one sample, compute per-atomic noise stats.
    Returns dict: atomic_func -> {abs_mean, abs_max, bias, std, n_elem, orig,
                                   clean_abs_mean, snr, diff_array}
    """
    qconv_to_input = build_qconv_to_input_map(sample_dir)
    results = {}

    for a in atomics:
        orig = a['orig_conv']
        kh, st, pd, _ = CONV_PARAMS[orig]
        input_fname = qconv_to_input.get(a['func'])
        if input_fname is None:
            continue
        in_path = os.path.join(sample_dir, input_fname)
        if not os.path.exists(in_path):
            continue

        qin = np.load(in_path)
        ic_lo = a['ic_id'] * a['ic_block']
        ic_hi = ic_lo + a['ic_size']
        input_slice = qin[:, ic_lo:ic_hi, :, :]

        w_full = weights[orig]
        oc_lo = a['oc_id'] * a['oc_block']
        oc_hi = oc_lo + a['oc_size']
        w_tile = w_full[oc_lo:oc_hi, ic_lo:ic_hi, :, :]

        clean_out, _, _ = noise_free_qconv(
            input_slice, w_tile, kernel_h=kh, stride=st, padding=pd,
            acc_mask=0, device='cpu',
        )
        clean_sq = clean_out.squeeze(0)

        # Load chip dump
        out_npy = None
        for f in os.listdir(sample_dir):
            if f.endswith(f'_{a["func"]}.npy'):
                out_npy = os.path.join(sample_dir, f)
                break
        if out_npy is None:
            continue

        dump = np.load(out_npy)
        dump_sq = dump[0, 0]
        dump_sel = dump_sq[:, :, a['valid_cols']].transpose(2, 0, 1).astype(np.int32)

        diff = dump_sel - clean_sq
        # int16 wrap
        diff = (((diff + 32768) & 0xFFFF) - 32768).astype(np.int32)

        clean_abs = float(np.abs(clean_sq).mean()) + 1e-12
        noise_rms = float(np.sqrt((diff.astype(np.float64) ** 2).mean()))

        results[a['func']] = {
            'abs_mean': float(np.abs(diff).mean()),
            'abs_max': int(np.abs(diff).max()),
            'bias': float(diff.mean()),
            'std': float(diff.std()),
            'n_elem': diff.size,
            'orig': orig,
            'clean_abs_mean': clean_abs,
            'snr_db': float(20 * np.log10(clean_abs / (noise_rms + 1e-12))),
            'chip_abs_mean': float(np.abs(dump_sel).mean()),
        }
    return results


def resolve_ckpt_from_eval(eval_path):
    """Extract checkpoint_path from eval result file header."""
    with open(eval_path) as f:
        for line in f:
            if 'checkpoint_path' in line and '...' in line:
                return line.split()[-1].strip()
    raise ValueError(f'No checkpoint_path found in {eval_path}')


def analyze_one_config(label, dump_dir, eval_path, ckpt_path, labels):
    """Run full per-sample analysis for one checkpoint config."""
    print(f'\n{"="*100}')
    print(f'  {label}')
    print(f'{"="*100}')
    print(f'  dump dir  : {dump_dir}')
    print(f'  eval file : {eval_path}')
    print(f'  ckpt      : {ckpt_path}')

    atomics = load_atomic_info(NPZ_PATH)
    weights = load_checkpoint_weights(ckpt_path)
    eval_samples = parse_eval_results(eval_path)

    print(f'  eval samples: {len(eval_samples)}')

    # Per-sample: correct/wrong + noise stats
    per_sample = []
    for s_idx, pred, scores in eval_samples:
        sample_dir = os.path.join(dump_dir, f'sample_{s_idx}')
        if not os.path.isdir(sample_dir):
            continue
        gt = int(labels[s_idx])
        correct = (pred == gt)
        noise_stats = compute_per_sample_noise(sample_dir, atomics, weights)
        if not noise_stats:
            continue

        # Aggregate across atomics
        total_abs_sum = sum(v['abs_mean'] * v['n_elem'] for v in noise_stats.values())
        total_elem = sum(v['n_elem'] for v in noise_stats.values())
        total_bias_sum = sum(v['bias'] * v['n_elem'] for v in noise_stats.values())
        max_abs = max(v['abs_max'] for v in noise_stats.values())

        # Per-orig-conv aggregation
        orig_stats = {}
        for v in noise_stats.values():
            o = v['orig']
            if o not in orig_stats:
                orig_stats[o] = {
                    'bias_sum': 0, 'abs_sum': 0, 'n': 0, 'std_sq_sum': 0,
                    'clean_abs_sum': 0, 'chip_abs_sum': 0, 'snr_sum': 0, 'snr_n': 0,
                }
            orig_stats[o]['bias_sum'] += v['bias'] * v['n_elem']
            orig_stats[o]['abs_sum'] += v['abs_mean'] * v['n_elem']
            orig_stats[o]['n'] += v['n_elem']
            orig_stats[o]['std_sq_sum'] += v['std'] ** 2 * v['n_elem']
            orig_stats[o]['clean_abs_sum'] += v['clean_abs_mean'] * v['n_elem']
            orig_stats[o]['chip_abs_sum'] += v['chip_abs_mean'] * v['n_elem']
            orig_stats[o]['snr_sum'] += v['snr_db']
            orig_stats[o]['snr_n'] += 1

        sorted_scores = sorted(scores, reverse=True)
        margin = sorted_scores[0] - sorted_scores[1]
        gt_score = scores[gt]
        gt_rank = sorted(range(10), key=lambda i: -scores[i]).index(gt)

        per_sample.append({
            'idx': s_idx,
            'correct': correct,
            'pred': pred,
            'gt': gt,
            'scores': scores,
            'margin': margin,
            'gt_score': gt_score,
            'gt_rank': gt_rank,
            'global_abs_mean': total_abs_sum / total_elem,
            'global_bias': total_bias_sum / total_elem,
            'global_max': max_abs,
            'orig_stats': {
                o: {
                    'bias': s['bias_sum'] / s['n'],
                    'abs_mean': s['abs_sum'] / s['n'],
                    'rms': float(np.sqrt(s['std_sq_sum'] / s['n'])),
                    'clean_abs': s['clean_abs_sum'] / s['n'],
                    'chip_abs': s['chip_abs_sum'] / s['n'],
                    'snr_db': s['snr_sum'] / s['snr_n'],
                }
                for o, s in orig_stats.items()
            },
            'noise_per_atomic': noise_stats,
        })

    correct_samples = [s for s in per_sample if s['correct']]
    wrong_samples = [s for s in per_sample if not s['correct']]

    print(f'\n  Analyzed: {len(per_sample)} samples '
          f'(correct={len(correct_samples)}, wrong={len(wrong_samples)})')

    # ---- Section 1: Global noise comparison ----
    print(f'\n  --- Global noise: correct vs wrong ---')
    print(f'  {"":40s} {"correct":>12s} {"wrong":>12s} {"delta":>12s}')
    print(f'  {"-"*76}')
    for metric in ['global_abs_mean', 'global_bias', 'global_max']:
        c_vals = [s[metric] for s in correct_samples]
        w_vals = [s[metric] for s in wrong_samples]
        c_mean = np.mean(c_vals) if c_vals else 0
        w_mean = np.mean(w_vals) if w_vals else 0
        print(f'  {metric:40s} {c_mean:12.2f} {w_mean:12.2f} {w_mean - c_mean:12.2f}')

    # ---- Section 2: Per-layer SNR and bias ----
    all_origs = sorted(set(o for s in per_sample for o in s['orig_stats']))
    print(f'\n  --- Per-layer SNR and signal/noise magnitude ---')
    print(f'  {"layer":12s} {"clean_abs":>10s} {"chip_abs":>10s} {"noise_abs":>10s}'
          f' {"bias":>10s} {"SNR_dB":>8s} {"bias/sig%":>10s}')
    print(f'  {"-"*72}')
    for orig in all_origs:
        vals = [s['orig_stats'][orig] for s in per_sample if orig in s['orig_stats']]
        clean = np.mean([v['clean_abs'] for v in vals])
        chip = np.mean([v['chip_abs'] for v in vals])
        noise = np.mean([v['abs_mean'] for v in vals])
        bias = np.mean([v['bias'] for v in vals])
        snr = np.mean([v['snr_db'] for v in vals])
        bias_pct = abs(bias) / (clean + 1e-12) * 100
        print(f'  {orig:12s} {clean:10.1f} {chip:10.1f} {noise:10.1f}'
              f' {bias:10.1f} {snr:8.1f} {bias_pct:9.1f}%')

    # ---- Section 3: Score margin ----
    print(f'\n  --- Score margin (top1 - top2) ---')
    c_margins = [s['margin'] for s in correct_samples]
    w_margins = [s['margin'] for s in wrong_samples]
    if c_margins:
        print(f'    correct: mean={np.mean(c_margins):.3f} std={np.std(c_margins):.3f} '
              f'min={np.min(c_margins):.3f} median={np.median(c_margins):.3f}')
    if w_margins:
        print(f'    wrong:   mean={np.mean(w_margins):.3f} std={np.std(w_margins):.3f} '
              f'min={np.min(w_margins):.3f} median={np.median(w_margins):.3f}')

    # ---- Section 4: GT class score and rank ----
    print(f'\n  --- GT class score and rank ---')
    c_gt_scores = [s['gt_score'] for s in correct_samples]
    w_gt_scores = [s['gt_score'] for s in wrong_samples]
    c_gt_ranks = [s['gt_rank'] for s in correct_samples]
    w_gt_ranks = [s['gt_rank'] for s in wrong_samples]
    print(f'    GT score:  correct mean={np.mean(c_gt_scores):.3f}  wrong mean={np.mean(w_gt_scores):.3f}')
    print(f'    GT rank:   correct mean={np.mean(c_gt_ranks):.2f}  wrong mean={np.mean(w_gt_ranks):.2f}')
    # rank distribution for wrong
    rank_hist = np.bincount([s['gt_rank'] for s in wrong_samples], minlength=10)
    print(f'    Wrong GT rank distribution: {dict(enumerate(rank_hist))}')

    return per_sample


def cross_checkpoint_analysis(all_results, labels):
    """Compare two checkpoints at per-sample level."""
    print(f'\n{"="*100}')
    print(f'  Cross-checkpoint comparison')
    print(f'{"="*100}')

    labels_list = list(all_results.keys())
    for lbl, samples in all_results.items():
        c = [s for s in samples if s['correct']]
        w = [s for s in samples if not s['correct']]
        c_abs = np.mean([s['global_abs_mean'] for s in c]) if c else 0
        w_abs = np.mean([s['global_abs_mean'] for s in w]) if w else 0
        c_bias = np.mean([np.abs(s['global_bias']) for s in c]) if c else 0
        w_bias = np.mean([np.abs(s['global_bias']) for s in w]) if w else 0
        print(f'\n  {lbl}:')
        print(f'    correct({len(c):3d}): abs_mean={c_abs:.2f}  |bias|={c_bias:.2f}')
        print(f'    wrong  ({len(w):3d}): abs_mean={w_abs:.2f}  |bias|={w_bias:.2f}')

    if len(labels_list) == 2:
        lbl_a, lbl_b = labels_list
        sa = {s['idx']: s for s in all_results[lbl_a]}
        sb = {s['idx']: s for s in all_results[lbl_b]}
        common = sorted(set(sa.keys()) & set(sb.keys()))

        # Contingency table
        both_c = sum(1 for i in common if sa[i]['correct'] and sb[i]['correct'])
        a_only = sum(1 for i in common if sa[i]['correct'] and not sb[i]['correct'])
        b_only = sum(1 for i in common if not sa[i]['correct'] and sb[i]['correct'])
        both_w = sum(1 for i in common if not sa[i]['correct'] and not sb[i]['correct'])
        n = len(common)

        print(f'\n  --- Per-sample agreement ({n} common samples) ---')
        print(f'  {"":30s} {lbl_b[:30]:>30s}')
        print(f'  {"":30s} {"correct":>15s} {"wrong":>15s}')
        print(f'  {lbl_a[:30]:30s}')
        print(f'    {"correct":15s}     {both_c:5d}         {a_only:5d}')
        print(f'    {"wrong":15s}     {b_only:5d}         {both_w:5d}')

        # Samples where B corrects but A doesn't: what's different?
        b_fixes = [i for i in common if not sa[i]['correct'] and sb[i]['correct']]
        a_fixes = [i for i in common if sa[i]['correct'] and not sb[i]['correct']]

        print(f'\n  --- Samples {lbl_b[:20]} fixes (A wrong -> B correct): {len(b_fixes)} ---')
        if b_fixes:
            # What's different about noise in these samples?
            a_bias_fixed = [np.abs(sa[i]['global_bias']) for i in b_fixes]
            b_bias_fixed = [np.abs(sb[i]['global_bias']) for i in b_fixes]
            a_abs_fixed = [sa[i]['global_abs_mean'] for i in b_fixes]
            b_abs_fixed = [sb[i]['global_abs_mean'] for i in b_fixes]
            a_margin = [sa[i]['margin'] for i in b_fixes]
            b_margin = [sb[i]['margin'] for i in b_fixes]

            print(f'    |bias|:   A={np.mean(a_bias_fixed):.2f}  B={np.mean(b_bias_fixed):.2f}')
            print(f'    abs_mean: A={np.mean(a_abs_fixed):.2f}  B={np.mean(b_abs_fixed):.2f}')
            print(f'    margin:   A={np.mean(a_margin):.3f}  B={np.mean(b_margin):.3f}')

            # Per-layer: where does B's noise differ for these fixed samples?
            all_origs = sorted(set(o for i in b_fixes for o in sa[i]['orig_stats']))
            print(f'\n    Per-layer bias for B-fixed samples:')
            print(f'    {"layer":12s} {"A_bias":>10s} {"B_bias":>10s} {"A_abs":>10s} {"B_abs":>10s}')
            print(f'    {"-"*54}')
            for orig in all_origs:
                ab = [sa[i]['orig_stats'][orig]['bias'] for i in b_fixes if orig in sa[i]['orig_stats']]
                bb = [sb[i]['orig_stats'][orig]['bias'] for i in b_fixes if orig in sb[i]['orig_stats']]
                aa = [sa[i]['orig_stats'][orig]['abs_mean'] for i in b_fixes if orig in sa[i]['orig_stats']]
                ba = [sb[i]['orig_stats'][orig]['abs_mean'] for i in b_fixes if orig in sb[i]['orig_stats']]
                print(f'    {orig:12s} {np.mean(ab):10.2f} {np.mean(bb):10.2f}'
                      f' {np.mean(aa):10.2f} {np.mean(ba):10.2f}')

        print(f'\n  --- Samples {lbl_a[:20]} fixes (B wrong -> A correct): {len(a_fixes)} ---')
        if a_fixes:
            a_bias_fixed = [np.abs(sa[i]['global_bias']) for i in a_fixes]
            b_bias_fixed = [np.abs(sb[i]['global_bias']) for i in a_fixes]
            a_margin = [sa[i]['margin'] for i in a_fixes]
            b_margin = [sb[i]['margin'] for i in a_fixes]
            print(f'    |bias|:   A={np.mean(a_bias_fixed):.2f}  B={np.mean(b_bias_fixed):.2f}')
            print(f'    margin:   A={np.mean(a_margin):.3f}  B={np.mean(b_margin):.3f}')

        # Both-wrong: how bad?
        print(f'\n  --- Both wrong ({both_w} samples): same or different predictions? ---')
        if both_w > 0:
            same_pred = sum(1 for i in common
                           if not sa[i]['correct'] and not sb[i]['correct']
                           and sa[i]['pred'] == sb[i]['pred'])
            print(f'    Same wrong prediction: {same_pred}/{both_w}')
            # GT rank for both-wrong
            a_gt_ranks = [sa[i]['gt_rank'] for i in common
                          if not sa[i]['correct'] and not sb[i]['correct']]
            b_gt_ranks = [sb[i]['gt_rank'] for i in common
                          if not sa[i]['correct'] and not sb[i]['correct']]
            print(f'    GT rank: A mean={np.mean(a_gt_ranks):.2f}  B mean={np.mean(b_gt_ranks):.2f}')

        # Per-layer SNR comparison across checkpoints (all samples)
        print(f'\n  --- Per-layer SNR comparison (all {n} samples) ---')
        all_origs = sorted(set(o for s in all_results[lbl_a] for o in s['orig_stats']))
        print(f'  {"layer":12s} {"A_snr_dB":>10s} {"B_snr_dB":>10s} {"A_bias%":>10s} {"B_bias%":>10s}')
        print(f'  {"-"*54}')
        for orig in all_origs:
            a_snr = [sa[i]['orig_stats'][orig]['snr_db'] for i in common if orig in sa[i]['orig_stats']]
            b_snr = [sb[i]['orig_stats'][orig]['snr_db'] for i in common if orig in sb[i]['orig_stats']]
            a_bp = [abs(sa[i]['orig_stats'][orig]['bias']) / (sa[i]['orig_stats'][orig]['clean_abs'] + 1e-12) * 100
                    for i in common if orig in sa[i]['orig_stats']]
            b_bp = [abs(sb[i]['orig_stats'][orig]['bias']) / (sb[i]['orig_stats'][orig]['clean_abs'] + 1e-12) * 100
                    for i in common if orig in sb[i]['orig_stats']]
            print(f'  {orig:12s} {np.mean(a_snr):10.1f} {np.mean(b_snr):10.1f}'
                  f' {np.mean(a_bp):9.1f}% {np.mean(b_bp):9.1f}%')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', default=os.path.join(CODEGEN, 'dataset/cifar10/labels.npy'))
    args = ap.parse_args()

    labels = np.load(args.labels)

    configs = [
        {
            'label': 'tmp01_refine_ndis32 (chip acc ~33%)',
            'dump_dir': os.path.join(CODEGEN, 'debugging/fpga/tmp01_refine_ndis32'),
            'eval_path': os.path.join(CODEGEN, 'eval_results/dataset_results_20260520_070000.txt'),
        },
        {
            'label': 'uqat_cycle4b_repro (chip acc ~53%)',
            'dump_dir': os.path.join(CODEGEN, 'debugging/fpga/uqat_cycle4b_repro'),
            'eval_path': os.path.join(CODEGEN, 'eval_results/dataset_results_20260520_073558.txt'),
        },
    ]

    all_results = {}
    for cfg in configs:
        ckpt_path = resolve_ckpt_from_eval(cfg['eval_path'])
        all_results[cfg['label']] = analyze_one_config(
            cfg['label'], cfg['dump_dir'], cfg['eval_path'], ckpt_path, labels,
        )

    cross_checkpoint_analysis(all_results, labels)

    print('\nDone.')


if __name__ == '__main__':
    main()
