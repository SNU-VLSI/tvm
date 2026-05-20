#!/usr/bin/env python3
"""Feature-1 compare: chip dump vs deploy debug pkl for the 4 single-atomic convs.

Atomic-split-free convs only — comparing one chip imcflow_main_N output against
one deploy PsumConv output. Multi-atomic convs need add-tree aggregation
(feature-2).

Usage:
    python scripts/compare_chip_vs_deploy_single_atomic.py
"""

import argparse
import os
import pickle
import numpy as np

CODEGEN_DIR = '/root/project/tvm/tvm_practice/test_imcflow/codegen'
DEFAULT_FPGA_DIR = os.path.join(CODEGEN_DIR, 'debugging/fpga')
PSUM_NPZ_PATH = os.path.join(
    CODEGEN_DIR, 'eval_dir/resnet8_subset31_pretrained_orig_evl.baremetal/psum_imcu_column_map.npz')

DEPLOY_NO_NOISE = '/root/project/CIM/debug_model.without_noise.pkl'
DEPLOY_WITH_NOISE = '/root/project/CIM/debug_model.with_noise.pkl'

# 4 single-atomic convs: (deploy_module_name, chip_dump_filename, imcflow_func_name)
SINGLE_ATOMIC = [
    ('layer1.block_int16.conv1',         '013_tvmgen_default_imcflow_main_0.npy',  'tvmgen_default_imcflow_main_0'),
    ('layer1.block_int16.conv2',         '024_tvmgen_default_imcflow_main_3.npy',  'tvmgen_default_imcflow_main_3'),
    ('layer2.block_int16.conv1',         '045_tvmgen_default_imcflow_main_6.npy',  'tvmgen_default_imcflow_main_6'),
    ('layer2.block_int16.downsample.1',  '081_tvmgen_default_imcflow_main_15.npy', 'tvmgen_default_imcflow_main_15'),
]


def normalize_chip(chip_raw, valid_cols, oc_size):
    """chip (1, 1, OH, OW, 64) int16 -> (1, oc_size, OH, OW) int32

    Squeeze the kernel-dim axis, transpose NHWC->NCHW, take valid_cols rows.
    """
    assert chip_raw.ndim == 5, f"unexpected chip shape {chip_raw.shape}"
    a = np.squeeze(chip_raw, axis=1)            # (1, OH, OW, 64)
    a = np.transpose(a, (0, 3, 1, 2))           # (1, 64, OH, OW)
    a = a[:, valid_cols[:oc_size], :, :]        # (1, oc_size, OH, OW)
    return a.astype(np.int32)


def summarize_diff(label, chip, deploy):
    diff = chip - deploy
    abs_diff = np.abs(diff)
    n = chip.size
    eq = (chip == deploy).sum()
    rel_match = eq / n
    cos = (chip.astype(np.float64).flatten() @ deploy.astype(np.float64).flatten()) / (
        np.linalg.norm(chip.astype(np.float64)) * np.linalg.norm(deploy.astype(np.float64)) + 1e-12)
    print(f'  [{label}] max_abs={abs_diff.max():>6d}  mean_abs={abs_diff.mean():.4f}  '
          f'rmse={np.sqrt((diff.astype(np.float64)**2).mean()):.4f}  '
          f'exact_match={eq}/{n} ({100*rel_match:.2f}%)  cos={cos:.6f}')
    if abs_diff.max() > 0:
        # show first 5 mismatching positions
        idx = np.argwhere(diff != 0)[:5]
        for i in idx:
            i = tuple(i)
            print(f'      diff at {i}: chip={chip[i]} deploy={deploy[i]} diff={diff[i]}')


def parse_args():
    ap = argparse.ArgumentParser(description='Compare chip dump vs deploy debug pkl')
    ap.add_argument('--dump-dir', type=str, default=None,
                    help='Chip dump directory containing sample_0/ (default: debugging/fpga)')
    ap.add_argument('--ckpt-alias', type=str, default=None,
                    help='Checkpoint alias; resolves to debugging/fpga/<alias>')
    return ap.parse_args()


def main():
    args = parse_args()
    ckpt_alias = args.ckpt_alias or os.environ.get('CKPT', '')
    if args.dump_dir:
        chip_dump_dir = os.path.join(args.dump_dir, 'sample_0')
    elif ckpt_alias:
        chip_dump_dir = os.path.join(DEFAULT_FPGA_DIR, ckpt_alias, 'sample_0')
    else:
        chip_dump_dir = os.path.join(DEFAULT_FPGA_DIR, 'sample_0')

    psum = np.load(PSUM_NPZ_PATH, allow_pickle=True)
    with open(DEPLOY_NO_NOISE, 'rb') as f:
        deploy_no = pickle.load(f)
    with open(DEPLOY_WITH_NOISE, 'rb') as f:
        deploy_yes = pickle.load(f)

    print('=' * 90)
    print(f'  CHIP DUMP  : {chip_dump_dir}')
    print(f'  DEPLOY no  : {DEPLOY_NO_NOISE}')
    print(f'  DEPLOY yes : {DEPLOY_WITH_NOISE}')
    print(f'  PSUM npz   : {PSUM_NPZ_PATH}')
    print('=' * 90)

    for deploy_name, chip_file, imcflow_func in SINGLE_ATOMIC:
        chip_raw = np.load(os.path.join(chip_dump_dir, chip_file))
        valid_cols = psum[f'valid_cols/{imcflow_func}']
        # Find this func's oc_size from the flat records
        func_idx = list(psum['func_names']).index(imcflow_func)
        oc_size = int(psum['oc_size'][func_idx])
        ic_size = int(psum['ic_size'][func_idx])
        orig_conv = str(psum['orig_conv'][func_idx])
        imce = int(psum['imce_linear_id'][func_idx])

        chip_norm = normalize_chip(chip_raw, valid_cols, oc_size)

        deploy_out_no = deploy_no[('Int16', 0, deploy_name, 'output')].numpy().astype(np.int32)
        deploy_out_yes = deploy_yes[('Int16', 0, deploy_name, 'output')].numpy().astype(np.int32)

        print(f'\n--- {deploy_name}  ({orig_conv}, oc={oc_size} ic={ic_size}, IMCE={imce}) ---')
        print(f'  chip raw   : {chip_raw.shape} {chip_raw.dtype}  ->  normalized {chip_norm.shape}')
        print(f'  deploy no  : {deploy_out_no.shape} {deploy_out_no.dtype}  '
              f'min={deploy_out_no.min()} max={deploy_out_no.max()} mean={deploy_out_no.mean():.4f}')
        print(f'  deploy yes : {deploy_out_yes.shape} {deploy_out_yes.dtype}  '
              f'min={deploy_out_yes.min()} max={deploy_out_yes.max()} mean={deploy_out_yes.mean():.4f}')
        print(f'  chip norm  : '
              f'min={chip_norm.min()} max={chip_norm.max()} mean={chip_norm.mean():.4f}')
        if chip_norm.shape == deploy_out_no.shape:
            summarize_diff('chip vs deploy-no-noise', chip_norm, deploy_out_no)
            summarize_diff('chip vs deploy-with-noise', chip_norm, deploy_out_yes)
        else:
            print(f'  SHAPE MISMATCH after normalize: chip {chip_norm.shape} vs deploy {deploy_out_no.shape}')

    print('\n' + '=' * 90)


if __name__ == '__main__':
    main()
