"""py_runner output vs CIM deploy debug pkl — ALL 8 conv layers.

For single-atomic convs the loop reduces to one normalize() (same as old script).
For multi-atomic convs:
  - locate every atomic func in NPZ sharing the same `orig_conv`
  - normalize each atomic to (1, oc_size, OH, OW) with its valid_cols
  - accumulate into result[1, total_oc, OH, OW] at offset oc_id*oc_block
    (IC blocks of the same oc_id are summed; OC blocks live in disjoint slices)
  - try plain sum first, fall back to incremental int16-saturated add if needed

The deploy pkl now stores per-sample tensors keyed by the second element of
the tuple, e.g. ('Int16', sample_idx, layer_name, 'output'). py_runner dumps
get overwritten on every run, so the simulator side must be re-run for the
matching sample_idx before invoking this script.

Usage:
    python scripts/compare_all_convs.py [--sample N] [--ckpt ALIAS]

If --ckpt is omitted, the alias is read from build_metadata.json
(recorded at compile time), then falls back to the registry default.
"""
import argparse
import json
import os, sys, pickle, glob
from collections import defaultdict
import numpy as np

CODEGEN = '/root/project/tvm/tvm_practice/test_imcflow/codegen'
CIM_DIR = os.environ.get('CIM_DIR', '/root/project/CIM')
EVAL_DIR = os.path.join(CODEGEN,
    'eval_dir/resnet8_subset31_pretrained_orig_evl.baremetal')
PY_RUNNER_DIR = os.path.join(EVAL_DIR, 'test_outputs/py_runner')
PSUM_NPZ = os.path.join(EVAL_DIR, 'psum_imcu_column_map.npz')
BUILD_METADATA = os.path.join(EVAL_DIR, 'build_metadata.json')


def _alias_from_metadata():
    """Read checkpoint_alias directly from build_metadata.json."""
    if not os.path.isfile(BUILD_METADATA):
        return None
    with open(BUILD_METADATA) as f:
        meta = json.load(f)
    return meta.get('checkpoint_alias')


def resolve_deploy_pkl(ckpt_alias=None, board='B2', vmode='half'):
    """Resolve checkpoint alias to debugging/<alias>/debug_model.with_noise.pkl.

    If no alias is given, read from build_metadata.json, then fall back
    to the registry default.
    """
    if not ckpt_alias:
        ckpt_alias = _alias_from_metadata()
    if not ckpt_alias:
        sys.path.insert(0, CIM_DIR)
        from checkpoints import get_default
        ckpt_alias = get_default(board, vmode)

    pkl_path = os.path.join(CODEGEN, 'debugging', ckpt_alias, 'debug_model.with_noise.pkl')
    if not os.path.isfile(pkl_path):
        raise SystemExit(
            f'Deploy pkl not found: {pkl_path}\n'
            f'Run: ./scripts/sync_artifacts.sh {ckpt_alias}'
        )
    return pkl_path, ckpt_alias


def resolve_dump_dir(sample_idx: int) -> str:
    """py_runner now writes per-sample subdirs (sample_<N>/). Prefer that
    layout; fall back to the flat dir if no sample subdir is present (legacy
    binaries that don't honor argv[6])."""
    sample_dir = os.path.join(PY_RUNNER_DIR, f'sample_{sample_idx}')
    if os.path.isdir(sample_dir):
        return sample_dir
    # Legacy fallback: dumps live directly under py_runner/ — print a warning
    # so the user knows the sim was likely run with an old binary.
    if os.path.isdir(PY_RUNNER_DIR) and glob.glob(os.path.join(PY_RUNNER_DIR, '???_*.npy')):
        print(f'[warn] no {sample_dir}; falling back to flat layout {PY_RUNNER_DIR}/.'
              f' Re-run sim with the updated host binary to isolate per-sample dumps.')
        return PY_RUNNER_DIR
    raise SystemExit(
        f'no py_runner dumps for sample {sample_idx}. Expected {sample_dir}/ or '
        f'flat {PY_RUNNER_DIR}/. Re-run simulator with --sample {sample_idx} first.'
    )

# orig_conv (relay) <-> deploy pkl layer name
CONV_NAME = [
    ('weight2_1', 'layer1.block_int16.conv1'),
    ('weight2_2', 'layer1.block_int16.conv2'),
    ('weight3_1', 'layer2.block_int16.conv1'),
    ('weight3_2', 'layer2.block_int16.conv2'),
    ('weight3_0', 'layer2.block_int16.downsample.1'),
    ('weight4_1', 'layer3.block_int16.conv1'),
    ('weight4_2', 'layer3.block_int16.conv2'),
    ('weight4_0', 'layer3.block_int16.downsample.1'),
]

INT16_MIN, INT16_MAX = -32768, 32767


def normalize(raw, valid_cols, oc_size):
    """(1, 1, OH, OW, 64) int16 -> (1, oc_size, OH, OW) int32."""
    assert raw.ndim == 5
    a = np.squeeze(raw, axis=1)
    a = np.transpose(a, (0, 3, 1, 2))
    a = a[:, valid_cols[:oc_size], :, :]
    return a.astype(np.int32)


def find_npy(dump_dir, func_name):
    """py_runner dump files are prefixed with a 3-digit node id."""
    matches = glob.glob(os.path.join(dump_dir, f'???_{func_name}.npy'))
    if not matches:
        return None
    return matches[0]


def aggregate(npz, orig_conv, dump_dir, saturated=False):
    """Sum all atomics belonging to ``orig_conv`` into (1, total_oc, OH, OW).

    ``saturated``: if True, clamp result to int16 after each per-atomic add
    (matches TVM saturating add). If False, plain int32 sum then no clamp.
    """
    fn = list(npz['func_names'])
    atomics = [i for i in range(len(fn)) if str(npz['orig_conv'][i]) == orig_conv]
    if not atomics:
        return None

    total_oc = int(npz['total_oc'][atomics[0]])
    oc_block = int(npz['oc_block'][atomics[0]])

    # Determine OH/OW from the first atomic's npy
    first = find_npy(dump_dir, str(npz['func_names'][atomics[0]]))
    if first is None:
        return None
    s = np.load(first).shape  # (1, 1, OH, OW, 64)
    OH, OW = int(s[2]), int(s[3])

    result = np.zeros((1, total_oc, OH, OW), dtype=np.int32)
    # Sort atomics so the add order is deterministic and matches the natural
    # (oc_id, ic_id) iteration.
    atomics.sort(key=lambda i: (int(npz['oc_id'][i]), int(npz['ic_id'][i])))

    for i in atomics:
        func = str(npz['func_names'][i])
        oc_size = int(npz['oc_size'][i])
        oc_id = int(npz['oc_id'][i])
        path = find_npy(dump_dir, func)
        if path is None:
            return None
        raw = np.load(path)
        valid_cols = npz[f'valid_cols/{func}']
        norm = normalize(raw, valid_cols, oc_size)  # (1, oc_size, OH, OW)
        off = oc_id * oc_block
        slot = result[:, off:off + oc_size, :, :]
        np.add(slot, norm, out=slot)
        if saturated:
            np.clip(slot, INT16_MIN, INT16_MAX, out=slot)
    return result


def diff(label, a, b):
    d = a - b
    ad = np.abs(d)
    n = a.size
    eq = int((a == b).sum())
    rmse = float(np.sqrt((d.astype(np.float64) ** 2).mean()))
    a_f = a.astype(np.float64).flatten()
    b_f = b.astype(np.float64).flatten()
    cos = float((a_f @ b_f) /
                (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-12))
    print(f'  [{label}] max_abs={int(ad.max()):>6d}  mean_abs={ad.mean():.4f}  '
          f'rmse={rmse:.4f}  exact={eq}/{n} ({100*eq/n:.2f}%)  cos={cos:.6f}')
    if ad.max() > 0:
        for idx in np.argwhere(d != 0)[:3]:
            t = tuple(idx)
            print(f'      diff at {t}: pyrun={int(a[t])} deploy={int(b[t])} '
                  f'diff={int(d[t])}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sample', '-s', type=int, default=0,
                        help='Sample index into the deploy pkl (second element of '
                             'the key tuple). The py_runner side must already be '
                             're-run for this sample. Default: 0')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Checkpoint alias (e.g. mapaware_bugfix_ndis32). '
                             'If omitted, read from build_metadata.json.')
    parser.add_argument('--board', type=str, default='B2')
    parser.add_argument('--vmode', type=str, default='half')
    args = parser.parse_args()
    sample_idx = int(args.sample)

    deploy_pkl, ckpt_alias = resolve_deploy_pkl(args.ckpt, args.board, args.vmode)

    npz = np.load(PSUM_NPZ, allow_pickle=True)
    with open(deploy_pkl, 'rb') as f:
        dep = pickle.load(f)

    # Sanity: confirm the requested sample idx exists in the pkl.
    avail_samples = sorted({k[1] for k in dep.keys()
                            if isinstance(k, tuple) and len(k) >= 2
                            and isinstance(k[1], int)})
    if sample_idx not in avail_samples:
        raise SystemExit(
            f'sample {sample_idx} not in pkl. Available: '
            f'{avail_samples[:5]}{" ..." if len(avail_samples) > 5 else ""} '
            f'(n={len(avail_samples)})'
        )

    dump_dir = resolve_dump_dir(sample_idx)

    print('=' * 90)
    print(f'  PY_RUNNER : {dump_dir}')
    print(f'  DEPLOY    : {deploy_pkl}')
    print(f'  CKPT      : {ckpt_alias}')
    print(f'  SAMPLE    : {sample_idx} (of {len(avail_samples)} stored)')
    print('=' * 90)

    n_atomics_total = defaultdict(int)
    for i in range(len(npz['func_names'])):
        n_atomics_total[str(npz['orig_conv'][i])] += 1

    for orig_conv, dep_name in CONV_NAME:
        n_atomics = n_atomics_total[orig_conv]
        flavor = 'single-atomic' if n_atomics == 1 else f'multi-atomic ({n_atomics} atomics)'
        print(f'\n--- {dep_name}  ({orig_conv}, {flavor}) ---')

        key = ('Int16', sample_idx, dep_name, 'output')
        if key not in dep:
            print(f'  pkl key missing: {key}')
            continue
        dep_out = dep[key].numpy().astype(np.int32)

        # Try plain sum first
        agg = aggregate(npz, orig_conv, dump_dir, saturated=False)
        if agg is None:
            print(f'  aggregate failed for {orig_conv} (missing atomic .npy?)')
            continue
        if agg.shape != dep_out.shape:
            print(f'  SHAPE MISMATCH: pyrun {agg.shape} vs deploy {dep_out.shape}')
            continue

        print(f'  pyrun agg : {agg.shape} min={agg.min()} max={agg.max()} '
              f'mean={agg.mean():.4f}')
        print(f'  deploy    : {dep_out.shape} min={dep_out.min()} max={dep_out.max()} '
              f'mean={dep_out.mean():.4f}')

        diff('plain int32 sum    ', agg, dep_out)
        if n_atomics > 1:
            agg_s = aggregate(npz, orig_conv, dump_dir, saturated=True)
            diff('int16 saturated sum', agg_s, dep_out)

    print('\n' + '=' * 90)


if __name__ == '__main__':
    main()
