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
import re
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



def build_qconv_to_input_map(dump_dir):
    """Map each imcflow_main_X dump to the preceding logical qconv input dump.

    The py_runner dump sequence is:
      imcflow_min_max_quantize -> optional split/bitpack -> imcflow_main_X

    For multi-atomic qconvs the per-atomic bitpack inputs may be channel-split,
    but the deploy pkl stores the logical qconv input before splitting. Keeping
    the most recent min_max_quantize dump gives that same logical tensor.
    """
    qconv_to_input = {}
    last_quant = None
    for fname in sorted(os.listdir(dump_dir)):
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


def to_numpy(value):
    """Convert torch tensors / numpy arrays from the deploy pkl to ndarray."""
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    if hasattr(value, 'numpy'):
        return value.numpy()
    return np.asarray(value)


def load_py_model_input(sample_idx, dump_dir):
    """Load the model input used by py_runner.

    New py_runner runs dump this as ``sample_<N>/model_input.npy``. For older
    runs, fall back to the exact ``test_inputs`` file that py_runner consumes.
    """
    candidates = [
        (os.path.join(dump_dir, 'model_input.npy'), 'py_runner dump'),
        (os.path.join(EVAL_DIR, 'test_inputs', f'sample_{sample_idx}',
                      'model_input.npy'), 'test_inputs/sample fallback'),
        (os.path.join(EVAL_DIR, 'test_inputs', 'model_input.npy'),
         'test_inputs legacy fallback'),
    ]
    for path, source in candidates:
        if os.path.exists(path):
            return np.load(path), path, source
    return None, None, None


def _candidate_model_input_keys(sample_idx):
    return [
        ('model_input', sample_idx),
        ('input_tensor', sample_idx),
        ('input', sample_idx),
        ('model_input', sample_idx, 'input'),
        ('FP', sample_idx, 'model_input'),
        ('FP', sample_idx, 'model_input', 'input'),
        ('Int16', sample_idx, 'model_input'),
        ('Int16', sample_idx, 'model_input', 'input'),
        ('model', sample_idx, 'input'),
    ]


def _slice_model_input_sample(arr, sample_idx):
    """Normalize common stored model-input layouts to NCHW single-sample."""
    if arr.ndim == 5 and arr.shape[0] > sample_idx and arr.shape[1:3] == (1, 3):
        return arr[sample_idx]
    if arr.ndim == 4 and arr.shape[0] > 1 and arr.shape[1] == 3:
        return arr[sample_idx:sample_idx + 1]
    if arr.ndim == 3 and arr.shape[0] == 3:
        return arr[None, ...]
    return arr


def load_deploy_model_input(dep, sample_idx):
    """Find the model input tensor in the deploy pkl, if present."""
    for key in _candidate_model_input_keys(sample_idx):
        if key in dep:
            return _slice_model_input_sample(to_numpy(dep[key]), sample_idx), key

    for key in ('model_input', 'input_tensor'):
        if key not in dep:
            continue
        arr = _slice_model_input_sample(to_numpy(dep[key]), sample_idx)
        return arr, key

    meta = dep.get(('meta', sample_idx))
    if isinstance(meta, dict):
        for key in ('model_input', 'input_tensor', 'input'):
            if key in meta:
                return _slice_model_input_sample(to_numpy(meta[key]), sample_idx), ('meta', sample_idx, key)

    shape_matches = []
    for key, value in dep.items():
        if not (isinstance(key, tuple) and len(key) >= 2 and key[1] == sample_idx):
            continue
        arr = _slice_model_input_sample(to_numpy(value), sample_idx)
        if arr.shape == (1, 3, 32, 32):
            shape_matches.append((key, arr))
    if shape_matches:
        def shape_rank(item):
            key, _ = item
            key_text = repr(key).lower()
            named = int(any(name in key_text for name in ('model', 'input', 'data', 'image')))
            return (-named, repr(key))
        key, arr = sorted(shape_matches, key=shape_rank)[0]
        return arr, key

    matches = []
    for key, value in dep.items():
        if not (isinstance(key, tuple) and len(key) >= 2 and key[1] == sample_idx):
            continue
        key_text = repr(key).lower()
        if 'model_input' not in key_text and not ('model' in key_text and 'input' in key_text):
            continue
        arr = to_numpy(value)
        if arr.ndim >= 2:
            matches.append((key, arr))

    if not matches:
        return None, None

    def rank(item):
        key, arr = item
        nchw3 = int(arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] == 3)
        return (-nchw3, arr.size, repr(key))

    key, arr = sorted(matches, key=rank)[0]
    return arr, key


def diff_numeric(label, a, b):
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    d = a64 - b64
    ad = np.abs(d)
    n = a64.size
    eq = int((a64 == b64).sum())
    tol = 0.0 if (np.issubdtype(a.dtype, np.integer) and
                  np.issubdtype(b.dtype, np.integer)) else max(1e-6, np.finfo(np.float32).eps)
    n_close = int((ad <= tol).sum())
    rmse = float(np.sqrt((d ** 2).mean()))
    a_f = a64.flatten()
    b_f = b64.flatten()
    cos = float((a_f @ b_f) /
                (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-12))
    print(f'  [{label}] max_abs={ad.max():.6g}  mean_abs={ad.mean():.6g}  '
          f'rmse={rmse:.6g}  exact={eq}/{n} ({100*eq/n:.2f}%)  '
          f'close(<={tol:.0e})={n_close}/{n}  cos={cos:.6f}')
    mask = ad > tol if tol > 0 else ad > 0
    if mask.any():
        for idx in np.argwhere(mask)[:3]:
            t = tuple(idx)
            print(f'      diff at {t}: pyrun={a64[t]:.9g} deploy={b64[t]:.9g} '
                  f'diff={d[t]:.9g}')


def compare_model_input(dep, sample_idx, dump_dir):
    print(f'\n--- model input ---')
    py_in, py_path, py_source = load_py_model_input(sample_idx, dump_dir)
    dep_in, dep_key = load_deploy_model_input(dep, sample_idx)

    if py_in is None:
        print('  py_runner model input missing: expected model_input.npy under '
              f'{dump_dir} or {os.path.join(EVAL_DIR, "test_inputs")}')
    else:
        print(f'  pyrun     : {py_in.shape} {py_in.dtype}  '
              f'min={py_in.min():.6g} max={py_in.max():.6g} '
              f'mean={py_in.mean():.6g}')
        print(f'              file={py_path} ({py_source})')
        if py_source != 'py_runner dump':
            print('  [warn] py_runner dump model_input.npy not found; using the '
                  'test_inputs file consumed by the simulator')

    if dep_in is None:
        print('  deploy model input missing in pkl. Expected a key like '
              "('model_input', sample_idx), ('FP', sample_idx, 'model_input'), "
              'or meta["model_input"].')
        return

    print(f'  deploy    : {dep_in.shape} {dep_in.dtype}  '
          f'min={dep_in.min():.6g} max={dep_in.max():.6g} '
          f'mean={dep_in.mean():.6g}')
    print(f'              key={dep_key}')
    if py_in is None:
        return
    if py_in.shape != dep_in.shape:
        print(f'  MODEL INPUT SHAPE MISMATCH: pyrun {py_in.shape} vs deploy {dep_in.shape}')
        return
    diff_numeric('model input', py_in, dep_in)


def load_qconv_input(npz, orig_conv, dump_dir, qconv_to_input):
    """Load py_runner's logical input tensor for ``orig_conv``.

    Returns (array, fname, all_input_fnames). ``array`` is NCHW int32.
    """
    fn = list(npz['func_names'])
    atomics = [i for i in range(len(fn)) if str(npz['orig_conv'][i]) == orig_conv]
    if not atomics:
        return None, None, []

    atomics.sort(key=lambda i: (int(npz['oc_id'][i]), int(npz['ic_id'][i])))
    input_fnames = []
    for i in atomics:
        func = str(npz['func_names'][i])
        fname = qconv_to_input.get(func)
        if fname is not None:
            input_fnames.append(fname)

    unique = sorted(set(input_fnames))
    if not unique:
        return None, None, []

    path = os.path.join(dump_dir, unique[0])
    if not os.path.exists(path):
        return None, unique[0], unique
    return np.load(path).astype(np.int32), unique[0], unique


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
    qconv_to_input = build_qconv_to_input_map(dump_dir)

    print('=' * 90)
    print(f'  PY_RUNNER : {dump_dir}')
    print(f'  DEPLOY    : {deploy_pkl}')
    print(f'  CKPT      : {ckpt_alias}')
    print(f'  SAMPLE    : {sample_idx} (of {len(avail_samples)} stored)')
    print('=' * 90)

    compare_model_input(dep, sample_idx, dump_dir)

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
        dep_out = to_numpy(dep[key]).astype(np.int32)

        # Try plain sum first
        agg = aggregate(npz, orig_conv, dump_dir, saturated=False)
        if agg is None:
            print(f'  aggregate failed for {orig_conv} (missing atomic .npy?)')
            continue
        if agg.shape != dep_out.shape:
            print(f'  SHAPE MISMATCH: pyrun {agg.shape} vs deploy {dep_out.shape}')
            continue

        input_key = ('Int16', sample_idx, dep_name, 'input')
        if input_key not in dep:
            print(f'  input pkl key missing: {input_key}')
        else:
            py_in, input_fname, all_input_fnames = load_qconv_input(
                npz, orig_conv, dump_dir, qconv_to_input)
            if py_in is None:
                detail = f' (candidate: {input_fname})' if input_fname else ''
                print(f'  input pyrun missing for {orig_conv}{detail}')
            else:
                if len(all_input_fnames) > 1:
                    print(f'  [warn] multiple candidate input dumps for {orig_conv}: '
                          f'{all_input_fnames}; using {input_fname}')
                dep_in = to_numpy(dep[input_key]).astype(np.int32)
                if py_in.shape != dep_in.shape:
                    print(f'  INPUT SHAPE MISMATCH: pyrun {py_in.shape} '
                          f'vs deploy {dep_in.shape}  ({input_fname})')
                else:
                    print(f'  input pyrun : {py_in.shape} min={py_in.min()} '
                          f'max={py_in.max()} mean={py_in.mean():.4f}  '
                          f'file={input_fname}')
                    print(f'  input deploy: {dep_in.shape} min={dep_in.min()} '
                          f'max={dep_in.max()} mean={dep_in.mean():.4f}')
                    diff('input quantized ', py_in, dep_in)

        print(f'  output pyrun: {agg.shape} min={agg.min()} max={agg.max()} '
              f'mean={agg.mean():.4f}')
        print(f'  output deploy: {dep_out.shape} min={dep_out.min()} max={dep_out.max()} '
              f'mean={dep_out.mean():.4f}')

        diff('plain int32 sum    ', agg, dep_out)
        if n_atomics > 1:
            agg_s = aggregate(npz, orig_conv, dump_dir, saturated=True)
            diff('int16 saturated sum', agg_s, dep_out)

    # ── final output (logit) ───────────────────────────────────────────────
    # TVM side: host binary writes the final GraphExecutor output as
    # ``output.npy`` in the same dump_dir. CIM side stores the post-Int16
    # path logits at ('Int16', sample, 'logit'). Both are (1, 10) float for
    # the classification model.
    print(f'\n--- final output (logit) ---')
    tvm_out_path = os.path.join(dump_dir, 'output.npy')
    cim_key = ('Int16', sample_idx, 'logit')
    if not os.path.exists(tvm_out_path):
        print(f'  py_runner output missing: {tvm_out_path}')
    elif cim_key not in dep:
        print(f'  pkl key missing: {cim_key}')
    else:
        tvm_out = np.load(tvm_out_path)
        cim_out = dep[cim_key].numpy()
        print(f'  pyrun     : {tvm_out.shape} {tvm_out.dtype}  '
              f'min={tvm_out.min():.4f} max={tvm_out.max():.4f} mean={tvm_out.mean():.4f}')
        print(f'  deploy    : {cim_out.shape} {cim_out.dtype}  '
              f'min={cim_out.min():.4f} max={cim_out.max():.4f} mean={cim_out.mean():.4f}')
        if tvm_out.shape != cim_out.shape:
            print(f'  SHAPE MISMATCH: pyrun {tvm_out.shape} vs deploy {cim_out.shape}')
        else:
            # Cast both to float64 for a uniform diff regardless of dtype mix
            # (TVM may emit int8/float32 depending on the head op; deploy is fp32).
            a = tvm_out.astype(np.float64)
            b = cim_out.astype(np.float64)
            d = a - b
            ad = np.abs(d)
            n = a.size
            eq = int((a == b).sum())
            # fp32 round-trip can leave sub-ULP noise even when the int psum path
            # was bitwise identical, so also report tolerance-based agreement.
            close_tol = max(1e-6, np.finfo(np.float32).eps)
            n_close = int((ad <= close_tol).sum())
            rmse = float(np.sqrt((d ** 2).mean()))
            a_f = a.flatten()
            b_f = b.flatten()
            cos = float((a_f @ b_f) /
                        (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-12))
            print(f'  [pyrun vs deploy logit] max_abs={ad.max():.3e}  '
                  f'mean_abs={ad.mean():.3e}  rmse={rmse:.3e}  '
                  f'exact={eq}/{n}  close(<={close_tol:.0e})={n_close}/{n}  '
                  f'cos={cos:.6f}')
            # Top-1 class agreement
            print(f'  top1 pyrun={int(np.argmax(a))}  deploy={int(np.argmax(b))}  '
                  f'{"AGREE" if np.argmax(a) == np.argmax(b) else "DISAGREE"}')
            # Show full vectors when small (use more digits so micro-diffs are visible)
            if n <= 32:
                print(f'  pyrun  vec: {[float(x) for x in a.flatten()]}')
                print(f'  deploy vec: {[float(x) for x in b.flatten()]}')

    print('\n' + '=' * 90)


if __name__ == '__main__':
    main()
