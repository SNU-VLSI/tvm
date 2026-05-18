"""Verify AccMask.BM_0000 acc-mode skip with controlled bitplane popcounts.

This script generates synthetic 1x1 conv tests where all four input bitplanes
have spatial popcount < 8. With AccMask.BM_0000, every activation bitplane is in
acc-mode, so the hardware should skip ADC quantization/noise and use the raw
digital accumulation. Therefore chip result must exactly match the test_conv
reference for every generated case.

The input is generated bitplane-by-bitplane:
  - choose independent active channel masks for abit 0..3
  - each mask size is profile[abit], always <= 7
  - combine the four bitplanes into one uint4 input tensor

Usage:
    python scripts/verify_acc_mask_skip.py --generate-only \
      --scan-val 0x0a --dda 1.13 --ddc 1.17 --ddl 0.006 --ddf 1.24

    python scripts/verify_acc_mask_skip.py \
      --connection root@HOST:PORT --board B2 --scan-val 0x0a \
      --dda 1.13 --ddc 1.17 --ddl 0.006 --ddf 1.24 --repeats 50
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import diagnose_noise_per_qconv as diag


MEASUREMENT_ROOT = "/root/project/imcflow/xilinx/measurement"
IMCFLOW_DIR = "/root/project/imcflow"
DEFAULT_OUT_DIR = os.path.join(diag.CODEGEN, "debugging/acc_mask_verify")

IC = 16
OC = 64
IH = 8
IW = 8
KH = 1
KW = 1
STRIDE = 1
PADDING = 0
OH = 8
OW = 8
ACC_MASK_NAME = "AccMask.BM_0000"

PROFILE_PRESETS = {
    "uniform": [
        (1, 1, 1, 1),
        (2, 2, 2, 2),
        (4, 4, 4, 4),
        (7, 7, 7, 7),
    ],
    "mixed": [
        (1, 2, 4, 7),
        (7, 4, 2, 1),
    ],
    "sparse": [
        (0, 1, 0, 7),
        (3, 0, 5, 0),
    ],
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--profiles", default="uniform,mixed,sparse",
                    help="Comma-separated preset names, or semicolon-separated custom profiles. "
                         "Preset names: uniform,mixed,sparse. Custom example: '1,2,4,7;7,4,2,1'.")
    ap.add_argument("--pairs-per-case", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--node", default="0,1", help="Target IMCE node as h_id,w_id (default: 0,1)")
    ap.add_argument("--isolated-column", action="store_true",
                    help="If set, only --column has nonzero weights. Default uses random weights for all OC.")
    ap.add_argument("--column", type=int, default=0,
                    help="Target OC column for --isolated-column (default: 0)")

    ap.add_argument("-c", "--connection", default=None,
                    help="Board SSH connection: user@host:port. Required unless --generate-only.")
    ap.add_argument("-b", "--board", default="B2", choices=["B1", "B2", "1128"])
    ap.add_argument("--scan-val", required=True)
    ap.add_argument("--skip-scan", action="store_true")
    ap.add_argument("--dda", type=float, required=True)
    ap.add_argument("--ddc", type=float, required=True)
    ap.add_argument("--ddl", type=float, required=True)
    ap.add_argument("--ddf", type=float, required=True)
    ap.add_argument("--ps-remote", "--ps_remote", dest="ps_remote", default=None,
                    help="Remote power supply RPC server HOST:PORT. If omitted, local config/<board>.json is used.")
    ap.add_argument("--gpu-remote", "--gpu_remote", dest="gpu_remote", default=None,
                    help="Remote GPU RPC server HOST:PORT. Optional; mainly used by planner error-analysis code.")
    ap.add_argument("--keyfile", default=str(Path.home() / ".ssh/id_ed25519"))
    ap.add_argument("--adcmode", default="ADCMode.SIX")
    ap.add_argument("--vmode", default="VMode.FULL")
    ap.add_argument("--multmode-set", default="MultModeSet.S4")
    ap.add_argument("--measurement-root", default=MEASUREMENT_ROOT)
    ap.add_argument("--imcflow-dir", default=IMCFLOW_DIR)
    ap.add_argument("--generate-only", action="store_true")
    ap.add_argument("--transfer-only", action="store_true")
    ap.add_argument("--dryrun", action="store_true")
    return ap.parse_args()


def parse_node(node):
    parts = str(node).split(",")
    if len(parts) != 2:
        raise ValueError("--node must be h_id,w_id")
    return int(parts[0]), int(parts[1])


def parse_profiles(spec):
    spec = str(spec).strip()
    if not spec:
        raise ValueError("--profiles cannot be empty")

    profiles = []
    # Custom profiles use ';' to avoid ambiguity with comma-separated preset names.
    if ";" in spec:
        for item in spec.split(";"):
            vals = tuple(int(x) for x in item.split(",") if x != "")
            validate_profile(vals)
            profiles.append(vals)
        return profiles

    for name in spec.split(","):
        name = name.strip()
        if name in PROFILE_PRESETS:
            profiles.extend(PROFILE_PRESETS[name])
        else:
            vals = tuple(int(x) for x in name.split(":") if x != "")
            if vals:
                validate_profile(vals)
                profiles.append(vals)
            else:
                raise ValueError(f"Unknown profile preset: {name}")

    # Preserve order, remove duplicates.
    unique = []
    seen = set()
    for p in profiles:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def validate_profile(profile):
    if len(profile) != diag.ABITS:
        raise ValueError(f"profile must have {diag.ABITS} values, got {profile}")
    if any(v < 0 or v >= 8 for v in profile):
        raise ValueError(f"all profile popcounts must be in [0, 7], got {profile}")
    if any(v > IC for v in profile):
        raise ValueError(f"profile popcount exceeds IC={IC}: {profile}")


def parse_connection(conn_str, keyfile):
    match = re.match(r"^([\w.-]+)@([\w.-]+):(\d+)$", conn_str or "")
    if not match:
        raise ValueError("Connection string must be: user@host:port")
    user, host, port = match.groups()
    return {"user": user, "host": host, "port": int(port), "keyfile": keyfile}


def import_measurement_stack(args):
    os.environ.setdefault("IMCFLOW_DIR", args.imcflow_dir)
    os.environ.setdefault("GENERATED", "generated")
    paths = [
        args.measurement_root,
        os.path.join(args.imcflow_dir, "pmap/ISA_sim/multi_core"),
        os.path.join(args.imcflow_dir, "pmap/ISA_sim/multi_core/test"),
        os.path.join(args.imcflow_dir, "pmap/compiler/src/python"),
        os.path.join(args.imcflow_dir, "pmap/include"),
    ]
    for p in reversed(paths):
        if p not in sys.path:
            sys.path.insert(0, p)

    from common import client
    from common.bulk_file_transfer import BulkFileTransfer
    from common.logging_config import setup_logging
    from executor import tune_executor
    from imcflow_sim.utils.acim_enum import ADCMode, VMode, MultModeSet, AccMask
    from imcflow_sim.utils.transform import transform_conv_output_to_3d
    from io_manager import instrument

    return {
        "client": client,
        "BulkFileTransfer": BulkFileTransfer,
        "setup_logging": setup_logging,
        "tune_executor": tune_executor,
        "ADCMode": ADCMode,
        "VMode": VMode,
        "MultModeSet": MultModeSet,
        "AccMask": AccMask,
        "transform_conv_output_to_3d": transform_conv_output_to_3d,
        "instrument": instrument,
    }


def apply_rpc_env_fallback(args):
    if not args.ps_remote and os.environ.get("PSM_RPC_IP") and os.environ.get("PSM_RPC_PORT"):
        args.ps_remote = f"{os.environ['PSM_RPC_IP']}:{os.environ['PSM_RPC_PORT']}"
    if not args.gpu_remote and os.environ.get("GPU_RPC_IP") and os.environ.get("GPU_RPC_PORT"):
        args.gpu_remote = f"{os.environ['GPU_RPC_IP']}:{os.environ['GPU_RPC_PORT']}"


def resolve_scan_val_path(args):
    scan_val = str(args.scan_val)
    if not scan_val.endswith(".npz"):
        return

    path = Path(scan_val).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([
            Path.cwd() / path,
            Path(diag.CODEGEN) / path,
        ])

    for candidate in candidates:
        if candidate.exists():
            args.scan_val = str(candidate.resolve())
            return

    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"--scan-val npz not found. Tried: {tried}")


def init_gpu_rpc_if_requested(args):
    if not args.gpu_remote:
        return
    m = re.match(r"^([\w.-]+):(\d+)$", args.gpu_remote)
    if not m:
        raise ValueError("--gpu-remote/--gpu_remote must be HOST:PORT")
    from planner.gpu_rpc import init_client
    init_client(m.group(1), int(m.group(2)))
    print(f"Using remote GPU RPC at {args.gpu_remote}")


def enum_from_arg(enum_cls, value):
    if isinstance(value, enum_cls):
        return value
    name = str(value).split(".")[-1]
    return enum_cls[name]


def bitplane_popcounts(data):
    counts = []
    for abit in range(diag.ABITS):
        bp = ((data >> abit) & 1).astype(np.int32)
        counts.append(bp.sum(axis=0))  # (IH, IW)
    return counts


def make_input_for_profile(profile, rng):
    data = np.zeros((IC, IH, IW), dtype=np.uint8)
    active_by_abit = []
    for abit, n_active in enumerate(profile):
        if n_active:
            active = rng.choice(IC, size=n_active, replace=False)
            data[active, :, :] = data[active, :, :] | np.uint8(1 << abit)
        else:
            active = np.array([], dtype=np.int64)
        active_by_abit.append([int(x) for x in sorted(active.tolist())])

    counts = bitplane_popcounts(data)
    for abit, expected in enumerate(profile):
        if not np.all(counts[abit] == expected):
            raise AssertionError(
                f"abit {abit} popcount mismatch: expected {expected}, got {np.unique(counts[abit]).tolist()}"
            )
        if expected >= 8:
            raise AssertionError(f"abit {abit} expected skip popcount must be <8, got {expected}")

    return data, active_by_abit


def make_weight(rng, isolated_column=False, column=0):
    if isolated_column:
        if not 0 <= column < OC:
            raise ValueError(f"--column must be in [0, {OC - 1}], got {column}")
        weight = np.zeros((OC, IC, KH, KW), dtype=np.int8)
        weight[column, :, :, :] = rng.integers(-8, 8, size=(IC, KH, KW), dtype=np.int8)
    else:
        weight = rng.integers(-8, 8, size=(OC, IC, KH, KW), dtype=np.int8)
    return weight


def make_case(profile, profile_idx, pair_idx, rng, out_dir, args):
    data, active_by_abit = make_input_for_profile(profile, rng)
    weight = make_weight(rng, args.isolated_column, args.column)

    profile_name = "_".join(str(v) for v in profile)
    case_id = f"profile{profile_idx:03d}_{profile_name}_pair{pair_idx:03d}"
    input_path = os.path.join(out_dir, f"{case_id}_input.npz")
    weight_path = os.path.join(out_dir, f"{case_id}_weight.npz")
    np.savez_compressed(input_path, arr_0=data)
    np.savez_compressed(weight_path, arr_0=weight)

    return {
        "case_id": case_id,
        "profile_idx": int(profile_idx),
        "pair_idx": int(pair_idx),
        "popcount_profile": [int(v) for v in profile],
        "actual_popcount_unique": [
            [int(x) for x in np.unique(c).tolist()] for c in bitplane_popcounts(data)
        ],
        "active_channels_by_abit": active_by_abit,
        "input_path": input_path,
        "weight_path": weight_path,
        "acc_mask": ACC_MASK_NAME,
        "expected_skip": True,
        "isolated_column": bool(args.isolated_column),
        "column": int(args.column) if args.isolated_column else None,
    }


def build_test_args(case, args, enums, test_num):
    h_id, w_id = parse_node(args.node)
    return {
        "IH": IH,
        "IW": IW,
        "IC": IC,
        "OC": OC,
        "kernel": KH,
        "stride": STRIDE,
        "padding": PADDING,
        "h_id": h_id,
        "w_id": w_id,
        "ic_idx": 0,
        "oc_idx": 0,
        "i_npz": case["input_path"],
        "w_npz": case["weight_path"],
        "scan_val": args.scan_val,
        "skip_scan": args.skip_scan,
        "wpattern": "1111",
        "adcmode": enum_from_arg(enums["ADCMode"], args.adcmode),
        "vmode": enum_from_arg(enums["VMode"], args.vmode),
        "multmode_set": enum_from_arg(enums["MultModeSet"], args.multmode_set),
        "acc_mask": enums["AccMask"].BM_0000,
        "runsim": 0,
        "test_num": test_num,
        "file_postfix": case["case_id"],
    }


def convert_to_256bit_words(values):
    if len(values) % 16 != 0:
        raise ValueError(f"Expected a multiple of 16 int16 values, got {len(values)}")
    words = []
    for i in range(0, len(values), 16):
        word = 0
        for j in range(16):
            word |= (int(values[i + j]) & 0xFFFF) << (j * 16)
        words.append(word)
    return words


def summarize_case(case, ref_stack, res_stack):
    # ref/res shape: (repeats, OC, OH, OW), int32 signed values.
    diff = res_stack.astype(np.int32) - ref_stack.astype(np.int32)
    abs_diff = np.abs(diff)
    n_elem = int(diff.size)
    nonzero_count = int((diff != 0).sum())
    return {
        "case_id": case["case_id"],
        "profile_idx": case["profile_idx"],
        "pair_idx": case["pair_idx"],
        "popcount_profile": ",".join(str(v) for v in case["popcount_profile"]),
        "acc_mask": case["acc_mask"],
        "expected_skip": bool(case["expected_skip"]),
        "n_repeats": int(ref_stack.shape[0]),
        "n_elem": n_elem,
        "zero_rate": float((diff == 0).mean()),
        "nonzero_count": nonzero_count,
        "diff_abs_max": int(abs_diff.max()) if n_elem else 0,
        "diff_mean": float(diff.mean()) if n_elem else 0.0,
        "diff_std": float(diff.std()) if n_elem else 0.0,
        "diff_min": int(diff.min()) if n_elem else 0,
        "diff_max": int(diff.max()) if n_elem else 0,
        "pass": bool(nonzero_count == 0 and (int(abs_diff.max()) if n_elem else 0) == 0),
    }


def execute_measurement(args, cases, test_args):
    stack = import_measurement_stack(args)
    setup_logging = stack["setup_logging"]
    client = stack["client"]
    BulkFileTransfer = stack["BulkFileTransfer"]
    tune_executor = stack["tune_executor"]
    instrument = stack["instrument"]
    transform_conv_output_to_3d = stack["transform_conv_output_to_3d"]
    init_gpu_rpc_if_requested(args)

    os.makedirs(os.path.join(args.out_dir, "logs"), exist_ok=True)
    setup_logging(os.path.join(args.out_dir, "logs"))

    if args.connection is None:
        raise ValueError("--connection is required unless --generate-only")
    conn = parse_connection(args.connection, args.keyfile)

    old_cwd = os.getcwd()
    os.chdir(args.measurement_root)
    try:
        ssh_client = client.open_ssh(conn["host"], conn["user"], conn["port"], keyfile=conn["keyfile"])
        bft_client = BulkFileTransfer(
            hostname=conn["host"],
            port=conn["port"],
            username=conn["user"],
            key_filename=conn["keyfile"],
            max_threads=1,
        )

        if args.dryrun:
            ps_manager = None
        elif args.ps_remote:
            m = re.match(r"^([\w.-]+):(\d+)$", args.ps_remote)
            if not m:
                raise ValueError("--ps-remote/--ps_remote must be HOST:PORT")
            print(f"Using remote power supply RPC at {args.ps_remote}")
            ps_manager = instrument.RemotePowerSupplyManager(
                m.group(1), int(m.group(2)), f"./config/{args.board}.json")
        else:
            ps_manager = instrument.PowerSupplyManager(f"./config/{args.board}.json")

        executor = tune_executor.BulkTuneExecutor(
            args_list=test_args,
            dda=args.dda,
            ddc=args.ddc,
            ddl=args.ddl,
            ddf=args.ddf,
            ps_manager=ps_manager,
            ssh_client=ssh_client,
            bft_client=bft_client,
            run_pre_execute=True,
            dryrun=args.dryrun,
            transfer_only=args.transfer_only,
        )

        executor.execute()
        if args.transfer_only or args.dryrun:
            ssh_client.close()
            return None, None, []

        all_ref = []
        all_res = []
        for repeat_idx in range(args.repeats):
            if repeat_idx > 0:
                executor.rerun()

            ref_arrays = []
            res_arrays = []
            for ref_flat, res_flat in executor.get_reference_and_result():
                ref_words = convert_to_256bit_words(ref_flat)
                res_words = convert_to_256bit_words(res_flat)
                ref_3d = transform_conv_output_to_3d(ref_words, (OC, OH, OW)).astype(np.int32)
                res_3d = transform_conv_output_to_3d(res_words, (OC, OH, OW)).astype(np.int32)
                ref_arrays.append(ref_3d)
                res_arrays.append(res_3d)
            all_ref.append(np.stack(ref_arrays, axis=0))
            all_res.append(np.stack(res_arrays, axis=0))

        ref_stack = np.stack(all_ref, axis=1)  # (case, repeat, OC, OH, OW)
        res_stack = np.stack(all_res, axis=1)
        summaries = [
            summarize_case(case, ref_stack[case_idx], res_stack[case_idx])
            for case_idx, case in enumerate(cases)
        ]

        ssh_client.close()
        return ref_stack, res_stack, summaries
    finally:
        os.chdir(old_cwd)


def write_manifest(path, args, profiles, cases):
    h_id, w_id = parse_node(args.node)
    with open(path, "w") as f:
        json.dump({
            "metadata": {
                "shape": {
                    "IC": IC, "OC": OC, "IH": IH, "IW": IW,
                    "KH": KH, "KW": KW, "stride": STRIDE, "padding": PADDING,
                    "OH": OH, "OW": OW,
                },
                "node": {"h_id": h_id, "w_id": w_id},
                "profiles": [[int(v) for v in p] for p in profiles],
                "all_expected_skip": True,
                "skip_condition": "AccMask.BM_0000 and every bitplane popcount < 8",
            },
            "measurement": {
                "scan_val": args.scan_val,
                "dda": args.dda,
                "ddc": args.ddc,
                "ddl": args.ddl,
                "ddf": args.ddf,
                "adcmode": args.adcmode,
                "vmode": args.vmode,
                "multmode_set": args.multmode_set,
                "acc_mask": ACC_MASK_NAME,
                "pairs_per_case": args.pairs_per_case,
                "repeats": args.repeats,
                "seed": args.seed,
            },
            "cases": cases,
        }, f, indent=2, sort_keys=True)


def main():
    args = parse_args()
    apply_rpc_env_fallback(args)
    resolve_scan_val_path(args)
    os.makedirs(args.out_dir, exist_ok=True)
    npz_dir = os.path.join(args.out_dir, "synthetic_npz")
    os.makedirs(npz_dir, exist_ok=True)

    profiles = parse_profiles(args.profiles)
    rng = np.random.default_rng(args.seed)

    cases = []
    for profile_idx, profile in enumerate(profiles):
        for pair_idx in range(args.pairs_per_case):
            cases.append(make_case(profile, profile_idx, pair_idx, rng, npz_dir, args))

    manifest_path = os.path.join(args.out_dir, "acc_mask_cases_manifest.json")
    write_manifest(manifest_path, args, profiles, cases)

    if args.generate_only:
        print(f"Generated {len(cases)} acc-mask verification cases")
        print(f"  profiles: {[list(p) for p in profiles]}")
        print(f"  manifest: {manifest_path}")
        print(f"  npz dir : {npz_dir}")
        return 0

    enums = import_measurement_stack(args)
    test_args = [build_test_args(case, args, enums, i) for i, case in enumerate(cases)]
    args_path = os.path.join(args.out_dir, "acc_mask_test_args.json")
    with open(args_path, "w") as f:
        printable = []
        for d in test_args:
            printable.append({
                k: str(v) if k in ("adcmode", "vmode", "multmode_set", "acc_mask") else v
                for k, v in d.items()
            })
        json.dump(printable, f, indent=2, sort_keys=True)

    ref_stack, res_stack, summaries = execute_measurement(args, cases, test_args)
    if ref_stack is None:
        print(f"Generated/transferred {len(cases)} cases; no result collection requested.")
        print(f"  manifest: {manifest_path}")
        print(f"  args    : {args_path}")
        return 0

    diff_stack = res_stack.astype(np.int32) - ref_stack.astype(np.int32)
    raw_path = os.path.join(args.out_dir, "acc_mask_measurements_raw.npz")
    np.savez_compressed(
        raw_path,
        ref=ref_stack,
        res=res_stack,
        diff=diff_stack,
        cases_json=json.dumps(cases, sort_keys=True),
    )

    summary_df = pd.DataFrame(summaries)
    summary_path = os.path.join(args.out_dir, "acc_mask_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"Measured {len(cases)} acc-mask verification cases x {args.repeats} repeats")
    print(f"  raw    : {raw_path}")
    print(f"  summary: {summary_path}")
    print(summary_df[[
        "case_id", "popcount_profile", "zero_rate", "nonzero_count", "diff_abs_max", "pass"
    ]].to_string(index=False))

    if not bool(summary_df["pass"].all()):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
