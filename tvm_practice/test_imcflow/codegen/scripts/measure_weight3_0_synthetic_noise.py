"""Generate and measure synthetic weight3_0 low-level noise cases.

Input is the candidate JSON emitted by analyze_weight3_0_ref_psum.py.  For each
candidate tuple, this script creates small 1x1 conv tests that force the target
(abit, wbit, raw_psum/adc_code, physical column) condition on the selected B2
IMCE node, then reuses the measurement BulkTuneExecutor to run the tests.

The synthetic conv shape intentionally matches the stable planner test shape:
  input  : IC=256, IH=1, IW=1
  weight : OC=64, IC=256, KH=KW=1
  conv   : stride=1, padding=0 -> OH=OW=1

The original weight3_0 layer shape is not reproduced here.  The target bitline
psum is determined by one dot-product over IC, so a 1x1 planner-style shape is
enough and avoids shape-specific chip hangs seen with the ResNet-like synthetic
shape.

Only the target physical output column gets nonzero weights.  The input bit-plane
popcount is kept >=8 so the target abit avoids the acc-mode skip path.

Usage:
    python scripts/measure_weight3_0_synthetic_noise.py \
      --candidates debugging/noise_lowlevel/weight3_0/weight3_0_candidate_tuples.json \
      --connection root@HOST:PORT --board B2 --scan-val 0x0a \
      --dda 1.13 --ddc 1.17 --ddl 0.006 --ddf 1.24

For generation-only smoke testing:
    python scripts/measure_weight3_0_synthetic_noise.py --generate-only --limit 1 ...
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
REMOTE_TEST_ROOT = "/home/root/imcflow/xilinx/petalinux-csrc"
DEFAULT_OUT_DIR = os.path.join(diag.CODEGEN, "debugging/noise_lowlevel/weight3_0_measure")

IC = 256
OC = 64
IH = 1
IW = 1
KH = 1
KW = 1
STRIDE = 1
PADDING = 0
OH = 1
OW = 1


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", default=None,
                    help="Candidate JSON from analyze_weight3_0_ref_psum.py")
    ap.add_argument("--noise-csv", default=None,
                    help="Reference noise CSV used for csv_diff_* comparison. "
                         "If omitted, use the CSV stats embedded in --candidates.")
    ap.add_argument("--recompute-summary", action="store_true",
                    help="Do not generate or measure cases. Reuse a raw measurement npz and "
                         "recompute synthetic_vs_csv_summary.csv, optionally with --noise-csv.")
    ap.add_argument("--raw-measurements", default=None,
                    help="Raw npz to reuse with --recompute-summary. Default: "
                         "<out-dir>/synthetic_measurements_raw.npz")
    ap.add_argument("--summary-out", default=None,
                    help="Output summary CSV path. Default: <out-dir>/synthetic_vs_csv_summary.csv")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--candidate-id", type=int, action="append", default=None,
                    help="Only use candidate_id(s) from the candidate JSON. Can be repeated.")
    ap.add_argument("--pairs-per-candidate", type=int, default=1,
                    help="Number of randomized input/weight channel assignments per candidate.")
    ap.add_argument("--repeats", type=int, default=100,
                    help="Number of chip executions per generated case.")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("-c", "--connection", default=None,
                    help="Board SSH connection: user@host:port. Required unless --generate-only.")
    ap.add_argument("-b", "--board", default="B2", choices=["B1", "B2", "1128"])
    ap.add_argument("--scan-val", default=None,
                    help="Scan value used for the CSV condition, e.g. 0x0a.")
    ap.add_argument("--skip-scan", action="store_true")
    ap.add_argument("--dda", type=float, default=None)
    ap.add_argument("--ddc", type=float, default=None)
    ap.add_argument("--ddl", type=float, default=None)
    ap.add_argument("--ddf", type=float, default=None)
    ap.add_argument("--ps-remote", "--ps_remote", dest="ps_remote", default=None,
                    help="Remote power supply RPC server HOST:PORT. If omitted, local config/<board>.json is used.")
    ap.add_argument("--gpu-remote", "--gpu_remote", dest="gpu_remote", default=None,
                    help="Remote GPU RPC server HOST:PORT. Optional; mainly used by planner error-analysis code.")
    ap.add_argument("--keyfile", default=str(Path.home() / ".ssh/id_ed25519"))
    ap.add_argument("--adcmode", default="ADCMode.SIX")
    ap.add_argument("--vmode", default="VMode.FULL")
    ap.add_argument("--multmode-set", default="MultModeSet.S4")
    ap.add_argument("--acc-mask", default="AccMask.BM_0000")
    ap.add_argument("--measurement-root", default=MEASUREMENT_ROOT)
    ap.add_argument("--imcflow-dir", default=IMCFLOW_DIR)
    ap.add_argument("--generate-only", action="store_true",
                    help="Only emit synthetic npz files and manifest; do not connect to the board.")
    ap.add_argument("--transfer-only", action="store_true",
                    help="Generate and upload tests, but skip board execution.")
    ap.add_argument("--dryrun", action="store_true",
                    help="Instantiate executor in dryrun mode; no result collection is attempted.")
    ap.add_argument("--force-board-reset", action="store_true",
                    help="Run 'make clear_time' on the board before execution so the next make run performs reset/warmup.")
    return ap.parse_args()


def validate_args(args):
    if args.recompute_summary:
        return
    if args.candidates is None:
        raise ValueError("--candidates is required unless --recompute-summary")
    if args.scan_val is None:
        raise ValueError("--scan-val is required unless --recompute-summary")
    for name in ("dda", "ddc", "ddl", "ddf"):
        if getattr(args, name) is None:
            raise ValueError(f"--{name} is required unless --recompute-summary")


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
    from common.chip_lock import ChipLock
    from common.logging_config import setup_logging
    from executor import tune_executor
    from imcflow_sim.utils.acim_enum import ADCMode, VMode, MultModeSet, AccMask
    from imcflow_sim.utils.transform import transform_conv_output_to_3d
    from io_manager import instrument

    return {
        "client": client,
        "BulkFileTransfer": BulkFileTransfer,
        "ChipLock": ChipLock,
        "setup_logging": setup_logging,
        "tune_executor": tune_executor,
        "ADCMode": ADCMode,
        "VMode": VMode,
        "MultModeSet": MultModeSet,
        "AccMask": AccMask,
        "transform_conv_output_to_3d": transform_conv_output_to_3d,
        "instrument": instrument,
    }


def load_candidates(path, limit, candidate_ids=None):
    with open(path) as f:
        payload = json.load(f)
    candidates = payload["candidates"]
    if candidate_ids is not None:
        wanted = set(int(v) for v in candidate_ids)
        candidates = [c for c in candidates if int(c["candidate_id"]) in wanted]
    if limit is not None:
        candidates = candidates[:limit]
    if not candidates:
        raise ValueError("No candidates selected")
    return payload.get("metadata", {}), candidates


def apply_reference_noise_csv(candidates, noise_csv):
    if noise_csv is None:
        return None

    csv_path = diag.resolve_noise_csv_path(noise_csv)
    csv_data = diag.load_noise_csv(csv_path)
    for candidate in candidates:
        pseudo_ch = int(candidate["pseudo_ch"])
        wbit = int(candidate["wbit"])
        abit = int(candidate["abit"])
        adc_code = int(candidate["adc_code"])
        csv_row = wbit * csv_data["n_refs"] + adc_code
        if pseudo_ch >= csv_data["probs"].shape[0]:
            raise ValueError(
                f"pseudo_ch={pseudo_ch} is outside reference CSV channel count "
                f"{csv_data['probs'].shape[0]} ({csv_path})")
        if csv_row >= csv_data["probs"].shape[1]:
            raise ValueError(
                f"CSV row={csv_row} for wbit={wbit}, adc_code={adc_code} is outside "
                f"reference CSV row count {csv_data['probs'].shape[1]} ({csv_path})")

        scale = diag.PSTEP * diag.W_SCALE[wbit] * (1 << abit)
        probs = csv_data["probs"][pseudo_ch, csv_row]
        nonzero = probs > 1e-12
        candidate["output_scale"] = float(scale)
        candidate["csv_diff_mean"] = float(csv_data["E"][pseudo_ch, csv_row])
        candidate["csv_diff_std"] = float(np.sqrt(csv_data["Var"][pseudo_ch, csv_row]))
        candidate["csv_diff_min"] = float(csv_data["diff_min"][pseudo_ch, csv_row])
        candidate["csv_diff_max"] = float(csv_data["diff_max"][pseudo_ch, csv_row])
        candidate["csv_out_mean"] = float(candidate["csv_diff_mean"] * scale)
        candidate["csv_out_std"] = float(candidate["csv_diff_std"] * abs(scale))
        candidate["csv_support_n"] = int(nonzero.sum())
        candidate["csv_prob_mass"] = float(probs.sum())
        candidate["csv_mode_diff"] = (
            float(csv_data["diff_bins"][np.argmax(probs)]) if probs.size else float("nan")
        )
        candidate["reference_noise_csv"] = csv_path
    return csv_path


def weight_value_for_wbit(wbit):
    if wbit == 3:
        return -8
    return 1 << wbit


def make_synthetic_case(candidate, pair_id, rng, out_dir):
    abit = int(candidate["abit"])
    wbit = int(candidate["wbit"])
    raw_psum = int(candidate["raw_psum"])
    target_popcount = int(candidate.get("target_popcount", max(raw_psum, 8)))
    valid_col = int(candidate["valid_col"])
    core_h = int(candidate["core_h"])
    core_w = int(candidate["core_w"])

    if target_popcount > IC:
        raise ValueError(f"target_popcount={target_popcount} exceeds IC={IC}")
    if raw_psum > target_popcount:
        raise ValueError(f"raw_psum={raw_psum} exceeds target_popcount={target_popcount}")
    if not 0 <= valid_col < OC:
        raise ValueError(f"valid_col={valid_col} is outside OC={OC}")

    active_input = rng.choice(IC, size=target_popcount, replace=False)
    active_weight = rng.choice(active_input, size=raw_psum, replace=False) if raw_psum else np.array([], dtype=np.int64)

    data = np.zeros((IC, IH, IW), dtype=np.uint8)
    weight = np.zeros((OC, IC, KH, KW), dtype=np.int8)

    data[active_input, :, :] = np.uint8(1 << abit)
    weight[valid_col, active_weight, 0, 0] = np.int8(weight_value_for_wbit(wbit))

    case_id = f"cand{int(candidate['candidate_id']):03d}_pair{pair_id:03d}"
    input_path = os.path.join(out_dir, f"{case_id}_input.npz")
    weight_path = os.path.join(out_dir, f"{case_id}_weight.npz")
    np.savez_compressed(input_path, arr_0=data)
    np.savez_compressed(weight_path, arr_0=weight)

    return {
        "case_id": case_id,
        "candidate_id": int(candidate["candidate_id"]),
        "pair_id": int(pair_id),
        "input_path": input_path,
        "weight_path": weight_path,
        "active_input_channels": [int(x) for x in sorted(active_input.tolist())],
        "active_weight_channels": [int(x) for x in sorted(active_weight.tolist())],
        "valid_col": valid_col,
        "core_h": core_h,
        "core_w": core_w,
        "abit": abit,
        "wbit": wbit,
        "raw_psum": raw_psum,
        "adc_code": int(candidate["adc_code"]),
        "pseudo_ch": int(candidate["pseudo_ch"]),
        "output_scale": float(candidate["output_scale"]),
        "csv_diff_mean": float(candidate["csv_diff_mean"]),
        "csv_diff_std": float(candidate["csv_diff_std"]),
        "csv_diff_min": float(candidate["csv_diff_min"]),
        "csv_diff_max": float(candidate["csv_diff_max"]),
        "csv_out_mean": float(candidate["csv_out_mean"]),
        "csv_out_std": float(candidate["csv_out_std"]),
        "csv_support_n": int(candidate.get("csv_support_n", -1)),
        "csv_prob_mass": float(candidate.get("csv_prob_mass", np.nan)),
        "csv_mode_diff": float(candidate.get("csv_mode_diff", np.nan)),
        "reference_noise_csv": candidate.get("reference_noise_csv"),
        "wpattern": candidate.get("wpattern", f"{1 << wbit:04b}"),
    }


def build_test_args(case, args, enums, test_num):
    adcmode = enum_from_arg(enums["ADCMode"], args.adcmode)
    vmode = enum_from_arg(enums["VMode"], args.vmode)
    multmode_set = enum_from_arg(enums["MultModeSet"], args.multmode_set)
    acc_mask = enum_from_arg(enums["AccMask"], args.acc_mask)
    return {
        "IH": IH,
        "IW": IW,
        "IC": IC,
        "OC": OC,
        "kernel": KH,
        "stride": STRIDE,
        "padding": PADDING,
        "h_id": int(case["core_h"]),
        "w_id": int(case["core_w"]),
        "ic_idx": 0,
        "oc_idx": 0,
        "i_npz": case["input_path"],
        "w_npz": case["weight_path"],
        "scan_val": scan_val_for_case(args.scan_val, case),
        "skip_scan": args.skip_scan,
        "wpattern": case["wpattern"],
        "adcmode": adcmode,
        "vmode": vmode,
        "multmode_set": multmode_set,
        "acc_mask": acc_mask,
        "runsim": 0,
        "test_num": test_num,
        "file_postfix": case["case_id"],
    }


def apply_rpc_env_fallback(args):
    if not args.ps_remote and os.environ.get("PSM_RPC_IP") and os.environ.get("PSM_RPC_PORT"):
        args.ps_remote = f"{os.environ['PSM_RPC_IP']}:{os.environ['PSM_RPC_PORT']}"
    if not args.gpu_remote and os.environ.get("GPU_RPC_IP") and os.environ.get("GPU_RPC_PORT"):
        args.gpu_remote = f"{os.environ['GPU_RPC_IP']}:{os.environ['GPU_RPC_PORT']}"


def scan_val_for_case(scan_val, case):
    scan_val = str(scan_val)
    if scan_val.startswith("0x") or scan_val.endswith(".npz"):
        return scan_val

    base = Path(scan_val)
    h = int(case["core_h"])
    w = int(case["core_w"])
    candidates = [
        base / "opt_scan" / f"imce_{h}_{w}.npz",
        base / f"imce_{h}_{w}.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"--scan-val scan-set does not contain imce_{h}_{w}.npz. Tried: {tried}")


def resolve_scan_val_path(args):
    scan_val = str(args.scan_val)
    if scan_val.startswith("0x"):
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

    if not scan_val.endswith(".npz"):
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


def signed_int16(arr):
    arr = np.asarray(arr, dtype=np.int64)
    return (((arr + 32768) & 0xFFFF) - 32768).astype(np.int32)


def summarize_case(case, ref_stack, res_stack):
    # ref/res shape: (repeats, OC, OH, OW)
    col = int(case["valid_col"])
    diff = signed_int16(res_stack[:, col] - ref_stack[:, col])
    ref_vals = ref_stack[:, col]
    scale = float(case["output_scale"])
    norm = diff.astype(np.float64) / scale if scale != 0 else np.full(diff.shape, np.nan)
    csv_min = float(case["csv_diff_min"])
    csv_max = float(case["csv_diff_max"])
    return {
        "case_id": case["case_id"],
        "candidate_id": case["candidate_id"],
        "pair_id": case["pair_id"],
        "valid_col": col,
        "pseudo_ch": case["pseudo_ch"],
        "abit": case["abit"],
        "wbit": case["wbit"],
        "raw_psum": case["raw_psum"],
        "adc_code": case["adc_code"],
        "n_samples": int(diff.size),
        "ref_min": int(ref_vals.min()),
        "ref_max": int(ref_vals.max()),
        "diff_mean": float(diff.mean()),
        "diff_std": float(diff.std()),
        "diff_min": int(diff.min()),
        "diff_max": int(diff.max()),
        "diff_bin_mean": float(np.nanmean(norm)),
        "diff_bin_std": float(np.nanstd(norm)),
        "diff_bin_min": float(np.nanmin(norm)),
        "diff_bin_max": float(np.nanmax(norm)),
        "csv_diff_mean": case["csv_diff_mean"],
        "csv_diff_std": case["csv_diff_std"],
        "csv_diff_min": csv_min,
        "csv_diff_max": csv_max,
        "csv_out_mean": case["csv_out_mean"],
        "csv_out_std": case["csv_out_std"],
        "csv_support_n": case.get("csv_support_n", -1),
        "csv_prob_mass": case.get("csv_prob_mass", np.nan),
        "csv_mode_diff": case.get("csv_mode_diff", np.nan),
        "reference_noise_csv": case.get("reference_noise_csv"),
        "csv_range_hit_pct": float(((norm >= csv_min) & (norm <= csv_max)).mean() * 100.0),
        "mean_shift_diff_bin": float(np.nanmean(norm) - case["csv_diff_mean"]),
        "std_ratio_diff_bin": float(np.nanstd(norm) / max(case["csv_diff_std"], 1e-12)),
    }


def build_summaries(cases, ref_stack, res_stack):
    if ref_stack.shape[0] != len(cases) or res_stack.shape[0] != len(cases):
        raise ValueError(
            f"Raw measurement case count mismatch: {len(cases)} cases, "
            f"ref shape={ref_stack.shape}, res shape={res_stack.shape}")
    return [
        summarize_case(case, ref_stack[case_idx], res_stack[case_idx])
        for case_idx, case in enumerate(cases)
    ]


def write_summary_csv(args, summaries):
    summary_df = pd.DataFrame(summaries)
    summary_path = args.summary_out or os.path.join(args.out_dir, "synthetic_vs_csv_summary.csv")
    os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    return summary_df, summary_path


def print_summary_table(summary_df):
    print(summary_df[[
        "case_id", "valid_col", "diff_bin_mean", "diff_bin_std", "csv_diff_mean",
        "csv_diff_std", "csv_range_hit_pct", "mean_shift_diff_bin",
    ]].to_string(index=False))


def recompute_summary(args):
    raw_path = args.raw_measurements or os.path.join(args.out_dir, "synthetic_measurements_raw.npz")
    with np.load(raw_path, allow_pickle=False) as raw:
        ref_stack = raw["ref"]
        res_stack = raw["res"]
        cases = json.loads(raw["cases_json"].item())

    reference_noise_csv = apply_reference_noise_csv(cases, args.noise_csv)
    summaries = build_summaries(cases, ref_stack, res_stack)
    summary_df, summary_path = write_summary_csv(args, summaries)

    print(f"Recomputed summary for {len(cases)} synthetic cases")
    print(f"  raw    : {raw_path}")
    print(f"  csv    : {reference_noise_csv or 'embedded case stats'}")
    print(f"  summary: {summary_path}")
    print_summary_table(summary_df)
    return 0


def execute_measurement(args, cases, test_args):
    stack = import_measurement_stack(args)
    setup_logging = stack["setup_logging"]
    client = stack["client"]
    BulkFileTransfer = stack["BulkFileTransfer"]
    tune_executor = stack["tune_executor"]
    instrument = stack["instrument"]
    transform_conv_output_to_3d = stack["transform_conv_output_to_3d"]
    ChipLock = stack["ChipLock"]
    init_gpu_rpc_if_requested(args)

    os.makedirs(os.path.join(args.out_dir, "logs"), exist_ok=True)
    setup_logging(os.path.join(args.out_dir, "logs"))

    if args.connection is None:
        raise ValueError("--connection is required unless --generate-only")
    conn = parse_connection(args.connection, args.keyfile)

    old_cwd = os.getcwd()
    chip_lock = None
    ssh_client = None
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

        if not args.dryrun and not args.transfer_only:
            chip_lock = ChipLock(ssh_client, script_name="measure_weight3_0_synthetic_noise.py")
            chip_lock.acquire()

        if args.force_board_reset and not args.dryrun:
            cmd = f"cd {REMOTE_TEST_ROOT} && make clear_time"
            status, stdout, stderr = client.run_remote_command_sshcli(ssh_client, cmd)
            out = stdout.read().decode("utf-8", errors="replace")
            err = stderr.read().decode("utf-8", errors="replace")
            print(f"Forced board reset marker clear: status={status}")
            if out.strip():
                print(out.strip())
            if err.strip():
                print(err.strip())
            if status != 0:
                raise RuntimeError(f"Failed to clear board reset markers with: {cmd}")

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
        summaries = build_summaries(cases, ref_stack, res_stack)

        return ref_stack, res_stack, summaries
    finally:
        if chip_lock is not None:
            chip_lock.release()
        if ssh_client is not None:
            ssh_client.close()
        os.chdir(old_cwd)


def main():
    args = parse_args()
    validate_args(args)
    if args.recompute_summary:
        return recompute_summary(args)

    apply_rpc_env_fallback(args)
    resolve_scan_val_path(args)
    os.makedirs(args.out_dir, exist_ok=True)
    npz_dir = os.path.join(args.out_dir, "synthetic_npz")
    os.makedirs(npz_dir, exist_ok=True)

    metadata, candidates = load_candidates(args.candidates, args.limit, args.candidate_id)
    reference_noise_csv = apply_reference_noise_csv(candidates, args.noise_csv)
    rng = np.random.default_rng(args.seed)

    cases = []
    for candidate in candidates:
        for pair_id in range(args.pairs_per_candidate):
            cases.append(make_synthetic_case(candidate, pair_id, rng, npz_dir))

    manifest_path = os.path.join(args.out_dir, "synthetic_cases_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "metadata": metadata,
            "shape": {
                "IC": IC,
                "OC": OC,
                "IH": IH,
                "IW": IW,
                "KH": KH,
                "KW": KW,
                "stride": STRIDE,
                "padding": PADDING,
                "OH": OH,
                "OW": OW,
            },
            "measurement": {
                "scan_val": args.scan_val,
                "reference_noise_csv": reference_noise_csv or metadata.get("noise_csv"),
                "dda": args.dda,
                "ddc": args.ddc,
                "ddl": args.ddl,
                "ddf": args.ddf,
                "adcmode": args.adcmode,
                "vmode": args.vmode,
                "multmode_set": args.multmode_set,
                "acc_mask": args.acc_mask,
                "pairs_per_candidate": args.pairs_per_candidate,
                "repeats": args.repeats,
                "seed": args.seed,
            },
            "cases": cases,
        }, f, indent=2, sort_keys=True)

    if args.generate_only:
        print(f"Generated {len(cases)} synthetic cases")
        print(f"  manifest: {manifest_path}")
        print(f"  npz dir : {npz_dir}")
        return 0

    enums = import_measurement_stack(args)
    test_args = [build_test_args(case, args, enums, i) for i, case in enumerate(cases)]
    args_path = os.path.join(args.out_dir, "synthetic_test_args.json")
    with open(args_path, "w") as f:
        printable = []
        for d in test_args:
            printable.append({k: str(v) if k in ("adcmode", "vmode", "multmode_set", "acc_mask") else v
                              for k, v in d.items()})
        json.dump(printable, f, indent=2, sort_keys=True)

    ref_stack, res_stack, summaries = execute_measurement(args, cases, test_args)
    if ref_stack is None:
        print(f"Generated/transferred {len(cases)} synthetic cases; no result collection requested.")
        print(f"  manifest: {manifest_path}")
        print(f"  args    : {args_path}")
        return 0

    raw_path = os.path.join(args.out_dir, "synthetic_measurements_raw.npz")
    np.savez_compressed(
        raw_path,
        ref=ref_stack,
        res=res_stack,
        diff=signed_int16(res_stack - ref_stack),
        cases_json=json.dumps(cases, sort_keys=True),
        candidates_json=json.dumps(candidates, sort_keys=True),
    )

    summary_df, summary_path = write_summary_csv(args, summaries)

    print(f"Measured {len(cases)} synthetic cases x {args.repeats} repeats")
    print(f"  raw    : {raw_path}")
    print(f"  summary: {summary_path}")
    print_summary_table(summary_df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
