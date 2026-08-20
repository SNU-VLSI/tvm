#!/usr/bin/env python3
"""VDD undervolt sweep for the 16-imce 127x127 power kernel.

Per point: set VDD (DDA/DDC stay at preset V1) -> settle -> measure rail
voltages -> one board run (DMM-bracketed kernel) -> pulse-extract idle/run
currents for VDD/DDA/DDC from the measurement2 per-sample records -> CSV row.

Failure policy (chip-wedge safety): a failing point is retried once; a second
failure records the point, STOPS the downward sweep, restores preset V1 and
runs a baseline sanity run. The PS is ALWAYS restored to V1 on exit.

Usage:  python3 experiments/vdd_sweep.py [--start 1.00] [--stop 0.70]
            [--step 0.01] [--out experiments/vdd_sweep_<date>.csv]
"""
import argparse
import csv
import datetime
import json
import os
import re
import statistics
import subprocess
import sys
import time

sys.path.insert(0, "/root/project/tvm/3rdparty/measurement_utils")
from ps_ctrl.rpc import RemotePowerSupplyManager  # noqa: E402

CODEGEN = "/root/project/tvm/.claude/worktrees/step-freerun-interleave/tvm_practice/test_imcflow/codegen"
MODEL_EVL = "one_1x1_quant_16imce_max127_evl.linux.bugfixoff"
BOARD = ["ssh", "-o", "BatchMode=yes", "-p", "1326", "root@147.46.117.99"]
MEAS2 = ["ssh", "-o", "BatchMode=yes", "measurement2"]
PS = ("147.46.117.49", 1331,
      "/home/jihoonpark/measurement_utils/example/configs/ps_B2_config.json")
REC_BASE = ("/home/jihoonpark/tvmgen_default_tvmgen_default_imcflow_main_0_"
            "round_imcflow_region1_main_0_tile0_{}_server.txt")
RAILS = ["vdd", "dda", "ddc"]

RUN_CMD = r"""
export DMM_BRIDGE_HOST=147.46.117.49 DMM_BRIDGE_PORT=9900
B=/home/root/tvm/tvm_practice/test_imcflow/codegen
cd $B/eval_dir/{evl}/host_binary_make/build
timeout 150 ./debug_execute_graph eval_dir/{evl} "$B" \
  mlf/executor-config/graph/default.graph mlf/parameters/default.params . 2>&1 | grep -E "\[DMM\] \[|POLLING ERROR"
""".format(evl=MODEL_EVL)

PULSE_PY = r"""
import re, statistics, json
out={}
for r in %s:
    f=%r.format(r)
    lines=open(f).readlines()
    v=[float(x) for x in re.findall(r"[-+0-9.eE]+", lines[-1])]
    idle=statistics.mean(v[:2000]); peak=max(v)
    thr=idle+0.4*(peak-idle)
    above=[i for i,x in enumerate(v) if x>thr]
    if not above or (peak-idle)<2e-4:
        out[r]={"idle_mA":idle*1e3,"run_mA":None,"len":0}
        continue
    i0,i1=above[0],above[-1]
    seg=v[i0:i1+1]
    out[r]={"idle_mA":idle*1e3,"run_mA":statistics.mean(seg)*1e3,
            "len":len(seg),"cov":len(above)/len(seg)}
print(json.dumps(out))
""" % (RAILS, REC_BASE)


def sh(cmd_list, script, timeout):
    return subprocess.run(cmd_list + [script], capture_output=True, text=True,
                          timeout=timeout)


def rec_linecount():
    r = sh(MEAS2, f"wc -l < {REC_BASE.format('vdd')}", 30)
    return int(r.stdout.strip() or 0)


def board_run():
    """One kernel run; returns (ok, raw_output)."""
    try:
        r = sh(BOARD, RUN_CMD, 200)
    except subprocess.TimeoutExpired:
        return False, "SSH-TIMEOUT"
    out = r.stdout + r.stderr
    ok = out.count("[DMM] [") >= 3 and "POLLING ERROR" not in out.replace(
        "POLLING ERROR 0", "")
    return ok, out


def extract_pulse():
    r = sh(MEAS2, "~/anaconda3/envs/meas/bin/python - <<'PYEOF'\n" + PULSE_PY + "\nPYEOF", 60)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=1.00)
    ap.add_argument("--stop", type=float, default=0.70)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--settle", type=float, default=3.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    date = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_csv = args.out or os.path.join(CODEGEN, "experiments", f"vdd_sweep_{date}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    git_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=CODEGEN,
                              capture_output=True, text=True).stdout.strip()
    md5 = subprocess.run(
        ["md5sum", f"{CODEGEN}/eval_dir/{MODEL_EVL}/host_binary_make/build/debug_execute_graph"],
        capture_output=True, text=True).stdout.split()[0]
    meta = {
        "date": date, "git_hash": git_hash, "model_evl": MODEL_EVL,
        "binary_md5": md5,
        "levers": "INPUT_REUSE nop0 flagfree pace0 drop-all feed_unroll4 load_unroll "
                  "ADCMODE=FOUR MULTMODE=Q1 BUGFIX=off MMIO_BARRIER=100 DMM_SAMPLES=20000",
        "fixed": "DDA=1.150 DDC=1.145 (preset V1)",
        "sweep": f"VDD {args.start}->{args.stop} step {args.step}",
        "expected_pulse_ms": 3.13,
    }
    with open(out_csv.replace(".csv", ".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    mgr = RemotePowerSupplyManager(*PS)
    cols = ["idx", "time", "vdd_set", "vdd_meas", "dda_meas", "ddc_meas",
            "idle_vdd_mA", "idle_dda_mA", "idle_ddc_mA",
            "run_vdd_mA", "run_dda_mA", "run_ddc_mA",
            "pulse_len", "status", "git_hash"]
    fcsv = open(out_csv, "w", newline="")
    w = csv.writer(fcsv); w.writerow(cols); fcsv.flush()

    def log(*a):
        print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

    n_steps = int(round((args.start - args.stop) / args.step)) + 1
    points = [round(args.start - i * args.step, 3) for i in range(n_steps)]
    stopped_reason = "completed"
    try:
        for idx, v in enumerate(points):
            mgr.set_voltage("VDD", v)
            time.sleep(args.settle)
            meas = {r: float(mgr.meas_voltage(r.upper())) for r in RAILS}
            status = "ok"; pulse = None
            for attempt in (1, 2):
                before = rec_linecount()
                ok, raw = board_run()
                after = rec_linecount()
                if ok and after > before:
                    pulse = extract_pulse()
                    if pulse and pulse["vdd"]["run_mA"] is not None:
                        break
                    status = f"no-pulse(attempt{attempt})"
                else:
                    status = f"run-fail(attempt{attempt}):{raw.strip()[:120]}"
                log(f"VDD={v}: attempt {attempt} failed -> {status}")
            row_p = pulse or {r: {"idle_mA": None, "run_mA": None, "len": 0} for r in RAILS}
            w.writerow([idx, time.strftime("%H:%M:%S"), v,
                        meas["vdd"], meas["dda"], meas["ddc"],
                        *(row_p[r]["idle_mA"] for r in RAILS),
                        *(row_p[r]["run_mA"] for r in RAILS),
                        row_p["vdd"]["len"], status, git_hash])
            fcsv.flush()
            if pulse and pulse["vdd"]["run_mA"] is not None:
                log(f"VDD={v}: run VDD/DDA/DDC = "
                    f"{row_p['vdd']['run_mA']:.2f}/{row_p['dda']['run_mA']:.2f}/"
                    f"{row_p['ddc']['run_mA']:.2f} mA (pulse {row_p['vdd']['len']})")
            else:
                stopped_reason = f"failed at VDD={v}"
                log(f"VDD={v}: FAILED twice -> stop sweep (record kept)")
                break
    finally:
        mgr.apply_preset("V1")
        log("preset V1 restored")
        # baseline sanity: one run at V1 to prove the chip is healthy
        time.sleep(3)
        ok, raw = board_run()
        log(f"baseline sanity after restore: {'OK' if ok else 'FAIL'}")
        w.writerow(["sanity", time.strftime("%H:%M:%S"), 1.0, "", "", "",
                    "", "", "", "", "", "", "", f"baseline-{'ok' if ok else 'FAIL'}",
                    git_hash])
        fcsv.close()
        log(f"CSV: {out_csv} | reason: {stopped_reason}")


if __name__ == "__main__":
    main()
