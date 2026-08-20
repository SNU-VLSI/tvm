#!/usr/bin/env python3
"""Greedy combined-floor search at 50MHz.

Start from a VERIFIED combined point and take single 0.01V steps, one rail
at a time (axis order by measured mW/step gain). A failing step blocks that
axis; after any failure run the append-checked V1 fingerprint gate (50MHz
ref) and, if corrupted, a 100MHz rescan cycle. Stops when all axes are
blocked -> that point IS the combined floor. CSV-logged like rail_sweep.
"""
import csv, importlib.util, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rs", os.path.join(HERE, "rail_sweep.py"))
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
from ps_ctrl.rpc import RemotePowerSupplyManager  # path set by rail_sweep import

START = {"VDD": 0.81, "DDA": 0.87, "DDC": 0.81}   # L1-verified deepest combined point
AXES = ["DDC", "DDA"]                              # step-power gain order; VDD already at floor
FLOORS = {"DDC": 0.70, "DDA": 0.70}
STEP = 0.01
FP50_REF, FP50_TOL = 27.8, 1.0
CODEGEN = rs.CODEGEN


def sh_meas(script):
    return subprocess.run(rs.MEAS2 + [script], capture_output=True, text=True, timeout=90)


def board_alive():
    r = subprocess.run(rs.BOARD[:-1] + ["-o", "ConnectTimeout=8", rs.BOARD[-1], "echo ok"],
                       capture_output=True, text=True, timeout=20)
    return "ok" in r.stdout


def run_point(mgr, volts, log):
    for rail, v in volts.items():
        mgr.set_voltage(rail, v)
    time.sleep(3)
    before = rs.rec_linecount()
    ok, raw = rs.board_run()
    after = rs.rec_linecount()
    if not (ok and after > before):
        log(f"  FAIL at {volts} ({raw.strip()[:80]})")
        return None
    p = rs.extract_pulse()
    if not p or p["vdd"]["run_mA"] is None or p["vdd"]["len"] < 280:
        log(f"  INVALID pulse at {volts} (len={p and p['vdd']['len']})")
        return None
    return p


def fp_gate(mgr, log):
    """V1 fingerprint at 50MHz, append-checked with one transient retry."""
    mgr.apply_preset("V1"); time.sleep(3)
    for t in (1, 2):
        before = rs.rec_linecount()
        rs.board_run()
        if rs.rec_linecount() > before:
            p = rs.extract_pulse()
            d = p["ddc"]["run_mA"] - p["ddc"]["idle_mA"]
            log(f"  [gate] FP50 ddc_delta={d:.2f} len={p['ddc']['len']}")
            return abs(d - FP50_REF) < FP50_TOL and p["ddc"]["len"] > 280
        log(f"  [gate] no-append (try {t})")
    return False


def rescan_cycle(log):
    log("  [gate] corrupted -> 100MHz rescan cycle")
    subprocess.run(["bash", os.path.join(CODEGEN, "tools", "pl_freq.sh"), "set", "100"],
                   capture_output=True)
    subprocess.run(rs.BOARD + ["""
      B=/home/root/tvm/tvm_practice/test_imcflow/codegen
      cd /home/root/imcflow/xilinx/measurement && timeout -s INT 0.5s ./program_scan_reg $B/scan_gen/scan_reg_files
      cd /home/root/imcflow/xilinx/petalinux-csrc && make clear_time >/dev/null 2>&1 && make warmup >/dev/null 2>&1"""],
                   capture_output=True, timeout=180)
    subprocess.run(["bash", os.path.join(CODEGEN, "tools", "pl_freq.sh"), "set", "50"],
                   capture_output=True)


def main():
    date = time.strftime("%Y%m%d_%H%M")
    out_csv = os.path.join(CODEGEN, "experiments", f"greedy_corner_{date}.csv")
    f = open(out_csv, "w", newline=""); w = csv.writer(f)
    w.writerow(["step", "vdd", "dda", "ddc", "run_vdd_mA", "run_dda_mA", "run_ddc_mA",
                "pulse_len", "P_mW", "status"]); f.flush()

    def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

    subprocess.run(["bash", os.path.join(CODEGEN, "tools", "pl_freq.sh"), "set", "50"],
                   capture_output=True)
    mgr = RemotePowerSupplyManager("147.46.117.49", 1331,
        "/home/jihoonpark/measurement_utils/example/configs/ps_B2_config.json")

    cur = dict(START)
    blocked = set()
    best = None
    step_i = 0
    try:
        # verify start point first
        p = run_point(mgr, cur, log)
        if p is None:
            log("START point failed?! abort"); return
        P = sum(p[r.lower()]["run_mA"] * cur[r] for r in ("VDD", "DDA", "DDC")) / 1000.0
        best = (dict(cur), p, P)
        w.writerow([step_i, cur["VDD"], cur["DDA"], cur["DDC"],
                    p["vdd"]["run_mA"], p["dda"]["run_mA"], p["ddc"]["run_mA"],
                    p["vdd"]["len"], f"{P*1000:.1f}", "ok-start"]); f.flush()
        log(f"start OK {cur} P={P*1000:.1f}mW")

        while len(blocked) < len(AXES):
            moved = False
            for ax in AXES:
                if ax in blocked:
                    continue
                cand = dict(cur); cand[ax] = round(cand[ax] - STEP, 3)
                if cand[ax] < FLOORS[ax]:
                    blocked.add(ax); continue
                step_i += 1
                log(f"try {ax} -> {cand[ax]}")
                p = run_point(mgr, cand, log)
                if p is None:
                    w.writerow([step_i, cand["VDD"], cand["DDA"], cand["DDC"],
                                "", "", "", 0, "", f"fail-block-{ax}"]); f.flush()
                    blocked.add(ax)
                    if not board_alive():
                        log("BOARD DEAD -> stop"); return
                    if not fp_gate(mgr, log):
                        rescan_cycle(log)
                        if not fp_gate(mgr, log):
                            log("RESCAN failed -> stop"); return
                    # re-establish current point after gate (rails back to cur)
                    continue
                P = sum(p[r.lower()]["run_mA"] * cand[r] for r in ("VDD", "DDA", "DDC")) / 1000.0
                cur = cand; best = (dict(cur), p, P); moved = True
                w.writerow([step_i, cur["VDD"], cur["DDA"], cur["DDC"],
                            p["vdd"]["run_mA"], p["dda"]["run_mA"], p["ddc"]["run_mA"],
                            p["vdd"]["len"], f"{P*1000:.1f}", "ok"]); f.flush()
                log(f"  OK {cur} P={P*1000:.1f}mW")
            if not moved and len(blocked) >= len(AXES):
                break
    finally:
        mgr.apply_preset("V1")
        subprocess.run(["bash", os.path.join(CODEGEN, "tools", "pl_freq.sh"), "set", "100"],
                       capture_output=True)
        if best:
            c, p, P = best
            tops = 1.425
            log(f"COMBINED FLOOR: {c} P={P*1000:.1f}mW TOPS/W={tops/P:.1f}")
        log(f"CSV: {out_csv} (V1 + 100MHz restored)")
        f.close()


if __name__ == "__main__":
    main()
