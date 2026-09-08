#!/usr/bin/env python3
"""Band-anomaly flakiness quantification: repeat the worst band cells N times.

For each target cell (VDD, f) at UNIFIED analog (DDA=DDC=VDD) with a DIRECT
jump from the V1/100MHz baseline (the exact condition that FAIL/WEDGE'd in the
shmoo sweep), run N trials and count PASS / FAIL(hang) / WEDGE. A nonzero but
<100% fail rate proves these are marginal/flaky cells, not deterministic
digital-fmax failures. Each trial resets to baseline first so trials are
independent. Kernel hang -> recover and continue; board wedge -> record and
stop (needs reboot). Restores V1 + 100MHz on exit.
CSV: vdd, mhz, analog, trial, pulse_len, exp_len, perconv_nC, verdict.
"""
import csv, importlib.util, os, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rs", os.path.join(HERE, "rail_sweep.py"))
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
from ps_ctrl.rpc import RemotePowerSupplyManager

CODEGEN = rs.CODEGEN
DT = 21e-6
NCONV = 16128
N_TRIALS = 20
CELLS = [(0.83, 62.5), (0.84, 65.2), (0.81, 62.5)]

def setf(mhz): subprocess.run(["bash", os.path.join(CODEGEN, "tools/pl_freq.sh"),
                               "set", str(int(round(mhz)))], capture_output=True)
def board_alive():
    r = subprocess.run(rs.BOARD[:-1] + ["-o", "ConnectTimeout=8", rs.BOARD[-1], "echo ok"],
                       capture_output=True, text=True, timeout=20)
    return "ok" in r.stdout
def run_once():
    b = rs.rec_linecount(); rs.board_run()
    return rs.extract_pulse() if rs.rec_linecount() > b else None
def per_conv_nC(p): return (p["dda"]["run_mA"] - p["dda"]["idle_mA"]) * p["dda"]["len"] * DT / NCONV * 1e6

def baseline(mgr, log, deep=False):
    mgr.apply_preset("V1"); setf(100); time.sleep(3)
    if deep:
        subprocess.run(rs.BOARD + ["pkill -f debug_execute_graph; "
                       "cd /home/root/imcflow/xilinx/petalinux-csrc && "
                       "make clear_time >/dev/null 2>&1 && make warmup >/dev/null 2>&1"],
                       capture_output=True, timeout=180)
    for t in (1, 2, 3):
        p = run_once()
        if p is None: continue
        d = p["ddc"]["run_mA"] - p["ddc"]["idle_mA"]
        if 44.9 < d < 47.7 and 140 < p["vdd"]["len"] < 160: return True
    return False

def main():
    date = time.strftime("%Y%m%d_%H%M")
    out = os.path.join(CODEGEN, "experiments", f"band_flaky_{date}.csv")
    f = open(out, "w", newline=""); w = csv.writer(f)
    w.writerow(["vdd", "mhz", "analog", "trial", "pulse_len", "exp_len",
                "perconv_nC", "verdict"]); f.flush()
    def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

    mgr = RemotePowerSupplyManager("147.46.117.49", 1331,
        "/home/jihoonpark/measurement_utils/example/configs/ps_B2_config.json")
    tally = {}
    try:
        for v, mhz in CELLS:
            tally[(v, mhz)] = {"PASS": 0, "FAIL": 0, "HANG": 0, "WEDGE": 0}
            log(f"===== cell {v}V/{mhz}MHz unified, direct, {N_TRIALS} trials =====")
            for i in range(1, N_TRIALS + 1):
                if not baseline(mgr, log):
                    if not board_alive(): log("  baseline WEDGE -> stop"); return
                    if not baseline(mgr, log, deep=True):
                        log("  baseline unrecoverable -> stop"); return
                mgr.set_voltage("VDD", v)
                mgr.set_voltage("DDA", v); mgr.set_voltage("DDC", v)
                time.sleep(3); setf(mhz); time.sleep(2)
                p = run_once() or run_once()
                exp = 149.0 * 99.99 / mhz
                if p is None:
                    alive = board_alive()
                    verdict = "HANG" if alive else "WEDGE"
                    w.writerow([v, f"{mhz:.1f}", v, i, 0, f"{exp:.0f}", "", verdict]); f.flush()
                    tally[(v, mhz)][verdict] += 1
                    log(f"  trial {i}: {verdict}")
                    if not alive: return
                    if not baseline(mgr, log, deep=True):
                        log("  recovery failed -> stop"); return
                    continue
                ln = p["vdd"]["len"]
                try: q = per_conv_nC(p)
                except TypeError: q = None
                lok = ln is not None and abs(ln - exp) < 0.10 * exp
                verdict = "PASS" if (lok and q is not None and q >= 2.0) else "FAIL"
                w.writerow([v, f"{mhz:.1f}", v, i, ln or 0, f"{exp:.0f}",
                            f"{q:.2f}" if q else "", verdict]); f.flush()
                tally[(v, mhz)][verdict] += 1
                log(f"  trial {i}: {verdict} (len={ln}/{exp:.0f}, q={q})")
    finally:
        try: mgr.apply_preset("V1"); log("PS->V1")
        except Exception as e: log(f"PS restore fail: {e}")
        try: setf(100); log("clk->100")
        except Exception as e: log(f"clk skip: {e}")
        f.close()
        for (v, mhz), t in tally.items():
            log(f"TALLY {v}V/{mhz}MHz: {t}")
        log(f"CSV: {out}")

if __name__ == "__main__":
    main()
