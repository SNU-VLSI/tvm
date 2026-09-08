#!/usr/bin/env python3
"""Band-anomaly hypothesis separation: analog level vs approach path.

For each band-FAIL cell (VDD, f), run a 2x2 of conditions:
  analog AN in {1.15, VDD-unified} x approach in {60MHz-run-first, direct-jump}
Every condition starts from a controlled baseline (V1 + 100MHz + one kernel
run) so history is uniform. 'approach' = after setting the target voltages,
run once at 60MHz (same-V adjacent PASS frequency), then switch to the band
frequency. 'direct' = jump straight from baseline to the band cell.

Interpretation: B2 (unified + approach) PASS -> warmup/approach hypothesis;
B2 FAIL but B1 (1.15 + approach) PASS -> analog level matters (with approach
as precondition); all FAIL -> neither rescues (marginal/flaky cell).

Kernel hang -> auto-recover (V1+100+warmup+fingerprint) and continue.
Board wedge -> stop. Restores V1 + 100MHz on exit.
CSV: vdd, mhz, analog, approach, stage, pulse_len, exp_len, perconv_nC, verdict.
"""
import csv, importlib.util, os, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rs", os.path.join(HERE, "rail_sweep.py"))
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
from ps_ctrl.rpc import RemotePowerSupplyManager

CODEGEN = rs.CODEGEN
DT = 21e-6
NCONV = 16128

CELLS = [(0.86, 62.5), (0.84, 62.5), (0.84, 75.0), (0.83, 62.5)]
# (tag, analog_or_None(None=unified=VDD), approach_via_60MHz)
CONDS = [("B1", 1.15, True), ("B2", None, True),
         ("A1", 1.15, False), ("A2", None, False)]

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

def gate(p, mhz):
    exp = 149.0 * 99.99 / mhz
    if p is None: return None, exp, None, None
    ln = p["vdd"]["len"]
    try: q = per_conv_nC(p)
    except TypeError: q = None
    lok = ln is not None and abs(ln - exp) < 0.10 * exp
    return ("PASS" if (lok and q is not None and q >= 2.0) else "FAIL"), exp, ln, q

def baseline(mgr, log, deep=False):
    """Reset to known-good V1/100MHz state; one clean run required."""
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
        if 44.9 < d < 47.7 and 140 < p["vdd"]["len"] < 160:
            log(f"    baseline ok (ddc={d:.2f} len={p['vdd']['len']})"); return True
    return False

def main():
    date = time.strftime("%Y%m%d_%H%M")
    out = os.path.join(CODEGEN, "experiments", f"band_ab_{date}.csv")
    f = open(out, "w", newline=""); w = csv.writer(f)
    w.writerow(["vdd", "mhz", "analog", "approach", "stage",
                "pulse_len", "exp_len", "perconv_nC", "verdict"]); f.flush()
    def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

    mgr = RemotePowerSupplyManager("147.46.117.49", 1331,
        "/home/jihoonpark/measurement_utils/example/configs/ps_B2_config.json")
    try:
        for v, mhz in CELLS:
            for tag, an_opt, approach in CONDS:
                an = an_opt if an_opt is not None else v
                log(f"===== {tag} cell {v}V/{mhz}MHz AN={an} approach={approach} =====")
                if not baseline(mgr, log):
                    if not board_alive(): log("  baseline WEDGE -> stop"); return
                    if not baseline(mgr, log, deep=True):
                        log("  baseline unrecoverable -> stop"); return
                mgr.set_voltage("VDD", v)
                mgr.set_voltage("DDA", an); mgr.set_voltage("DDC", an)
                time.sleep(3)
                if approach:
                    setf(60); time.sleep(2)
                    p = run_once()
                    verdict, exp, ln, q = gate(p, 59.99)
                    w.writerow([v, f"{mhz:.1f}", an, tag, "approach60",
                                ln or 0, f"{exp:.0f}", f"{q:.2f}" if q else "",
                                verdict or "HANG"]); f.flush()
                    log(f"  approach 60MHz -> {verdict or 'HANG'}")
                    if verdict is None:
                        if not board_alive(): log("  WEDGE -> stop"); return
                        continue  # approach itself hung; skip target this cond
                setf(mhz); time.sleep(2)
                p = run_once() or run_once()
                verdict, exp, ln, q = gate(p, mhz)
                if verdict is None:
                    alive = board_alive()
                    w.writerow([v, f"{mhz:.1f}", an, tag, "target", 0,
                                f"{exp:.0f}", "", "HANG" if alive else "WEDGE"]); f.flush()
                    if not alive: log(f"  target WEDGE -> stop"); return
                    log(f"  target -> HANG")
                else:
                    w.writerow([v, f"{mhz:.1f}", an, tag, "target",
                                ln or 0, f"{exp:.0f}", f"{q:.2f}" if q else "",
                                verdict]); f.flush()
                    log(f"  target {mhz}MHz -> {verdict} (len={ln}/{exp:.0f}, q={q})")
    finally:
        try: mgr.apply_preset("V1"); log("PS->V1")
        except Exception as e: log(f"PS restore fail: {e}")
        try: setf(100); log("clk->100")
        except Exception as e: log(f"clk skip: {e}")
        f.close(); log(f"CSV: {out}")

if __name__ == "__main__":
    main()
