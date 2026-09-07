#!/usr/bin/env python3
"""Verify specific (VDD, freq) shmoo cells with one run each.

For each cell: set VDD (analog held at V1 headroom -> safe, these are all
inferred-PASS below f_max), set clock, one run, gate on pulse-length +
per-conversion charge. Appends PASS/FAIL to a CSV. Restores V1+100MHz on exit.
"""
import csv, importlib.util, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rs", os.path.join(HERE, "rail_sweep.py"))
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
from ps_ctrl.rpc import RemotePowerSupplyManager

CODEGEN = rs.CODEGEN
DT = 21e-6
NCONV = 16128

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

# (VDD, freq_MHz) cells to verify -- loaded from json (safe PASS-expected)
import json as _json
_todo = _json.load(open("/root/.claude/jobs/4343f03a/tmp/shmoo_div_remain.json"))
# sort by VDD desc, then freq asc (minimize voltage changes; run low-f-safe order)
CELLS = sorted([(round(v,2), round(f,1)) for v, f in _todo],
               key=lambda t: (-t[0], t[1]))

def main():
    date = time.strftime("%Y%m%d_%H%M")
    out = os.path.join(CODEGEN, "experiments", f"verify_cells_{date}.csv")
    f = open(out, "w", newline=""); w = csv.writer(f)
    w.writerow(["vdd", "mhz", "analog", "pulse_len", "exp_len", "perconv_nC",
                "verdict"]); f.flush()
    def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

    mgr = RemotePowerSupplyManager("147.46.117.49", 1331,
        "/home/jihoonpark/measurement_utils/example/configs/ps_B2_config.json")
    AN = 1.15  # analog V1 headroom (safe for inferred-PASS cells)
    try:
        cur_v = None
        for v, mhz in CELLS:
            if v != cur_v:
                mgr.set_voltage("DDA", AN); mgr.set_voltage("DDC", AN)
                mgr.set_voltage("VDD", v); cur_v = v; time.sleep(3)
                log(f"===== VDD={v} (analog {AN}) =====")
            setf(mhz); time.sleep(2)
            p = run_once() or run_once()
            exp = 149.0 * 99.99 / mhz
            if p is None:
                verdict = "RUN-FAIL"
                if not board_alive():
                    w.writerow([v, f"{mhz:.1f}", AN, 0, f"{exp:.0f}", "", "WEDGE"]); f.flush()
                    log(f"  {mhz:.1f}MHz WEDGE -> stop"); return
            else:
                lok = abs(p["vdd"]["len"] - exp) < 0.10 * exp
                q = per_conv_nC(p)
                verdict = "PASS" if (lok and q >= 2.0) else f"FAIL(len={lok},q={q:.1f})"
            w.writerow([v, f"{mhz:.1f}", AN, p["vdd"]["len"] if p else 0,
                        f"{exp:.0f}", f"{q:.2f}" if p else "",
                        verdict]); f.flush()
            log(f"  {mhz:.1f}MHz -> {verdict}"
                + (f" (len={p['vdd']['len']}/{exp:.0f})" if p else ""))
    finally:
        try: mgr.apply_preset("V1"); log("PS->V1")
        except Exception as e: log(f"PS restore fail: {e}")
        try: setf(100); log("clk->100")
        except Exception as e: log(f"clk restore skip: {e}")
        f.close(); log(f"CSV: {out}")

if __name__ == "__main__":
    main()
