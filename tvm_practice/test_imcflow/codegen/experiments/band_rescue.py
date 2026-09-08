#!/usr/bin/env python3
"""Band-anomaly rescue: can raising DDA=DDC make the 62-75MHz FAIL cells pass?

For each unified-voltage band-FAIL cell (VDD, f): hold VDD at the unified
value, ladder the analog rails DOWN from 1.15V in 0.05 steps, one run per
point. First point (1.15) PASS proves analog-rescue; the last PASS before
the first FAIL is the analog threshold. After the first FAIL move to the
next cell (kernel hang -> auto-recover; board wedge -> stop). Restores
V1 + 100MHz on exit. CSV: vdd, mhz, analog, pulse_len, exp_len, nC, verdict.
"""
import csv, importlib.util, os, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rs", os.path.join(HERE, "rail_sweep.py"))
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
from ps_ctrl.rpc import RemotePowerSupplyManager

CODEGEN = rs.CODEGEN
DT = 21e-6
NCONV = 16128

# unified-voltage band FAIL cells (measured), high V first
CELLS = [(0.86, 62.5), (0.85, 65.2), (0.84, 62.5), (0.84, 65.2), (0.84, 75.0),
         (0.83, 62.5), (0.83, 65.2), (0.82, 62.5), (0.81, 62.5), (0.81, 65.2)]
AN_LADDER = [1.15, 1.10, 1.05, 1.00, 0.95, 0.90]

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

def recover(mgr, log):
    mgr.apply_preset("V1"); setf(100); time.sleep(3)
    subprocess.run(rs.BOARD + ["pkill -f debug_execute_graph; "
                   "cd /home/root/imcflow/xilinx/petalinux-csrc && "
                   "make clear_time >/dev/null 2>&1 && make warmup >/dev/null 2>&1"],
                   capture_output=True, timeout=180)
    for t in (1, 2, 3):
        p = run_once()
        if p is None: continue
        d = p["ddc"]["run_mA"] - p["ddc"]["idle_mA"]
        log(f"    recover fp try{t}: ddc={d:.2f}mA len={p['vdd']['len']}")
        if 44.9 < d < 47.7 and 140 < p["vdd"]["len"] < 160: return True
    return False

def main():
    date = time.strftime("%Y%m%d_%H%M")
    out = os.path.join(CODEGEN, "experiments", f"band_rescue_{date}.csv")
    f = open(out, "w", newline=""); w = csv.writer(f)
    w.writerow(["vdd", "mhz", "analog", "pulse_len", "exp_len", "perconv_nC", "verdict"]); f.flush()
    def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

    mgr = RemotePowerSupplyManager("147.46.117.49", 1331,
        "/home/jihoonpark/measurement_utils/example/configs/ps_B2_config.json")
    try:
        for v, mhz in CELLS:
            log(f"===== cell VDD={v} f={mhz}MHz =====")
            mgr.set_voltage("VDD", v)
            setf(mhz)
            for an in AN_LADDER:
                if an <= v: break
                mgr.set_voltage("DDA", an); mgr.set_voltage("DDC", an)
                time.sleep(3)
                p = run_once() or run_once()
                exp = 149.0 * 99.99 / mhz
                if p is None:
                    alive = board_alive()
                    w.writerow([v, f"{mhz:.1f}", an, 0, f"{exp:.0f}", "",
                                "FAIL-HANG" if alive else "WEDGE"]); f.flush()
                    if not alive:
                        log(f"  AN={an} WEDGE -> stop"); return
                    log(f"  AN={an} FAIL(hang) -> recover, next cell")
                    if not recover(mgr, log): log("  recovery failed -> stop"); return
                    break
                ln = p["vdd"]["len"]
                try: q = per_conv_nC(p)
                except TypeError: q = None
                lok = ln is not None and abs(ln - exp) < 0.10 * exp
                verdict = "PASS" if (lok and q is not None and q >= 2.0) else "FAIL"
                w.writerow([v, f"{mhz:.1f}", an, ln or 0, f"{exp:.0f}",
                            f"{q:.2f}" if q is not None else "", verdict]); f.flush()
                log(f"  AN={an} -> {verdict} (len={ln}/{exp:.0f}, q={q})")
                if verdict == "FAIL": break   # threshold found, next cell
    finally:
        try: mgr.apply_preset("V1"); log("PS->V1")
        except Exception as e: log(f"PS restore fail: {e}")
        try: setf(100); log("clk->100")
        except Exception as e: log(f"clk skip: {e}")
        f.close(); log(f"CSV: {out}")

if __name__ == "__main__":
    main()
