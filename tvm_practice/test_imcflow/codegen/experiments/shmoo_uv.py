#!/usr/bin/env python3
"""Unified-voltage shmoo: VDD=DDA=DDC swept together, vs frequency (divisor grid).

For each voltage v (0.80..1.00, 0.01 step): set all three rails to v, then scan
the divisor-grid frequencies at/below that voltage's f_max ceiling (safe,
PASS-expected), one run each, gate on pulse-length + per-conversion charge.
Records PASS/FAIL per (v, f). Restores V1 + 100MHz on exit. WEDGE -> stop.

Frequency grid = PL0_REF divisors 30..12 (49.99..124.99 MHz), 71.42MHz(div21)
EXCLUDED (reproducible wedge, cause TBD).
"""
import csv, importlib.util, os, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rs", os.path.join(HERE, "rail_sweep.py"))
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
from ps_ctrl.rpc import RemotePowerSupplyManager

P = 1499.85
DIVS = [d for d in range(30, 11, -1) if d not in (21,22)]  # skip div21(71.4),div22(68.2) reproducible wedge
FREQS = [round(P / d, 1) for d in DIVS]            # high f = small div
DT = 21e-6
NCONV = 16128
CODEGEN = rs.CODEGEN

# f_max ceiling per unified voltage (conservative; refined as data comes in).
# Single-rail-all-low behaves like the digital VDD limit, so reuse the VDD f_max
# curve but stay one grid-step conservative at the low end.
def fmax_ceiling(v):
    if v >= 1.00: return 125.0
    if v >= 0.98: return 125.0
    if v >= 0.95: return 115.4
    if v >= 0.90: return 100.0
    if v >= 0.85: return 88.2
    return 62.5   # 0.80-0.84: unknown, cap low for safety

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

def main():
    date = time.strftime("%Y%m%d_%H%M")
    out = os.path.join(CODEGEN, "experiments", f"shmoo_uv_{date}.csv")
    f = open(out, "w", newline=""); w = csv.writer(f)
    w.writerow(["volt", "mhz", "pulse_len", "exp_len", "perconv_nC", "verdict"]); f.flush()
    def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

    mgr = RemotePowerSupplyManager("147.46.117.49", 1331,
        "/home/jihoonpark/measurement_utils/example/configs/ps_B2_config.json")
    try:
        import os as _os
        _hi=int(round(float(_os.environ.get("SHMOO_VHI","1.00"))*100))
        _lo=int(round(float(_os.environ.get("SHMOO_VLO","0.80"))*100))
        for vi in range(_hi, _lo-1, -1):
            v = round(vi / 100, 2)
            for r in ("VDD", "DDA", "DDC"):
                mgr.set_voltage(r, v)
            time.sleep(3)
            log(f"===== V={v} (all rails) =====")
            fm = fmax_ceiling(v)
            # wedge-prone band 60-72MHz (div25..21) widens as voltage drops:
            # 71.4 wedges at 1.0V, 68.2 at 0.91-0.92V, 62.5 at 0.86V. Below 0.87V
            # skip div24/23 (62.5, 65.2) too; div21/22 are globally excluded.
            _freqs = [x for x in FREQS if x <= fm + 1e-6]
            if v <= 0.87:
                _freqs = [x for x in _freqs if not (62.0 <= x <= 72.0)]
            for mhz in _freqs:
                setf(mhz); time.sleep(2)
                p = run_once() or run_once()
                exp = 149.0 * 99.99 / mhz
                if p is None:
                    w.writerow([v, f"{mhz:.1f}", 0, f"{exp:.0f}", "", "WEDGE"]); f.flush()
                    log(f"  {mhz:.1f}MHz WEDGE -> stop")
                    if not board_alive(): return
                    return
                lok = abs(p["vdd"]["len"] - exp) < 0.10 * exp
                q = per_conv_nC(p)
                verdict = "PASS" if (lok and q >= 2.0) else "FAIL"
                w.writerow([v, f"{mhz:.1f}", p["vdd"]["len"], f"{exp:.0f}",
                            f"{q:.2f}", verdict]); f.flush()
                log(f"  {mhz:.1f}MHz -> {verdict} (len={p['vdd']['len']}/{exp:.0f})")
    finally:
        try: mgr.apply_preset("V1"); log("PS->V1")
        except Exception as e: log(f"PS restore fail: {e}")
        try: setf(100); log("clk->100")
        except Exception as e: log(f"clk skip: {e}")
        f.close(); log(f"CSV: {out}")

if __name__ == "__main__":
    main()
