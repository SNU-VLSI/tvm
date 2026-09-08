"""Rescue 65.2 MHz band cells via gradual voltage descent (no baseline reset).

The 65.2 MHz (div23) cells wedge when jumped into cold from a V1/100 baseline
(both a 60MHz-warmup and a divisor-walk approach wedged). This instead starts
at a KNOWN-PASS high voltage with the clock already at 65.2 MHz, confirms PASS,
then lowers VDD=DDA=DDC one 0.01 V step at a time WITHOUT resetting the clock
or preset between steps -- the accelerator stays live in-band the whole time,
so each step is a tiny perturbation rather than a cold cross-band jump.

Records PASS/FAIL per (V, 65.2). A hang (board alive, no record) is a FAIL for
that step; the descent continues from the next lower voltage after a light
re-warm. A board WEDGE (SSH dead) stops the run (needs reboot). Starts from
V_START (default 0.90, a measured PASS) down to V_STOP (default 0.81).
Restores V1 + 100MHz on exit.
CSV: volt, mhz, pulse_len, exp_len, perconv_nC, verdict.
"""
import csv, importlib.util, os, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rs", os.path.join(HERE, "rail_sweep.py"))
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
from ps_ctrl.rpc import RemotePowerSupplyManager

CODEGEN = rs.CODEGEN
DT = 21e-6
NCONV = 16128
MHZ = 65.2
V_START = float(os.environ.get("BAND_VSTART", "0.90"))
V_STOP = float(os.environ.get("BAND_VSTOP", "0.81"))

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
def is_pass(p, mhz):
    if p is None: return False, None, None
    ln = p["vdd"]["len"]; exp = 149.0 * 99.99 / mhz
    try: q = per_conv_nC(p)
    except TypeError: q = None
    lok = ln is not None and abs(ln - exp) < 0.10 * exp
    return (lok and q is not None and q >= 2.0), ln, q
def setv(mgr, v):
    for r in ("VDD", "DDA", "DDC"): mgr.set_voltage(r, v)

def main():
    date = time.strftime("%Y%m%d_%H%M")
    out = os.path.join(CODEGEN, "experiments", f"band_descend_{date}.csv")
    f = open(out, "w", newline=""); w = csv.writer(f)
    w.writerow(["volt", "mhz", "pulse_len", "exp_len", "perconv_nC", "verdict"]); f.flush()
    def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

    mgr = RemotePowerSupplyManager("147.46.117.49", 1331,
        "/home/jihoonpark/measurement_utils/example/configs/ps_B2_config.json")
    exp = 149.0 * 99.99 / MHZ
    try:
        # start at a known-PASS voltage WITH the clock already at 65.2
        setv(mgr, V_START); time.sleep(3)
        setf(MHZ); time.sleep(2)
        p = run_once() or run_once()
        ok, ln, q = is_pass(p, MHZ)
        w.writerow([f"{V_START:.2f}", f"{MHZ:.1f}", ln or 0, f"{exp:.0f}",
                    f"{q:.2f}" if q else "", "PASS" if ok else ("FAIL" if p is not None else "WEDGE")]); f.flush()
        log(f"START {V_START:.2f}V @ {MHZ}MHz -> {'PASS' if ok else 'FAIL/NO-REC'} (len={ln}, q={q})")
        if not ok:
            if p is None and not board_alive(): log("start WEDGE -> stop"); return
            log("start not PASS; continuing descent anyway")

        vi = int(round(V_START * 100)) - 1
        vlo = int(round(V_STOP * 100))
        while vi >= vlo:
            v = round(vi / 100, 2)
            setv(mgr, v); time.sleep(2)          # clock stays at 65.2, no reset
            p = run_once() or run_once()
            if p is None:
                alive = board_alive()
                w.writerow([f"{v:.2f}", f"{MHZ:.1f}", 0, f"{exp:.0f}", "",
                            "WEDGE" if not alive else "FAIL"]); f.flush()
                if not alive:
                    log(f"  {v:.2f}V -> WEDGE -> stop"); return
                log(f"  {v:.2f}V -> FAIL(hang); re-warm, keep descending")
                # light re-warm at this voltage in-band, then continue
                setf(60.0); time.sleep(2); run_once(); setf(MHZ); time.sleep(2)
                vi -= 1; continue
            ok, ln, q = is_pass(p, MHZ)
            w.writerow([f"{v:.2f}", f"{MHZ:.1f}", ln or 0, f"{exp:.0f}",
                        f"{q:.2f}" if q else "", "PASS" if ok else "FAIL"]); f.flush()
            log(f"  {v:.2f}V -> {'PASS' if ok else 'FAIL'} (len={ln}/{exp:.0f}, q={q})")
            vi -= 1
    finally:
        try: mgr.apply_preset("V1"); log("PS->V1")
        except Exception as e: log(f"PS restore fail: {e}")
        try: setf(100); log("clk->100")
        except Exception as e: log(f"clk skip: {e}")
        f.close(); log(f"CSV: {out}")

if __name__ == "__main__":
    main()
