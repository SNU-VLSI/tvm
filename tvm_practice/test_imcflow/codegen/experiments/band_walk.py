"""Rescue remaining band cells via graded approach (60MHz warmup, then 62.5 walk).

The 65.2 MHz (div23) band cells fail as SoC WEDGE at low unified voltage, so
per-cell retry can't help (a wedge needs a reboot). This tries, for each cell,
increasingly "gentle" approaches that were never tested on 65.2:
  strategy A: hold target V, run 60MHz (a guaranteed-PASS freq) to warm the
              IMC/linebuffer, THEN jump to the target freq.
  strategy B: after A's 60MHz warmup, step UP one divisor at a time through the
              band (60 -> 62.5 -> target) so the linebuffer/IMC never sees a
              cold cross-band jump.
Each cell: try A, if not PASS (and board still alive) try B; PASS -> next cell.
Board WEDGE at any point -> record and stop (needs reboot). Unified analog
(DDA=DDC=VDD). Restores V1 + 100MHz on exit.
CSV: vdd, mhz, analog, strategy, pulse_len, exp_len, perconv_nC, verdict.
"""
import csv, importlib.util, json, os, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rs", os.path.join(HERE, "rail_sweep.py"))
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
from ps_ctrl.rpc import RemotePowerSupplyManager

CODEGEN = rs.CODEGEN
DT = 21e-6
NCONV = 16128
TODO = os.environ.get("BAND_TODO", "/root/.claude/jobs/4343f03a/tmp/shmoo_uv_all_todo.json")
CELLS = [(round(v, 2), round(f, 1)) for v, f in json.load(open(TODO))]
# strategies to try per cell, in order (env override e.g. "B" or "A,B")
STRATS = tuple(os.environ.get("BAND_STRATS", "A,B").split(","))
# divisor-grid frequencies to walk up through (must include 60 and 62.5)
WALK = [60.0, 62.5, 65.2, 68.2, 75.0]

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
    out = os.path.join(CODEGEN, "experiments", f"band_walk_{date}.csv")
    f = open(out, "w", newline=""); w = csv.writer(f)
    w.writerow(["vdd", "mhz", "analog", "strategy", "pulse_len", "exp_len",
                "perconv_nC", "verdict"]); f.flush()
    def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

    mgr = RemotePowerSupplyManager("147.46.117.49", 1331,
        "/home/jihoonpark/measurement_utils/example/configs/ps_B2_config.json")

    def set_target_v(v):
        mgr.set_voltage("VDD", v); mgr.set_voltage("DDA", v); mgr.set_voltage("DDC", v)
        time.sleep(3)

    def attempt(v, mhz, strat, log):
        """Return ('PASS'|'FAIL'|'WEDGE', ln, q). Caller handles WEDGE stop."""
        set_target_v(v)
        if strat == "A":
            setf(60.0); time.sleep(2)
            if run_once() is None and not board_alive(): return "WEDGE", None, None
            setf(mhz); time.sleep(2)
        else:  # B: walk up 60 -> ... -> mhz
            for wf in [x for x in WALK if x <= mhz + 1e-6]:
                setf(wf); time.sleep(2)
                p = run_once()
                if p is None and not board_alive(): return "WEDGE", None, None
        p = run_once() or run_once()
        if p is None:
            return ("WEDGE" if not board_alive() else "FAIL"), None, None
        ok, ln, q = is_pass(p, mhz)
        return ("PASS" if ok else "FAIL"), ln, q

    try:
        for v, mhz in CELLS:
            log(f"===== cell {v}V/{mhz}MHz unified =====")
            got = False
            for strat in STRATS:
                if not baseline(mgr, log):
                    if not board_alive(): log("  baseline WEDGE -> stop"); return
                    if not baseline(mgr, log, deep=True): log("  unrecoverable -> stop"); return
                verdict, ln, q = attempt(v, mhz, strat, log)
                exp = 149.0 * 99.99 / mhz
                w.writerow([v, f"{mhz:.1f}", v, strat, ln or 0, f"{exp:.0f}",
                            f"{q:.2f}" if q else "", verdict]); f.flush()
                log(f"  strategy {strat} -> {verdict}" + (f" (len={ln}/{exp:.0f}, q={q:.2f})" if q else ""))
                if verdict == "WEDGE": log("  -> stop (reboot needed)"); return
                if verdict == "PASS": got = True; break
            if not got:
                log(f"  cell {v}V/{mhz}MHz: no PASS via A or B")
    finally:
        try: mgr.apply_preset("V1"); log("PS->V1")
        except Exception as e: log(f"PS restore fail: {e}")
        try: setf(100); log("clk->100")
        except Exception as e: log(f"clk skip: {e}")
        f.close(); log(f"CSV: {out}")

if __name__ == "__main__":
    main()
