"""Re-measure remaining band FAIL cells, retrying until PASS (flaky rescue).

The 62-75 MHz band FAILs proved to be flaky/marginal (band_ab: 16/16 PASS on
re-run). This retries each remaining band-FAIL cell at UNIFIED analog
(DDA=DDC=VDD) with a DIRECT jump from the V1/100MHz baseline, up to MAX_RETRY
times, stopping that cell as soon as it PASSes. Records the outcome per cell.

Kernel hang -> recover (V1+100+warmup+fingerprint), counts as one failed try,
retry. Board wedge -> record WEDGE and stop (needs reboot). Each try resets to
baseline so tries are independent. Restores V1 + 100MHz on exit.
CSV: vdd, mhz, analog, tries, pulse_len, exp_len, perconv_nC, verdict.
"""
import csv, importlib.util, json, os, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rs", os.path.join(HERE, "rail_sweep.py"))
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
from ps_ctrl.rpc import RemotePowerSupplyManager

CODEGEN = rs.CODEGEN
DT = 21e-6
NCONV = 16128
MAX_RETRY = 8
TODO = os.environ.get("BAND_TODO", "/root/.claude/jobs/4343f03a/tmp/shmoo_uv_all_todo.json")
CELLS = [(round(v, 2), round(f, 1)) for v, f in json.load(open(TODO))]

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
    out = os.path.join(CODEGEN, "experiments", f"band_retry_{date}.csv")
    f = open(out, "w", newline=""); w = csv.writer(f)
    w.writerow(["vdd", "mhz", "analog", "tries", "pulse_len", "exp_len",
                "perconv_nC", "verdict"]); f.flush()
    def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

    mgr = RemotePowerSupplyManager("147.46.117.49", 1331,
        "/home/jihoonpark/measurement_utils/example/configs/ps_B2_config.json")
    try:
        for v, mhz in CELLS:
            log(f"===== cell {v}V/{mhz}MHz unified, direct, up to {MAX_RETRY} tries =====")
            exp = 149.0 * 99.99 / mhz
            got = False
            for i in range(1, MAX_RETRY + 1):
                if not baseline(mgr, log):
                    if not board_alive(): log("  baseline WEDGE -> stop"); return
                    if not baseline(mgr, log, deep=True):
                        log("  baseline unrecoverable -> stop"); return
                mgr.set_voltage("VDD", v)
                mgr.set_voltage("DDA", v); mgr.set_voltage("DDC", v)
                time.sleep(3); setf(mhz); time.sleep(2)
                p = run_once() or run_once()
                if p is None:
                    if not board_alive():
                        w.writerow([v, f"{mhz:.1f}", v, i, 0, f"{exp:.0f}", "", "WEDGE"]); f.flush()
                        log(f"  try {i}: WEDGE -> stop"); return
                    log(f"  try {i}: HANG -> recover, retry")
                    if not baseline(mgr, log, deep=True):
                        log("  recovery failed -> stop"); return
                    continue
                ln = p["vdd"]["len"]
                try: q = per_conv_nC(p)
                except TypeError: q = None
                lok = ln is not None and abs(ln - exp) < 0.10 * exp
                if lok and q is not None and q >= 2.0:
                    w.writerow([v, f"{mhz:.1f}", v, i, ln, f"{exp:.0f}",
                                f"{q:.2f}", "PASS"]); f.flush()
                    log(f"  try {i}: PASS (len={ln}/{exp:.0f}, q={q:.2f})")
                    got = True; break
                log(f"  try {i}: FAIL (len={ln}/{exp:.0f}, q={q})")
            if not got:
                w.writerow([v, f"{mhz:.1f}", v, MAX_RETRY, 0, f"{exp:.0f}", "",
                            "FAIL"]); f.flush()
                log(f"  cell {v}V/{mhz}MHz: no PASS in {MAX_RETRY} tries")
    finally:
        try: mgr.apply_preset("V1"); log("PS->V1")
        except Exception as e: log(f"PS restore fail: {e}")
        try: setf(100); log("clk->100")
        except Exception as e: log(f"clk skip: {e}")
        f.close(); log(f"CSV: {out}")

if __name__ == "__main__":
    main()
