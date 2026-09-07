#!/usr/bin/env python3
"""Unified-voltage shmoo, 2nd pass: measure an explicit (v, f) cell list.

Same gate as shmoo_uv.py (pulse-length within 10% + per-conversion charge
>= 2.0 nC) but iterates a JSON cell list instead of the below-ceiling grid,
so it can probe the above-f_max region for real measured FAILs. All three
rails (VDD=DDA=DDC) set to v. FAIL -> continue (graceful). No record but
board alive = kernel HANG (functional fail, not SoC wedge): record FAIL,
recover (V1 + 100MHz + warmup + fingerprint), continue. Board unreachable =
WEDGE: record and stop (needs reboot). Restores V1 + 100MHz on exit.

Cell list: env SHMOO_CELLS (json path), default shmoo_uv_all_todo.json.
"""
import csv, importlib.util, json, os, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rs", os.path.join(HERE, "rail_sweep.py"))
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
from ps_ctrl.rpc import RemotePowerSupplyManager

CODEGEN = rs.CODEGEN
DT = 21e-6
NCONV = 16128
CELLS = [(round(v, 2), round(f, 1)) for v, f in json.load(open(
    os.environ.get("SHMOO_CELLS", "/root/.claude/jobs/4343f03a/tmp/shmoo_uv_all_todo.json")))]

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

def wedge_reboot_recover(mgr, log):
    """WEDGE + TAPO_AUTO_REBOOT=1: power-cycle the board plug via tapo_ctl,
    wait for SSH, then full re-normalize (clk 100 + scan + warmup + V1 +
    fingerprint). Returns True when the board is back and fingerprint-clean."""
    mgr.apply_preset("V1")
    r = subprocess.run(["python3", os.path.join(CODEGEN, "tools/tapo_ctl.py"), "reboot"],
                       capture_output=True, text=True, timeout=120)
    log(f"  tapo reboot: rc={r.returncode} {r.stdout.strip()} {r.stderr.strip()}")
    if r.returncode != 0: return False
    deadline = time.time() + 420
    while time.time() < deadline:
        if board_alive(): break
        time.sleep(15)
    else:
        log("  board did not come back within 7min"); return False
    time.sleep(20)  # let services settle
    setf(100)
    subprocess.run(rs.BOARD + ["B=/home/root/tvm/tvm_practice/test_imcflow/codegen; "
                   "cd /home/root/imcflow/xilinx/measurement && "
                   "timeout -s INT 0.5s ./program_scan_reg $B/scan_gen/scan_reg_files; "
                   "cd /home/root/imcflow/xilinx/petalinux-csrc && "
                   "make clear_time >/dev/null 2>&1 && make warmup >/dev/null 2>&1"],
                   capture_output=True, timeout=300)
    for t in (1, 2, 3, 4):
        p = run_once()
        if p is None: continue
        d = p["ddc"]["run_mA"] - p["ddc"]["idle_mA"]
        log(f"  post-reboot fp try{t}: ddc={d:.2f}mA len={p['vdd']['len']}")
        if 44.9 < d < 47.7 and 140 < p["vdd"]["len"] < 160: return True
    return False

def recover(mgr, log):
    """After a kernel hang (board alive): safe V/f, kill leftovers, warmup,
    fingerprint. Returns True when the board is back to the known-good state."""
    mgr.apply_preset("V1"); setf(100); time.sleep(3)
    subprocess.run(rs.BOARD + ["pkill -f debug_execute_graph; "
                   "cd /home/root/imcflow/xilinx/petalinux-csrc && "
                   "make clear_time >/dev/null 2>&1 && make warmup >/dev/null 2>&1"],
                   capture_output=True, timeout=180)
    for t in (1, 2):
        p = run_once()
        if p is None: continue
        d = p["ddc"]["run_mA"] - p["ddc"]["idle_mA"]
        log(f"  recover fp try{t}: ddc={d:.2f}mA len={p['vdd']['len']}")
        if 44.9 < d < 47.7 and 140 < p["vdd"]["len"] < 160: return True
    return False

def main():
    date = time.strftime("%Y%m%d_%H%M")
    out = os.path.join(CODEGEN, "experiments", f"shmoo_uv_{date}.csv")
    f = open(out, "w", newline=""); w = csv.writer(f)
    w.writerow(["volt", "mhz", "pulse_len", "exp_len", "perconv_nC", "verdict"]); f.flush()
    def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

    mgr = RemotePowerSupplyManager("147.46.117.49", 1331,
        "/home/jihoonpark/measurement_utils/example/configs/ps_B2_config.json")
    try:
        cur_v = None
        for v, mhz in CELLS:
            if v != cur_v:
                for r in ("VDD", "DDA", "DDC"):
                    mgr.set_voltage(r, v)
                cur_v = v; time.sleep(3)
                log(f"===== V={v} (all rails) =====")
            setf(mhz); time.sleep(2)
            p = run_once() or run_once()
            exp = 149.0 * 99.99 / mhz
            if p is None:
                if board_alive():
                    w.writerow([v, f"{mhz:.1f}", 0, f"{exp:.0f}", "", "FAIL"]); f.flush()
                    log(f"  {mhz:.1f}MHz -> FAIL (hang, board alive) — recovering")
                    if recover(mgr, log):
                        cur_v = None  # force rail re-set for next cell
                        continue
                    log("  recovery fingerprint failed -> stop"); return
                w.writerow([v, f"{mhz:.1f}", 0, f"{exp:.0f}", "", "WEDGE"]); f.flush()
                if os.environ.get("TAPO_AUTO_REBOOT") == "1":
                    log(f"  {mhz:.1f}MHz WEDGE -> auto-reboot via tapo")
                    if wedge_reboot_recover(mgr, log):
                        cur_v = None
                        continue
                    log("  auto-reboot recovery failed -> stop"); return
                log(f"  {mhz:.1f}MHz WEDGE -> stop")
                return
            # a FAIL run may have no detectable pulse on some rail -> None fields
            ln = p["vdd"]["len"]
            try: q = per_conv_nC(p)
            except TypeError: q = None
            lok = ln is not None and abs(ln - exp) < 0.10 * exp
            verdict = "PASS" if (lok and q is not None and q >= 2.0) else "FAIL"
            w.writerow([v, f"{mhz:.1f}", ln or 0, f"{exp:.0f}",
                        f"{q:.2f}" if q is not None else "", verdict]); f.flush()
            log(f"  {mhz:.1f}MHz -> {verdict} (len={ln}/{exp:.0f}, q={q})")
    finally:
        try: mgr.apply_preset("V1"); log("PS->V1")
        except Exception as e: log(f"PS restore fail: {e}")
        try: setf(100); log("clk->100")
        except Exception as e: log(f"clk skip: {e}")
        f.close(); log(f"CSV: {out}")

if __name__ == "__main__":
    main()
