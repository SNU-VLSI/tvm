#!/usr/bin/env python3
"""Frequency-raising probe at preset V1 (divisor-grid single steps).

Gates per step (no fingerprint ref exists at new freqs, so physics gates):
  (1) pulse_len within +-10% of 149 * 100/f  (ran at the new speed)
  (2) DDA delta-charge within +-10% of the 100MHz reference (ALL conversions
      really happened -- per-conversion ADC charge is frequency-invariant)
Failure => hang-class: NO retry, board-alive check, restore 100MHz + rescan
+ fingerprint, stop. Reports max working f and TOPS = 2.85 * f/100.
"""
import csv, importlib.util, os, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rs", os.path.join(HERE, "rail_sweep.py"))
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
from ps_ctrl.rpc import RemotePowerSupplyManager

PARENT = 1499.85  # MHz
DIVS = [14, 13, 12, 11, 10, 9, 8]      # 107.1, 115.4, 125.0, 136.4, 150.0, 166.7, 187.5
BASE_LEN, BASE_F = 149.0, 99.99
DDA_QREF = None  # set from a 100MHz reference run
CODEGEN = rs.CODEGEN
DT = 21e-6


def set_freq(mhz):
    subprocess.run(["bash", os.path.join(CODEGEN, "tools", "pl_freq.sh"), "set", str(int(round(mhz)))],
                   capture_output=True)


def board_alive():
    r = subprocess.run(rs.BOARD[:-1] + ["-o", "ConnectTimeout=8", rs.BOARD[-1], "echo ok"],
                       capture_output=True, text=True, timeout=20)
    return "ok" in r.stdout


def one_run():
    before = rs.rec_linecount()
    ok, raw = rs.board_run()
    if rs.rec_linecount() > before:
        return rs.extract_pulse()
    return None


def dda_charge(p):
    return (p["dda"]["run_mA"] - p["dda"]["idle_mA"]) * p["dda"]["len"] * DT  # mA*s ~ uC/1000


def rescan_and_fp(log):
    set_freq(100)
    subprocess.run(rs.BOARD + ["""
      B=/home/root/tvm/tvm_practice/test_imcflow/codegen
      cd /home/root/imcflow/xilinx/measurement && timeout -s INT 0.5s ./program_scan_reg $B/scan_gen/scan_reg_files
      cd /home/root/imcflow/xilinx/petalinux-csrc && make clear_time >/dev/null 2>&1 && make warmup >/dev/null 2>&1"""],
                   capture_output=True, timeout=180)
    p = one_run() or one_run()
    if p:
        d = p["ddc"]["run_mA"] - p["ddc"]["idle_mA"]
        log(f"  post-fail FP100 ddc_delta={d:.2f} len={p['ddc']['len']}")
        return abs(d - 46.3) < 1.4
    return False


def main():
    date = time.strftime("%Y%m%d_%H%M")
    out_csv = os.path.join(CODEGEN, "experiments", f"freq_up_{date}.csv")
    f = open(out_csv, "w", newline=""); w = csv.writer(f)
    w.writerow(["mhz", "div", "pulse_len", "run_vdd_mA", "run_dda_mA", "run_ddc_mA",
                "dda_uC", "P_mW", "TOPS", "TOPSW", "status"]); f.flush()

    def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

    mgr = RemotePowerSupplyManager("147.46.117.49", 1331,
        "/home/jihoonpark/measurement_utils/example/configs/ps_B2_config.json")
    mgr.apply_preset("V1")
    set_freq(100)
    time.sleep(2)

    # 100MHz reference (warm + charge ref), one transient retry
    p = one_run() or one_run()
    if not p:
        log("100MHz reference run failed -> abort"); return
    qref = dda_charge(p)
    log(f"100MHz ref: len={p['vdd']['len']} dda_charge={qref*1e3:.2f}uC")
    V = {"vdd": 1.0, "dda": 1.15, "ddc": 1.145}
    best = (99.99, p)
    try:
        for div in DIVS:
            mhz = PARENT / div
            set_freq(mhz)
            time.sleep(2)
            log(f"try {mhz:.1f}MHz (div {div})")
            p = one_run()  # NO retry: hang-class boundary
            ok = False
            if p:
                exp_len = BASE_LEN * BASE_F / mhz
                q = dda_charge(p)
                len_ok = abs(p["vdd"]["len"] - exp_len) < 0.10 * exp_len
                q_ok = abs(q - qref) < 0.10 * qref
                ok = len_ok and q_ok
                P = sum(p[r]["run_mA"] * V[r] for r in V) / 1000.0
                tops = 2.85 * mhz / 99.99
                w.writerow([f"{mhz:.1f}", div, p["vdd"]["len"],
                            f"{p['vdd']['run_mA']:.2f}", f"{p['dda']['run_mA']:.2f}",
                            f"{p['ddc']['run_mA']:.2f}", f"{q*1e3:.2f}",
                            f"{P*1000:.1f}", f"{tops:.2f}", f"{tops/P:.1f}",
                            "ok" if ok else f"gate-fail(len_ok={len_ok},q_ok={q_ok})"]); f.flush()
                if ok:
                    log(f"  OK {mhz:.1f}MHz len={p['vdd']['len']} q={q*1e3:.2f}uC "
                        f"P={P*1000:.1f}mW TOPS={tops:.2f} TOPS/W={tops/P:.1f}")
                    best = (mhz, p)
                else:
                    log(f"  GATE FAIL at {mhz:.1f} (len={p['vdd']['len']} exp={exp_len:.0f}, "
                        f"q={q*1e3:.2f} ref={qref*1e3:.2f})")
            else:
                w.writerow([f"{mhz:.1f}", div, 0, "", "", "", "", "", "", "", "run-fail"]); f.flush()
                log(f"  RUN FAIL at {mhz:.1f}MHz")
            if not ok:
                if not board_alive():
                    log("BOARD DEAD -> stop"); return
                if not rescan_and_fp(log):
                    log("chip not restored -> stop"); return
                break
    finally:
        set_freq(100)
        mgr.apply_preset("V1")
        log(f"MAX WORKING f = {best[0]:.1f}MHz -> TOPS = {2.85*best[0]/99.99:.2f}")
        log(f"CSV: {out_csv} (V1 + 100MHz restored)")
        f.close()


if __name__ == "__main__":
    main()
