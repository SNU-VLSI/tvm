#!/usr/bin/env python3
"""Reproduce the two published max-TOPS-kernel operating points on chip3/B2.

Points (from CIM/figures/peak_eff_data.csv):
  peak_tops : VDD=1.00, DDA=DDC=1.02, 125 MHz  -> ref 182.4 mW, 3.56 TOPS, 19.5 TOPS/W
  peak_eff  : VDD=0.81, DDA=DDC=0.66,  50 MHz  -> ref  42.7 mW, 1.425 TOPS, 33.4 TOPS/W

Per point: V1 fingerprint gate @100MHz -> set freq -> set rails -> settle ->
N kernel runs (DMM bracket) -> charge-integral currents -> P = sum(run_mA*V)
-> TOPS/W, mean +/- std vs reference. Kernel hang -> auto-recover
(V1+100+warmup+fingerprint) and retry the point once; wedge -> stop.
Restores V1 + 100 MHz on exit.

Usage:
  python3 experiments/repro_peaks.py [--points peak_tops,peak_eff]
      [--runs 3] [--input-tag ones]
`--input-tag` only labels the CSV (swap inputs with gen_rand4_input.py first).
"""
import argparse, csv, importlib.util, os, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rs", os.path.join(HERE, "rail_sweep.py"))
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
from ps_ctrl.rpc import RemotePowerSupplyManager

CODEGEN = rs.CODEGEN

POINTS = {
    # name: (VDD, DDA=DDC, MHz, ref_mW, ref_TOPSW)
    "peak_tops": (1.00, 1.02, 125.0, 182.4, 19.5),
    "peak_eff":  (0.81, 0.66,  50.0,  42.7, 33.4),
    "tops_an115":  (1.00, 1.15, 125.0, 0, 0),   # rand4: more analog headroom
    "tops_115mhz": (1.00, 1.02, 115.4, 0, 0),   # rand4: one grid step down
    # rand4 re-measure of the mid peak-eff points (ones f_max first, fallback -1 step)
    "p085_88":  (0.85, 0.83,  88.2, 93.1, 27.0),
    "p085_83":  (0.85, 0.83,  83.3, 0, 0),
    "p090_100": (0.90, 0.86, 100.0, 113.6, 25.1),
    "p090_94":  (0.90, 0.86,  93.7, 0, 0),
    "p095_115": (0.95, 0.96, 115.4, 152.4, 21.6),
    "p095_107": (0.95, 0.96, 107.1, 0, 0),
    # rand4 recon: analog-headroom vs deeper-freq at 0.85V
    "p085_88_an115": (0.85, 1.15, 88.2, 0, 0),
    "p085_83_an115": (0.85, 1.15, 83.3, 0, 0),
    "p085_79": (0.85, 0.83, 78.9, 0, 0),
    "p085_75": (0.85, 0.83, 75.0, 0, 0),
    "p090_88": (0.90, 0.86, 88.2, 0, 0),   # rand4: two steps down
    "p095_100": (0.95, 0.96, 100.0, 0, 0), # rand4: two steps down
}

def setf(mhz): subprocess.run(["bash", os.path.join(CODEGEN, "tools/pl_freq.sh"),
                               "set", str(int(round(mhz)))], capture_output=True)
def board_alive():
    r = subprocess.run(rs.BOARD[:-1] + ["-o", "ConnectTimeout=8", rs.BOARD[-1], "echo ok"],
                       capture_output=True, text=True, timeout=20)
    return "ok" in r.stdout
def run_once():
    b = rs.rec_linecount(); rs.board_run()
    return rs.extract_pulse() if rs.rec_linecount() > b else None

FP_RANGE = [44.9, 47.7]   # ones-input DDC delta @V1/100MHz; input-dependent

def fingerprint(mgr, log, tries=4):
    mgr.apply_preset("V1"); setf(100); time.sleep(3)
    for t in range(1, tries + 1):
        p = run_once()
        if p is None: log(f"  fp try{t}: no record"); continue
        d = p["ddc"]["run_mA"] - p["ddc"]["idle_mA"]
        log(f"  fp try{t}: ddc={d:.2f}mA len={p['vdd']['len']}")
        if FP_RANGE[0] < d < FP_RANGE[1] and 140 < p["vdd"]["len"] < 160: return True
    return False

def recover(mgr, log):
    mgr.apply_preset("V1"); setf(100); time.sleep(3)
    subprocess.run(rs.BOARD + ["pkill -f debug_execute_graph; "
                   "cd /home/root/imcflow/xilinx/petalinux-csrc && "
                   "make clear_time >/dev/null 2>&1 && make warmup >/dev/null 2>&1"],
                   capture_output=True, timeout=180)
    return fingerprint(mgr, log, tries=3)

def measure_point(mgr, log, w, fcsv, name, vdd, an, mhz, runs, tag):
    """Returns list of per-run dicts, or None on unrecoverable failure."""
    log(f"===== {name}: VDD={vdd} DDA=DDC={an} f={mhz}MHz ({runs} runs) =====")
    setf(mhz)
    # rails: going UP first is safe in either direction order; set VDD then analog
    mgr.set_voltage("VDD", vdd)
    mgr.set_voltage("DDA", an); mgr.set_voltage("DDC", an)
    time.sleep(3)
    results = []
    exp_len = 149.0 * 99.99 / mhz
    for i in range(1, runs + 1):
        p = run_once()
        if p is None:
            alive = board_alive()
            log(f"  run{i}: {'HANG' if alive else 'WEDGE'}")
            w.writerow([name, tag, vdd, an, mhz, i, "", "", "", "", "", "",
                        "HANG" if alive else "WEDGE"]); fcsv.flush()
            if not alive: return None
            if not recover(mgr, log): return None
            # re-apply point and continue
            setf(mhz); mgr.set_voltage("VDD", vdd)
            mgr.set_voltage("DDA", an); mgr.set_voltage("DDC", an); time.sleep(3)
            continue
        ln = p["vdd"]["len"]
        if ln is None or abs(ln - exp_len) > 0.10 * exp_len:
            log(f"  run{i}: bad pulse len={ln}/{exp_len:.0f} -> skip")
            w.writerow([name, tag, vdd, an, mhz, i, "", "", "", ln, f"{exp_len:.0f}",
                        "", "LEN-FAIL"]); fcsv.flush()
            continue
        mw = (p["vdd"]["run_mA"] * vdd + p["dda"]["run_mA"] * an
              + p["ddc"]["run_mA"] * an)
        tops = 2.85 * mhz / 100.0
        topsw = tops / (mw / 1000.0)
        results.append({"mW": mw, "TOPSW": topsw, "p": p})
        w.writerow([name, tag, vdd, an, mhz, i,
                    f"{p['vdd']['run_mA']:.2f}", f"{p['dda']['run_mA']:.2f}",
                    f"{p['ddc']['run_mA']:.2f}", ln, f"{exp_len:.0f}",
                    f"{mw:.1f}", f"{topsw:.1f}"]); fcsv.flush()
        log(f"  run{i}: P={mw:.1f}mW  TOPS/W={topsw:.1f}  "
            f"(vdd {p['vdd']['run_mA']:.1f} / dda {p['dda']['run_mA']:.1f} / "
            f"ddc {p['ddc']['run_mA']:.1f} mA, len={ln})")
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default="peak_tops,peak_eff")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--input-tag", default="ones")
    ap.add_argument("--fp-range", default=None,
                    help="min,max DDC delta mA for the V1/100MHz gate (input-dependent)")
    args = ap.parse_args()
    if args.fp_range:
        FP_RANGE[:] = [float(x) for x in args.fp_range.split(",")]

    date = time.strftime("%Y%m%d_%H%M")
    out = os.path.join(CODEGEN, "experiments", f"repro_peaks_{date}.csv")
    fcsv = open(out, "w", newline=""); w = csv.writer(fcsv)
    w.writerow(["point", "input", "vdd", "an", "mhz", "run",
                "vdd_mA", "dda_mA", "ddc_mA", "pulse_len", "exp_len",
                "P_mW", "TOPS_per_W"]); fcsv.flush()
    def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

    mgr = RemotePowerSupplyManager(*rs.PS)
    try:
        if not fingerprint(mgr, log):
            log("fingerprint gate FAILED -> abort (board not in known-good state)")
            return
        for name in args.points.split(","):
            vdd, an, mhz, ref_mw, ref_tw = POINTS[name]
            res = measure_point(mgr, log, w, fcsv, name, vdd, an, mhz,
                                args.runs, args.input_tag)
            if res is None:
                log(f"{name}: unrecoverable -> stop"); return
            if res:
                import statistics as st
                mws = [r["mW"] for r in res]; tws = [r["TOPSW"] for r in res]
                log(f"{name} [{args.input_tag}] mean P={st.mean(mws):.1f}mW"
                    f" (ref {ref_mw})  TOPS/W={st.mean(tws):.1f} (ref {ref_tw})"
                    + (f"  std P={st.stdev(mws):.1f}" if len(mws) > 1 else ""))
            # V1 between points for a clean baseline
            mgr.apply_preset("V1"); time.sleep(2)
    finally:
        try: mgr.apply_preset("V1"); log("PS->V1")
        except Exception as e: log(f"PS restore fail: {e}")
        try: setf(100); log("clk->100")
        except Exception as e: log(f"clk skip: {e}")
        fcsv.close(); log(f"CSV: {out}")

if __name__ == "__main__":
    main()
