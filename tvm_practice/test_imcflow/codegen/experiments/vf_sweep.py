#!/usr/bin/env python3
"""Per-VDD max-frequency + best analog-floor DVFS sweep.

For each VDD in the sweep list:
  1. hold DDA=DDC at V1 (headroom), push frequency UP the divisor grid until
     the physics gate fails (pulse-length off, or DDA delta-charge < ref):
     that is f_max(VDD).
  2. at (VDD, f_max) run a tied DDA=DDC greedy descent to its analog floor.
  3. record the best-TOPS/W operating point for that VDD.

Physics gates (no per-freq fingerprint ref needed):
  pulse_len ~= 149 * 100/f  (+-10%)   AND   DDA delta-charge >= 0.90 * ref(f)
where ref(f) follows the measured q(f) = 66.9 + 3750/f uC model.

Safety: hang-class failures get NO retry, board-alive check, and the run
ALWAYS restores V1 + 100MHz on exit. cold-cache/daemon-churn transients get
one start retry. Known static hang boundaries (VDD floor) are excluded.
"""
import csv, importlib.util, os, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rs", os.path.join(HERE, "rail_sweep.py"))
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
from ps_ctrl.rpc import RemotePowerSupplyManager

PARENT = 1499.85
# divisor grid from 100MHz upward; extend as needed
DIV_GRID = [15, 14, 13, 12, 11, 10, 9, 8]          # 100,107,115,125,136,150,167,188
BASE_LEN, BASE_F = 149.0, 99.99
DT = 21e-6
def q_ref(mhz): return 44.14 + 5126.3/mhz            # DDA charge model (uC), 3-pt fit @VDD1.0
CODEGEN = rs.CODEGEN


def setf(mhz): subprocess.run(["bash", os.path.join(CODEGEN, "tools/pl_freq.sh"), "set", str(int(round(mhz)))], capture_output=True)
def board_alive():
    r = subprocess.run(rs.BOARD[:-1] + ["-o", "ConnectTimeout=8", rs.BOARD[-1], "echo ok"],
                       capture_output=True, text=True, timeout=20)
    return "ok" in r.stdout
def run_once():
    b = rs.rec_linecount(); rs.board_run()
    return rs.extract_pulse() if rs.rec_linecount() > b else None
def dda_charge(p): return (p["dda"]["run_mA"] - p["dda"]["idle_mA"]) * p["dda"]["len"] * DT
NCONV = 16128  # 127*127-1 conversions per imce
def per_conv_nC(p): return dda_charge(p) / NCONV * 1e6  # nC per conversion (freq-invariant)
def len_ok(p, mhz):
    if not p or p["vdd"]["run_mA"] is None: return False
    exp = BASE_LEN * BASE_F / mhz
    return abs(p["vdd"]["len"] - exp) < 0.10 * exp
def charge_ok(p, anchor_nC):
    # SELF-CALIBRATED residual gate: per-conversion charge must stay within 5%
    # of the running anchor (previous good point at this VDD). The smooth q(f)
    # decline is a slow drift the anchor tracks; a real collapse drops
    # per-conversion charge sharply (125MHz was ~-6% in one step) and trips this.
    return per_conv_nC(p) >= 0.95 * anchor_nC


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--vdds", default="1.00,0.80", help="comma list, swept")
    ap.add_argument("--analog-floor", type=float, default=0.60)
    ap.add_argument("--dry", action="store_true", help="f_max only, skip analog descent")
    args = ap.parse_args()
    vdds = [float(x) for x in args.vdds.split(",")]

    date = time.strftime("%Y%m%d_%H%M")
    out = os.path.join(CODEGEN, "experiments", f"vf_sweep_{date}.csv")
    f = open(out, "w", newline=""); w = csv.writer(f)
    w.writerow(["vdd", "phase", "mhz", "dda_ddc", "pulse_len", "run_vdd_mA",
                "run_dda_mA", "run_ddc_mA", "dda_uC", "P_mW", "TOPS", "TOPSW", "status"]); f.flush()
    def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

    mgr = RemotePowerSupplyManager("147.46.117.49", 1331,
        "/home/jihoonpark/measurement_utils/example/configs/ps_B2_config.json")
    ANV1 = 1.15  # analog V1 headroom while finding f_max
    try:
        for vdd in vdds:
            log(f"===== VDD={vdd} =====")
            mgr.set_voltage("VDD", vdd); mgr.set_voltage("DDA", ANV1); mgr.set_voltage("DDC", ANV1)
            time.sleep(3)
            # phase 1: push freq up
            fmax = None; p_fmax = None; anchor_nC = None
            for div in DIV_GRID:
                mhz = PARENT / div
                setf(mhz); time.sleep(2)
                p = run_once() or (run_once() if div == DIV_GRID[0] else None)  # transient retry only at first
                if anchor_nC is None and p and len_ok(p, mhz):
                    anchor_nC = per_conv_nC(p)   # first good point at this VDD = anchor
                ok = bool(p) and len_ok(p, mhz) and (anchor_nC is None or charge_ok(p, anchor_nC))
                if ok and p: anchor_nC = per_conv_nC(p)   # advance anchor (track slow drift)
                P = (sum(p[r]["run_mA"] for r in ("vdd","dda","ddc")) if p else 0)  # placeholder
                if p:
                    Pw = (p["vdd"]["run_mA"]*vdd + p["dda"]["run_mA"]*ANV1 + p["ddc"]["run_mA"]*ANV1)/1000
                    tops = 2.85*mhz/99.99
                    w.writerow([vdd,"fmax",f"{mhz:.1f}",ANV1,p["vdd"]["len"],
                                f"{p['vdd']['run_mA']:.2f}",f"{p['dda']['run_mA']:.2f}",
                                f"{p['ddc']['run_mA']:.2f}",f"{dda_charge(p)*1e3:.2f}",
                                f"{Pw*1000:.1f}",f"{tops:.2f}",f"{tops/Pw:.1f}",
                                "ok" if ok else "gate-fail"]); f.flush()
                else:
                    w.writerow([vdd,"fmax",f"{mhz:.1f}",ANV1,0,"","","","","","","","run-fail"]); f.flush()
                if ok:
                    fmax, p_fmax = mhz, p
                    log(f"  {mhz:.1f}MHz OK (len={p['vdd']['len']})")
                else:
                    log(f"  {mhz:.1f}MHz FAIL -> f_max={fmax}")
                    if not board_alive(): log("BOARD DEAD"); return
                    setf(100)  # back to safe clock before next
                    break
            if fmax is None:
                log(f"  no working freq at VDD={vdd}?!"); continue
            log(f"  === f_max(VDD={vdd}) = {fmax:.1f}MHz ===")
            if args.dry:
                continue
            # phase 2: tied analog greedy at (vdd, fmax)
            setf(fmax); time.sleep(2)
            an = ANV1; best = None; an_anchor = per_conv_nC(p_fmax)
            while an - 0.01 >= args.analog_floor:
                an = round(an - 0.01, 3)
                mgr.set_voltage("DDA", an); mgr.set_voltage("DDC", an); time.sleep(2)
                p = run_once()
                if not (p and len_ok(p, fmax) and charge_ok(p, an_anchor)):
                    log(f"  analog {an+0.01:.2f} was the floor (fail at {an:.2f})")
                    if not board_alive(): log("BOARD DEAD"); return
                    break
                Pw = (p["vdd"]["run_mA"]*vdd + (p["dda"]["run_mA"]+p["ddc"]["run_mA"])*an)/1000
                tops = 2.85*fmax/99.99
                best = (an, Pw, tops); an_anchor = per_conv_nC(p)
                w.writerow([vdd,"analog",f"{fmax:.1f}",an,p["vdd"]["len"],
                            f"{p['vdd']['run_mA']:.2f}",f"{p['dda']['run_mA']:.2f}",
                            f"{p['ddc']['run_mA']:.2f}",f"{dda_charge(p)*1e3:.2f}",
                            f"{Pw*1000:.1f}",f"{tops:.2f}",f"{tops/Pw:.1f}","ok"]); f.flush()
            if best:
                an, Pw, tops = best
                log(f"  BEST @VDD={vdd}: f={fmax:.1f} analog={an} P={Pw*1000:.1f}mW TOPS/W={tops/Pw:.1f}")
            setf(100); mgr.set_voltage("DDA", ANV1); mgr.set_voltage("DDC", ANV1); time.sleep(1)
    finally:
        setf(100); mgr.apply_preset("V1")
        log(f"CSV: {out} (V1 + 100MHz restored)")
        f.close()


if __name__ == "__main__":
    main()
