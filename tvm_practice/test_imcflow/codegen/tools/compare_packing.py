#!/usr/bin/env python3
"""
A/B comparison harness for the IMCFLOW_PACK_BN_MINMAX packing optimization.

Runs the SAME model twice -- once with packing OFF (baseline) and once ON --
into two separate eval_dirs, then reports a side-by-side comparison of:

  * active IMCE count per region (how many IMCEs actually emit instructions)
  * accelerator busy cycles per region (via tools/rtl_region_cycles.py, the
    imcflow_state_o HW signal in the .fsdb -- host poll/transfer excluded)
  * total busy cycles / us

Packing folds standalone BN / min_max_quantize IMCEs into the producing qconv's
IMCE, so the win shows up as FEWER active IMCEs (more layers per region / bigger
models mappable) and -- ideally -- equal-or-fewer busy cycles for the same work.

The two runs use DISTINCT eval_dir suffixes so their codegen/fsdb/outputs never
clobber each other:
    <model>_evl.<os>[.bugfixoff]            <- packing OFF
    <model>_evl.<os>[.bugfixoff].packon     <- packing ON  (IMCFLOW_EVAL_SUFFIX)

so you can re-point rtl_region_cycles.py at each afterwards.

USAGE
    # full A/B: codegen + RTL both ways, then compare (needs VCS + Verdi env)
    python tools/compare_packing.py --model resnet8_subset31_pretrained_orig \
        --ckpt n32_signed_sample

    # only regenerate codegen (no RTL) -- compares active-IMCE counts only
    python tools/compare_packing.py --model <m> --stop-at codegen

    # compare two ALREADY-RUN eval_dirs (no run), e.g. after a manual RTL pass
    python tools/compare_packing.py --compare-dirs <off_eval_dir> <on_eval_dir>

    # cycles method: fsdb (accurate, needs Verdi) or poll (log estimate)
    python tools/compare_packing.py --model <m> --method poll

Environment the RTL path needs (same as a normal --stop-at simulate run; see
CLAUDE.md "Running the BUGFIX-off RTL co-simulation"): IMCFLOW_BUGFIX=off,
IMCFLOW_RUNNER=rtl, IMCFLOW_HOST_OS, IMCFLOW_HOST_ISA, IMCFLOW_DIR,
SNPSLMD_LICENSE_FILE, CKPT (or --ckpt). This script sets IMCFLOW_PACK_BN_MINMAX
and IMCFLOW_EVAL_SUFFIX per run; everything else it inherits.
"""

import os
import re
import sys
import glob
import json
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
CODEGEN_DIR = os.path.dirname(HERE)  # .../test_imcflow/codegen
RTL_CYCLES = os.path.join(HERE, "rtl_region_cycles.py")

# imce.cpp per-node branch header:  if (hid == R && wid == C) { // imce_R_C
# NOTE: the SAME (hid,wid) branch appears once PER LAUNCH in a region's imce.cpp,
# so we must de-dup by (hid,wid) -- an IMCE is "active" if ANY of its launch
# blocks emits a real builtin. Counting raw branch matches over-counts by the
# number of launches.
_IMCE_HEADER = re.compile(r'if \(hid == \d+ && wid == \d+\) \{ // (imce_\d+_\d+)')
_IMCE_SPLIT = re.compile(r'if \(hid == \d+ && wid == \d+\) \{ // imce_\d+_\d+')
# a block is "active" if it emits any real IMCE builtin (not just GET_CORE/STOP)
_REAL_BUILTIN = re.compile(r'__builtin_IMCE_(?!GET_CORE|STOP)')
_REGION_DIR = re.compile(r'_imcflow_(region\d+)_main_\d+')


def active_imces_per_region(eval_dir):
    """Return {region_name: [sorted active imce names]} from the built imce.cpp.

    An IMCE (unique hid,wid) counts once, even though its code appears in
    multiple per-launch blocks within the file; it is active if any block emits
    a real builtin.
    """
    out = {}
    build = os.path.join(eval_dir, "build")
    for f in sorted(glob.glob(os.path.join(build, "tvmgen*", "imce.cpp"))):
        m = _REGION_DIR.search(f)
        region = m.group(1) if m else os.path.basename(os.path.dirname(f))
        src = open(f).read()
        names = _IMCE_HEADER.findall(src)
        bodies = _IMCE_SPLIT.split(src)[1:]  # aligned with names
        active = {}
        for name, body in zip(names, bodies):
            head = body.split("else if")[0]
            is_active = bool(_REAL_BUILTIN.search(head))
            # OR across launch blocks of the same imce
            active[name] = active.get(name, False) or is_active
        out[region] = sorted(n for n, a in active.items() if a)
    return out


def _run(cmd, env=None, cwd=CODEGEN_DIR):
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, env=env, cwd=cwd).returncode


def run_one(model, pack_on, stop_at, ckpt, extra_env):
    """Codegen (and optionally RTL) one side. Returns (eval_dir, rc)."""
    env = dict(os.environ)
    env["IMCFLOW_PACK_BN_MINMAX"] = "1" if pack_on else "0"
    # Distinct eval_dir for the packing-ON run so artifacts don't collide.
    if pack_on:
        env["IMCFLOW_EVAL_SUFFIX"] = env.get("IMCFLOW_EVAL_SUFFIX", "") + ".packon"
    if ckpt:
        env["CKPT"] = ckpt
    env.update(extra_env)
    label = "ON " if pack_on else "OFF"
    print(f"[compare_packing] === run packing {label} (stop_at={stop_at}) ===",
          flush=True)
    rc = _run([sys.executable, "-u", "main.py", "--model", model,
               "--stop-at", stop_at], env=env)
    eval_dir = resolve_eval_dir(model, env)
    return eval_dir, rc


def resolve_eval_dir(model, env):
    """Best-effort reconstruct the eval_dir path this run wrote to."""
    host_os = env.get("IMCFLOW_HOST_OS", "baremetal")
    base = f"{model}_evl.{host_os}"
    bugfix = env.get("IMCFLOW_BUGFIX", "off").strip().lower()
    if bugfix != "on":
        base += ".bugfixoff"
    base += env.get("IMCFLOW_EVAL_SUFFIX", "")
    return os.path.join(CODEGEN_DIR, "eval_dir", base)


def cycles_for(eval_dir, method):
    """Invoke rtl_region_cycles.py --json; return its dict or None if unavailable."""
    if not os.path.isdir(os.path.join(eval_dir, "logs", "rtl_runner")):
        return None
    try:
        out = subprocess.check_output(
            [sys.executable, RTL_CYCLES, eval_dir, "--method", method, "--json"],
            cwd=CODEGEN_DIR, stderr=subprocess.DEVNULL)
        return json.loads(out)
    except (subprocess.CalledProcessError, ValueError):
        return None


def _cyc_map(cyc):
    """region -> busy_cycles from rtl_region_cycles JSON, or {} if None."""
    if not cyc:
        return {}
    return {r["region"]: r["busy_cycles"] for r in cyc.get("regions", [])}


# Candidate output artifacts to compare for numerical equivalence, in priority
# order. The transformed CPU golden (hw-accurate qconv on CPU) is the gold
# standard: packing only rewrites the offloaded subgraph's structure, so this
# must be bit-identical OFF vs ON. RTL output is used if both sides ran RTL.
_OUTPUT_CANDIDATES = [
    # transformed CPU golden (hw-accurate qconv on CPU) -- the gold standard;
    # written by --stop-at validate/simulate to test_references/.
    "test_references/cpu_reference_output_transformed.npy",
    "test_references/cpu_reference_output.npy",
    "test_outputs/cpu_reference_output_transformed.npy",
    "test_outputs/cpu_reference_output.npy",
    "test_outputs/rtl_runner/output.npy",
    "test_outputs/py_runner/output.npy",
]


def numerical_equivalence(off_dir, on_dir):
    """Compare OFF vs ON output tensors bit-for-bit. Returns a result dict.

    Packing is a graph rewrite + remapping only, so functionality MUST be
    identical: the same artifact on both sides must be array_equal.
    """
    import numpy as np
    for rel in _OUTPUT_CANDIDATES:
        po = os.path.join(off_dir, rel)
        pn = os.path.join(on_dir, rel)
        if os.path.isfile(po) and os.path.isfile(pn):
            a = np.load(po)
            b = np.load(pn)
            if a.shape != b.shape:
                return {"artifact": rel, "status": "SHAPE_MISMATCH",
                        "detail": f"{a.shape} vs {b.shape}"}
            exact = bool(np.array_equal(a, b))
            if exact:
                return {"artifact": rel, "status": "IDENTICAL", "n": a.size}
            nmis = int(np.count_nonzero(a != b))
            maxd = None
            if np.issubdtype(a.dtype, np.number):
                maxd = float(np.max(np.abs(a.astype("float64")
                                           - b.astype("float64"))))
            return {"artifact": rel, "status": "DIFFERS",
                    "n": a.size, "n_mismatch": nmis, "max_abs_diff": maxd}
    return {"artifact": None, "status": "NO_OUTPUT",
            "detail": "no comparable output on both sides "
                      "(run --stop-at validate or simulate)"}


def report(off_dir, on_dir, method):
    off_imce = active_imces_per_region(off_dir)
    on_imce = active_imces_per_region(on_dir)
    off_cyc = _cyc_map(cycles_for(off_dir, method))
    on_cyc = _cyc_map(cycles_for(on_dir, method))

    regions = sorted(set(off_imce) | set(on_imce),
                     key=lambda r: int(re.sub(r"\D", "", r) or 0))

    print("\n" + "=" * 74)
    print("PACKING A/B COMPARISON  (OFF = baseline, ON = IMCFLOW_PACK_BN_MINMAX)")
    print("=" * 74)
    print(f"OFF eval_dir: {off_dir}")
    print(f"ON  eval_dir: {on_dir}")
    have_cyc = bool(off_cyc or on_cyc)
    if not have_cyc:
        print("(no RTL busy-cycle data found -- run with RTL, or --compare-dirs "
              "on completed runs. Showing active-IMCE counts only.)")

    hdr = f"\n{'region':<9}{'IMCE off':>9}{'IMCE on':>9}{'Δimce':>7}"
    if have_cyc:
        hdr += f"{'cyc off':>11}{'cyc on':>11}{'Δcyc':>9}{'Δ%':>7}"
    print(hdr)
    print("-" * len(hdr))

    tot_off_i = tot_on_i = 0
    tot_off_c = tot_on_c = 0
    for r in regions:
        no = len(off_imce.get(r, []))
        non = len(on_imce.get(r, []))
        tot_off_i += no
        tot_on_i += non
        line = f"{r:<9}{no:>9}{non:>9}{non - no:>+7}"
        if have_cyc:
            co = off_cyc.get(r)
            cn = on_cyc.get(r)
            tot_off_c += co or 0
            tot_on_c += cn or 0
            co_s = str(co) if co is not None else "-"
            cn_s = str(cn) if cn is not None else "-"
            if co and cn:
                d = cn - co
                pct = 100.0 * d / co
                line += f"{co_s:>11}{cn_s:>11}{d:>+9}{pct:>+6.1f}%"
            else:
                line += f"{co_s:>11}{cn_s:>11}{'N/A':>9}{'N/A':>7}"
        print(line)

    print("-" * len(hdr))
    tline = f"{'TOTAL':<9}{tot_off_i:>9}{tot_on_i:>9}{tot_on_i - tot_off_i:>+7}"
    if have_cyc:
        if tot_off_c and tot_on_c:
            d = tot_on_c - tot_off_c
            pct = 100.0 * d / tot_off_c
            tline += f"{tot_off_c:>11}{tot_on_c:>11}{d:>+9}{pct:>+6.1f}%"
        else:
            tline += f"{tot_off_c or '-':>11}{tot_on_c or '-':>11}{'partial':>9}{'':>7}"
    print(tline)
    print()
    saved = tot_off_i - tot_on_i
    print(f"IMCEs reclaimed by packing: {saved} "
          f"({tot_off_i} -> {tot_on_i} active)")
    if have_cyc and tot_off_c and tot_on_c:
        d = tot_on_c - tot_off_c
        verb = "fewer" if d < 0 else "more"
        print(f"Total busy cycles: {tot_off_c} -> {tot_on_c} "
              f"({abs(d)} {verb}, {100.0*d/tot_off_c:+.1f}%)")
    elif have_cyc:
        print("Total busy cycles: partial (one side's RTL did not complete -- "
              "see per-region N/A above).")

    # Functional equivalence: packing is graph-rewrite + remap only, so the
    # output MUST be bit-identical. This is the correctness gate.
    eq = numerical_equivalence(off_dir, on_dir)
    print()
    if eq["status"] == "IDENTICAL":
        print(f"Functional equivalence: ✓ BIT-IDENTICAL "
              f"({eq['artifact']}, {eq['n']} elems)")
    elif eq["status"] == "NO_OUTPUT":
        print(f"Functional equivalence: (not checked) {eq['detail']}")
    elif eq["status"] == "SHAPE_MISMATCH":
        print(f"Functional equivalence: ✗ SHAPE MISMATCH on {eq['artifact']} "
              f"({eq['detail']})")
    else:
        print(f"Functional equivalence: ✗ DIFFERS on {eq['artifact']} "
              f"({eq['n_mismatch']}/{eq['n']} elems differ, "
              f"max|Δ|={eq['max_abs_diff']})")
    print()


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="model registry name (e.g. "
                    "resnet8_subset31_pretrained_orig)")
    ap.add_argument("--ckpt", default=None, help="CKPT value for the run")
    ap.add_argument("--stop-at", default="simulate",
                    choices=["codegen", "compile", "validate", "simulate"],
                    help="codegen = IMCE-count compare only (no RTL/CPU run); "
                         "validate = also run CPU golden -> functional-equivalence "
                         "check (no RTL); "
                         "simulate = full A/B with busy cycles (default)")
    ap.add_argument("--method", choices=["fsdb", "poll"], default="fsdb",
                    help="busy-cycle method for rtl_region_cycles.py")
    ap.add_argument("--compare-dirs", nargs=2, metavar=("OFF_DIR", "ON_DIR"),
                    help="skip running; just compare two existing eval_dirs")
    args = ap.parse_args(argv)

    if args.compare_dirs:
        report(args.compare_dirs[0], args.compare_dirs[1], args.method)
        return 0

    if not args.model:
        ap.error("--model is required unless --compare-dirs is given")

    extra_env = {}
    off_dir, rc_off = run_one(args.model, False, args.stop_at, args.ckpt, extra_env)
    if rc_off not in (0,):
        print(f"[compare_packing] WARNING: packing-OFF run exited rc={rc_off} "
              f"(continuing; results may be partial)", file=sys.stderr)
    on_dir, rc_on = run_one(args.model, True, args.stop_at, args.ckpt, extra_env)
    if rc_on not in (0,):
        print(f"[compare_packing] WARNING: packing-ON run exited rc={rc_on} "
              f"(likely a sync deadlock -- cycles will show N/A)", file=sys.stderr)

    report(off_dir, on_dir, args.method)
    return 0


if __name__ == "__main__":
    sys.exit(main())
