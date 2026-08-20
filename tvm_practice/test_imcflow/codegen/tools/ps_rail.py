#!/usr/bin/env python3
"""Remote power-supply rail control / voltage sweep via the measurement2 rpyc
server (ps-rpc-server, default 147.46.117.49:1331) -- NO direct ssh needed.

Uses the vendored ps_ctrl.rpc.RemotePowerSupplyManager client against the
server-side config (default: ~/measurement_utils/example/configs/ps_B2_config.json,
rails DDA/DDC/DDF/DVDD/VDD/DDL, PRESET V1 = the tuned baseline).

Subcommands
  status [RAIL ...]          read setpoint + measured V/I (read-only)
  set RAIL=V [RAIL=V ...]    set voltages (safety-bounded vs PRESET V1; --force to override)
  preset [NAME]              apply a named preset (default V1) -- the restore point
  sweep --points "SPEC;SPEC;..." --cmd 'SHELL'
                             for each point (e.g. "DDC=1.10,VDD=1.05"): set rails,
                             settle, run SHELL with PS_POINT/PS_<RAIL> env vars,
                             then ALWAYS restore PRESET V1 at the end (also on error).

Safety
  - voltages outside [LOW_FRAC, HIGH_FRAC] x preset-V1 value are refused
    without --force (defaults 0.70 / 1.02: this tool is for UNDER-volting sweeps).
  - every action is appended as JSON to --log (default ~/.ps_rail_log.jsonl).
  - outputs are never toggled on/off by this tool.

Examples
  python3 tools/ps_rail.py status VDD DDA DDC
  python3 tools/ps_rail.py set DDC=1.10
  python3 tools/ps_rail.py preset            # restore V1
  python3 tools/ps_rail.py sweep \
      --points "DDC=1.145;DDC=1.10;DDC=1.05;DDC=1.00" \
      --settle 3 \
      --cmd 'ssh -p 1326 root@147.46.117.99 ". /home/root/run_power_kernel.sh"'
    # each iteration exports PS_POINT="DDC=1.10" and PS_DDC="1.10" to SHELL
"""
import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (
    os.path.join(_HERE, "..", "..", "..", "..", "3rdparty", "measurement_utils"),
    "/root/project/tvm/3rdparty/measurement_utils",
):
    _cand = os.path.abspath(_cand)
    if os.path.isdir(os.path.join(_cand, "ps_ctrl")) and _cand not in sys.path:
        sys.path.insert(0, _cand)
        break

DEFAULT_HOST = os.environ.get("PSM_RPC_HOST", "147.46.117.49")
DEFAULT_PORT = int(os.environ.get("PSM_RPC_PORT", "1331"))
DEFAULT_CONFIG = os.environ.get(
    "PSM_CONFIG", "~/measurement_utils/example/configs/ps_B2_config.json")
RESTORE_PRESET = os.environ.get("PSM_RESTORE_PRESET", "V1")
LOW_FRAC, HIGH_FRAC = 0.70, 1.02


def _connect(args):
    from ps_ctrl.rpc import RemotePowerSupplyManager
    return RemotePowerSupplyManager(args.host, args.port, args.config)


def _log(args, record):
    record = dict(record, t=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(os.path.expanduser(args.log), "a") as f:
        f.write(json.dumps(record) + "\n")


def _preset_vols(mgr, preset=RESTORE_PRESET):
    """PRESET voltage reference (server-side dict via rpyc netref)."""
    out = {}
    for rail, st in mgr.presets[preset].items():
        if "VOL" in st:
            out[str(rail)] = float(st["VOL"])
    return out


def _parse_assign(items):
    pairs = []
    for it in items:
        rail, _, val = it.partition("=")
        if not val:
            raise SystemExit(f"bad RAIL=V spec: {it!r}")
        pairs.append((rail.strip().upper(), float(val)))
    return pairs


def _check_bounds(pairs, ref, force):
    for rail, v in pairs:
        if rail not in ref:
            raise SystemExit(f"unknown rail {rail} (preset has: {sorted(ref)})")
        lo, hi = ref[rail] * LOW_FRAC, ref[rail] * HIGH_FRAC
        if not (lo <= v <= hi) and not force:
            raise SystemExit(
                f"{rail}={v}V outside safety window [{lo:.3f}, {hi:.3f}] "
                f"(preset {ref[rail]}V); use --force to override")


def cmd_status(args):
    mgr = _connect(args)
    rails = [r.upper() for r in args.rails] or sorted(_preset_vols(mgr))
    for r in rails:
        try:
            sp = mgr.get_voltage(r)
            m = mgr.meas_voltage(r)
            print(f"{r}: set={sp} meas={m}")
        except Exception as e:  # rail may be off/unknown
            print(f"{r}: ERROR {e}")
    _log(args, {"op": "status", "rails": rails})


def cmd_set(args):
    mgr = _connect(args)
    pairs = _parse_assign(args.assign)
    ref = _preset_vols(mgr)
    _check_bounds(pairs, ref, args.force)
    before = {r: mgr.get_voltage(r) for r, _ in pairs}
    for r, v in pairs:
        mgr.set_voltage(r, v)
    after = {r: mgr.get_voltage(r) for r, _ in pairs}
    print(f"set: {dict(pairs)} (before={before}, readback={after})")
    _log(args, {"op": "set", "pairs": dict(pairs), "before": before, "after": after})


def cmd_preset(args):
    mgr = _connect(args)
    name = args.name or RESTORE_PRESET
    mgr.apply_preset(name)
    print(f"preset {name} applied")
    _log(args, {"op": "preset", "name": name})


def cmd_sweep(args):
    mgr = _connect(args)
    ref = _preset_vols(mgr)
    points = [p.strip() for p in args.points.split(";") if p.strip()]
    parsed = [(_parse_assign(p.split(","))) for p in points]
    for pairs in parsed:
        _check_bounds(pairs, ref, args.force)

    results = []
    try:
        for spec, pairs in zip(points, parsed):
            for r, v in pairs:
                mgr.set_voltage(r, v)
            time.sleep(args.settle)
            meas = {r: str(mgr.meas_voltage(r)) for r, _ in pairs}
            env = dict(os.environ, PS_POINT=spec,
                       **{f"PS_{r}": str(v) for r, v in pairs})
            print(f"=== [ps_rail sweep] point {spec} (meas {meas}) ===", flush=True)
            _log(args, {"op": "sweep-point", "point": spec, "meas": meas})
            rc = 0
            if args.cmd:
                rc = subprocess.call(args.cmd, shell=True, env=env)
            results.append({"point": spec, "rc": rc})
            if rc != 0 and not args.keep_going:
                print(f"[ps_rail] cmd rc={rc} at {spec}; aborting sweep", flush=True)
                break
    finally:
        # ALWAYS park the rails back on the tuned baseline.
        try:
            mgr.apply_preset(RESTORE_PRESET)
            print(f"[ps_rail] restored preset {RESTORE_PRESET}", flush=True)
            _log(args, {"op": "restore", "preset": RESTORE_PRESET})
        except Exception as e:
            print(f"[ps_rail] RESTORE FAILED: {e} -- rails may be off-baseline!",
                  file=sys.stderr, flush=True)
            _log(args, {"op": "restore-FAILED", "err": str(e)})
    print(json.dumps(results))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="server-side ps config json path")
    ap.add_argument("--log", default="~/.ps_rail_log.jsonl")
    sub = ap.add_subparsers(dest="op", required=True)

    p = sub.add_parser("status"); p.add_argument("rails", nargs="*"); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("set"); p.add_argument("assign", nargs="+", metavar="RAIL=V")
    p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_set)
    p = sub.add_parser("preset"); p.add_argument("name", nargs="?"); p.set_defaults(fn=cmd_preset)
    p = sub.add_parser("sweep")
    p.add_argument("--points", required=True, help='e.g. "DDC=1.145;DDC=1.10,VDD=1.05"')
    p.add_argument("--cmd", default="", help="shell command per point (PS_* env exposed)")
    p.add_argument("--settle", type=float, default=3.0)
    p.add_argument("--keep-going", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_sweep)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
