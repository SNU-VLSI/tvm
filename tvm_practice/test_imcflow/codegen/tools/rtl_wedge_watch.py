#!/usr/bin/env python3
"""Live wedge watcher for IMCFlow RTL co-simulation.

Wedge signature: the host keeps polling the accelerator state register
(vcs_sim.log keeps appending "Processing READ (REG via AXI): addr=0x00000000"
lines) while the waveform dump (.fsdb) has effectively stopped growing --
nothing but clocks is toggling, i.e. a NoC/sync deadlock. This detects that
well before the host's 20000-poll timeout, so debugging can start early.

Usage:
  python tools/rtl_wedge_watch.py <eval_dir> [options]

Watches <eval_dir>/logs/rtl_runner/{vcs_sim.log, *.fsdb}. Both paths are
re-resolved every tick, so the watcher can be started before the run creates
them. One status line per tick (tail -f friendly). On wedge detection it
prints a loud alert, writes <eval_dir>/logs/rtl_runner/WEDGE_DETECTED with a
snapshot, and exits 2 (unless --keep-going). Exits 0 when the simulation
ends (vcs_sim.log reports client disconnect, or everything goes quiet).

Typical use from a run script:
  python tools/rtl_wedge_watch.py "$EVAL_DIR" >> "$WATCH_LOG" 2>&1 &
"""

import argparse
import glob
import os
import sys
import time

POLL_PAT = b"Processing READ (REG via AXI): addr=0x00000000"
END_PAT = b"Client disconnected"


def find_files(rtl_dir):
    vcs = os.path.join(rtl_dir, "vcs_sim.log")
    vcs = vcs if os.path.isfile(vcs) else None
    fsdbs = glob.glob(os.path.join(rtl_dir, "*.fsdb"))
    fsdb = max(fsdbs, key=os.path.getmtime) if fsdbs else None
    return vcs, fsdb


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dir", help="model eval_dir (contains logs/rtl_runner)")
    ap.add_argument("--interval", type=float, default=20.0,
                    help="seconds between checks (default 20)")
    ap.add_argument("--patience", type=int, default=4,
                    help="consecutive quiet ticks before declaring a wedge (default 4; "
                         "fsdb flushes are chunky ~40-60s, so a healthy run can look "
                         "quiet for 2-3 ticks at interval=20)")
    ap.add_argument("--min-fsdb-growth", type=int, default=65536,
                    help="fsdb bytes/tick below which the design counts as quiet "
                         "(default 64KiB; healthy compute grows MBs/tick)")
    ap.add_argument("--min-polls", type=int, default=10,
                    help="min poll READs/tick to count the host as actively polling")
    ap.add_argument("--end-patience", type=int, default=9,
                    help="ticks with zero vcs+fsdb growth before assuming the sim ended")
    ap.add_argument("--keep-going", action="store_true",
                    help="do not exit on wedge detection; keep logging")
    ap.add_argument("--kill", action="store_true",
                    help="on wedge detection, kill the simv/gem5 processes so the "
                         "run fails fast (fsim/fsdb logs are already flushed; saves "
                         "the remaining poll budget, typically 15+ min)")
    ap.add_argument("--max-polls", type=int, default=8000,
                    help="declare a wedge when a single span's poll count exceeds "
                         "this (healthy spans stay well under; default 8000)")
    args = ap.parse_args()

    rtl_dir = os.path.join(os.path.abspath(args.eval_dir), "logs", "rtl_runner")
    flag_path = os.path.join(rtl_dir, "WEDGE_DETECTED")

    vcs_off = 0          # bytes of vcs_sim.log already scanned
    span_polls = 0       # polls accumulated since the design last made progress
    fsdb_size = 0
    total_polls = 0
    quiet = 0            # consecutive polling-but-no-waveform ticks
    idle = 0             # consecutive completely-dead ticks
    alerted = False
    t0 = time.time()

    print(f"[watch] watching {rtl_dir} interval={args.interval}s "
          f"patience={args.patience} min_fsdb_growth={args.min_fsdb_growth}B", flush=True)

    while True:
        time.sleep(args.interval)
        vcs, fsdb = find_files(rtl_dir)
        now = int(time.time() - t0)

        if vcs is None:
            print(f"[watch] t={now}s waiting for vcs_sim.log ...", flush=True)
            continue

        # incremental scan of vcs_sim.log for poll READs and end marker
        new_polls, saw_end = 0, False
        vcs_size = os.path.getsize(vcs)
        if vcs_size < vcs_off:          # new run truncated/recreated the log
            vcs_off, total_polls = 0, 0
        if vcs_size > vcs_off:
            with open(vcs, "rb") as f:
                f.seek(vcs_off)
                chunk = f.read()
            vcs_off = vcs_size
            new_polls = chunk.count(POLL_PAT)
            saw_end = END_PAT in chunk
        total_polls += new_polls

        d_fsdb = 0
        fsdb_mb = 0.0
        if fsdb is not None:
            sz = os.path.getsize(fsdb)
            d_fsdb = max(0, sz - fsdb_size)
            fsdb_size = sz
            fsdb_mb = sz / 1e6

        if saw_end:
            print(f"[watch] t={now}s simulation ended (client disconnected). "
                  f"polls_total={total_polls} fsdb={fsdb_mb:.1f}MB", flush=True)
            return 0

        polling = new_polls >= args.min_polls
        computing = d_fsdb >= args.min_fsdb_growth

        if polling and not computing:
            quiet += 1
            span_polls += new_polls
        elif computing or new_polls > 0:
            quiet = 0
            # fsdb grew -> the design made progress; restart the span counter
            if computing:
                span_polls = 0
            else:
                span_polls += new_polls

        if new_polls == 0 and d_fsdb == 0:
            idle += 1
            if idle >= args.end_patience:
                print(f"[watch] t={now}s no activity for {idle} ticks; assuming run ended. "
                      f"polls_total={total_polls}", flush=True)
                return 0
        else:
            idle = 0

        verdict = "OK"
        if quiet:
            verdict = f"QUIET {quiet}/{args.patience}"
        print(f"[watch] t={now}s polls+={new_polls} (total {total_polls}) "
              f"fsdb+={d_fsdb/1e6:.2f}MB (total {fsdb_mb:.1f}MB) {verdict}", flush=True)

        if span_polls > args.max_polls and not alerted:
            alerted = True
            msg = (f"WEDGE (poll-budget) at t={now}s: {span_polls} polls since last "
                   f"fsdb progress (> --max-polls {args.max_polls}). total_polls={total_polls}\n"
                   f"Next: python tools/fsim_stall_report.py {args.eval_dir}\n")
            print("=" * 70 + f"\n[watch] *** {msg}" + "=" * 70, flush=True)
            try:
                with open(flag_path, "w") as f:
                    f.write(msg)
            except OSError as e:
                print(f"[watch] could not write {flag_path}: {e}", flush=True)
            if args.kill:
                _kill_sim()
            if not args.keep_going:
                return 2

        if quiet >= args.patience and not alerted:
            alerted = True
            msg = (f"WEDGE SUSPECTED at t={now}s: host polled {new_polls}/tick for "
                   f"{quiet} ticks (~{int(quiet*args.interval)}s) while fsdb grew "
                   f"<{args.min_fsdb_growth}B/tick. total_polls={total_polls} "
                   f"fsdb={fsdb_mb:.1f}MB\n"
                   f"Next: python tools/fsim_stall_report.py {args.eval_dir}\n")
            print("=" * 70 + f"\n[watch] *** {msg}" + "=" * 70, flush=True)
            try:
                with open(flag_path, "w") as f:
                    f.write(msg)
            except OSError as e:
                print(f"[watch] could not write {flag_path}: {e}", flush=True)
            if args.kill:
                _kill_sim()
            if not args.keep_going:
                return 2


def _kill_sim():
    """Kill the simv/gem5 pair so the wedged run fails fast. fsim logs and the
    fsdb are flushed continuously, so post-mortem diagnosis loses nothing."""
    import subprocess
    for pat in ("simv_imcflow_gem5", "gem5.opt"):
        try:
            out = subprocess.run(["pkill", "-f", pat], capture_output=True)
            print(f"[watch] pkill -f {pat} -> rc={out.returncode}", flush=True)
        except OSError as e:
            print(f"[watch] pkill {pat} failed: {e}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
