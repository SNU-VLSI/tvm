#!/usr/bin/env python3
"""
Per-region accelerator busy-cycle extractor for IMCFlow RTL (VCS) co-simulation.

Reports, for each accelerator region, how many cycles the imcflow accelerator was
actually computing (host transfer / poll overhead excluded), and the wall-clock
time at the accelerator clock (default 100 MHz).

TWO measurement methods:

  --method fsdb  (DEFAULT, accurate): read the RTL hardware busy signal
      ``imcflow_state_o`` directly from the run's .fsdb. This wire is 1 exactly
      while the accelerator is busy and 0 otherwise, so it is immune to host poll
      cadence and backdoor transfers. Requires Verdi's fsdb2vcd on PATH.

  --method poll  (log-only estimate, no fsdb tools needed): infer busy time from
      vcs_sim.log. In this backdoor co-sim the host spins on the status register
      (``Processing READ (REG ...)``) while the accelerator runs, so the poll span
      approximates busy time. Empirically within ~1% of the fsdb signal, but it is
      an estimate (poll cadence is host-CPU speed, not hardware speed).

Region boundaries come from the RESET_GEN / "resuming normal operation" markers in
vcs_sim.log (one per region); region names are pulled from gem5_output.log.

Clock: testbench CyclTime = 10 ns (100 MHz). fsdb/VCD timescale is 1 ps and
vcs_sim.log timestamps are ps, so 1 cycle = 10000 ps at 100 MHz.

Usage:
    python rtl_region_cycles.py <eval_dir>                 # fsdb signal (accurate)
    python rtl_region_cycles.py <eval_dir> --method poll   # log-only estimate
    python rtl_region_cycles.py <eval_dir> --clk-mhz 100
    python rtl_region_cycles.py <eval_dir> --json

    # <eval_dir> may be the eval dir itself or its logs/rtl_runner subdir.

Example:
    python tools/rtl_region_cycles.py \
        eval_dir/resnet8_subset31_pretrained_orig_evl.baremetal
"""

import os
import re
import sys
import json
import glob
import argparse
import subprocess
import tempfile
from pathlib import Path


# Hardware busy signal (top-level wire in the testbench). 1 == accelerator busy.
BUSY_SIGNAL = 'imcflow_state_o'
TOP_SCOPE = 'testbench_imcflow_gem5'

# Region-start marker in vcs_sim.log.
_RESUME = 'Reset sequence complete, resuming normal operation'
_TS = re.compile(r'^\[(\d+)\]\s*(.*)')
_REGION_NAME = re.compile(r'imcflow_main_\d+_round_imcflow_(region\d+)')

# host is only polling the status reg -> accelerator busy (poll method).
_POLL_PREFIX = '[SV] Processing READ (REG'
_HOST_SUBSTR = (
    'Processing WRITE (DMEM', 'Processing READ (DMEM', 'Processing WRITE (IMEM',
    'Processing WRITE (REG', 'Asserting reset', 'RESET_GEN',
)


def _resolve_log_dir(eval_dir: Path) -> Path:
    if (eval_dir / 'vcs_sim.log').is_file():
        return eval_dir
    cand = eval_dir / 'logs' / 'rtl_runner'
    if (cand / 'vcs_sim.log').is_file():
        return cand
    raise FileNotFoundError(
        f"vcs_sim.log not found under {eval_dir} (looked in {eval_dir} and {cand})")


def parse_regions(log_dir: Path):
    """Return (region_starts_ps, final_ts_ps) from vcs_sim.log markers."""
    starts, final = [], 0
    with open(log_dir / 'vcs_sim.log') as f:
        for line in f:
            m = _TS.match(line)
            if not m:
                continue
            final = int(m.group(1))
            if _RESUME in m.group(2):
                starts.append(int(m.group(1)))
    starts.sort()
    return starts, final


def region_names(log_dir: Path, n: int):
    names, seen = [], set()
    p = log_dir / 'gem5_output.log'
    if p.is_file():
        with open(p) as f:
            for line in f:
                m = _REGION_NAME.search(line)
                if m and m.group(1) not in seen:
                    seen.add(m.group(1))
                    names.append(m.group(1))
    while len(names) < n:
        names.append(f'region{len(names) + 1}')
    return names[:n]


# --- fsdb method -----------------------------------------------------------
def busy_intervals_from_fsdb(log_dir: Path):
    """Convert the run's fsdb to a tiny VCD (top scope only) and return the list
    of (rise_ps, fall_ps) intervals where imcflow_state_o == 1."""
    fsdbs = glob.glob(str(log_dir / '*.fsdb'))
    if not fsdbs:
        raise FileNotFoundError(f"no .fsdb in {log_dir}")
    fsdb = max(fsdbs, key=os.path.getsize)

    with tempfile.TemporaryDirectory() as td:
        vcd = os.path.join(td, 'top.vcd')
        # -level 1: just the top scope (where the wire lives) -> fast, small.
        cmd = ['fsdb2vcd', fsdb, '-s', TOP_SCOPE, '-level', '1', '-o', vcd]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=900)

        # find the VCD id-code for BUSY_SIGNAL
        sig_id = None
        for line in open(vcd):
            if line.startswith('$var') and f' {BUSY_SIGNAL} ' in line:
                sig_id = line.split()[3]
                break
        if sig_id is None:
            raise RuntimeError(f"{BUSY_SIGNAL} not found in VCD header")

        intervals, cur, rise = [], 0, None
        rise_tok, fall_tok = '1' + sig_id, '0' + sig_id
        for line in open(vcd):
            line = line.rstrip('\n')
            if line.startswith('#'):
                cur = int(line[1:])
            elif line == rise_tok:
                rise = cur
            elif line == fall_tok and rise is not None:
                intervals.append((rise, cur))
                rise = None
        return intervals, os.path.basename(fsdb)


# --- poll method -----------------------------------------------------------
def _poll_class(rest: str) -> str:
    if rest.startswith(_POLL_PREFIX):
        return 'POLL'
    if rest.startswith('[SRAM_DIRECT]') or rest.startswith('[SV] Read data'):
        return 'HOST'
    if any(s in rest for s in _HOST_SUBSTR):
        return 'HOST'
    return ''


def busy_from_poll(log_dir: Path, s, e):
    """Estimate accelerator-busy ps in [s, e) via host poll/transfer split."""
    seq = []
    with open(log_dir / 'vcs_sim.log') as f:
        for line in f:
            m = _TS.match(line)
            if not m:
                continue
            t = int(m.group(1))
            if not (s <= t < e):
                continue
            c = _poll_class(m.group(2))
            if c:
                seq.append((t, c))
    if not seq:
        return float(e - s)
    host = comp = 0.0
    for (t, c), (t2, _c) in zip(seq, seq[1:]):
        dur = t2 - t
        if c == 'HOST':
            host += dur
        else:
            comp += dur
    head, tail = seq[0][0] - s, e - seq[-1][0]
    comp += head if seq[0][1] == 'POLL' else 0
    comp += tail if seq[-1][1] == 'POLL' else 0
    return comp


# --- driver ----------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('eval_dir', help='eval dir or its logs/rtl_runner subdir')
    ap.add_argument('--method', choices=['fsdb', 'poll'], default='fsdb',
                    help='fsdb = hardware busy signal (accurate, default); '
                         'poll = vcs_sim.log estimate (no fsdb tools needed)')
    ap.add_argument('--clk-mhz', type=float, default=100.0,
                    help='accelerator clock in MHz (default 100 = 10 ns period)')
    ap.add_argument('--json', action='store_true', help='emit JSON instead of a table')
    args = ap.parse_args(argv)

    ps_per_cycle = 1e6 / args.clk_mhz
    log_dir = _resolve_log_dir(Path(args.eval_dir))
    starts, final = parse_regions(log_dir)
    if not starts:
        print(f"error: no region markers in {log_dir}/vcs_sim.log", file=sys.stderr)
        return 2
    names = region_names(log_dir, len(starts))
    ends = starts[1:] + [final]

    src = None
    rows = []
    if args.method == 'fsdb':
        try:
            intervals, src = busy_intervals_from_fsdb(log_dir)
        except (FileNotFoundError, RuntimeError, subprocess.SubprocessError) as ex:
            print(f"error: fsdb method failed ({ex}); retry with --method poll",
                  file=sys.stderr)
            return 3
        for name, s, e in zip(names, starts, ends):
            busy = sum(min(b, e) - max(a, s)
                       for a, b in intervals if min(b, e) > max(a, s))
            npulse = sum(1 for a, b in intervals if min(b, e) > max(a, s))
            rows.append({'region': name, 'busy_ns': busy / 1000.0,
                         'busy_cycles': round(busy / ps_per_cycle), 'pulses': npulse})
    else:
        src = 'vcs_sim.log (poll estimate)'
        for name, s, e in zip(names, starts, ends):
            busy = busy_from_poll(log_dir, s, e)
            rows.append({'region': name, 'busy_ns': busy / 1000.0,
                         'busy_cycles': round(busy / ps_per_cycle), 'pulses': None})

    tb = sum(r['busy_ns'] for r in rows)
    tcyc = round(tb * 1000 / ps_per_cycle)

    if args.json:
        print(json.dumps({'method': args.method, 'source': src,
                          'clk_mhz': args.clk_mhz, 'regions': rows,
                          'total': {'busy_us': tb / 1000.0, 'busy_cycles': tcyc}},
                         indent=2))
        return 0

    has_pulse = args.method == 'fsdb'
    print(f"Method: {args.method}  ({'HW signal imcflow_state_o' if has_pulse else 'poll estimate'})")
    print(f"Source: {src}")
    print(f"Accelerator clock: {args.clk_mhz:g} MHz\n")
    hdr = f"{'Region':<10}{'busy(us)':>11}{'busy cyc':>11}" + ("{:>9}".format('#pulses') if has_pulse else "")
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        line = f"{r['region']:<10}{r['busy_ns']/1000:>11.2f}{r['busy_cycles']:>11d}"
        if has_pulse:
            line += f"{r['pulses']:>9d}"
        print(line)
    print('-' * len(hdr))
    print(f"{'TOTAL':<10}{tb/1000:>11.2f}{tcyc:>11d}")
    print()
    print(f"accelerator busy (host excluded): {tb/1000:.2f} us = {tcyc} cycles @ {args.clk_mhz:g} MHz")
    return 0


if __name__ == '__main__':
    sys.exit(main())
