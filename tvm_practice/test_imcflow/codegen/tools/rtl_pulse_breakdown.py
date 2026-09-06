#!/usr/bin/env python3
"""Busy-pulse phase breakdown (policy / imem-program / weight-load / execute): policy / imem-program / weight-load / execute.

Per busy pulse (imcflow_state_o=1), gather per-class event timestamps inside
the pulse window from fsim logs:
  policy  POLICY_WRITE          (router u_policy_table logs; runtime inode writes)
  imem    cmd=CMD_IMEM_WRITE    (IMCE ctrl: NoC imem distribution)
  imcu    cmd=CMD_IMCU_WRITE    (IMCE ctrl: weight write into IMCU array)
  exec    OP_STEP|OP_MM_QUANT|OP_MULTL|OP_ADD(vec)  (compute)
A pulse is then SPLIT at class-transition boundaries (first timestamp of the
next class) so mixed pulses (imem->weight) are attributed by actual sub-span.
Output: JSON {run: [{region, phases: {name: cycles}}]}.
"""
import sys, os, re, json, glob, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rtl_region_cycles import busy_intervals_from_fsdb, parse_regions, _resolve_log_dir
from pathlib import Path

CYC = 10000
TS = re.compile(r'^\[(\d+)\]')
CLASSES = [
    ('policy', r'POLICY_WRITE',                        '*u_policy_table.log'),
    ('imem',   r'cmd=CMD_IMEM_WRITE',                  '*imce_node.imce.u_imce_ctrl.u_ctrl_pl.log'),
    ('imcu',   r'cmd=CMD_IMCU_WRITE',                  '*imce_node.imce.u_imce_ctrl.u_ctrl_pl.log'),
    ('exec',   r'opcode=OP_STEP|opcode=OP_MM_QUANT|opcode=OP_MULTL|opcode=OP_ADD,', '*imce_node.imce.u_imce_ctrl.u_ctrl_pl.log'),
]
ORDER = ['policy', 'imem', 'imcu', 'exec']
LABEL = {'policy': 'policy-program', 'imem': 'imem-program',
         'imcu': 'weight-load', 'exec': 'execute'}

def collect(log_dir):
    ev = {}
    for name, pat, g in CLASSES:
        p = re.compile(pat)
        ts = []
        for f in glob.glob(str(log_dir / 'fsim_logs' / g)):
            for line in open(f, errors='replace'):
                m = TS.match(line)
                if m and p.search(line):
                    ts.append(int(m.group(1)))
        ev[name] = sorted(ts)
    return ev

def in_win(ts, s, e):
    i = bisect.bisect_left(ts, s); j = bisect.bisect_right(ts, e)
    return ts[i:j]

def one(eval_dir):
    ld = _resolve_log_dir(Path(eval_dir))
    iv, _ = busy_intervals_from_fsdb(ld)
    starts, final = parse_regions(ld)
    bounds = [(st, (starts[i+1] if i+1 < len(starts) else final+1))
              for i, st in enumerate(starts)]
    ev = collect(ld)
    out = []
    for ri, (rs, re_) in enumerate(bounds):
        phases = {LABEL[c]: 0 for c in ORDER}
        for (s, e) in iv:
            if not (rs <= s < re_):
                continue
            # classes present in this pulse, in canonical order
            present = [(c, in_win(ev[c], s, e)) for c in ORDER]
            present = [(c, t) for c, t in present if t]
            if not present:
                phases.setdefault('other', 0)
                phases['other'] += (e - s) // CYC
                continue
            # split pulse at first-event boundaries of each subsequent class
            cuts = [s] + [t[0] for _, t in present[1:]] + [e]
            for (c, _), a, b in zip(present, cuts, cuts[1:]):
                phases[LABEL[c]] += (b - a) // CYC
        out.append({'region': f'region{ri+1}', 'phases': phases})
    return out

if __name__ == '__main__':
    res = {}
    for spec in sys.argv[1:]:
        name, path = spec.split('=', 1)
        res[name] = one(path)
    json.dump(res, sys.stdout, indent=1)

# Usage:
#   python tools/rtl_pulse_breakdown.py <name>=<eval_dir> [...] > breakdown.json
# Pulse structure (verified on packed/off resnet8 + ds_cnn, 2026-09):
#   pulse1 = policy-program (inode runtime POLICY_WRITE to router tables)
#   pulse2 = imem-program (NoC CMD_IMEM_WRITE) then weight-load (CMD_IMCU_WRITE),
#            split at the first CMD_IMCU_WRITE timestamp
#   pulse3 = execute
