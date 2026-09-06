#!/usr/bin/env python3
"""Execute-phase composition profiler for IMCFlow RTL co-sim.

Within each region's EXECUTE busy pulse (see rtl_pulse_breakdown.py), walk every
IMCE ctrl-pipeline fsim log and attribute wall time per core to opcode classes:

  imcu       OP_STEP / OP_MM_QUANT               (conv array + quantize)
  vpu        OP_MULTL/MULTH/ADD/SUB/XOR/AND/OR/SRAI/SLLI  (vector post-ops)
  noc-wait   OP_SEND / OP_RECV (issue + stall holding them)
  sync-wait  OP_STANDBY / OP_SET_FLAG
  ctrl       ADDI/BNE/JUMP/NOP/etc.
  done       after OP_STOP until window end

Attribution: consecutive log events (t_i -> t_{i+1}) charge [t_i, t_{i+1}) to
the class of the event at t_i (EXECUTE by its opcode; STALL_START by its
holding_opcode). Idle cores (no events in window) are excluded.

Usage:
  python tools/rtl_exec_profile.py <eval_dir> [--json]
  python tools/rtl_exec_profile.py name1=<eval_dir1> name2=<eval_dir2> ... --json
"""
import sys, os, re, json, glob, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rtl_region_cycles import busy_intervals_from_fsdb, parse_regions, _resolve_log_dir
from pathlib import Path

CYC = 10000  # ps per cycle @100MHz
LINE = re.compile(r'^\[(\d+)\]\s*\|\s*(EXECUTE|STALL_START)\s*\|.*?(?:opcode=|holding_opcode=)(OP_[A-Z_0-9]*)?')
EXEC_MARK = re.compile(r'opcode=OP_STEP|opcode=OP_MM_QUANT|opcode=OP_MULTL|opcode=OP_ADD,')

IMCU = {'OP_STEP', 'OP_MM_QUANT'}
VPU = {'OP_MULTL', 'OP_MULTH', 'OP_ADD', 'OP_SUB', 'OP_XOR', 'OP_AND', 'OP_OR',
       'OP_SRAI', 'OP_SRLI', 'OP_SLLI', 'OP_MULTI'}
NOC = {'OP_SEND', 'OP_RECV'}
SYNC = {'OP_STANDBY', 'OP_SET_FLAG'}

def clsof(op):
    if op in IMCU: return 'imcu'
    if op in VPU: return 'vpu'
    if op in NOC: return 'noc-wait'
    if op in SYNC: return 'sync-wait'
    if op == 'OP_STOP': return 'done'
    return 'ctrl'

def exec_windows(log_dir):
    """EXECUTE pulses = busy pulses containing compute opcodes."""
    iv, _ = busy_intervals_from_fsdb(log_dir)
    marks = []
    for f in glob.glob(str(log_dir / 'fsim_logs' / '*imce_node.imce.u_imce_ctrl.u_ctrl_pl.log')):
        for line in open(f, errors='replace'):
            if EXEC_MARK.search(line):
                m = LINE.match(line)
                if m:
                    marks.append(int(m.group(1)))
    marks.sort()
    wins = []
    for s, e in iv:
        i = bisect.bisect_left(marks, s)
        if i < len(marks) and marks[i] <= e:
            wins.append((s, e))
    return wins

def profile(eval_dir):
    ld = _resolve_log_dir(Path(eval_dir))
    wins = exec_windows(ld)
    agg = {}
    total_win_cyc = sum((e - s) // CYC for s, e in wins)
    per_core = {}
    for f in glob.glob(str(ld / 'fsim_logs' / '*imce_node.imce.u_imce_ctrl.u_ctrl_pl.log')):
        core = re.search(r'core_row_(\d)_\.core_col_(\d)_', f)
        core = f"imce_{core.group(1)}_{core.group(2)}" if core else os.path.basename(f)
        ev = []
        for line in open(f, errors='replace'):
            m = LINE.match(line)
            if m:
                ev.append((int(m.group(1)), m.group(3) or ''))
        if not ev:
            continue
        acct = {}
        for s, e in wins:
            i = bisect.bisect_left(ev, (s, ''))
            # event holding the pipe at window start
            if i > 0:
                cur_t, cur_op = s, ev[i - 1][1]
            elif i < len(ev):
                cur_t, cur_op = ev[i][0], ev[i][1]
                i += 1
            else:
                continue
            stopped = False
            while i < len(ev) and ev[i][0] <= e:
                c = clsof(cur_op)
                acct[c] = acct.get(c, 0) + (ev[i][0] - cur_t)
                cur_t, cur_op = ev[i]
                if cur_op == 'OP_STOP':
                    stopped = True
                i += 1
            c = 'done' if stopped else clsof(cur_op)
            acct[c] = acct.get(c, 0) + (e - cur_t)
        if any(v for k, v in acct.items() if k != 'done'):
            per_core[core] = {k: v // CYC for k, v in acct.items()}
            for k, v in acct.items():
                agg[k] = agg.get(k, 0) + v // CYC
    return {'exec_windows_cyc': total_win_cyc, 'n_active_cores': len(per_core),
            'core_cycles': agg, 'per_core': per_core}

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--json']
    as_json = '--json' in sys.argv
    res = {}
    for spec in args:
        name, path = spec.split('=', 1) if '=' in spec else (os.path.basename(spec), spec)
        res[name] = profile(path)
    if as_json:
        json.dump(res, sys.stdout, indent=1)
    else:
        for name, r in res.items():
            tot = sum(r['core_cycles'].values()) or 1
            print(f"\n{name}: exec window {r['exec_windows_cyc']} cyc x {r['n_active_cores']} cores")
            for k in ('imcu', 'vpu', 'noc-wait', 'sync-wait', 'ctrl', 'done'):
                v = r['core_cycles'].get(k, 0)
                print(f"  {k:<10} {v:>10} core-cyc  {100*v/tot:5.1f}%")
