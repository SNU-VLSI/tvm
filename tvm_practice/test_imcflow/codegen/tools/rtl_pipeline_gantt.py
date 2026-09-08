#!/usr/bin/env python3
"""Per-region pipeline Gantt from RTL fsim logs (+ latency cross-check).

For each region's EXECUTE busy pulse (imcflow_state_o=1 window containing
compute opcodes), extract every IMCE's activity from its ctrl-pipeline fsim
log and draw an overlapped timeline (one row per core):

  compute   EXECUTE of OP_STEP/OP_MM_QUANT (IMCU) or vector ALU ops (VPU)
  noc       EXECUTE of OP_SEND/OP_RECV (data actually moving)
  wait      STALL_START intervals + OP_STANDBY/OP_SET_FLAG (blocked)

Attribution: interval [t_i, t_{i+1}) belongs to the event at t_i.
Rows are annotated with the cell's op label from HWNodeMap when available.

Cross-check: for each region, the longest per-core active span must equal the
fsdb execute-window length (pipeline ends when the last core stops); the table
prints both and the per-core busy fraction.

Usage:
  python tools/rtl_pipeline_gantt.py <eval_dir> -o out.png [--regions 1,2]
"""
import sys, os, re, glob, bisect, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rtl_region_cycles import busy_intervals_from_fsdb, parse_regions, _resolve_log_dir
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CYC = 10000
LINE = re.compile(r'^\[(\d+)\]\s*\|\s*(EXECUTE|STALL_START)\s*\|.*?(?:opcode=|holding_opcode=)(OP_[A-Z_0-9]*)?')
EXEC_MARK = re.compile(r'opcode=OP_STEP|opcode=OP_MM_QUANT|opcode=OP_MULTL|opcode=OP_ADD,')
IMCU = {'OP_STEP', 'OP_MM_QUANT'}
VPU = {'OP_MULTL', 'OP_MULTH', 'OP_ADD', 'OP_SUB', 'OP_XOR', 'OP_AND', 'OP_OR',
       'OP_SRAI', 'OP_SRLI', 'OP_SLLI', 'OP_MULTI'}
NOC = {'OP_SEND', 'OP_RECV'}
GCOL = {'compute': '#4878a8', 'noc': '#e8a33d', 'wait': '#e3e3e3'}

def group(kind, op):
    if kind == 'STALL_START':
        return 'wait'
    if op in IMCU or op in VPU:
        return 'compute'
    if op in NOC:
        return 'noc'
    if op in ('OP_STANDBY', 'OP_SET_FLAG'):
        return 'wait'
    return 'compute'   # ctrl ops count as pipeline-active

def core_events(log_dir):
    ev = {}
    for f in glob.glob(str(log_dir / 'fsim_logs' / '*imce_node.imce.u_imce_ctrl.u_ctrl_pl.log')):
        m = re.search(r'core_row_(\d)_\.core_col_(\d)_', f)
        core = f"imce_{m.group(1)}_{m.group(2)}" if m else os.path.basename(f)
        lst = []
        for line in open(f, errors='replace'):
            mm = LINE.match(line)
            if mm:
                lst.append((int(mm.group(1)), mm.group(2), mm.group(3) or ''))
        if lst:
            ev[core] = lst
    return ev

def exec_windows(log_dir, ev):
    """Execute pulse = LAST busy pulse inside each region's marker bounds
    (pulse1=policy, pulse2=imem+weight, pulse3=execute; see rtl_pulse_breakdown)."""
    iv, _ = busy_intervals_from_fsdb(log_dir)
    starts, final = parse_regions(log_dir)
    bounds = [(st, (starts[i+1] if i+1 < len(starts) else final+1))
              for i, st in enumerate(starts)]
    wins = []
    for rs, re_ in bounds:
        inside = [(s, e) for (s, e) in iv if rs <= s < re_]
        if inside:
            wins.append(inside[-1])
    return wins

def segments(lst, s, e):
    """Merged (start, end, group) segments for one core inside [s,e]."""
    keys = [t for (t, _, _) in lst]
    i = bisect.bisect_left(keys, s)
    if i > 0:
        cur_t, cur_g = s, group(lst[i-1][1], lst[i-1][2])
    elif i < len(lst):
        cur_t, cur_g = lst[i][0], group(lst[i][1], lst[i][2])
        i += 1
    else:
        return [], None, None
    segs, first_act, last_act = [], None, None
    def push(a, b, g):
        nonlocal first_act, last_act
        if b <= a:
            return
        if segs and segs[-1][2] == g and segs[-1][1] == a:
            segs[-1] = (segs[-1][0], b, g)
        else:
            segs.append((a, b, g))
        if g != 'wait':
            if first_act is None:
                first_act = a
            last_act = b
    stopped_at = None
    while i < len(lst) and lst[i][0] <= e:
        push(cur_t, lst[i][0], cur_g)
        cur_t = lst[i][0]
        cur_g = group(lst[i][1], lst[i][2])
        if lst[i][2] == 'OP_STOP' and stopped_at is None:
            stopped_at = lst[i][0]
        i += 1
    push(cur_t, stopped_at if stopped_at else e, cur_g)
    return segs, first_act, (stopped_at or last_act)

def cell_labels(eval_dir):
    """{region_idx: {core: 'op-chain'}} from generated imce.cpp markers
    (noc_mapping_viz.parse_impl_ops keys sections by imce_<hid>_<wid>)."""
    try:
        from noc_mapping_viz import parse_impl_ops
        impl = parse_impl_ops(str(eval_dir))
        out = {}
        for func, nodes in impl.items():
            m = re.search(r'region(\d+)', func)
            if not m:
                continue
            ri = int(m.group(1)) - 1
            out[ri] = {core: '+'.join(tags)[:22] for core, tags in nodes.items()}
        return out
    except Exception:
        return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('eval_dir'); ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--regions', default='')
    ap.add_argument('--render', choices=['dominant', 'presence'], default='presence',
                    help="bin coloring: 'dominant' = largest-share class per bin "
                         "(thin mm/add bursts vanish); 'presence' = priority "
                         "compute>noc>wait, so any compute in a bin shows (default)")
    a = ap.parse_args()
    ld = _resolve_log_dir(Path(a.eval_dir))
    ev = core_events(ld)
    wins = exec_windows(ld, ev)
    sel = [int(x) - 1 for x in a.regions.split(',')] if a.regions else range(len(wins))
    wins = [wins[i] for i in sel]
    labels = cell_labels(a.eval_dir)

    fig, axes = plt.subplots(len(wins), 1, figsize=(13, 3.2 * len(wins)), squeeze=False)
    print(f"{'region':<9}{'exec-window':>12}{'max-span':>10}{'coverage':>10}{'match':>7}   per-core busy% (non-wait)")
    for ri, (s, e) in enumerate(wins):
        ax = axes[ri][0]
        rows = []
        for core in sorted(ev):
            segs, fa, la = segments(ev[core], s, e)
            if segs and any(g != 'wait' for _, _, g in segs):
                rows.append((core, segs, fa, la))
        max_span = 0
        last_end = s
        busys = []
        NB = 3000  # visual bins per window (dominant class per bin)
        binw = max(1, (e - s) // NB)
        for yi, (core, segs, fa, la) in enumerate(rows):
            import numpy as np
            acc = {g: np.zeros(NB + 1) for g in GCOL}
            for (a0, b0, g) in segs:
                i0, i1 = int((a0 - s) // binw), int((b0 - s) // binw)
                if i0 == i1:
                    acc[g][min(i0, NB)] += b0 - a0
                else:
                    acc[g][min(i0, NB)] += (i0 + 1) * binw - (a0 - s)
                    acc[g][min(i1, NB)] += (b0 - s) - i1 * binw
                    if i1 > i0 + 1:
                        acc[g][min(i0, NB)+1:min(i1, NB)] += binw
            names = list(GCOL)
            stack = np.stack([acc[g] for g in names])
            if a.render == 'presence':
                # priority compute > noc > wait: a bin shows compute if ANY
                # compute cycles landed in it (keeps 2-6cyc mm/add bursts visible)
                dom = np.full(NB + 1, -1)
                for gi in reversed(range(len(names))):   # wait, noc, compute
                    dom = np.where(stack[gi] > 0, gi, dom)
            else:
                dom = np.where(stack.sum(0) > 0, stack.argmax(0), -1)
            # runs of equal dominant class -> one broken_barh batch per class
            spans = {g: [] for g in names}
            j = 0
            while j <= NB:
                if dom[j] < 0:
                    j += 1
                    continue
                k = j
                while k + 1 <= NB and dom[k + 1] == dom[j]:
                    k += 1
                spans[names[dom[j]]].append((j * binw / CYC, (k - j + 1) * binw / CYC))
                j = k + 1
            for g, sp in spans.items():
                if sp:
                    ax.broken_barh(sp, (yi - 0.38, 0.76), facecolors=GCOL[g], edgecolor='none')
            span = (la - fa) if fa is not None else 0
            max_span = max(max_span, span)
            if la:
                last_end = max(last_end, la)
            act = sum(b0 - a0 for a0, b0, g in segs if g != 'wait')
            busys.append(f"{core.split('_',1)[1]}:{100*act/(e-s):.0f}")
        win = (e - s) // CYC
        cover = (last_end - s) // CYC
        ok = abs(cover - win) <= max(5, win // 50)
        print(f"region{ri+1:<3}{win:>12}{max_span//CYC:>10}{cover:>10}{'OK' if ok else 'DIFF':>7}   {' '.join(busys)}")
        ax.set_yticks(range(len(rows)))
        rlab = labels.get(ri, {})
        ax.set_yticklabels([f"{c}  {rlab.get(c,'')}" for c, _, _, _ in rows], fontsize=7)
        ax.set_xlim(0, win)
        ax.set_xlabel(f'cycles from execute-window start (window = {win} cyc = {win/100:.1f} µs @100MHz)',
                      fontsize=8)
        ax.set_title(f'region{ri+1} pipeline occupancy (blue=compute, orange=NoC xfer, grey=wait)',
                     fontsize=9)
        ax.grid(axis='x', alpha=0.25)
        ax.invert_yaxis()
    fig.suptitle(os.path.basename(os.path.normpath(a.eval_dir)), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(a.out, dpi=150)
    print(f'saved {a.out}')

if __name__ == '__main__':
    main()
