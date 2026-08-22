#!/usr/bin/env python3
"""Post-mortem stall/wedge report from IMCFlow RTL fsim logs.

Codifies the manual wedge-debugging survey: for every node it extracts the
last activity timestamp, the instruction it is stuck on, and traffic
counters, then sorts nodes by who stopped first. The earliest staller among
recently-active nodes is usually the head of the blocking chain (everything
later stalled waiting on it, directly or through NoC backpressure).

Usage:
  python tools/fsim_stall_report.py <eval_dir> [--since T] [--disasm]
      [--region region3] [--context 8]

  <eval_dir>   model eval_dir containing logs/rtl_runner/fsim_logs/
  --since T    only count traffic (STEP/RECV/SEND) at/after sim time T
               (use to isolate the wedged launch; timestamps are the [..]
               prefixes in the fsim logs)
  --disasm     print disassembly context around each stuck node's pc
               (needs the build dir; pick it with --region)
  --region S   substring selecting the build dir for --disasm
               (default: last build dir alphabetically, usually the last region)
  --context N  disasm lines around the stuck pc (default 8)

Reading the output:
  - Nodes are sorted by last-activity time, earliest (first staller) first.
  - IDLE-tagged nodes (last activity long before the global max) usually just
    weren't enabled in the wedged launch -- ignore them.
  - inode STANDBY op1: 254/255 = all-inode barrier, small value = per-word
    data rendezvous with an imce flag.
  - imce holding_opcode comes from the last STALL_START event.
"""

import argparse
import glob
import os
import re
import subprocess
import sys

FSIM_RE = re.compile(r"core_row_(\d)_\.core_col_(\d)_")
TIME_RE = re.compile(rb"^\[\s*(\d+)\]")
IMCE_EXEC_RE = re.compile(rb"\[\s*(\d+)\] \| EXECUTE \| \{pc=(\d+), opcode=(\w+)")
FIFO_RE = re.compile(rb"fifo_id=(\d+)")
IMCE_STALL_RE = re.compile(rb"STALL_START \| \{holding_opcode=(\w*)\}")
INODE_EXEC_RE = re.compile(
    rb"\[\s*(\d+)\] \| EX_(?:START|END) \| \{opcode=(\w+), op1=(-?\d+), op2=(-?\d+).*?pc=(\d+)")

DEFAULT_OBJDUMP = "/root/project/llvm-project/builddir/bin/llvm-objdump"


def node_name(row, col):
    return f"inode_{row}_0" if col == 0 else f"imce_{row}_{col}"


def parse_imce(path, since):
    last = {"time": -1, "pc": None, "op": None}
    holding = ""
    steps = stops = sends = standbys = setflags = 0
    recv_fifo = {}
    with open(path, "rb") as f:
        for line in f:
            m = IMCE_EXEC_RE.search(line)
            if m:
                t = int(m.group(1))
                op = m.group(3).decode()
                last = {"time": t, "pc": int(m.group(2)), "op": op}
                if t < since:
                    continue
                if op == "OP_STEP":
                    steps += 1
                elif op == "OP_STOP":
                    stops += 1
                elif op == "OP_SEND":
                    sends += 1
                elif op == "OP_STANDBY":
                    standbys += 1
                elif op == "OP_SET_FLAG":
                    setflags += 1
                elif op == "OP_RECV":
                    fm = FIFO_RE.search(line)
                    if fm:
                        fid = int(fm.group(1))
                        recv_fifo[fid] = recv_fifo.get(fid, 0) + 1
                continue
            m = IMCE_STALL_RE.search(line)
            if m:
                holding = m.group(1).decode()
    return {"last": last, "holding": holding, "steps": steps, "stops": stops,
            "sends": sends, "standbys": standbys, "setflags": setflags,
            "recv_fifo": recv_fifo}


def parse_inode(path, since):
    last = {"time": -1, "pc": None, "op": None, "op1": None}
    sends = recvs = 0
    last_standby_op1 = None
    recent_pcs = []
    with open(path, "rb") as f:
        for line in f:
            m = INODE_EXEC_RE.search(line)
            if not m:
                continue
            t = int(m.group(1))
            op = m.group(2).decode()
            pc = int(m.group(5))
            last = {"time": t, "pc": pc, "op": op, "op1": int(m.group(3))}
            recent_pcs.append(pc)
            if len(recent_pcs) > 60:
                recent_pcs.pop(0)
            if t < since:
                continue
            if op == "OP_SEND":
                sends += 1
            elif op == "OP_RECV":
                recvs += 1
            if op == "OP_STANDBY":
                last_standby_op1 = int(m.group(3))
    lo = min(recent_pcs) if recent_pcs else None
    hi = max(recent_pcs) if recent_pcs else None
    return {"last": last, "sends": sends, "recvs": recvs,
            "last_standby_op1": last_standby_op1, "pc_window": (lo, hi)}


def disasm_context(build_dir, name, pc, context, objdump):
    out = os.path.join(build_dir, f"{name}_imem.out")
    if not os.path.isfile(out):
        return f"  (no {out})"
    try:
        txt = subprocess.run([objdump, "-d", out], capture_output=True,
                             text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"  (objdump failed: {e})"
    target = f"{pc * 4:x}:"
    lines = txt.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith(target):
            lo = max(0, i - context // 2)
            block = lines[lo:i] + [ln + "   <=== STUCK HERE"] + lines[i + 1:i + 1 + context // 2]
            return "\n".join("  " + b for b in block)
    return f"  (pc {pc} / byte addr {pc*4:#x} not found in {os.path.basename(out)})"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dir")
    ap.add_argument("--since", type=int, default=0)
    ap.add_argument("--disasm", action="store_true")
    ap.add_argument("--region", default=None)
    ap.add_argument("--context", type=int, default=8)
    ap.add_argument("--objdump", default=os.environ.get("IMCFLOW_LLVM_OBJDUMP", DEFAULT_OBJDUMP))
    args = ap.parse_args()

    ev = os.path.abspath(args.eval_dir)
    fsim = os.path.join(ev, "logs", "rtl_runner", "fsim_logs")
    if not os.path.isdir(fsim):
        sys.exit(f"no fsim_logs at {fsim}")

    rows = []
    for path in glob.glob(os.path.join(fsim, "*u_ctrl_pl.log")):
        m = FSIM_RE.search(path)
        if m:
            info = parse_imce(path, args.since)
            rows.append(("imce", node_name(int(m.group(1)), int(m.group(2))), info))
    for path in glob.glob(os.path.join(fsim, "*inode.u_intf_node.ex_stage.log")):
        m = FSIM_RE.search(path)
        if m:
            info = parse_inode(path, args.since)
            rows.append(("inode", node_name(int(m.group(1)), int(m.group(2))), info))

    if not rows:
        sys.exit(f"no node logs matched under {fsim}")

    rows.sort(key=lambda r: r[2]["last"]["time"])
    tmax = max(r[2]["last"]["time"] for r in rows)
    idle_cut = tmax - max(1, int(0.10 * tmax))  # inactive in final 10% => likely idle
    head = None

    print(f"# stall report: {ev}")
    if args.since:
        print(f"# traffic counted since t={args.since}")
    print(f"# global last activity t={tmax}; sorted earliest-staller first\n")
    hdr = f"{'node':<11} {'last_t':>13} {'pc':>4} {'stuck_on':<12} {'detail'}"
    print(hdr)
    print("-" * len(hdr))
    for kind, name, info in rows:
        last = info["last"]
        tag = " IDLE?" if last["time"] < idle_cut else ""
        if kind == "imce":
            fifo = ",".join(f"f{k}:{v}" for k, v in sorted(info["recv_fifo"].items()))
            detail = (f"STEP={info['steps']} STOP={info['stops']} SEND={info['sends']} "
                      f"STBY={info['standbys']} SETF={info['setflags']} RECV[{fifo}]")
            stuck = info["holding"] or (last["op"] or "?")
        else:
            sb = info["last_standby_op1"]
            sbs = ("barrier" if sb in (254, 255) else f"flag={sb}") if sb is not None else "-"
            lo, hi = info["pc_window"]
            detail = (f"SEND={info['sends']} RECV={info['recvs']} lastSTBY={sbs} "
                      f"recent_pc=[{lo}..{hi}]")
            stuck = f"{last['op']}({last['op1']})" if last["op"] else "?"
        print(f"{name:<11} {last['time']:>13} {str(last['pc']):>4} {stuck:<12} {detail}{tag}")
        if head is None and not tag and last["time"] >= 0:
            head = (name, last)
    if head:
        print(f"\n>>> suspected head of blocking chain: {head[0]} "
              f"(first non-idle staller, stuck at pc={head[1]['pc']} on {head[1]['op']})")

    if args.disasm:
        builds = sorted(glob.glob(os.path.join(ev, "build", "*", "")))
        if args.region:
            builds = [b for b in builds if args.region in b]
        if not builds:
            sys.exit("no build dir for --disasm (check --region)")
        bd = builds[-1]
        print(f"\n# disasm context from {bd}")
        for kind, name, info in rows:
            last = info["last"]
            if last["time"] < idle_cut or last["pc"] is None:
                continue
            print(f"\n== {name} pc={last['pc']} ({last['op']}) ==")
            print(disasm_context(bd, name, last["pc"], args.context, args.objdump))


if __name__ == "__main__":
    main()
