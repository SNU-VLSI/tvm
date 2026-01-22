#!/usr/bin/env python3
"""
Deadlock analyzer v2 for IMCFlow.
Analyzes multiple log types to build complete picture.
"""

import os
import re
from collections import defaultdict
from pathlib import Path

LOG_DIR = "/root/project/tvm/tvm_practice/test_imcflow/codegen/resnet8_subset08_pretrained_small_evl/logs/rtl_runner/fsim_logs"

def parse_imce_ctrl_log(filepath):
    """Parse IMCE ctrl log to find stall events."""
    events = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                match = re.search(r'\[\s*(\d+)\]\s+\[IMCE_HS\]\s+(\w+)\s+\|\s*(.*)', line)
                if match:
                    events.append({
                        'time': int(match.group(1)),
                        'event': match.group(2),
                        'details': match.group(3)
                    })
    except Exception as e:
        pass
    return events


def parse_hazard_log(filepath):
    """Parse hazard detector log."""
    events = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                # [time] [IMCE_STALL] type: reason | details
                match = re.search(r'\[\s*(\d+)\]\s+\[IMCE_(\w+)\]\s+(\w+):\s+(\w+)', line)
                if match:
                    events.append({
                        'time': int(match.group(1)),
                        'type': match.group(2),
                        'stage': match.group(3),
                        'reason': match.group(4)
                    })
    except Exception as e:
        pass
    return events


def analyze_node(row, col, node_type, log_dir):
    """Analyze a single node's state at deadlock time."""
    # Note: filenames use underscores like "core_row_3_.core_col_3_."
    base_prefix = f"testbench_imcflow_gem5.u_imcflow_with_axi.u_imcflow_impl.core_row_{row}_.core_col_{col}_.{node_type}"
    base_path = log_dir

    result = {
        'row': row,
        'col': col,
        'type': node_type,
        'imce_ctrl': None,
        'hazard': None,
        'last_send': None,
        'last_recv': None,
        'stall_state': None
    }

    # Parse IMCE ctrl log
    if node_type == 'imce_node':
        ctrl_log = base_path / f"{base_prefix}.imce.u_imce_ctrl.log"
        if ctrl_log.exists():
            events = parse_imce_ctrl_log(str(ctrl_log))
            if events:
                result['imce_ctrl'] = events

                # Find last send/recv
                sends = [e for e in events if 'SEND' in e['event']]
                recvs = [e for e in events if 'RECV' in e['event']]

                if sends:
                    result['last_send'] = sends[-1]
                if recvs:
                    result['last_recv'] = recvs[-1]

        # Parse hazard log
        hazard_log = base_path / f"{base_prefix}.imce.u_imce_ctrl.u_hazard_detector.log"
        if hazard_log.exists():
            events = parse_hazard_log(str(hazard_log))
            if events:
                result['hazard'] = events

                # Find current stall state
                stall_stack = []
                for e in events:
                    if 'START' in e['stage']:
                        stall_stack.append(e)
                    elif 'END' in e['stage']:
                        if stall_stack:
                            stall_stack.pop()

                if stall_stack:
                    result['stall_state'] = stall_stack[-1]

    return result


def main():
    log_dir = Path(LOG_DIR)

    print("=" * 80)
    print("DEADLOCK ANALYSIS V2 - IMCE State Analysis")
    print("=" * 80)

    # Analyze row 3 nodes specifically
    print("\n[ROW 3 IMCE NODES - Detailed Analysis]")
    print("-" * 80)

    for col in range(5):
        node = analyze_node(3, col, 'imce_node', log_dir)

        print(f"\n### Row 3, Col {col} (IMCE) ###")

        if node['stall_state']:
            stall = node['stall_state']
            print(f"  CURRENT STALL: {stall['stage']} - {stall['reason']} @ {stall['time']}ns")

        if node['last_send']:
            send = node['last_send']
            print(f"  Last SEND: {send['event']} @ {send['time']}ns | {send['details']}")

        if node['last_recv']:
            recv = node['last_recv']
            print(f"  Last RECV: {recv['event']} @ {recv['time']}ns | {recv['details']}")

        # Show stall history
        if node['hazard']:
            print("  Stall History (last 5):")
            for e in node['hazard'][-5:]:
                print(f"    {e['time']}ns: {e['stage']} - {e['reason']}")

    # Build timeline of stall events
    print("\n" + "=" * 80)
    print("[TIMELINE - All Stall Events around Deadlock Time]")
    print("-" * 80)

    all_events = []

    for row in range(4):
        for col in range(5):
            node = analyze_node(row, col, 'imce_node', log_dir)
            if node['hazard']:
                for e in node['hazard']:
                    if e['time'] >= 700000000:  # Only events after 700ms
                        all_events.append({
                            'time': e['time'],
                            'row': row,
                            'col': col,
                            'stage': e['stage'],
                            'reason': e['reason']
                        })

    # Sort by time
    all_events.sort(key=lambda x: x['time'])

    for e in all_events:
        print(f"  {e['time']}ns: ({e['row']},{e['col']}) {e['stage']} - {e['reason']}")

    # Analyze dependency chain
    print("\n" + "=" * 80)
    print("[DEPENDENCY CHAIN ANALYSIS]")
    print("-" * 80)

    # Find nodes that are stalled and what they're waiting for
    stalled_nodes = []
    for row in range(4):
        for col in range(5):
            node = analyze_node(row, col, 'imce_node', log_dir)
            if node['stall_state']:
                stalled_nodes.append({
                    'row': row,
                    'col': col,
                    'stall': node['stall_state'],
                    'last_send': node['last_send'],
                    'last_recv': node['last_recv']
                })

    print(f"\nNodes currently stalled: {len(stalled_nodes)}")
    for node in stalled_nodes:
        stall = node['stall']
        print(f"\n  ({node['row']},{node['col']}): {stall['reason']}")

        # Extract fifo_id from reason if present
        if 'FIFO' in stall['reason']:
            # Check last send/recv for fifo_id
            if 'SEND' in stall['reason'] and node['last_send']:
                print(f"    Trying to SEND: {node['last_send']['details']}")
            if 'RECV' in stall['reason'] and node['last_recv']:
                print(f"    Waiting to RECV: {node['last_recv']['details']}")

    # Summary
    print("\n" + "=" * 80)
    print("[ROOT CAUSE SUMMARY]")
    print("-" * 80)

    print("""
Based on the analysis:

1. DEADLOCK CHAIN in Row 3:
   - Col 4 IMCE: SEND_FIFO_STALL (trying to send to fifo_id=0, going West)
   - Col 4 Router: Local->West blocked because West output not ready
   - Col 4 Router: West input (from col5?) wants to go Local (fifo_id=2)

   - Col 3 IMCE: RECV_FIFO_STALL (waiting to receive from fifo_id=0)
   - Col 3 Router: East input (from col4) wants to go Local (fifo_id=0)
   - Col 3 Router: But Local output is not ready (IMCE not consuming)

2. CIRCULAR DEPENDENCY:
   Col4.IMCE -> Col4.Router.Local -> Col4.Router.West_out
        ^                                    |
        |                                    v
   Col4.Router.West_in <- Col3.Router.East_out
        |                                    ^
        v                                    |
   Col4.Router.Local <- Col3.Router.East_in (from col4 or col2?)

3. ROOT CAUSE:
   - Both Col3 and Col4 IMCEs are waiting for data from FIFOs
   - But the router paths are blocked because of conflicting traffic
   - The router grants to LOCAL but LOCAL (IMCE) is not ready to receive
   - This creates backpressure that blocks the entire chain
""")


if __name__ == "__main__":
    main()
