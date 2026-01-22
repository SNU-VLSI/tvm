#!/usr/bin/env python3
"""
Deadlock analyzer for IMCFlow arbiter logs.
Parses arbiter logs and identifies circular dependency chains.
"""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

LOG_DIR = "/root/project/tvm/tvm_practice/test_imcflow/codegen/resnet8_subset08_pretrained_small_evl/logs/rtl_runner/fsim_logs"

def parse_arbiter_log_tail(filepath, num_lines=200):
    """Parse the tail of an arbiter log file and extract the last cycle's state."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()[-num_lines:]
    except Exception as e:
        return None

    # Find the last complete cycle
    cycles = []
    current_cycle = None

    current_section = None

    for line in lines:
        if "========== ARBITER CYCLE ==========" in line:
            if current_cycle:
                cycles.append(current_cycle)
            # Extract timestamp
            match = re.search(r'\[\s*(\d+)\]', line)
            timestamp = int(match.group(1)) if match else 0
            current_cycle = {
                'timestamp': timestamp,
                'requests': [],
                'grants': [],
                'transferred': [],
                'blocked': []
            }
            current_section = None
        elif current_cycle:
            if "REQUESTS:" in line:
                current_section = 'requests'
            elif "GRANTS:" in line:
                current_section = 'grants'
            elif "TRANSFERRED:" in line:
                current_section = 'transferred'
            elif "BLOCKED:" in line:
                current_section = 'blocked'
            elif "PRIORITY" in line:
                current_section = None
            elif current_section and line.strip().startswith('['):
                # Parse request/grant/transfer/block entry
                # Format: [timestamp]   DIR: details
                match = re.search(r'\]\s+([LNESW]):', line)
                if match:
                    direction = match.group(1)
                    ready_match = re.search(r'ready=(\d)', line)
                    ready = ready_match.group(1) == '1' if ready_match else None
                    dst_match = re.search(r'dst(?:_vec)?=([LNESW]+)', line)
                    dst = dst_match.group(1) if dst_match else None
                    fifo_match = re.search(r'fifo_id=(\d+)', line)
                    fifo_id = int(fifo_match.group(1)) if fifo_match else None
                    cmd_match = re.search(r'cmd=(\w+)', line)
                    cmd = cmd_match.group(1) if cmd_match else None

                    entry = {
                        'dir': direction,
                        'ready': ready,
                        'dst': dst,
                        'fifo_id': fifo_id,
                        'cmd': cmd
                    }
                    current_cycle[current_section].append(entry)

    if current_cycle:
        cycles.append(current_cycle)

    return cycles[-1] if cycles else None


def extract_node_info(filename):
    """Extract row, col, and node type from filename."""
    # Pattern: core_row_X_.core_col_Y_.(imce_node|inode)
    match = re.search(r'core_row_(\d+)_\.core_col_(\d+)_\.(\w+)', filename)
    if match:
        return {
            'row': int(match.group(1)),
            'col': int(match.group(2)),
            'type': match.group(3)
        }
    return None


def analyze_deadlock():
    """Analyze all arbiter logs and find deadlock chains."""
    log_dir = Path(LOG_DIR)
    arbiter_logs = list(log_dir.glob("*arbiter.log"))

    print(f"Found {len(arbiter_logs)} arbiter logs")
    print("=" * 80)

    # Parse all arbiter logs
    node_states = {}

    for log_path in arbiter_logs:
        node_info = extract_node_info(log_path.name)
        if not node_info:
            continue

        cycle = parse_arbiter_log_tail(str(log_path))
        if cycle:
            key = (node_info['row'], node_info['col'], node_info['type'])
            node_states[key] = {
                'info': node_info,
                'cycle': cycle,
                'path': log_path.name
            }

    # Find nodes with pending requests (granted but not transferred)
    print("\n[DEADLOCK ANALYSIS - Last Cycle State]")
    print("=" * 80)

    deadlock_nodes = []

    for key, state in sorted(node_states.items()):
        row, col, node_type = key
        cycle = state['cycle']

        # Check if there are requests that are granted but not transferred
        granted_not_transferred = []
        for grant in cycle['grants']:
            transferred = any(t['dir'] == grant['dir'] for t in cycle['transferred'])
            if not transferred:
                # Find corresponding request for more details
                req = next((r for r in cycle['requests'] if r['dir'] == grant['dir']), None)
                if req:
                    granted_not_transferred.append({
                        'from': grant['dir'],
                        'to': grant['dst'],
                        'ready': req['ready'],
                        'fifo_id': req.get('fifo_id'),
                        'cmd': req.get('cmd')
                    })

        if granted_not_transferred:
            deadlock_nodes.append({
                'row': row,
                'col': col,
                'type': node_type,
                'timestamp': cycle['timestamp'],
                'stalled': granted_not_transferred
            })

    # Print deadlock nodes
    print(f"\nNodes with stalled transfers (Granted but not Transferred):")
    print("-" * 80)

    for node in sorted(deadlock_nodes, key=lambda x: (x['row'], x['col'])):
        print(f"\n[Row {node['row']}, Col {node['col']}] ({node['type']}) @ {node['timestamp']}ns")
        for stall in node['stalled']:
            fifo_info = f", fifo_id={stall['fifo_id']}" if stall['fifo_id'] is not None else ""
            cmd_info = f", cmd={stall['cmd']}" if stall['cmd'] else ""
            print(f"  {stall['from']} -> {stall['to']}: ready={stall['ready']}{fifo_info}{cmd_info}")

    # Build dependency graph
    print("\n" + "=" * 80)
    print("[DEPENDENCY GRAPH]")
    print("-" * 80)

    # Direction to coordinate offset mapping
    dir_to_offset = {
        'N': (-1, 0),  # North = row-1
        'S': (1, 0),   # South = row+1
        'E': (0, 1),   # East = col+1
        'W': (0, -1),  # West = col-1
        'L': (0, 0)    # Local = same node
    }

    dependencies = []

    for node in deadlock_nodes:
        row, col = node['row'], node['col']
        for stall in node['stalled']:
            src_dir = stall['from']
            dst_dir = stall['to']

            # Where is the source coming from?
            src_offset = dir_to_offset.get(src_dir, (0, 0))
            src_row = row - src_offset[0]  # Reverse because incoming
            src_col = col - src_offset[1]

            # Where is it trying to go?
            if dst_dir == 'L':
                dst_row, dst_col = row, col
                dst_desc = f"LOCAL (fifo_id={stall.get('fifo_id')})"
            else:
                dst_offset = dir_to_offset.get(dst_dir, (0, 0))
                dst_row = row + dst_offset[0]
                dst_col = col + dst_offset[1]
                dst_desc = f"({dst_row}, {dst_col})"

            if src_dir == 'L':
                src_desc = f"LOCAL ({row}, {col})"
            else:
                src_desc = f"({src_row}, {src_col})"

            dep = f"({row},{col}).{src_dir} -> ({row},{col}).{dst_dir} [wants to reach {dst_desc}]"
            dependencies.append(dep)
            print(dep)

    # Analyze circular dependencies
    print("\n" + "=" * 80)
    print("[CIRCULAR DEPENDENCY ANALYSIS]")
    print("-" * 80)

    # Build a graph of blocked paths
    # Node: (row, col, port)
    # Edge: blocked path
    graph = defaultdict(list)

    for node in deadlock_nodes:
        row, col = node['row'], node['col']
        for stall in node['stalled']:
            src_dir = stall['from']
            dst_dir = stall['to']

            # Source node
            if src_dir == 'L':
                src_node = (row, col, 'L')
            else:
                src_offset = dir_to_offset[src_dir]
                src_node = (row - src_offset[0], col - src_offset[1],
                           {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}[src_dir])

            # Destination node
            if dst_dir == 'L':
                dst_node = (row, col, 'L')
            else:
                dst_offset = dir_to_offset[dst_dir]
                dst_node = (row + dst_offset[0], col + dst_offset[1],
                           {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}.get(dst_dir, dst_dir))

            # This path is blocked
            graph[src_node].append({
                'blocked_at': (row, col, dst_dir),
                'wants_to_reach': dst_node,
                'fifo_id': stall.get('fifo_id'),
                'ready': stall['ready']
            })

    # Print graph
    for src, edges in sorted(graph.items()):
        print(f"\nFrom ({src[0]},{src[1]}).{src[2]}:")
        for edge in edges:
            blocked_at = edge['blocked_at']
            wants = edge['wants_to_reach']
            fifo = f" [fifo={edge['fifo_id']}]" if edge['fifo_id'] is not None else ""
            ready = f" ready={edge['ready']}" if edge['ready'] is not None else ""
            print(f"  -> blocked at ({blocked_at[0]},{blocked_at[1]}).{blocked_at[2]}, wants ({wants[0]},{wants[1]}).{wants[2]}{fifo}{ready}")

    # Summary
    print("\n" + "=" * 80)
    print("[SUMMARY]")
    print("-" * 80)

    # Group by row
    by_row = defaultdict(list)
    for node in deadlock_nodes:
        by_row[node['row']].append(node)

    for row in sorted(by_row.keys()):
        nodes = by_row[row]
        print(f"\nRow {row}: {len(nodes)} stalled nodes")
        cols = sorted(set(n['col'] for n in nodes))
        print(f"  Columns: {cols}")

        # Check for East-West chain
        ew_chain = []
        for node in sorted(nodes, key=lambda x: x['col']):
            for stall in node['stalled']:
                if stall['to'] in ['E', 'W']:
                    ew_chain.append((node['col'], stall['from'], stall['to']))

        if ew_chain:
            print(f"  E-W chain: {ew_chain}")

        # Check for local blockages
        local_blocks = []
        for node in nodes:
            for stall in node['stalled']:
                if stall['to'] == 'L':
                    local_blocks.append((node['col'], stall['from'], stall.get('fifo_id')))

        if local_blocks:
            print(f"  Local blocks: {local_blocks}")


if __name__ == "__main__":
    analyze_deadlock()
