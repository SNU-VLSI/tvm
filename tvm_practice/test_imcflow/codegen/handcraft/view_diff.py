#!/usr/bin/env python3
"""
View diff between build/ and build.orig/ cpp files

This tool shows the differences between generated cpp files in build/ and build.orig/
directories for easy comparison during development and debugging.

Usage:
    python view_diff.py <model_name> <target> [--region <name>] [--no-color] [--context <lines>]

Arguments:
    model_name: Model evaluation directory (e.g., resnet8_subset31_pretrained_orig_evl)
    target: Target file type to compare (inode or imce)

Examples:
    python view_diff.py resnet8_subset31_pretrained_orig_evl inode
    python view_diff.py resnet8_subset31_pretrained_orig_evl imce --region region2
    python view_diff.py resnet8_subset12_pretrained_small_evl inode --region region1 --context 5
"""

import argparse
import difflib
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def find_region_dirs(build_dir: Path, region_filter: Optional[str] = None) -> List[Path]:
    """
    Find all region directories in the build directory.

    Args:
        build_dir: Path to build directory
        region_filter: Optional substring to filter regions (e.g., "region2")

    Returns:
        List of region directory paths
    """
    if not build_dir.exists():
        return []

    regions = []
    for item in sorted(build_dir.iterdir()):
        if item.is_dir() and item.name.startswith('tvmgen_'):
            if region_filter:
                # Check if region_filter is in directory name
                if region_filter.lower() in item.name.lower():
                    regions.append(item)
            else:
                regions.append(item)

    return regions


def extract_region_name(dir_name: str) -> str:
    """
    Extract human-readable region name from directory name.

    Example:
        tvmgen_default_tvmgen_default_imcflow_main_46_round_imcflow_region2_main_5
        -> region2
    """
    match = re.search(r'region(\d+)', dir_name)
    if match:
        return f"region{match.group(1)}"
    return dir_name


def extract_node_from_line(line: str) -> Optional[str]:
    """
    Extract node identifier from a line containing conditional comments.

    Example:
        "  if (hid == 1 && wid == 4) { // imce_1_4" -> "imce_1_4"
        "  else if (hid == 0 && wid == 2) { // imce_0_2" -> "imce_0_2"

    Args:
        line: Source code line

    Returns:
        Node identifier or None if not found
    """
    # Match patterns like: if (hid == X && wid == Y) { // imce_X_Y
    # Also match inode patterns like: if (iid == X) { // inode_X
    match = re.search(r'//\s*(imce_\d+_\d+|inode_\d+)', line)
    if match:
        return match.group(1)
    return None


def generate_unified_diff(original_path: Path, modified_path: Path,
                         context_lines: int = 3) -> str:
    """
    Generate unified diff between two files.

    Args:
        original_path: Path to original file (build.orig/)
        modified_path: Path to modified file (build/)
        context_lines: Number of context lines to show

    Returns:
        Unified diff string
    """
    if not original_path.exists():
        return f"Error: Original file not found: {original_path}\n"

    if not modified_path.exists():
        return f"Error: Modified file not found: {modified_path}\n"

    with open(original_path, 'r') as f:
        original_lines = f.readlines()

    with open(modified_path, 'r') as f:
        modified_lines = f.readlines()

    diff = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile=f"build.orig/{original_path.parent.name}/{original_path.name}",
        tofile=f"build/{modified_path.parent.name}/{modified_path.name}",
        n=context_lines
    )

    # Join diff lines, stripping the trailing newline from each since we'll add them back
    return ''.join(diff).rstrip('\n')


def annotate_diff_with_nodes(diff_text: str, modified_path: Path) -> str:
    """
    Annotate diff output with node identifiers (imce_X_Y or inode_X).

    Reads the modified file to map line numbers to nodes, then inserts
    node headers when the diff enters a new node's code block.

    Args:
        diff_text: Unified diff output
        modified_path: Path to the modified file

    Returns:
        Annotated diff text with node headers
    """
    if not diff_text or not modified_path.exists():
        return diff_text

    # Read modified file to build line -> node mapping
    with open(modified_path, 'r') as f:
        file_lines = f.readlines()

    # Build a mapping of line numbers to nodes
    line_to_node = {}
    current_node = None

    for line_num, line in enumerate(file_lines, start=1):
        node = extract_node_from_line(line)
        if node:
            current_node = node
        if current_node:
            line_to_node[line_num] = current_node

    # Process diff and insert node headers
    diff_lines = diff_text.split('\n')
    annotated_lines = []
    last_printed_node = None

    for line in diff_lines:
        # Parse hunk header to get line numbers
        if line.startswith('@@'):
            # Extract the line number from new file side (e.g., @@ -100,7 +100,8 @@)
            match = re.search(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
            if match:
                current_line = int(match.group(1))
            annotated_lines.append(line)
            continue

        # For context and added lines, check if we need to print node header
        if line.startswith(' ') or line.startswith('+'):
            # Context or added line - check node at current line
            if 'current_line' in locals() and current_line in line_to_node:
                node = line_to_node[current_line]
                if node != last_printed_node:
                    # Insert node header (add empty line before only if not first node)
                    if last_printed_node is not None:
                        annotated_lines.append('')
                    annotated_lines.append(f"--- Diff for {node} ---")
                    last_printed_node = node

            if 'current_line' in locals():
                current_line += 1

        annotated_lines.append(line)

    return '\n'.join(annotated_lines)


def colorize_diff(diff_text: str) -> str:
    """Add ANSI colors to diff output"""
    if not diff_text:
        return diff_text

    lines = diff_text.split('\n')
    colored_lines = []

    for line in lines:
        if line.startswith('--- Diff for '):
            # Node header - use bold magenta
            colored_lines.append(f"\033[1;35m{line}\033[0m")  # Bold Magenta
        elif line.startswith('+++') or line.startswith('---'):
            colored_lines.append(f"\033[1;33m{line}\033[0m")  # Bold Yellow
        elif line.startswith('+'):
            colored_lines.append(f"\033[32m{line}\033[0m")  # Green
        elif line.startswith('-'):
            colored_lines.append(f"\033[31m{line}\033[0m")  # Red
        elif line.startswith('@@'):
            colored_lines.append(f"\033[36m{line}\033[0m")  # Cyan
        else:
            colored_lines.append(line)

    return '\n'.join(colored_lines)


def get_diff_stats(diff_text: str) -> Tuple[int, int]:
    """
    Calculate diff statistics.

    Returns:
        Tuple of (additions, deletions)
    """
    if not diff_text:
        return (0, 0)

    additions = 0
    deletions = 0

    for line in diff_text.split('\n'):
        if line.startswith('+') and not line.startswith('+++'):
            additions += 1
        elif line.startswith('-') and not line.startswith('---'):
            deletions += 1

    return (additions, deletions)


def main():
    parser = argparse.ArgumentParser(
        description='View diff between build/ and build.orig/ cpp files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python view_diff.py resnet8_subset31_pretrained_orig_evl inode
  python view_diff.py resnet8_subset31_pretrained_orig_evl imce --region region2
  python view_diff.py big_conv_evl inode --region region1 --context 5
        """
    )

    parser.add_argument('model_name',
                        help='Model evaluation directory name')
    parser.add_argument('target',
                        choices=['inode', 'imce'],
                        help='Target file type to compare (inode.cpp or imce.cpp)')
    parser.add_argument('--region',
                        type=str,
                        help='Filter to specific region (e.g., "region2")')
    parser.add_argument('--no-color',
                        action='store_true',
                        help='Disable colored diff output')
    parser.add_argument('--context', '-c',
                        type=int,
                        default=3,
                        help='Number of context lines to show (default: 3)')
    parser.add_argument('--stats', '-s',
                        action='store_true',
                        help='Show diff statistics (additions/deletions)')

    args = parser.parse_args()

    # Construct paths
    model_dir = Path(args.model_name)
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}", file=sys.stderr)
        return 1

    build_dir = model_dir / 'build'
    build_orig_dir = model_dir / 'build.orig'

    if not build_dir.exists():
        print(f"Error: build/ directory not found: {build_dir}", file=sys.stderr)
        return 1

    if not build_orig_dir.exists():
        print(f"Error: build.orig/ directory not found: {build_orig_dir}", file=sys.stderr)
        return 1

    # Find regions to compare
    regions = find_region_dirs(build_dir, args.region)

    if not regions:
        if args.region:
            print(f"Error: No regions found matching filter: {args.region}", file=sys.stderr)
        else:
            print(f"Error: No region directories found in {build_dir}", file=sys.stderr)
        return 1

    # Target filename
    target_file = f"{args.target}.cpp"

    print(f"Model: {args.model_name}")
    print(f"Target: {target_file}")
    print(f"Regions to compare: {len(regions)}")
    print(f"Context lines: {args.context}")
    print("=" * 80)
    print()

    # Compare each region
    total_additions = 0
    total_deletions = 0
    regions_with_changes = 0

    for region_dir in regions:
        region_name = extract_region_name(region_dir.name)

        # Paths to compare
        modified_file = region_dir / target_file
        original_file = build_orig_dir / region_dir.name / target_file

        if not modified_file.exists():
            print(f"Warning: {modified_file} not found, skipping", file=sys.stderr)
            continue

        if not original_file.exists():
            print(f"Warning: {original_file} not found, skipping", file=sys.stderr)
            continue

        # Generate diff
        diff_text = generate_unified_diff(original_file, modified_file, args.context)

        # Skip if no changes
        if not diff_text or diff_text.startswith("Error:"):
            if diff_text.startswith("Error:"):
                print(diff_text, file=sys.stderr)
            continue

        # Check if there are actual changes (not just header lines)
        diff_lines = diff_text.split('\n')
        has_changes = any(line.startswith(('+', '-')) and
                         not line.startswith(('+++', '---'))
                         for line in diff_lines)

        if not has_changes:
            continue

        regions_with_changes += 1

        # Calculate stats
        additions, deletions = get_diff_stats(diff_text)
        total_additions += additions
        total_deletions += deletions

        # Print region header
        print(f"{'=' * 80}")
        print(f"Region: {region_name} ({region_dir.name})")
        if args.stats:
            print(f"Stats: +{additions} -{deletions}")
        print(f"{'=' * 80}")

        # Annotate diff with node information
        annotated_diff = annotate_diff_with_nodes(diff_text, modified_file)

        # Print diff
        if args.no_color:
            print(annotated_diff)
        else:
            print(colorize_diff(annotated_diff))

        print()

    # Summary
    print("=" * 80)
    print(f"Summary: {regions_with_changes} region(s) with changes")
    if args.stats:
        print(f"Total: +{total_additions} -{total_deletions}")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
