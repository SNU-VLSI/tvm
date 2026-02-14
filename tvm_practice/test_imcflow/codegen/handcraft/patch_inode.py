#!/usr/bin/env python3
"""
Patch inode.cpp based on mem_layout.txt

This tool patches inode.cpp files to update WR_IMEM variables (var2, loop counts)
based on the memory layout generated from imce.cpp builds.

Usage:
    python patch_inode.py <eval_dir> [--region <name>] [--dry-run] [--verbose]

Examples:
    python patch_inode.py ../resnet8_subset31_pretrained_orig_evl --dry-run
    python patch_inode.py ../resnet8_subset31_pretrained_orig_evl --region region2
"""

import argparse
import difflib
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ImemBlock:
    """Represents an IMCE imem block from mem_layout.txt"""
    rel: int
    size: int

    @property
    def loop_count(self) -> int:
        return self.size // 32


def parse_mem_layout(layout_path: str) -> Dict[str, Dict[str, ImemBlock]]:
    """
    Parse mem_layout.txt and extract imce_X_Y_imem blocks per region.

    Returns:
        Dict[region_name, Dict[imce_name, ImemBlock]]
        Example: {'tvmgen_..._region2_main_5': {'imce_3_1': ImemBlock(rel=10368, size=704)}}
    """
    with open(layout_path, 'r') as f:
        content = f.read()

    result = {}
    current_region = None

    # Pattern to match region name
    region_pattern = r"'(tvmgen_default_[^']+)':\s*FuncMemoryLayout"

    # Pattern to match imce imem DataBlock
    # DataBlock(imce_3_1_imem, size=704, rel=10368, addr=211200)
    imem_block_pattern = r"DataBlock\(imce_(\d+)_(\d+)_imem,\s*size=(\d+),\s*rel=(\d+)"

    lines = content.split('\n')
    for line in lines:
        # Check for region start
        region_match = re.search(region_pattern, line)
        if region_match:
            current_region = region_match.group(1)
            if current_region not in result:
                result[current_region] = {}

        # Check for imem block
        if current_region:
            imem_match = re.search(imem_block_pattern, line)
            if imem_match:
                row = int(imem_match.group(1))
                col = int(imem_match.group(2))
                size = int(imem_match.group(3))
                rel = int(imem_match.group(4))

                imce_name = f"imce_{row}_{col}"
                result[current_region][imce_name] = ImemBlock(rel=rel, size=size)

    return result


def filter_regions(layout: Dict[str, Dict[str, ImemBlock]],
                   region_filter: Optional[str]) -> Dict[str, Dict[str, ImemBlock]]:
    """Filter regions by name substring match"""
    if not region_filter:
        return layout
    return {k: v for k, v in layout.items() if region_filter in k}


def find_imem_section(lines: List[str], imce_name: str) -> Optional[Tuple[int, int]]:
    """
    Find the start and end line indices for an imem write section.

    Returns:
        Tuple of (start_idx, end_idx) or None if not found
    """
    start_marker = f"// generate: imem write: {imce_name}"
    end_marker = f"// endgenerate: imem write: {imce_name}"

    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if start_marker in line:
            start_idx = i
        elif end_marker in line and start_idx is not None:
            end_idx = i
            break

    if start_idx is not None and end_idx is not None:
        return (start_idx, end_idx)
    return None


def patch_imem_section(lines: List[str], start: int, end: int,
                       imce_name: str, block: ImemBlock, verbose: bool = False) -> List[str]:
    """
    Patch an imem write section with new rel and loop_count values.

    Returns:
        New list of lines for this section
    """
    section = lines[start:end + 1]
    new_section = []
    loop_count = block.loop_count

    # Extract the WR_IMEM line to get the target_id (last argument)
    target_id = None
    for line in section:
        wr_imem_match = re.search(r'__builtin_INODE_WR_IMEM\([^,]+,\s*\d+,\s*(\d+)\)', line)
        if wr_imem_match:
            target_id = wr_imem_match.group(1)
            break

    if target_id is None:
        # Can't find target_id, return original section
        return section

    # Get indentation from first line after start marker
    indent = ""
    for line in section[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = line[:len(line) - len(stripped)]
            break

    # Build new section
    new_section.append(section[0])  # // generate: imem write: imce_X_Y

    # var2 = XXX line
    new_section.append(f"{indent}var2 = {block.rel};")

    # __builtin_INODE_SET_ADDR_CNT(0)
    new_section.append(f"{indent}__builtin_INODE_SET_ADDR_CNT(0);")

    if loop_count == 1:
        # Single call pattern
        new_section.append(f"{indent}// generate. loop count == 1")
        new_section.append(f"{indent}__builtin_INODE_WR_IMEM(var2 + 0*32, 0, {target_id});")
        new_section.append(f"{indent}// endgenerate")
    else:
        # For loop pattern
        new_section.append(f"{indent}for (int i1 = 0; i1 < {loop_count}; i1++) {{ // generate")
        new_section.append(f"{indent}  __builtin_INODE_WR_IMEM(var2 + i1*32, 0, {target_id});")
        new_section.append(f"{indent}}} // endgenerate")

    new_section.append(section[-1])  # // endgenerate: imem write: imce_X_Y

    if verbose:
        print(f"  Patched {imce_name}: rel={block.rel}, loop_count={loop_count}")

    return new_section


def patch_inode_cpp(cpp_path: str, imem_blocks: Dict[str, ImemBlock],
                    verbose: bool = False) -> str:
    """
    Patch an inode.cpp file with new imem block values.

    Returns:
        Patched file content as string
    """
    with open(cpp_path, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    result_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if this line starts an imem write section
        matched_imce = None
        for imce_name in imem_blocks:
            if f"// generate: imem write: {imce_name}" in line:
                matched_imce = imce_name
                break

        if matched_imce:
            # Find the section bounds
            section_bounds = find_imem_section(lines, matched_imce)
            if section_bounds:
                start, end = section_bounds
                # Patch the section
                new_section = patch_imem_section(
                    lines, start, end, matched_imce,
                    imem_blocks[matched_imce], verbose
                )
                result_lines.extend(new_section)
                i = end + 1
                continue

        result_lines.append(line)
        i += 1

    return '\n'.join(result_lines)


def generate_diff(original: str, patched: str, file_path: str) -> str:
    """Generate unified diff in git diff style"""
    original_lines = original.splitlines(keepends=True)
    patched_lines = patched.splitlines(keepends=True)

    # Ensure lines end with newline for proper diff
    if original_lines and not original_lines[-1].endswith('\n'):
        original_lines[-1] += '\n'
    if patched_lines and not patched_lines[-1].endswith('\n'):
        patched_lines[-1] += '\n'

    diff = difflib.unified_diff(
        original_lines,
        patched_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm='\n'
    )
    return ''.join(diff)


def colorize_diff(diff_text: str) -> str:
    """Add ANSI colors to diff output"""
    lines = diff_text.split('\n')
    colored_lines = []

    for line in lines:
        if line.startswith('+++') or line.startswith('---'):
            colored_lines.append(f"\033[1m{line}\033[0m")  # Bold
        elif line.startswith('+'):
            colored_lines.append(f"\033[32m{line}\033[0m")  # Green
        elif line.startswith('-'):
            colored_lines.append(f"\033[31m{line}\033[0m")  # Red
        elif line.startswith('@@'):
            colored_lines.append(f"\033[36m{line}\033[0m")  # Cyan
        else:
            colored_lines.append(line)

    return '\n'.join(colored_lines)


def patch_inode_for_eval_dir(eval_dir: str, input_build_dir: str, output_build_dir: str, verbose: bool = False) -> int:
    """
    Patch inode.cpp files based on mem_layout.txt.

    Reads inode.cpp from input_build_dir and writes inode.patched.cpp to output_build_dir.

    Args:
        eval_dir: Path to eval directory containing mem_layout.txt
        input_build_dir: Directory containing source inode.cpp files
        output_build_dir: Directory to write inode.patched.cpp files
        verbose: If True, print detailed output

    Returns:
        0 on success, 1 on error
    """
    layout_path = os.path.join(eval_dir, 'mem_layout.txt')

    if not os.path.exists(layout_path):
        print(f"Error: mem_layout.txt not found at {layout_path}")
        return 1

    layout = parse_mem_layout(layout_path)
    if not layout:
        print(f"No regions found in mem_layout.txt")
        return 1

    print(f"Patching inode.cpp -> inode.patched.cpp ({len(layout)} region(s))...")

    patched_count = 0
    for region_name, imem_blocks in sorted(layout.items()):
        cpp_path = os.path.join(input_build_dir, region_name, 'inode.cpp')
        patched_cpp_path = os.path.join(output_build_dir, region_name, 'inode.patched.cpp')

        if not os.path.exists(cpp_path):
            if verbose:
                print(f"  Warning: {cpp_path} not found, skipping")
            continue

        patched = patch_inode_cpp(cpp_path, imem_blocks, verbose)

        with open(patched_cpp_path, 'w') as f:
            f.write(patched)

        with open(cpp_path, 'r') as f:
            if f.read() != patched:
                print(f"  Patched: {region_name}/inode.patched.cpp")
                patched_count += 1

    if patched_count > 0:
        print(f"Patched {patched_count} file(s)")
    else:
        print(f"No changes needed (copied as inode.patched.cpp)")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Patch inode.cpp files based on mem_layout.txt, output to inode.patched.cpp'
    )
    parser.add_argument('eval_dir', help='Evaluation directory (e.g., resnet8_subset31_pretrained_orig_evl)')
    parser.add_argument('--region', type=str, help='Filter to specific region (e.g., "region2")')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show diff without modifying files')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable colored diff output')

    args = parser.parse_args()

    # mem_layout.txt is in ../eval_dir/ (parent codegen directory)
    # inode.cpp files are in ./eval_dir/build/ (current handcraft directory)
    layout_path = os.path.join('..', args.eval_dir, 'mem_layout.txt')
    build_dir = os.path.join(args.eval_dir, 'build')

    if not os.path.exists(layout_path):
        print(f"Error: mem_layout.txt not found at {layout_path}")
        return 1

    if not os.path.exists(build_dir):
        print(f"Error: build directory not found at {build_dir}")
        return 1

    layout = parse_mem_layout(layout_path)

    if args.verbose:
        print(f"Parsed {len(layout)} regions from mem_layout.txt")

    # Filter regions if specified
    layout = filter_regions(layout, args.region)

    if not layout:
        print(f"No regions found matching filter: {args.region}")
        return 1

    print(f"Processing {len(layout)} region(s)")

    # Process each region
    for region_name, imem_blocks in sorted(layout.items()):
        cpp_path = os.path.join(build_dir, region_name, 'inode.cpp')
        patched_cpp_path = os.path.join(build_dir, region_name, 'inode.patched.cpp')

        if not os.path.exists(cpp_path):
            print(f"Warning: {cpp_path} not found, skipping")
            continue

        if args.verbose:
            print(f"\nRegion: {region_name}")
            print(f"  imem blocks: {list(imem_blocks.keys())}")

        # Read original content
        with open(cpp_path, 'r') as f:
            original = f.read()

        # Patch the file
        patched = patch_inode_cpp(cpp_path, imem_blocks, args.verbose)

        if original == patched:
            print(f"  {region_name}: No changes needed")
            if not args.dry_run:
                # Still write to patched file for consistency
                with open(patched_cpp_path, 'w') as f:
                    f.write(patched)
            continue

        if args.dry_run:
            # Show diff
            rel_path = os.path.relpath(cpp_path, args.eval_dir)
            diff = generate_diff(original, patched, rel_path)
            if diff:
                print(f"\n=== {region_name} ===")
                if args.no_color:
                    print(diff)
                else:
                    print(colorize_diff(diff))
        else:
            # Write patched file to inode.patched.cpp (preserves original)
            with open(patched_cpp_path, 'w') as f:
                f.write(patched)
            print(f"  Patched: {patched_cpp_path}")

    return 0


if __name__ == '__main__':
    exit(main())
