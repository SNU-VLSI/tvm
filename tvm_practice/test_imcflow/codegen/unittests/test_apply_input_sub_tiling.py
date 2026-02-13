"""
Unit tests for apply_input_sub_tiling function.

This script tests the apply_input_sub_tiling function in isolation
with various input configurations.
"""
import math
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


# Mock classes to simulate the real objects
@dataclass
class MockMemBlock:
    """Mock memory block with size attribute"""
    size: int


@dataclass
class MockEdge:
    """Mock tensor edge"""
    name: str


def debug_print(msg):
    """Debug print function"""
    print(f"  [DEBUG] {msg}")


def apply_input_sub_tiling(
    output_tile_specs: Dict[int, Tuple[List[int], List[int]]],
    trimmed_input_tiles: Dict[str, Tuple[List[int], List[int]]],
    input_tensor_info: List[Tuple],
    output_tensor_info: List[Tuple],
    memory_limit: int,
    func_name: str = "test_func"
) -> Tuple[Dict[int, Tuple[List[int], List[int]]], Dict[str, Tuple[List[int], List[int]]]]:
    """
    Apply hierarchical input sub-tiling when input tiles exceed memory limit.

    This is a standalone version of the function for testing.

    Args:
        output_tile_specs: {out_idx: (bases, sizes)}
        trimmed_input_tiles: {var_name: (trimmed_bases, trimmed_sizes)}
        input_tensor_info: List of (edge, height, width, channels, elem_size, inode_name, mem_block, var_name)
        output_tensor_info: List of (edge, height, width, channels, elem_size, inode_name, mem_block)
        memory_limit: Maximum memory per tile in bytes
        func_name: Function name for debug output

    Returns:
        (new_output_tile_specs, new_trimmed_input_tiles)
    """
    debug_print("Applying input sub-tiling due to memory limit")

    # Get memory per row for each input variable
    input_mem_per_row = {}
    for edge, height, width, channels, elem_size, inode_name, mem_block, var_name in input_tensor_info:
        if height > 0:
            input_mem_per_row[var_name] = mem_block.size / height
        else:
            input_mem_per_row[var_name] = mem_block.size

    # Get memory per row for each output
    output_mem_per_row = {}
    for out_idx, (edge, height, width, channels, elem_size, inode_name, mem_block) in enumerate(output_tensor_info):
        if height > 0:
            output_mem_per_row[out_idx] = mem_block.size / height
        else:
            output_mem_per_row[out_idx] = mem_block.size

    num_tiles = len(list(output_tile_specs.values())[0][0])

    # For each tile, find the input with largest memory requirement
    # and determine how many sub-tiles are needed
    tile_sub_tile_info = []  # [(tile_idx, num_sub_tiles, max_var, sub_tile_size, total_input_size)]

    for tile_idx in range(num_tiles):
        max_input_mem = 0
        max_var = None
        max_input_size = 0

        for var_name, (trimmed_bases, trimmed_sizes) in trimmed_input_tiles.items():
            if tile_idx < len(trimmed_sizes):
                tile_input_size = trimmed_sizes[tile_idx]
                mem = tile_input_size * input_mem_per_row.get(var_name, 0)
                if mem > max_input_mem:
                    max_input_mem = mem
                    max_var = var_name
                    max_input_size = tile_input_size

        # Calculate how many sub-tiles needed
        if max_input_mem > memory_limit and max_input_size > 0:
            # Estimate max rows that fit
            max_rows_per_subtile = max(1, int(memory_limit / input_mem_per_row.get(max_var, 1)))
            num_sub_tiles = math.ceil(max_input_size / max_rows_per_subtile)
            sub_tile_size = math.ceil(max_input_size / num_sub_tiles)
            tile_sub_tile_info.append((tile_idx, num_sub_tiles, max_var, sub_tile_size, max_input_size))
            debug_print(f"  Tile {tile_idx}: input {max_var} needs {num_sub_tiles} sub-tiles of ~{sub_tile_size} rows (total {max_input_size})")
        else:
            tile_sub_tile_info.append((tile_idx, 1, None, 0, 0))

    # Build new expanded tile specs
    new_output_tile_specs = {out_idx: ([], []) for out_idx in output_tile_specs.keys()}
    new_trimmed_input_tiles = {var_name: ([], []) for var_name in trimmed_input_tiles.keys()}

    for tile_idx, num_sub_tiles, max_var, sub_tile_size, total_input_size in tile_sub_tile_info:
        if num_sub_tiles == 1:
            # No sub-tiling needed, copy as-is
            for out_idx, (out_bases, out_sizes) in output_tile_specs.items():
                new_output_tile_specs[out_idx][0].append(out_bases[tile_idx])
                new_output_tile_specs[out_idx][1].append(out_sizes[tile_idx])

            for var_name, (in_bases, in_sizes) in trimmed_input_tiles.items():
                new_trimmed_input_tiles[var_name][0].append(in_bases[tile_idx])
                new_trimmed_input_tiles[var_name][1].append(in_sizes[tile_idx])
        else:
            # Sub-tiling needed
            # Get original output info for this tile
            out_bases_orig = {out_idx: output_tile_specs[out_idx][0][tile_idx] for out_idx in output_tile_specs}
            out_sizes_orig = {out_idx: output_tile_specs[out_idx][1][tile_idx] for out_idx in output_tile_specs}

            # Get original trimmed input info for this tile
            in_bases_orig = {var_name: trimmed_input_tiles[var_name][0][tile_idx] for var_name in trimmed_input_tiles}
            in_sizes_orig = {var_name: trimmed_input_tiles[var_name][1][tile_idx] for var_name in trimmed_input_tiles}

            # Get base for the max_var
            max_var_base = in_bases_orig[max_var]

            for sub_idx in range(num_sub_tiles):
                is_last_subtile = (sub_idx == num_sub_tiles - 1)

                # Output: size=0 for intermediate, actual size for last
                for out_idx in output_tile_specs.keys():
                    new_output_tile_specs[out_idx][0].append(out_bases_orig[out_idx])
                    if is_last_subtile:
                        new_output_tile_specs[out_idx][1].append(out_sizes_orig[out_idx])
                    else:
                        new_output_tile_specs[out_idx][1].append(0)

                # Input: divide into sub-tiles
                for var_name in trimmed_input_tiles.keys():
                    if var_name == max_var:
                        # This is the variable being sub-tiled
                        sub_base = max_var_base + sub_idx * sub_tile_size
                        sub_size = min(sub_tile_size, max_var_base + total_input_size - sub_base)
                        new_trimmed_input_tiles[var_name][0].append(sub_base)
                        new_trimmed_input_tiles[var_name][1].append(max(0, sub_size))
                    else:
                        # Other variables: proportional split
                        orig_base = in_bases_orig[var_name]
                        orig_size = in_sizes_orig[var_name]
                        if total_input_size > 0 and orig_size > 0:
                            ratio = sub_tile_size / total_input_size
                            other_sub_size = int(orig_size * ratio)
                            other_sub_base = orig_base + sub_idx * other_sub_size
                            new_trimmed_input_tiles[var_name][0].append(other_sub_base)
                            new_trimmed_input_tiles[var_name][1].append(other_sub_size)
                        else:
                            # Just repeat with size=0 for intermediate, orig for last
                            new_trimmed_input_tiles[var_name][0].append(orig_base)
                            if is_last_subtile:
                                new_trimmed_input_tiles[var_name][1].append(orig_size)
                            else:
                                new_trimmed_input_tiles[var_name][1].append(0)

    new_num_iterations = len(new_output_tile_specs[0][0]) if new_output_tile_specs else 0
    debug_print(f"[{func_name}] Input sub-tiling: {num_tiles} tiles -> {new_num_iterations} iterations")

    return new_output_tile_specs, new_trimmed_input_tiles


def print_tile_specs(name: str, output_specs: Dict, input_specs: Dict):
    """Pretty print tile specifications"""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print("  Output tile specs:")
    for out_idx, (bases, sizes) in output_specs.items():
        print(f"    out[{out_idx}]: bases={bases}, sizes={sizes}")
    print("  Input tile specs:")
    for var_name, (bases, sizes) in input_specs.items():
        print(f"    {var_name}: bases={bases}, sizes={sizes}")


def test_case_1_no_subtiling_needed():
    """
    Test Case 1: No sub-tiling needed (all tiles fit in memory)

    Setup:
    - 3 output tiles, each with 1 row
    - 1 input variable with 3 rows per output row (kernel=3)
    - Memory limit: 1000 bytes
    - Input mem per row: 100 bytes -> 3 rows = 300 bytes < 1000 (fits!)
    """
    print("\n" + "="*70)
    print("TEST CASE 1: No sub-tiling needed (tiles fit in memory)")
    print("="*70)

    # Output: 3 tiles, each processing 1 row
    output_tile_specs = {
        0: ([0, 1, 2], [1, 1, 1])  # bases=[0,1,2], sizes=[1,1,1]
    }

    # Input: for 3x3 conv, need 3 input rows for first tile, then 1 for subsequent
    # After trimming (halo removal): [3, 1, 1] -> actual new rows needed
    trimmed_input_tiles = {
        "input0": ([0, 3, 4], [3, 1, 1])  # First tile needs rows 0-2, then 3, then 4
    }

    # Input tensor info: height=5, 100 bytes per row
    input_tensor_info = [
        (MockEdge("input"), 5, 10, 10, 1, "inode0", MockMemBlock(500), "input0")
        # 500 bytes / 5 rows = 100 bytes per row
    ]

    # Output tensor info: height=3
    output_tensor_info = [
        (MockEdge("output"), 3, 10, 10, 1, "onode0", MockMemBlock(300))
    ]

    memory_limit = 1000  # 300 bytes < 1000, no sub-tiling needed

    print_tile_specs("BEFORE", output_tile_specs, trimmed_input_tiles)

    new_out, new_in = apply_input_sub_tiling(
        output_tile_specs, trimmed_input_tiles,
        input_tensor_info, output_tensor_info,
        memory_limit
    )

    print_tile_specs("AFTER", new_out, new_in)

    # Verify: should be unchanged
    assert new_out[0][0] == [0, 1, 2], f"Output bases should be unchanged, got {new_out[0][0]}"
    assert new_out[0][1] == [1, 1, 1], f"Output sizes should be unchanged, got {new_out[0][1]}"
    assert new_in["input0"][0] == [0, 3, 4], f"Input bases should be unchanged, got {new_in['input0'][0]}"
    assert new_in["input0"][1] == [3, 1, 1], f"Input sizes should be unchanged, got {new_in['input0'][1]}"

    print("\n✅ Test Case 1 PASSED: No sub-tiling applied as expected")


def test_case_2_first_tile_needs_subtiling():
    """
    Test Case 2: First tile needs sub-tiling

    Setup:
    - 3 output tiles, each with 1 row
    - First input tile needs 8 rows (too big)
    - Memory limit allows only 2 rows at a time
    - Expected: first tile split into 4 sub-tiles
    """
    print("\n" + "="*70)
    print("TEST CASE 2: First tile needs sub-tiling")
    print("="*70)

    output_tile_specs = {
        0: ([0, 1, 2], [1, 1, 1])
    }

    # First tile needs 8 input rows, too big!
    trimmed_input_tiles = {
        "input0": ([0, 8, 9], [8, 1, 1])
    }

    # 1000 bytes / 10 rows = 100 bytes per row
    input_tensor_info = [
        (MockEdge("input"), 10, 10, 10, 1, "inode0", MockMemBlock(1000), "input0")
    ]

    output_tensor_info = [
        (MockEdge("output"), 3, 10, 10, 1, "onode0", MockMemBlock(300))
    ]

    # Memory limit: 200 bytes -> 2 rows max per sub-tile
    # 8 rows / 2 = 4 sub-tiles needed
    memory_limit = 200

    print_tile_specs("BEFORE", output_tile_specs, trimmed_input_tiles)
    print(f"\n  Memory limit: {memory_limit} bytes")
    print(f"  Input mem per row: 100 bytes")
    print(f"  First tile needs 8 rows = 800 bytes > {memory_limit} -> needs sub-tiling")
    print(f"  Expected: 4 sub-tiles of 2 rows each")

    new_out, new_in = apply_input_sub_tiling(
        output_tile_specs, trimmed_input_tiles,
        input_tensor_info, output_tensor_info,
        memory_limit
    )

    print_tile_specs("AFTER", new_out, new_in)

    # Verify structure
    # Original: 3 tiles -> 4 (first tile) + 1 + 1 = 6 iterations
    assert len(new_out[0][0]) == 6, f"Expected 6 iterations, got {len(new_out[0][0])}"

    # Output sizes: first 3 should be 0 (intermediate), 4th should be 1 (last of first tile)
    # Then 1, 1 for remaining tiles
    expected_out_sizes = [0, 0, 0, 1, 1, 1]
    assert new_out[0][1] == expected_out_sizes, f"Expected output sizes {expected_out_sizes}, got {new_out[0][1]}"

    # Output bases: first 4 should all be 0 (same output tile), then 1, 2
    expected_out_bases = [0, 0, 0, 0, 1, 2]
    assert new_out[0][0] == expected_out_bases, f"Expected output bases {expected_out_bases}, got {new_out[0][0]}"

    # Input: first tile (8 rows) split into 4 sub-tiles of 2 rows
    # bases: 0, 2, 4, 6, then 8, 9
    expected_in_bases = [0, 2, 4, 6, 8, 9]
    assert new_in["input0"][0] == expected_in_bases, f"Expected input bases {expected_in_bases}, got {new_in['input0'][0]}"

    # sizes: 2, 2, 2, 2, 1, 1
    expected_in_sizes = [2, 2, 2, 2, 1, 1]
    assert new_in["input0"][1] == expected_in_sizes, f"Expected input sizes {expected_in_sizes}, got {new_in['input0'][1]}"

    print("\n✅ Test Case 2 PASSED: First tile correctly sub-tiled into 4 parts")


def test_case_3_multiple_tiles_need_subtiling():
    """
    Test Case 3: Multiple tiles need sub-tiling

    Setup:
    - 3 output tiles
    - All input tiles need sub-tiling (different amounts)
    """
    print("\n" + "="*70)
    print("TEST CASE 3: Multiple tiles need sub-tiling")
    print("="*70)

    output_tile_specs = {
        0: ([0, 1, 2], [1, 1, 1])
    }

    # All tiles need sub-tiling: 6, 4, 4 rows
    trimmed_input_tiles = {
        "input0": ([0, 6, 10], [6, 4, 4])
    }

    # 1400 bytes / 14 rows = 100 bytes per row
    input_tensor_info = [
        (MockEdge("input"), 14, 10, 10, 1, "inode0", MockMemBlock(1400), "input0")
    ]

    output_tensor_info = [
        (MockEdge("output"), 3, 10, 10, 1, "onode0", MockMemBlock(300))
    ]

    # Memory limit: 200 bytes -> 2 rows max
    # Tile 0: 6 rows -> 3 sub-tiles
    # Tile 1: 4 rows -> 2 sub-tiles
    # Tile 2: 4 rows -> 2 sub-tiles
    # Total: 3 + 2 + 2 = 7 iterations
    memory_limit = 200

    print_tile_specs("BEFORE", output_tile_specs, trimmed_input_tiles)
    print(f"\n  Memory limit: {memory_limit} bytes")
    print(f"  Expected sub-tiles: tile0=3, tile1=2, tile2=2 -> 7 total iterations")

    new_out, new_in = apply_input_sub_tiling(
        output_tile_specs, trimmed_input_tiles,
        input_tensor_info, output_tensor_info,
        memory_limit
    )

    print_tile_specs("AFTER", new_out, new_in)

    # Verify total iterations
    assert len(new_out[0][0]) == 7, f"Expected 7 iterations, got {len(new_out[0][0])}"

    # Output sizes: [0,0,1, 0,1, 0,1] - last of each group gets size=1
    expected_out_sizes = [0, 0, 1, 0, 1, 0, 1]
    assert new_out[0][1] == expected_out_sizes, f"Expected {expected_out_sizes}, got {new_out[0][1]}"

    # Output bases: [0,0,0, 1,1, 2,2]
    expected_out_bases = [0, 0, 0, 1, 1, 2, 2]
    assert new_out[0][0] == expected_out_bases, f"Expected {expected_out_bases}, got {new_out[0][0]}"

    print("\n✅ Test Case 3 PASSED: Multiple tiles correctly sub-tiled")


def test_case_4_multiple_inputs():
    """
    Test Case 4: Multiple input variables (like ResNet skip connection)

    Setup:
    - 2 input variables: main path and residual
    - Main path needs sub-tiling, residual doesn't (smaller)
    """
    print("\n" + "="*70)
    print("TEST CASE 4: Multiple input variables")
    print("="*70)

    output_tile_specs = {
        0: ([0, 1, 2], [1, 1, 1])
    }

    # Main path needs 8 rows (big), residual needs 2 rows (small)
    trimmed_input_tiles = {
        "main_input": ([0, 8, 9], [8, 1, 1]),
        "residual": ([0, 2, 3], [2, 1, 1])
    }

    # Main: 100 bytes/row, Residual: 50 bytes/row
    input_tensor_info = [
        (MockEdge("main"), 10, 10, 10, 1, "inode0", MockMemBlock(1000), "main_input"),
        (MockEdge("res"), 4, 10, 5, 1, "inode1", MockMemBlock(200), "residual")
    ]

    output_tensor_info = [
        (MockEdge("output"), 3, 10, 10, 1, "onode0", MockMemBlock(300))
    ]

    # Memory limit: 200 bytes
    # Main: 8 rows * 100 = 800 > 200 -> needs 4 sub-tiles
    # Residual: 2 rows * 50 = 100 < 200 -> fits
    memory_limit = 200

    print_tile_specs("BEFORE", output_tile_specs, trimmed_input_tiles)
    print(f"\n  Memory limit: {memory_limit} bytes")
    print(f"  Main input: 100 bytes/row, first tile 8 rows = 800 > 200 -> 4 sub-tiles")
    print(f"  Residual: 50 bytes/row, first tile 2 rows = 100 < 200 -> fits")

    new_out, new_in = apply_input_sub_tiling(
        output_tile_specs, trimmed_input_tiles,
        input_tensor_info, output_tensor_info,
        memory_limit
    )

    print_tile_specs("AFTER", new_out, new_in)

    # Verify: 4 + 1 + 1 = 6 iterations
    assert len(new_out[0][0]) == 6, f"Expected 6 iterations, got {len(new_out[0][0])}"

    # Main input should be split
    assert len(new_in["main_input"][0]) == 6

    # Residual should also have 6 entries (proportionally split for first tile)
    assert len(new_in["residual"][0]) == 6

    print("\n✅ Test Case 4 PASSED: Multiple inputs handled correctly")


def test_case_5_multiple_outputs():
    """
    Test Case 5: Multiple output tensors

    Setup:
    - 2 output variables (like multi-output layer)
    - Sub-tiling should apply to both
    """
    print("\n" + "="*70)
    print("TEST CASE 5: Multiple output tensors")
    print("="*70)

    output_tile_specs = {
        0: ([0, 1, 2], [1, 1, 1]),
        1: ([0, 1, 2], [1, 1, 1])
    }

    trimmed_input_tiles = {
        "input0": ([0, 6, 7], [6, 1, 1])
    }

    input_tensor_info = [
        (MockEdge("input"), 8, 10, 10, 1, "inode0", MockMemBlock(800), "input0")
    ]

    output_tensor_info = [
        (MockEdge("output0"), 3, 10, 10, 1, "onode0", MockMemBlock(300)),
        (MockEdge("output1"), 3, 10, 10, 1, "onode1", MockMemBlock(300))
    ]

    memory_limit = 200  # 6 rows * 100 = 600 > 200 -> 3 sub-tiles

    print_tile_specs("BEFORE", output_tile_specs, trimmed_input_tiles)

    new_out, new_in = apply_input_sub_tiling(
        output_tile_specs, trimmed_input_tiles,
        input_tensor_info, output_tensor_info,
        memory_limit
    )

    print_tile_specs("AFTER", new_out, new_in)

    # Both outputs should have same structure
    assert new_out[0][0] == new_out[1][0], "Both outputs should have same bases"
    assert new_out[0][1] == new_out[1][1], "Both outputs should have same sizes"

    # 3 + 1 + 1 = 5 iterations
    assert len(new_out[0][0]) == 5, f"Expected 5 iterations, got {len(new_out[0][0])}"

    print("\n✅ Test Case 5 PASSED: Multiple outputs handled correctly")


def test_case_6_edge_case_single_tile():
    """
    Test Case 6: Single tile that needs sub-tiling
    """
    print("\n" + "="*70)
    print("TEST CASE 6: Single tile needing sub-tiling")
    print("="*70)

    output_tile_specs = {
        0: ([0], [1])
    }

    trimmed_input_tiles = {
        "input0": ([0], [10])  # Single tile needs 10 rows
    }

    input_tensor_info = [
        (MockEdge("input"), 10, 10, 10, 1, "inode0", MockMemBlock(1000), "input0")
    ]

    output_tensor_info = [
        (MockEdge("output"), 1, 10, 10, 1, "onode0", MockMemBlock(100))
    ]

    memory_limit = 200  # 10 rows * 100 = 1000 > 200 -> 5 sub-tiles

    print_tile_specs("BEFORE", output_tile_specs, trimmed_input_tiles)

    new_out, new_in = apply_input_sub_tiling(
        output_tile_specs, trimmed_input_tiles,
        input_tensor_info, output_tensor_info,
        memory_limit
    )

    print_tile_specs("AFTER", new_out, new_in)

    # Should have 5 iterations
    assert len(new_out[0][0]) == 5, f"Expected 5 iterations, got {len(new_out[0][0])}"

    # Output sizes: [0, 0, 0, 0, 1]
    expected_out_sizes = [0, 0, 0, 0, 1]
    assert new_out[0][1] == expected_out_sizes, f"Expected {expected_out_sizes}, got {new_out[0][1]}"

    # Input sizes: [2, 2, 2, 2, 2]
    expected_in_sizes = [2, 2, 2, 2, 2]
    assert new_in["input0"][1] == expected_in_sizes, f"Expected {expected_in_sizes}, got {new_in['input0'][1]}"

    print("\n✅ Test Case 6 PASSED: Single tile correctly sub-tiled")


def test_case_7_exact_fit():
    """
    Test Case 7: Input exactly fits memory limit (boundary case)
    """
    print("\n" + "="*70)
    print("TEST CASE 7: Exact fit (boundary case)")
    print("="*70)

    output_tile_specs = {
        0: ([0, 1], [1, 1])
    }

    trimmed_input_tiles = {
        "input0": ([0, 2], [2, 1])  # First tile exactly 200 bytes
    }

    input_tensor_info = [
        (MockEdge("input"), 3, 10, 10, 1, "inode0", MockMemBlock(300), "input0")
    ]

    output_tensor_info = [
        (MockEdge("output"), 2, 10, 10, 1, "onode0", MockMemBlock(200))
    ]

    memory_limit = 200  # 2 rows * 100 = 200 == 200 (exactly fits)

    print_tile_specs("BEFORE", output_tile_specs, trimmed_input_tiles)
    print(f"\n  Memory: 2 rows * 100 bytes = 200 == limit -> should NOT sub-tile")

    new_out, new_in = apply_input_sub_tiling(
        output_tile_specs, trimmed_input_tiles,
        input_tensor_info, output_tensor_info,
        memory_limit
    )

    print_tile_specs("AFTER", new_out, new_in)

    # Should be unchanged (exact fit, no sub-tiling)
    assert new_out[0][0] == [0, 1], f"Expected unchanged bases"
    assert new_out[0][1] == [1, 1], f"Expected unchanged sizes"

    print("\n✅ Test Case 7 PASSED: Exact fit handled correctly (no sub-tiling)")


def test_case_8_docstring_example():
    """
    Test Case 8: Example from docstring

    From docstring:
      Original: output bases=[0,1,2], sizes=[1,1,1], input sizes=[8,0,0] (first needs 8 rows)
      If 8 rows too big, split into 4 sub-tiles of 2 rows each:
      Result: output bases=[0,0,0,0,1,2], sizes=[0,0,0,1,1,1]
              input bases=[0,2,4,6,8,8], sizes=[2,2,2,2,0,0]
    """
    print("\n" + "="*70)
    print("TEST CASE 8: Docstring example verification")
    print("="*70)

    output_tile_specs = {
        0: ([0, 1, 2], [1, 1, 1])
    }

    # Note: input bases [0, 8, 8] means tile 1 and 2 start at same place
    # sizes [8, 0, 0] means only first tile needs input
    trimmed_input_tiles = {
        "input0": ([0, 8, 8], [8, 0, 0])
    }

    # 800 bytes / 8 rows = 100 bytes per row
    input_tensor_info = [
        (MockEdge("input"), 8, 10, 10, 1, "inode0", MockMemBlock(800), "input0")
    ]

    output_tensor_info = [
        (MockEdge("output"), 3, 10, 10, 1, "onode0", MockMemBlock(300))
    ]

    # Memory limit: 200 bytes -> 2 rows max -> 4 sub-tiles for 8 rows
    memory_limit = 200

    print_tile_specs("BEFORE", output_tile_specs, trimmed_input_tiles)
    print(f"\n  Expected from docstring:")
    print(f"    output bases=[0,0,0,0,1,2], sizes=[0,0,0,1,1,1]")
    print(f"    input bases=[0,2,4,6,8,8], sizes=[2,2,2,2,0,0]")

    new_out, new_in = apply_input_sub_tiling(
        output_tile_specs, trimmed_input_tiles,
        input_tensor_info, output_tensor_info,
        memory_limit
    )

    print_tile_specs("AFTER", new_out, new_in)

    # Verify against docstring
    expected_out_bases = [0, 0, 0, 0, 1, 2]
    expected_out_sizes = [0, 0, 0, 1, 1, 1]
    expected_in_bases = [0, 2, 4, 6, 8, 8]
    expected_in_sizes = [2, 2, 2, 2, 0, 0]

    assert new_out[0][0] == expected_out_bases, f"Output bases: expected {expected_out_bases}, got {new_out[0][0]}"
    assert new_out[0][1] == expected_out_sizes, f"Output sizes: expected {expected_out_sizes}, got {new_out[0][1]}"
    assert new_in["input0"][0] == expected_in_bases, f"Input bases: expected {expected_in_bases}, got {new_in['input0'][0]}"
    assert new_in["input0"][1] == expected_in_sizes, f"Input sizes: expected {expected_in_sizes}, got {new_in['input0'][1]}"

    print("\n✅ Test Case 8 PASSED: Docstring example verified")


def run_all_tests():
    """Run all test cases"""
    print("\n" + "="*70)
    print("  RUNNING ALL TEST CASES FOR apply_input_sub_tiling")
    print("="*70)

    try:
        test_case_1_no_subtiling_needed()
        test_case_2_first_tile_needs_subtiling()
        test_case_3_multiple_tiles_need_subtiling()
        test_case_4_multiple_inputs()
        test_case_5_multiple_outputs()
        test_case_6_edge_case_single_tile()
        test_case_7_exact_fit()
        test_case_8_docstring_example()

        print("\n" + "="*70)
        print("  ALL TESTS PASSED! ✅")
        print("="*70)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
