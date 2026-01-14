#!/usr/bin/env python3
"""
Scan Register Policy Table Generator

This script generates NoC paths and PolicyTable_2D for scan register value propagation.
Each IMCE node receives scan register values from its master inode through the NoC.

The scan register value propagation is done independently before other operations,
so it doesn't need to consider other operation dependencies.
"""

import os
import sys
import json
import pprint
from typing import Dict, List, Tuple, Optional

# Add TVM to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../python"))

import tvm
from tvm import relay
from tvm.contrib.imcflow import ImcflowDeviceConfig, NodeID, RouterEntry


class ScanRegPolicyGenerator:
    """
    Generate NoC paths and policy tables for scan register value propagation.
    
    Each IMCE node needs to receive scan register values from its master inode.
    This happens before any other computation, so paths are independent.
    """
    
    def __init__(self):
        self.NoCPaths = {}
        self.PolicyTable_2D = {}
        self.table_capacity = 32
        
    def construct_noc_path(self) -> Dict:
        """
        Construct NoC paths for scan register values.
        
        For each IMCE node, create a path from its master inode to the IMCE.
        Path format: {NodeID (destination IMCE): (source_inode, dest_imce, None)}
        
        Returns:
            Dict mapping each IMCE NodeID to its path tuple (source, dest, split_idx)
        """
        print("\n" + "="*60)
        print("CONSTRUCTING NOC PATHS FOR SCAN REGISTER VALUES")
        print("="*60)
        
        noc_paths = {}
        
        # For each IMCE node, create a path from its master inode
        for imce_node in NodeID.imces():
            # Get the master inode for this IMCE
            master_inode = imce_node.master()
            
            # Create path entry: (source, destination, split_idx)
            # split_idx is None because scan register is not split
            path_entry = (master_inode, imce_node, None)
            
            # Use IMCE node as key (similar to instruction edges in constructNoCPathDict)
            noc_paths[imce_node] = path_entry
            
            print(f"  Path: {master_inode.name} -> {imce_node.name}")
        
        self.NoCPaths = noc_paths
        print(f"\nTotal paths created: {len(noc_paths)}")
        return noc_paths
    
    def gen_policy_table(self) -> Dict:
        """
        Generate PolicyTable_2D for scan register paths.
        
        This follows the same logic as PolicyTableGenerator.generate_policy_table()
        but only for scan register paths (inode -> imce).
        
        Returns:
            Dict mapping NodeID to list of policy table entries
        """
        print("\n" + "="*60)
        print("GENERATING POLICY TABLE FOR SCAN REGISTER")
        print("="*60)
        
        if not self.NoCPaths:
            raise ValueError("NoCPaths not constructed. Call construct_noc_path() first.")
        
        # Initialize policy tables for all nodes
        # Each policy table starts with an all-zeros entry at address 0
        zero_entry = {
            "Local": {"enable": False, "chunk_index": 0, "addr": 0, "ksel": 0},
            "North": {"enable": False, "addr": 0},
            "East": {"enable": False, "addr": 0},
            "South": {"enable": False, "addr": 0},
            "West": {"enable": False, "addr": 0}
        }
        policy_tables = {node_id: [zero_entry.copy()] for node_id in NodeID}
        
        def get_direction(source_coord: Tuple[int, int], dest_coord: Tuple[int, int]) -> Optional[str]:
            """Determine the direction from source to destination."""
            if source_coord[1] < dest_coord[1]:
                return "East"
            elif source_coord[1] > dest_coord[1]:
                return "West"
            elif source_coord[0] < dest_coord[0]:
                return "South"
            elif source_coord[0] > dest_coord[0]:
                return "North"
            return None
        
        def check_path_capacity(path_coords: List[Tuple[int, int]]) -> bool:
            """Check if all nodes in the path have available capacity."""
            for coord in path_coords:
                node = NodeID.from_coord(coord[0], coord[1])
                if len(policy_tables[node]) >= self.table_capacity:
                    return False
            return True
        
        def get_path_coords(source_coord: Tuple[int, int], 
                          dest_coord: Tuple[int, int], 
                          is_xy_routing: bool = True) -> List[Tuple[int, int]]:
            """
            Get list of coordinates for the routing path.
            
            Args:
                source_coord: Starting coordinate (row, col)
                dest_coord: Destination coordinate (row, col)
                is_xy_routing: If True, route X then Y; if False, route Y then X
            
            Returns:
                List of coordinates along the path (excluding source)
            """
            path_coords = []
            current_coord = source_coord
            
            if is_xy_routing:
                # Move horizontally first (X direction)
                while current_coord[1] != dest_coord[1]:
                    next_coord = (current_coord[0],
                                current_coord[1] + (1 if current_coord[1] < dest_coord[1] else -1))
                    path_coords.append(next_coord)
                    current_coord = next_coord
                
                # Then vertically (Y direction)
                while current_coord[0] != dest_coord[0]:
                    next_coord = (current_coord[0] + (1 if current_coord[0] < dest_coord[0] else -1),
                                current_coord[1])
                    path_coords.append(next_coord)
                    current_coord = next_coord
            else:
                # Move vertically first (Y direction)
                while current_coord[0] != dest_coord[0]:
                    next_coord = (current_coord[0] + (1 if current_coord[0] < dest_coord[0] else -1),
                                current_coord[1])
                    path_coords.append(next_coord)
                    current_coord = next_coord
                
                # Then horizontally (X direction)
                while current_coord[1] != dest_coord[1]:
                    next_coord = (current_coord[0],
                                current_coord[1] + (1 if current_coord[1] < dest_coord[1] else -1))
                    path_coords.append(next_coord)
                    current_coord = next_coord
            
            # Check policy table capacity along the designated routing path
            if not check_path_capacity(path_coords):
                # If X-Y fails, try Y-X routing
                path_coords = get_path_coords(source_coord, dest_coord, False)
                if not check_path_capacity(path_coords):
                    raise ValueError(f"Routing failed for both X-Y and Y-X from {source_coord} to {dest_coord}!")
            
            return path_coords
        
        def create_policy_entry(source_node: NodeID, dest_node: NodeID) -> None:
            """
            Create policy table entries for a path from source to destination.
            
            Args:
                source_node: Source NodeID (inode)
                dest_node: Destination NodeID (imce)
            """
            source_coord = NodeID.to_coord(source_node)
            dest_coord = NodeID.to_coord(dest_node)
            
            # If same node, no routing needed
            if source_coord == dest_coord:
                print(f"  Warning: Source and dest are the same node: {source_node.name}")
                return
            
            # Get routing path
            path_coords = get_path_coords(source_coord, dest_coord, is_xy_routing=True)
            
            current_coord = source_coord
            current_node = source_node
            
            # Create entries along the path
            for next_coord in path_coords:
                direction = get_direction(current_coord, next_coord)
                next_node = NodeID.from_coord(next_coord[0], next_coord[1])
                
                # Create routing entry for current node
                entry = {
                    "Local": {"enable": False, "chunk_index": 0, "addr": 0, "ksel": 0},
                    "North": {"enable": False, "addr": 0},
                    "East": {"enable": False, "addr": 0},
                    "South": {"enable": False, "addr": 0},
                    "West": {"enable": False, "addr": 0}
                }
                
                target_addr = len(policy_tables[next_node])
                entry[direction]["addr"] = target_addr
                entry[direction]["enable"] = True
                policy_tables[current_node].append(entry)
                
                # Move to next node
                current_coord = next_coord
                current_node = next_node
            
            # Create final entry for destination node (enable Local)
            final_entry = {
                "Local": {"enable": True, "chunk_index": 0, "addr": 0, "ksel": 0},
                "North": {"enable": False, "addr": 0},
                "East": {"enable": False, "addr": 0},
                "South": {"enable": False, "addr": 0},
                "West": {"enable": False, "addr": 0}
            }
            policy_tables[dest_node].append(final_entry)
            
            print(f"  Created path: {source_node.name} -> {dest_node.name} "
                  f"(hops: {len(path_coords)})")
        
        # Generate policy table entries for all scan register paths
        for imce_node, (source_node, dest_node, split_idx) in self.NoCPaths.items():
            create_policy_entry(source_node, dest_node)
        
        self.PolicyTable_2D = policy_tables
        
        # Print summary
        print(f"\n{'='*60}")
        print("POLICY TABLE SUMMARY")
        print(f"{'='*60}")
        for node_id in NodeID:
            entry_count = len(policy_tables[node_id])
            if entry_count > 1:  # Skip nodes with only zero entry
                print(f"{node_id.name:12s}: {entry_count:3d} entries "
                      f"({entry_count}/{self.table_capacity} capacity)")
        
        return policy_tables
    
    def export_policy_table(self, output_path: str) -> None:
        """
        Export the policy table to a text file for inspection.
        
        Args:
            output_path: Path to output file
        """
        if not self.PolicyTable_2D:
            raise ValueError("Policy table not generated. Call gen_policy_table() first.")
        
        with open(output_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("SCAN REGISTER POLICY TABLE\n")
            f.write("="*60 + "\n\n")
            
            for node_id in NodeID:
                entries = self.PolicyTable_2D[node_id]
                if len(entries) <= 1:  # Skip nodes with only zero entry
                    continue
                
                f.write(f"\n{node_id.name} ({len(entries)} entries):\n")
                f.write("-" * 60 + "\n")
                
                for idx, entry in enumerate(entries):
                    f.write(f"  Entry {idx}:\n")
                    for direction in ["Local", "North", "East", "South", "West"]:
                        dir_entry = entry[direction]
                        if dir_entry["enable"]:
                            if direction == "Local":
                                f.write(f"    {direction:5s}: enable=True, "
                                       f"chunk_index={dir_entry['chunk_index']}, "
                                       f"addr={dir_entry['addr']}, "
                                       f"ksel={dir_entry['ksel']}\n")
                            else:
                                f.write(f"    {direction:5s}: enable=True, "
                                       f"addr={dir_entry['addr']}\n")
                    f.write("\n")
        
        print(f"\nPolicy table exported to: {output_path}")
    
    def export_noc_paths(self, output_path: str) -> None:
        """
        Export NoC paths to a text file.
        
        Args:
            output_path: Path to output file
        """
        if not self.NoCPaths:
            raise ValueError("NoC paths not constructed. Call construct_noc_path() first.")
        
        with open(output_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("SCAN REGISTER NOC PATHS\n")
            f.write("="*60 + "\n\n")
            
            for imce_node, (source_node, dest_node, split_idx) in sorted(
                self.NoCPaths.items(), key=lambda x: x[0].value):
                f.write(f"{imce_node.name}: {source_node.name} -> {dest_node.name} "
                       f"(split_idx: {split_idx})\n")
        
        print(f"NoC paths exported to: {output_path}")


def run_scan_reg_policy_generation(output_dir: str = "./scan_reg_policy") -> None:
    """
    Main function to run scan register policy table generation.
    
    Args:
        output_dir: Directory to save output files
    """
    print("\n" + "="*60)
    print("SCAN REGISTER POLICY TABLE GENERATION")
    print("="*60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create generator instance
    generator = ScanRegPolicyGenerator()
    
    # Step 1: Construct NoC paths
    noc_paths = generator.construct_noc_path()
    
    # Step 2: Generate policy table
    policy_table = generator.gen_policy_table()
    
    # Step 3: Export results
    generator.export_noc_paths(os.path.join(output_dir, "scan_reg_noc_paths.txt"))
    generator.export_policy_table(os.path.join(output_dir, "scan_reg_policy_table.txt"))
    
    print("\n" + "="*60)
    print("SCAN REGISTER POLICY GENERATION COMPLETE")
    print("="*60)
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print(f"Total IMCE nodes: {len(NodeID.imces())}")
    print(f"Total paths created: {len(noc_paths)}")
    
    return generator


if __name__ == "__main__":
    """
    Command-line interface for scan register policy generation.
    
    Usage:
        python scan_reg_policy_gen.py [output_dir]
    
    Example:
        python scan_reg_policy_gen.py ./my_output
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate NoC paths and policy table for scan register values"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./scan_reg_policy",
        help="Output directory for generated files (default: ./scan_reg_policy)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Run the generation
    try:
        generator = run_scan_reg_policy_generation(output_dir=args.output_dir)
        
        if args.verbose:
            print("\n" + "="*60)
            print("DETAILED POLICY TABLE")
            print("="*60)
            pprint.pprint(generator.PolicyTable_2D)
        
        print("\n✅ Success!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
