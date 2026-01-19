"""
Test file for NoC Path Visualization

This test creates mock NoC paths and verifies the visualization output.
"""
import sys
import os
import tempfile

# Add TVM python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../python'))

from tvm.contrib.imcflow import (
    ImcflowDeviceConfig,
    NodeID,
    TensorEdge,
    TensorID,
    TensorEdgeInfo,
    RouterEntry,
)
from tvm.relay.backend.contrib.imcflow.transform import TensorPathVisualizer


def create_mock_router_entry(node_id):
    """Create a mock RouterEntry with the given node_id."""
    return RouterEntry(router_id=node_id, address=0, data={})


def create_mock_tensor_edge(src_graph_id, dst_graph_id, tensor_type='odata'):
    """Create a mock TensorEdge."""
    src_id = TensorID(graph_node_id=src_graph_id, tensor_type=tensor_type)
    dst_id = TensorID(graph_node_id=dst_graph_id, tensor_type=tensor_type)
    return TensorEdge(src_id=src_id, dst_id=dst_id, split_idx=None)


def create_mock_edge_info(path_node_ids):
    """Create mock TensorEdgeInfo with routing path."""
    policy_info = [create_mock_router_entry(node_id) for node_id in path_node_ids]
    return TensorEdgeInfo(policy_info=policy_info, data_block=None, fifo_id=1)


def setup_mock_config(func_name, edges_and_paths):
    """
    Setup mock ImcflowDeviceConfig with test data.

    Args:
        func_name: Name of the function
        edges_and_paths: List of (TensorEdge, src_node, dst_node, path_node_ids)
    """
    config = ImcflowDeviceConfig()

    # Clear any existing data for this function
    config.NoCPaths[func_name] = {}
    config.TensorEdgeListDict[func_name] = []

    # Setup NoCPaths and TensorEdgetoInfo
    for edge, src_node, dst_node, path_node_ids in edges_and_paths:
        config.NoCPaths[func_name][edge] = (src_node, dst_node, 0)
        config.TensorEdgetoInfo[edge] = create_mock_edge_info(path_node_ids)
        config.TensorEdgeListDict[func_name].append(edge)

    return config


def test_noc_visualization():
    """Main test function."""
    print("=" * 60)
    print("NoC Visualization Test")
    print("=" * 60)

    # Create output directory
    output_dir = tempfile.mkdtemp(prefix="noc_vis_test_")
    print(f"Output directory: {output_dir}")

    func_name = "test_function"

    # Define test cases: (edge, src_node, dst_node, path_node_ids)
    # Each path tests different scenarios
    edges_and_paths = [
        # Path 1: Simple horizontal path (row 0, col 1->3)
        (
            create_mock_tensor_edge(10, 11, 'odata'),
            NodeID.imce_0_1, NodeID.imce_0_3,
            [NodeID.imce_0_1, NodeID.imce_0_2, NodeID.imce_0_3]
        ),

        # Path 2: Simple vertical path (col 1, row 0->2)
        (
            create_mock_tensor_edge(20, 21, 'odata'),
            NodeID.imce_0_1, NodeID.imce_2_1,
            [NodeID.imce_0_1, NodeID.imce_1_1, NodeID.imce_2_1]
        ),

        # Path 3: L-shaped (right then down): inode_0_0 -> imce_0_1 -> imce_1_1 -> imce_1_3
        # This tests corner handling: horizontal then vertical then horizontal
        (
            create_mock_tensor_edge(30, 31, 'odata'),
            NodeID.inode_0_0, NodeID.imce_1_3,
            [NodeID.inode_0_0, NodeID.imce_0_1, NodeID.imce_1_1, NodeID.imce_1_2, NodeID.imce_1_3]
        ),

        # Path 4: L-shaped sharing initial segment with Path 3
        # inode_0_0 -> imce_0_1 -> imce_1_1 -> imce_2_1
        (
            create_mock_tensor_edge(40, 41, 'odata'),
            NodeID.inode_0_0, NodeID.imce_2_1,
            [NodeID.inode_0_0, NodeID.imce_0_1, NodeID.imce_1_1, NodeID.imce_2_1]
        ),

        # Path 5: Reverse L (left then up): imce_2_3 -> imce_2_1 -> imce_1_1 -> imce_0_1
        (
            create_mock_tensor_edge(50, 51, 'odata'),
            NodeID.imce_2_3, NodeID.imce_0_1,
            [NodeID.imce_2_3, NodeID.imce_2_2, NodeID.imce_2_1, NodeID.imce_1_1, NodeID.imce_0_1]
        ),
    ]

    print(f"Created {len(edges_and_paths)} test paths:")
    for i, (edge, src, dst, path) in enumerate(edges_and_paths):
        path_str = " -> ".join([n.name for n in path])
        print(f"  Path {i+1}: {path_str}")

    # Setup mock config
    setup_mock_config(func_name, edges_and_paths)
    print(f"\nSetup mock config for function: {func_name}")

    # Create visualizer and generate visualization
    print("\nGenerating visualization...")
    visualizer = TensorPathVisualizer(output_dir=output_dir)
    visualizer.visualize_function(func_name)

    # List generated files
    func_dir = os.path.join(output_dir, func_name)
    if os.path.exists(func_dir):
        print(f"\nGenerated files in {func_dir}:")
        for f in sorted(os.listdir(func_dir)):
            fpath = os.path.join(func_dir, f)
            size = os.path.getsize(fpath)
            print(f"  {f} ({size} bytes)")

    # Return the output paths for inspection
    group_data_path = os.path.join(func_dir, "group_data.png")
    odata_path = os.path.join(func_dir, "odata.png")

    print("\n" + "=" * 60)
    print(f"Test complete!")
    print(f"Main output: {group_data_path}")
    print("=" * 60)

    return output_dir, group_data_path, odata_path


if __name__ == "__main__":
    output_dir, group_data_path, odata_path = test_noc_visualization()
    print(f"\nOutput files:")
    print(f"  {group_data_path}")
    print(f"  {odata_path}")
