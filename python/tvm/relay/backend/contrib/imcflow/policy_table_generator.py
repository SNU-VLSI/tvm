"""
Policy Table Generator for NoC Routing (Phase 3)

DEPRECATED: This module has been merged into policy_table_builder.py.
Please use policy_table_builder.py directly.

This module now re-exports from policy_table_builder for backward compatibility.
"""

import warnings

warnings.warn(
    "policy_table_generator module is deprecated. "
    "Please import from policy_table_builder instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export all classes from policy_table_builder for backward compatibility
from .policy_table_builder import (
    PolicyEntry,
    NodeCapacityError,
    PolicyTableGenerator,
    EdgeInfoGenerator,
    MemoryAllocator,
    PathTreeNode,
    MulticastTree,
    PathTreeBuildResult,
    getInnerNodeID,
)

# Re-export the top-level PolicyTableBuilder
from .policy_table_builder import PolicyTableBuilder as NewPolicyTableBuilder

# Alias PolicyTableGenerator to the old class name for backward compatibility
# (The old PolicyTableBuilder is now PolicyTableGenerator in the new file)
PolicyTableBuilder = PolicyTableGenerator


def generate_policy_tables(
    tree_result: PathTreeBuildResult,
    noc_paths,
    func_name: str,
    table_capacity: int = 32,
):
    """Convenience function to generate policy tables and update device config.

    DEPRECATED: Use PolicyTableBuilder.build() from policy_table_builder instead.

    Args:
        tree_result: Result from Phase 2 (PathTreeBuilder)
        noc_paths: Original NoCPaths dictionary
        func_name: Name of the function being processed
        table_capacity: Maximum entries per node

    Returns:
        Generated policy tables
    """
    from tvm.contrib.imcflow import ImcflowDeviceConfig

    # Phase 3a: Generate policy tables
    generator = PolicyTableGenerator(table_capacity=table_capacity)
    policy_tables = generator.generate(tree_result, noc_paths)

    # Phase 3b: Generate EdgeInfo
    edge_info_gen = EdgeInfoGenerator(
        policy_tables,
        generator.get_router_entries(),
        noc_paths,
    )
    edge_info_gen.generate(func_name)

    # Phase 3c: Allocate memory
    allocator = MemoryAllocator(policy_tables)
    allocator.allocate(func_name)

    # Store in device config
    ImcflowDeviceConfig().PolicyTableDict[func_name] = policy_tables

    return policy_tables


__all__ = [
    'PolicyEntry',
    'NodeCapacityError',
    'PolicyTableBuilder',
    'PolicyTableGenerator',
    'EdgeInfoGenerator',
    'MemoryAllocator',
    'generate_policy_tables',
]
