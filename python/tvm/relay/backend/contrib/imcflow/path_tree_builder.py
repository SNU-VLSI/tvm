"""
Path Tree Builder for NoC Routing (Phase 2)

DEPRECATED: This module has been merged into policy_table_builder.py.
Please use policy_table_builder.py directly.

This module now re-exports from policy_table_builder for backward compatibility.
"""

import warnings

warnings.warn(
    "path_tree_builder module is deprecated. "
    "Please import from policy_table_builder instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export all classes from policy_table_builder for backward compatibility
from .policy_table_builder import (
    PathTreeNode,
    MulticastTree,
    PathTreeBuildResult,
    PathTreeBuilder,
    build_path_trees,
)

# Re-export types for backward compatibility
from .joint_pnr_ilp import (
    Coord,
    Edge,
    Direction,
    HWCommodity,
)

__all__ = [
    'PathTreeNode',
    'MulticastTree',
    'PathTreeBuildResult',
    'PathTreeBuilder',
    'build_path_trees',
]
