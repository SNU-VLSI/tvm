"""Context and helper utilities for IMCFlow code generation.

This module provides the BuilderContext class that encapsulates:
1. Shared state (edges, codeblocks, composite context)
2. A wrapped relay.Call with context-aware helper methods

Usage in handlers:
    def handle(self, call_ctx: BuilderContext) -> None:
        hid = call_ctx.get_hid()
        in_edges = call_ctx.get_input_edges()
        out_edge = call_ctx.get_output_edge()
"""

import re
from typing import Optional, List, Dict, Any
from tvm import relay
from tvm.contrib.imcflow import TensorID, TensorEdge
from tvm.relay.backend.contrib.imcflow.transform import getNodeID
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from tvm.relay.backend.contrib.imcflow.codeblock import CodePhase
from tvm.relay.backend.contrib.imcflow.imce_codeblock import ConvBlock


class BuilderContext:
    """Unified context for code generation with wrapped relay.Call.

    This class combines:
    1. Global state (edges, codeblocks, composite context)
    2. Per-call context (wrapped relay.Call with helper methods)

    This enables a clean API where handlers receive a context object with
    everything they need:
        def handle(self, call_ctx: BuilderContext) -> None:
            hid = call_ctx.get_hid()
            edges = call_ctx.get_input_edges()

    Attributes:
        call: The wrapped relay.Call object
        edges: Set of TensorEdge instances from InternalEdgeAnnotator
        codeblocks: CodeBlockManager for accumulating generated blocks
        curr_composite_id: Current composite function's node ID (or None)
        curr_conv_block: Current ConvBlock being processed (or None)
        last_tuple_idx: Last tuple index seen (used by MinMaxQuantBlock)
    """

    def __init__(self, call: relay.Call, edges: set, codeblocks,
                 curr_composite_id: Optional[int] = None,
                 curr_conv_block: Optional[ConvBlock] = None,
                 last_tuple_idx: Optional[int] = None):
        """Initialize context with call and shared state.

        Args:
            call: The relay.Call to wrap
            edges: Set of TensorEdge instances
            codeblocks: CodeBlockManager (ImceCodeBlockManager or InodeCodeBlockManager)
            curr_composite_id: Current composite context (optional)
            curr_conv_block: Current ConvBlock (optional)
            last_tuple_idx: Last tuple index (optional)
        """
        self.call = call
        self.edges = edges
        self.codeblocks = codeblocks
        self.curr_composite_id = curr_composite_id
        self.curr_conv_block = curr_conv_block
        self.last_tuple_idx = last_tuple_idx

    def fork(self, call: relay.Call) -> 'BuilderContext':
        """Create a new context for a different call, preserving shared state.

        This is useful when a handler needs to process multiple calls.

        Args:
            call: The new relay.Call to wrap

        Returns:
            New BuilderContext with same shared state but different call
        """
        return BuilderContext(
            call=call,
            edges=self.edges,
            codeblocks=self.codeblocks,
            curr_composite_id=self.curr_composite_id,
            curr_conv_block=self.curr_conv_block,
            last_tuple_idx=self.last_tuple_idx
        )

    # Hardware and node mapping helpers

    def get_hid(self):
        """Get hardware node ID for this call.

        Uses current composite context if available.

        Returns:
            Hardware node ID from DevConfig
        """
        node_id = self.curr_composite_id or getNodeID(self.call)
        return DevConfig().get_hw_node(node_id)

    def get_graph_node_id(self):
        """Get graph node ID for this call.

        Returns a tuple (composite_id, call_id) if inside a composite,
        otherwise returns just the call_id.

        Returns:
            Node ID or tuple of (composite_id, node_id)
        """
        if self.curr_composite_id:
            return (self.curr_composite_id, getNodeID(self.call))
        else:
            return getNodeID(self.call)

    def get_gid(self):
        """Get generalized graph ID for this call.

        Handles special case of imcflow composite functions.

        Returns:
            Graph node ID (may be tuple for composites)
        """
        if (hasattr(self.call, "op") and
            isinstance(self.call.op, relay.Function) and
            "Composite" in self.call.op.attrs and
            re.match(r"imcflow\..*", self.call.op.attrs["Composite"])):
            gid = (getNodeID(self.call), getNodeID(self.call.op.body))
        else:
            gid = getNodeID(self.call)
        return gid

    # Tensor ID helpers

    def get_tensor_id(self, tag: str) -> TensorID:
        """Get TensorID for this call with a specific tag.

        Args:
            tag: Tensor tag (e.g., "data", "weight", "odata")

        Returns:
            TensorID instance
        """
        return TensorID(self.get_graph_node_id(), tag)

    # Edge query helpers

    def get_input_edge(self) -> TensorEdge:
        """Get the single input edge for this call.

        Returns:
            The input TensorEdge

        Raises:
            AssertionError: If there is not exactly one input edge
        """
        edges = self.get_input_edges()
        assert len(edges) == 1, "Input edge must be unique"
        return edges[0]

    def get_input_edges(self) -> List[TensorEdge]:
        """Get all input edges for this call.

        Returns:
            List of input TensorEdge instances
        """
        return [edge for edge in self.edges
                if edge.dst_inner_gid_match(getNodeID(self.call))]

    def get_output_edge(self) -> TensorEdge:
        """Get the single output edge for this call.

        Returns:
            The output TensorEdge

        Raises:
            AssertionError: If there is not exactly one output edge
        """
        edges = self.get_output_edges()
        assert len(edges) == 1, "Output edge must be unique"
        return edges[0]

    def get_output_edges(self) -> List[TensorEdge]:
        """Get all output edges for this call.

        Returns:
            List of output TensorEdge instances
        """
        return [edge for edge in self.edges
                if edge.src_inner_gid_match(getNodeID(self.call))]

    def get_tensor_edge_from_tag(self, tag: str) -> TensorEdge:
        """Get tensor edge for this call with a specific tag.

        Args:
            tag: Tensor tag to look up

        Returns:
            TensorEdge instance from DevConfig
        """
        tid = self.get_tensor_id(tag)
        te = DevConfig().get_tensor_edge(tid)
        return te

    # Code block query helpers

    def get_conv_block_by_hid(self, hid) -> Optional[ConvBlock]:
        """Find ConvBlock for a given hardware node ID.

        Searches through EXEC phase blocks to find the ConvBlock.

        Args:
            hid: Hardware node ID

        Returns:
            ConvBlock instance or None if not found
        """
        for block in self.codeblocks.blocks[hid][CodePhase.EXEC]:
            if isinstance(block, ConvBlock):
                return block
        return None

    # Call argument helpers

    def get_arg_keys(self) -> List[str]:
        """Get argument names for this call.

        Returns:
            List of argument names
        """
        return [arg.name for arg in self.call.op.arguments]

    def get_arg_dict(self) -> Dict[str, Any]:
        """Get dictionary mapping argument names to values.

        Returns:
            Dictionary of {arg_name: arg_value}
        """
        args = {}
        for idx, arg in enumerate(self.call.op.arguments):
            args[arg.name] = self.call.args[idx]
        return args

    def get_arg_shape_dict(self) -> Dict[str, tuple]:
        """Get dictionary mapping argument names to shapes.

        Returns:
            Dictionary of {arg_name: shape_tuple}
        """
        args_shape = {}
        for idx, arg in enumerate(self.call.op.arguments):
            args_shape[arg.name] = tuple(self.call.type_args[idx].shape)
        return args_shape

    # Debug helpers

    def inspect(self, tmp):
        """Debug utility to inspect node mappings for this call.

        Prints hardware node ID and tensor IDs.

        Args:
            tmp: Object being inspected
        """
        graph_node_id = self.get_graph_node_id()
        if graph_node_id in DevConfig().HWNodeMap.keys():
            hw_node_id = DevConfig().HWNodeMap[graph_node_id]
        else:
            hw_node_id = None
        tid = DevConfig().get_tensor_ids_from_graph_node_id(graph_node_id)
        print(f"{tmp.__class__} graph_node_id: {graph_node_id}, "
              f"hw_node_id: {hw_node_id}, tid: {tid}")
        print("----------------------")
