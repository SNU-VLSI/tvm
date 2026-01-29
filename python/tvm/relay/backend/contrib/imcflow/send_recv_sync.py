"""
Send-Recv Pair Synchronization Support

This module provides UUID-based synchronization for send-recv pairs across
multi-node hardware. Each send-recv pair (including multicasts) is assigned
a unique UUID, and all participating nodes synchronize after send/recv operations.
"""

import tvm
from tvm import relay
from typing import List, Dict, Set, Tuple
from tvm.contrib.imcflow import TensorEdge, NodeID, TensorEdgeInfo
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from tvm.relay.op.contrib.imcflow import CustomIDToNode
from tvm.relay.backend.contrib.imcflow.transform_utils import getInnerNodeID
import logging


class SendRecvPair:
    """Represents a send-recv pair with multicast support"""

    def __init__(self, uuid: int, sender_node: NodeID, receiver_nodes: Set[NodeID], edges: List[TensorEdge]):
        self.uuid = uuid
        self.sender_node = sender_node
        self.receiver_nodes = receiver_nodes  # Can be multiple for multicast
        self.edges = edges  # All edges in this pair share the same UUID

    @property
    def all_nodes(self) -> List[NodeID]:
        """Returns all participating nodes (sender + all receivers)"""
        return [self.sender_node] + sorted(list(self.receiver_nodes), key=lambda x: x.value)

    def __repr__(self):
        receiver_str = ','.join([r.name for r in sorted(self.receiver_nodes, key=lambda x: x.value)])
        return f"Pair(uuid={self.uuid}, {self.sender_node.name}->[{receiver_str}])"


class SendRecvPairManager:
    """Manages send-recv pair UUID assignment with multicast support

    Groups tensor edges by source graph node to handle multicasts.
    For example, if node A sends to both B and C, edges A->B and A->C
    are grouped into one pair with UUID assigned to all three nodes.
    """

    def __init__(self, edges: List[TensorEdge], exclude_const: bool = True, filter_contention: bool = True):
        """Initialize pair manager and assign UUIDs

        Args:
            edges: List of tensor edges to process
            exclude_const: If True, skip constant edges (no sync needed)
        """
        self.pairs: Dict[int, SendRecvPair] = {}  # {uuid: SendRecvPair}
        self.edge_to_pair: Dict[TensorEdge, SendRecvPair] = {}  # {edge: SendRecvPair}
        self.exclude_const = exclude_const
        self.filter_contention = filter_contention
        self._assign_uuids(edges)
        if self.filter_contention:
            self._filter_pairs_with_contention()

        # Log assignment results
        print(f"[SendRecvPairManager] Assigned {len(self.pairs)} UUIDs for {len(edges)} edges")
        for pair in sorted(self.pairs.values(), key=lambda p: p.uuid):
            print(f"  {pair}")

    def _filter_pairs_with_contention(self):
      """
      Filter send-recv pairs for nodes that have multiple recvs or multiple sends
      """
      participation_count: Dict[NodeID, Tuple[int, int]] = {}
      # Count participation
      for pair in self.pairs.values(): 
          # Count sender
          if pair.sender_node not in participation_count:
              participation_count[pair.sender_node] = (0,0)
          send_count, recv_count = participation_count[pair.sender_node]
          participation_count[pair.sender_node] = (send_count + 1, recv_count)

          # Count receivers
          for rnode in pair.receiver_nodes:
              if rnode not in participation_count:
                  participation_count[rnode] = (0,0)
              send_count, recv_count = participation_count[rnode]
              participation_count[rnode] = (send_count, recv_count + 1)

      # Identify nodes with contention by role
      nodes_with_send_contention: Set[NodeID] = set()  # send_count > 1
      nodes_with_recv_contention: Set[NodeID] = set()  # recv_count > 1
      for node, (send_count, recv_count) in participation_count.items():
          if send_count > 1:
              nodes_with_send_contention.add(node)
          if recv_count > 1:
              nodes_with_recv_contention.add(node)

      # Filter pairs - only keep if contention matches the role
      filtered_pairs: Dict[int, SendRecvPair] = {}
      filtered_edge_to_pair: Dict[TensorEdge, SendRecvPair] = {}
      for pair in self.pairs.values():
          # Keep if sender has send contention
          sender_has_contention = pair.sender_node in nodes_with_send_contention
          # Keep if any receiver has recv contention
          receiver_has_contention = any(rnode in nodes_with_recv_contention for rnode in pair.receiver_nodes)

          if sender_has_contention or receiver_has_contention:
              filtered_pairs[pair.uuid] = pair
              for edge in pair.edges:
                  filtered_edge_to_pair[edge] = pair
      # Update
      # self.pairs = filtered_pairs
      # self.edge_to_pair = filtered_edge_to_pair
      self.pairs = {}
      self.edge_to_pair = {}


    def _assign_uuids(self, edges: List[TensorEdge]):
        """Assign UUIDs to send-recv pairs

        Groups edges by source graph node ID to handle multicasts.
        Each group gets a unique UUID (0-255).
        """
        print(f"[DEBUG _assign_uuids] Input edges ({len(edges)}):")
        for e in edges:
            print(f"  {e}")

        # Filter out constant edges if requested
        filtered_edges = []
        for edge in edges:
            if self.exclude_const:
                src_graph_id = edge.src_id.graph_node_id
                try:
                    from tvm.relay.dataflow_pattern import is_constant
                    ConstPat = is_constant()
                    if ConstPat.match(CustomIDToNode()[src_graph_id]):
                        print(f"[DEBUG _assign_uuids] Skipping constant edge: {edge}")
                        continue  # Skip constant edges
                except (KeyError, Exception):
                    pass
            filtered_edges.append(edge)

        print(f"[DEBUG _assign_uuids] Filtered edges ({len(filtered_edges)}):")
        for e in filtered_edges:
            print(f"  {e}")

        # Group edges by source graph node (handles multicast)
        # Key: (outer_src_gid, inner_src_gid)
        edge_groups: Dict[Tuple, List[TensorEdge]] = {}

        for edge in filtered_edges:
            src_gid = edge.src_id.graph_node_id
            # Normalize to tuple format
            if isinstance(src_gid, tuple):
                key = src_gid
            else:
                key = (src_gid,)

            if key not in edge_groups:
                edge_groups[key] = []
            edge_groups[key].append(edge)

        print(f"[DEBUG _assign_uuids] Edge groups:")
        for key, group in sorted(edge_groups.items(), key=lambda x: str(x[0])):
            print(f"  key={key}:")
            for e in group:
                print(f"    {e}")

        # Assign UUIDs to each group
        uuid = 1  # Start from 1 (0 is reserved for flag clear)
        for src_gid_key, group_edges in sorted(edge_groups.items(), key=lambda x: str(x[0])):
            if uuid > 255:
                raise RuntimeError(f"UUID overflow: more than 255 send-recv pairs in function")

            # Determine sender node (from first edge's src)
            first_edge = group_edges[0]
            sender_hw_node = self._get_hw_node(first_edge.src_id)

            # Collect all receiver nodes
            receiver_nodes = set()
            for edge in group_edges:
                recv_node = self._get_hw_node(edge.dst_id)
                # Handle tuple hw node (from split operations)
                if isinstance(recv_node, tuple):
                    for node in recv_node:
                        receiver_nodes.add(node)
                else:
                    receiver_nodes.add(recv_node)

            # Skip if sender == receiver (constant edge or same-node, no real communication)
            if len(receiver_nodes) == 1 and sender_hw_node in receiver_nodes:
                print(f"[DEBUG _assign_uuids] Skipping sender==receiver: {sender_hw_node} -> {receiver_nodes}")
                continue

            # Create pair
            pair = SendRecvPair(uuid, sender_hw_node, receiver_nodes, group_edges)
            self.pairs[uuid] = pair

            # Map each edge to this pair
            print(f"[DEBUG _assign_uuids] Created {pair}, mapping edges:")
            for edge in group_edges:
                self.edge_to_pair[edge] = pair
                print(f"    {edge}")

            uuid += 1

    def _get_hw_node(self, tensor_id) -> NodeID:
        """Get hardware node ID for a tensor ID"""
        gid = tensor_id.graph_node_id
        if isinstance(gid, tuple):
            node = CustomIDToNode()[gid[-1]]
            if isinstance(node, relay.Constant):
                return DevConfig().get_hw_node(gid[1]) # Constant node - inode
            else:
                return DevConfig().get_hw_node(gid[0]) # other nodes in composite -> imce
        else:
          hw_node = DevConfig().get_hw_node(gid)
          return hw_node

    def get_pair(self, edge: TensorEdge, needs_sync=True) -> SendRecvPair:
        """Get the send-recv pair for a given edge

        If edge's dst is a split node, recursively find edges starting from
        that split node and return the pair from those edges.
        Args:
            edge: The tensor edge to look up
            needs_sync: If True, only return pair if it needs sync (using needs_sync method)
        """
        # Direct lookup first
        pair = self.edge_to_pair.get(edge, None)
        if pair is not None:
            return pair

        # If not found, check if dst is a split node
        dst_gid = edge.dst_id.graph_node_id
        if isinstance(dst_gid, tuple):
            dst_gid = dst_gid[-1]

        try:
            dst_node = CustomIDToNode()[dst_gid]
            if hasattr(dst_node, 'op') and hasattr(dst_node.op, 'name') and dst_node.op.name == "split":
                # dst is split node - find edges starting from this split node
                for registered_edge, registered_pair in self.edge_to_pair.items():
                    src_gid = registered_edge.src_id.graph_node_id
                    if isinstance(src_gid, tuple):
                        src_gid = src_gid[-1]
                    if src_gid == dst_gid:
                        print(f"[DEBUG get_pair] Split node detected: {edge} -> found pair via {registered_edge}")
                        return registered_pair
        except (KeyError, Exception):
            pass

        return None

    def get_uuid(self, edge: TensorEdge) -> int:
        """Get UUID for a given edge"""
        pair = self.get_pair(edge)
        return pair.uuid if pair else None

    def needs_sync(self, edge: TensorEdge) -> bool:
        """Check if this edge needs synchronization"""
        return edge in self.edge_to_pair

    def get_participating_nodes(self, edge: TensorEdge) -> List[NodeID]:
        """Get all nodes participating in sync for this edge"""
        pair = self.get_pair(edge)
        return pair.all_nodes if pair else []
