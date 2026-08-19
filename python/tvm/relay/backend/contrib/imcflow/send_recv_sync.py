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
from tvm.contrib.imcflow import bugfix_off_mode
from tvm.relay.op.contrib.imcflow import CustomIDToNode
from tvm.relay.op.contrib.imcflow import pack_bn_minmax_mode, residual_in_region_mode
from tvm.relay.backend.contrib.imcflow.transform_utils import getInnerNodeID
import logging


# IMCFLOW_PACK_BN_MINMAX: dedicated node-flag value for the packed-postop const
# pacing rendezvous (inode SendBlock <-> imce RecvConstBlock). It MUST differ
# from the value-1 data/residual input rendezvous: the packed const pacing runs
# during the CONFIG phase on the inode's (scalar) flag register, and a residual-
# add consumer that begins computing early (e.g. region2 imce_1_2 doing
# STANDBY(inode, 1)) would otherwise alias this config-phase SET_FLAG(1) pulse
# and race ahead of the actual per-tile data SEND -> hard deadlock. 253 is
# reserved (pair UUIDs cap at 251 under packing; 252 is PACK_BN_DATA_SYNC_FLAG;
# 254/255 are the barrier senses; emitted flag values are otherwise <= 4), so it
# can never collide. Lever OFF ->
# the pacing windows are not emitted -> byte-identical.
PACK_CONST_SYNC_FLAG = 253

# IMCFLOW_PACK_BN_MINMAX: dedicated node-flag value for the de-fused standalone-BN
# DATA-input rendezvous (producer conv pre-send STANDBY <-> standalone-BN receiver
# SETFLAG window; see is_defused_standalone_bn_data_edge). It MUST differ from the
# value-1 pipeline rendezvous: the de-fused BN's receiver node (region3 imce_2_1)
# ALSO presents flag 1 during the CONFIG phase for the packed-postop const pacing
# (inode STANDBY(imce_2_1, 1)). A flag-1 data window on that SAME node aliases the
# config-phase flag-1 rendezvous -> on the 2nd kernel launch the inode's
# STANDBY(imce_2_1,1) sees the wrong sense (expected 1, actual 0) and wedges the
# inter-inode func_out-drain barrier (fsim: inode_2_0 EX_STALL_START STANDBY
# expected=0x1 actual=0x0). 252 is below the pack UUID cap (lowered to 251 under
# packing so no pair aliases it) and distinct from 253/254/255, so it never
# collides. Lever OFF -> the window is not emitted -> byte-identical.
PACK_BN_DATA_SYNC_FLAG = 252


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
        # Marker B (sibling-producer STANDBY): {data_producer_edge -> sibling_rhs_hwnode}
        # populated by _build_sibling_standby_map from the raw edge list.
        self._sibling_standby: Dict[TensorEdge, NodeID] = {}
        # Marker B' (flag-3 data-multicast barrier): a data-path producer whose
        # SEND is a MULTICAST (>=2 imce receivers) and whose fused-consumer
        # receiver has >=2 inter-node imce producers (data+rhs). Both the sender
        # STANDBY and the receiver LOAD_LB/RECV window for this producer's data
        # use flag 3 (distinct from flag-1 input/pipeline and flag-2 Fix-B).
        # {data_producer_hwnode.value -> sorted list[receiver hwnode.value]}
        self._flag3_sender_targets: Dict[int, List[int]] = {}
        # set of data-producer hwnode.value that drive a flag-3 window on the
        # receiver side (the LOAD_LB/RECV from this producer gets SETFLAG(3)).
        self._flag3_producers: Set[int] = set()
        # Full raw edge list (incl. const edges excluded from self.pairs). Needed
        # by pack-const per-consumer pacing to discover all inode->imce const
        # endpoints. Only consumed under pack_bn_minmax_mode() -> OFF unaffected.
        self._all_edges_for_pacing = list(edges)
        self._assign_uuids(edges)
        self._build_sibling_standby_map(edges)
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
      # Update -- ACTIVATE the contention filter (option A). The disable
      # (self.pairs={}) came from a 2026-01-23 merge with no stated reason and
      # emits ZERO sync -> deadlock/X-fatal. handcraft needs SETFLAG/STANDBY
      # rendezvous; this restores the pair set so the imce/inode codeblocks emit
      # them. (Contention-only may under-cover 1:1 pipeline SENDs; measure vs
      # handcraft's 22 SETFLAG / 16 STANDBY and widen if needed.)
      #
      # BUGFIX knob: knob=on (bugfix_off_mode()==False) reproduces a8af, whose
      # filter_contention=True path emitted ZERO pairs (self.pairs={}) -> all
      # SEND/RECV/LOAD_LB are BARE. knob=off keeps the 934 activated filter.
      if bugfix_off_mode():
        self.pairs = filtered_pairs
        self.edge_to_pair = filtered_edge_to_pair
      else:
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
        # 255 is reserved for the all-inode barrier (SyncAllINodes). Under the
        # BN/minmax packing lever the barrier is sense-reversing over {254,255}
        # (both reserved), 253 is reserved for the packed-postop const pacing
        # rendezvous (PACK_CONST_SYNC_FLAG), and 252 is reserved for the de-fused
        # standalone-BN data rendezvous (PACK_BN_DATA_SYNC_FLAG), so pair UUIDs
        # must stay <= 251. residual_in_region keeps its prior <= 253 cap (no
        # pack-const pacing).
        if pack_bn_minmax_mode():
            _uuid_max = 251
        elif residual_in_region_mode():
            _uuid_max = 253
        else:
            _uuid_max = 255
        for src_gid_key, group_edges in sorted(edge_groups.items(), key=lambda x: str(x[0])):
            if uuid > _uuid_max:
                raise RuntimeError(f"UUID overflow: more than {_uuid_max} send-recv pairs in function")

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

    def _build_sibling_standby_map(self, edges: List[TensorEdge]):
        """Marker B (RTL-derived): sibling-producer STANDBY for 2-producer fused
        consumers.

        A fused/composite consumer whose graph_node_id is a tuple (outer, inner)
        can be fed by >=2 *inter-node imce* producers -- e.g. region2 imce_3_2
        (composite 63) receives its conv `data`/lhs from imce_2_1 and its fused
        add `rhs` from imce_2_2. The `data`-path producer's SEND must rendezvous
        with the *rhs*-path sibling producer (so both operands land before the
        consumer's fused STEP/ADD), NOT with the consumer itself. handcraft
        imce_2_1 emits exactly `STANDBY(imce_2_2)` and no STANDBY on imce_3_2.

        This map records, for each such data-path producer edge, the sibling
        rhs-path producer hw node. get_pre_send_sync() then emits STANDBY on the
        sibling and SUPPRESSES the receiver STANDBY for that edge.

        region1 exclusion: region1 has no fused node fed by two inter-node imce
        producers (its fused conv+op takes conv-data from one imce + a local
        const), so no consumer here has >=2 inter-node imce producers -> the map
        stays empty and get_pre_send_sync() is unchanged there.
        """
        # Group inter-node edges by the consumer's composite OUTER graph id.
        # Only tuple dst (composite/fused) consumers are candidates.
        by_consumer: Dict = {}  # outer_id -> list[(edge, dst_tensor_type, src_hwnode)]
        for edge in edges:
            dst_gid = edge.dst_id.graph_node_id
            if not isinstance(dst_gid, tuple):
                continue  # plain consumer -> not a fused node
            # Skip constant producers (no real inter-node imce SEND).
            try:
                from tvm.relay.dataflow_pattern import is_constant
                if is_constant().match(CustomIDToNode()[edge.src_id.graph_node_id]):
                    continue
            except (KeyError, Exception):
                pass
            src_hw = self._get_hw_node(edge.src_id)
            dst_hw = self._get_hw_node(edge.dst_id)
            # only inter-node imce->imce producer edges qualify
            if isinstance(src_hw, tuple) or isinstance(dst_hw, tuple):
                continue
            if not (src_hw.is_imce() and dst_hw.is_imce()):
                continue
            if src_hw == dst_hw:
                continue  # local edge, same hw node
            outer_id = dst_gid[0]
            by_consumer.setdefault(outer_id, []).append(
                (edge, edge.dst_id.tensor_type, src_hw))

        for outer_id, entries in by_consumer.items():
            # distinct inter-node imce producer hw nodes feeding this consumer
            producer_hwnodes = {src_hw for (_, _, src_hw) in entries}
            if len(producer_hwnodes) < 2:
                continue  # single-producer fused node -> no sibling rendezvous
            # data/lhs path edge(s) and the rhs-path producer hw node
            rhs_entries = [e for e in entries if e[1] == "rhs"]
            data_entries = [e for e in entries if e[1] == "data"]
            if not rhs_entries or not data_entries:
                continue
            # sibling = rhs-path producer hw node (single expected)
            sibling_hw = rhs_entries[0][2]
            for (data_edge, _, _) in data_entries:
                self._sibling_standby[data_edge] = sibling_hw
                print(f"[Marker B] data-producer edge {data_edge} -> "
                      f"sibling STANDBY({sibling_hw.value}) (consumer composite {outer_id})")

            # Marker B' (flag-3 data-multicast barrier). The data-path producer
            # (e.g. region3 imce_0_3) whose quantized odata is MULTICAST to two
            # imce receivers -- the fused consumer (imce_1_2) AND the sibling conv
            # (imce_1_3) -- must gate its SEND on BOTH receivers at flag 3, and
            # both receivers must open a matching SETFLAG(3) window around the
            # LOAD_LB/RECV of this producer's data. handcraft imce_0_3:
            #   STANDBY(imce_1_2,3); STANDBY(imce_1_3,3); SEND...
            # This differs from Marker B (which targets only the rhs sibling at
            # flag 1) because the producer's SEND is itself a multicast whose
            # receiver set spans both branch convs.
            #
            # DISCRIMINATOR (verified against passing handcraft): flag 3 is used
            # by handcraft in EXACTLY ONE place -- region3 imce_0_3. The naive
            # "multicast into a 2-producer fused consumer" test also matches
            # region2 imce_2_1 (consumer composite 63) and region3 imce_2_2
            # (composite 99), both of which handcraft leaves WITHOUT flag 3.
            # The true separator (H1) is the OP feeding the producer: region3
            # imce_0_3 is a STANDALONE min_max_quantize fed by a STANDALONE
            # residual `add` (its feeder edge's src gid is a PLAIN int), whereas
            # imce_2_1 / imce_2_2 are fed by a fused conv/BN COMPOSITE (feeder
            # src gid is a TUPLE). We additionally require the exact-diamond (H3)
            # receiver set == {fused-consumer, rhs-sibling} to harden against
            # future graphs (changes nothing for these three cases). region1 has
            # no imce->imce multicast at all, so it never reaches here.
            for (data_edge, _, data_src_hw) in data_entries:
                dpair = self.edge_to_pair.get(data_edge, None)
                if dpair is None:
                    continue
                imce_recvs = sorted(
                    {r for r in dpair.receiver_nodes if r.is_imce()},
                    key=lambda x: x.value)
                if len(imce_recvs) < 2:
                    continue  # not a multicast -> Marker B (flag 1) handles it

                # H1: producer must be fed by a STANDALONE node (plain-int src
                # gid), NOT a fused conv/BN composite (tuple src gid). The
                # producer's own split gid is a plain int in all cases, so test
                # the gid of the node FEEDING the producer.
                producer_gid = data_edge.src_id.graph_node_id
                if isinstance(producer_gid, tuple):
                    producer_gid = producer_gid[-1]
                feeder_is_composite = any(
                    ((e.dst_id.graph_node_id[-1]
                      if isinstance(e.dst_id.graph_node_id, tuple)
                      else e.dst_id.graph_node_id) == producer_gid)
                    and isinstance(e.src_id.graph_node_id, tuple)
                    for e in edges
                )
                if feeder_is_composite:
                    continue  # fused conv/BN producer -> not the region3 quantize

                # H3 (hardening): the multicast receiver set must be exactly
                # {fused-consumer, rhs-sibling}. Excludes region3 composite 99
                # (producer imce_2_2 fans out to a THIRD node).
                consumer_hw = self._get_hw_node(data_edge.dst_id)
                if isinstance(consumer_hw, tuple):
                    consumer_hw = consumer_hw[0]
                expected = sorted({sibling_hw, consumer_hw}, key=lambda x: x.value)
                if imce_recvs != expected:
                    continue

                self._flag3_sender_targets[data_src_hw.value] = [
                    r.value for r in imce_recvs]
                self._flag3_producers.add(data_src_hw.value)
                print(f"[Marker B'] flag-3 data multicast: producer "
                      f"imce({data_src_hw.value}) STANDBY "
                      f"{[r.value for r in imce_recvs]} @flag3 "
                      f"(consumer composite {outer_id})")

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

    # ------------------------------------------------------------------
    # handcraft-matched sync classification / emission
    #
    # Sync uses literal flag values (1 for set, 0 for clear) -- NOT pair.uuid.
    # imce<->imce pipeline sync is ASYMMETRIC:
    #   * receiver side wraps its RECV/LOAD_LB burst in one window:
    #       SETFLAG(1); RECV/LOAD_LB x N; SETFLAG(0)   (no STANDBY)
    #   * sender side emits ONE pre-send STANDBY(receiver, 1) just before the
    #     SEND loop (not per-SEND).
    # inode->imce *data input* (from a graph input Var) receiver side is:
    #       SETFLAG(1); STANDBY(inode, 1); SETFLAG(0); RECV
    # inode->imce weight/const and imce->inode output are bare (no sync).
    # ------------------------------------------------------------------

    def is_sender_inode(self, edge: TensorEdge) -> bool:
        """True if the sender hw node of this edge's pair is an inode."""
        pair = self.get_pair(edge)
        if pair is None:
            return False
        return pair.sender_node.is_inode()

    def is_inode_data_input_recv(self, edge: TensorEdge) -> bool:
        """True if this RECV is a data input coming from an inode into the main
        pipeline (as opposed to a fused/composite receiver).

        In handcraft the multicast input edge (inode_0_0 -> imce_0_1, imce_0_2)
        is received bare by imce_0_1 (dst graph_node_id is a composite tuple)
        but with a STANDBY(inode,1) window by imce_0_2 (dst graph_node_id is a
        plain int = main pipeline entry). We reproduce that by requiring the
        sender to be an inode AND the *this receiver's* dst graph_node_id to be
        a plain (non-tuple) id.
        """
        pair = self.get_pair(edge)
        if pair is None or not pair.sender_node.is_inode():
            return False
        dst_gid = edge.dst_id.graph_node_id
        return not isinstance(dst_gid, tuple)

    def is_residual_data_input_recv(self, edge: TensorEdge) -> bool:
        """IMCFLOW_RESIDUAL_IN_REGION: True if this RECV is one of the TWO data
        operands of an in-region residual ADD (a fused VecOpBlock: BN + relu +
        multiply + add) that lives in the SAME region as its producers.

        Unlike is_inode_data_input_recv (which requires an inode sender AND a
        plain-int dst = the OLD cross-region 2-inode add), the in-region residual
        add is a MIXED pair:
          - main path : imce_3_2 (an IMCE conv) -> data  (fifo 2)
          - skip path : inode_0_0 (an INODE)    -> data  (fifo 3)
        Both operands land as composite `data` edges whose dst graph_node_id is a
        TUPLE (the vecops composite), so is_inode_data_input_recv drops both and
        the OLD merged-window machinery never fires -> both RECVs are bare ->
        producer/consumer desync -> region1 tiled-launch deadlock.

        Detection (lever-gated so OFF stays byte-identical):
          * residual_in_region_mode() ON
          * edge is a paired `data` RECV whose dst is a composite tuple
          * receiver hw node is an imce
          * that receiver has >= 2 DISTINCT data-input producers (any mix of imce
            and inode), i.e. it is a converging residual add.
        The >=2-producer requirement structurally excludes every ordinary
        single-input pipeline `data` RECV (which has exactly one producer), so
        this only ever matches the residual add.
        """
        if not residual_in_region_mode():
            return False
        pair = self.get_pair(edge)
        if pair is None:
            return False
        if edge.dst_id.tensor_type != "data":
            return False
        dst_gid = edge.dst_id.graph_node_id
        if not isinstance(dst_gid, tuple):
            return False  # plain-int dst is handled by is_inode_data_input_recv
        recv_hw = self._get_hw_node(edge.dst_id)
        if isinstance(recv_hw, tuple):
            recv_hw = recv_hw[0]
        if recv_hw is None or not recv_hw.is_imce():
            return False
        return len(self._residual_data_producers(recv_hw)) >= 2

    def _residual_data_producers(self, recv_hw: NodeID) -> List[NodeID]:
        """Distinct hw nodes that SEND a composite `data` operand into recv_hw
        (imce). Includes BOTH imce and inode senders -- the residual add's mixed
        producer pair. Ascending-value ordered, deduplicated. Only meaningful
        under residual_in_region_mode(); callers gate on the lever.
        """
        producers = []
        seen = set()
        for p in self.pairs.values():
            if recv_hw not in p.receiver_nodes:
                continue
            for e in p.edges:
                # only the edges whose OWN dst is recv_hw with a composite data
                rn = self._get_hw_node(e.dst_id)
                if isinstance(rn, tuple):
                    rn = rn[0]
                if rn != recv_hw:
                    continue
                if e.dst_id.tensor_type != "data":
                    continue
                if not isinstance(e.dst_id.graph_node_id, tuple):
                    continue
                if p.sender_node.value not in seen:
                    seen.add(p.sender_node.value)
                    producers.append(p.sender_node)
                break
        producers.sort(key=lambda x: x.value)
        return producers

    def is_residual_multicast_conv_input_recv(self, edge: TensorEdge) -> bool:
        """IMCFLOW_RESIDUAL_IN_REGION: True if this RECV is the PLAIN-int-dst
        consumer (a standalone min_max_quantize / conv head, e.g. imce_0_1) of a
        model-input MULTICAST whose SAME source TensorID ALSO fans out to an
        in-region residual ADD composite (tuple dst, e.g. imce_0_2's skip input)
        in the same func.

        Under residual_in_region the model input TensorID(-11,odata) is one
        physical INODE multicast reaching BOTH:
          * this plain-int-dst consumer   -- is_inode_data_input_recv True
          * a composite tuple-dst residual add -- is_residual_data_input_recv True
        The residual-add consumer keeps its per-word SETFLAG(1);STANDBY(inode,1);
        SETFLAG(0) window and paces the ONE inode SEND per word. To exactly mirror
        the proven cross-region baseline (IMCFLOW_RESIDUAL_IN_REGION OFF), where
        the inode STANDBYs ONLY the windowed residual consumer and THIS consumer
        is a BARE fanout receiver (drains via fifo backpressure), we:
          * (inode side) drop this consumer from the pre-send rendezvous target
            set, leaving the inode to STANDBY only the residual-add consumer.
          * (imce side)  emit this consumer's data RECV window BARE (None, None).

        Detection (lever-gated so OFF stays byte-identical):
          * residual_in_region_mode() ON
          * `edge` is an inode->imce data input with a PLAIN-int dst
            (is_inode_data_input_recv True)
          * SOME OTHER edge sharing THIS edge's source TensorID is an in-region
            residual-add operand (is_residual_data_input_recv True).
        The residual-add co-fanout requirement structurally excludes every
        ordinary inode data input (no residual sibling), so this only ever
        matches the residual-in-region model-input multicast head.
        """
        if not residual_in_region_mode():
            return False
        if not self.is_inode_data_input_recv(edge):
            return False
        src_id = edge.src_id
        # A sibling edge shares this multicast's source TensorID and lands on an
        # in-region residual add (composite tuple dst). Scan all paired edges.
        for p in self.pairs.values():
            for e in p.edges:
                if e is edge:
                    continue
                if e.src_id is not src_id:
                    continue
                if self.is_residual_data_input_recv(e):
                    return True
        return False

    def is_packed_postop_const_edge(self, edge: TensorEdge) -> bool:
        """IMCFLOW_PACK_BN_MINMAX capacity-deadlock guard (BUGFIX-off RTL).

        When BN + min_max_quantize are folded into the qconv composite (IMCE
        packing), a single packed conv (e.g. region2 qconv_bn_multiply_add,
        node 72) grows extra inode->imce CONSTANT operands -- the BN
        fused_scale/fused_bias and the multiply/add `scale` operands (dst
        tensor_type "rhs"). These const SENDs are BARE (const edges are
        excluded from the pair manager, so they carry no rendezvous) and are
        pushed into the NoC eagerly. On a packed conv the burst is large enough
        (config + fused_scale x2 + fused_bias x2 + mult x2 + add x2 = 9 words,
        plus the sibling minmax's min/max on the same inode->column path) to
        overflow the inode send FIFO before the (transitively pipeline-blocked)
        receiving imce drains it. The inode then wedges on PUSH_STALL and can
        never reach either its all-inode 255 barrier or its pipeline-root
        data-input SEND -> the observed region2 hard deadlock (imce_0_2 starved
        forever on its input STANDBY).

        Fix: pace these extra packed-postop consts with the SAME per-word
        inode<->imce flag-1 rendezvous the data input already uses, so the inode
        cannot outrun the imce's const drain. Both the inode SEND (pre-send
        STANDBY/SETFLAG in inode_codeblock.SendBlock) and the imce RECV
        (SETFLAG window in RecvConstBlock) key on THIS predicate so they stay in
        lockstep.

        DISCRIMINATOR / OFF-invariance: returns True only when
        pack_bn_minmax_mode() is on AND the edge is an inode->imce const whose
        dst is a fused/composite (tuple gid) post-op operand of a packed conv
        (tensor_type in {fused_scale, fused_bias, rhs, scale}). With the lever
        OFF these folded post-ops do not exist (BN/minmax are separate IMCEs),
        so the mode short-circuits to False for every edge -> the const-send /
        const-recv path is byte-identical to the OFF build. Config and weight
        (tensor_type "config"/"weight") are excluded: they exist in the OFF
        build too and must stay bare to preserve byte-identity there.
        """
        if not pack_bn_minmax_mode():
            return False
        # dst must be a fused/composite post-op operand of a packed conv
        dst_gid = edge.dst_id.graph_node_id
        if not isinstance(dst_gid, tuple):
            return False
        if getattr(edge.dst_id, "tensor_type", None) not in (
                "fused_scale", "fused_bias", "rhs", "scale", "min", "max"):
            return False
        # src must be a constant fed by an inode; dst must resolve to an imce.
        # (const edges are unpaired -- derive hw nodes directly.)
        try:
            src_hw = self._get_hw_node(edge.src_id)
            dst_hw = self._get_hw_node(edge.dst_id)
        except Exception:
            return False
        if isinstance(dst_hw, tuple):
            dst_hw = dst_hw[0]
        if src_hw is None or dst_hw is None:
            return False
        if isinstance(src_hw, tuple):
            return False
        return src_hw.is_inode() and dst_hw.is_imce()

    def packed_postop_const_endpoints(self, edge: TensorEdge):
        """(inode_hw, imce_hw) for a packed-postop const edge, or None.

        Used by the inode SendBlock (pre-send STANDBY on the imce) and the imce
        RecvConstBlock (SETFLAG window + STANDBY on the inode) to build the
        per-word flag rendezvous WITHOUT a pair (const edges are unpaired).
        """
        if not self.is_packed_postop_const_edge(edge):
            return None
        src_hw = self._get_hw_node(edge.src_id)
        dst_hw = self._get_hw_node(edge.dst_id)
        if isinstance(dst_hw, tuple):
            dst_hw = dst_hw[0]
        if isinstance(src_hw, tuple):
            return None
        return (src_hw, dst_hw)

    def pack_const_sync_flag(self) -> int:
        """Node-flag value presented by the inode during packed-postop const
        pacing (SendBlock SET_FLAG / RecvConstBlock STANDBY-on-inode).

        Returns 1 by default (byte-identical to the pack-const introduction),
        and PACK_CONST_SYNC_FLAG (253) ONLY when this function contains a
        receiver fed by >=2 inode data inputs (the 2-inode residual-add pattern,
        e.g. region2 imce_1_2). That consumer does STANDBY(inode, 1) in the DATA
        phase and would otherwise alias the inode's CONFIG-phase pack-const
        SET_FLAG(1) pulse -> it races ahead of the real per-tile SEND and the
        whole region wedges (region2 TILE0/1 hard deadlock). Regions with no such
        consumer (region1) keep flag 1 -> byte-identical to the passing form.

        Cached (function-scoped): the pair set is fixed after construction.
        """
        if not pack_bn_minmax_mode():
            return 1
        cached = getattr(self, "_pack_const_flag_cache", None)
        if cached is not None:
            return cached
        # Count inode data-input senders per receiver hw node across all pairs.
        recv_inode_senders: Dict[int, Set[int]] = {}
        for pair in self.pairs.values():
            if not pair.sender_node.is_inode():
                continue
            for edge in pair.edges:
                if not self.is_inode_data_input_recv(edge):
                    continue
                rnode = self._get_hw_node(edge.dst_id)
                if isinstance(rnode, tuple):
                    rnode = rnode[0]
                if rnode is None or not rnode.is_imce():
                    continue
                recv_inode_senders.setdefault(rnode.value, set()).add(
                    pair.sender_node.value)
        has_two_inode_add = any(len(s) >= 2 for s in recv_inode_senders.values())
        flag = PACK_CONST_SYNC_FLAG if has_two_inode_add else 1
        self._pack_const_flag_cache = flag
        return flag

    def _pack_const_multi_consumer_inodes(self) -> Dict[int, List[int]]:
        """{inode_value: sorted[imce_value,...]} for inodes that pace >=2 DISTINCT
        imce consumers via packed-postop const pacing.

        The pack-const go-pulse (inode SET_FLAG(pack_const_sync_flag)) is a SHARED
        scalar flag on the inode; every consumer of that inode does
        STANDBY(inode, flag). When ONE inode paces two consumers (e.g. region3
        inode_2_0 -> the de-fused standalone BN imce_2_1 AND the fused conv
        imce_2_2), the pulse meant for consumer A is also visible to consumer B,
        which steals it -> A's inode-side STANDBY(A, 0) never clears -> wedge.
        Only arises once a de-fused standalone BN adds a SECOND pack-const
        consumer to an inode. Cached (pair set fixed post-construction).
        """
        cached = getattr(self, "_pack_const_multi_cache", None)
        if cached is not None:
            return cached
        inode_consumers: Dict[int, Set[int]] = {}
        if pack_bn_minmax_mode():
            for e in self._iter_all_edges():
                eps = self.packed_postop_const_endpoints(e)
                if eps is None:
                    continue
                inode_hw, imce_hw = eps
                inode_consumers.setdefault(inode_hw.value, set()).add(imce_hw.value)
        result = {k: sorted(v) for k, v in inode_consumers.items() if len(v) >= 2}
        self._pack_const_multi_cache = result
        return result

    def pack_const_go_flag(self, inode_hw, imce_hw) -> int:
        """Per-consumer go-pulse value for packed-postop const pacing.

        Default = pack_const_sync_flag() (1 or 253), byte-identical to the
        single-consumer form. When `inode_hw` paces >=2 distinct consumers, each
        consumer gets a DISTINCT value drawn from a safe band (>=4 below the
        reserved 251..255 and above the <=4 emitted data/barrier values), so a
        SET_FLAG for one consumer cannot be sampled by another. Both the inode
        SendBlock and the imce RecvConstBlock call this with the SAME
        (inode, imce) pair, so they agree on the value.
        """
        base = self.pack_const_sync_flag()
        multi = self._pack_const_multi_consumer_inodes()
        consumers = multi.get(inode_hw.value)
        if not consumers:
            return base
        # Distinct value per consumer within a safe band [200, 250]. The band
        # avoids emitted data/barrier values (<=4), the pair-UUID cap (251), and
        # 252/253/254/255. Index by position in the sorted consumer list so the
        # inode and imce sides derive the same value deterministically.
        idx = consumers.index(imce_hw.value)
        return 200 + idx

    def _iter_all_edges(self):
        """Yield every TensorEdge seen during construction (paired + const).

        Const edges are unpaired (excluded from self.pairs), so pack-const
        endpoints must be discovered from the full edge list stashed at build.
        """
        edges = getattr(self, "_all_edges_for_pacing", None)
        if edges is not None:
            for e in edges:
                yield e
            return
        seen = set()
        for pair in self.pairs.values():
            for e in pair.edges:
                key = id(e)
                if key not in seen:
                    seen.add(key)
                    yield e

    def _receiver_is_output_node(self, recv_hw_node) -> bool:
        """True iff every inter-node edge sent BY recv_hw_node goes to an inode
        (the receiver is a pipeline-terminal / output node whose result leaves to
        an inode, e.g. imce_3_1 -> inode_3_0). If it forwards to another imce it
        is a mid-chain node (e.g. imce_2_1 -> imce_3_1). Terminal nodes with a
        fused sender take a BARE edge; mid-chain nodes keep the flag=1 sync.
        """
        sends = [p for p in self.pairs.values() if p.sender_node == recv_hw_node]
        if not sends:
            return True  # no onward NoC send -> terminal
        return all(r.is_inode() for p in sends for r in p.receiver_nodes)

    def _node_is_standalone_bn(self, inner_gid) -> bool:
        """True iff the relay node at custom-id `inner_gid` is a STANDALONE
        batch_norm (the body of a de-fused BN's RecvSendWrapper, annotation
        "bn_standalone" in imce_operation_handlers.BatchNormHandler), i.e. a
        batch_norm op / imcflow.*batch_norm composite. Used ONLY under
        pack_bn_minmax_mode() to detect the 2-pass de-fused BN receiver."""
        try:
            node = CustomIDToNode()[inner_gid]
        except Exception:
            return False
        if not isinstance(node, relay.Call):
            return False
        opn = node.op
        # plain op form (nn.batch_norm / imcflow.fused_batch_norm)
        if isinstance(opn, tvm.ir.Op) and opn.name in (
                "nn.batch_norm", "imcflow.fused_batch_norm"):
            return True
        # composite Function form: Composite name mentions batch_norm
        if isinstance(opn, relay.Function) and opn.attrs and "Composite" in opn.attrs:
            cname = str(opn.attrs["Composite"])
            return "batch_norm" in cname
        return False

    def is_defused_standalone_bn_data_edge(self, edge: TensorEdge) -> bool:
        """IMCFLOW_PACK_BN_MINMAX 2-pass de-fuse launch-boundary sync guard.

        The size-aware de-fuse (transform.stamp_pack_atomic_keys +
        pack_bn_minmax_exclude_set) drops an overflowing spatial conv's BN into a
        STANDALONE BN IMCE (region3 imce_2_1, gid (111,105)), fed by the now
        BN-less bare conv (imce_1_1, gid 110). That conv->BN `data` edge is
        classified BARE on both sides by _is_conv_data_into_terminal_or_fanout /
        _is_composite_boundary_edge (imce_2_1 is a pipeline terminal). Bare/bare
        is fine for a genuinely FUSED terminal wrapper, but the de-fused
        standalone BN's flat num_blocks*64 data-RECV loop has no launch-boundary
        rendezvous, so on the 2nd kernel launch the producer conv (no pre-send
        STANDBY) and the starved standalone BN cannot re-sync -> the inter-inode
        func_out-drain barrier wedges (fsim-confirmed region3 launch-2 deadlock:
        imce_1_1 pc4 RECV_CFG <- inode_1_0 barrier <- inode_0_0 <- inode_2_0
        func_out1 drain <- imce_2_1 data-starved <- imce_1_1).

        Fires ONLY when ALL hold (so OFF and every non-de-fused edge -- incl. the
        fan-out standalone BN imce_2_2 which is mid-chain and re-synced downstream
        -- are untouched, byte-identical):
          * pack_bn_minmax_mode() is ON,
          * the exclusion set is non-empty (a de-fuse actually happened),
          * dst is a composite (tuple gid) `data` operand,
          * the receiver hw-node's inner relay node is a STANDALONE batch_norm,
          * the receiver is a pipeline TERMINAL (_receiver_is_output_node),
          * the sender is an imce (inter-node imce->imce).
        """
        if not pack_bn_minmax_mode():
            return False
        from tvm.relay.op.contrib.imcflow import pack_bn_minmax_exclude_set
        if not pack_bn_minmax_exclude_set():
            return False
        dst_gid = edge.dst_id.graph_node_id
        if not isinstance(dst_gid, tuple):
            return False
        if getattr(edge.dst_id, "tensor_type", None) != "data":
            return False
        if not self._node_is_standalone_bn(dst_gid[-1]):
            return False
        pair = self.get_pair(edge)
        if pair is None:
            return False
        if pair.sender_node.is_inode():
            return False
        recv_hw = self._get_hw_node(edge.dst_id)
        if isinstance(recv_hw, tuple):
            recv_hw = recv_hw[0]
        if recv_hw is None or not recv_hw.is_imce():
            return False
        return self._receiver_is_output_node(recv_hw)

    def _is_composite_boundary_edge(self, edge: TensorEdge) -> bool:
        """Bare (no pre-send STANDBY, no RECV window) iff the SENDER is a fused/
        composite op (src graph_node_id is a tuple) AND the RECEIVER is an output
        node (sends only to inode). Verified against all three cases:
          R2 imce_3_2->imce_3_1: src tuple, imce_3_1->inode -> BARE.
          R1 imce_2_1->imce_3_1: src PLAIN (38) -> synced (fails src-tuple).
          R1 imce_1_1->imce_2_1: src tuple, but imce_2_1->imce_3_1 (imce) so
                                 receiver is mid-chain -> synced (fails output).
        The RTL flag is one scalar reg/node; a flag=1 rendezvous on this
        fused-sender->terminal edge aliases with the sender's same-iteration
        flag-1 uses and wedges region2 TILE0. Marker-A NoC-rhs/sibling edges are
        handled earlier and not reached here.
        """
        try:
            if not isinstance(edge.src_id.graph_node_id, tuple):
                return False
        except Exception:
            return False
        pair = self.get_pair(edge)
        if pair is None:
            return False
        return all(self._receiver_is_output_node(r) for r in pair.receiver_nodes)

    def _receiver_is_fanout_producer(self, recv_hw_node) -> bool:
        """True iff `recv_hw_node` forwards its result onward to >=2 DISTINCT imce
        hw nodes (a NoC fan-out / multicast producer). In region2 imce_2_1 sends
        its BN+minmax output to BOTH imce_2_2 (conv data) and imce_3_2 (fused add
        data); imce_1_3 (add) sends to imce_1_2 and imce_0_3. A single-output
        mid-chain node (region1 imce_1_1 -> imce_2_1 only) is NOT a fan-out
        producer.
        """
        imce_targets = set()
        for p in self.pairs.values():
            if p.sender_node != recv_hw_node:
                continue
            for r in p.receiver_nodes:
                if r.is_imce():
                    imce_targets.add(r)
        return len(imce_targets) >= 2

    def _is_conv_data_into_terminal_or_fanout(self, edge: TensorEdge) -> bool:
        """Fix C (region2-only, handcraft-derived): a plain conv-odata SEND feeding
        the composite `data` input of a fused wrapper is BARE (no pre-send STANDBY,
        no receiver SETFLAG window) iff that wrapper node is either a pipeline
        TERMINAL (sends only to inode) OR a NoC FAN-OUT producer (>=2 distinct imce
        receivers). It stays SYNCED for a single-output mid-chain receiver.

        This is the exact split handcraft applies to the identically-shaped
        `(conv, odata) -> ((composite, inner), data)` edge:
          R2 imce_0_2 -> imce_0_1  : imce_0_1 -> inode  (terminal)  -> BARE   (hc)
          R2 imce_1_1 -> imce_2_1  : imce_2_1 -> imce_2_2,imce_3_2 (fanout) -> BARE (hc)
          R1 imce_1_2 -> imce_1_1  : imce_1_1 -> imce_2_1  (single)  -> SYNCED (hc)
        The node-level topology of all three is otherwise isomorphic, so the
        receiver's onward fan-out is the only discriminator surfaced here.

        region1 has NO conv->composite-data edge whose receiver is terminal or
        fan-out (its only such edge, imce_1_2->imce_1_1, is single-output), so this
        predicate is False for every region1 edge -> region1 is unchanged.
        """
        # Fix C+ (region3 imce_1_2): the sender may be a PLAIN conv odata OR a
        # FUSED conv+add odata (src tuple). handcraft bares BOTH when the dst is a
        # fused wrapper's `data` and the receiver fans out. The dst-tuple +
        # tensor_type=="data" pair below is model-unique to imce_1_2->imce_2_2, so
        # admitting fused senders here flips exactly that one edge (region1
        # imce_1_1->imce_2_1 has a PLAIN-int dst; region3 imce_3_3->imce_3_2 has
        # tensor_type=="lhs" -- both still fail the checks below and stay synced).
        # Dropping imce_1_2's per-iteration STANDBY(12,1) pre-send removes a
        # pipeline-serializing rendezvous -> region3 throughput recovery.
        # receiver tensor must be the composite `data` input of a fused wrapper.
        dst_gid = edge.dst_id.graph_node_id
        if not isinstance(dst_gid, tuple):
            return False
        if getattr(edge.dst_id, "tensor_type", None) != "data":
            return False
        pair = self.get_pair(edge)
        if pair is None:
            return False
        # inter-node imce->imce only (inode senders handled elsewhere).
        if pair.sender_node.is_inode():
            return False
        # every receiver must be terminal OR fan-out (single-output mid-chain stays synced)
        return all(
            self._receiver_is_output_node(r) or self._receiver_is_fanout_producer(r)
            for r in pair.receiver_nodes
        )

    def _node_has_input_rendezvous(self, hw_node: NodeID) -> bool:
        """Fix B predicate part 1: True iff `hw_node` is the receiver of an
        inode *data-input* rendezvous in this same function -- i.e. it emits a
        flag=1 input window (SETFLAG(1); STANDBY(inode,1); SETFLAG(0)) in the
        same loop iteration. In region2 only imce_1_3(node8) qualifies: it RECVs
        its add lhs/rhs from inode_0_0(0) and inode_1_0(5).

        Implementation: scan pairs for one whose sender is an inode and whose
        receiver set includes hw_node, and where that edge is classified as a
        main-pipeline data input (is_inode_data_input_recv). Region1 has no
        imce->imce multicast producer, so its convs never reach Fix B anyway,
        but this predicate is also False for region1 pipeline entries fed by an
        inode data input that are NOT multicast producers.
        """
        for pair in self.pairs.values():
            if not pair.sender_node.is_inode():
                continue
            if hw_node not in pair.receiver_nodes:
                continue
            for e in pair.edges:
                if self.is_inode_data_input_recv(e):
                    return True
        return False

    def _is_fixb_multicast_edge(self, edge: TensorEdge) -> bool:
        """Fix B target discriminator.

        An edge is a Fix B (flag=2) multicast-barrier edge iff:
          (a) its pair is a MULTICAST (>=2 receiver hw nodes), AND
          (b) the pair's SENDER hw node also performs an inode-fed input
              rendezvous in the same iteration (_node_has_input_rendezvous).

        Only region2 imce_1_3 (sends odata to imce_1_2=7 and imce_0_3=3 while
        RECVing its add operands from inode_0_0/inode_1_0) satisfies both. The
        RTL flag reg is one scalar per node (imce_ctrl.sv), so the output
        multicast rendezvous MUST use a distinct flag (2) from the input
        rendezvous (1) or they alias and deadlock. region1 has no imce->imce
        multicast producer -> (a) is never true there -> region1 unaffected.
        """
        pair = self.get_pair(edge)
        if pair is None:
            return False
        # (a) multicast: >=2 distinct receiver hw nodes, all imce
        imce_receivers = [r for r in pair.receiver_nodes if r.is_imce()]
        if len(imce_receivers) < 2:
            return False
        # (b) sender also has an inode-fed input rendezvous this iteration
        return self._node_has_input_rendezvous(pair.sender_node)

    def get_output_flag(self, edge: TensorEdge) -> int:
        """Flag slot for this edge's SEND/RECV rendezvous: 2 for a Fix B
        multicast-barrier edge, 1 otherwise."""
        return 2 if self._is_fixb_multicast_edge(edge) else 1

    def get_recv_window_sync(self, edge: TensorEdge):
        """Return (pre_lines, post_lines) to wrap a receiver's RECV/LOAD_LB burst.

        pre_lines are emitted before the burst, post_lines after. Returns
        (None, None) if this edge needs no receiver-side window (bare).
        """
        pair = self.get_pair(edge)
        if pair is None:
            return None, None

        # Marker B' (flag-3 data-multicast barrier) receiver side: the LOAD_LB /
        # RECV of this producer's `data` opens a SETFLAG(3) window that the
        # producer's STANDBY(receiver, 3) waits on. Both branch-B receivers
        # (fused consumer imce_1_2 and sibling conv imce_1_3) wrap their data
        # LOAD_LB from imce_0_3 in SETFLAG(3)...SETFLAG(0). Takes precedence over
        # the bare (Marker A has_noc_rhs) / flag-1 defaults below. Only the
        # data-path edge from a flag-3 producer qualifies (the rhs edge from the
        # sibling keeps its own flag-1 window).
        if (pair.sender_node.is_imce()
                and pair.sender_node.value in self._flag3_producers
                and edge.dst_id.tensor_type == "data"):
            return (["__builtin_IMCE_SETFLAG(3);"],
                    ["__builtin_IMCE_SETFLAG(0);"])

        # Fix B receiver window: this receiver is fed by a Fix B multicast
        # producer (imce_1_3). It waits on the sender with flag=2:
        #   SETFLAG(2); STANDBY(sender, 2); SETFLAG(0)
        # (handcraft imce_0_3 / imce_1_2 receiving imce_1_3's odata). Emitted as
        # pre_lines that fully close before the RECV.
        if (not pair.sender_node.is_inode()
                and self._is_fixb_multicast_edge(edge)):
            pre = [
                "__builtin_IMCE_SETFLAG(2);",
                f"__builtin_IMCE_STANDBY({pair.sender_node.value}, 2);",
                "__builtin_IMCE_SETFLAG(0);",
            ]
            return pre, []

        # Fused-sender -> output-node edge is bare on the receiver side too
        # (handcraft imce_3_1 receiving from fused imce_3_2 has no SETFLAG window).
        if (not pair.sender_node.is_inode()
                and self._is_composite_boundary_edge(edge)):
            return None, None

        # IMCFLOW_PACK_BN_MINMAX de-fuse fix (takes precedence over the Fix C bare
        # return below): re-arm the receiver-side SETFLAG window around the de-
        # fused standalone BN's per-iteration data RECV burst so it rendezvouses
        # with the producer conv's pre-send STANDBY (get_pre_send_sync). Emitted as
        # pre=SETFLAG(f) / post=SETFLAG(0) wrapping the RECV burst -- the
        # RecvSendWrapper's create_loop re-enters this window every outer-loop
        # iteration, matching the producer's per-STEP SEND-burst so neither side
        # can outrun the other across a kernel-launch boundary. Both sides key on
        # is_defused_standalone_bn_data_edge + pack_const_sync_flag(); OFF /
        # non-de-fused edges never reach here -> byte-identical.
        if (not pair.sender_node.is_inode()
                and self.is_defused_standalone_bn_data_edge(edge)):
            # PACK_BN_DATA_SYNC_FLAG (252), NOT 1: this node ALSO presents flag 1
            # in the CONFIG phase for the inode's packed-postop const pacing
            # (INODE_STANDBY(this_node, 1)). A flag-1 EXEC-loop data window
            # aliases it and wedges the launch-2 inode barrier. The producer
            # pre-send STANDBY (get_pre_send_sync) waits on this same value 252.
            return ([f"__builtin_IMCE_SETFLAG({PACK_BN_DATA_SYNC_FLAG});"],
                    ["__builtin_IMCE_SETFLAG(0);"])

        # Fix C receiver side: the composite `data` RECV of a terminal/fan-out
        # wrapper fed by a plain conv is bare too (handcraft imce_0_1 / imce_2_1
        # RECV their pipeline `data` with NO SETFLAG window). region1's single-
        # output mid-chain receiver keeps its window (predicate False).
        if (not pair.sender_node.is_inode()
                and self._is_conv_data_into_terminal_or_fanout(edge)):
            return None, None

        if pair.sender_node.is_inode():
            # inode -> imce. Only the main-pipeline data input gets a window
            # (STANDBY on the sending inode); everything else (weights already
            # excluded as const; fused/composite receivers) is bare.
            #
            # IMCFLOW_RESIDUAL_IN_REGION: the plain-int-dst consumer of a model-
            # input multicast that is ALSO fed to an in-region residual add
            # (imce_0_1: standalone min_max_quantize co-fed with imce_0_2's skip)
            # must be BARE, mirroring the proven OFF baseline where this fanout
            # receiver drains via fifo backpressure while the residual-add
            # consumer's per-word window paces the single inode SEND. Its RECV
            # window (num_blocks==4 wrapped as one window = per-4-word) otherwise
            # desyncs the inode's per-word SEND -> region1 tiled-launch deadlock.
            # Lever OFF -> predicate False -> the normal window below (unchanged).
            if self.is_residual_multicast_conv_input_recv(edge):
                return None, None
            if self.is_inode_data_input_recv(edge):
                pre = [
                    "__builtin_IMCE_SETFLAG(1);",
                    f"__builtin_IMCE_STANDBY({pair.sender_node.value}, 1);",
                    "__builtin_IMCE_SETFLAG(0);",
                ]
                # window closes before the RECV -> everything in pre_lines
                return pre, []
            return None, None

        # imce -> imce pipeline: burst window, no STANDBY on receiver side.
        return ["__builtin_IMCE_SETFLAG(1);"], ["__builtin_IMCE_SETFLAG(0);"]

    def is_multiblock_fusedadd_input_edge(self, edge: TensorEdge) -> bool:
        """Silicon-deadlock discriminator (used by IMCFLOW_MULTIBLOCK_FUSEDADD_BARE).

        True iff `edge` is an inode->imce data-input edge whose RECEIVER is a
        2-inode-fed fused-add (Fix-D merged case) with num_blocks > 1 (i.e. the
        consumer's output has > 16 channels, so its RECV window is re-armed per
        block -- ResNet8 region3 imce_1_1 has 32ch -> 2 blocks). Both the
        consumer window (imce side) and the producer 4-phase handshake (inode
        side) key on THIS predicate so they stay in lockstep when the lever
        bares them. region2 imce_1_3 (16ch -> 1 block) returns False and keeps
        its rendezvous.
        """
        pair = self.get_pair(edge)
        if pair is None or not pair.sender_node.is_inode():
            return False
        if not self.is_inode_data_input_recv(edge):
            return False
        recv_hw = self._get_hw_node(edge.dst_id)
        if isinstance(recv_hw, tuple):
            recv_hw = recv_hw[0]
        # count DISTINCT inode senders feeding data inputs into this receiver
        inode_senders = set()
        for p in self.pairs.values():
            if not p.sender_node.is_inode():
                continue
            if recv_hw not in p.receiver_nodes:
                continue
            for e in p.edges:
                if self.is_inode_data_input_recv(e):
                    inode_senders.add(p.sender_node.value)
                    break
        if len(inode_senders) < 2:
            return False  # not a 2-inode fused-add -> Fix-D merge doesn't apply
        # num_blocks = ceil(consumer output channels / 16). Compute from the
        # consumer relay call's output shape (NCHW: channel = dim 1).
        try:
            import math as _math
            from tvm.relay.frontend.common import infer_shape
            dst_gid = edge.dst_id.graph_node_id
            inner = dst_gid[-1] if isinstance(dst_gid, tuple) else dst_gid
            node = CustomIDToNode()[inner]
            out_shape = infer_shape(node)
            if isinstance(out_shape, (list, tuple)) and len(out_shape) and \
               isinstance(out_shape[0], (list, tuple)):
                out_shape = out_shape[0]
            channels = out_shape[1]
            num_blocks = _math.ceil(channels / 16.0)
            return num_blocks > 1
        except Exception:
            return False

    def fusedadd_consumer_num_blocks(self, edge: TensorEdge):
        """Silicon-SAFE redesign helper (IMCFLOW_MULTIBLOCK_FUSEDADD_SAFE).

        For an inode->imce data-input edge feeding a 2-inode fused-add consumer,
        return (consumer_node_value, num_blocks) where num_blocks =
        ceil(consumer output channels / 16). Returns None if `edge` is not such
        an edge (so the SAFE handshake only replaces the 2-inode fused-add
        rendezvous and every other edge is byte-identical). Mirrors the detector
        in is_multiblock_fusedadd_input_edge but returns the block count (which
        the token scheme needs) rather than a bool, and does NOT gate on
        num_blocks>1 (num_blocks==1 == region2 imce_1_3 is also made safe: its
        single window carries token (1,2) with the same interlock -- see NOTE).
        """
        pair = self.get_pair(edge)
        if pair is None or not pair.sender_node.is_inode():
            return None
        if not self.is_inode_data_input_recv(edge):
            return None
        recv_hw = self._get_hw_node(edge.dst_id)
        if isinstance(recv_hw, tuple):
            recv_hw = recv_hw[0]
        if recv_hw is None or not recv_hw.is_imce():
            return None
        # require >=2 DISTINCT inode senders feeding data inputs into recv_hw
        inode_senders = set()
        for p in self.pairs.values():
            if not p.sender_node.is_inode():
                continue
            if recv_hw not in p.receiver_nodes:
                continue
            for e in p.edges:
                if self.is_inode_data_input_recv(e):
                    inode_senders.add(p.sender_node.value)
                    break
        if len(inode_senders) < 2:
            return None
        try:
            import math as _math
            from tvm.relay.frontend.common import infer_shape
            dst_gid = edge.dst_id.graph_node_id
            inner = dst_gid[-1] if isinstance(dst_gid, tuple) else dst_gid
            node = CustomIDToNode()[inner]
            out_shape = infer_shape(node)
            if isinstance(out_shape, (list, tuple)) and len(out_shape) and \
               isinstance(out_shape[0], (list, tuple)):
                out_shape = out_shape[0]
            channels = out_shape[1]
            num_blocks = _math.ceil(channels / 16.0)
            return (recv_hw.value, int(num_blocks))
        except Exception:
            return None

    def collect_inode_data_input_edges(self, edges: List[TensorEdge]) -> List[TensorEdge]:
        """Return the subset of `edges` that are inode->imce *data input*
        rendezvous edges (the ones get_recv_window_sync wraps in a
        SETFLAG(1);STANDBY(inode,1);SETFLAG(0) window).

        Used by Fix D to detect a receiver fed by >=2 inode data inputs (region2
        imce_1_3: lhs from inode_0_0, rhs from inode_1_0) so their per-edge
        windows can be MERGED into one. Order is preserved from `edges`.
        """
        out = []
        for edge in edges:
            pair = self.get_pair(edge)
            if pair is None or not pair.sender_node.is_inode():
                continue
            if self.is_inode_data_input_recv(edge):
                out.append(edge)
        return out

    def get_merged_inode_input_window(self, edges: List[TensorEdge]):
        """Fix D: merge multiple inode data-input RECV windows into ONE.

        When a receiver node RECVs its data operands from >=2 distinct inode
        senders (region2 imce_1_3: add lhs from inode_0_0, rhs from inode_1_0),
        the RTL flag reg is a single scalar per node (imce_ctrl.sv). Emitting a
        separate SETFLAG(1)..STANDBY..SETFLAG(0) window per edge toggles that
        flag 1->0->1->0, which desynchronizes from the later flag=2 output
        barrier and deadlocks (imce_1_3 STANDBY STALL). handcraft instead uses a
        SINGLE window that opens once, STANDBYs on every inode sender, then
        closes once, followed by all the RECVs:

            SETFLAG(1); STANDBY(s1,1); STANDBY(s2,1); ...; SETFLAG(0);
            RECV(f1); RECV(f2); ...

        Returns the merged pre_lines (window that closes before ALL the RECVs)
        for the given inode-data-input edges, or None when there are <2 such
        edges (so region1 -- which has at most one inode data input per receiver
        -- is unaffected and keeps its existing per-edge window). STANDBY order
        follows ASCENDING sender value to match handcraft (STANDBY(0,1) then
        STANDBY(5,1)).
        """
        input_edges = self.collect_inode_data_input_edges(edges)
        # distinct inode sender hw nodes feeding this receiver's data operands
        senders = []
        seen = set()
        for edge in input_edges:
            pair = self.get_pair(edge)
            s = pair.sender_node
            if s.value not in seen:
                seen.add(s.value)
                senders.append(s)
        if len(senders) < 2:
            return None  # not a multi-inode-input receiver -> no merge (region1)
        senders.sort(key=lambda x: x.value)
        pre = ["__builtin_IMCE_SETFLAG(1);"]
        for s in senders:
            pre.append(f"__builtin_IMCE_STANDBY({s.value}, 1);")
        pre.append("__builtin_IMCE_SETFLAG(0);")
        return pre

    def collect_residual_data_input_edges(self, edges: List[TensorEdge]) -> List[TensorEdge]:
        """IMCFLOW_RESIDUAL_IN_REGION: the subset of `edges` that are the TWO
        data operands of the in-region residual ADD (mixed imce+inode producers,
        composite `data` dst). Empty unless the lever is ON. Order preserved.
        """
        if not residual_in_region_mode():
            return []
        return [e for e in edges if self.is_residual_data_input_recv(e)]

    def get_merged_residual_input_window(self, edges: List[TensorEdge]):
        """IMCFLOW_RESIDUAL_IN_REGION: ONE merged input window for the in-region
        residual add's TWO data operands (main from imce_3_2, skip from
        inode_0_0). Mirrors get_merged_inode_input_window's Fix-D protocol but
        collects MIXED imce+inode producers (the inode-only variant drops the
        imce edge and returns None -> no window at all -> deadlock).

            SETFLAG(1); STANDBY(main_producer,1); STANDBY(skip_producer,1);
            SETFLAG(0); RECV(main..); RECV(skip..)

        The single scalar flag reg (imce_ctrl.sv) means ONE window covers BOTH
        producers (two toggling windows would 1->0->1->0 and desync -- the Fix D
        lesson). Producers STANDBY in ASCENDING node value. Returns the pre_lines
        or None when there are < 2 residual data producers (so nothing but the
        real residual add is affected). OFF -> [] collect -> None here.
        """
        input_edges = self.collect_residual_data_input_edges(edges)
        senders = []
        seen = set()
        for edge in input_edges:
            pair = self.get_pair(edge)
            if pair is None:
                continue
            s = pair.sender_node
            if s.value not in seen:
                seen.add(s.value)
                senders.append(s)
        if len(senders) < 2:
            return None
        senders.sort(key=lambda x: x.value)
        pre = ["__builtin_IMCE_SETFLAG(1);"]
        for s in senders:
            pre.append(f"__builtin_IMCE_STANDBY({s.value}, 1);")
        pre.append("__builtin_IMCE_SETFLAG(0);")
        return pre

    def residual_add_consumer_of(self, edge: TensorEdge):
        """IMCFLOW_RESIDUAL_IN_REGION: if `edge` is a producer's SEND whose dst
        is an in-region residual add's `data` operand, return the consumer add's
        hw NodeID (so the producer can pre-send STANDBY(add, 1)); else None.

        Used by BOTH producers' pre-send paths:
          * imce_3_2 (IMCE): its odata SEND -> add gets STANDBY(add,1) (imce side)
          * inode_0_0 (INODE): its skip-data SEND -> add gets STANDBY(add,1) too
        matched to the consumer's SETFLAG(1) window (get_merged_residual_input_
        window). Returns None when OFF or not a residual data edge.
        """
        if not residual_in_region_mode():
            return None
        if not self.is_residual_data_input_recv(edge):
            return None
        recv_hw = self._get_hw_node(edge.dst_id)
        if isinstance(recv_hw, tuple):
            recv_hw = recv_hw[0]
        return recv_hw

    def get_pre_send_sync(self, edge: TensorEdge):
        """Return the list of pre-send sync lines for a sender's SEND loop.

        For an imce->imce pipeline edge this is a single STANDBY(receiver, 1)
        emitted once before the SEND loop. For an imce->inode output edge (or
        any edge whose sole receiver is an inode) it is bare (empty list).
        Returns [] when no pre-send sync is needed.
        """
        pair = self.get_pair(edge)

        # IMCFLOW_RESIDUAL_IN_REGION: an IMCE producer (imce_3_2) whose odata is a
        # `data` operand of the in-region residual add pre-sends STANDBY(add, 1),
        # matching the consumer's merged SETFLAG(1) window
        # (get_merged_residual_input_window). WITHOUT this the conv-odata edge
        # into the fused add wrapper is (correctly, for OFF) classified as a
        # composite-boundary / conv-into-fanout edge below and emitted BARE ->
        # producer/consumer desync -> deadlock. Placed FIRST so it takes
        # precedence over those bare predicates. Gated on the lever + the mixed-
        # pair detector, so OFF (and every non-residual imce->composite edge)
        # is untouched. Inode producers reach the same STANDBY(add,1) via
        # inode_codeblock's _get_presend_sync_code_str; both use flag value 1 so
        # they rendezvous with the single consumer window.
        if (pair is not None
                and pair.sender_node.is_imce()
                and self.is_residual_data_input_recv(edge)):
            add_hw = self.residual_add_consumer_of(edge)
            if add_hw is not None:
                return [f"__builtin_IMCE_STANDBY({add_hw.value}, 1);"]

        # Marker B' (flag-3 data-multicast barrier): a data-path producer whose
        # odata is MULTICAST to two imce receivers (fused consumer + sibling conv)
        # gates its SEND on BOTH receivers at flag 3. Takes precedence over the
        # Marker B sibling STANDBY (which would emit only the rhs sibling @flag1).
        # handcraft imce_0_3: STANDBY(imce_1_2,3); STANDBY(imce_1_3,3).
        if (pair is not None
                and pair.sender_node.is_imce()
                and pair.sender_node.value in self._flag3_sender_targets):
            targets = self._flag3_sender_targets[pair.sender_node.value]
            return [f"__builtin_IMCE_STANDBY({t}, 3);" for t in targets]

        # Marker B: a data/lhs-path producer feeding a 2-producer fused consumer
        # rendezvouses with its SIBLING rhs-path producer, NOT with the consumer.
        # Emit only STANDBY(sibling); suppress the receiver-node STANDBYs.
        sibling_hw = self._sibling_standby.get(edge, None)
        if sibling_hw is not None:
            return [f"__builtin_IMCE_STANDBY({sibling_hw.value}, 1);"]

        if pair is None:
            return []

        # Fix B: multicast producer that also has an inode-fed input rendezvous
        # (imce_1_3). The output multicast uses a 2-PHASE flag=2 barrier so it
        # does NOT alias the same node's flag=1 input window:
        #   STANDBY(r,2) x N; SETFLAG(2); STANDBY(r,0) x N; SETFLAG(0)
        # handcraft imce_1_3 emits the receivers in DESCENDING value order
        # (STANDBY(7,..) then STANDBY(3,..)), so mirror that here.
        if self._is_fixb_multicast_edge(edge):
            recv_nodes = sorted(
                [r for r in pair.receiver_nodes if r.is_imce()],
                key=lambda x: x.value, reverse=True)
            lines = []
            for rnode in recv_nodes:
                lines.append(f"__builtin_IMCE_STANDBY({rnode.value}, 2);")
            lines.append("__builtin_IMCE_SETFLAG(2);")
            for rnode in recv_nodes:
                lines.append(f"__builtin_IMCE_STANDBY({rnode.value}, 0);")
            lines.append("__builtin_IMCE_SETFLAG(0);")
            return lines

        # IMCFLOW_PACK_BN_MINMAX de-fuse fix (takes precedence over the two bare
        # predicates below): the de-fused standalone-BN's `data` edge (region3
        # imce_1_1->imce_2_1) would otherwise be bared as a conv->terminal edge,
        # leaving the standalone BN with no launch-boundary rendezvous -> 2nd-
        # launch deadlock. Re-arm a per-block pre-send STANDBY on the receiving
        # standalone BN so the producer conv cannot outrun it across the launch
        # boundary; matched by get_recv_window_sync's block-outer receiver window
        # (both keyed on is_defused_standalone_bn_data_edge). Gated on
        # pack_bn_minmax_mode()+exclusion set -> OFF / non-de-fused edges untouched.
        if self.is_defused_standalone_bn_data_edge(edge):
            # PACK_BN_DATA_SYNC_FLAG (252), NOT 1: the receiver standalone-BN node
            # (imce_2_1) presents flag 1 during the CONFIG phase for the inode's
            # packed-postop const pacing (INODE_STANDBY(imce_2_1, 1)). A flag-1
            # data window aliases it and wedges the launch-2 inode barrier (fsim:
            # inode_2_0 STANDBY expected=0x1 actual=0x0). 252 is reserved (pair
            # UUID cap lowered to 251) so it never collides.
            return [f"__builtin_IMCE_STANDBY({r.value}, {PACK_BN_DATA_SYNC_FLAG});"
                    for r in sorted(pair.receiver_nodes, key=lambda x: x.value)
                    if r.is_imce()]

        # Fused-sender -> output-node edge is bare (no pre-send STANDBY). Removes
        # the spurious STANDBY(16,1) on region2 imce_3_2->imce_3_1. region1
        # imce_1_1->imce_2_1 (fused sender but mid-chain receiver) and
        # imce_2_1->imce_3_1 (plain sender) both fail the predicate -> stay synced.
        if self._is_composite_boundary_edge(edge):
            return []

        # Fix C: plain conv-odata -> composite `data` of a terminal/fan-out wrapper
        # is bare on the sender side (handcraft R2 imce_0_2->imce_0_1 STANDBY(1,1)
        # and imce_1_1->imce_2_1 STANDBY(11,1) are absent). region1's analogous
        # edge has a single-output mid-chain receiver -> predicate False -> synced.
        if self._is_conv_data_into_terminal_or_fanout(edge):
            return []

        # Fix F (region3 imce_2_2): an imce->imce MULTICAST SEND (>=2 imce
        # receivers) must NOT pre-send-STANDBY on a receiver that consumes this
        # SEND's data via a BARE LOAD_LB. Those receivers open no matching
        # receiver-side SETFLAG window (their rendezvous rides a SECOND inter-node
        # producer's RECV window instead), so a STANDBY on them can never clear ->
        # the sender gates its own SEND on consumers it is feeding -> circular
        # deadlock (fsim-confirmed: imce_2_2 STANDBY(13/17/18) vs consumers
        # starved on their first LOAD_LB). IR test (verified byte-exact vs
        # handcraft on every imce->imce multicast in all 4 regions): keep only
        # receivers reached via a PLAIN (non-tuple graph_node_id) dst edge; drop
        # receivers reached ONLY via TUPLE (composite/fused-wrapper) dsts.
        # region1 has ZERO imce->imce multicasts (every imce sender is
        # single-receiver) so this branch never fires there -> region1 unchanged.
        imce_receivers = [r for r in pair.receiver_nodes if r.is_imce()]
        if len(imce_receivers) >= 2:
            # map each receiver hw node -> does it have any plain (non-tuple) dst?
            recv_has_plain = {}
            for e in pair.edges:
                rn = self._get_hw_node(e.dst_id)
                rns = rn if isinstance(rn, tuple) else (rn,)
                is_plain = not isinstance(e.dst_id.graph_node_id, tuple)
                for one in rns:
                    recv_has_plain[one] = recv_has_plain.get(one, False) or is_plain
            lines = []
            for rnode in sorted(imce_receivers, key=lambda x: x.value):
                if recv_has_plain.get(rnode, False):
                    lines.append(f"__builtin_IMCE_STANDBY({rnode.value}, 1);")
            return lines

        lines = []
        for rnode in sorted(pair.receiver_nodes, key=lambda x: x.value):
            if rnode.is_imce():
                lines.append(f"__builtin_IMCE_STANDBY({rnode.value}, 1);")
        return lines
