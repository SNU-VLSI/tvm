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
from tvm.contrib.imcflow import CodegenContext
from tvm.relay.op.contrib.imcflow import CustomIDToNode
from tvm.relay.op.contrib.imcflow import pack_bn_minmax_mode, residual_in_region_mode, residual_inode_buffer_mode
from tvm.relay.op.contrib.imcflow import region_merge_mode
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

# C1b (C) wave-launch realization: dedicated IMCE-flag value for the per-wave
# completion rendezvous. A core reused across launch waves SETFLAGs this as the
# last act of its wave-(k-1) program (after its final SEND, before STOP); the
# owning inode STANDBYs on it before re-WR_IMEM(wave k), so the previous wave's
# program has fully run before its IMEM is overwritten (WR_IMEM-vs-in-flight-
# fetch race fix). Reserved like 252/253/254/255: under REGION_MERGE (the only
# path that uses waves) the pair-UUID cap drops to 249 so no data pair aliases
# it. Set on the IMCE flag register (distinct from the INODE 254/255 barrier
# senses). Off merge -> never emitted -> byte-identical.
WAVE_DONE_FLAG = 250

# Option A (merged region1): MONOTONIC PHASE-TOKEN CYCLE for the paced region-
# input MULTICAST rendezvous (the `-11` model input reaching BOTH a handshake-
# gated conv-head consumer imce_0_2 AND the in-region residual add's skip operand
# imce_1_2). This REPLACES the earlier single repeated flag 249 (retired): a
# single value toggling 249->0->249->0 each of the ~1024 packet iterations is a
# REPEATED edge that collapses/ambiguates across iterations -- a consumer passed
# its STANDBY(inode,249) on a STALE 249 from the previous iteration and ran one
# iteration ahead of the producer -> data-stream re-arm wedge (RTL region1, fsim:
# inode_0_0 stuck at STANDBY(2,249) while imce_0_2 already past its window at
# RECV). This is exactly the "1->0->1->0 toggle race" the Fix-D / SAFE_TOKEN
# comments warn about.
#
# Fix (mirrors IMCFLOW_MULTIBLOCK_FUSEDADD_SAFE / SAFE_TOKEN_BASE, imcflow.py:504-
# 550, adapted to 1-producer -> 2-consumer per-packet multicast): cycle K distinct
# token BLOCKS; iteration i uses block b = i % K with a token PAIR
#   T1 = base + 2b   (consumer READY invite, raised on the consumer's own flag)
#   T2 = base + 2b + 1 (producer GO release, raised on the inode's flag)
# Consecutive iterations use DIFFERENT tokens (no collapsible repeated edge), and
# the RECV is the ACK: a consumer cannot loop to block b+1 (raise its next T1)
# until the producer SENT block b (which required it observed the consumer's clear
# after T2) -> skew bounded to < 1 iteration -> a token cannot go stale by a full
# K-cycle. K=4 (=> 8 token values) gives ample margin.
#
# Token base space: values occupy [PACED_MULTICAST_TOKEN_BASE ..
# +2*PACED_MULTICAST_NUM_BLOCKS-1] = 241..248, placed JUST under 249 and BELOW the
# 250-255 reserved senses; the merge pair-UUID cap drops to 240 (below 241) so no
# data pair aliases any token. All THREE sites (both consumer recv windows + the
# inode merged presend) derive their tokens from paced_multicast_token(block) --
# the single source -- and unroll the per-packet loop by K with LITERAL tokens per
# block (INODE cannot lower a runtime `iter % K` flag value: backend 'Cannot
# select and'; and no generated code has ever used a runtime-variable flag value,
# so we stay on the proven literal path). Merge-gated -> non-merged / OFF never
# emit any token -> byte-identical.
PACED_MULTICAST_TOKEN_BASE = 241
PACED_MULTICAST_NUM_BLOCKS = 4  # K: cycles blocks 0..K-1; tokens 241..248

# Words per token WINDOW. One paced-multicast token-block covers ONE PIXEL of the
# `-11` model input, whose min_max_quantize consumer RECVs it as a group of 4
# BITPLANES (MinmaxQuantBlock._num_blocks == 4). So each consumer token-window
# wraps 4 RECVs, and -- to stay lockstep -- each producer (inode) token-window MUST
# wrap the SAME 4 SENDs (NOT 1). The earlier 1-SEND-per-window inode unroll made
# the inode advance its token 4x faster than the consumer: block b's window
# released only 1 of the 4 words the consumer awaited, so the consumer stalled on
# its 2nd RECV while the inode had already moved to block b+1's READY token ->
# wedge (inode STANDBY(243) vs consumer stuck at 2 of 4 RECVs). This constant is
# the SINGLE SOURCE for that group size; a codegen assert ties it to the consumer
# block's num_blocks so a future bitplane-count change fails loud rather than
# silently desyncing the two unrolls.
PACED_MULTICAST_WORDS_PER_WINDOW = 4


def paced_multicast_token(block):
    """SINGLE SOURCE for the paced region-input multicast phase-token pair.

    Given the per-iteration block index b (0..PACED_MULTICAST_NUM_BLOCKS-1),
    return (t1, t2) = (base+2b, base+2b+1). t1 is the consumer READY invite
    (consumer's own flag); t2 is the producer GO release (inode's flag). ALL
    three emission sites (inode merged presend + both imce consumer recv windows)
    MUST call this -- no hardcoded token values anywhere. See the
    PACED_MULTICAST_TOKEN_BASE block above for the interlock rationale.
    """
    b = int(block) % PACED_MULTICAST_NUM_BLOCKS
    return PACED_MULTICAST_TOKEN_BASE + 2 * b, PACED_MULTICAST_TOKEN_BASE + 2 * b + 1


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
        # pack-const pacing). Under REGION_MERGE (wave-launch), 250 is additionally
        # reserved for the per-wave completion rendezvous (WAVE_DONE_FLAG), so the
        # pair cap drops one further to 249. Merge mode implies pack+resid+
        # inode_buffer, so this branch must be checked BEFORE pack_bn_minmax_mode.
        if region_merge_mode():
            # 241..248 reserved for the paced region-input multicast phase-token
            # cycle (PACED_MULTICAST_TOKEN_BASE .. +2K-1, K=4, Option A) in
            # addition to 250-255 and 249 (now unused/retired), so the pair cap
            # drops to 240 (below the token base) -- no data pair aliases a token.
            _uuid_max = 240
        elif pack_bn_minmax_mode():
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
                #
                # C1b (C) MERGE FIX: this "feeder must be plain-int" test is a
                # PRE-MERGE assumption. It was tuned to fire Marker B' only for the
                # non-merged region3 shape (a standalone min_max_quantize fed by a
                # standalone add) and to reject a fused conv/BN composite feeder.
                # Under REGION_MERGE=2 the SAME diamond (one imce multicast SEND ->
                # {fused-add consumer, rhs sibling}) reappears but the multicast
                # producer is a fused conv+bn+minmax COMPOSITE (e.g. node67 imce_0_2
                # in resnet8 subset31). With the composite excluded, get_pre_send_sync
                # falls through to Marker B and emits STANDBY(rhs-sibling, 1) ONLY --
                # the multicast's SECOND receiver (the fused consumer) gets NO
                # producer-side rendezvous. Since the multicast is ONE physical NoC
                # stream into shallow (depth-2) recv fifos, the un-gated receiver's
                # fifo backs up, the SEND wedges mid-stream (fsim: node67 sent 16,
                # consumers stalled at pc5 having taken 13-15), the fused add never
                # gets its data, and OP_STOP never fires -> wave0 pipeline deadlock.
                # The multicast-into-diamond TOPOLOGY (len(imce_recvs)>=2 above + the
                # exact {sibling,consumer} H3 guard below) is what Marker B' actually
                # needs; the feeder's plain-vs-composite nature is irrelevant to the
                # fifo-lockstep requirement. So only apply the composite-feeder
                # exclusion PRE-MERGE (keeps non-merged .packresid byte-identical);
                # under region_merge keep the composite-fed multicast producer.
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
                if feeder_is_composite and not region_merge_mode():
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

    def _edge_dst_wave(self, edge) -> int:
        """Launch wave of a pack-const edge = its IMCE (dst) endpoint's wave.

        Under IMCFLOW_REGION_MERGE a core is reused across waves; two pack-const
        consumers on the SAME (inode, imce) but DIFFERENT waves (e.g. region2
        node64 residual-add mult-const in wave 0 AND node92 conv config in wave 1,
        both imce_0_1 <- inode_0_0) would otherwise share one go-flag and alias
        across the wave boundary (v14 wedge: imce_0_1 waits the wave-0 mult-const
        flag 200 forever while the inode's flag-200 sends are node92's wave-1
        consts). So the go-flag must be per-(inode, imce, WAVE). Returns 0 for
        non-merged / single-wave / unmapped -> the (imce, 0) key collapses to the
        old (imce)-only behavior -> byte-identical."""
        try:
            fn = CodegenContext().func_name
            wm = DevConfig().GraphNodeToWavePerFunc.get(fn, {}) or {}
            gid = edge.dst_id.graph_node_id
            if gid in wm:
                return wm[gid]
            if isinstance(gid, tuple):
                if gid[-1] in wm:
                    return wm[gid[-1]]
                if gid[0] in wm:
                    return wm[gid[0]]
        except Exception:
            pass
        return 0

    def _pack_const_multi_consumer_inodes(self) -> Dict[int, List[tuple]]:
        """{inode_value: sorted[(imce_value, wave),...]} for inodes that pace >=2
        DISTINCT (imce, wave) pack-const consumers.

        The pack-const go-pulse (inode SET_FLAG(pack_const_sync_flag)) is a SHARED
        scalar flag on the inode; every consumer of that inode does
        STANDBY(inode, flag). When ONE inode paces two consumers -- either two
        DIFFERENT imce cores (region3 inode_2_0 -> de-fused BN imce_2_1 AND fused
        conv imce_2_2) OR the SAME core in TWO WAVES (region2 merge: imce_0_1
        wave0 node64 mult-const AND wave1 node92 config, both from inode_0_0) --
        the pulse for consumer A is also visible to B -> theft -> wedge. Keying on
        (imce_value, WAVE) makes both the multi-core and the cross-wave cases
        distinct. Cached (pair set fixed post-construction). Non-merged: every
        wave==0 -> (imce, 0) == the old imce-only set -> byte-identical.
        """
        cached = getattr(self, "_pack_const_multi_cache", None)
        if cached is not None:
            return cached
        inode_consumers: Dict[int, Set[tuple]] = {}
        if pack_bn_minmax_mode():
            for e in self._iter_all_edges():
                eps = self.packed_postop_const_endpoints(e)
                if eps is None:
                    continue
                inode_hw, imce_hw = eps
                w = self._edge_dst_wave(e)
                inode_consumers.setdefault(inode_hw.value, set()).add((imce_hw.value, w))
        result = {k: sorted(v) for k, v in inode_consumers.items() if len(v) >= 2}
        self._pack_const_multi_cache = result
        return result

    def pack_const_go_flag(self, inode_hw, imce_hw, wave: int = 0) -> int:
        """Per-(consumer, wave) go-pulse value for packed-postop const pacing.

        Default = pack_const_sync_flag() (1 or 253), byte-identical to the
        single-consumer form. When `inode_hw` paces >=2 distinct (imce, wave)
        consumers, each gets a DISTINCT value in a safe band, so a SET_FLAG for
        one consumer cannot be sampled by another -- INCLUDING the same core in a
        different wave (v14 cross-wave alias fix). Both the inode SendBlock and the
        imce RecvConstBlock call this with the SAME (inode, imce, wave) so they
        agree. Non-merged -> wave==0 for all -> byte-identical to the prior form.
        """
        base = self.pack_const_sync_flag()
        multi = self._pack_const_multi_consumer_inodes()
        consumers = multi.get(inode_hw.value)
        if not consumers:
            return base
        # Distinct value per (imce, wave) consumer within a safe band. The band
        # base is 200; the cap must stay below the reserved 249 (pair-UUID cap),
        # 250 (WAVE_DONE), 251, 252/253, 254/255. Index by position in the sorted
        # (imce, wave) list so inode and imce sides derive the same value.
        key = (imce_hw.value, wave)
        idx = consumers.index(key) if key in consumers else consumers.index(
            next((c for c in consumers if c[0] == imce_hw.value), key))
        flag = 200 + idx
        assert flag < 249, (
            f"pack_const_go_flag {flag} >= 249 reserved band; inode {inode_hw} "
            f"paces {len(consumers)} (imce,wave) consumers -- exceeds [200,248].")
        return flag

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

    def _is_residual_add_odata_multicast(self, edge: TensorEdge) -> bool:
        """IMCFLOW_RESIDUAL_IN_REGION: True iff `edge` carries a standalone ADD's
        odata that MULTICASTS to >=2 imce consumers.

        This is the b2.res-projection fan-out: the previous block's residual add
        result (%26) is re-quantized twice with different min/max -- once for the
        main downsample conv (imce_0_2) and once for the projection shortcut conv
        (imce_1_1). Both quantize consumers RECV the SAME multicast word.

        Such an edge structurally matches _is_fixb_multicast_edge (multicast +
        the add sender also RECVs its two add operands = an input rendezvous), so
        it wrongly gets the Fix B flag-2 window. But a flag-2 per-word rendezvous
        shared by a producer + 2 consumers is racy: the producer's SETFLAG(2)/
        SETFLAG(0) pulses outrun the slower consumers (RTL: producer completes 64
        iters, consumers stall at ~4 -> lost wakeup). The correct behaviour is
        BARE (NoC valid/ready + fifo backpressure pace it) -- the residual
        philosophy. Gated on the lever; OFF -> False -> unchanged.
        """
        if not residual_in_region_mode():
            return False
        pair = self.get_pair(edge)
        if pair is None:
            return False
        if len([r for r in pair.receiver_nodes if r.is_imce()]) < 2:
            return False
        try:
            src_node = CustomIDToNode()[getInnerNodeID(edge.src_id.graph_node_id)]
            import tvm
            return (isinstance(src_node, tvm.relay.expr.Call)
                    and isinstance(src_node.op, tvm.ir.Op)
                    and src_node.op.name == "add")
        except (KeyError, AttributeError, Exception):
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

        EXCLUSION (residual-in-region): the residual-add-odata multicast
        (b2.res projection: add result re-quantized for main + skip) also
        satisfies (a)+(b) but its flag-2 3-way rendezvous is racy and must be
        BARE -- see _is_residual_add_odata_multicast.
        """
        pair = self.get_pair(edge)
        if pair is None:
            return False
        # (a) multicast: >=2 distinct receiver hw nodes, all imce
        imce_receivers = [r for r in pair.receiver_nodes if r.is_imce()]
        if len(imce_receivers) < 2:
            return False
        # residual-add odata multicast -> bare, not flag-2 (racy 3-way barrier)
        if self._is_residual_add_odata_multicast(edge):
            return False
        # C1b launch-aware PnR (region-merge): wave-sharing can co-locate a
        # conv producer that ALSO has an inode-fed input rendezvous with a
        # fan-out to >=2 imce consumer nodes -- structurally (a)+(b), but this
        # is the SAME racy 3-way flag-2 rendezvous the residual-add case above
        # excludes (producer SETFLAG(2)/SETFLAG(0) pulses outrun the slower
        # consumers -> lost wakeup). The proven baseline runs every imce->imce
        # conv multicast BARE (NoC valid/ready + fifo backpressure pace it; the
        # handshake map is empty for stock ResNet8). So under the merge lever we
        # bare these too. Gated on the lever -> OFF is byte-identical (stock
        # imce_1_3 flag-2 case in the non-merged region2 is untouched).
        if region_merge_mode() and pair.sender_node.is_imce():
            return False
        # (b) sender also has an inode-fed input rendezvous this iteration
        return self._node_has_input_rendezvous(pair.sender_node)

    def get_output_flag(self, edge: TensorEdge) -> int:
        """Flag slot for this edge's SEND/RECV rendezvous: 2 for a Fix B
        multicast-barrier edge, 1 otherwise."""
        return 2 if self._is_fixb_multicast_edge(edge) else 1

    def get_recv_window_sync(self, edge: TensorEdge, token_block=None):
        """Return (pre_lines, post_lines) to wrap a receiver's RECV/LOAD_LB burst.

        pre_lines are emitted before the burst, post_lines after. Returns
        (None, None) if this edge needs no receiver-side window (bare).

        `token_block` (Option A paced multicast only): the LITERAL per-iteration
        block index b in 0..PACED_MULTICAST_NUM_BLOCKS-1. The paced-multicast
        consumer window (below) uses paced_multicast_token(b) so consecutive
        unrolled iterations carry DISTINCT tokens (no repeated-edge re-arm race).
        None -> block 0 (single-shot / non-unrolled callers stay byte-identical
        for every NON-paced edge, which never reads token_block).
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
            # Option A (merged region1): a region-input skip that lands on a
            # composite `data` operand AND is co-multicast with a handshake-gated
            # conv-head consumer must NOT be bare -- pace it in the SAME per-packet
            # inode window as its co-consumer (and the inode merged presend,
            # inode_codeblock._get_presend_sync_code_str). BOTH multicast consumers
            # (the paced skip AND its handshake-gated co-consumer) use a DISTINCT
            # flag PACED_MULTICAST_SYNC_FLAG (249), NOT 1: the inode CONFIG phase
            # raises flag 1 for packed-postop const pacing, and an EXEC data window
            # on flag 1 aliases those stale pulses (one consumer passes STANDBY on
            # a config-stage 1 before the inode reaches the data rendezvous ->
            # config-stage mutual-wait wedge, the PACK_BN_DATA_SYNC_FLAG 252 class).
            # Window is pre-only (closes before the RECV) -> adds pacing only, does
            # NOT reorder the RECVs. Narrow-gated -> region2 / non-merged / OFF bare
            # or the plain flag-1 path.
            _paced_mc = (self.is_paced_region_input_residual_skip(edge)
                         or self.is_paced_multicast_handshake_consumer(edge))
            if _paced_mc:
                # MONOTONIC PHASE-TOKEN, interlocked, RECV-as-ack (mirrors SAFE,
                # adapted to 1-producer -> this-consumer per-packet). Block b's
                # token pair (t1, t2) comes from paced_multicast_token(b) -- the
                # SINGLE source shared with the inode presend. Per iteration:
                #   SETFLAG(t1);          announce READY on THIS consumer's flag
                #   STANDBY(inode, t2);   wait producer GO on the inode's flag
                #   SETFLAG(0);           clear (producer STANDBYs this)
                #   RECV(2)x4             the RECV is the ACK (cannot complete
                #                         until the inode SENT -> C can't re-arm
                #                         t1 for block b+1 before the producer
                #                         consumed block b).
                # The unrolled caller (RecvSendWrapper) passes token_block=b for
                # each of the K bodies; consecutive blocks use DISTINCT tokens so
                # there is no repeated-edge cross-iteration self-alias. Window is
                # PRE-only (closes before the RECV) -> pacing only, RECV order
                # unchanged. Non-paced edges never reach here -> byte-identical.
                t1, t2 = paced_multicast_token(token_block or 0)
                pre = [
                    f"__builtin_IMCE_SETFLAG({t1});",
                    f"__builtin_IMCE_STANDBY({pair.sender_node.value}, {t2});",
                    "__builtin_IMCE_SETFLAG(0);",
                ]
                return pre, []
            if self.is_inode_data_input_recv(edge):
                pre = [
                    "__builtin_IMCE_SETFLAG(1);",
                    f"__builtin_IMCE_STANDBY({pair.sender_node.value}, 1);",
                    "__builtin_IMCE_SETFLAG(0);",
                ]
                # window closes before the RECV -> everything in pre_lines
                return pre, []
            return None, None

        # IMCFLOW_RESIDUAL_IN_REGION: the residual-add odata multicast consumer
        # is fully BARE (paired with the bared sender pre-send, get_pre_send_sync).
        # A self-flag window here is harmless but the matching sender no longer
        # STANDBYs it, so drop it for a clean bare RECV. OFF -> predicate False.
        if self._is_residual_add_odata_multicast(edge):
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
        if residual_in_region_mode():
            # Task #8 iter4 GENERATION GUARD (pre-clear): with pure level flags
            # the per-word 4-phase can one-side complete -- the consumer's
            # STANDBY(s,1) eats the sender's STALE flag=1 left from the PREVIOUS
            # word (the sender had not yet lowered), so the consumer runs one
            # word ahead and parks in RECV while the sender waits for a pulse
            # that was already consumed (fsim: hub imce_0_1 in RECV + inode_0_0
            # in STANDBY(1,1) simultaneously, wedged at word ~16 when the
            # downstream drain timing shifts). Waiting for every sender to be
            # CLEARED (flag==0) before raising the next window forces strict
            # word-generation alternation: consumer closes word k -> sender
            # lowers+sends -> pre-clear passes -> word k+1 window opens. No
            # circular wait (the close->lower->pre-clear chain is acyclic).
            # Gated on the lever; OFF baseline stays byte-identical.
            pre = []
            for s in senders:
                pre.append(f"__builtin_IMCE_STANDBY({s.value}, 0);")
            pre.append("__builtin_IMCE_SETFLAG(1);")
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

    def is_paced_region_input_residual_skip(self, edge: TensorEdge) -> bool:
        """IMCFLOW_RESIDUAL_IN_REGION (merged region1): True iff `edge` is a
        REGION-INPUT (identity fanout, src gid < 0) that lands on a COMPOSITE
        (tuple-dst) `data` operand of an in-region op (e.g. b1.res add / vecops on
        imce_1_2), AND the SAME source TensorID is ALSO multicast to a HANDSHAKE-
        GATED conv-head consumer (a plain-int-dst inode data input,
        is_inode_data_input_recv True) in the same func.

        NOTE: this composite consumer's OTHER operand is a `lhs` (not `data`), so
        _residual_data_producers() (which counts only `data` dsts) finds ONE
        producer -> is_residual_data_input_recv() is False for it. Hence this
        predicate is defined STRUCTURALLY (gid<0 + composite `data` dst + imce
        receiver + a handshake-gated multicast sibling), NOT via
        is_residual_data_input_recv.

        Why it exists: normally a region-input fanout to a composite `data` dst is
        BARE (get_recv_window_sync -> inode branch -> neither
        is_residual_multicast_conv_input_recv nor is_inode_data_input_recv ->
        None). That's fine when it drains via fanout-lead. But merged region1
        makes `-11` a MULTICAST to BOTH imce_0_2 (handshake-gated b1 conv-head)
        AND imce_1_2 (bare vecops add). imce_1_2's bare `-11` RECV is INTERLEAVED
        with a flagged lhs RECV; when the lhs producer lags, imce_1_2 stalls
        mid-loop and stops draining the shared multicast -> the handshake-gated
        co-consumer imce_0_2 starves -> the inode's per-packet rendezvous never
        re-arms -> the all-inode barrier hangs (RTL region1 factor-1 wedge). Fix
        (Option A): pace this skip in the SAME per-packet inode flag-1 window as
        its handshake-gated co-consumer.

        Gate is DELIBERATELY narrow (lever-OFF / non-merged / region2 -> False):
          * residual_in_region_mode() ON
          * edge is an inode->imce paired `data` RECV with a COMPOSITE tuple dst
            whose receiver is an imce
          * its src is a region input (graph_node_id < 0)
          * some sibling edge sharing this src TensorID is a handshake-gated
            plain-int-dst inode data input (is_inode_data_input_recv True).
        region2's residual add (node 77) takes TWO SEPARATE region inputs into
        one add via lhs/rhs (no shared multicast, no is_inode_data_input_recv
        sibling) -> False.

        ALSO gated on region_merge_mode(): the NON-merged combined region1
        contains a structurally matching multicast (input -> quant + the
        b1.res skip-scale producer) that is PROVEN bit-exact with bare
        pacing -- pacing it would change a verified program, so this
        predicate only fires for merged regions (where the bare co-consumer
        is the interleaved-RECV residual vecops that actually deadlocks).
        """
        if not (residual_in_region_mode() and region_merge_mode()):
            return False
        pair = self.get_pair(edge)
        if pair is None:
            return False
        if not pair.sender_node.is_inode():
            return False
        if edge.dst_id.tensor_type != "data":
            return False
        if not isinstance(edge.dst_id.graph_node_id, tuple):
            return False
        recv_hw = self._get_hw_node(edge.dst_id)
        if isinstance(recv_hw, tuple):
            recv_hw = recv_hw[0]
        if recv_hw is None or not recv_hw.is_imce():
            return False
        _sgid = getattr(edge.src_id, "graph_node_id", None)
        if not (isinstance(_sgid, int) and _sgid < 0):
            return False
        src_id = edge.src_id
        for p in self.pairs.values():
            for e in p.edges:
                if e is edge:
                    continue
                if e.src_id is not src_id:
                    continue
                if self.is_inode_data_input_recv(e):
                    return True
        return False

    def is_paced_multicast_handshake_consumer(self, edge: TensorEdge) -> bool:
        """Option A (merged region1): True iff `edge` is the HANDSHAKE-GATED
        conv-head consumer (is_inode_data_input_recv, plain-int dst, e.g. imce_0_2)
        of a region-input MULTICAST whose SAME source TensorID ALSO feeds a paced
        residual skip (is_paced_region_input_residual_skip True, e.g. imce_1_2).

        This is the SIBLING of is_paced_region_input_residual_skip. Both consumers
        of the paced multicast must use the SAME distinct rendezvous flag
        (PACED_MULTICAST_SYNC_FLAG) so the inode's single merged window paces them
        in lockstep without aliasing the CONFIG-phase flag 1. Merge-gated ->
        non-merged / region2 / OFF -> False (no paced skip sibling).
        """
        if not (residual_in_region_mode() and region_merge_mode()):
            return False
        if not self.is_inode_data_input_recv(edge):
            return False
        src_id = edge.src_id
        for p in self.pairs.values():
            for e in p.edges:
                if e is edge:
                    continue
                if e.src_id is not src_id:
                    continue
                if self.is_paced_region_input_residual_skip(e):
                    return True
        return False

    def edge_is_paced_multicast(self, edge: TensorEdge) -> bool:
        """Option A convenience: True iff `edge` is EITHER paced-multicast side
        (the residual skip OR its handshake-gated conv-head co-consumer). Used by
        the consumer-loop unroller (create_loop_from_call) to decide whether to
        cycle the phase-token by PACED_MULTICAST_NUM_BLOCKS. Merge-gated via both
        predicates -> non-merged / OFF -> False."""
        return (self.is_paced_region_input_residual_skip(edge)
                or self.is_paced_multicast_handshake_consumer(edge))

    def has_paced_multicast_edge(self, edges) -> bool:
        """True iff ANY edge in `edges` is a paced region-input multicast
        consumer edge (see edge_is_paced_multicast). Lets a RecvSendWrapper /
        create_loop_from_call detect a paced-multicast consumer node so its
        per-packet loop is unrolled by K with a distinct phase-token per block."""
        return any(self.edge_is_paced_multicast(e) for e in (edges or []))

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
        bared_operand = False
        for edge in input_edges:
            # BARE operands ride pure NoC valid/ready and must NOT be waited
            # on in the window (their sender never raises a flag -> instant
            # wedge). Three kinds, each lockstep with the matching sender-side
            # presend skip:
            #   * resbuf_out: RESBUF resend is emitted flag-free
            #     (ResidResendFuncoutInterleavedBlock).
            #   * IMCE producer: its scalar flag is already overloaded by its
            #     own input pacing -- the window would eat a spurious toggle
            #     as a fake ACK (RTL-traced generation race).
            #   * REGION-INPUT src (identity-skip fanout, src gid < 0): paced
            #     by rhs-fifo backpressure + fanout-lead.
            # Dedicated INODE senders (e.g. an entry add fed by 2 inodes)
            # KEEP the window -- that protocol is region2-proven.
            src_tt = getattr(edge.src_id, "tensor_type", None)
            if residual_inode_buffer_mode() and src_tt == "resbuf_out":
                bared_operand = True
                continue
            pair = self.get_pair(edge)
            if pair is None:
                continue
            s = pair.sender_node
            if residual_inode_buffer_mode() and s.is_imce():
                bared_operand = True
                continue
            _sgid = getattr(edge.src_id, "graph_node_id", None)
            if isinstance(_sgid, int) and _sgid < 0:
                bared_operand = True
                continue
            if s.value not in seen:
                seen.add(s.value)
                senders.append(s)
        if bared_operand:
            # The window covers only the remaining (windowed) producers, even
            # a SINGLE one -- falling into the generic len<2 -> None branch
            # would kill the window while its sender's presend STANDBY(add,1)
            # remains (mutual first-word deadlock, L1 unit test). All bare ->
            # return [] (NOT None): no window lines, but the merged pairwise
            # RECV order is KEPT (imce_codeblock joins [] to "") -- edge-outer
            # RECV order overflows the IMCE register file (no spill support).
            if not senders:
                return []
        elif len(senders) < 2:
            return None
        senders.sort(key=lambda x: x.value)
        # iter4 generation guard (pre-clear) -- same stale-level race as
        # get_merged_inode_input_window; see the comment there. Always under
        # the residual lever here (this function is residual-only).
        pre = []
        for s in senders:
            pre.append(f"__builtin_IMCE_STANDBY({s.value}, 0);")
        pre.append("__builtin_IMCE_SETFLAG(1);")
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
            # Under the RESBUF lever this edge is BARE: the consumer window
            # drops IMCE senders (flag-overload race, see
            # get_merged_residual_input_window) -- LOCKSTEP: skip the
            # producer-side STANDBY(add,1) too.
            if residual_inode_buffer_mode():
                return []
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

        # IMCFLOW_RESIDUAL_IN_REGION: the residual-add odata multicast to two
        # re-quantize consumers is fully BARE on the sender side. A per-word
        # STANDBY on BOTH plain-dst consumers (Fix F below) locks the producer to
        # the SLOWER branch each word; the two branches' downstream convs drain at
        # different rates, so the producer stalls (RTL: imce_0_1 computes 4 then
        # blocks). NoC valid/ready + fifo backpressure pace the multicast without
        # the per-word rendezvous (the residual philosophy; matches the bared
        # receiver windows in get_recv_window_sync). OFF -> predicate False.
        if self._is_residual_add_odata_multicast(edge):
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
