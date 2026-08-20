from tvm.relay.backend.contrib.imcflow.codeblock import *
from tvm.contrib.imcflow import DataBlock, InstEdgeInfo, TensorID, TensorEdge, TensorEdgeInfo
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from tvm.contrib.imcflow import bugfix_off_mode
from tvm.contrib.imcflow import drop_psum_send, drop_psum_keep_every
from tvm.contrib.imcflow import step_freerun_n, step_freerun_factors
from tvm.contrib.imcflow import feed_sync_per_pixel
from tvm.contrib.imcflow import qconv_nop_delay_cnt
from tvm.contrib.imcflow import input_reuse
from tvm.contrib.imcflow import input_reuse_feed_flagfree, input_reuse_feed_pace_nops
from tvm.contrib.imcflow import imcu_intra_drain_nops
from tvm.relay.op.contrib.imcflow import CustomIDToNode
from tvm.relay.backend.contrib.imcflow.transform import getInnerNodeID
from textwrap import indent
import math
import pdb

NOP_LOOP_CNTS = 10


def _wrap_freerun_passes(body, reps):
  """STEP_FREERUN root cause #2: wrap `body` (one per-pass feed/config) in a `reps`-trip
  outer loop, nested into <=16384 hardware-loop factors (INODE 14-bit limit) via
  step_freerun_factors(reps) -- the SAME factorization the imce uses so the (1+N) pass
  count is byte-identical on both sides. reps<=1 -> body unchanged (no freerun)."""
  if reps <= 1:
    return body
  if reps <= 16384:
    return SimpleFor(reps, body, "freerun_pass")
  factors = step_freerun_factors(reps)
  wrapped = SimpleFor(factors[-1], body, "freerun_pass")
  for f in reversed(factors[:-1]):
    wrapped = SimpleFor(f, wrapped, "freerun_pass")
  return wrapped


class InodeCodeBlock(CodeBlock):
  def __init__(self, annotation: str = ""):
    super().__init__(annotation)
    # Subclasses should build their structure into self.body in __init__
    self.body = SequentialBlock()

  def _render(self) -> str:
    return self.body.render()


class PolicyUpdateBlock(InodeCodeBlock):
  """ Code block for updating policy table for given inode's hw node id  """

  def __init__(self, node_id: NodeID, annotation: str = ""):
    super().__init__(annotation)
    assert node_id.is_inode(), "PolicyUpdateBlock can only be used for inode"
    self.node_id = node_id
    self._build()

  def _build(self):
    assert self.node_id.is_inode(), "PolicyUpdateCodeBlock can only be used for inode"
    same_row_node_ids = [self.node_id] + self.node_id.slaves()
    same_row_node_ids.sort(key=lambda id: id.to_coord(1))

    for id in same_row_node_ids:
      db = DevConfig().CurrFuncMemLayout.get_data_block_by_edge(f"{id.name}_policy")
      if db is None:
        continue
      var = UniqueVar("policy_table_start_address", dtype="int")
      loop_count = math.ceil(db.size / 32)

      self.body.add(TextBlock(f"{var} = {db.offset};"))
      
      # FIXME: maybe we should leave the loop optimization to llvm?
      if loop_count > 5:
        # Using lambda for SimpleFor body to inject 'iter' variable
        self.body.add(SimpleFor(loop_count, 
            lambda iter, wid=id.to_coord(1): f"__builtin_INODE_PU({var} + {iter}*32, 0, {iter}, {wid});"))
      else:
        for i in range(loop_count):
          self.body.add(TextBlock(f"__builtin_INODE_PU({var}, {i*32}, {i}, {id.to_coord(1)});"))


class WriteIMEMBlock(InodeCodeBlock):
  """ Code block for writing IMEM given InstEdgeInfo """

  def __init__(self, edge_info: InstEdgeInfo, annotation: str = ""):
    super().__init__(annotation)
    self.edge_info = edge_info
    self._build()

  def _build(self):
    db = self.edge_info.data_block
    policy_addr = self.edge_info.policy_info[0].address

    var = UniqueVar("imem_start_address", dtype="int")
    self.body.add(TextBlock(f"{var} = {db.offset};"))
    self.body.add(TextBlock(f"__builtin_INODE_SET_ADDR_CNT(0);"))

    self.body.add(SimpleFor(math.ceil(db.size / 32),
                      lambda iter: f"__builtin_INODE_WR_IMEM({var} + {iter}*32, 0, {policy_addr});"))


class WriteIMCUBlock(InodeCodeBlock):
  """ Code block for writing IMCU weights given the master inode's hid  """

  def __init__(self, node_id: NodeID, annotation: str = ""):
    super().__init__(annotation)
    assert node_id.is_inode(), "WriteIMCUBlock can only be used for inode"
    self.node_id = node_id
    self._build()

  def _build(self):
    # Intra-inode IMCU drain (IMCFLOW_IMCU_INTRA_DRAIN, default 0=OFF): when an
    # inode streams >=2 consecutive WR_IMCU bursts (e.g. region3 inode_3_0's two
    # 256-word bursts), the back-to-back streaming overruns the IMCU write path on
    # real silicon and wedges region3 at kernel entry (BUGFIX-off RTL tolerates
    # it). Inserting a NOP-delay loop BETWEEN the bursts lets the first burst fully
    # commit before the second starts. NopLoopBlock is purely local to this inode
    # (no NoC handshake) so it cannot deadlock. drain=0 -> byte-identical to stock.
    # See imcflow.imcu_intra_drain_nops().
    drain = imcu_intra_drain_nops()
    first_burst = True
    region = DevConfig().CurrFuncMemLayout[f"{self.node_id.name}_data"]
    for db in region.blocks.values():
      if isinstance(db.id, TensorEdge) and "weight" == db.id.src_id.tensor_type:
        info = DevConfig().get_tensor_edge_info(db.id)
        assert info.fifo_id == 1, f"IMCU fifo id should be set to 1 (although not used), but got {info.fifo_id} for {db.id}"
        var = UniqueVar("imcu_start_address", dtype="int")

        # Drain between consecutive bursts within this inode (not before the first).
        if drain > 0 and not first_burst:
          self.body.add(NopLoopBlock(drain, "imcu intra-inode drain between WR_IMCU bursts"))
        first_burst = False

        self.body.add(TextBlock(f"{var} = {db.offset};"))
        self.body.add(TextBlock(f"__builtin_INODE_SET_ADDR_CNT(0);"))
        self.body.add(SimpleFor(math.ceil(db.size / 32),
                          lambda iter, addr=info.policy_info[0].address: f"__builtin_INODE_WR_IMCU({var} + {iter}*32, 0, {addr});"))


class RecvBlock(InodeCodeBlock):
  """ Code block for receiving data from given fifo id """

  def __init__(self, builder, block: DataBlock, fifo_id: int, annotation: str = ""):
    super().__init__(annotation)
    self.builder = builder
    self.block = block
    self.fifo_id = fifo_id
    if self.block.tiling_info is not None:
      self._build_tiled()
    else:
      self._build()

  def _is_func_out_psum_recv(self) -> bool:
    """True iff this tiled RECV collects a conv psum output (dst tensor_type starts
    with 'func_out'), i.e. it consumes the imce ConvBlock's psum SEND. Only this edge
    is freerun-scaled (the imce repeats its psum SEND (1+N)x); input/weight RECVs are
    separate blocks and must NOT be scaled."""
    e = self.block.id[0] if isinstance(self.block.id, list) else self.block.id
    if not isinstance(e, TensorEdge):
      return False
    dst_tt = getattr(e.dst_id, "tensor_type", "")
    return isinstance(dst_tt, str) and dst_tt.startswith("func_out")

  def _build(self):
    recv_count = math.ceil(self.block.size / 32)
    var = UniqueVar("recv_data_base_address", dtype="int")

    self.body.add(TextBlock(f"{var} = {self.block.offset};"))

    # Per-packet sync: Receive one packet, then sync immediately
    def recv_body_with_sync(iter, var=var, fid=self.fifo_id):
      code = f"__builtin_INODE_RECV({var} + {iter}*32, 0, 0, {fid});\n"
      # Add sync after each recv
      sync_code = self._get_recv_sync_code_str()
      if sync_code:
        code += sync_code
      return code
    self.body.add(SimpleFor(recv_count, recv_body_with_sync))

  def _build_tiled(self):
    fifo_id = self.fifo_id
    if isinstance(self.block.id, list):
      block_id = self.block.id[0]
    else:
      block_id = self.block.id
    assert isinstance(block_id, TensorEdge), "Tiled recv block must be a tensor edge"
    target_edge = block_id

    cnt_addr_var = UniqueVar(f"{target_edge.simple_name()}_recv_data_base_address", dtype="int", pointer_type=True)
    _recv_cnt_address_block = DevConfig().MemLayout[self.builder.func_name].get_data_block_by_id(f"{target_edge.simple_name()}_cnt_base_addr")
    _recv_cnt_address = _recv_cnt_address_block.offset
    self.body.add(TextBlock(f"{cnt_addr_var} = (int*)({_recv_cnt_address});"))

    loop_cnt_var = UniqueVar(f"{target_edge.simple_name()}_tile_loop_count", dtype="int")
    base_var = UniqueVar("recv_data_base_address", dtype="int")
    self.body.add(TextBlock(f"{base_var} = {self.block.offset};"))

    # Power-measurement lever (IMCFLOW_STEP_FREERUN): the producing imce repeats its
    # whole conv body (1+N)x (row loop x(1+N) in ConvBlock._build_structure), so it
    # emits (1+N)x as many psum SENDs on this func_out edge. Scale this func_out RECV
    # count by (1+N) too, else imce sends (1+N)x more than the inode receives -> the
    # func_out fifo deadlocks / consistency fails (observed: 768 send vs 256 recv at
    # N=2, STEP_FREERUN-alone). Only the func_out (imce->inode psum collector) tiled
    # RECV -- input/weight RECVs are separate blocks and are NOT freerun-scaled. The
    # DROP_PSUM keep path below applies its OWN (1+N) scaling on the kept count, so
    # this multiply is for the plain (non-drop) path only. N=0 -> unchanged (x1).
    _fr_recv_mult = (1 + step_freerun_n()) if (step_freerun_n() > 0
                                               and self._is_func_out_psum_recv()) else 1
    if _fr_recv_mult != 1 and not drop_psum_send():
      # LOAD-USE HAZARD (see SendBlock._build_tiled): `cnt[0] * M` compiles LOAD then
      # INODE_MULI back-to-back with no bubble -> MULI reads a stale register ->
      # garbage loop count -> deadlock. Split: LOAD into temp, hazard nop, then MULI.
      _raw_recv_var = UniqueVar(f"{target_edge.simple_name()}_tile_loop_count_raw", dtype="int")
      self.body.add(TextBlock(f"{_raw_recv_var} = {cnt_addr_var}[0];"))
      # Register-barrier nop (see SendBlock): "+r" operand forces a bubble between
      # INODE_LOAD and INODE_MULI (a plain volatile nop is scheduled away).
      self.body.add(TextBlock(f"__asm__ volatile(\"nop\" : \"+r\"({_raw_recv_var}));")) # LOAD-USE HAZARD
      self.body.add(TextBlock(f"{loop_cnt_var} = {_raw_recv_var} * {_fr_recv_mult};"))
    else:
      self.body.add(TextBlock(f"{loop_cnt_var} = {cnt_addr_var}[0];"))
    self.body.add(TextBlock(f"__asm__ volatile(\"nop\");")) # BUGFIX_LOAD_USE_HAZARD

    # Max-throughput lever (IMCFLOW_DROP_PSUM): the producing imce drops its psum
    # SEND for garbage output, so this matching inode RECV loop would wedge waiting
    # for packets that never arrive. Drop it too (keep the count read + nop for
    # imem/label parity). Only the func_out (imce->inode output collector) tiled
    # RECV is affected; input/weight RECVs are separate blocks. Gated by env;
    # default OFF -> byte-identical.
    if drop_psum_send():
      _keep_k = drop_psum_keep_every()
      if _keep_k <= 0:
        # TRUE drop-all (keep=0): the producing imce drops all psum sends (see
        # ConvBlock._build_structure _drop_all) -> no packets arrive -> omit the
        # whole RECV loop. ★2026-08-20 BUGFIX-off RTL-proven symmetric-valid; the
        # old "wedges on chip" note was a codegen asymmetry misdiagnosis.
        self.body.add(TextBlock(f"// [DROP_PSUM] omitted tiled INODE_RECV loop ({loop_cnt_var} iters, keep=0)"))
        return
      # K-keep: the producing imce keeps its BLK-block psum drain per K PIXELS
      # (out_fifo drain; imce_codeblock ConvBlock._build_structure splits the pixel
      # loop into keep/skip tiers -- 1 drain per K pixels). The imce SENDs only the
      # kept pixels' BLK packets, and they arrive contiguously (skip pixels send
      # nothing), so the inode just RECVs the kept count in a flat COUNTED loop.
      # The INODE, like the IMCE, only supports counted hardware loops AND cannot
      # select a runtime `/K` (sra) or `%K`/`if` (and+br_cc) -> "Cannot select".
      # So compute the kept count as a COMPILE-TIME literal from the static packet
      # count (tiling_info.pkt_cnts), divided by K in Python. Valid because the
      # func_out drain is a single-tile block (tiling_factor==1 -> one pkt_cnts
      # entry). K MUST divide the per-tile packet count (choose K | pixels, e.g.
      # K=8 for an 8-wide col_group); assert if not so the imbalance surfaces at
      # compile time instead of an RTL fifo deadlock.
      _pkt_cnts = getattr(self.block.tiling_info, "pkt_cnts", None) if self.block.tiling_info else None
      assert _pkt_cnts and len(_pkt_cnts) >= 1, \
        f"[DROP_PSUM keep] func_out block {target_edge.simple_name()} has no static pkt_cnts"
      _total_pkts = int(_pkt_cnts[0])
      assert _total_pkts % int(_keep_k) == 0, \
        (f"[DROP_PSUM keep] K={_keep_k} must divide the func_out packet count "
         f"{_total_pkts} (choose K | packets so imce kept-SEND == inode kept-RECV; "
         f"else fifo deadlock)")
      _kept_pkts = _total_pkts // int(_keep_k)
      # STEP_FREERUN composes MULTIPLICATIVELY with K-keep: the producing imce
      # repeats its whole conv body (1+N) times (row loop x(1+N) in
      # ConvBlock._build_structure), so it emits (1+N)x as many KEPT psum SENDs.
      # The inode RECV count must match = kept_per_pass * (1+N), else imce sends
      # (1+N)x more than inode receives -> fifo deadlock (observed: 3232 vs 32 at
      # N=100,K=8). keep-tier drain (spatial, per K pixels) and freerun (temporal,
      # x(1+N) passes) are orthogonal and both scale the RECV count.
      _fr = step_freerun_n()
      _reps = (1 + _fr) if _fr > 0 else 1
      _kept_total = _kept_pkts * _reps
      self.body.add(TextBlock(
        f"// [DROP_PSUM] keep {_kept_pkts}/{_total_pkts} psum RECVs per pass (1 per "
        f"{int(_keep_k)} pixels) x {_reps} freerun passes = {_kept_total} total"))
      def recv_body_kept(iter, base_addr_var=base_var, fid=fifo_id):
        code = f"__builtin_INODE_RECV({base_addr_var} + {iter}*32, 0, 0, {fid});\n"
        sync_code = self._get_recv_sync_code_str()
        if sync_code:
          code += sync_code
        return code
      # Nest into counted hardware-loop factors <= 16384 (IMCE/INODE 14-bit limit)
      # when the total exceeds it; a single SimpleFor(>16384) makes clang crash.
      if _kept_total > 16384:
        _wrapped = None
        for _lvl, _f in enumerate(reversed(step_freerun_factors(_kept_total))):
          if _wrapped is None:
            _wrapped = SimpleFor(_f, recv_body_kept)
          else:
            _inner = _wrapped
            _wrapped = SimpleFor(_f, _inner)
        self.body.add(_wrapped)
      else:
        self.body.add(SimpleFor(_kept_total, recv_body_kept))
      return

    # Per-packet sync: Receive one packet, then sync immediately
    def recv_body_with_sync(iter, base_addr_var=base_var, fid=fifo_id):
      code = f"__builtin_INODE_RECV({base_addr_var} + {iter}*32, 0, 0, {fid});\n"
      # Add sync after each recv
      sync_code = self._get_recv_sync_code_str()
      if sync_code:
        code += sync_code
      return code
    self.body.add(SimpleFor(loop_cnt_var, recv_body_with_sync))

    # TEMPORARILY DISABLED: Testing UUID-based sync instead
    # Consumer INODE waits for producer IMCE to signal completion
    # Find producer IMCE ID from the tensor edge
    # SYNC_OUTPUT_FLAG = 2
    # producer_imce_ids = set()

    # Get tensor edge info to find the producer
    # if target_edge in DevConfig().TensorEdgetoInfo:
    #   te_info = DevConfig().TensorEdgetoInfo[target_edge]
    #   if te_info.policy_info:
    #     # The first router in policy_info is the source (producer)
    #     producer_router_id = te_info.policy_info[0].router_id
    #     if producer_router_id.is_imce():
    #       producer_imce_ids.add(producer_router_id.value)

    # Add STANDBY calls for each producer IMCE
    # for producer_imce_id in sorted(producer_imce_ids):
    #   # Find producer name for annotation
    #   producer_name = None
    #   if target_edge in DevConfig().TensorEdgetoInfo:
    #     te_info = DevConfig().TensorEdgetoInfo[target_edge]
    #     if te_info.policy_info and te_info.policy_info[0].router_id.value == producer_imce_id:
    #       producer_name = te_info.policy_info[0].router_id.name

    #   if producer_name:
    #     self.body.add(TextBlock(f"__builtin_INODE_STANDBY({producer_imce_id}, {SYNC_OUTPUT_FLAG}); // {producer_name}"))
    #   else:
    #     self.body.add(TextBlock(f"__builtin_INODE_STANDBY({producer_imce_id}, {SYNC_OUTPUT_FLAG});"))

    # Sync is already added inside the loop (per-packet sync), no need to add again here

  def _get_recv_sync_code_str(self):
    """Get sync code as a string (for inline insertion in loops) for recv.

    handcraft (knob=off): imce->inode output RECV is BARE (no SETFLAG/STANDBY).
    The inode just receives; the producing imce does the (bare) SEND. Return "".

    BUGFIX knob: knob=on (bugfix_off_mode()==False) reproduces a8af's per-RECV
    RECEIVER-pattern sync (SETFLAG(uuid); STANDBY(sender,uuid); SETFLAG(0)).
    """
    if bugfix_off_mode():
      return ""

    if not hasattr(self.builder, 'pair_manager') or self.builder.pair_manager is None:
      return ""
    edge = self._get_edge()
    if edge is None:
      return ""
    pair = self.builder.pair_manager.get_pair(edge)
    if pair is None:
      return ""
    dst_hw_node = self._get_hw_node_from_edge(edge)
    if dst_hw_node is None:
      return ""
    sync_lines = []
    sync_lines.append(f"__builtin_INODE_SET_FLAG({pair.uuid});")
    sync_lines.append(f"__builtin_INODE_STANDBY({pair.sender_node.value}, {pair.uuid});")
    sync_lines.append(f"__builtin_INODE_SET_FLAG(0);")
    return "\n".join(sync_lines) + "\n"

  def _add_sync_after_recv(self):
    """Add synchronization block after recv operation"""
    if not hasattr(self.builder, 'pair_manager') or self.builder.pair_manager is None:
      return  # No pair manager, skip sync

    # Get the tensor edge for this recv block
    edge = self._get_edge()
    if edge is None:
      return

    pair = self.builder.pair_manager.get_pair(edge)
    if pair is None:
      return  # No sync needed for this edge

    # Get current node - receiver node
    # For recv, we need to determine which node is receiving
    # The edge.dst_id should give us the tensor ID, and we can get hw node from there
    dst_hw_node = self._get_hw_node_from_edge(edge)
    if dst_hw_node is None:
      return

    # Add sync block
    sync_annotation = f"sync after recv: uuid={pair.uuid}, edge={edge}"
    sync_block = SyncPairINode(dst_hw_node, pair.all_nodes, pair.uuid, sync_annotation)
    self.body.add(sync_block)

  def _get_edge(self):
    """Get the tensor edge associated with this recv block"""
    if isinstance(self.block.id, list):
      return self.block.id[0] if len(self.block.id) > 0 else None
    return self.block.id if isinstance(self.block.id, TensorEdge) else None

  def _get_hw_node_from_edge(self, edge: TensorEdge):
    """Get hardware node for receiver from edge"""
    try:
      dst_gid = edge.dst_id.graph_node_id
      if isinstance(dst_gid, tuple):
        dst_gid = dst_gid[-1]
      hw_node = DevConfig().get_hw_node(dst_gid)
      # Handle tuple hw_node (from split operations)
      if isinstance(hw_node, tuple):
        # Use the first receiver node for sync
        return hw_node[0] if len(hw_node) > 0 else None
      return hw_node
    except Exception:
      return None


class RecvBlockInterleaved(InodeCodeBlock):
  """ Code block for receiving data from given fifo id interleaved """

  def __init__(self, builder, blocks: List[DataBlock], fifo_ids: List[int], annotation: str = ""):
    super().__init__(annotation)
    assert len(blocks) == len(fifo_ids), "# of blocks and fifo_ids must be equal"
    self.builder = builder
    self.blocks = blocks
    self.fifo_ids = fifo_ids
    self._build()

  def _build(self):
    # Collect block info
    info_list = []
    for block, fifo_id in zip(self.blocks, self.fifo_ids):
      recv_count = math.ceil(block.size / 32)
      info_list.append({
          'block': block,
          'recv_count': recv_count,
          'offset': block.offset,
          'fid': fifo_id
      })

    # Sort unique recv_counts to define intervals
    counts = sorted(list(set(x['recv_count'] for x in info_list)))

    current_base = 0
    for limit in counts:
      duration = limit - current_base
      if duration <= 0:
        continue

      # Identify blocks active in this interval
      active_infos = [x for x in info_list if x['recv_count'] > current_base]

      # Generate loop for this interval
      loop_body = SequentialBlock()
      for x in active_infos:
        var = UniqueVar("recv_offset_address", dtype="int")
        loop_body.add(TextBlock(f"{var} = {x['offset']};"))
        loop_body.add(TextBlock(f"__builtin_INODE_RECV({var} + ({f'({current_base} + {iter})' if current_base > 0 else iter})*32, 0, 0, {x['fid']});"))
        # Add sync after each recv in interleaved block
        self._add_sync_for_edge(x['block'])

      self.body.add(SimpleFor(duration, TextBlock(loop_body.render())))

      current_base = limit

  def _add_sync_for_edge(self, block: DataBlock):
    """Add synchronization for a specific edge in interleaved recv"""
    if not hasattr(self.builder, 'pair_manager') or self.builder.pair_manager is None:
      return

    # Get edge from block
    if isinstance(block.id, list):
      edge = block.id[0] if len(block.id) > 0 else None
    else:
      edge = block.id if isinstance(block.id, TensorEdge) else None

    if edge is None:
      return

    pair = self.builder.pair_manager.get_pair(edge)
    if pair is None:
      return

    # Get current node (receiver)
    try:
      dst_gid = edge.dst_id.graph_node_id
      if isinstance(dst_gid, tuple):
        dst_gid = dst_gid[-1]
      current_node = DevConfig().get_hw_node(dst_gid)
      if isinstance(current_node, tuple):
        current_node = current_node[0]
    except Exception:
      return

    # Add sync block
    sync_annotation = f"sync after interleaved recv: uuid={pair.uuid}, edge={edge}"
    sync_block = SyncPairINode(current_node, pair.all_nodes, pair.uuid, sync_annotation)
    self.body.add(sync_block)

class NopLoopBlock(InodeCodeBlock):
  """ Code block for a simple loop with NOP body, used for timing or synchronization purposes """

  def __init__(self, loop_count: int, annotation: str = ""):
    super().__init__(annotation)
    self.loop_count = loop_count
    self._build()

  def _build(self):
    self.body.add(SimpleFor(self.loop_count, TextBlock(f"__asm__ volatile(\"nop\");")))

class SendBlock(InodeCodeBlock):
  """ Code block for sending data from given fifo id """

  def __init__(self, builder, block: DataBlock, edge_info: TensorEdgeInfo, annotation: str = ""):
    super().__init__(annotation)
    self.builder = builder
    self.block = block
    self.edge_info = edge_info
    if self.block.tiling_info is not None:
      self._build_tiled()
    else:
      self._build()
    # STEP_FREERUN is applied INSIDE _build() by lengthening recv_count (the
    # per-packet rendezvous loop), NOT by wrapping self.body in a new outer loop --
    # an unsynchronized outer back-edge slips the scalar-flag toggle phase at the
    # rep seam and deadlocks (chip-proven). See recv_count *= (1+N) in _build().

  def _is_conv_activation_feed(self) -> bool:
    """True iff this SEND is the conv/dwconv ACTIVATION feed (paired with the imce
    ConvBlock LOAD_LB). Weight/const/config/psum SENDs -> False, so they are NEVER
    repeated under STEP_FREERUN. The activation edge is the one policy_table_builder
    (:499-513) gives dst tensor_type "data" + base fifo_id 0 feeding a qconv/qdwconv."""
    e = self.block.id if isinstance(self.block.id, TensorEdge) else None
    if e is None:
      return False
    if getattr(e.dst_id, "tensor_type", None) != "data":
      return False
    if getattr(self.edge_info, "fifo_id", -1) != 0:
      return False
    try:
      dst = CustomIDToNode()[getInnerNodeID(e.dst_id.graph_node_id)]
      return dst.op.name in ("nn.imcflow_qconv", "nn.imcflow_qdwconv")
    except Exception:
      # fifo_id==0 + dst "data" already uniquely identifies the activation feed;
      # fall back to that if the node lookup is unavailable.
      return True

  def _is_conv_config_send(self) -> bool:
    """True iff this SEND is the conv/dwconv CONFIG write (dst tensor_type 'config'
    feeding a qconv/qdwconv). Under STEP_FREERUN root cause #2 the imce re-issues
    RECV_CFG once per pass to reset the linebuffer (imce_codeblock ConvBlock
    ._build_structure), so this config SEND must ALSO repeat (1+N)x in lockstep. All
    OTHER config/const/weight sends stay single (this predicate is config-only)."""
    e = self.block.id if isinstance(self.block.id, TensorEdge) else None
    if e is None:
      return False
    if getattr(e.dst_id, "tensor_type", None) != "config":
      return False
    try:
      dst = CustomIDToNode()[getInnerNodeID(e.dst_id.graph_node_id)]
      return dst.op.name in ("nn.imcflow_qconv", "nn.imcflow_qdwconv")
    except Exception:
      # dst "config" uniquely identifies a config write; if the node lookup fails,
      # still treat it as the conv config (the only config edge in a single-qconv).
      return True

  def _conv_config_interleave_lines(self):
    """STEP_FREERUN config-interleave (RTL/chip wedge fix): return the C line(s) that
    re-SEND this conv's CONFIG packet, to be emitted at the START of EACH freerun data
    pass (inside the data feed's per-pass loop). This mirrors the imce, which issues
    RECV_CFG at the head of every freerun pass (imce_codeblock ConvBlock: _ConfigRecvLine
    inside the (1+N) SimpleFor). The OLD design batched all (1+N) config SENDs in the INIT
    segment before the imce drains them; with the depth-2 config RECV FIFO (params.svh)
    and INODE_SEND blocking on a full FIFO, that overflows and mutually deadlocks (imce
    stalls at OP_RECV, inode stalls in its feed loop -- fsim-confirmed at N=3). Emitting
    ONE config SEND per data pass keeps config-FIFO pending <=1, in lockstep with the
    imce's per-pass RECV_CFG.

    Returns "" when this is NOT a conv activation feed under freerun, or when the sibling
    config edge cannot be resolved (caller then leaves the INIT batch in place)."""
    if not (step_freerun_n() > 0 and self._is_conv_activation_feed()):
      return ""
    e = self.block.id[0] if isinstance(self.block.id, list) else self.block.id
    if not isinstance(e, TensorEdge):
      return ""
    # the config edge shares this data edge's dst conv node (dst tensor_type "config").
    dst_inner = getInnerNodeID(e.dst_id.graph_node_id)
    cfg_edge = None
    for te in DevConfig().TensorEdgetoInfo.keys():
      if (getattr(te.dst_id, "tensor_type", None) == "config"
          and getInnerNodeID(te.dst_id.graph_node_id) == dst_inner):
        cfg_edge = te
        break
    if cfg_edge is None:
      return ""
    cfg_info = DevConfig().get_tensor_edge_info(cfg_edge)
    cfg_db = DevConfig().CurrFuncMemLayout.get_data_block_by_edge(cfg_edge)
    if cfg_info is None or cfg_db is None:
      return ""
    cfg_fid = cfg_info.fifo_id
    cfg_pa = cfg_info.policy_info[0].address
    cfg_off = cfg_db.offset
    cfg_pkts = math.ceil(cfg_db.size / 32)  # 1 for a single-qconv
    nop = (NopLoopBlock(qconv_nop_delay_cnt()).render() + "\n" if (DevConfig().single_qconv and qconv_nop_delay_cnt() > 0) else "")
    lines = "// [STEP_FREERUN] per-pass config reload (interleaved, config-FIFO safe)\n"
    for p in range(cfg_pkts):
      lines += f"__builtin_INODE_SEND({cfg_off} + {p}*32, 0, {cfg_pa}, {cfg_fid});\n"
      lines += nop
    return lines

  def _build(self):
    recv_count = math.ceil(self.block.size / 32)
    # Power-measurement lever (IMCFLOW_STEP_FREERUN): LENGTHEN the activation feed's
    # per-packet rendezvous loop by (1+N) so it matches the imce's (1+N)x-longer row
    # loop as ONE continuous SETFLAG/STANDBY toggle stream. Scaling recv_count (not
    # wrapping self.body in a new outer loop) keeps the toggle phase monotonic across
    # the whole length -> no scalar-flag lost-wakeup deadlock at a rep seam (the
    # chip-proven failure of the old outer-wrap). Only the conv activation feed; N=0
    # -> unchanged. SimpleFor -> hardware loop; the spread/flat loops below inherit
    # the multiplied count. Nest if it exceeds 16384 elsewhere (spread loop already
    # divides by eff; recv_count*(1+N) stays a valid trip since factors compose).
    if step_freerun_n() > 0 and self._is_conv_activation_feed():
      recv_count *= (1 + step_freerun_n())
    fifo_id = self.edge_info.fifo_id
    assert fifo_id >= 0, "fifo id should be assigned to a positive id"
    next_policy_addr = self.edge_info.policy_info[0].address

    # STEP_FREERUN root cause #2: the imce re-issues RECV_CFG once per pass (1+N total:
    # 1 INIT + N freerun) to RESET the linebuffer (all_recived) so it accepts a fresh
    # H*W each pass. This config SEND must repeat (1+N)x in lockstep. Each repeat sends
    # the SAME single 32B config packet (offset 0), NOT distinct packets -- the config
    # is one packet; a distinct-address loop (var+iter*32) would read OOB garbage. So
    # emit the whole (bare) config SEND wrapped in SimpleFor(1+N) at fixed offset 0.
    # Only the conv config edge; consistency (codegen.py) scales this edge's send by
    # (1+N) to match the imce config RECV. Handled here (early return) to bypass the
    # distinct-packet loops below.
    if step_freerun_n() > 0 and self._is_conv_config_send():
      # STEP_FREERUN config-interleave (RTL/chip wedge fix): the (1+N) config reloads are
      # NO LONGER batched here in the INIT segment. Batching them ahead of the imce's
      # consumption overflows the depth-2 config RECV FIFO (INODE_SEND blocks on a full
      # FIFO) and mutually deadlocks with the imce's OP_RECV wait (fsim-confirmed at N=3).
      # Instead, ONE config SEND is emitted at the head of EACH freerun DATA pass by the
      # activation feed's _build_tiled (_conv_config_interleave_lines), in lockstep with
      # the imce's per-pass RECV_CFG -> config-FIFO pending stays <=1. Mirrors the imce,
      # which likewise suppresses its INIT-phase RECV_CFG and does all (1+N) inside the
      # freerun loop (RecvConstBlock._render + ConvBlock Option-B symmetric passes). So
      # emit NOTHING in INIT for the conv config under freerun. N=0 -> normal path below.
      self.body.add(TextBlock(
          "// [STEP_FREERUN] conv config reload moved to per-pass data feed (INIT batch "
          "suppressed for config-FIFO safety)"))
      return

    var = UniqueVar("send_data_base_address", dtype="int")
    self.body.add(TextBlock(f"{var} = {self.block.offset};"))

    # data-input SEND is preceded by a pre-send rendezvous with the receiving
    # imce (weight/const SEND is bare -> _get_presend_sync_code_str returns "").
    # Sync-granularity contract (DESIGN §2.3): emit the rendezvous once per
    # `producer_send_per_sync` packets. Default (None) == 1 == per-packet, i.e.
    # byte-identical to the previous hardcoded behavior.
    # NOTE: `iter` here is the C loop-variable NAME (a string like "i1") for
    # count>1, or the int 0 for count==1 -- it is NOT a Python integer we can do
    # arithmetic on. So for the default per-packet case we emit the handshake
    # unconditionally (as before); only a contracted value >1 wraps it in a
    # C-level `if (iter % N == 0)` guard.
    # chip_acc_measure reconcile (DESIGN §3.5 order contract): pre-send
    # rendezvous BEFORE the SEND; the single_qconv nop_delay (v2/atomic-conv
    # timing) AFTER the SEND. single_qconv off (ResNet8 / v1 multi-core) ->
    # nop_delay == "" -> byte-identical to the 934ec1001 output.
    # BUGFIX knob: knob=on (bugfix_off_mode()==False) reproduces a8af's
    # per-packet POST-send SENDER-pattern sync (SEND then _get_sync_code_str);
    # knob=off keeps the 934 pre-send rendezvous below.
    # Max-throughput feed-spread (IMCFLOW_FEED_SPREAD): the INODE_SEND fifo_id is
    # a compile-time immediate (int_INODE_SEND ImmArg<3>), so a per-packet
    # rotating fifo_id cannot be a runtime `iter % n` expression -- we UNROLL the
    # flat send loop by `eff` and emit a literal rotated fifo_id per unrolled
    # packet. `eff` is >1 ONLY for the conv activation edge (spread_fifo_n set in
    # policy_table_builder); weight/const/other SENDs have spread_fifo_n==0 ->
    # eff==1 -> the classic single flat loop (byte-identical). `eff` divides the
    # bitplane count so the unrolled packet j (j in 0..eff-1) selects the SAME
    # fifo as the matching IMCE LOAD_LB bitplane. recv_count is a multiple of the
    # 4-bitplane pixel stride, hence of eff (eff|4); a remainder tail is emitted
    # defensively.
    eff = self.edge_info.effective_spread_n(4)

    if not bugfix_off_mode():
      def send_body_with_sync(iter, var=var, next_policy_addr=next_policy_addr, fifo_id=fifo_id):
        code = f"__builtin_INODE_SEND({var} + {iter}*32, 0, {next_policy_addr}, {fifo_id});\n"
        sync_code = self._get_sync_code_str_a8af()
        if sync_code:
          code += sync_code
        return code

      if eff > 1:
        def spread_group_a8af(iter, var=var, next_policy_addr=next_policy_addr):
          body = ""
          for j in range(eff):
            fid = self.edge_info.spread_fifo_id(j, 4)
            body += f"__builtin_INODE_SEND({var} + (({iter})*{eff} + {j})*32, 0, {next_policy_addr}, {fid});\n"
            sync_code = self._get_sync_code_str_a8af()
            if sync_code:
              body += sync_code
          return body
        self.body.add(SimpleFor(recv_count // eff, spread_group_a8af))
        for r in range(recv_count - (recv_count % eff), recv_count):
          fid = self.edge_info.spread_fifo_id(r, 4)
          self.body.add(TextBlock(send_body_with_sync(r, fifo_id=fid).rstrip("\n")))
        return
      self.body.add(SimpleFor(recv_count, send_body_with_sync))
      return

    send_per_sync = getattr(self.edge_info, "producer_send_per_sync", None) or 1
    nop_delay = (NopLoopBlock(qconv_nop_delay_cnt()).render() + "\n" if (DevConfig().single_qconv and qconv_nop_delay_cnt() > 0) else "")
    def send_body_with_sync(iter, var=var, next_policy_addr=next_policy_addr,
                            fifo_id=fifo_id, send_per_sync=send_per_sync, nop_delay=nop_delay):
      code = ""
      pre = self._get_presend_sync_code_str(iter_var=iter)
      if pre:
        if send_per_sync == 1:
          code += pre                      # per-packet (default, unchanged)
        else:
          # group N packets under one handshake: guard at C level.
          guarded = "".join(f"  {ln}\n" for ln in pre.splitlines() if ln.strip())
          code += f"if (({iter}) % {send_per_sync} == 0) {{\n{guarded}}}\n"
      code += f"__builtin_INODE_SEND({var} + {iter}*32, 0, {next_policy_addr}, {fifo_id});\n"
      code += nop_delay
      return code

    if eff > 1:
      def spread_group_hc(iter, var=var, next_policy_addr=next_policy_addr,
                          send_per_sync=send_per_sync, nop_delay=nop_delay):
        body = ""
        for j in range(eff):
          fid = self.edge_info.spread_fifo_id(j, 4)
          pre = self._get_presend_sync_code_str(iter_var=f"(({iter})*{eff} + {j})")
          if pre:
            if send_per_sync == 1:
              body += pre
            else:
              guarded = "".join(f"  {ln}\n" for ln in pre.splitlines() if ln.strip())
              body += f"if ((({iter})*{eff} + {j}) % {send_per_sync} == 0) {{\n{guarded}}}\n"
          body += f"__builtin_INODE_SEND({var} + (({iter})*{eff} + {j})*32, 0, {next_policy_addr}, {fid});\n"
          body += nop_delay
        return body
      self.body.add(SimpleFor(recv_count // eff, spread_group_hc))
      for r in range(recv_count - (recv_count % eff), recv_count):
        fid = self.edge_info.spread_fifo_id(r, 4)
        self.body.add(TextBlock(send_body_with_sync(r, fifo_id=fid).rstrip("\n")))
      return
    # Silicon-SAFE lever (non-tiled path): same step-by-num_blocks unroll with
    # LITERAL phase-tokens as _build_tiled. recv_count is a Python int here so we
    # step a plain C for by num_blocks. Only fires when this SEND feeds a 2-inode
    # fused-add consumer (else None -> byte-identical legacy loop below).
    safe_nb = self.is_safe_fusedadd_send()
    if safe_nb is not None and eff == 1:
      _cval, nb = safe_nb
      safe_iv = UniqueVar(f"send_safe_iv_{fifo_id}", dtype="int")
      inner = ""
      for b in range(nb):
        pre = self._get_presend_sync_code_str(safe_block=b)
        if pre:
          inner += indent(pre.rstrip("\n"), "  ") + "\n"
        inner += (f"  __builtin_INODE_SEND({var} + ({safe_iv} + {b})*32, 0, "
                  f"{next_policy_addr}, {fifo_id});\n")
        if nop_delay:
          inner += indent(nop_delay.rstrip("\n"), "  ") + "\n"
      self.body.add(TextBlock(
          f"for (int {safe_iv} = 0; {safe_iv} < {recv_count}; "
          f"{safe_iv} += {nb}) {{ // SAFE fused-add step-by-{nb} loop\n{inner}}}"))
      return
    self.body.add(SimpleFor(recv_count, send_body_with_sync))

  def _build_tiled(self):
    # tiling_info = self.block.tiling_info
    fifo_id = self.edge_info.fifo_id
    assert fifo_id >= 0, "fifo id should be assigned to a positive id"
    next_policy_addr = self.edge_info.policy_info[0].address
    if isinstance(self.block.id, list):
      block_id = self.block.id[0]
    else:
      block_id = self.block.id
    assert isinstance(block_id, TensorEdge), "Tiled send block must be a tensor edge"
    target_edge = block_id

    cnt_addr_var = UniqueVar(f"{target_edge.simple_name()}_send_data_base_address", dtype="int", pointer_type=True)
    _send_cnt_address_block = DevConfig().MemLayout[self.builder.func_name].get_data_block_by_id(f"{target_edge.simple_name()}_cnt_base_addr")
    _send_cnt_address = _send_cnt_address_block.offset
    self.body.add(TextBlock(f"{cnt_addr_var} = (int*)({_send_cnt_address});"))

    loop_cnt_var = UniqueVar(f"{target_edge.simple_name()}_tile_loop_count", dtype="int")
    base_var = UniqueVar("send_data_base_address", dtype="int")
    self.body.add(TextBlock(f"{base_var} = {self.block.offset};"))

    # Power-measurement lever (IMCFLOW_STEP_FREERUN): the tiled feed loop trips on
    # the RUNTIME tile count cnt_addr_var[0]; scale it by (1+N) so the activation
    # feed emits (1+N)x SENDs, matching the imce's (1+N)x-longer row loop as ONE
    # continuous rendezvous stream (no new outer loop -> no scalar-flag toggle-phase
    # deadlock at a rep seam; chip-proven the outer wrap stalls). Only the conv
    # activation feed; N=0 -> unchanged.
    # STEP_FREERUN root cause #2: the feed is NOT scaled by (1+N) here. Instead the
    # WHOLE per-pass feed (reading the SAME cnt[0] input packets at offset 0..cnt-1) is
    # wrapped in an OUTER (1+N)x freerun loop below (`_freerun_passes`), so each pass
    # RE-READS the same input buffer (the imce re-configs the linebuffer per pass to
    # accept it again). A flat `cnt[0]*(1+N)` feed would instead read addresses past the
    # input buffer (OOB garbage -> X-fatal on the send path, RTL-confirmed at pass 1).
    # Using the natural cnt[0] here ALSO removes the earlier load-use hazard (no MULI on
    # the loaded count). N=0 -> single pass, byte-identical.
    # INPUT_REUSE (IMCFLOW_INPUT_REUSE, DON'T-CARE power kernel): the host stored only
    # ONE ROW of input (transform.py shrank block.size to height_offset), but the config
    # H*W (and pkt_cnts) still expect the FULL H*row_pkts SENDs so the array does H*W
    # STEPs. Emit the feed as a COMPILE-TIME NESTED loop: for H rows { for row_pkts
    # packets { SEND base + p*32 } }. The inner offset uses ONLY the inner index
    # (0..row_pkts-1), so every SEND reads within the 1-row buffer (NO OOB, NO runtime
    # modulo which INODE can't lower). Each row re-reads the SAME buffer -> DON'T-CARE OK.
    # H and row_pkts are compile-time: row_pkts = block.size/32 (1 row), total from
    # pkt_cnts, H = total/row_pkts. Only the conv activation feed; else fall through.
    if input_reuse() and self._is_conv_activation_feed() and self.block.tiling_info is not None:
      _row_pkts = math.ceil(self.block.size / 32)          # 1-row buffer packets (W*C/32)
      _total_pkts = int(self.block.tiling_info.pkt_cnts[0]) # full H*row_pkts
      assert _row_pkts > 0 and _total_pkts % _row_pkts == 0, \
        f"[INPUT_REUSE] total pkts {_total_pkts} not a multiple of row pkts {_row_pkts}"
      _h_rows = _total_pkts // _row_pkts                    # config H
      _nop = NopLoopBlock(qconv_nop_delay_cnt()).render() + "\n" if (DevConfig().single_qconv and qconv_nop_delay_cnt() > 0) else ""
      # FLAG-FREE feed (input_reuse_feed_flagfree, default ON): the per-packet
      # flag-1 rendezvous writes syn_reg[this_inode] on every packet and thereby
      # clobbers the END-phase barrier flag (SET_FLAG 254/255) this same inode set
      # -- a slow peer still in STANDBY(this_inode, 254/255) then lost-wakes. Since
      # the output is DON'T-CARE we drop the handshake and PACE the feed with a
      # purely-local INODE nop delay instead (imce free-runs its LOAD_LB/STEP; the
      # nop keeps the feed from overrunning the depth-2 data RECV fifo). The barrier
      # flag is never touched by the feed -> no lost-wakeup, stock barrier is fine.
      _flagfree = input_reuse_feed_flagfree()
      _pace = (NopLoopBlock(input_reuse_feed_pace_nops()).render() + "\n"
               if (_flagfree and input_reuse_feed_pace_nops() > 0) else "")
      self.body.add(TextBlock(f"{base_var} = {self.block.offset};"))
      def _inner_pkt(p, base=base_var, pa=next_policy_addr, fid=fifo_id, nop=_nop,
                     flagfree=_flagfree, pace=_pace):
        # inner index p in [0, row_pkts) -> offset p*32 stays inside the 1-row buffer.
        if flagfree:
          # flag-free: local pacing nop, then SEND (no syn_reg write).
          return pace + f"__builtin_INODE_SEND({base} + {p}*32, 0, {pa}, {fid});\n" + nop
        pre = self._get_presend_sync_code_str(iter_var=p)
        body = (pre or "") + f"__builtin_INODE_SEND({base} + {p}*32, 0, {pa}, {fid});\n" + nop
        return body
      # H rows, each re-sends the full row (row_pkts packets, offset reset to 0).
      _inner_loop = SimpleFor(_row_pkts, _inner_pkt, f"{target_edge.simple_name()}_reuse_row_pkts")
      self.body.add(SimpleFor(_h_rows, _inner_loop, f"{target_edge.simple_name()}_reuse_rows"))
      return

    self.body.add(TextBlock(f"{loop_cnt_var} = {cnt_addr_var}[0];"))
    self.body.add(TextBlock(f"__asm__ volatile(\"nop\");")) # BUGFIX_LOAD_USE_HAZARD
    _freerun_passes = (1 + step_freerun_n()) if (step_freerun_n() > 0
                                                 and self._is_conv_activation_feed()) else 1

    # STEP_FREERUN config-interleave (RTL/chip wedge fix): the conv config reload is
    # emitted at the HEAD of each freerun pass (SequentialBlock[config_send, data_feed]),
    # in lockstep with the imce's per-pass RECV_CFG, so the depth-2 config RECV FIFO never
    # holds >1 pending packet. The INIT-phase config batch is suppressed (_build). When
    # freerun is off / this isn't the activation feed, the config line is "" and this is a
    # no-op wrapper (byte-identical). See _conv_config_interleave_lines for the rationale.
    _cfg_reload_lines = self._conv_config_interleave_lines()
    def _wrap_freerun_with_config(per_pass_body):
      if _freerun_passes <= 1 or not _cfg_reload_lines:
        return _wrap_freerun_passes(per_pass_body, _freerun_passes)
      one_pass = SequentialBlock([TextBlock(_cfg_reload_lines.rstrip("\n")), per_pass_body])
      return _wrap_freerun_passes(one_pass, _freerun_passes)

    # Max-throughput feed-spread (IMCFLOW_FEED_SPREAD): the tiled data-input SEND
    # must rotate a LITERAL fifo_id per packet (int_INODE_SEND fifo_id is an
    # ImmArg). The INODE ISA cannot lower arithmetic on the loop index for the
    # fifo select (no `and`, no `/`, no unsigned compare -- those fail INODE
    # instruction selection), and fully unrolling all packets overflows the small
    # inode imem. The INODE-safe form (verified to compile) is a STEP-BY-eff
    # hardware loop `for (i=0; i<var6; i += eff)` whose body emits `eff` unrolled
    # SENDs at constant offsets `(i+j)*32` (ADDI+shift, supported) with literal
    # rotated fifo_ids. This needs NO division and preserves the runtime tile
    # count. `var6` is a whole-pixel multiple (4 bitplanes/pixel) and eff | 4, so
    # the step evenly covers all packets. eff==1 (flag off / non-conv edge) keeps
    # the original per-packet loop byte-identical.
    # Max-throughput lever (IMCFLOW_FEED_PREFETCH): when prefetch is active on
    # this conv activation edge, step the tiled SEND by width=P*4 and unroll that
    # many SENDs over fifos 0..width-1, so the inode pushes P pixels' worth of
    # bitplanes into P*4 distinct RECV fifos ahead of the IMCE's consumption. This
    # matches the IMCE col_group P-pixel unroll (fifo (p*4+b)). Falls back to the
    # plain feed-spread width (effective_spread_n) when prefetch is off.
    _pf = self.edge_info.prefetch_group(4)
    if _pf is not None:
      eff = _pf[1]
      _prefetch_fids = [j % 8 for j in range(eff)]
    else:
      eff = self.edge_info.effective_spread_n(4)
      _prefetch_fids = None
    spread_iv = UniqueVar(f"{target_edge.simple_name()}_spread_iv", dtype="int") if eff > 1 else None

    def _spread_fid(j):
      return _prefetch_fids[j] if _prefetch_fids is not None else self.edge_info.spread_fifo_id(j, 4)

    def _spread_loop(inner_lines):
      return (f"for (int {spread_iv} = 0; {spread_iv} < {loop_cnt_var}; "
              f"{spread_iv} += {eff}) {{ // feed-spread group loop\n"
              f"{inner_lines}}}")

    # BUGFIX knob: knob=on (bugfix_off_mode()==False) reproduces a8af's
    # per-packet POST-send SENDER-pattern sync; knob=off keeps the 934 pre-send
    # rendezvous below.
    if not bugfix_off_mode():
      def send_body_with_sync(iter, base_addr_var=base_var, policy_addr=next_policy_addr, fid=fifo_id):
        code = f"__builtin_INODE_SEND({base_addr_var} + {iter}*32, 0, {policy_addr}, {fid});\n"
        sync_code = self._get_sync_code_str_a8af()
        if sync_code:
          code += sync_code
        return code

      if eff > 1:
        # a8af POST-send sync (empty "" for the plain conv data edge) re-derived
        # per unrolled SEND to preserve its per-packet semantics.
        inner = ""
        for j in range(eff):
          fid = _spread_fid(j)
          inner += (f"  __builtin_INODE_SEND({base_var} + ({spread_iv} + {j})*32, 0, "
                    f"{next_policy_addr}, {fid});\n")
          sync_code = self._get_sync_code_str_a8af()
          if sync_code:
            inner += indent(sync_code.rstrip("\n"), "  ") + "\n"
        self.body.add(TextBlock(_spread_loop(inner)))
        return
      self.body.add(SimpleFor(loop_cnt_var, send_body_with_sync))
      return

    # handcraft: data-input SEND is preceded (per-packet) by a pre-send
    # rendezvous with the receiving imce. weight/const SEND is bare.
    # chip_acc_measure reconcile (DESIGN §3.5): pre-send rendezvous BEFORE the
    # SEND, single_qconv nop_delay AFTER (== "" when single_qconv off).
    nop_delay = (NopLoopBlock(qconv_nop_delay_cnt()).render() + "\n" if (DevConfig().single_qconv and qconv_nop_delay_cnt() > 0) else "")
    def send_body_with_sync(iter, base_addr_var=base_var, policy_addr=next_policy_addr, fid=fifo_id, nop_delay=nop_delay):
      code = ""
      pre = self._get_presend_sync_code_str(iter_var=iter)
      if pre:
        code += pre
      code += f"__builtin_INODE_SEND({base_addr_var} + {iter}*32, 0, {policy_addr}, {fid});\n"
      code += nop_delay
      return code

    # Silicon-SAFE lever: this data edge feeds a 2-inode fused-add consumer.
    # INODE cannot lower a runtime `iter % num_blocks` flag value (backend
    # 'Cannot select and'), so emit a STEP-BY-num_blocks hardware loop with
    # num_blocks UNROLLED bodies, each carrying LITERAL phase-tokens (block b:
    # base+2b / base+2b+1). Mirrors the feed-spread step-by-eff idiom above.
    # eff==1 always here (fused-add data isn't feed-spread). Word (i+b) maps to
    # consumer block b -- exactly the consumer's per-block unroll order.
    safe_nb = self.is_safe_fusedadd_send()
    if safe_nb is not None and eff == 1:
      _cval, nb = safe_nb
      safe_iv = UniqueVar(f"{target_edge.simple_name()}_safe_iv", dtype="int")
      inner = ""
      for b in range(nb):
        pre = self._get_presend_sync_code_str(safe_block=b)
        if pre:
          inner += indent(pre.rstrip("\n"), "  ") + "\n"
        inner += (f"  __builtin_INODE_SEND({base_var} + ({safe_iv} + {b})*32, 0, "
                  f"{next_policy_addr}, {fifo_id});\n")
        if nop_delay:
          inner += indent(nop_delay.rstrip("\n"), "  ") + "\n"
      self.body.add(TextBlock(
          f"for (int {safe_iv} = 0; {safe_iv} < {loop_cnt_var}; "
          f"{safe_iv} += {nb}) {{ // SAFE fused-add step-by-{nb} loop\n{inner}}}"))
      return

    if eff > 1:
      inner = ""
      # Max-throughput lever (IMCFLOW_FEED_SYNC_PER_PIXEL): emit ONE pre-send
      # rendezvous before the whole `eff`-packet group (per pixel) instead of one
      # per packet, matching the imce's single per-pixel LOAD_LB window (both 1:1).
      # DON'T-CARE output; safe when FEED_SPREAD>=eff (each packet in its own
      # depth-2 fifo, so all eff fit in flight before the single drain). Default
      # OFF -> per-packet rendezvous unchanged.
      _sync_per_pixel = feed_sync_per_pixel()
      if _sync_per_pixel:
        pre = self._get_presend_sync_code_str(iter_var=f"({spread_iv})")
        if pre:
          inner += indent(pre.rstrip("\n"), "  ") + "\n"
      for j in range(eff):
        fid = self.edge_info.spread_fifo_id(j, 4)
        if not _sync_per_pixel:
          pre = self._get_presend_sync_code_str(iter_var=f"({spread_iv} + {j})")
          if pre:
            inner += indent(pre.rstrip("\n"), "  ") + "\n"
        inner += (f"  __builtin_INODE_SEND({base_var} + ({spread_iv} + {j})*32, 0, "
                  f"{next_policy_addr}, {fid});\n")
        if nop_delay:
          inner += indent(nop_delay.rstrip("\n"), "  ") + "\n"
      # STEP_FREERUN: wrap the per-pass spread feed (reads offset 0..cnt-1) in a
      # (1+N)x outer loop so each pass RE-READS the same input buffer (matching the
      # imce's per-pass linebuffer reset). spread_iv resets per pass -> no OOB read.
      # Nest the (1+N) outer loop into <=16384 factors (INODE 14-bit hw-loop limit),
      # same step_freerun_factors(1+N) the imce uses -> identical pass count both sides.
      self.body.add(_wrap_freerun_with_config(TextBlock(_spread_loop(inner))))
      return
    self.body.add(_wrap_freerun_with_config(SimpleFor(loop_cnt_var, send_body_with_sync)))

  def is_safe_fusedadd_send(self):
    """SAFE lever: return (consumer_value, num_blocks) if THIS SendBlock's data
    edge feeds a 2-inode fused-add consumer (so its per-word SEND loop must be
    unrolled by num_blocks with literal phase-tokens), else None. Used by the
    caller to switch the flat SEND loop into a step-by-num_blocks unrolled loop
    (INODE cannot lower a runtime `iter % nb` flag value -- verified: backend
    'Cannot select and' -- so tokens must be compile-time literals)."""
    from tvm.contrib.imcflow import multiblock_fusedadd_safe
    if not multiblock_fusedadd_safe():
      return None
    if not hasattr(self.builder, 'pair_manager') or self.builder.pair_manager is None:
      return None
    edge_or_edges = self._get_edge()
    if edge_or_edges is None:
      return None
    edges = edge_or_edges if isinstance(edge_or_edges, list) else [edge_or_edges]
    pm = self.builder.pair_manager
    for e in edges:
      nb = pm.fusedadd_consumer_num_blocks(e)
      if nb is not None:
        return nb  # (consumer_value, num_blocks)
    return None

  def _get_presend_sync_code_str(self, iter_var=None, safe_block=None):
    """Get PRE-send rendezvous code (handcraft, SENDER side for data input).

    handcraft inode_0_0 data-input SEND (per-packet):
        STANDBY(imce, 1); SET_FLAG(1); STANDBY(imce, 0); SET_FLAG(0); SEND
    Only emitted for inode -> imce *data input* edges whose receiver is the
    main-pipeline imce (plain-int dst id, e.g. imce_0_2). weight/const SENDs and
    the fused/composite receiver (imce_0_1) get NO pre-send (bare).

    `iter_var` is the C loop-variable NAME (unused by the SAFE path -- kept for
    the legacy signature). `safe_block` is the LITERAL (Python int) block index
    b in 0..num_blocks-1 for the SAFE unrolled path; when not None, the SAFE
    branch emits its monotonic phase-token with literal values base+2b/base+2b+1
    (INODE requires literal flag values).
    """
    if not hasattr(self.builder, 'pair_manager') or self.builder.pair_manager is None:
      return ""

    edge_or_edges = self._get_edge()
    if edge_or_edges is None:
      return ""
    edges = edge_or_edges if isinstance(edge_or_edges, list) else [edge_or_edges]

    pm = self.builder.pair_manager
    # Silicon-deadlock lever (IMCFLOW_MULTIBLOCK_FUSEDADD_BARE): drop this
    # inode-side 4-phase pre-send handshake for a MULTI-BLOCK 2-inode fused-add
    # receiver (region3 imce_1_1). Kept in LOCKSTEP with the consumer window
    # (RecvSendWrapper drops the matching SETFLAG window under the same lever);
    # if only one side were bared the other would STANDBY forever. region2
    # imce_1_3 (num_blocks==1) fails the predicate -> handshake preserved.
    from tvm.contrib.imcflow import (multiblock_fusedadd_bare,
                                     multiblock_fusedadd_safe, SAFE_TOKEN_BASE)
    if multiblock_fusedadd_bare() and any(
        pm.is_multiblock_fusedadd_input_edge(e) for e in edges):
      return ""

    # Silicon-SAFE lever (IMCFLOW_MULTIBLOCK_FUSEDADD_SAFE): REPLACE the per-word
    # 4-phase 1/0 toggle with a monotonic phase-token, fully-interlocked,
    # order-independent handshake keyed on the per-word block index
    # b = (iter_var % num_blocks). Producer Pk, block b:
    #     STANDBY(C, 2b+1); SET_FLAG(2b+1); STANDBY(C, 2b+2); SEND
    # (matching consumer sets 2b+1 to invite, waits both producers' 2b+1, sets
    # 2b+2 to release, then RECVs -- the RECV is the ack that lets C advance).
    # Distinct token values per block (no repeated 1->0->1 edge to collapse) +
    # RECV-as-ack interlock => lost-wakeup-proof on silicon AND simv. See
    # multiblock_fusedadd_safe() docstring / DESIGN_region3_fusedadd_redesign.md.
    if multiblock_fusedadd_safe() and safe_block is not None:
      safe_targets = []  # (consumer_value, num_blocks)
      for e in edges:
        nb = pm.fusedadd_consumer_num_blocks(e)
        if nb is not None and nb not in safe_targets:
          safe_targets.append(nb)
      if safe_targets:
        b = int(safe_block)
        rdy = SAFE_TOKEN_BASE + 2 * b        # literal (INODE requires literal flag)
        go = SAFE_TOKEN_BASE + 2 * b + 1
        lines = []
        for cval, _nb in sorted(safe_targets):
          lines.append(f"__builtin_INODE_STANDBY({cval}, {rdy}); // SAFE: wait consumer ready(block {b})")
          lines.append(f"__builtin_INODE_SET_FLAG({rdy});        // SAFE: announce ready(block {b})")
          lines.append(f"__builtin_INODE_STANDBY({cval}, {go});  // SAFE: wait consumer GO(block {b})")
        return "\n".join(lines) + "\n"
    # IMCFLOW_PACK_BN_MINMAX capacity fix: the packed-conv EXTRA post-op consts
    # (BN fused_scale/bias + multiply/add scale rhs) are otherwise BARE and
    # overflow the NoC send FIFO before the (pipeline-blocked) imce drains them
    # -> region2 hard deadlock. Pace them with the same per-word flag-1
    # rendezvous the data input uses, keyed on the receiving imce derived from
    # the edge (const edges are unpaired). Lockstep with RecvConstBlock's window
    # (both gate on pair_manager.is_packed_postop_const_edge). Lever OFF ->
    # predicate False -> bare (byte-identical).
    for e in edges:
      eps = pm.packed_postop_const_endpoints(e)
      if eps is not None:
        _inode_hw, imce_hw = eps
        return (
          f"__builtin_INODE_STANDBY({imce_hw.value}, 1); // pack-const sync with {imce_hw.name}\n"
          f"__builtin_INODE_SET_FLAG(1);\n"
          f"__builtin_INODE_STANDBY({imce_hw.value}, 0);\n"
          f"__builtin_INODE_SET_FLAG(0);\n"
        )

    # Rendezvous only with data-input receivers (plain-int dst) of this SEND.
    # Each edge maps to exactly ONE receiver (its own dst hw node); do NOT pull
    # in the other multicast receivers (e.g. the fused imce_0_1 whose dst is a
    # tuple must stay bare).
    target_imces = []
    for e in edges:
      if not pm.is_inode_data_input_recv(e):
        continue
      rnode = pm._get_hw_node(e.dst_id)
      if isinstance(rnode, tuple):
        rnode = rnode[0]
      if rnode is not None and rnode.is_imce() and rnode not in target_imces:
        target_imces.append(rnode)

    if not target_imces:
      return ""

    target_imces.sort(key=lambda x: x.value)

    sync_lines = []
    for rnode in target_imces:
      sync_lines.append(f"__builtin_INODE_STANDBY({rnode.value}, 1); // sync with {rnode.name} before SEND")
      sync_lines.append(f"__builtin_INODE_SET_FLAG(1);")
      sync_lines.append(f"__builtin_INODE_STANDBY({rnode.value}, 0);")
      sync_lines.append(f"__builtin_INODE_SET_FLAG(0);")
    return "\n".join(sync_lines) + "\n"

  def _get_sync_code_str_a8af(self):
    """a8af POST-send SENDER-pattern sync (knob=on fallback).

    SENDER: STANDBY(receiver, uuid); SETFLAG(uuid); STANDBY(receiver, 0); SETFLAG(0)
    emitted AFTER each SEND. Verbatim from a8af _get_sync_code_str (debug prints
    preserved for byte-identity of stdout side-effects only; they do not affect
    the emitted .cpp).
    """
    if not hasattr(self.builder, 'pair_manager') or self.builder.pair_manager is None:
      print(f"[DEBUG _get_sync_code_str] No pair_manager")
      return ""

    edge_or_edges = self._get_edge()
    if edge_or_edges is None:
      print(f"[DEBUG _get_sync_code_str] Edge is None")
      return ""

    if isinstance(edge_or_edges, list):
      edge = None
      for e in edge_or_edges:
        if e.split_idx is not None:
          edge = e
          break
      if edge is None:
        edge = edge_or_edges[0] if edge_or_edges else None
      print(f"[DEBUG _get_sync_code_str] Multicast: selected edge={edge} from {edge_or_edges}")
    else:
      edge = edge_or_edges

    if edge is None:
      print(f"[DEBUG _get_sync_code_str] Edge is None after multicast handling")
      return ""

    pair = self.builder.pair_manager.get_pair(edge)
    if pair is None:
      print(f"[DEBUG _get_sync_code_str] No pair for edge: {edge}")
      return ""

    if len(pair.receiver_nodes) == 1 and pair.sender_node in pair.receiver_nodes:
      print(f"[DEBUG _get_sync_code_str] Skipping sync: sender==receiver for edge={edge}")
      return ""

    current_node = self.edge_info.policy_info[0].router_id

    print(f"[DEBUG _get_sync_code_str] edge={edge}")
    print(f"[DEBUG _get_sync_code_str] pair.uuid={pair.uuid}, pair.sender_node={pair.sender_node}, pair.receiver_nodes={pair.receiver_nodes}")
    print(f"[DEBUG _get_sync_code_str] pair.all_nodes={pair.all_nodes}")
    print(f"[DEBUG _get_sync_code_str] current_node={current_node} (type={type(current_node)})")

    sync_lines = []
    for node in pair.all_nodes:
      print(f"[DEBUG _get_sync_code_str] Checking node={node}, node != current_node = {node != current_node}")
      if node != current_node:
        sync_lines.append(f"__builtin_INODE_STANDBY({node.value}, {pair.uuid});")
    sync_lines.append(f"__builtin_INODE_SET_FLAG({pair.uuid});")
    for node in pair.all_nodes:
      if node != current_node:
        sync_lines.append(f"__builtin_INODE_STANDBY({node.value}, 0);")
    sync_lines.append(f"__builtin_INODE_SET_FLAG(0);")

    print(f"[DEBUG _get_sync_code_str] Generated sync_lines: {sync_lines}")
    return "\n".join(sync_lines) + "\n"

  def _add_sync_after_send(self):
    """Add synchronization block after send operation"""
    if not hasattr(self.builder, 'pair_manager') or self.builder.pair_manager is None:
      return  # No pair manager, skip sync

    # Get the tensor edge for this send block
    edge = self._get_edge()
    if edge is None:
      return

    pair = self.builder.pair_manager.get_pair(edge)
    if pair is None:
      return  # No sync needed for this edge

    # Get current node from edge_info
    current_node = self.edge_info.policy_info[0].router_id

    # Add sync block
    sync_annotation = f"sync after send: uuid={pair.uuid}, edge={edge}"
    sync_block = SyncPairINode(current_node, pair.all_nodes, pair.uuid, sync_annotation)
    self.body.add(sync_block)

  def _get_edge(self):
    """Get the tensor edge associated with this send block

    Returns:
      - List of TensorEdge if multicast (block.id is list)
      - Single TensorEdge otherwise
    """
    if isinstance(self.block.id, list):
      return self.block.id  # Return list for multicast handling
    if isinstance(self.block.id, TensorEdge):
      return self.block.id
    return None

class SendBlockInterleaved(InodeCodeBlock):
  """ Code block for sending data from given fifo id """

  def __init__(self, builder, blocks: List[DataBlock], edge_infos: List[TensorEdgeInfo], annotation: str = ""):
    super().__init__(annotation)
    assert len(blocks) == len(edge_infos), "# of blocks and fifo_ids must be equal"
    self.builder = builder
    self.blocks = blocks
    self.edge_infos = edge_infos
    self._build()

  def _build(self):
    # Collect block info
    info_list = []
    for block, edge_info in zip(self.blocks, self.edge_infos):
      recv_count = math.ceil(block.size / 32)
      next_policy_addr = edge_info.policy_info[0].address
      fifo_id = edge_info.fifo_id
      info_list.append({
          'owner': block,
          'recv_count': recv_count,
          'offset': block.offset,
          'policy': next_policy_addr,
          'fid': fifo_id,
          'edge_info': edge_info
      })

    # Sort unique recv_counts to define intervals
    counts = sorted(list(set(x['recv_count'] for x in info_list)))

    current_base = 0
    for limit in counts:
      duration = limit - current_base
      if duration <= 0:
        continue

      # Identify blocks active in this interval
      active_infos = [x for x in info_list if x['recv_count'] > current_base]

      # Generate loop for this interval.
      # BUGFIX knob: knob=off appends "\n"+nop_delay after the SEND (934); knob=on
      # (bugfix_off_mode()==False) reproduces a8af's bare SEND line (no trailing
      # newline / nop_delay).
      if bugfix_off_mode():
        nop_delay = (NopLoopBlock(qconv_nop_delay_cnt()).render() + "\n" if (DevConfig().single_qconv and qconv_nop_delay_cnt() > 0) else "")
        _suffix = "\n"
      else:
        nop_delay = ""
        _suffix = ""
      for x in active_infos:
        # var = UniqueVar("send_offset_address", dtype="int")
        var = UniqueVar(x['owner'], dtype="int")
        self.body.add(TextBlock(f"{var} = {x['offset']};"))
        self.body.add(SimpleFor(duration,
            lambda iter, base=current_base, offset_var=var, policy=x['policy'], fid=x['fid'], nop_delay=nop_delay, _suffix=_suffix:
              f"__builtin_INODE_SEND({offset_var} + ({f'{base} + {iter}' if base > 0 else iter})*32, 0, {policy}, {fid});" + _suffix + nop_delay))

        # Add sync after each send in interleaved block
        self._add_sync_for_edge(x['owner'], x['edge_info'])

      current_base = limit

  def _add_sync_for_edge(self, block: DataBlock, edge_info: TensorEdgeInfo):
    """Add synchronization for a specific edge in interleaved send"""
    if not hasattr(self.builder, 'pair_manager') or self.builder.pair_manager is None:
      return

    # Get edge from block
    if isinstance(block.id, list):
      edge = block.id[0] if len(block.id) > 0 else None
    else:
      edge = block.id if isinstance(block.id, TensorEdge) else None

    if edge is None:
      return

    pair = self.builder.pair_manager.get_pair(edge)
    if pair is None:
      return

    # Get current node
    current_node = edge_info.policy_info[0].router_id

    # Add sync block
    sync_annotation = f"sync after interleaved send: uuid={pair.uuid}, edge={edge}"
    sync_block = SyncPairINode(current_node, pair.all_nodes, pair.uuid, sync_annotation)
    self.body.add(sync_block)


class IMCEComputeBlock(InodeCodeBlock):
  """ Code block for sending data from given fifo id """

  def __init__(self, policy_addr, annotation: str = ""):
    super().__init__(annotation)
    self.policy_addr = policy_addr
    self._build()

  def _build(self):
    self.body.add(TextBlock(f"__builtin_INODE_IMCE_COMPUTE(0, {self.policy_addr});"))


class StandbyAndIntrtBlock(InodeCodeBlock):
  def __init__(self, node_ids: List[NodeID], annotation: str = ""):
    super().__init__(annotation)
    self.node_ids = node_ids
    self._build()

  def _build(self):
    for node in self.node_ids:
      self.body.add(TextBlock(f"__builtin_INODE_STANDBY({node.value}, 1);"))

    nops = " ".join([f"\"nop\\n\"" for _ in range(len(self.node_ids))])
    self.body.add(TextBlock(f"__asm__ volatile({nops});"))

    self.body.add(TextBlock(f"__builtin_INODE_DONE();"))
    self.body.add(TextBlock(f"__builtin_INODE_INTRT(0);"))
    self.body.add(TextBlock(f"__builtin_INODE_HALT();"))

class DoneAndIntrtBlock(InodeCodeBlock):
  def __init__(self, annotation: str = ""):
    super().__init__(annotation)
    self._build()

  def _build(self):
    self.body.add(TextBlock(f"__builtin_INODE_DONE();"))
    self.body.add(TextBlock(f"__builtin_INODE_INTRT(0);"))

class SyncAllINodes(InodeCodeBlock):
  # Sense-reversing barrier support (IMCFLOW_PACK_BN_MINMAX): the stock barrier
  # writes a REUSED constant flag (255) then clears it to 0. Under packing the
  # per-inode workload is redistributed, so inodes arrive at the barrier with
  # more skew; a fast inode can SET_FLAG(255) -> ... -> SET_FLAG(0) and re-enter
  # the NEXT barrier's SET_FLAG(255) before a slow inode's STANDBY samples the
  # first one -> the slow inode waits on a 255 that was cleared -> lost wakeup
  # (flags are a persistent LEVEL sampled by STANDBY, not an edge; controller.sv).
  #
  # Fix: give consecutive barriers ALTERNATING sense values (254 <-> 255) and
  # DROP the clear-to-0. Because the flag register is persistent, a slow inode
  # still sees the previous barrier's value latched until it samples it; a fast
  # inode re-entering the next barrier writes the OTHER value, so it can never
  # erase the value a straggler is still waiting on (a straggler is at most one
  # barrier behind: it cannot pass barrier k until every peer reached k, so no
  # peer can be at k+2 while it is at k). 254/255 are reserved for this: pair
  # UUIDs are capped at 253 in SendRecvPairManager when the lever is on.
  #
  # A module-level program-order counter gives every barrier instance (across
  # all inodes at the same logical barrier) the SAME sense, because codegen
  # emits the 4 per-inode SyncAllINodes for one logical barrier consecutively
  # -- but to be robust to interleaving we derive the sense from an EXPLICIT
  # per-logical-barrier index passed by the caller, defaulting to the counter.
  _sense_counter = 0

  # Barrier flag window. The 2-value (254/255) alternation is only lost-wakeup-safe
  # when the module-order counter increments EXACTLY once per logical barrier and in
  # PROGRAM order. In practice the counter is advanced from several codegen sites
  # (sync_inrt_clear called both inside initialize() and again after finalize(), the
  # conditional serialize_imcu gate, ...) whose call order does NOT always match the
  # final CodePhase emission order -> two ADJACENT barriers can get the SAME value
  # (observed: 254,255,254,254,255 -> the 4th barrier repeats 254, so a straggler in
  # STANDBY(x,254) at barrier 3 cannot tell barrier 4 apart -> lost-wakeup, exactly
  # the INPUT_REUSE END-barrier hang). Widening the window to N distinct values makes
  # a same-value COLLISION impossible unless two barriers are >=N apart -- and a
  # straggler is at most ONE barrier behind, so any N>=2 that never repeats
  # consecutively suffices. We use a 6-value window (250..255) under INPUT_REUSE so
  # even a counter that is a few steps out of program-order never yields an adjacent
  # duplicate. pair UUIDs are lowered to <=249 in SendRecvPairManager to match.
  # INPUT_REUSE uses a 2-phase (arrive, ack) barrier drawn from a WIDE window so
  # that no two barriers within the worst-case straggler distance ever reuse a
  # flag pair. The feed inode runs ~6x longer than an idle inode (4096-packet
  # feed), so a straggler can be several barriers behind; with only 3 pairs
  # (250/251, 252/253, 254/255) barrier N and N+3 collided -> lost-wakeup. Because
  # the INPUT_REUSE feed is FLAG-FREE, pair UUIDs 1..199 are entirely unused here,
  # so we take a large barrier window 200..249 = 25 disjoint (arrive, ack) pairs.
  # 25 pairs >> the ~10 barriers per inode, so NO barrier ever reuses a pair within
  # one region program -> the 2-phase barrier is collision-free regardless of skew.
  _SENSE_LO_WIDE = 200   # 200..249 -> 25 (arrive,ack) pairs (INPUT_REUSE)
  _SENSE_WIDE_PAIRS = 25
  _SENSE_LO_NARROW = 254  # 254..255 -> 2 (pack_bn_minmax legacy)

  @classmethod
  def _sense_span(cls):
    from tvm.contrib.imcflow import input_reuse as _ir
    return (cls._SENSE_LO_WIDE, 6) if _ir() else (cls._SENSE_LO_NARROW, 2)

  @classmethod
  def next_sense(cls):
    """Advance and return the next barrier sense token.

    Legacy (pack_bn_minmax): a single alternating value 254/255 (1-phase barrier).

    INPUT_REUSE: a (arrive, ack) PAIR drawn from the 250..255 window, cycling
    through 3 disjoint pairs (250,251)/(252,253)/(254,255). The 2-phase barrier
    (SyncAllINodes._build tuple branch) needs two distinct flags per barrier, and
    cycling 3 pairs means an adjacent barrier never reuses either value -- so even
    with an out-of-program-order counter no two neighbouring barriers collide.
    Returns a tuple under INPUT_REUSE, a scalar otherwise."""
    from tvm.contrib.imcflow import input_reuse as _ir
    if _ir():
      pair = cls._sense_counter % cls._SENSE_WIDE_PAIRS  # 25 disjoint pairs
      cls._sense_counter += 1
      lo = cls._SENSE_LO_WIDE + 2 * pair     # 200, 202, ..., 248
      return (lo, lo + 1)                    # (arrive, ack)
    lo, span = cls._sense_span()
    v = lo + (cls._sense_counter % span)
    cls._sense_counter += 1
    return v

  @classmethod
  def reset_sense(cls):
    cls._sense_counter = 0

  def __init__(self, node_id : NodeID, annotation: str = "", sense: int = None,
               participants=None):
    super().__init__(annotation)
    self.node_id = node_id
    # sense=None -> stock behavior (255 + clear-to-0), byte-identical to before.
    # sense=254/255 -> sense-reversing barrier (no clear), used under packing.
    self.sense = sense
    # participants: the inodes that join THIS barrier. Default = all inodes (legacy
    # all-inode rendezvous). Under INPUT_REUSE the caller passes only the
    # participating (feed/collect) inodes so idle spectator inodes neither wait on
    # nor are waited on -> removes the skew that lost-wakes the barrier. If this
    # node is not itself a participant, emit nothing (it will HALT on its own).
    self.participants = list(participants) if participants is not None else list(NodeID.inodes())
    self._build()

  def _build(self):
    # Non-participant inode: skip this barrier entirely (it neither raises nor
    # waits on any flag for this rendezvous). Used under INPUT_REUSE for idle
    # spectator inodes so they don't inject skew.
    if self.node_id not in self.participants:
      return
    if self.sense is None:
      # Stock barrier: reused UUID=255 + clear-to-0. Byte-identical to the
      # pre-fix codegen (lever OFF / non-packing path).
      INODE_SYNC_UUID = 255
      self.body.add(TextBlock(f"__builtin_INODE_SET_FLAG({INODE_SYNC_UUID});"))
      for node in self.participants:
        if node != self.node_id:
          self.body.add(TextBlock(f"__builtin_INODE_STANDBY({node.value}, {INODE_SYNC_UUID});"))
      nops = " ".join([f"\"nop\\n\"" for _ in range(len(self.participants))])
      self.body.add(TextBlock(f"__asm__ volatile({nops});"))
      self.body.add(TextBlock(f"__builtin_INODE_SET_FLAG(0);"))
    elif isinstance(self.sense, tuple):
      # TWO-PHASE (arrive + ack) barrier: needed when the inter-inode SKEW can
      # exceed the spacing between consecutive barriers (INPUT_REUSE: the feed
      # inode races through several END barriers in ~350ns while an idle inode is
      # still entering the first STANDBY -> the fast inode overwrites its own flag
      # to the NEXT barrier's value before the straggler samples the current one,
      # so a single-phase sense-reversing barrier ALSO lost-wakes). A 1-phase
      # barrier only guarantees "everyone ARRIVED"; the fast inode is then free to
      # run ahead. The 2-phase barrier additionally guarantees "everyone SAW that
      # everyone arrived" before ANY inode leaves, so no inode can advance (and
      # clobber its flag) until every peer has latched this barrier's arrive value.
      #   Phase 1 (arrive): SET_FLAG(a); STANDBY(peer,a) for all  -> all arrived
      #   Phase 2 (ack):    SET_FLAG(k); STANDBY(peer,k) for all  -> all saw arrive
      # Distinct a/k per barrier (from the widened sense window) so neither phase
      # value collides with an adjacent barrier's. This is the only variant that
      # survives unbounded skew; used under INPUT_REUSE.
      s_arrive, s_ack = self.sense
      self.body.add(TextBlock(f"__builtin_INODE_SET_FLAG({s_arrive});"))
      for node in self.participants:
        if node != self.node_id:
          self.body.add(TextBlock(f"__builtin_INODE_STANDBY({node.value}, {s_arrive});"))
      self.body.add(TextBlock(f"__builtin_INODE_SET_FLAG({s_ack});"))
      for node in self.participants:
        if node != self.node_id:
          self.body.add(TextBlock(f"__builtin_INODE_STANDBY({node.value}, {s_ack});"))
      nops = " ".join([f"\"nop\\n\"" for _ in range(2 * len(self.participants))])
      self.body.add(TextBlock(f"__asm__ volatile({nops});"))
    else:
      # Sense-reversing barrier: write this barrier's sense, wait for all peers
      # to present it, and DO NOT clear it (persistence protects stragglers).
      sense = self.sense
      self.body.add(TextBlock(f"__builtin_INODE_SET_FLAG({sense});"))
      for node in self.participants:
        if node != self.node_id:
          self.body.add(TextBlock(f"__builtin_INODE_STANDBY({node.value}, {sense});"))
      nops = " ".join([f"\"nop\\n\"" for _ in range(len(self.participants))])
      self.body.add(TextBlock(f"__asm__ volatile({nops});"))

class Standby(InodeCodeBlock):
  def __init__(self, node_ids: List[NodeID], annotation: str = ""):
    super().__init__(annotation)
    self.node_ids = node_ids
    self._build()

  def _build(self):
    # Use UUID=255 for INODE-to-INODE sync to avoid conflict with SendRecvPairManager UUIDs (1-254)
    INODE_SYNC_UUID = 255
    for node in self.node_ids:
      self.body.add(TextBlock(f"__builtin_INODE_STANDBY({node.value}, {INODE_SYNC_UUID});"))

    nops = " ".join([f"\"nop\\n\"" for _ in range(len(self.node_ids))])
    self.body.add(TextBlock(f"__asm__ volatile({nops});"))


class SetFlagAndHaltBlock(InodeCodeBlock):
  def __init__(self, annotation: str = ""):
    super().__init__(annotation)
    self._build()

  def _build(self):
    self.body.add(TextBlock(f"__builtin_INODE_SET_FLAG(1);"))
    self.body.add(TextBlock(f"__builtin_INODE_HALT();"))

class HaltBlock(InodeCodeBlock):
  def __init__(self, annotation: str = ""):
    super().__init__(annotation)
    self._build()

  def _build(self):
    self.body.add(TextBlock(f"__builtin_INODE_HALT();"))


class SetFlag(InodeCodeBlock):
  def __init__(self, annotation: str = ""):
    super().__init__(annotation)
    self._build()

  def _build(self):
    self.body.add(TextBlock(f"__builtin_INODE_SET_FLAG(1);"))


class ClearFlag(InodeCodeBlock):
  def __init__(self, annotation: str = ""):
    super().__init__(annotation)
    self._build()

  def _build(self):
    self.body.add(TextBlock(f"__builtin_INODE_SET_FLAG(0);"))


class SyncPairINode(InodeCodeBlock):
  """Synchronize nodes after send/recv using UUID-based barrier"""

  def __init__(self, current_node: NodeID, participating_nodes: List[NodeID], uuid: int, annotation: str = ""):
    super().__init__(annotation)
    self.current_node = current_node
    self.participating_nodes = participating_nodes
    self.uuid = uuid
    self._build()

  def _build(self):
    # Set flag with UUID
    self.body.add(TextBlock(f"__builtin_INODE_SET_FLAG({self.uuid});"))

    # Wait for all other participating nodes
    for node in self.participating_nodes:
      if node != self.current_node:
        self.body.add(TextBlock(f"__builtin_INODE_STANDBY({node.value}, {self.uuid});"))

    # Add nops for timing (one per participating node)
    nops = " ".join([f"\"nop\\n\"" for _ in range(len(self.participating_nodes))])
    self.body.add(TextBlock(f"__asm__ volatile({nops});"))

    # Clear flag
    self.body.add(TextBlock(f"__builtin_INODE_SET_FLAG(0);"))


class InodeCodeBlockManager(NodeCodeBlockManager):
  """A class that manages and generates code blocks for inodes."""

  def __init__(self, func_name: str):
    UniqueVar.reset()
    self.blocks = {key: {CodePhase.INIT: [], CodePhase.EXEC: [], CodePhase.EXEC_TILE: [], CodePhase.END: []}
                   for key in self.nodes}
    self.func_name = func_name

  @property
  def nodes(self) -> List[NodeID]:
    return NodeID.inodes()

  @property
  def target(self) -> str:
    return "inode"

  def render_phase(self, node: NodeID, phase: CodePhase) -> str:
    blocks = self.blocks[node][phase]
    if len(blocks) == 0:
      return ""

    if phase is CodePhase.EXEC_TILE:
      # Tiling phase: wrap in for loop
      seq_block = SequentialBlock()
      for codeblock in blocks:
        seq_block.add(codeblock)
      code = f"{indent(SimpleFor(DevConfig().ImcflowFuncMap[self.func_name].tiling_factor, seq_block).render(), '  ')}\n"
    else:
      # None tiling phase: just render
      code = ""
      for codeblock in blocks:
        code += f"{indent(codeblock.render(), '  ')}\n"

    return code

  def generate_body(self) -> str:
    code = ""
    first = True
    for node in self.nodes:
      condition = f"if" if first else f"else if"
      code += f"{condition} (hid == {node.to_coord(0)} && wid == {node.to_coord(1)}) {{ // {node.name}\n"
      code += self.render_phase(node, CodePhase.INIT)
      code += self.render_phase(node, CodePhase.EXEC)
      code += self.render_phase(node, CodePhase.EXEC_TILE)
      code += self.render_phase(node, CodePhase.END)
      code += "}\n"
      first = False
    return code

  def start_block(self) -> str:
    code = (
      "#include \"../common_decl.h\"\n"
      f"void {self.func_name}() {{\n"
      "  int hid = __builtin_INODE_GET_CORE_HID();\n"
      "  int wid = 0;\n"
      f"{indent(UniqueVar.get_decls_str(), '  ')}\n"
    )
    return code

  def end_block(self) -> str:
    return "}\n"



"""
  __builtin_INODE_SEND(1, 1, 1, 1);
  __builtin_INODE_RECV(1, 1, 1, 1);
  __builtin_INODE_LAYERINIT();
  __builtin_INODE_IMCE_COMPUTE(1);

  __builtin_INODE_WR_IMEM(1, 1, 1);
  __builtin_INODE_WR_IMCU(1, 1, 1);
  __builtin_INODE_WR_REG(1, 1, 1);
  __builtin_INODE_SET_ADDR_CNT(1);

  __builtin_INODE_SET_FLAG(1);
  __builtin_INODE_STANDBY(1, 1);

  __builtin_INODE_DONE();
  __builtin_INODE_HALT();
  __builtin_INODE_INTRT(1);

  __builtin_INODE_PU(addr, imm, rs, slv_node_id);

  int a = __builtin_INODE_GET_CORE_HID();
  int b = __builtin_INODE_GET_CORE_WID();
"""
