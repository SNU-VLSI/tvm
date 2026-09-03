from tvm.relay.backend.contrib.imcflow.codeblock import *
from tvm.contrib.imcflow import DataBlock, InstEdgeInfo, TensorID, TensorEdge, TensorEdgeInfo
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from tvm.contrib.imcflow import bugfix_off_mode
from tvm.contrib.imcflow import drop_psum_send
from tvm.contrib.imcflow import step_freerun_n, step_freerun_factors
from tvm.contrib.imcflow import imcu_intra_drain_nops
from tvm.contrib.imcflow import resid_fill_lead_groups
from tvm.contrib.imcflow import resid_fanout_lead_words
from tvm.relay.op.contrib.imcflow import CustomIDToNode, residual_in_region_mode
from tvm.relay.backend.contrib.imcflow.transform import getInnerNodeID
from textwrap import indent
import math
import pdb

NOP_LOOP_CNTS = 10


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

  def __init__(self, edge_info: InstEdgeInfo, annotation: str = "", wave: int = None):
    super().__init__(annotation)
    self.edge_info = edge_info
    # C1b (C): when wave is given, load THAT wave's IMEM blob (per-(core,wave)
    # program). wave=None -> the single/legacy .data_block (byte-identical stock).
    self.wave = wave
    self._build()

  def _build(self):
    if self.wave is not None:
      db = self.edge_info.get_wave_data_block(self.wave)
    else:
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

    self.body.add(TextBlock(f"{loop_cnt_var} = {cnt_addr_var}[0];"))
    self.body.add(TextBlock(f"__asm__ volatile(\"nop\");")) # BUGFIX_LOAD_USE_HAZARD

    # Max-throughput lever (IMCFLOW_DROP_PSUM): the producing imce drops its psum
    # SEND for garbage output, so this matching inode RECV loop would wedge waiting
    # for packets that never arrive. Drop it too (keep the count read + nop for
    # imem/label parity). Only the func_out (imce->inode output collector) tiled
    # RECV is affected; input/weight RECVs are separate blocks. Gated by env;
    # default OFF -> byte-identical.
    if drop_psum_send():
      self.body.add(TextBlock(f"// [DROP_PSUM] omitted tiled INODE_RECV loop ({loop_cnt_var} iters)"))
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
    # Barrier residue-clear: the consumer's merged window PRE-CLEARS
    # (STANDBY(sender,0)) per word, but the sense-reversing all-inode barrier
    # leaves this inode's flag at 254/255 -- without a one-time SET_FLAG(0)
    # the first pre-clear and this sender's rendezvous mutually wait forever.
    # Gated on the lever + this SEND actually having a data rendezvous (OFF
    # byte-identical). Must be render-time deferred: at build time the
    # pair_manager is not attached yet, so the probe would return "".
    if residual_in_region_mode():
      self.body.add(DeferredTextBlock(
          lambda: "__builtin_INODE_SET_FLAG(0); // iter4b: clear barrier residue for pre-clear"
          if self._get_presend_sync_code_str(iter_var="0") else ""))
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
    nop_delay = NopLoopBlock(NOP_LOOP_CNTS).render() + "\n" if DevConfig().single_qconv else ""
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
    # Option A (paced multicast, non-tiled path): step-by-(K*W) token unroll, same
    # as _build_tiled but recv_count is a Python int here. Only fires for a paced
    # region-input multicast SEND (merged region1); else -> byte-identical loop.
    # W = PACED_MULTICAST_WORDS_PER_WINDOW: each token-block window wraps W SENDs
    # (the W bitplanes of ONE pixel) to MATCH the consumer's W-RECV window; a
    # 1-SEND-per-window unroll advances the inode token W times faster than the
    # consumer -> block-boundary desync wedge. K blocks x W words = K*W step.
    if self.is_paced_multicast_send() and eff == 1:
      from tvm.relay.backend.contrib.imcflow.send_recv_sync import (
          PACED_MULTICAST_NUM_BLOCKS, PACED_MULTICAST_WORDS_PER_WINDOW)
      K = PACED_MULTICAST_NUM_BLOCKS
      W = PACED_MULTICAST_WORDS_PER_WINDOW
      pmc_iv = UniqueVar(f"send_pmc_iv_{fifo_id}", dtype="int")
      inner = ""
      for b in range(K):
        pre = self._get_presend_sync_code_str(token_block=b)
        if pre:
          inner += indent(pre.rstrip("\n"), "  ") + "\n"
        for w in range(W):
          inner += (f"  __builtin_INODE_SEND({var} + ({pmc_iv} + {b}*{W} + {w})*32, "
                    f"0, {next_policy_addr}, {fifo_id});\n")
          if nop_delay:
            inner += indent(nop_delay.rstrip("\n"), "  ") + "\n"
      self.body.add(TextBlock(
          f"for (int {pmc_iv} = 0; {pmc_iv} < {recv_count}; "
          f"{pmc_iv} += {K*W}) {{ // paced-mc token step-by-{K}x{W} loop\n{inner}}}"))
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
    _fr_mult = (1 + step_freerun_n()) if (step_freerun_n() > 0 and self._is_conv_activation_feed()) else 1
    if _fr_mult != 1:
      self.body.add(TextBlock(f"{loop_cnt_var} = {cnt_addr_var}[0] * {_fr_mult};"))
    else:
      self.body.add(TextBlock(f"{loop_cnt_var} = {cnt_addr_var}[0];"))
    self.body.add(TextBlock(f"__asm__ volatile(\"nop\");")) # BUGFIX_LOAD_USE_HAZARD

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
    nop_delay = NopLoopBlock(NOP_LOOP_CNTS).render() + "\n" if DevConfig().single_qconv else ""
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

    # Option A (paced region-input multicast, merged region1): UNROLL the tiled
    # SEND loop by K = PACED_MULTICAST_NUM_BLOCKS with a LITERAL phase-token per
    # block, exactly mirroring the SAFE fused-add unroll above AND the consumer's
    # per-block unroll (imce create_loop_from_call). Each unrolled body b emits the
    # interlocked window paced_multicast_token(b) then the SEND, so consecutive
    # packets carry DISTINCT tokens (no repeated-flag re-arm race). loop_cnt_var
    # (var6) is a whole-pixel multiple and K|count (K=4) so the step divides
    # evenly. eff==1 always here (the `-11` multicast input isn't feed-spread).
    if self.is_paced_multicast_send() and eff == 1:
      from tvm.relay.backend.contrib.imcflow.send_recv_sync import (
          PACED_MULTICAST_NUM_BLOCKS, PACED_MULTICAST_WORDS_PER_WINDOW)
      K = PACED_MULTICAST_NUM_BLOCKS
      W = PACED_MULTICAST_WORDS_PER_WINDOW  # SENDs per token-window = consumer's
                                            # W-RECV window (bitplanes of one pixel)
      pmc_iv = UniqueVar(f"{target_edge.simple_name()}_pmc_iv", dtype="int")
      inner = ""
      for b in range(K):
        pre = self._get_presend_sync_code_str(token_block=b)
        if pre:
          inner += indent(pre.rstrip("\n"), "  ") + "\n"
        for w in range(W):
          inner += (f"  __builtin_INODE_SEND({base_var} + ({pmc_iv} + {b}*{W} + {w})*32, "
                    f"0, {next_policy_addr}, {fifo_id});\n")
          if nop_delay:
            inner += indent(nop_delay.rstrip("\n"), "  ") + "\n"
      self.body.add(TextBlock(
          f"for (int {pmc_iv} = 0; {pmc_iv} < {loop_cnt_var}; "
          f"{pmc_iv} += {K*W}) {{ // paced-mc token step-by-{K}x{W} loop\n{inner}}}"))
      return

    if eff > 1:
      inner = ""
      for j in range(eff):
        fid = self.edge_info.spread_fifo_id(j, 4)
        pre = self._get_presend_sync_code_str(iter_var=f"({spread_iv} + {j})")
        if pre:
          inner += indent(pre.rstrip("\n"), "  ") + "\n"
        inner += (f"  __builtin_INODE_SEND({base_var} + ({spread_iv} + {j})*32, 0, "
                  f"{next_policy_addr}, {fid});\n")
        if nop_delay:
          inner += indent(nop_delay.rstrip("\n"), "  ") + "\n"
      self.body.add(TextBlock(_spread_loop(inner)))
      return
    self.body.add(SimpleFor(loop_cnt_var, send_body_with_sync))

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

  def is_paced_multicast_send(self):
    """Option A: True iff THIS SendBlock's data edge(s) form a paced region-input
    MULTICAST (the `-11` input feeding BOTH imce_0_2 and imce_1_2 in merged
    region1). The per-packet SEND loop must then be UNROLLED by
    PACED_MULTICAST_NUM_BLOCKS with a LITERAL phase-token per block (INODE cannot
    lower a runtime `iter % K` flag value). Mirrors is_safe_fusedadd_send. Gated
    (via the predicates) on residual_in_region_mode() AND region_merge_mode() ->
    non-merged / OFF / region2 -> False -> the flat SEND loop below (byte-id)."""
    if not hasattr(self.builder, 'pair_manager') or self.builder.pair_manager is None:
      return False
    edge_or_edges = self._get_edge()
    if residual_in_region_mode() and isinstance(self.block.id, list):
      edge_or_edges = self.block.id
    if edge_or_edges is None:
      return False
    edges = edge_or_edges if isinstance(edge_or_edges, list) else [edge_or_edges]
    return self.builder.pair_manager.has_paced_multicast_edge(edges)

  def _get_presend_sync_code_str(self, iter_var=None, safe_block=None, token_block=None):
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
    # IMCFLOW_RESIDUAL_IN_REGION: a single multicast SendBlock over a shared
    # DataBlock (the OFF-style [0]-only param send) must derive its rendezvous
    # targets from EVERY edge of the multicast -- _get_edge() returns only
    # .id[0], and if that happens to be the bare tuple-dst consumer the
    # windowed plain-dst consumer (e.g. subset18 region1 imce_0_2) becomes
    # invisible here, the SEND goes out bare, and the consumer STANDBYs on a
    # flag the inode never raises (RTL region1 launch3 wedge). Lever OFF ->
    # unchanged (single-edge ids).
    if residual_in_region_mode() and isinstance(self.block.id, list):
      edge_or_edges = self.block.id
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
        inode_hw, imce_hw = eps
        # The inode-side SET_FLAG value comes from pm.pack_const_go_flag(inode,
        # imce): pack_const_sync_flag() (1 default, 253 with a 2-inode residual
        # add) for a single-consumer inode -- byte-identical -- but a DISTINCT
        # per-consumer value when this inode paces >=2 pack-const consumers. The
        # go-pulse is a SHARED scalar flag on the inode; if two consumers both
        # STANDBY(inode, <same value>) the pulse for one is stolen by the other
        # (region3: de-fused standalone BN imce_2_1 AND fused conv imce_2_2 both
        # STANDBY(inode_2_0, 253) -> theft -> STANDBY(consumer,0) never clears ->
        # wedge). A distinct value per consumer makes the pulse unambiguous. The
        # imce still raises its OWN flag=1 and the inode waits for it. Lockstep
        # with RecvConstBlock's window (both read pack_const_go_flag()).
        # Wave-aware: the same (inode, imce) pair may pace pack-const in >1 wave
        # (merge core reuse); pass THIS edge's consumer wave so inode & imce derive
        # the SAME distinct flag per wave (v14 cross-wave alias fix). Non-merged ->
        # wave 0 -> byte-identical.
        pc_flag = pm.pack_const_go_flag(inode_hw, imce_hw, pm._edge_dst_wave(e))
        # per-window READY token (invite) -- MUST match the imce RecvConstBlock's
        # SETFLAG for THIS const edge. Both derive it from the SAME edge `e`, so
        # they agree by construction (lockstep). DISTINCT per (const-window) on the
        # imce breaks the shared-flag-1 lost-wakeup where two pacers' windows to one
        # imce collapsed the level flag. <=1 window on the imce -> 1 (byte-id).
        ready_flag = pm.pack_const_ready_token(imce_hw, e)
        return (
          f"__builtin_INODE_STANDBY({imce_hw.value}, {ready_flag}); // pack-const sync with {imce_hw.name}\n"
          f"__builtin_INODE_SET_FLAG({pc_flag});\n"
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
      # IMCFLOW_RESIDUAL_IN_REGION: skip the plain-int-dst consumer of a model-
      # input multicast that is ALSO fed to an in-region residual add (imce_0_1:
      # standalone min_max_quantize, co-fed with imce_0_2's skip). Mirroring the
      # proven OFF baseline, the inode STANDBYs ONLY the windowed residual-add
      # consumer (imce_0_2, added below) and this fanout consumer stays a BARE
      # receiver (its RECV window is likewise bared in send_recv_sync). If the
      # inode kept STANDBY(imce_0_1,1) here, imce_0_1 (now bare, no SETFLAG)
      # would never raise flag 1 -> inode wedges. Lever OFF -> predicate False
      # -> byte-identical (imce_0_1 stays a rendezvous target as before).
      if pm.is_residual_multicast_conv_input_recv(e):
        continue
      rnode = pm._get_hw_node(e.dst_id)
      if isinstance(rnode, tuple):
        rnode = rnode[0]
      if rnode is not None and rnode.is_imce() and rnode not in target_imces:
        target_imces.append(rnode)

    # IMCFLOW_RESIDUAL_IN_REGION: this same skip SEND is ALSO multicast to the
    # in-region residual add (imce_0_2), whose dst is a composite tuple ->
    # is_inode_data_input_recv() above skips it, so the add would never see the
    # inode's flag. Add the residual add receiver as a second rendezvous target
    # so the inode pre-send STANDBYs BOTH imce_0_1 AND the add (imce_0_2) in ONE
    # window (below), matching the consumer's merged SETFLAG(1) window. Lever
    # OFF -> residual_data_input receivers empty -> byte-identical.
    residual_target = False
    if residual_in_region_mode():
      for e in edges:
        # Option A (merged region1): a region-input skip landing on a composite
        # `data` operand (b1.res add / vecops, imce_1_2) is NOT classified as
        # is_residual_data_input_recv (its sibling operand is `lhs`, not `data`,
        # so _residual_data_producers finds 1). When it is co-MULTICAST with a
        # handshake-gated conv-head consumer, it MUST be paced -> admit it here so
        # the inode STANDBYs it too (merged 2-target window below). LOCKSTEP with
        # get_recv_window_sync (imce side gives it the matching flag-1 window
        # under the SAME predicate). Narrow-gated -> OFF / region2 unaffected.
        _paced_skip = pm.is_paced_region_input_residual_skip(e)
        if not pm.is_residual_data_input_recv(e) and not _paced_skip:
          continue
        # Bare identity rhs: a REGION-INPUT operand (identity skip, src gid <
        # 0) is paced by rhs-fifo backpressure + fanout-lead, not flags.
        # LOCKSTEP: the add-side window drops this sender too
        # (get_merged_residual_input_window, same predicate). EXCEPTION: a
        # paced skip (above) KEEPS the receiver as a rendezvous target.
        _sgid = getattr(e.src_id, "graph_node_id", None)
        if isinstance(_sgid, int) and _sgid < 0 and not _paced_skip:
          continue
        rnode = pm._get_hw_node(e.dst_id)
        if isinstance(rnode, tuple):
          rnode = rnode[0]
        if rnode is not None and rnode.is_imce() and rnode not in target_imces:
          target_imces.append(rnode)
          residual_target = True

    if not target_imces:
      return ""

    target_imces.sort(key=lambda x: x.value)

    # Option A (merged region1): if ANY edge of this SEND is a paced region-input
    # MULTICAST (skip OR its handshake-gated co-consumer), the merged window uses
    # a MONOTONIC PHASE-TOKEN pair (t1, t2) from paced_multicast_token(token_block)
    # -- NOT a single repeated flag. The old single flag (249) re-armed the SAME
    # value every packet iteration -> a consumer passed its STANDBY on a STALE
    # token and ran an iteration ahead -> data-stream re-arm wedge. The caller
    # (SendBlock._build_tiled) UNROLLS this SEND loop by K and passes token_block=b
    # per block so consecutive iterations carry DISTINCT tokens. Lockstep with both
    # consumers' recv windows (get_recv_window_sync, same paced_multicast_token).
    # Non-paced multicasts keep flag 1 (byte-identical).
    from tvm.relay.backend.contrib.imcflow.send_recv_sync import paced_multicast_token
    _is_paced_mc = any(
        pm.is_paced_region_input_residual_skip(e)
        or pm.is_paced_multicast_handshake_consumer(e)
        for e in edges)

    # Paced-multicast interlocked window (RECV-as-ack, order-independent). Per
    # block b, tokens (t1=base+2b READY, t2=base+2b+1 GO):
    #   STANDBY(rnode, t1) for each consumer  -- wait each consumer's READY (t1
    #                                            raised on the consumer's own flag)
    #   SET_FLAG(t2)                          -- producer GO on the inode flag
    #   STANDBY(rnode, 0) for each            -- wait each consumer clear
    #   SET_FLAG(0)                           -- clear the inode flag, then SEND
    # Matches the consumer `SETFLAG(t1); STANDBY(inode, t2); SETFLAG(0); RECV`.
    # Distinct tokens per consecutive block (no collapsible repeated edge) + RECV
    # as ack bound the skew to < 1 iter. token_block is the LITERAL block index
    # from the unrolled caller (INODE requires a literal flag value -- backend
    # cannot lower a runtime `iter % K`).
    if _is_paced_mc and len(target_imces) >= 2:
      t1, t2 = paced_multicast_token(token_block or 0)
      sync_lines = []
      for rnode in target_imces:
        sync_lines.append(f"__builtin_INODE_STANDBY({rnode.value}, {t1}); // paced-mc READY {rnode.name} (block {int(token_block or 0)})")
      sync_lines.append(f"__builtin_INODE_SET_FLAG({t2}); // paced-mc GO")
      for rnode in target_imces:
        sync_lines.append(f"__builtin_INODE_STANDBY({rnode.value}, 0);")
      sync_lines.append(f"__builtin_INODE_SET_FLAG(0);")
      return "\n".join(sync_lines) + "\n"

    # When a residual add receiver joins the target set the SEND is a single
    # MIXED multicast to >=2 imces: emit ONE merged window (single scalar inode
    # flag) that STANDBYs every target's SETFLAG(1), sets flag once, waits every
    # target to clear, clears once -- then SEND. Two separate per-target windows
    # would toggle the inode flag 1->0->1->0 and race the second consumer (Fix D
    # lesson, inode side). The OFF path keeps the original per-target 4-phase
    # emission byte-identical.
    if residual_target and len(target_imces) >= 2:
      sync_lines = []
      for rnode in target_imces:
        sync_lines.append(f"__builtin_INODE_STANDBY({rnode.value}, 1); // sync with {rnode.name} before SEND")
      sync_lines.append(f"__builtin_INODE_SET_FLAG(1);")
      for rnode in target_imces:
        sync_lines.append(f"__builtin_INODE_STANDBY({rnode.value}, 0);")
      sync_lines.append(f"__builtin_INODE_SET_FLAG(0);")
      return "\n".join(sync_lines) + "\n"

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
    # IMCFLOW_RESIDUAL_IN_REGION: a model-input multicast fans out to TWO
    # distinct receivers (b1 conv entry AND the residual skip) that share ONE
    # DataBlock, so block.id lists BOTH edges. add_send_block emits a SEPARATE
    # per-edge SendBlock for each, but block.id (shared) would make every one
    # rendezvous with the SAME (first) receiver -> the 2nd fan-out SEND waits on
    # the wrong node's flag (region1 imce_0_1 SEND standing by on imce_0_2) ->
    # count mismatch + wrong-node wedge. Prefer this block's OWN per-edge owner
    # (edge_info.owner, set per fan-out edge in add_send_block) so each SEND
    # rendezvouses with its own consumer. Lever OFF -> predicate False -> the
    # original block.id path -> byte-identical.
    if residual_in_region_mode() and isinstance(self.block.id, list):
      own = getattr(self.edge_info, "owner", None)
      if own is not None and own in self.block.id:
        return own
    if isinstance(self.block.id, list):
      return self.block.id  # Return list for multicast handling
    if isinstance(self.block.id, TensorEdge):
      return self.block.id
    return None

class SendBlockResidualFanoutInterleaved(InodeCodeBlock):
  """IMCFLOW_RESIDUAL_IN_REGION: one inode source fans out a model-input identity
  to TWO distinct converging consumers (the b1 conv-entry minmax AND the in-region
  residual add's skip). Both streams reach the SAME add, so the pipeline needs BOTH
  concurrently. Emitting them as SEPARATE per-edge send blocks (full burst to A,
  THEN full burst to B) SERIALIZES them: consumer B starves while A's burst runs,
  the converge at the add stalls, backpressure freezes the sender mid-burst and the
  first tile wedges forever (region1 first-tile 20000-poll deadlock).

  Fix: emit ONE loop over the shared (tiled) word count whose body alternates, per
  word i: [presend-sync_A; SEND_A(word i)] [presend-sync_B; SEND_B(word i)]. Each
  edge keeps its OWN fifo / policy / per-word rendezvous semantics (bare for the
  conv-entry consumer, the 4-phase flag rendezvous for the residual-add consumer) --
  only the EMISSION is interleaved, so neither consumer starves.

  Reuses per-edge SendBlock instances purely to obtain each edge's exact presend
  rendezvous (edge_info.owner selects the right receiver) and SEND parameters; they
  are NOT registered as separate code blocks. Lever OFF -> never constructed
  (visit_function only routes residual_fanout_edges here) -> byte-identical.
  """

  def __init__(self, builder, blocks: List[DataBlock], edge_infos: List[TensorEdgeInfo], annotation: str = ""):
    super().__init__(annotation)
    assert len(blocks) == len(edge_infos), "# of blocks and edge_infos must be equal"
    assert len(blocks) >= 2, "residual fanout interleave needs >= 2 edges"
    self.builder = builder
    self.blocks = blocks
    self.edge_infos = edge_infos
    # Per-edge SendBlock helpers (own edge_info -> own owner -> own rendezvous).
    self._helpers = []
    for db, ei in zip(self.blocks, self.edge_infos):
      h = SendBlock(builder, db, ei, annotation="")
      # Discard the helper's self-built body; we only borrow its per-word sync
      # (_get_presend_sync_code_str) below, never rendering the helper itself.
      h.body = SequentialBlock()
      self._helpers.append(h)
    self._build()

  def _build(self):
    # Barrier residue-clear (see the SendBlock.__init__ prologue for the
    # rationale); this class discards the helper bodies, so emit it here too.
    if residual_in_region_mode():
      self.body.add(DeferredTextBlock(
          lambda: "__builtin_INODE_SET_FLAG(0); // iter4b: clear barrier residue for pre-clear"
          if any(h._get_presend_sync_code_str(iter_var="0") for h in self._helpers)
          else ""))
    # All fan-out edges share ONE model-input DataBlock (same base offset) and are
    # tiled on the SAME runtime packet count. Verify and use a single tile-count
    # loop. Each edge still SENDs on its own fifo/policy (distinct routing entries).
    tiled = all(db.tiling_info is not None for db in self.blocks)

    # Resolve the tile packet-count pointer from the first edge (all identical).
    if tiled:
      first = self.blocks[0]
      first_edge = first.id[0] if isinstance(first.id, list) else first.id
      cnt_addr_var = UniqueVar(
          f"{first_edge.simple_name()}_fanout_cnt_base_address",
          dtype="int", pointer_type=True)
      _cnt_block = DevConfig().MemLayout[self.builder.func_name].get_data_block_by_id(
          f"{first_edge.simple_name()}_cnt_base_addr")
      self.body.add(TextBlock(f"{cnt_addr_var} = (int*)({_cnt_block.offset});"))
      loop_cnt_var = UniqueVar(f"{first_edge.simple_name()}_fanout_tile_loop_count", dtype="int")
      self.body.add(TextBlock(f"{loop_cnt_var} = {cnt_addr_var}[0];"))
      self.body.add(TextBlock('__asm__ volatile("nop");'))  # BUGFIX_LOAD_USE_HAZARD
      trip = loop_cnt_var
    else:
      # Non-tiled fallback: fixed word count (all edges same size).
      trip = math.ceil(self.blocks[0].size / 32)

    # Per-edge base-offset vars + SEND parameters.
    infos = []
    for db, ei, h in zip(self.blocks, self.edge_infos, self._helpers):
      infos.append({
          "offset": db.offset,
          "policy": ei.policy_info[0].address,
          "fifo": ei.fifo_id,
          "helper": h,
      })

    def _base_var(x):
      """Lazily emit `varN = <offset>;` once for this edge and return the var.

      Emitted only by the paths that actually index off a base (same_route /
      legacy lockstep); the fanout-lead path walks cursors instead.
      """
      if "base_var" not in x:
        x["base_var"] = UniqueVar("fanout_send_base_address", dtype="int")
        self.body.add(TextBlock(f"{x['base_var']} = {x['offset']};"))
      return x["base_var"]

    nop_delay = NopLoopBlock(NOP_LOOP_CNTS).render() + "\n" if DevConfig().single_qconv else ""

    # ★ True-multicast collapse. In ResNet8's b1.res the two fan-out edges (b1 conv
    # entry + residual-add skip) resolve to the SAME router policy entry AND the SAME
    # inode send FIFO (all edges: policy 1, fifo 2) -- one INODE_SEND on that entry is
    # MULTICAST and reaches BOTH imces at once (imce_0_2's RECV(2) is even annotated
    # "-> imce_0_1, imce_0_2"). Emitting a SEPARATE SEND per edge then pushes each word
    # TWICE into the single multicast FIFO: both consumers receive it twice -> RECV
    # FIFO overflow / send backpressure -> first-tile wedge (~word 11/416). So when all
    # edges share (policy, fifo, base), emit ONE SEND per word, preceded by the UNION of
    # every edge's per-word rendezvous (bare for the conv entry, the 4-phase flag for the
    # residual add), so the single multicast word is paced by every consumer that needs a
    # handshake. If edges have DISTINCT policy/fifo (genuine separate streams) fall back
    # to the word-interleaved per-edge emission (each keeps its own route+rendezvous).
    same_route = (len({(x["policy"], x["fifo"]) for x in infos}) == 1
                  and len({b.offset for b in self.blocks}) == 1)

    if same_route:
      _policy = infos[0]["policy"]
      _fifo = infos[0]["fifo"]
      _base = _base_var(infos[0])

      def multicast_body(iter, infos=infos, policy=_policy, fifo=_fifo, base=_base,
                         nop_delay=nop_delay):
        code = ""
        for x in infos:
          pre = x["helper"]._get_presend_sync_code_str(iter_var=iter)
          if pre:
            code += pre
        code += (f"__builtin_INODE_SEND({base} + {iter}*32, 0, {policy}, {fifo}); "
                 f"// multicast fan-out to all consumers\n")
        code += nop_delay
        return code

      self.body.add(SimpleFor(trip, multicast_body))
      return

    # FANOUT-LEAD (distinct routes only; see resid_fanout_lead_words for the
    # full pacing rationale). The add's rhs stream cannot deliver anything
    # until the main path primes, so the conv-head stream must run LEAD words
    # ahead: prime LEAD conv words, then per word [add-stream (w);
    # conv-stream (w+LEAD)], then drain the add-stream tail. The same_route
    # true-multicast above is one single-credit stream and cannot lead.
    lead = resid_fanout_lead_words()

    if lead <= 0:
      # Legacy word-lockstep interleave (knob explicitly 0).
      def interleaved_body(iter, infos=infos, nop_delay=nop_delay):
        code = ""
        for x in infos:
          pre = x["helper"]._get_presend_sync_code_str(iter_var=iter)
          if pre:
            code += pre
          code += (f"__builtin_INODE_SEND({_base_var(x)} + {iter}*32, 0, "
                   f"{x['policy']}, {x['fifo']});\n")
          code += nop_delay
        return code
      self.body.add(SimpleFor(trip, interleaved_body))
      return

    def _is_add_stream(x):
      """True for the residual-ADD stream (the lagged side of the schedule).

      Classified by EDGE (dst is the residual add's data operand), falling
      back to 'has a presend rendezvous' for any windowed edge. Render-time
      only (needs pair_manager); loop bodies call this lazily.
      """
      h = x["helper"]
      if bool(h._get_presend_sync_code_str(iter_var="0")):
        return True
      pm = getattr(self.builder, "pair_manager", None)
      ee = h._get_edge() if pm is not None else None
      if ee is None:
        return False
      return any(pm.is_residual_data_input_recv(e)
                 for e in (ee if isinstance(ee, list) else [ee]))

    # CURSOR addressing only (cur += 32): the INODE ISA has no reg-reg
    # arithmetic, so `(iter + var)*32` / `var + var` fail isel. The conv
    # cursor runs prime(0..lead-1) then steady(lead..trip-1) continuously;
    # the add cursor runs steady(0..steady_cnt-1) then drain(..trip-1).
    # For a runtime trip we assume trip > lead (steady_cnt would go negative
    # and the drain would overrun otherwise); the int case clamps.
    if isinstance(trip, int):
      lead = min(lead, trip)
    cur_conv = UniqueVar("fanout_lead_conv_cursor", dtype="int")
    cur_add = UniqueVar("fanout_lead_add_cursor", dtype="int")
    self.body.add(TextBlock(f"{cur_conv} = {infos[0]['offset']};"))
    self.body.add(TextBlock(f"{cur_add} = {infos[0]['offset']};"))
    steady_cnt = UniqueVar("fanout_lead_steady_count", dtype="int")
    self.body.add(TextBlock(f"{steady_cnt} = {trip} - {lead};"))

    def _sends(pred, cur, iter=None, with_sync=False):
      code = ""
      for x in infos:
        if not pred(x):
          continue
        if with_sync:
          code += x["helper"]._get_presend_sync_code_str(iter_var=iter)
        code += f"__builtin_INODE_SEND({cur}, 0, {x['policy']}, {x['fifo']});\n"
        if with_sync:
          code += nop_delay
      return code + f"{cur} = {cur} + 32;\n"

    self.body.add(SimpleFor(
        lead,
        lambda iter: _sends(lambda x: not _is_add_stream(x), cur_conv),
        annotation="resid fanout-lead prime (conv stream ahead)"))
    self.body.add(SimpleFor(
        steady_cnt,
        lambda iter: (_sends(_is_add_stream, cur_add, iter, with_sync=True)
                      + _sends(lambda x: not _is_add_stream(x), cur_conv)),
        annotation="resid fanout-lead steady"))
    self.body.add(SimpleFor(
        lead,
        lambda iter: _sends(_is_add_stream, cur_add, iter, with_sync=True),
        annotation="resid fanout-lead drain (add-stream tail)"))


class ResidResendFuncoutInterleavedBlock(InodeCodeBlock):
  """RESBUF fill / resend / func_out-collect, interleaved on the owning inode.

  The in-region residual add is a single in-order imce: per `group`-word
  output group it RECVs `group` rhs words (this inode's resend) + `group`
  main-path words, computes, and SENDs `group` func_out words back here.
  Any schedule that lets ONE of the three inode-side streams run dry or run
  unboundedly ahead deadlocks the region (sequential whole-loops, group
  lockstep, and word-level pacing all wedged on RTL); the working schedule
  is FILL-LEAD -- see resid_fill_lead_groups() for the analysis and the
  measured LAG window.

  All sends/recvs here are BARE (NoC valid/ready paced; the add-side window
  drops its matching STANDBY in get_merged_residual_input_window). Gated by
  residual_inode_buffer_mode() at the call site; OFF / no-RESBUF -> never
  constructed -> byte-identical.
  """

  def __init__(self, builder, resbuf_db: DataBlock, resend_info: TensorEdgeInfo,
               funcout_db: DataBlock, funcout_fifo_id: int, group: int = 4,
               collector_fifo_id: int = None, auto_lag: int = None,
               annotation: str = ""):
    super().__init__(annotation)
    self.builder = builder
    self.resbuf_db = resbuf_db
    self.resend_info = resend_info
    self.funcout_db = funcout_db
    self.funcout_fifo_id = funcout_fifo_id
    self.group = group
    # When set, the RESBUF collector fill (hop A, from the skip producer imce)
    # is folded into this block's schedule instead of a standalone leading
    # whole-buffer drain (which starves the add: the skip producer only
    # trickles words as the input streams).
    self.collector_fifo_id = collector_fifo_id
    # Geometry-derived fill-lead LAG (codegen._auto_fill_lead_groups); the
    # IMCFLOW_RESID_FILL_LEAD env overrides. None + no env -> hard error at
    # _build (no magic fallback; the window is geometry-dependent).
    self.auto_lag = auto_lag
    self._build()

  def _build(self):
    # RESBUF buffer is fully allocated (task #10) -> fixed word count. The func_out
    # collector on this inode is the same length (both hops of the same add, 256
    # words in ResNet8's factor-1 tile); we drive BOTH by the resbuf word count so
    # the resend/funcout stay in lockstep group-by-group.
    total = math.ceil(self.resbuf_db.size / 32)
    g = self.group
    ngroups = total // g
    tail = total - ngroups * g

    resend_policy = self.resend_info.policy_info[0].address
    resend_fifo = self.resend_info.fifo_id
    fout_fifo = self.funcout_fifo_id
    coll_fifo = self.collector_fifo_id

    # FILL-LEAD schedule (see resid_fill_lead_groups for the deadlock analysis
    # and the measured LAG window): prime the fill LAG groups ahead -- the
    # whole buffer is allocated, so fill-ahead is always safe -- then per
    # group resend(g) -> funcout(g) -> fill(g+LAG), then drain the tail.
    # LAG resolution: env override > geometry-derived auto. NO magic fallback:
    # the window is geometry-dependent (b3's measured 14 wedges a 16-group
    # buffer), so a failed geometry walk must fail HERE at compile time -- a
    # silently-wrong LAG costs a 20000-poll RTL wedge instead. Escape hatch:
    # set IMCFLOW_RESID_FILL_LEAD explicitly.
    if coll_fifo is not None:
      _env = resid_fill_lead_groups()
      if _env is not None:
        lag = _env
      elif self.auto_lag is not None:
        lag = self.auto_lag
      else:
        raise RuntimeError(
            "RESBUF fill-lead LAG could not be auto-derived from graph "
            "geometry (codegen._auto_fill_lead_groups returned None) and "
            "IMCFLOW_RESID_FILL_LEAD is unset. Fix the geometry walk for "
            "this graph pattern or set the env override explicitly.")
    else:
      lag = 0
    lag = min(lag, ngroups)

    if coll_fifo is not None and lag > 0:
      # Prime is WORD-level, but steady/drain keep the GROUP rhythm: the add
      # RECVs a whole g-word rhs group before emitting ANY output (word-level
      # resend/funcout pacing wedges mid-group, iter7 RTL). All sends/recvs
      # are BARE (NoC valid/ready paces them; the add-side window dropped its
      # matching STANDBY in get_merged_residual_input_window).
      lead_w = lag * g

      # CURSOR addressing only (cur += 32): the INODE ISA has no reg-reg
      # arithmetic and full (g*4+j)*32 index math overflowed inode_3_0's
      # 256-word imem. All three cursors advance strictly sequentially
      # across prime/steady/drain.
      cur_fill = UniqueVar("resid_fill_cursor", dtype="int")
      cur_resend = UniqueVar("resid_resend_cursor", dtype="int")
      cur_fout = UniqueVar("resid_funcout_cursor", dtype="int")
      self.body.add(TextBlock(f"{cur_fill} = {self.resbuf_db.offset};"))
      self.body.add(TextBlock(f"{cur_resend} = {self.resbuf_db.offset};"))
      self.body.add(TextBlock(f"{cur_fout} = {self.funcout_db.offset};"))

      def _step(cur, op):
        return f"{op}\n{cur} = {cur} + 32;\n"

      self.body.add(SimpleFor(
          lead_w,
          lambda iter: _step(cur_fill,
                             f"__builtin_INODE_RECV({cur_fill}, 0, 0, {coll_fifo});"),
          annotation="resid fill prime (fill-lead)"))

      def steady_body(iter, g=g):
        code = ""
        for _ in range(g):
          code += _step(cur_resend,
                        f"__builtin_INODE_SEND({cur_resend}, 0, {resend_policy}, {resend_fifo});")
        for _ in range(g):
          code += _step(cur_fout,
                        f"__builtin_INODE_RECV({cur_fout}, 0, 0, {fout_fifo});")
        for _ in range(g):
          code += _step(cur_fill,
                        f"__builtin_INODE_RECV({cur_fill}, 0, 0, {coll_fifo});")
        return code

      self.body.add(SimpleFor(
          ngroups - lag, steady_body,
          annotation="resid fill-lead steady: resend/funcout + fill-ahead groups"))

      def drain_body(iter, g=g):
        code = ""
        for _ in range(g):
          code += _step(cur_resend,
                        f"__builtin_INODE_SEND({cur_resend}, 0, {resend_policy}, {resend_fifo});")
        for _ in range(g):
          code += _step(cur_fout,
                        f"__builtin_INODE_RECV({cur_fout}, 0, 0, {fout_fifo});")
        return code

      self.body.add(SimpleFor(
          lag, drain_body,
          annotation="resid fill-lead drain: resend/funcout tail groups"))
      return

    # Legacy lockstep path (LAG=0 explicit, or no folded collector): per
    # group fill G -> resend G -> funcout G, base+index addressing.
    resend_base = UniqueVar("resid_resend_base_address", dtype="int")
    funcout_base = UniqueVar("resid_funcout_base_address", dtype="int")
    self.body.add(TextBlock(f"{resend_base} = {self.resbuf_db.offset};"))
    self.body.add(TextBlock(f"{funcout_base} = {self.funcout_db.offset};"))

    def group_body(iter, g=g):
      code = ""
      if coll_fifo is not None:
        for j in range(g):
          code += (f"__builtin_INODE_RECV({resend_base} + (({iter})*{g} + {j})*32, "
                   f"0, 0, {coll_fifo});\n")
      for j in range(g):
        code += (f"__builtin_INODE_SEND({resend_base} + (({iter})*{g} + {j})*32, 0, "
                 f"{resend_policy}, {resend_fifo});\n")
      for j in range(g):
        code += (f"__builtin_INODE_RECV({funcout_base} + (({iter})*{g} + {j})*32, "
                 f"0, 0, {fout_fifo});\n")
      return code

    self.body.add(SimpleFor(ngroups, group_body,
                            annotation="resid resend/funcout interleave"))

    # Defensive tail (ResNet8 group|total, so tail==0 -> emits nothing).
    for r in range(ngroups * g, ngroups * g + tail):
      if coll_fifo is not None:
        self.body.add(TextBlock(
            f"__builtin_INODE_RECV({resend_base} + {r}*32, 0, 0, {coll_fifo});"))
      self.body.add(TextBlock(
          f"__builtin_INODE_SEND({resend_base} + {r}*32, 0, "
          f"{resend_policy}, {resend_fifo});"))
      self.body.add(TextBlock(
          f"__builtin_INODE_RECV({funcout_base} + {r}*32, 0, 0, {fout_fifo});"))


class ResidCollectResendInterleavedBlock(InodeCodeBlock):
  """RESBUF collector-fill + resend, interleaved on the owning inode -- for a
  SAME-WAVE residual RESBUF whose add consumer does NOT feed a func_out this
  inode collects (i.e. the add output stays in-region, e.g. -> quantize).

  WHY (fsim-proven, v17 region2 deadlock): the plain path emits a STANDALONE
  full-buffer collector RECV (all N words up-front) THEN the resend SEND. The
  skip producer (an imce conv/bn) only TRICKLES its output words as the region
  streams, so demanding all N words before ANY resend starves the fed add:
  the add's rhs never arrives, so it never consumes -> its input backpressures
  -> the producer can't emit the remaining skip words -> the collector never
  completes -> the inode never reaches the next wave barrier -> global barrier
  desync (inode_3_0 stuck at -100001 collector word 64/1024 while inode_2_0
  already ARRIVEd at the wave-1 barrier). This is the exact hazard the plain
  path's own comment warns about; the func_out case already avoids it via
  ResidResendFuncoutInterleavedBlock. This block gives the func_out-less
  residual RESBUF the same non-blocking treatment: FILL-LEAD group interleave
  of collector-RECV(g) -> resend-SEND(g), so each group flows to the consumer
  immediately, un-backpressuring the producer.

  Two streams only (no func_out): prime LAG groups of collector-fill ahead
  (the whole buffer is allocated, so fill-ahead is safe), then per group
  resend(g) -> fill(g+LAG), then drain the resend tail. Mirrors the proven
  FILL-LEAD schedule of ResidResendFuncoutInterleavedBlock with the func_out
  third stream removed. BARE sends/recvs (NoC valid/ready paced; the add-side
  window dropped its matching STANDBY). Adds/renumbers no flags. Gated by
  residual_inode_buffer_mode() at the call site -> OFF -> never constructed ->
  byte-identical.
  """

  def __init__(self, builder, resbuf_db: DataBlock, resend_info: TensorEdgeInfo,
               collector_fifo_id: int, group: int = 4, auto_lag: int = None,
               annotation: str = ""):
    super().__init__(annotation)
    self.builder = builder
    self.resbuf_db = resbuf_db
    self.resend_info = resend_info
    self.collector_fifo_id = collector_fifo_id
    self.group = group
    self.auto_lag = auto_lag
    self._build()

  def _build(self):
    total = math.ceil(self.resbuf_db.size / 32)
    g = self.group
    ngroups = total // g
    tail = total - ngroups * g

    resend_policy = self.resend_info.policy_info[0].address
    resend_fifo = self.resend_info.fifo_id
    coll_fifo = self.collector_fifo_id

    # FILL-LEAD LAG resolution: env override > geometry-derived auto. No magic
    # fallback (a silently-wrong LAG costs a 20000-poll RTL wedge). Same policy
    # as ResidResendFuncoutInterleavedBlock.
    _env = resid_fill_lead_groups()
    if _env is not None:
      lag = _env
    elif self.auto_lag is not None:
      lag = self.auto_lag
    else:
      raise RuntimeError(
          "RESBUF fill-lead LAG could not be auto-derived from graph geometry "
          "(codegen._auto_fill_lead_groups returned None) and "
          "IMCFLOW_RESID_FILL_LEAD is unset. Fix the geometry walk for this "
          "graph pattern or set the env override explicitly.")
    lag = min(max(lag, 0), ngroups)

    # CURSOR addressing only (cur += 32): the INODE ISA has no reg-reg
    # arithmetic (mirrors the func_out block's cursor rationale).
    cur_fill = UniqueVar("resid_coll_fill_cursor", dtype="int")
    cur_resend = UniqueVar("resid_coll_resend_cursor", dtype="int")
    self.body.add(TextBlock(f"{cur_fill} = {self.resbuf_db.offset};"))
    self.body.add(TextBlock(f"{cur_resend} = {self.resbuf_db.offset};"))

    def _step(cur, op):
      return f"{op}\n{cur} = {cur} + 32;\n"

    if lag > 0:
      lead_w = lag * g
      self.body.add(SimpleFor(
          lead_w,
          lambda iter: _step(cur_fill,
                             f"__builtin_INODE_RECV({cur_fill}, 0, 0, {coll_fifo});"),
          annotation="resid-collect fill prime (fill-lead)"))

      def steady_body(iter, g=g):
        code = ""
        for _ in range(g):
          code += _step(cur_resend,
                        f"__builtin_INODE_SEND({cur_resend}, 0, {resend_policy}, {resend_fifo});")
        for _ in range(g):
          code += _step(cur_fill,
                        f"__builtin_INODE_RECV({cur_fill}, 0, 0, {coll_fifo});")
        return code

      self.body.add(SimpleFor(
          ngroups - lag, steady_body,
          annotation="resid-collect fill-lead steady: resend + fill-ahead groups"))

      def drain_body(iter, g=g):
        code = ""
        for _ in range(g):
          code += _step(cur_resend,
                        f"__builtin_INODE_SEND({cur_resend}, 0, {resend_policy}, {resend_fifo});")
        return code

      self.body.add(SimpleFor(
          lag, drain_body,
          annotation="resid-collect fill-lead drain: resend tail groups"))
    else:
      # LAG==0: strict per-group lockstep collect(g) -> resend(g).
      def group_body(iter, g=g):
        code = ""
        for _ in range(g):
          code += _step(cur_fill,
                        f"__builtin_INODE_RECV({cur_fill}, 0, 0, {coll_fifo});")
        for _ in range(g):
          code += _step(cur_resend,
                        f"__builtin_INODE_SEND({cur_resend}, 0, {resend_policy}, {resend_fifo});")
        return code

      self.body.add(SimpleFor(ngroups, group_body,
                              annotation="resid-collect lockstep: collect + resend"))

    # Word tail (ResNet8 group|total, so tail==0 emits nothing): collect then
    # resend each remaining word, keeping the two cursors in step.
    for _r in range(tail):
      self.body.add(TextBlock(
          _step(cur_fill, f"__builtin_INODE_RECV({cur_fill}, 0, 0, {coll_fifo});")))
      self.body.add(TextBlock(
          _step(cur_resend,
                f"__builtin_INODE_SEND({cur_resend}, 0, {resend_policy}, {resend_fifo});")))


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
        nop_delay = NopLoopBlock(NOP_LOOP_CNTS).render() + "\n" if DevConfig().single_qconv else ""
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

  @classmethod
  def next_sense(cls):
    """Advance and return the next alternating sense value (254/255)."""
    v = 254 + (cls._sense_counter % 2)
    cls._sense_counter += 1
    return v

  @classmethod
  def reset_sense(cls):
    cls._sense_counter = 0

  def __init__(self, node_id : NodeID, annotation: str = "", sense: int = None,
               two_phase: bool = False):
    super().__init__(annotation)
    self.node_id = node_id
    # sense=None -> stock behavior (255 + clear-to-0), byte-identical to before.
    # sense=254/255 -> single-phase sense-reversing barrier (no clear), packing.
    # two_phase=True -> skew-ROBUST two-phase barrier (arrive 254 / release 255),
    #   used ONLY by the wave-launch per-wave barriers (merge mode). See _build.
    self.sense = sense
    self.two_phase = two_phase
    self._build()

  def _build(self):
    if self.two_phase:
      # Two-phase (skew-robust) all-inode barrier. REQUIRED for the wave-launch
      # per-wave barriers: those introduce a NEW pattern (re-WR_IMEM barrier then
      # COMPUTE barrier within ONE host RUN, with asymmetric per-inode WR_IMEM
      # between them) that the single-phase sense-reversing barrier CANNOT survive.
      # RTL fact (controller.sv:148,63 + hazard_control.sv:97): sync_reg is ONE
      # scalar per inode, overwritten by SET_FLAG, and STANDBY stalls on strict
      # `!=`. So a fast inode that SET_FLAG(arrive) then SET_FLAG(next) ERASES the
      # arrive value a lagging peer is still `==`-waiting on -> lost wakeup (the v9
      # wedge: inode_2_0 raced re-WR_IMEM(255)->COMPUTE(254), erasing 255 before
      # slow peers sampled it; peers STANDBY(inode_2_0,255) forever).
      #
      # Textbook two-phase barrier: (1) ARRIVE = SET_FLAG(A); STANDBY(all peers,A)
      # -> everyone has arrived. (2) RELEASE = SET_FLAG(R); STANDBY(all peers,R)
      # -> everyone has ACKNOWLEDGED phase 1. No inode overwrites A with R until it
      # passed its own phase-1 STANDBY (saw all A's); it overwrites R only at the
      # NEXT barrier's phase 1, by which point every inode has passed phase 2 (saw
      # all R's) and no one is still reading R. Hence a FIXED (A,R)=(254,255) pair
      # is safe for EVERY barrier regardless of arrival skew -- no counter needed.
      A, R = 254, 255
      peers = [n for n in NodeID.inodes() if n != self.node_id]
      nops = " ".join([f"\"nop\\n\"" for _ in range(len(NodeID.inodes()))])
      # Phase 1: arrive.
      self.body.add(TextBlock(f"__builtin_INODE_SET_FLAG({A});"))
      for node in peers:
        self.body.add(TextBlock(f"__builtin_INODE_STANDBY({node.value}, {A});"))
      self.body.add(TextBlock(f"__asm__ volatile({nops});"))
      # Phase 2: release (acknowledge everyone saw phase 1 before A can be reused).
      self.body.add(TextBlock(f"__builtin_INODE_SET_FLAG({R});"))
      for node in peers:
        self.body.add(TextBlock(f"__builtin_INODE_STANDBY({node.value}, {R});"))
      self.body.add(TextBlock(f"__asm__ volatile({nops});"))
    elif self.sense is None:
      # Stock barrier: reused UUID=255 + clear-to-0. Byte-identical to the
      # pre-fix codegen (lever OFF / non-packing path).
      INODE_SYNC_UUID = 255
      self.body.add(TextBlock(f"__builtin_INODE_SET_FLAG({INODE_SYNC_UUID});"))
      for node in NodeID.inodes():
        if node != self.node_id:
          self.body.add(TextBlock(f"__builtin_INODE_STANDBY({node.value}, {INODE_SYNC_UUID});"))
      nops = " ".join([f"\"nop\\n\"" for _ in range(len(NodeID.inodes()))])
      self.body.add(TextBlock(f"__asm__ volatile({nops});"))
      self.body.add(TextBlock(f"__builtin_INODE_SET_FLAG(0);"))
    else:
      # Sense-reversing barrier: write this barrier's sense, wait for all peers
      # to present it, and DO NOT clear it (persistence protects stragglers).
      sense = self.sense
      self.body.add(TextBlock(f"__builtin_INODE_SET_FLAG({sense});"))
      for node in NodeID.inodes():
        if node != self.node_id:
          self.body.add(TextBlock(f"__builtin_INODE_STANDBY({node.value}, {sense});"))
      nops = " ".join([f"\"nop\\n\"" for _ in range(len(NodeID.inodes()))])
      self.body.add(TextBlock(f"__asm__ volatile({nops});"))

class Standby(InodeCodeBlock):
  def __init__(self, node_ids: List[NodeID], annotation: str = "", uuid: int = None):
    super().__init__(annotation)
    self.node_ids = node_ids
    # uuid=None -> stock 255 (byte-identical for all existing callers). C1b (C)
    # wave-launch passes WAVE_DONE_UUID to wait on a specific imce completion flag.
    self.uuid = uuid
    self._build()

  def _build(self):
    # Use UUID=255 for INODE-to-INODE sync to avoid conflict with SendRecvPairManager UUIDs (1-254)
    INODE_SYNC_UUID = 255 if self.uuid is None else self.uuid
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

  def generate_body(self, wave=None) -> str:
    # C1b (C): inode programs are single (policy/weights static; per-wave IMEM
    # swapping is IMCE-only), so `wave` is accepted for signature parity with the
    # base manager but ignored -- the inode body is always emitted whole.
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
