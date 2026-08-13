from abc import *
from typing import *
from copy import copy
import math
import os
from pprint import pprint
from tvm import relay
from tvm.relay.op.contrib.imcflow import CustomIDToNode
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from tvm.contrib.imcflow import NodeID, TensorID, TensorEdge, TensorEdgeInfo
from tvm.contrib.imcflow import bugfix_off_mode
from tvm.contrib.imcflow import drop_psum_send, drop_psum_keep_every
from tvm.contrib.imcflow import multiblock_fusedadd_bare, multiblock_fusedadd_safe, SAFE_TOKEN_BASE
from tvm.relay.op.op_attrs import Conv2DAttrs
from tvm.relay.backend.contrib.imcflow.conv_util import ConvUtil
from tvm.relay.backend.contrib.imcflow.codeblock import *
from tvm.relay.backend.contrib.imcflow.layout import apply_layout_to_type
from tvm.relay.backend.contrib.imcflow.transform_utils import getInnerNodeID, getOuterNodeID, get_type, getNodeID
from tvm.relay.dataflow_pattern import *
from textwrap import indent
import logging
import pdb
from dataclasses import dataclass

# Environment variable to enable MULTL overflow bug SW fix.
# Set IMCFLOW_BUGFIX_OVERFLOW_SW=1/0 to explicitly enable/disable the SW fix.
# If UNSET, the default is coupled to the master IMCFLOW_BUGFIX knob: the
# BUGFIX-off RTL lacks the HW overflow fix, so codegen SW-compensates by
# default when the master knob is off (bugfix_off_mode()==True); when the
# master knob is on (BUGFIX-on RTL) the default is off. An explicit
# IMCFLOW_BUGFIX_OVERFLOW_SW always wins.
def _is_multl_swfix_enabled():
  val = os.environ.get("IMCFLOW_BUGFIX_OVERFLOW_SW")
  if val is not None:
    return val == "1"
  from tvm.contrib.imcflow import overflow_sw_default_on
  return overflow_sw_default_on()

if TYPE_CHECKING:
  from .builder_context import BuilderContext

ConstPat = is_constant()

# for debugging
send_num_map = {}
recv_num_map = {}

# P4 (DESIGN §2.4, invariant II): flag-rendezvous handshake counters. Keyed by
# a flag-rendezvous pair uuid: {uuid: {"producer": n, "consumer": {node_value: n}}}.
# The producer barrier (get_pre_send_sync) and each consumer window
# (get_recv_window_sync) increment these at emission time, multiplied by the
# enclosing SimpleFor loop trips (same mechanism add_to_map uses). validate then
# asserts producer_handshakes == each consumer's window count, so a future
# granularity/flag mismatch surfaces at COMPILE time instead of a 20000-poll RTL
# hang. Only populated for edges the codeblocks explicitly contract (dwconv
# output today); ResNet8 never calls add_flag_handshake -> map stays empty ->
# no effect / byte-identical.
handshake_num_map = {}

@dataclass
class RecvSendNum:
  dir       : str  = ""  # "recv" or "send"
  total_num : int  = 1

  def set_iter(self, iter_num):
    self.total_num = self.total_num * iter_num
  
  def __int__(self):
    try:
      self.total_num = int(self.total_num)
    except:
      self.total_num = self.total_num.value
    return self.total_num
  
  def __eq__(self, other):
    if not isinstance(other, RecvSendNum):
      return False
    return (self.dir == other.dir and
            self.total_num == other.total_num)

def add_to_map(edge, count, is_send=True):
  if edge.dst_id.graph_node_id == 8:
    pass
  print(f"[add_recv_send_map] edge: {edge}, count: {count}, is_send: {is_send}")
  print(f"    loop count stack: {SimpleFor.count_stack}")
  outer_loop_count = 1 if len(SimpleFor.count_stack) == 0 else int(math.prod(SimpleFor.count_stack))
  target_map = send_num_map if is_send else recv_num_map
  count.set_iter(outer_loop_count)
  print(f"    adjusted count with outer loops ({outer_loop_count}): {count.total_num}")
  if edge in target_map:
    print(f"    previous count: {target_map[edge].total_num}, adding {count.total_num}")
    target_map[edge].total_num += count.total_num
    print(f"    new count: {target_map[edge].total_num}")
  else:
    target_map[edge] = count
    print(f"    new count: {target_map[edge].total_num}")

class ImceCodeBlock(CodeBlock):
  def __init__(self, annotation: str = ""):
    super().__init__(annotation)

  @abstractmethod
  def _render(self) -> str:
    pass


class SyncPairIMCE(ImceCodeBlock):
  """Synchronize IMCE nodes after send/recv using UUID-based barrier"""

  def __init__(self, current_node: NodeID, participating_nodes: List[NodeID], uuid: int, annotation: str = ""):
    super().__init__(annotation)
    self.current_node = current_node
    self.participating_nodes = participating_nodes
    self.uuid = uuid

  def _render(self) -> str:
    code = TextBlock("")

    # Set flag with UUID
    code += f"__builtin_IMCE_SETFLAG({self.uuid});"

    # Wait for all other participating nodes
    for node in self.participating_nodes:
      if node != self.current_node:
        code += f"__builtin_IMCE_STANDBY({node.value}, {self.uuid});"

    # Add nops for timing (one per participating node)
    nops = " ".join([f"\"nop\\n\"" for _ in range(len(self.participating_nodes))])
    code += f"__asm__ volatile({nops});"

    # Clear flag
    code += f"__builtin_IMCE_SETFLAG(0);"

    return code.render()


class ImceCallCodeBlock(ImceCodeBlock):
  num_in_edges = None

  def __init__(self, call: 'BuilderContext', annotation: str = ""):
    super().__init__(annotation)
    self.call = call
    self.in_edges = call.get_input_edges()
    self.out_edges = call.get_output_edges()
    self.prev_op = None
    # Skip edge count assertion when inside a composite (edges may not be individually tracked)
    # The composite handler will manage edges at the composite level
    if self.num_in_edges is not None and call.curr_composite_id is None:
      assert len(self.in_edges) == self.num_in_edges, \
        f"{self.__class__.__name__}: expected {self.num_in_edges} input edges, got {len(self.in_edges)}"
  
  def get_graph_node_id(self) -> NodeID:
    return getNodeID(self.call.call)

  def _make_unique_input_var_for_post_op(self, edge, i=None):
    if self.prev_op and (edge in self.prev_op.out_edges):
      edge = self.prev_op
    if i is not None:
      return UniqueVar((edge, i))
    else:
      return UniqueVar(edge)

  @property
  def num_blocks(self) -> int:
    if self.prev_op:
      return self.prev_op.num_out_blocks

    # Calculate from relay call output shape (channel dimension / 16)
    # Works for all standalone ops: BatchNorm, Add, Mult, etc.
    from tvm.relay.frontend.common import infer_shape
    out_shape = infer_shape(self.call.call)
    # Handle tuple outputs (e.g., from batch_norm)
    if isinstance(out_shape, list):
      out_shape = out_shape[0]
    # NCHW layout: channel is dim 1
    channels = out_shape[1]
    return math.ceil(channels / 16.0)

  @property
  def num_out_blocks(self) -> int:
    if self.prev_op:
      return self.prev_op.num_out_blocks
    else:
      # Standalone: same as num_blocks
      return self.num_blocks

  def __repr__(self):
    return f"{self.__class__.__name__}(gid: {self.call.get_gid()})"

# ======================================================
# helper functions
# ======================================================
def get_total_bytes(ttype):
    elem_count = math.prod(list(ttype.shape))
    if isinstance(elem_count, tvm.tir.expr.IntImm):
      elem_count = elem_count.value
    dtype = ttype.dtype
    if "int8" in dtype or "uint8" in dtype:
      bytes_per_elem = 1
    elif "int16" in dtype or "uint16" in dtype:
      bytes_per_elem = 2
    elif "int32" in dtype or "uint32" in dtype or "float32" in dtype:
      bytes_per_elem = 4
    else:
      raise RuntimeError(f"Unsupported dtype {dtype} in create_loop_from_call")
    total_bytes = elem_count * bytes_per_elem
    return total_bytes

class LoadLBBlock(ImceCodeBlock):
  """ Code block for receiving data from given fifo id to the line buffer """

  def __init__(self, count: int, repeat: int, edge: TensorEdge, edge_info: TensorEdgeInfo, builder=None, annotation: str = "", bare: bool = False, prefetch_fifo_base: int = None):
    super().__init__(annotation)
    self.count = count
    self.repeat = repeat
    self.edge = edge
    self.edge_info = edge_info
    self.builder = builder
    self.bare = bare
    # Max-throughput lever (IMCFLOW_FEED_PREFETCH): when set, this LOAD_LB burst's
    # `repeat` bitplanes use fifos prefetch_fifo_base..+repeat-1 (a per-pixel slot
    # in the P*repeat-wide window) instead of the edge's own spread. The caller
    # (_build_structure_a8af col_group unroll) assigns a distinct base per pixel
    # in the P-pixel group so 2 pixels' bitplanes occupy 8 distinct RECV fifos.
    self.prefetch_fifo_base = prefetch_fifo_base
    self._rz_uuid = None
    self._rz_consumer = None

    # BUGFIX knob: knob=on (bugfix_off_mode()==False) reproduces a8af's
    # per-LOAD_LB SETFLAG/STANDBY(uuid) sync; knob=off keeps the 934 handcraft
    # window sync below.
    if not bugfix_off_mode():
      self._build_a8af()
      return

    load_fifo_id = self.edge_info.fifo_id
    annotation = f"{self.edge}, {self.edge_info.node_info_str}"

    # handcraft window sync: wrap the whole LOAD_LB burst (the `repeat` inner
    # loop) in a single SETFLAG(1)...SETFLAG(0) window, with NO per-LOAD_LB
    # STANDBY. Non-cyclic (no pair) edges emit bare LOAD_LB.
    # `bare=True` forces a bare burst even for paired edges -- used by Marker A
    # (has_noc_rhs) nodes, whose data LOAD_LB carries NO window in handcraft (the
    # SETFLAG window there wraps the fused add-rhs RECV instead).
    if bare:
      pre_lines, post_lines = (None, None)
      # Marker B' override: a flag-3 data-multicast producer's data LOAD_LB must
      # carry a SETFLAG(3) window even on a Marker-A (has_noc_rhs) node. The
      # producer STANDBYs this receiver at flag 3, so a bare LOAD_LB here would
      # never satisfy that barrier (region3 imce_1_2 data from imce_0_3).
      forced = self._get_window_sync()
      if forced != (None, None):
        pre_lines, post_lines = forced
    else:
      pre_lines, post_lines = self._get_window_sync()

    # invariant II bookkeeping (see handshake_num_map): count this LOAD_LB window
    # ONLY when it is a flag-2 producer-gated rendezvous (the dwconv-output
    # Fix-B barrier: pre-lines contain STANDBY(producer, 2)). ResNet8's conv
    # pipeline windows are SETFLAG(1)...SETFLAG(0) with NO STANDBY -> excluded, so
    # the handshake map stays empty for ResNet8 and the (II) assert never fires
    # there. The matching producer barrier (build_mmquant_for_dwconv) is likewise
    # gated on SETFLAG(2).
    if pre_lines and any("STANDBY" in ln and ", 2)" in ln for ln in pre_lines):
      pm = getattr(self.builder, "pair_manager", None) if self.builder is not None else None
      pair = pm.get_pair(self.edge) if pm is not None else None
      if pair is not None:
        self._rz_uuid = pair.uuid
        # the consumer is THIS node = the LOAD_LB receiver of self.edge
        rn = pm._get_hw_node(self.edge.dst_id)
        self._rz_consumer = rn[0].value if isinstance(rn, tuple) else rn.value

    # Max-throughput feed-spread (IMCFLOW_FEED_SPREAD): fifo_id is a compile-time
    # immediate, so spreading requires UNROLLING the bitplane loop with a literal
    # rotated fifo_id per bitplane (see _build_a8af for the same rationale). eff==1
    # (flag off / repeat==1) keeps the classic single-fifo emission byte-identical.
    eff = self.edge_info.effective_spread_n(self.repeat)

    def load_lb_line(iter, fid=load_fifo_id, annot=annotation):
      return f"__builtin_IMCE_LOAD_LB({fid}); // {annot}\n"

    # Sync-granularity (DESIGN §2 invariant II): the inode data-input rendezvous
    # window is `SETFLAG(1); STANDBY(inode, 1); SETFLAG(0)` (get_recv_window_sync
    # inode branch -> the only window carrying a `STANDBY(..., 1)`). The inode
    # SendBlock does ONE such flag-1 handshake PER SEND packet, and each packet
    # (32B) is exactly one LOAD_LB. When `repeat > 1` (dwconv: `repeat` == bitplane
    # count == LOAD_LBs per output-column window) the default "one window around
    # the whole `repeat` burst" gives the imce ONE handshake per `repeat` packets
    # while the inode issues `repeat` handshakes -> a `repeat`:1 rendezvous-freq
    # mismatch that wedges the very first STANDBY (imce_1_1 STANDBY(0,1) hangs;
    # inode never re-raises flag 1). MobileNetV1's first depthwise reads the model
    # input straight from the inode, so this inode-fed dwconv (repeat=4) is the
    # first path to hit it -- ResNet8 conv from the inode has repeat==1 (per-window
    # == per-LOAD_LB, no change), and DS-CNN dwconv is fed by an upstream imce
    # (flag-2 window, `STANDBY(..., 2)`, excluded below). Fix: for the inode flag-1
    # window with repeat>1, emit the window PER LOAD_LB (matching the inode's
    # per-packet handshake).
    is_inode_input_flag1_window = bool(
        pre_lines
        and any("STANDBY" in ln and ", 1)" in ln for ln in pre_lines)
    )
    per_load_lb_window = (
        bugfix_off_mode()
        and self.repeat > 1
        and is_inode_input_flag1_window
    )

    def windowed_load_lb_line(iter, fid=load_fifo_id, annot=annotation):
      body = ""
      if pre_lines:
        body += "\n".join(pre_lines) + "\n"
      body += f"__builtin_IMCE_LOAD_LB({fid}); // {annot}\n"
      if post_lines:
        body += "\n".join(post_lines) + "\n"
      return body

    def _spread_bitplane_lines(line_fn):
      # unroll the `repeat` bitplane loop, one literal rotated fifo_id each.
      out = ""
      for b in range(self.repeat):
        fid = self.edge_info.spread_fifo_id(b, self.repeat)
        out += line_fn(b, fid=fid)
      return out

    def burst_body(iter):
      # one window per outer-loop iteration wrapping the `repeat` LOAD_LBs
      body = ""
      if per_load_lb_window:
        # inode-fed dwconv: one flag-1 handshake PER LOAD_LB to match the inode
        # SendBlock's per-packet rendezvous frequency.
        if self.repeat > 1 and eff > 1:
          body += _spread_bitplane_lines(windowed_load_lb_line) + "\n"
        else:
          body += SimpleFor(self.repeat, windowed_load_lb_line).render() + "\n"
        return body
      if pre_lines:
        body += "\n".join(pre_lines) + "\n"
      if self.repeat > 1 and eff > 1:
        body += _spread_bitplane_lines(load_lb_line) + "\n"
      elif self.repeat > 1:
        body += SimpleFor(self.repeat, load_lb_line).render() + "\n"
      else:
        body += load_lb_line(iter)
      if post_lines:
        body += "\n".join(post_lines) + "\n"
      return body

    self.body = SimpleFor(self.count, burst_body, "load_block")

  def _get_window_sync(self):
    """Return (pre_lines, post_lines) for the LOAD_LB burst window."""
    if self.builder is None or not hasattr(self.builder, 'pair_manager') or self.builder.pair_manager is None:
      return None, None
    return self.builder.pair_manager.get_recv_window_sync(self.edge)

  # ---- a8af (knob=on) fallback: per-LOAD_LB SETFLAG/STANDBY(uuid) sync ----
  def _build_a8af(self):
    load_fifo_id = self.edge_info.fifo_id
    annotation = f"{self.edge}, {self.edge_info.node_info_str}"

    # Max-throughput feed-spread (IMCFLOW_FEED_SPREAD): the LOAD_LB fifo_id is a
    # compile-time immediate (int_IMCE_LOAD_LB ImmArg<0>), so we cannot emit
    # LOAD_LB(iter % n) as a runtime expression -- we UNROLL the inner `repeat`
    # (bitplane) loop and emit a literal rotated fifo_id per bitplane instead of
    # the single pinned fifo_id. eff==1 (flag off / repeat==1) -> the classic
    # single-fifo emission (byte-identical). See DESIGN in TensorEdgeInfo
    # .effective_spread_n: eff divides `repeat`, so the per-pixel bitplane index
    # b matches the INODE SEND's flat-index rotation for the same word.
    eff = self.edge_info.effective_spread_n(self.repeat)

    # Per-packet sync: Load one packet, then sync immediately
    def inner_body_with_sync(iter, fid=load_fifo_id, annot=annotation):
      code = f"__builtin_IMCE_LOAD_LB({fid}); // {annot}\n"
      sync_code = self._get_sync_code_str_a8af()
      if sync_code:
        code += sync_code
      return code

    # Max-throughput lever (IMCFLOW_FEED_PREFETCH): a per-pixel prefetch slot was
    # assigned by the col_group unroll (prefetch_fifo_base). Emit this pixel's
    # `repeat` bitplanes at fifos base..base+repeat-1 (a distinct slot in the
    # P*repeat-wide window) so the NEXT pixel's bitplanes land in different RECV
    # fifos and are resident during the current compute.
    if self.repeat > 1 and self.prefetch_fifo_base is not None:
      base = self.prefetch_fifo_base
      def prefetch_inner(iter):
        body = ""
        for b in range(self.repeat):
          body += inner_body_with_sync(b, fid=(base + b) % 8)
        return body
      self.body = SimpleFor(self.count, prefetch_inner, "load_block_prefetch")
    elif self.repeat > 1 and eff > 1:
      # spread: unroll the bitplane loop, one literal fifo_id per bitplane
      def spread_inner(iter):
        body = ""
        for b in range(self.repeat):
          fid = self.edge_info.spread_fifo_id(b, self.repeat)
          body += inner_body_with_sync(b, fid=fid)
        return body
      self.body = SimpleFor(self.count, spread_inner, "load_block")
    elif self.repeat > 1:
      inner_loop = SimpleFor(self.repeat, inner_body_with_sync)
      self.body = SimpleFor(self.count, inner_loop, "load_block")
    else:
      self.body = SimpleFor(self.count, inner_body_with_sync, "load_block")

  def _get_sync_code_str_a8af(self):
    """a8af per-LOAD_LB RECEIVER-pattern sync (SETFLAG(uuid); STANDBY(sender,uuid); SETFLAG(0))."""
    if self.builder is None or not hasattr(self.builder, 'pair_manager') or self.builder.pair_manager is None:
      return ""
    pair = self.builder.pair_manager.get_pair(self.edge)
    if pair is None:
      return ""
    try:
      dst_gid = self.edge.dst_id.graph_node_id
      if isinstance(dst_gid, tuple):
        dst_gid = dst_gid[-1]
      current_node = DevConfig().get_hw_node(dst_gid)
      if isinstance(current_node, tuple):
        current_node = current_node[0]
    except Exception:
      return ""
    sync_lines = []
    sync_lines.append(f"__builtin_IMCE_SETFLAG({pair.uuid});")
    sync_lines.append(f"__builtin_IMCE_STANDBY({pair.sender_node.value}, {pair.uuid});")
    sync_lines.append(f"__builtin_IMCE_SETFLAG(0);")
    return "\n".join(sync_lines) + "\n"

  def _render(self) -> str:
    add_to_map(self.edge, RecvSendNum("recv", self.count * self.repeat), is_send=False)
    # invariant II: one consumer window per outer load_block iteration
    # (self.count), scaled by the enclosing DWConv row/col SimpleFor nest.
    # knob=off only (handshake_num_map is part of the 934/P4 sync work).
    if bugfix_off_mode() and self._rz_uuid is not None:
      outer = 1 if len(SimpleFor.count_stack) == 0 else int(math.prod(SimpleFor.count_stack))
      entry = handshake_num_map.setdefault(self._rz_uuid, {"producer": 0, "consumer": {}})
      entry["consumer"][self._rz_consumer] = (
          entry["consumer"].get(self._rz_consumer, 0) + self.count * outer)
    return self.body.render()


class RecvConstBlock(ImceCodeBlock):
  """ Code block for receiving constant from given fifo id into a variable """
  # FIXME: Add support for initializing QREGs to zero
  num_in_edges = 1

  class ConstType(Enum):
    MIN = "MIN"
    MAX = "MAX"
    CONFIG = "CONFIG"
    SCAN = "SCAN"
    NORMAL = "NORMAL"

  def __init__(self, in_edge: TensorEdge, type : ConstType, annotation: str = "", recv_count: int = None):
    super().__init__(annotation)
    self.in_edge = in_edge
    self.type = type
    self.recv_map = {}

    te_info = DevConfig().get_tensor_edge_info_with_id_dir(
        self.in_edge.dst_id, "in")  # a hack to get the tensor edge info
    assert te_info, "Tensor edge info not found"

    assert len(te_info) == 1, "Multiple tensor edge infos found for the given edge"
    te_info = te_info[0]

    size = DevConfig().CurrFuncMemLayout.get_data_block_by_edge(self.in_edge).size
    base_addr = DevConfig().CurrFuncMemLayout.get_data_block_by_edge(
        self.in_edge).base_address
    assert base_addr % 32 == 0, "Base address must be a multiple of 32"

    self.te_info = te_info
    # Use provided recv_count if given, otherwise calculate from size
    if recv_count is not None:
      self.recv_count = recv_count
    else:
      self.recv_count = math.ceil(size / 32.0)  # recv operates on 32-byte word

  def _render(self) -> str:
    owner_edge = self.te_info.owner
    add_to_map(owner_edge, RecvSendNum("recv",  self.recv_count), is_send=False)
    code = TextBlock("")
    for i in range(self.recv_count):
      var = UniqueVar((self.in_edge, i))
      var.set_static()
      if self.type == RecvConstBlock.ConstType.MIN:
        code += f"__builtin_IMCE_RECV_MIN({self.te_info.fifo_id});"
      elif self.type == RecvConstBlock.ConstType.MAX:
        code += f"__builtin_IMCE_RECV_MAX({self.te_info.fifo_id});"
      elif self.type == RecvConstBlock.ConstType.CONFIG:
        code += f"__builtin_IMCE_RECV_CFG({self.te_info.fifo_id});"
      elif self.type == RecvConstBlock.ConstType.SCAN:
        code += f"{var}_{i} = __builtin_IMCE_RECV({self.te_info.fifo_id});"
        code += f"{var}_{i} = __builtin_IMCE_SCAN_RW({var}_{i});"
      elif self.type == RecvConstBlock.ConstType.NORMAL:
        code += f"{var} = __builtin_IMCE_RECV({self.te_info.fifo_id});"
      else:
        raise ValueError(f"Unknown ConstType: {self.type}")
    return code.render()


class IMCERecvBlock(ImceCodeBlock):
  """
  Code block for receiving data from a fifo.
  Calls add_to_map during _render() so that count_stack is properly populated.
  """
  def __init__(self, var_name: str, fifo_id: int, owner_edge: TensorEdge, annotation: str = ""):
    super().__init__(annotation)
    self.var_name = var_name
    self.fifo_id = fifo_id
    self.owner_edge = owner_edge

  def _render(self) -> str:
    # Call add_to_map at render time when count_stack is properly populated
    add_to_map(self.owner_edge, RecvSendNum("recv", 1), is_send=False)
    return f"{self.var_name} = __builtin_IMCE_RECV({self.fifo_id}); // {self.annotation}"

class IMCESendBlock(ImceCodeBlock):
  """
  Code block for sending data to a fifo.
  Calls add_to_map during _render() so that count_stack is properly populated.
  """
  def __init__(self, var_name: str, fifo_id: int, policy_addr : int, owner_edges: TensorEdge, annotation: str = ""):
    super().__init__(annotation)
    self.var_name = var_name
    self.fifo_id = fifo_id
    self.policy_addr = policy_addr
    # list if multicast
    self.owner_edges = owner_edges if isinstance(owner_edges, list) else [owner_edges]

  def _render(self) -> str:
    # Call add_to_map at render time when count_stack is properly populated
    for owner_edge in self.owner_edges:
      add_to_map(owner_edge, RecvSendNum("send", 1), is_send=True)
    return f"__builtin_IMCE_SEND({self.policy_addr}, {self.var_name}, {self.fifo_id}, 0); // {self.annotation}"

class VecBlock(ImceCallCodeBlock):
  """
  VecBlock is base class for implementing R,I type vector operations.
  Only generates computation. RECV/SEND handled by wrapper or ConvBlock.
  """
  num_in_edges = 2

  def __init__(self, call: 'BuilderContext', annotation: str = ""):
    """ Code block for vector operations """
    super().__init__(call, annotation)
    self.op_name = self._op_name()
    self.imm_value = self._get_imm_value()
    # Constant operand info (set by handler when one operand is constant)
    self.const_edge = None  # TensorEdge for RecvConstBlock
    self.const_arg_idx = None  # Argument index where constant appears (0 or 1)
    # Overridable num_blocks for standalone wrapper (set by create_loop_from_call)
    self._num_blocks = None
    self._num_out_blocks = None

  @property
  def num_blocks(self) -> int:
    """Return num_blocks, using override if set by wrapper."""
    if self._num_blocks is not None:
      return self._num_blocks
    return super().num_blocks

  @property
  def num_out_blocks(self) -> int:
    """Return num_out_blocks, using override if set by wrapper."""
    if self._num_out_blocks is not None:
      return self._num_out_blocks
    return super().num_out_blocks

  def set_const_info(self, const_edge: TensorEdge, arg_idx: int):
    """Set constant operand information.

    Args:
        const_edge: The TensorEdge for the constant operand (used by RecvConstBlock)
        arg_idx: The argument index where the constant appears (0 or 1)
    """
    self.const_edge = const_edge
    self.const_arg_idx = arg_idx

  @abstractmethod
  def _get_imm_value(self) -> int:
    pass

  @abstractmethod
  def _op_name(self) -> str:
    pass

  def _build_var_ins(self, i: int) -> list:
    """Build input variable list for block iteration i, handling constants.

    Handles three cases:
    1. Both operands from in_edges (no constant)
    2. One operand from in_edges, one constant
    3. One operand from prev_op (VecOpBlock chaining), one constant
    """
    # Build variables from non-constant edges only
    non_const_edges = [edge for edge in self.in_edges if edge != self.const_edge]
    var_ins_from_edges = [self._make_unique_input_var_for_post_op(
        edge, i) for edge in non_const_edges]

    if self.const_edge is not None:
      const_var = UniqueVar((self.const_edge, i))
      # Determine the non-constant operand variable
      if var_ins_from_edges:
        # Case 2: One operand from non-constant in_edges
        non_const_var = var_ins_from_edges[0]
      elif self.prev_op is not None:
        # Case 3: No non-constant in_edges but prev_op exists (VecOpBlock chaining)
        non_const_var = UniqueVar((self.prev_op, i))
      else:
        # Should not happen - no input source available
        raise ValueError(f"VecBlock {self} has const_edge but no input source")

      if self.const_arg_idx == 0:
        return [const_var, non_const_var]
      else:
        return [non_const_var, const_var]
    return var_ins_from_edges

  def _render(self) -> str:
    """Generate only computation, no RECV/SEND."""
    code = TextBlock("")

    for i in range(self.num_blocks):
      var_ins = self._build_var_ins(i)
      var_o = UniqueVar((self, i))
      var_in_str = ", ".join([f"{var_i}" for var_i in var_ins])
      # e.g. __builtin_IMCE_ADD(a, b, 15);
      code += f"{var_o} = __builtin_IMCE_{self.op_name}({var_in_str}, {self.imm_value});"

    return code.render()


class AddBlock(VecBlock):
  def _get_imm_value(self) -> int:
    return 15  # src_mask

  def _op_name(self) -> str:
    return "ADD"


class DivBlock(VecBlock):
  def _get_imm_value(self) -> int:
    return 15  # src_mask

  def _op_name(self) -> str:
    return "DIV"


class MultlBlock(VecBlock):
  """MULTL operation block with optional SW fix for hardware overflow bug.

  If IMCFLOW_BUGFIX_OVERFLOW_SW=1, uses MultlSWFixOp for overflow bug fix.
  Otherwise, uses simple MULTL instruction (inherited from VecBlock).
  """

  def _get_imm_value(self) -> int:
    return 15  # src_mask

  def _op_name(self) -> str:
    return "MULTL"

  def _render(self) -> str:
    """Generate MULTL with optional SW fix for overflow bug."""
    if not _is_multl_swfix_enabled():
      # Use standard VecBlock rendering
      return super()._render()

    # SW fix enabled: use MultlSWFixOp
    code = TextBlock("")

    for i in range(self.num_blocks):
      var_ins = self._build_var_ins(i)
      var_o = UniqueVar((self, i))
      code += MultlSWFixOp(var_ins[0], var_ins[1], var_o, (self, i), self.imm_value)

    return code.render()


class MulthBlock(VecBlock):
  def _get_imm_value(self) -> int:
    return 15  # src_mask

  def _op_name(self) -> str:
    return "MULTH"


class MultlSWFixOp(ImceCodeBlock):
  """Composable MULTL with SW fix for hardware overflow bug.

  MULTL bug software fix algorithm (14-bit immediate constraint):
  - Overflow occurs when H and L signs mismatch
  - Replace with saturate value on overflow (H_sign XOR 0x7FFF)
  - Uses only immediate values 1 and 15 (within 14-bit range)

  Can be used standalone or composed into other blocks.
  """

  def __init__(self, var_a, var_b, var_out, block_id, imm_value: int = 15, annotation: str = ""):
    """Initialize MultlSWFixOp.

    Args:
        var_a: First input variable (UniqueVar or str)
        var_b: Second input variable (UniqueVar or str)
        var_out: Output variable (UniqueVar or str)
        block_id: Unique identifier for intermediate variables (e.g., (self, i))
        imm_value: Immediate value for src_mask (default 15)
        annotation: Optional annotation string
    """
    super().__init__(annotation)
    self.var_a = var_a
    self.var_b = var_b
    self.var_out = var_out
    self.block_id = block_id
    self.imm_value = imm_value

  def _render(self) -> str:
    code = TextBlock("")

    # Intermediate variables (unique per block_id)
    var_L = UniqueVar((self.block_id, "L"))
    var_H = UniqueVar((self.block_id, "H"))
    var_neg1 = UniqueVar((self.block_id, "neg1"))
    var_const_7fff = UniqueVar((self.block_id, "const_7fff"))
    var_H_sign = UniqueVar((self.block_id, "H_sign"))
    var_L_sign = UniqueVar((self.block_id, "L_sign"))
    var_mismatch = UniqueVar((self.block_id, "mismatch"))
    var_saturate_val = UniqueVar((self.block_id, "saturate_val"))
    var_not_mismatch = UniqueVar((self.block_id, "not_mismatch"))
    var_part1 = UniqueVar((self.block_id, "part1"))
    var_part2 = UniqueVar((self.block_id, "part2"))

    # MULTL/MULTH operations
    code += f"{var_L} = __builtin_IMCE_MULTL({self.var_a}, {self.var_b}, {self.imm_value});"
    code += f"{var_H} = __builtin_IMCE_MULTH({self.var_a}, {self.var_b}, {self.imm_value});"

    # Constant generation (uses zero register directly, immediate 1 only)
    code += f"{var_neg1} = __builtin_IMCE_SUBI(0, 1);"  # 0 - 1 = -1 (0xFFFF), uses zero register
    code += f"{var_const_7fff} = __builtin_IMCE_SRLI({var_neg1}, 1);"  # -1 >>> 1 = 0x7FFF (32767)

    # Sign extraction
    code += f"{var_H_sign} = __builtin_IMCE_SRAI({var_H}, 15);"  # -1 or 0
    code += f"{var_L_sign} = __builtin_IMCE_SRAI({var_L}, 15);"  # -1 or 0

    # Detect sign mismatch
    code += f"{var_mismatch} = __builtin_IMCE_XOR({var_H_sign}, {var_L_sign}, {self.imm_value});"

    # Determine saturate value
    code += f"{var_saturate_val} = __builtin_IMCE_XOR({var_H_sign}, {var_const_7fff}, {self.imm_value});"

    # Masking and synthesis
    code += f"{var_not_mismatch} = __builtin_IMCE_XOR({var_mismatch}, {var_neg1}, {self.imm_value});"
    code += f"{var_part1} = __builtin_IMCE_AND({var_mismatch}, {var_saturate_val}, {self.imm_value});"
    code += f"{var_part2} = __builtin_IMCE_AND({var_not_mismatch}, {var_L}, {self.imm_value});"
    code += f"{self.var_out} = __builtin_IMCE_OR({var_part1}, {var_part2}, {self.imm_value});"

    return code.render()


class ReLUBlock(VecBlock):
  num_in_edges = 1

  def _get_imm_value(self) -> int:
    return 0  # immediate value for MAXI

  def _op_name(self) -> str:
    return "MAXI"


class MinmaxQuantBlock(ImceCallCodeBlock):
  """
  MinmaxQuantBlock for min/max quantization operations.
  Only generates computation. RECV/SEND handled by wrapper or ConvBlock.
  """
  num_in_edges = 3

  def __init__(self, call: 'BuilderContext', o_split_idx: int, annotation: str = ""):
    """ Code block for min/max quantization """
    super().__init__(call, annotation)
    self._num_blocks = 4
    self.channels = None
    self.o_split_idx = o_split_idx

  @property
  def num_blocks(self) -> int:
    return self._num_blocks

  @property
  def num_out_blocks(self) -> int:
    return 4  # FIXED in MinmaxQuantBlock

  def consumer_is_non_multicast_split(self):
    assert len(self.out_edges) == 1, "Only one output edge is expected"
    out_edge = self.out_edges[0]
    dst_gid = out_edge.dst_id.graph_node_id
    dst_node = CustomIDToNode()[getInnerNodeID(dst_gid)]
    if isinstance(dst_node, relay.Call) and dst_node.op.name == "split":
      split_info = DevConfig().SplitInfo[self.call.func_name][getNodeID(dst_node)]
      is_multicast = split_info["is_multi_cast"]
      channels = split_info["channels"]
      num_splits = split_info["num_splits"]
      return (not is_multicast), channels, num_splits
    else:
      return False, None, None
  
  def get_split_consumer_edge_info(self):
    assert len(self.out_edges) == 1, "Only one output edge is expected"
    out_edge = self.out_edges[0]
    dst_gid = out_edge.dst_id.graph_node_id
    dst_node = CustomIDToNode()[getInnerNodeID(dst_gid)]
    assert isinstance(dst_node, relay.Call) and dst_node.op.name == "split", "Output consumer must be a split"

    split_out_edges = DevConfig().get_tensor_edges_from_graph_node_id(
      graph_node_id=dst_gid,
      dir="out"
    )

    # Sort by split_idx so that edges[i] corresponds to split.i (ch_group i)
    split_out_edges.sort(key=lambda e: e.split_idx if e.split_idx is not None else 0)

    split_out_edge_infos = [
      DevConfig().get_tensor_edge_info(out_edge) for out_edge in split_out_edges
    ]

    return split_out_edges, split_out_edge_infos
  
  def build_mmquant_for_dwconv(self):
    code = SequentialBlock()
    is_non_multicast_split, channels, num_splits = self.consumer_is_non_multicast_split()
    ch_group_nums = math.ceil(channels / 32)

    call_node = self.call.call
    arg = call_node.args[0]
    layout = DevConfig().LayoutMap[arg]
    vir_ttype = get_type(self.call.module, arg)
    N, C, H, W = vir_ttype.shape
    assert channels == C.value, "Channel size mismatch"
    
    for in_edge in self.in_edges:
      if in_edge.dst_id.tensor_type == "data":
        break
    in_edge_info = DevConfig().get_tensor_edge_info(in_edge)
    out_edge = self.out_edges[0]
    split_out_edges, split_out_edge_infos = self.get_split_consumer_edge_info()
    out_edge_info = DevConfig().get_tensor_edge_info(out_edge)
    loop_cnt = N * H * W

    # BUGFIX knob: knob=on (bugfix_off_mode()==False) reproduces a8af's bare
    # RECV/compute/SEND loop (no P3 input window, no P4 output barrier, global
    # (self,i) qreg vars). knob=off keeps the 934/P3/P4 rendezvous below.
    if not bugfix_off_mode():
      for ch_group_idx in range(ch_group_nums):
        # recv
        for i in range(2): # 32 -> 16 x 2
          code += IMCERecvBlock(
            var_name=UniqueVar((in_edge, 2*ch_group_idx+i)),
            fifo_id=in_edge_info.fifo_id,
            owner_edge=in_edge,
            annotation=f"dwconv minmaxquant recv ch_group {ch_group_idx}, part {i}"
          )
        # compute
        src_mask = 15
        for i in range(2):
          qreg_idx = i
          var_i = UniqueVar((in_edge, 2*ch_group_idx+i))
          var_o = UniqueVar((self, 2*ch_group_idx+i))
          code += f"__builtin_IMCE_MM_QUANT({var_i}, 0, {src_mask}, {qreg_idx});"
        # get_qreg
        for i in range(4):
          var_o = UniqueVar((self, i))
          code += f"{var_o} = __builtin_IMCE_GET_QREG({i});"
        # send
        for i in range(4):
          split_out_edge_info = split_out_edge_infos[ch_group_idx]
          code += IMCESendBlock(
            var_name=UniqueVar((self, i)),
            fifo_id=split_out_edge_info.fifo_id,
            policy_addr=split_out_edge_info.policy_info[0].address,
            owner_edges=split_out_edges[ch_group_idx],
            annotation=f"dwconv minmaxquant send ch_group {ch_group_idx}, part {i}"
          )
      return SimpleFor(loop_cnt, code, "dwconv_minmaxquant_loop")

    # P3 (DESIGN §2.3): inode->imce data-input rendezvous window on the dwconv
    # minmaxquant RECV. The inode SEND loop does one handshake per packet; each
    # imce RECV consumes exactly one packet, so the matching receiver window
    # (SETFLAG(1);STANDBY(inode,1);SETFLAG(0)) must wrap EACH RECV (contract
    # consumer_recv_per_sync=1). Emitting it per-pixel instead of per-RECV
    # desynchronizes the handshake (the input-starvation deadlock). The window
    # comes from the edge's contract via get_recv_window_sync (same source a
    # plain conv uses); bare if the edge is not an inode data input.
    recv_per_sync = getattr(in_edge_info, "consumer_recv_per_sync", None) or 1
    _pm = getattr(getattr(self.call, "builder", None), "pair_manager", None)
    recv_pre, recv_post = (None, None)
    if _pm is not None:
      recv_pre, recv_post = _pm.get_recv_window_sync(in_edge)

    # P4 (DESIGN §5-P4): producer-side output rendezvous on the dwconv
    # minmaxquant SEND. imce_0_1's split odata is MULTICAST to two imce receivers
    # (imce_0_2 and the real DWCONV consumer imce_1_1). Both wrap their LOAD_LB in
    # a flag-2 window (SETFLAG(2); STANDBY(imce_0_1,2); SETFLAG(0)) but the
    # producer never raised flag 2 -> both STANDBYs wedge (the middle-stage
    # deadlock). The matching producer barrier comes from the SAME pair via
    # get_pre_send_sync (a 2-phase flag-2 rendezvous:
    #   STANDBY(r,2)xN; SETFLAG(2); STANDBY(r,0)xN; SETFLAG(0)).
    # It fires ONCE per pixel (== one window per pixel on each receiver) and MUST
    # precede the SENDs: each receiver's window CLOSES (SETFLAG(0)) before its
    # LOAD_LB, so the producer's phase-down completes without waiting on the data
    # transfer, then the 8 SENDs flow under fifo backpressure (depth 2). Because
    # both split edges share the pair, both give the same barrier -> emit it once.
    # A single global flag slot per node makes this a node-level rendezvous, so
    # both ch_groups' SENDs are covered by the one barrier.
    send_barrier = None
    if _pm is not None:
      send_barrier = _pm.get_pre_send_sync(split_out_edges[0])

    recv_idx = 0  # counts RECVs to place a window every recv_per_sync RECVs
    for ch_group_idx in range(ch_group_nums):
      # recv
      for i in range(2): # 32 -> 16 x 2
        if recv_pre and (recv_idx % recv_per_sync == 0):
          code += "\n".join(recv_pre) + "\n"
        code += IMCERecvBlock(
          var_name=UniqueVar((in_edge, 2*ch_group_idx+i)),
          fifo_id=in_edge_info.fifo_id,
          owner_edge=in_edge,
          annotation=f"dwconv minmaxquant recv ch_group {ch_group_idx}, part {i}"
        )
        if recv_post and (recv_idx % recv_per_sync == recv_per_sync - 1):
          code += "\n".join(recv_post) + "\n"
        recv_idx += 1

      # compute
      src_mask = 15
      for i in range(2):
        qreg_idx = i
        var_i = UniqueVar((in_edge, 2*ch_group_idx+i))
        var_o = UniqueVar((self, 2*ch_group_idx+i))
        code += f"__builtin_IMCE_MM_QUANT({var_i}, 0, {src_mask}, {qreg_idx});"

      # get_qreg -- into per-ch_group vars so the SENDs can be hoisted after the
      # one-per-pixel output barrier without qreg aliasing across ch_groups.
      for i in range(4):
        var_o = UniqueVar((self, ch_group_idx, i))
        code += f"{var_o} = __builtin_IMCE_GET_QREG({i});"

    # P4: one output rendezvous per pixel, then all SENDs (see comment above).
    if send_barrier:
      code += "\n".join(send_barrier) + "\n"
      # invariant II: record loop_cnt producer handshakes for this pair (the
      # barrier is emitted once here but runs once per pixel inside the
      # dwconv_minmaxquant_loop, which is not yet on the count_stack). Gated on a
      # real flag-rendezvous barrier (SETFLAG appears) so a bare [] send_barrier
      # records nothing.
      _pair0 = _pm.get_pair(split_out_edges[0]) if _pm is not None else None
      if _pair0 is not None and any("SETFLAG(2)" in ln for ln in send_barrier):
        handshake_num_map.setdefault(_pair0.uuid, {"producer": 0, "consumer": {}})
        handshake_num_map[_pair0.uuid]["producer"] += loop_cnt

    for ch_group_idx in range(ch_group_nums):
      for i in range(4):
        split_out_edge_info = split_out_edge_infos[ch_group_idx]
        code += IMCESendBlock(
          var_name=UniqueVar((self, ch_group_idx, i)),
          fifo_id=split_out_edge_info.fifo_id,
          policy_addr=split_out_edge_info.policy_info[0].address,
          owner_edges=split_out_edges[ch_group_idx],
          annotation=f"dwconv minmaxquant send ch_group {ch_group_idx}, part {i}"
        )

    for_code = SimpleFor(loop_cnt, code, "dwconv_minmaxquant_loop")

    return for_code

  def _render(self) -> str:
    """Generate only computation, no RECV/SEND."""
    # Find data edge from in_edges or use prev_op if inside composite
    data_edge = next(
        (edge for edge in self.in_edges if edge.dst_id.tensor_type == "data"),
        None
    )
    if data_edge is None and self.prev_op is not None:
      # Inside composite: use prev_op's output as data source
      data_edge = self.prev_op

    code = TextBlock("")

    is_non_multicast_split, _, _ = self.consumer_is_non_multicast_split()
    assert is_non_multicast_split==False, "MinmaxQuantBlock for non-multicast split is handled separately."

    # Determine channels: from explicit setting, prev_op, or default to 64
    channels = self.channels
    if channels is None and self.prev_op is not None:
      channels = getattr(self.prev_op, 'channels', None) or getattr(self.prev_op, 'num_out_blocks', 4) * 16
    if channels is None:
      channels = 64  # Default

    #TODO: maybe we need to clear qreg before
    src_masks = [min(15, channels - i - 1) for i in range(0, channels, 16)]

    for i, src_mask in enumerate(src_masks):
      var_i = self._make_unique_input_var_for_post_op(data_edge, i)
      qreg_start_idx = i + 4 * self.o_split_idx
      code += f"__builtin_IMCE_MM_QUANT({var_i}, 0, {src_mask}, {qreg_start_idx});"

    # NOTE: currently, it is not possible to have consequtive 4*(MM_QUANT -> QREG)s.
    # Instead of the below code,
    #  __builtin_IMCE_MM_QUANT(var267, 0, 15, 0);
    #  var274 = __builtin_IMCE_GET_QREG(0);
    #  __builtin_IMCE_MM_QUANT(var269, 0, 15, 1);
    #  var275 = __builtin_IMCE_GET_QREG(1);
    #  __builtin_IMCE_MM_QUANT(var271, 0, 15, 2);
    #  var276 = __builtin_IMCE_GET_QREG(2);
    #  __builtin_IMCE_MM_QUANT(var273, 0, 15, 3);
    #  var277 = __builtin_IMCE_GET_QREG(3);
    # We put MM_QUANTs block first, then GET_QREGS.
    # Otherwise, it results in llvm artifact of moving qregs into vector registers.
    # e.g. vaddi %v3 %qreg2 0
    for i in range(self.num_out_blocks):
      # Get QREG result for this block
      var_o = UniqueVar((self, i))
      code += f"{var_o} = __builtin_IMCE_GET_QREG({i});"

    return code.render()


class ConcatBlock(ImceCallCodeBlock):
  min_in_edges = 2

  """
  Code block for concatenating multiple tensors
  FIXME: needs to look upon, since concat can happen not only in bitplanes...
         concat is occured in channel axis. it means store data in register file.
         We don't need to OR at this situation.
  """

  def __init__(self, call: 'BuilderContext', annotation: str = ""):
    """ Code block for min/max quantization """
    super().__init__(call, annotation)
    self.or_concat=None
    self.channel=None
    self.channel_block_size=None
    self.set_type()
    self.set_channel()
    self.set_channel_block_size()
    assert len(
        self.in_edges) >= self.min_in_edges, "At least two input edges are required"
  
  def set_type(self):
    """
      If input type is C16 vector, CONCAT.
      If conv_input, OR_CONCAT
    """
    last_arg = self.call.call.args[0].fields[-1]
    assert isinstance(last_arg, relay.Call), "Last argument must be a relay.Call"
    if last_arg.op.name == "qnn.imcflow_min_max_quantize":
      self.or_concat = 1
    else:
      self.or_concat = 0
    print(f"[ConcatBlock] or_concat: {self.or_concat}")
  
  def set_channel(self):
    call_node = self.call.call
    vir_ttype = get_type(self.call.module, call_node)
    N, C, H, W= vir_ttype.shape
    self.channel = C.value
    print(f"[ConcatBlock] channel: {self.channel}")
  
  def set_channel_block_size(self):
    last_arg = self.call.call.args[0].fields[-1]
    layout = DevConfig().LayoutMap[last_arg]
    if layout.value == "NCHW16c":
      self.channel_block_size = 16
    elif layout.value == "NCHW64c":
      self.channel_block_size = 64
    elif layout.value == "NHWC16c":
      self.channel_block_size = 16
    elif layout.value == "NHWC64c":
      self.channel_block_size = 64

  
  @property
  def num_blocks(self) -> int:
    if self.or_concat:
      return 4 * len(self.in_edges)  # input bitplanes : 4
    else:
      return math.ceil(self.channel/16) // len(self.in_edges)
  
  @property
  def num_out_blocks(self) -> int:
    if self.or_concat:
      return 4 # input bitplanes : 4
    else:
      return math.ceil(self.channel/16)

  def _render(self) -> str:
    num_bitplanes = 4
    src_mask = 15

    code = TextBlock("")

    external_in_edges = [e for e in self.in_edges if e in DevConfig().TensorEdgetoInfo]
    internal_in_edge = (set(self.in_edges) - set(external_in_edges))
    in_edges_sorted = []
    for arg in self.call.call.args[0].fields:
      if isinstance(arg, relay.Var):
        arg = self.call.conv_pending_info["param_to_arg"][arg]
        if isinstance(arg, relay.Call) and isinstance(arg.op, tvm.ir.Op): # normal call
          pass
        elif isinstance(arg, relay.Call) and isinstance(arg.op, relay.Function): # composite call
          arg = arg.op.body
        else:
          raise RuntimeError("unexpected arg type in concat")
      for e in self.in_edges:
        if getInnerNodeID(e.src_id.graph_node_id) == getNodeID(arg):
          in_edges_sorted.append(e)
          break

    assert self.call.curr_composite_id is not None, "standalone concat will be handled at handler"
    if not self.or_concat: # input is vector form
      for in_edge in in_edges_sorted:
        for b in range(self.num_blocks):
          var_i = self._make_unique_input_var_for_post_op(in_edge, b)
          var_o = UniqueVar((self, b + self.num_blocks * in_edges_sorted.index(in_edge)))
          code += f"{var_o} = {var_i};"
    else: # input is bitplane form
      external_in_edges = [e for e in self.in_edges if e in DevConfig().TensorEdgetoInfo]
      internal_in_edge = (set(self.in_edges) - set(external_in_edges)).pop()
      assert len(internal_in_edge) == 0, "no internal edge is expected in OR concat"
      for i in range(num_bitplanes):
        var_o = UniqueVar((self, i))
        for ext_edge in external_in_edges:
          var_e = UniqueVar((ext_edge, i))
          fifo_id = DevConfig().get_tensor_edge_info(ext_edge).fifo_id
          code += f"{var_e} = __builtin_IMCE_RECV({fifo_id});"
          code += f"{var_o} = __builtin_IMCE_OR({var_i}, {var_e}, {src_mask});"

    return code.render()

class SplitBlock(ImceCallCodeBlock):
  num_in_edges = 1

  """ Code block for splitting a tensor into multiple tensors """

  def __init__(self, call: 'BuilderContext', annotation: str = ""):
    super().__init__(call, annotation)
    first_policies = [DevConfig().get_tensor_edge_info(
        out_edge).policy_info[0] for out_edge in self.out_edges]
    fifo_ids = [DevConfig().get_tensor_edge_info(
        out_edge).fifo_id for out_edge in self.out_edges]
    assert all(policy == first_policies[0]
               for policy in first_policies), "All output edges must have the same first policy info"
    assert all(fid == fifo_ids[0]
               for fid in fifo_ids), "All output edges must have the same fifo id"

  def _render(self) -> str:
    return ""


class ConvBlock(ImceCallCodeBlock):
  """ Code block for receiving conv input data from given fifo id """
  num_in_edges = 3

  def __init__(self, call: 'BuilderContext', shapes: dict, conv_attrs: Conv2DAttrs,
               post_ops: List[ImceCallCodeBlock] = None, annotation: str = ""):
    super().__init__(call, annotation)
    self.builder = call.builder  # Store builder reference for sync support
    self.conv = ConvUtil(shapes["data"][2], shapes["data"][3],
                         conv_attrs.padding[0], conv_attrs.strides[0],
                         conv_attrs.kernel_size[0], conv_attrs.kernel_size[1])
    self.post_ops = post_ops if post_ops is not None else []
    self.total_in_read_counts = self.conv.get_total_input_read_counts()
    self.origin_hw = shapes["data"][2] * shapes["data"][3]  # H * W
    self.remain = self.origin_hw - self.total_in_read_counts

    self.out_channels = conv_attrs.channels.value
    assert self.out_channels <= 64, "convolution output channel is greater than 64"
    # assert self.out_channels%16 == 0, "Until now, conv output channel should be multiple of 16"
    self.in_channels  = conv_attrs.in_channels.value

    # Link post-ops
    prev = self
    for op in self.post_ops:
      op.prev_op = prev
      prev = op

    # Marker A (RTL-derived boundary-STEP reorder) discriminator.
    # has_noc_rhs == True when a post_op (e.g. fused add) takes one operand over
    # an INTER-NODE NoC edge from a DIFFERENT imce (the add-rhs of region2
    # imce_3_2 arrives from imce_2_2). With BUGFIX-off imce_ctrl, a stall-capable
    # NoC RECV sitting between the boundary col_group's last LOAD_LB and its STEP
    # lets valid drop mid-stall and mis-consume hs_remaining -> stale STEP latch
    # -> deadlock. handcraft avoids it by doing the boundary col_group's final
    # LOAD_LB AFTER that RECV window (right before STEP). We reproduce that only
    # when has_noc_rhs (so region1's local-vec post-ops -- Multl/BN/Minmax with no
    # inter-node rhs -- are provably excluded and keep their default ordering).
    #
    # BUGFIX knob: knob=on (bugfix_off_mode()==False) reproduces a8af, which had
    # no Marker A logic -> leave these inert (False/None) and dispatch
    # _build_structure/_build_loop_body to the a8af path below.
    if bugfix_off_mode():
      self.has_noc_rhs = self._compute_has_noc_rhs()
      # Marker A sibling case (imce_2_2): a PLAIN conv (no post_op) whose output
      # is the inter-node NoC `rhs` of a downstream fused add.
      self.noc_rhs_sender_receiver = self._compute_noc_rhs_sender_receiver()
      self.body = self._build_structure()
    else:
      self.has_noc_rhs = False
      self.noc_rhs_sender_receiver = None
      self.body = self._build_structure_a8af()

  def _compute_noc_rhs_sender_receiver(self):
    """Return the receiver hw node value if this PLAIN conv feeds a fused add's
    inter-node NoC `rhs`, else None. Used for the imce_2_2 boundary reorder."""
    if self.post_ops:
      return None  # only plain convs (imce_2_2), fused convs use has_noc_rhs
    try:
      my_hw = DevConfig().get_hw_node(self.get_graph_node_id())
    except Exception:
      my_hw = None
    for edge in self.out_edges:
      dst_gid = edge.dst_id.graph_node_id
      if not isinstance(dst_gid, tuple):
        continue  # plain consumer, not fused
      if edge.dst_id.tensor_type != "rhs":
        continue  # only the fused add rhs slot
      te_info = DevConfig().TensorEdgetoInfo.get(edge, None)
      if te_info is None or te_info.fifo_id == TensorEdgeInfo.LOCAL_FIFO:
        continue
      try:
        dst_hw = DevConfig().get_hw_node(dst_gid)
      except Exception:
        continue
      if isinstance(dst_hw, tuple):
        dst_hw = dst_hw[0]
      if dst_hw is not None and dst_hw.is_imce() and dst_hw != my_hw:
        print(f"[Marker A sibling] {self.annotation}: sends NoC rhs to fused "
              f"consumer imce {dst_hw} via {edge}")
        return dst_hw
    return None

  def _compute_has_noc_rhs(self) -> bool:
    """True iff a post_op consumes an inter-node NoC operand from another imce."""
    if not self.post_ops:
      return False
    # edges produced internally by conv or any post_op (conv->add etc.)
    internal_edges = set()
    for op in [self] + self.post_ops:
      internal_edges.update(op.out_edges)
    try:
      my_hw = DevConfig().get_hw_node(self.get_graph_node_id())
    except Exception:
      my_hw = None
    for op in self.post_ops:
      for edge in op.in_edges:
        if edge in internal_edges:
          continue  # internal (conv->add) edge, not NoC
        # constant operand -> local, not NoC
        try:
          if ConstPat.match(CustomIDToNode()[edge.src_id.graph_node_id]):
            continue
        except KeyError:
          pass
        te_info = DevConfig().TensorEdgetoInfo.get(edge, None)
        if te_info is None or te_info.fifo_id == TensorEdgeInfo.LOCAL_FIFO:
          continue  # same-hw-node (local) operand, not inter-node NoC
        try:
          src_hw = DevConfig().get_hw_node(edge.src_id.graph_node_id)
        except Exception:
          continue
        if isinstance(src_hw, tuple):
          src_hw = src_hw[0]
        # inter-node operand from a DIFFERENT imce -> fused NoC rhs
        if src_hw is not None and src_hw.is_imce() and src_hw != my_hw:
          print(f"[Marker A] {self.annotation}: has_noc_rhs via edge {edge} "
                f"(src imce {src_hw})")
          return True
    return False

  @property
  def num_blocks(self) -> int:
    # conv have to recieve 4 bit planes.
    return 4  # FIXED in ConvBlock

  @property
  def num_out_blocks(self) -> int:
    # return self.out_channels//16
    return 4

  def _prefetch_group_for_load_edge(self):
    """Return (P, width) prefetch group for the conv activation load edge, or None
    (see TensorEdgeInfo.prefetch_group). Used to unroll the col_group loop for
    IMCFLOW_FEED_PREFETCH. repeat == num_blocks (4 conv bitplanes)."""
    for edge in self.in_edges:
      try:
        arg_id = edge.src_id.graph_node_id
        if ConstPat.match(CustomIDToNode()[arg_id]):
          continue
      except KeyError:
        pass
      te_infos = DevConfig().get_tensor_edge_info_with_id_dir(edge.dst_id, "in")
      if not te_infos:
        continue
      te_info = te_infos[0]
      if te_info is None or te_info.fifo_id == TensorEdgeInfo.LOCAL_FIFO:
        continue
      return te_info.prefetch_group(self.num_blocks)
    return None

  # -------- a8af (knob=on) fallback: no Marker A / padding-drain / window sync --
  def _build_loop_body_a8af(self, recv_count: int, prefetch_fifo_base: int = None) -> CodeBlock:
    load_info = []
    for edge in self.in_edges:
      te_infos = DevConfig().get_tensor_edge_info_with_id_dir(edge.dst_id, "in")
      assert len(te_infos) == 1, "more than one te_info found!"
      te_info = te_infos[0]
      try:
        arg_id = edge.src_id.graph_node_id
        if ConstPat.match(CustomIDToNode()[arg_id]):
          continue
      except KeyError:
        pass
      load_info.append({"edge": edge, "te_info": te_info})

    assert len(load_info) == 1, "there should be exactly one load edge"
    load_edge = load_info[0]["edge"]
    load_edge_info = load_info[0]["te_info"]
    self.load_edge_info = load_edge_info

    comp = SequentialBlock()
    comp.add(LoadLBBlock(recv_count, self.num_blocks, load_edge, load_edge_info, builder=self.builder, prefetch_fifo_base=prefetch_fifo_base))
    comp.add(TextBlock("__builtin_IMCE_STEP();\n"))

    # Max-throughput lever (IMCFLOW_DROP_PSUM): DON'T-CARE output. The post_imcu
    # out_fifo is drained by OP_STEP itself (imce_ctrl.sv:69
    # compute_if.ready==OP_STEP), NOT by GET_CREG; GET_CREG only reads the latched
    # creg from the regfile and SEND ships it. So for garbage output we can drop
    # BOTH GET_CREG and SEND without touching the STEP<->feed handshake. Only for
    # a plain conv (no post_ops consume the creg vars). Gated by env; default OFF.
    _drop_psum = drop_psum_send() and not self.post_ops
    if not _drop_psum:
      creg_code = TextBlock("")
      for i in range(self.num_out_blocks):
        var_o = UniqueVar((self, i))
        creg_code += f"{var_o} = __builtin_IMCE_GET_CREG((short){i});"
      comp.add(creg_code)
    else:
      comp.add(TextBlock("// [DROP_PSUM] omitted 4x GET_CREG\n"))

    if self.post_ops:
      all_in_edges = copy(self.in_edges)
      all_out_edges = copy(self.out_edges)
      for op in self.post_ops:
        all_in_edges += op.in_edges
        all_out_edges += op.out_edges
      internal_edges = set(all_out_edges)

      for op in self.post_ops:
        op_external_edges = [e for e in op.in_edges if e not in internal_edges and e != load_edge]

        for edge in op_external_edges:
          if edge in DevConfig().TensorEdgetoInfo:
            te_info = DevConfig().TensorEdgetoInfo[edge]
          else:
            continue

          if te_info and te_info.fifo_id == TensorEdgeInfo.LOCAL_FIFO:
            continue

          try:
            arg_id = edge.src_id.graph_node_id
            if ConstPat.match(CustomIDToNode()[arg_id]):
              continue
          except KeyError:
            pass

          if te_info.fifo_id != 0:
            for i in range(op.num_blocks):
              var_i = UniqueVar((edge, i))
              if var_i.static:
                continue
              annotation = f"{edge}, {te_info.node_info_str}"
              owner_edge = te_info.owner
              recv_code = IMCERecvBlock(str(var_i), te_info.fifo_id, owner_edge, annotation)
              comp.add(recv_code)

              if self.builder and hasattr(self.builder, 'pair_manager') and self.builder.pair_manager:
                pair = self.builder.pair_manager.get_pair(edge)
                if pair:
                  sync_annotation = f"sync after IMCE recv: uuid={pair.uuid}, edge={edge}"
                  sync_code = SequentialBlock()
                  sync_code.add(f"// {sync_annotation}")
                  sync_code.add(f"__builtin_IMCE_SETFLAG({pair.uuid});")
                  sync_code.add(f"__builtin_IMCE_STANDBY({pair.sender_node.value}, {pair.uuid});")
                  sync_code.add(f"__builtin_IMCE_SETFLAG(0);")
                  comp.add(sync_code)

        comp.add(op)
        print(f"[ConvBlock] post_op {type(op).__name__}: external_edges={op_external_edges}")

    if self.post_ops:
      send_edges = list(set(all_out_edges) - set(all_in_edges))
      last_out_edges = self.post_ops[-1].out_edges
      assert (set(send_edges) == set(last_out_edges)), "currently doesn't support middle op SEND"
      send_block = self.post_ops[-1]
      num_out_blocks = send_block.num_out_blocks

      print(f"[ConvBlock] with post ops : send_edges: {send_edges}, send_block: {type(send_block).__name__}")
      return RecvSendWrapper(comp, self.num_blocks, num_out_blocks, send_block, [], send_edges, builder=self.builder)
    else:
      recv_edges = self.in_edges
      send_edges = self.out_edges
      send_block = self
      num_blocks = self.num_blocks
      num_out_blocks = self.num_out_blocks
      return RecvSendWrapper(comp, num_blocks, num_out_blocks, send_block, recv_edges, send_edges, builder=self.builder)

  def _build_structure_a8af(self) -> CodeBlock:
    row_pattern = self.conv.extract_2d_pattern()
    pprint(f"[ConvBlock] row pattern for node {getNodeID(self.call.call)}:")
    pprint(row_pattern)
    # Max-throughput lever (IMCFLOW_FEED_PREFETCH): determine the prefetch group
    # (P pixels, width=P*repeat fifos) for the conv activation load edge. When
    # active AND the col_group pixel count is a multiple of P, UNROLL the col_group
    # loop by P: the P pixels in each unrolled group use distinct fifo slots
    # (pixel p -> base p*repeat), so 2 pixels' bitplanes occupy 8 RECV fifos and
    # the next pixel is resident during the current compute. None -> unchanged.
    pf = self._prefetch_group_for_load_edge()

    root = SequentialBlock()
    for idx, row_pat in enumerate(row_pattern):
      outer_body = SequentialBlock()
      tag = self.annotation + f"_row_group{idx}"
      for inner_idx, pat in enumerate(row_pat["pattern"]):
         cnt = pat["count"]
         if pf is not None and cnt % pf[0] == 0 and cnt >= pf[0]:
           p_group, width = pf
           group_body = SequentialBlock()
           for p in range(p_group):
             group_body.add(self._build_loop_body_a8af(pat["pattern"],
                                                       prefetch_fifo_base=p * 4))
           inner_loop = SimpleFor(cnt // p_group, group_body,
                                  f"{tag}_col_group{inner_idx}_prefetch{p_group}")
         else:
           inner_loop = SimpleFor(cnt,
                                  self._build_loop_body_a8af(pat["pattern"]),
                                  f"{tag}_col_group{inner_idx}")
         outer_body.add(inner_loop)
      outer_loop = SimpleFor(row_pat["count"], outer_body, f"{tag}_outer_loop(iterate row offset)")
      root.add(outer_loop)

    if self.remain > 0:
      print(f"[ConvBlock] node {getNodeID(self.call.call)} has remaining pixels to read: {self.remain}")
      tail_body = TextBlock(f"__builtin_IMCE_RECV({self.load_edge_info.fifo_id});")
      tail_loop = SimpleFor(self.remain*4, tail_body, f"{self.annotation}_tail_loop")
      add_to_map(self.load_edge_info.owner, RecvSendNum("recv", self.remain*4), is_send=False)
      root.add(tail_loop)

    return root

  def _build_loop_body(self, recv_count: int, skip_presend: bool = False,
                       reorder_final_load: bool = False,
                       pad_drain: bool = False) -> CodeBlock:
    load_info = []
    for edge in self.in_edges:
      te_infos = DevConfig().get_tensor_edge_info_with_id_dir(edge.dst_id, "in")
      assert len(te_infos) == 1, "more than one te_info found!"
      te_info = te_infos[0]
      try:
        arg_id = edge.src_id.graph_node_id
        if ConstPat.match(CustomIDToNode()[arg_id]):
          continue
      except KeyError:
        # If the node is not found in CustomIDToNode, treat it as non-constant
        pass
      load_info.append({"edge": edge, "te_info": te_info})

    assert len(load_info) == 1, "there should be exactly one load edge"
    load_edge = load_info[0]["edge"]
    load_edge_info = load_info[0]["te_info"]
    self.load_edge_info = load_edge_info
    self.load_edge = load_edge  # saved for the tail_loop window sync

    # Marker A reorder applies to (a) the fused NoC-rhs post_op path (imce_3_2)
    # and (b) the plain conv that sends its odata as a fused add's NoC rhs
    # (imce_2_2). Both peel the boundary col_group's final iteration.
    is_boundary_colgroup = reorder_final_load  # preserved before the rebind below
    reorder_fused = reorder_final_load and self.has_noc_rhs and bool(self.post_ops) and not skip_presend
    reorder_plain_rhs = (reorder_final_load and not self.post_ops
                         and self.noc_rhs_sender_receiver is not None and not skip_presend)
    reorder_final_load = reorder_fused

    comp = SequentialBlock()
    # A pure-padding row-group feeding a fused consumer (skip_presend) produces
    # NO real output: handcraft emits only dummy SEND(0)xN for this flush row,
    # with no LOAD_LB / STEP / GET_CREG / pre-send STANDBY. Computing a real STEP
    # here (as the default path does) makes imce_1_2 run one extra STEP-burst +
    # STANDBY(6,1) that imce_1_1 does not consume -> back-pressure -> imce_1_2 rx
    # X-fatal. So for skip_presend rows, skip the compute and let the SEND path
    # emit dummies (RecvSendWrapper detects skip_presend). imce_2_1's flush row
    # feeds a PLAIN consumer (imce_3_1) so skip_presend is False there -> keeps
    # the real STEP, matching handcraft's asymmetry.
    def _build_creg():
      creg_code = TextBlock("")
      for i in range(self.num_out_blocks):
        var_o = UniqueVar((self, i))
        creg_code += f"{var_o} = __builtin_IMCE_GET_CREG((short){i});"
      return creg_code

    if reorder_plain_rhs:
      # Marker A (imce_2_2): boundary col_group with the pre-send STANDBY relocated
      # INTO the LOAD window, between LOAD_LB x(n-1) and the final LOAD_LB. The data
      # LOAD window (SETFLAG) still wraps all n loads; the SEND then carries no
      # pre-send STANDBY (skip_presend=True on the wrapper).
      # handcraft: SETFLAG(1); LOAD x(n-1); STANDBY(recv,1); LOAD x1; SETFLAG(0);
      #            STEP; CREG; SEND (bare).
      recv_hw = self.noc_rhs_sender_receiver
      pre_lines, post_lines = (None, None)
      if self.builder and hasattr(self.builder, 'pair_manager') and self.builder.pair_manager:
        pre_lines, post_lines = self.builder.pair_manager.get_recv_window_sync(load_edge)
      if pre_lines:
        comp.add("\n".join(pre_lines))
      comp.add(LoadLBBlock(recv_count, self.num_blocks - 1, load_edge, load_edge_info,
                           builder=self.builder, bare=True))
      comp.add(TextBlock(f"//! we wait IMCE node that requires receiving from this node to be ready\n"
                         f"__builtin_IMCE_STANDBY({recv_hw.value}, 1);\n"))
      comp.add(LoadLBBlock(recv_count, 1, load_edge, load_edge_info, builder=self.builder, bare=True))
      if post_lines:
        comp.add("\n".join(post_lines))
      comp.add(TextBlock("__builtin_IMCE_STEP();\n"))
      comp.add(_build_creg())
      # SEND real data, but pre-send STANDBY suppressed (relocated into the window).
      return RecvSendWrapper(comp, self.num_blocks, self.num_out_blocks, self,
                             [], self.out_edges, builder=self.builder,
                             suppress_presend_only=True)

    if reorder_final_load:
      # Marker A boundary col_group: split the num_blocks bitplane LOAD_LBs into
      # (n-1) BEFORE the fused-rhs RECV window and 1 AFTER it, immediately before
      # STEP -- so STEP is issued only after the stall-capable NoC RECV has
      # drained, keeping hs_remaining stable (RTL BUGFIX-off deadlock avoidance).
      # handcraft: LOAD_LB x(n-1) [bare] ; RECV window ; LOAD_LB x1 ; STEP ; CREG.
      comp.add(LoadLBBlock(recv_count, self.num_blocks - 1, load_edge, load_edge_info,
                           builder=self.builder, bare=True))
      # post_op fused-rhs RECV window is emitted here (see _emit_postop_recv below)
      self._emit_postop_recv(comp, load_edge)
      comp.add(TextBlock("//! this is last iteration before padding; final load after recv\n"))
      # Fix G (region3 imce_3_2): boundary STEP rendezvous. When this node's data
      # LOAD_LB producer MULTICASTS to a sibling S, and S also feeds this node's
      # post-op lhs over NoC (cyclic coupling), the boundary STEP must wait for S
      # to have finished sending its psum for this iteration -- otherwise STEP
      # fires while the sibling's SEND handshake is mid-flight, mis-consuming
      # hs_remaining -> stale STEP latch -> deadlock (fsim-confirmed: imce_3_2
      # STEP stall). handcraft: STANDBY(imce_3_3, uuid) before the final LOAD_LB,
      # paired with SETFLAG(uuid) after imce_3_3's SEND. region2 imce_3_2 (both
      # edges single-target, self-fed add) and region1 (no sibling multicast) and
      # imce_1_2 (no reorder_final_load) all fail the predicate -> no barrier.
      bstep = self._get_boundary_step_barrier(load_edge)
      if bstep:
        comp.add(TextBlock(bstep))
      comp.add(LoadLBBlock(recv_count, 1, load_edge, load_edge_info, builder=self.builder, bare=True))
      comp.add(TextBlock("__builtin_IMCE_STEP();\n"))
      comp.add(_build_creg())
      # compute the post_op(s) now that both operands are present
      for op in self.post_ops:
        comp.add(op)
      # fall through to the SEND wrapper below (post_op RECV already emitted)
      last_out_edges = self.post_ops[-1].out_edges
      all_in_edges = copy(self.in_edges)
      all_out_edges = copy(self.out_edges)
      for op in self.post_ops:
        all_in_edges += op.in_edges
        all_out_edges += op.out_edges
      send_edges = list(set(all_out_edges) - set(all_in_edges))
      assert (set(send_edges) == set(last_out_edges)), "currently doesn't support middle op SEND"
      send_block = self.post_ops[-1]
      return RecvSendWrapper(comp, self.num_blocks, send_block.num_out_blocks, send_block,
                             [], send_edges, builder=self.builder, skip_presend=skip_presend,
                             post_send_setflag_uuid=self._get_boundary_step_notify_uuid(load_edge))

    # Padding-row NoC-operand DRAIN (region3 imce_3_3 AND imce_3_2 bottom zero-pad
    # row_group). A pure-padding row of a FUSED conv+add whose add operand arrives
    # over the NoC (has_noc_rhs) MUST still RECV+drain that operand even on a zero
    # output row and emit NO STEP/GET_CREG (there is no valid conv window; a STEP
    # over an empty bottom_pad line buffer never retires -> hard hang). The
    # producer keeps sending its full tail, so dropping the RECV overruns the
    # input FIFO. Two entry conditions:
    #   * skip_presend=True  -> node feeds a FUSED consumer (imce_3_3 -> imce_3_2)
    #   * pad_drain=True     -> node feeds a PLAIN consumer (imce_3_2 -> imce_3_1),
    #     which the skip_row_presend gate (feeds_fused_consumer) misses.
    # handcraft zero-pad tail (both): SETFLAG(1); RECV x4; SETFLAG(0);
    #   ADD(.,0) x4; [STANDBY(recv,1)]; SEND(result) x4  -- conv psum zeroed, NoC
    # operand still consumed and forwarded. ADD is commutative so ADD(0,x) is used
    # uniformly (functionally identical to handcraft's ADD(x,0) on imce_3_2).
    if (skip_presend or pad_drain) and self.has_noc_rhs and self.post_ops:
      # drain the fused post_op's NoC-rhs (emits its SETFLAG(1);RECV;SETFLAG(0)).
      self._emit_postop_recv(comp, load_edge)
      send_block = self.post_ops[-1]
      # rhs edge = the post_op external in-edge that is NOT the conv load edge and
      # not an internal/const edge (the imce_2_3 -> imce_3_3 rhs).
      all_out_edges = copy(self.out_edges)
      for op in self.post_ops:
        all_out_edges += op.out_edges
      internal_edges = set(all_out_edges)
      rhs_edge = None
      for e in send_block.in_edges:
        if e in internal_edges or e == load_edge:
          continue
        te = DevConfig().TensorEdgetoInfo.get(e, None)
        if te is None or te.fifo_id == TensorEdgeInfo.LOCAL_FIFO:
          continue
        try:
          if ConstPat.match(CustomIDToNode()[e.src_id.graph_node_id]):
            continue
        except (KeyError, Exception):
          pass
        rhs_edge = e
        break
      # ADD(0, rhs) per output block, writing the send_block's output vars so the
      # RecvSendWrapper below sends the real results (matching handcraft ADD(0,x)).
      add_code = TextBlock("")
      for i in range(send_block.num_out_blocks):
        var_o = UniqueVar((send_block, i))
        var_rhs = UniqueVar((rhs_edge, i)) if rhs_edge is not None else "0"
        add_code += f"{var_o} = __builtin_IMCE_ADD(0, {var_rhs}, 15);"
      comp.add(add_code)
      last_out_edges = send_block.out_edges
      send_edges = list(last_out_edges)
      # Both cases send REAL ADD results (not dummy). Pre-send STANDBY differs by
      # consumer type, matching handcraft's padding rows exactly:
      #   * fused consumer (skip_presend, imce_3_3 -> imce_3_2): KEEP the pre-send
      #     STANDBY(receiver,1) -> skip_presend=False on the wrapper.
      #   * plain consumer (pad_drain-only, imce_3_2 -> imce_3_1): DROP the pre-send
      #     STANDBY (handcraft padding SEND has none) -> suppress_presend_only=True
      #     (real SEND, no STANDBY).
      if skip_presend:
        return RecvSendWrapper(comp, self.num_blocks, send_block.num_out_blocks, send_block,
                               [], send_edges, builder=self.builder, skip_presend=False)
      return RecvSendWrapper(comp, self.num_blocks, send_block.num_out_blocks, send_block,
                             [], send_edges, builder=self.builder,
                             suppress_presend_only=True)

    if not skip_presend:
      comp.add(LoadLBBlock(recv_count, self.num_blocks, load_edge, load_edge_info,
                           builder=self.builder, bare=self.has_noc_rhs))
      comp.add(TextBlock("__builtin_IMCE_STEP();\n"))
      comp.add(_build_creg())

    if self.post_ops and not skip_presend:
      # (skip_presend flush row: no post_op RECV/compute either -- only dummy SEND)
      # Calculate internal edges (outputs from one op that are inputs to another)
      all_in_edges = copy(self.in_edges)
      all_out_edges = copy(self.out_edges)
      for op in self.post_ops:
        all_in_edges += op.in_edges
        all_out_edges += op.out_edges
      internal_edges = set(all_out_edges)  # edges produced by conv or post_ops

      # For each post_op, generate its external RECVs right before the op
      for op in self.post_ops:
        self._emit_postop_recv(comp, load_edge, only_op=op)
        # Add the post_op after its RECVs
        comp.add(op)

    if self.post_ops:
      last_out_edges = self.post_ops[-1].out_edges
      if not skip_presend:
        send_edges = list(set(all_out_edges) - set(all_in_edges))
        assert (set(send_edges) == set(last_out_edges)), "currently doesn't support middle op SEND"
      else:
        # flush row: comp is empty (no compute); SEND target from post_op directly.
        send_edges = list(last_out_edges)
      send_block = self.post_ops[-1]
      num_out_blocks = send_block.num_out_blocks

      print(f"[ConvBlock] with post ops : send_edges: {send_edges}, send_block: {type(send_block).__name__}")
      # Pass empty recv_edges to RecvSendWrapper since we already handled post_op_recv_edges above
      # skip pre-send STANDBY for pure-padding col_groups (whole zero row_group).
      # Fix G producer: imce_3_3 (fused conv+add sending its psum to a sibling that
      # co-receives the same data multicast) reaches this non-reorder post_ops path
      # for its SENDs. Emit SETFLAG(uuid) after each SEND so the sibling consumer's
      # boundary STANDBY(this_node, uuid) can clear. Predicate returns None for all
      # non-coupled nodes -> no notify -> region1/2 unaffected.
      # ONLY on the boundary col_group (handcraft emits SETFLAG(5) exactly twice,
      # at the boundary tiles, NOT every iteration). Emitting it every iteration
      # (the earlier bug) floods the sibling's flag credit -> the barrier becomes
      # trivially satisfied AND adds a per-iteration rendezvous that makes region3
      # ~14x slower than handcraft (1388 vs >20000 polls) -> host poll timeout
      # (throughput collapse, NOT a real deadlock).
      _notify_uuid = (self._get_boundary_step_notify_uuid(load_edge)
                      if (is_boundary_colgroup and not skip_presend) else None)
      return RecvSendWrapper(comp, self.num_blocks, num_out_blocks, send_block, [], send_edges,
                             builder=self.builder, skip_presend=skip_presend,
                             post_send_setflag_uuid=_notify_uuid)
    else:
      recv_edges = self.in_edges
      send_edges = self.out_edges
      send_block = self
      num_blocks = self.num_blocks
      num_out_blocks = self.num_out_blocks
      return RecvSendWrapper(comp, num_blocks, num_out_blocks, send_block, recv_edges, send_edges,
                             builder=self.builder, skip_presend=skip_presend)

  def _get_boundary_step_barrier(self, load_edge):
    """Fix G: return the one-way STANDBY line that gates the boundary STEP on the
    sibling psum producer, or "" when the cyclic-coupling predicate is false.

    Predicate (IR-decidable, verified byte-exact vs handcraft): the data LOAD_LB
    producer of this node MULTICASTS (>=2 imce receivers) to a sibling S, AND S is
    the NoC source of one of this node's post-op external (lhs) edges. Only then
    does the boundary STEP need to wait for S's psum SEND (handcraft STANDBY(S,uuid)).
    The uuid is the sibling-lhs edge's own SendRecvPair uuid (same object queried on
    both sides -> matching value), NOT the hand-chosen literal 5.
    """
    if not (self.builder and hasattr(self.builder, 'pair_manager') and self.builder.pair_manager):
      return ""
    pm = self.builder.pair_manager
    # 1. data producer must multicast to >=2 imce (this node + a sibling)
    data_pair = pm.get_pair(load_edge)
    if data_pair is None:
      return ""
    imce_recvs = [r for r in data_pair.receiver_nodes if r.is_imce()]
    if len(imce_recvs) < 2:
      return ""
    try:
      my_hw = DevConfig().get_hw_node(self.get_graph_node_id())
    except Exception:
      return ""
    sibling_nodes = set()
    for r in imce_recvs:
      if r != my_hw:
        sibling_nodes.add(r)
    if not sibling_nodes:
      return ""
    # 2. one of this node's post-op external (lhs) NoC edges must be sourced from
    #    a sibling S that co-receives the data multicast.
    all_out_edges = copy(self.out_edges)
    for op in self.post_ops:
      all_out_edges += op.out_edges
    internal_edges = set(all_out_edges)
    for op in self.post_ops:
      for edge in op.in_edges:
        if edge in internal_edges or edge == load_edge:
          continue
        # ONLY the lhs operand carries the sibling-psum rendezvous. In region2
        # imce_3_2 the sibling (imce_2_2) feeds the RHS instead, and the data
        # LOAD_LB comes from a DIFFERENT producer (imce_2_1) that only happens to
        # multicast -> the predicate must reject rhs, else region2 gets a spurious
        # STANDBY(12,7) and deadlocks. handcraft's barrier is lhs-only.
        if getattr(edge.dst_id, "tensor_type", None) != "lhs":
          continue
        if edge not in DevConfig().TensorEdgetoInfo:
          continue
        te_info = DevConfig().TensorEdgetoInfo[edge]
        if te_info is None or te_info.fifo_id in (0, TensorEdgeInfo.LOCAL_FIFO):
          continue
        try:
          src_hw = DevConfig().get_hw_node(edge.src_id.graph_node_id)
          if isinstance(src_hw, tuple):
            src_hw = src_hw[0]
        except Exception:
          continue
        if src_hw in sibling_nodes:
          lhs_pair = pm.get_pair(edge)
          if lhs_pair is None:
            continue
          # one-way rendezvous: wait for the sibling's SETFLAG(uuid) posted after
          # its SEND of this lhs edge (handcraft STANDBY(18,5)).
          return (f"//! wait sibling that sends psum into this node before STEP\n"
                  f"__builtin_IMCE_STANDBY({lhs_pair.sender_node.value}, {lhs_pair.uuid});\n")
    return ""

  def _get_boundary_step_notify_uuid(self, load_edge):
    """Fix G producer side: return the uuid this node must SETFLAG after its SEND
    to release a sibling's boundary STEP, or None when the predicate is false.

    Mirror of _get_boundary_step_barrier from the producer's view: this node
    co-receives the data multicast (load_edge, >=2 imce receivers) AND sends one of
    its outputs to a SIBLING co-receiver of that same multicast. The sibling is the
    boundary-STEP consumer waiting on STANDBY(this_node, uuid); uuid is that out
    edge's own SendRecvPair uuid (same object both sides -> matching value).
    """
    if not (self.builder and hasattr(self.builder, 'pair_manager') and self.builder.pair_manager):
      return None
    pm = self.builder.pair_manager
    data_pair = pm.get_pair(load_edge)
    if data_pair is None:
      return None
    imce_recvs = [r for r in data_pair.receiver_nodes if r.is_imce()]
    if len(imce_recvs) < 2:
      return None
    # this node is the LOAD_LB receiver of load_edge -> derive my_hw from the edge
    # dst (get_hw_node on the conv graph id can be None; the edge dst is reliable).
    try:
      my_hw = DevConfig().get_hw_node(load_edge.dst_id.graph_node_id)
      if isinstance(my_hw, tuple):
        my_hw = my_hw[0]
    except Exception:
      my_hw = None
    if my_hw is None or my_hw not in imce_recvs:
      # fall back: any receiver we also send an output to counts as sibling below
      my_hw = None
    siblings = set(r for r in imce_recvs if r != my_hw)
    # the real SEND edge to the sibling is the post_op's out_edge (fused add
    # odata -> sibling lhs), not the conv's own out_edge -- include both.
    candidate_out = list(self.out_edges)
    for op in self.post_ops:
      candidate_out += op.out_edges
    for edge in candidate_out:
      # only the sibling's LHS-fed rendezvous (region3). region2's imce_2_2 sends
      # to a sibling's RHS slot -> reject so region2 gets no spurious SETFLAG.
      if getattr(edge.dst_id, "tensor_type", None) != "lhs":
        continue
      if edge not in DevConfig().TensorEdgetoInfo:
        continue
      te_info = DevConfig().TensorEdgetoInfo[edge]
      if te_info is None or te_info.fifo_id in (0, TensorEdgeInfo.LOCAL_FIFO):
        continue
      try:
        dst_hw = DevConfig().get_hw_node(edge.dst_id.graph_node_id)
        if isinstance(dst_hw, tuple):
          dst_hw = dst_hw[0]
      except Exception:
        continue
      if dst_hw in siblings:
        out_pair = pm.get_pair(edge)
        if out_pair is not None:
          return out_pair.uuid
    return None

  def _emit_postop_recv(self, comp: SequentialBlock, load_edge, only_op=None):
    """Emit the external (fused-rhs) RECV window+blocks for post_op(s).

    Shared by the normal path (RECV emitted after STEP/CREG, before compute) and
    the Marker A reorder path (RECV emitted before the final LOAD_LB+STEP). Emits
    only RECVs, not the post_op compute. `only_op` restricts to a single post_op.
    """
    all_out_edges = copy(self.out_edges)
    for op in self.post_ops:
      all_out_edges += op.out_edges
    internal_edges = set(all_out_edges)  # edges produced by conv or post_ops

    ops = [only_op] if only_op is not None else list(self.post_ops)
    for op in ops:
      op_external_edges = [e for e in op.in_edges if e not in internal_edges and e != load_edge]
      for edge in op_external_edges:
        if edge in DevConfig().TensorEdgetoInfo:
          te_info = DevConfig().TensorEdgetoInfo[edge]
        else:
          continue
        if te_info and te_info.fifo_id == TensorEdgeInfo.LOCAL_FIFO:
          continue  # local edge, no need to recv
        try:
          arg_id = edge.src_id.graph_node_id
          if ConstPat.match(CustomIDToNode()[arg_id]):
            continue  # constant edge
        except KeyError:
          pass
        if te_info.fifo_id != 0:
          # handcraft window sync: wrap ALL blocks' RECVs for this edge in one
          # SETFLAG(1)...SETFLAG(0) window (no per-RECV STANDBY) for imce<->imce
          # pipeline; STANDBY(inode,1) window for inode data input.
          recv_blocks = []
          for i in range(op.num_blocks):
            var_i = UniqueVar((edge, i))
            if var_i.static:
              continue
            annotation = f"{edge}, {te_info.node_info_str}"
            owner_edge = te_info.owner
            recv_blocks.append(IMCERecvBlock(str(var_i), te_info.fifo_id, owner_edge, annotation))
          if not recv_blocks:
            continue
          pre_lines, post_lines = (None, None)
          if self.builder and hasattr(self.builder, 'pair_manager') and self.builder.pair_manager:
            pre_lines, post_lines = self.builder.pair_manager.get_recv_window_sync(edge)
          if pre_lines:
            comp.add("\n".join(pre_lines))
          for rb in recv_blocks:
            comp.add(rb)
          if post_lines:
            comp.add("\n".join(post_lines))

  def _build_structure(self) -> CodeBlock:
    """
    row pattern example:
    [
      {'count': 1, 'pattern': [
        {'count': 1, 'pattern': 6}, {'count': 2, 'pattern': 1}, {'count': 1, 'pattern': 0}]
      }, 
      {'count': 2, 'pattern': [
        {'count': 1, 'pattern': 2}, {'count': 2, 'pattern': 1}, {'count': 1, 'pattern': 0}
      ]}, 
      {'count': 1, 'pattern': [
        {'count': 4, 'pattern': 0}
      ]}
    ]
    """
    row_pattern = self.conv.extract_2d_pattern()
    pprint(f"[ConvBlock] row pattern for node {getNodeID(self.call.call)}:")
    pprint(row_pattern)
    root = SequentialBlock()
    for idx, row_pat in enumerate(row_pattern):
      # row_pat["count"] : number of rows that share the same pattern
      # row_pat["pattern"] : pattern for a row. list of {count, pattern}. pattern is the read count for a output pixel
      
      outer_body = SequentialBlock()
      tag = self.annotation + f"_row_group{idx}"

      # A row_group whose every col_group has load pattern 0 is pure zero
      # padding (trailing rows). handcraft emits its SEND without the pre-send
      # STANDBY rendezvous ONLY when this conv feeds a fused/composite consumer
      # (out edge dst graph_node_id is a tuple, e.g. imce_1_2 -> imce_1_1's fused
      # conv sends literal 0). A conv feeding a plain consumer (imce_2_1 ->
      # imce_3_1) still emits the rendezvous even for the padding row.
      row_is_padding = all(pat["pattern"] == 0 for pat in row_pat["pattern"])
      # Look at the EXTERNAL inter-node SEND edge (the post-op's output when this
      # node is a fused conv+op), NOT the internal conv->add edge. A fused
      # conv+add always has an internal conv->add edge with a tuple dst, which
      # would make EVERY fused node look like it "feeds a fused consumer" and
      # wrongly dummy-out its flush row -- dropping a NoC RECV the producer still
      # feeds (region2 imce_3_2: add-rhs arrives from imce_2_2 -> undrained FIFO
      # -> deadlock). The real discriminator is whether the node's SEND leaves to
      # a fused consumer (dst is a tuple) vs a plain consumer (int dst).
      external_out_edges = (self.post_ops[-1].out_edges
                            if self.post_ops else self.out_edges)
      feeds_fused_consumer = any(
          isinstance(e.dst_id.graph_node_id, tuple) for e in external_out_edges)
      skip_row_presend = row_is_padding and feeds_fused_consumer

      # A pure-padding row of a FUSED conv+add whose operand arrives over the NoC
      # must DRAIN-and-forward without computing (no STEP/GET_CREG): the producer
      # keeps sending its tail, so the RECV must still fire or its input FIFO
      # overruns -> hard hang (waveform: imce_3_2 STEP never retires in bottom_pad
      # row -> imce_3_3 STANDBY(17,1) -> imce_2_3 SEND blocked -> imce_3_1 RECV).
      # This holds REGARDLESS of whether the SEND target is fused (imce_3_3 ->
      # imce_3_2) or plain (imce_3_2 -> imce_3_1); the earlier skip_row_presend
      # gate wrongly required feeds_fused_consumer, so imce_3_2 (plain consumer)
      # fell through to the compute path and emitted a spurious STEP over an empty
      # (bottom_pad, S3_valid=0) line buffer. handcraft emits RECV+ADD(.,0)+SEND,
      # no STEP. Decoupled from feeds_fused_consumer here.
      pad_drain_row = row_is_padding and self.has_noc_rhs and bool(self.post_ops)

      # Marker A: boundary col_group = the LAST non-zero-load col_group in this
      # row_group (the one immediately before pure zero-padding). Its FINAL
      # iteration is the "padding boundary" where the fused NoC-rhs RECV (has_noc_rhs,
      # imce_3_2) or the relocated pre-send STANDBY (noc_rhs_sender, imce_2_2) must
      # sit before the last LOAD_LB+STEP.
      # The peel only makes sense at a REAL padding boundary: the last non-zero
      # col_group must be FOLLOWED by a pure-zero-padding col_group (pattern==0)
      # in this row. 3x3/pad-1 convs (imce_3_2, region2 imce_2_2/3_2) have that
      # trailing zero-pad group -> peel (STEP hs_remaining guard / Fix G barrier).
      # 1x1/pad-0 convs (region3 imce_1_2/1_3) have NO zero-pad col_group -> their
      # last group is ordinary steady-state data; handcraft emits it as one
      # monolithic loop with NO peel. Peeling them adds a spurious second RECV
      # window + LOAD split on every outer row -> ~2x rendezvous on the hot path
      # -> region3 throughput collapse (14x slower, host poll timeout). So gate
      # boundary_idx on trailing-zero-pad existence (byte-exact vs handcraft).
      boundary_idx = None
      if self.has_noc_rhs or self.noc_rhs_sender_receiver is not None:
        last_nonzero = None
        has_trailing_pad = False
        for j, pat in enumerate(row_pat["pattern"]):
          if pat["pattern"] > 0:
            last_nonzero = j
          elif last_nonzero is not None:
            has_trailing_pad = True  # a zero-load col_group AFTER the last real one
        if has_trailing_pad:
          boundary_idx = last_nonzero

      for inner_idx, pat in enumerate(row_pat["pattern"]):
         if inner_idx == boundary_idx and pat["count"] >= 1:
           # peel the final iteration and reorder its last LOAD_LB after the RECV
           n_normal = pat["count"] - 1
           if n_normal > 0:
             outer_body.add(SimpleFor(
                 n_normal,
                 self._build_loop_body(pat["pattern"], skip_presend=skip_row_presend),
                 f"{tag}_col_group{inner_idx}"))
           outer_body.add(SimpleFor(
               1,
               self._build_loop_body(pat["pattern"], skip_presend=skip_row_presend,
                                     reorder_final_load=True),
               f"{tag}_col_group{inner_idx}_boundary"))
           continue
         inner_loop = SimpleFor(pat["count"],
                                self._build_loop_body(pat["pattern"], skip_presend=skip_row_presend,
                                                      pad_drain=pad_drain_row),
                                f"{tag}_col_group{inner_idx}")
         outer_body.add(inner_loop)

      outer_loop = SimpleFor(row_pat["count"], outer_body, f"{tag}_outer_loop(iterate row offset)")
      root.add(outer_loop)
    
    # read remaining pixels if any
    if self.remain > 0:
      print(f"[ConvBlock] node {getNodeID(self.call.call)} has remaining pixels to read: {self.remain}")
      fid = self.load_edge_info.fifo_id
      # If the load edge is cyclic (paired), the producer (e.g. imce_0_3) gates
      # every one of its iterations on a matching SETFLAG(1) rendezvous. The main
      # LOAD_LB loop wraps each 4-value block in a SETFLAG(1)...SETFLAG(0) window
      # (LoadLBBlock), but the tail RECVs default to bare RECV with NO flag. Once
      # the consumer enters this flag-less tail, the producer's per-iteration
      # STANDBY blocks forever -> region2 TILE0 deadlock (fsim-confirmed root
      # cause: imce_0_2 tail starves imce_0_3.STANDBY(2,1)). handcraft wraps the
      # tail in the same window: remain x [SETFLAG(1); RECV x4; SETFLAG(0)].
      pre_lines, post_lines = (None, None)
      if (getattr(self, "load_edge", None) is not None and self.builder
          and hasattr(self.builder, 'pair_manager') and self.builder.pair_manager):
        pre_lines, post_lines = self.builder.pair_manager.get_recv_window_sync(self.load_edge)
      if pre_lines:
        # windowed tail: one SETFLAG(1)...SETFLAG(0) per 4-value block
        def tail_window_body(iter, fid=fid, pre=pre_lines, post=post_lines):
          body = "\n".join(pre) + "\n"
          body += SimpleFor(4, TextBlock(f"__builtin_IMCE_RECV({fid});")).render() + "\n"
          body += "\n".join(post) + "\n"
          return body
        tail_loop = SimpleFor(self.remain, tail_window_body, f"{self.annotation}_tail_loop")
      else:
        tail_body = TextBlock(f"__builtin_IMCE_RECV({fid});")
        tail_loop = SimpleFor(self.remain*4, tail_body, f"{self.annotation}_tail_loop")
      add_to_map(self.load_edge_info.owner, RecvSendNum("recv", self.remain*4), is_send=False)
      root.add(tail_loop)

    return root

  def _render(self) -> str:
    return self.body.render()


class DWConvBlock(ImceCallCodeBlock):
  """Code block for depthwise convolution using OP_DWCONV instruction.

  Depthwise convolution operates on:
  - Input: BSHR (bit shift register), loaded via linebuffer
  - Weight: GPR registers (received via FIFO, not IMCU like standard conv)

  Key parameters from ISA (A-type instruction):
  - bshr_sel: Input channel group index (16 channels per group)
  - src_mask: Mask for which input channels in group are used
  - shift_amt: Weight bitplane weight (2^shift_amt), 8 bitplanes total
  - dwresult_valid: Set to 1 when ending accumulation for all weight bitplanes

  Weight layout: ic16_pad16_kh3_kw3 - 8 bitplanes per channel group
  Total weight registers: 8 * num_channel_groups

  Only 3x3 kernel is supported in hardware.
  """
  num_in_edges = 3  # data, weight, (bias or config)

  def __init__(self, call: 'BuilderContext', shapes: dict, conv_attrs,
               weight_edge: TensorEdge = None,
               post_ops: List[ImceCallCodeBlock] = None, annotation: str = ""):
    super().__init__(call, annotation)
    self.builder = call.builder
    self.shapes = shapes
    self.conv_attrs = conv_attrs
    self.weight_edge = weight_edge
    self.post_ops = post_ops if post_ops is not None else []

    # Extract convolution parameters
    self.kernel_h = conv_attrs.kernel_size[0].value if hasattr(conv_attrs.kernel_size[0], 'value') else conv_attrs.kernel_size[0]
    self.kernel_w = conv_attrs.kernel_size[1].value if hasattr(conv_attrs.kernel_size[1], 'value') else conv_attrs.kernel_size[1]
    self.stride = conv_attrs.strides[0].value if hasattr(conv_attrs.strides[0], 'value') else conv_attrs.strides[0]
    self.padding = conv_attrs.padding[0].value if hasattr(conv_attrs.padding[0], 'value') else conv_attrs.padding[0]

    # Depthwise conv: in_channels == out_channels == groups
    self.channels = conv_attrs.channels.value if hasattr(conv_attrs.channels, 'value') else conv_attrs.channels

    # Validate 3x3 kernel constraint
    if self.kernel_h != 3 or self.kernel_w != 3:
      logging.warning(f"DWConvBlock: Only 3x3 kernel supported, got {self.kernel_h}x{self.kernel_w}")

    # Input spatial dimensions
    self.in_h = shapes["data"][2]
    self.in_w = shapes["data"][3]

    # Calculate output spatial dimensions
    self.out_h = (self.in_h + 2 * self.padding - self.kernel_h) // self.stride + 1
    self.out_w = (self.in_w + 2 * self.padding - self.kernel_w) // self.stride + 1

    # Number of input channel groups (16 channels per group due to BSHR constraint)
    self.num_channel_groups = math.ceil(self.channels / 16)

    # Number of weight bitplanes (8 bitplanes per channel group for int8 weights)
    self.num_weight_bitplanes = 8

    # Total weight registers: 8 bitplanes × num_channel_groups
    self.num_weight_regs = self.num_weight_bitplanes * self.num_channel_groups

    # Weight variables - populated here to reference vars created by RecvConstBlock in INIT phase
    # RecvConstBlock uses UniqueVar((edge, i)) so we use the same to get the same variable names
    self.weight_vars = []
    if self.weight_edge is not None:
      for i in range(self.num_weight_regs):
        var = UniqueVar((self.weight_edge, i))
        self.weight_vars.append(var)

    # Initialize ConvUtil for 2D pattern extraction (like ConvBlock)
    self.conv = ConvUtil(self.in_h, self.in_w,
                         self.padding, self.stride,
                         self.kernel_h, self.kernel_w)
    self.total_in_read_counts = self.conv.get_total_input_read_counts()
    self.origin_hw = self.in_h * self.in_w  # H * W
    self.remain = self.origin_hw - self.total_in_read_counts

    # Will be set by _build_loop_body
    self.load_edge_info = None

    # Link post-ops
    prev = self
    for op in self.post_ops:
      op.prev_op = prev
      prev = op

    self.body = self._build_structure()

  @property
  def num_blocks(self) -> int:
    # Depthwise conv processes 4 bitplanes (for input)
    return 4

  @property
  def num_out_blocks(self) -> int:
    return self.num_channel_groups
    # return 4

  # def _build_weight_recv(self) -> CodeBlock:
  #   """Receive DW conv weights into GPR registers.

  #   DW conv weights are sent via FIFO (unlike standard conv weights which
  #   go to IMCU). Each weight register contains one bitplane of weights
  #   for a channel group.

  #   Returns:
  #       CodeBlock with RECV instructions for all weight registers
  #   """
  #   if self.weight_edge is None:
  #     raise ValueError("DWConvBlock: weight_edge is None, cannot generate weight RECV")

  #   te_infos = DevConfig().get_tensor_edge_info_with_id_dir(
  #       self.weight_edge.dst_id, "in")

  #   if not te_infos:
  #     raise ValueError(f"DWConvBlock: No tensor edge info found for weight edge {self.weight_edge}")

  #   te_info = te_infos[0]

  #   weight_code = TextBlock("")
  #   self.weight_vars = []

  #   # Receive all weight registers
  #   # Layout: for each channel group, receive 8 bitplane weights
  #   for i in range(self.num_weight_regs):
  #     var = UniqueVar((self.weight_edge, i))
  #     var.set_static()  # Mark as static so it persists across loop iterations
  #     weight_code += f"{var} = __builtin_IMCE_RECV({te_info.fifo_id}); // weight bitplane {i}"
  #     self.weight_vars.append(var)

  #   # Track recv count for validation
  #   add_to_map(self.weight_edge, RecvSendNum("recv", self.num_weight_regs), is_send=False)

  #   return weight_code

  def _build_dwconv_compute(self) -> CodeBlock:
    """Build DWCONV computation for one output pixel.

    For each output pixel:
    1. Iterate over channel groups (bshr_sel)
    2. For each channel group, iterate over 8 weight bitplanes (shift_amt)
    3. On the last bitplane (shift_amt=7), set dwresult_valid=1 to capture result

    Returns:
        CodeBlock with DWCONV instructions
    """
    comp = SequentialBlock()

    # Add comment header
    comp.add(TextBlock(f"// OP_DWCONV: {self.channels} channels, {self.num_channel_groups} groups, {self.num_weight_bitplanes} bitplanes\n"))

    # Iterate over actual channel groups
    for bshr_sel in range(self.num_channel_groups):
      # Calculate src_mask for this channel group
      # src_mask indicates which channels in the 16-channel group are valid
      channels_in_group = min(16, self.channels - bshr_sel * 16)
      src_mask = channels_in_group - 1  # 0-indexed mask (15 for full group)

      # BUGFIX knob (bugfix_off_mode()): a DWCONV group with channels_in_group<16
      # (e.g. MobileNetV1's 8-channel depthwise) drives only the valid lanes. The
      # vpu masks the unused blocks (block_mask -> vpu_wmask, BWEN active-high =
      # mask-out) so the destination GPR's UPPER lanes are NEVER written and stay
      # X. When imce_1_1 SENDs that raw partial psum straight to the next imce
      # (split standalone-DWCONV path), the NoC TX carries X on the undriven lanes;
      # BUGFIX-off has no X-masking so the flow_if `word_test` (^data !== 'x)
      # $fatals after 100 such words (imce_intf_tx[4] = imce_1_1 local TX). Widen
      # src_mask to 15 so ALL 16 blocks are block_selected => wmask=0 => every lane
      # is WRITTEN (upper lanes accumulate the zero-padded upper input channels ->
      # defined). The downstream minmaxquant only reads the valid channels, so the
      # extra (zero) upper lanes are numerically inert. ResNet8 has no DWCONV;
      # DS-CNN dwconv groups are full 16-lane (channels_in_group==16) => predicate
      # never fires => byte-identical.
      if bugfix_off_mode() and channels_in_group < 16:
        src_mask = 15

      # Iterate over weight bitplanes (8 bitplanes for int8 weights)
      for shift_amt in range(self.num_weight_bitplanes):
        weight_idx = bshr_sel * self.num_weight_bitplanes + shift_amt

        if weight_idx >= len(self.weight_vars):
          raise ValueError(f"DWConvBlock: weight_idx {weight_idx} out of range for weight_vars length {len(self.weight_vars)}")

        weight_var = self.weight_vars[weight_idx]

        # dwresult_valid = 1 on the last bitplane of each channel group
        dwresult_valid = 1 if shift_amt == self.num_weight_bitplanes - 1 else 0

        if dwresult_valid:
          # Last bitplane: capture result in output variable
          out_var = UniqueVar((self, bshr_sel))
          # Builtin signature: __builtin_IMCE_DWCONV(weight, shift_amt, dwresult_valid, src_mask, bshr_sel)
          comp.add(TextBlock(f"{out_var} = __builtin_IMCE_DWCONV({weight_var}, {shift_amt}, {dwresult_valid}, {src_mask}, {bshr_sel});\n"))
        else:
          # Intermediate bitplane: no result capture, accumulates internally
          comp.add(TextBlock(f"__builtin_IMCE_DWCONV({weight_var}, {shift_amt}, {dwresult_valid}, {src_mask}, {bshr_sel});\n"))
    
    # for bshr_sel in range(self.num_channel_groups, 2):
    #   out_var = UniqueVar((self, bshr_sel))
    #   comp.add(TextBlock(f"{out_var} = __builtin_IMCE_DWCONV({weight_var}, {shift_amt}, {dwresult_valid}, {src_mask}, {bshr_sel}); // this is dummy \n"))

    # Initialize zero outputs for unused channel groups (padding to 64 channels for NCHW64C layout)
    # num_out_blocks is always 4 (64 channels), but actual channels may be less
    # for bshr_sel in range(2, 4):
    #   out_var = UniqueVar((self, bshr_sel))
    #   comp.add(TextBlock(f"{out_var} = (short16)0; // padding for NCHW64C layout\n"))

    return comp

  def _build_loop_body(self, recv_count: int) -> CodeBlock:
    """Build the inner loop body for depthwise convolution.

    For each output pixel:
    1. Load input data to linebuffer (LOAD_LB)
    2. Execute OP_DWCONV for each weight bitplane and channel group
    3. SEND results
    """
    # Get load edge info
    load_info = []
    for edge in self.in_edges:
      te_infos = DevConfig().get_tensor_edge_info_with_id_dir(edge.dst_id, "in")
      if not te_infos:
        continue
      te_info = te_infos[0]
      try:
        arg_id = edge.src_id.graph_node_id
        if ConstPat.match(CustomIDToNode()[arg_id]):
          continue
      except KeyError:
        pass
      load_info.append({"edge": edge, "te_info": te_info})

    assert len(load_info) == 1, "DWConvBlock: should have exactly one data load edge"
    load_edge = load_info[0]["edge"]
    load_edge_info = load_info[0]["te_info"]
    self.load_edge_info = load_edge_info

    comp = SequentialBlock()

    # 1. Load input to linebuffer
    comp.add(LoadLBBlock(recv_count, self.num_blocks, load_edge, load_edge_info, builder=self.builder))

    # 2. Execute DWCONV computation
    comp.add(self._build_dwconv_compute())

    # Handle post-ops if any
    if self.post_ops:
      all_in_edges = copy(self.in_edges)
      all_out_edges = copy(self.out_edges)
      for op in self.post_ops:
        all_in_edges += op.in_edges
        all_out_edges += op.out_edges
      internal_edges = set(all_out_edges)

      for op in self.post_ops:
        op_external_edges = [e for e in op.in_edges if e not in internal_edges and e != load_edge]
        for edge in op_external_edges:
          if edge in DevConfig().TensorEdgetoInfo:
            te_info = DevConfig().TensorEdgetoInfo[edge]
          else:
            continue
          if te_info and te_info.fifo_id == TensorEdgeInfo.LOCAL_FIFO:
            continue
          try:
            arg_id = edge.src_id.graph_node_id
            if ConstPat.match(CustomIDToNode()[arg_id]):
              continue
          except KeyError:
            pass
          if te_info.fifo_id != 0:
            # BUGFIX knob: knob=on (bugfix_off_mode()==False) reproduces a8af's
            # bare RECVs (no window sync) for the dwconv post-op external inputs.
            if not bugfix_off_mode():
              for i in range(op.num_blocks):
                var_i = UniqueVar((edge, i))
                if var_i.static:
                  continue
                annotation = f"{edge}, {te_info.node_info_str}"
                owner_edge = te_info.owner
                comp.add(IMCERecvBlock(str(var_i), te_info.fifo_id, owner_edge, annotation))
              continue
            recv_blocks = []
            for i in range(op.num_blocks):
              var_i = UniqueVar((edge, i))
              if var_i.static:
                continue
              annotation = f"{edge}, {te_info.node_info_str}"
              owner_edge = te_info.owner
              recv_blocks.append(IMCERecvBlock(str(var_i), te_info.fifo_id, owner_edge, annotation))
            if not recv_blocks:
              continue
            pre_lines, post_lines = (None, None)
            if self.builder and hasattr(self.builder, 'pair_manager') and self.builder.pair_manager:
              pre_lines, post_lines = self.builder.pair_manager.get_recv_window_sync(edge)
            if pre_lines:
              comp.add("\n".join(pre_lines))
            for rb in recv_blocks:
              comp.add(rb)
            if post_lines:
              comp.add("\n".join(post_lines))
        comp.add(op)

    # Wrap with RECV/SEND
    if self.post_ops:
      send_edges = list(set(all_out_edges) - set(all_in_edges))
      send_block = self.post_ops[-1]
      num_out_blocks = send_block.num_out_blocks
      return RecvSendWrapper(comp, self.num_blocks, num_out_blocks, send_block, [], send_edges, builder=self.builder)
    else:
      return RecvSendWrapper(comp, self.num_blocks, self.num_out_blocks, self, self.in_edges, self.out_edges, builder=self.builder)

  def _build_structure(self) -> CodeBlock:
    """Build the overall loop structure for depthwise convolution.

    Uses ConvUtil.extract_2d_pattern() like ConvBlock for proper loop optimization.

    Structure:
    1. INIT phase: Receive weights into GPR registers
    2. EXEC phase: Nested loops over output pixels using 2D pattern
    3. TAIL phase: Read remaining pixels if any

    row pattern example (from ConvBlock):
    [
      {'count': 1, 'pattern': [
        {'count': 1, 'pattern': 6}, {'count': 2, 'pattern': 1}, {'count': 1, 'pattern': 0}]
      },
      {'count': 2, 'pattern': [
        {'count': 1, 'pattern': 2}, {'count': 2, 'pattern': 1}, {'count': 1, 'pattern': 0}
      ]},
      {'count': 1, 'pattern': [
        {'count': 4, 'pattern': 0}
      ]}
    ]
    """
    from pprint import pprint

    root = SequentialBlock()

    # Note: Weight reception moved to DwConvHandler (RecvConstBlock in INIT phase)
    # Weight variables are referenced via self.weight_vars (populated in __init__)

    # 1. Main loop using 2D pattern (same as ConvBlock)
    row_pattern = self.conv.extract_2d_pattern()
    pprint(f"[DWConvBlock] row pattern for node {getNodeID(self.call.call)}:")
    pprint(row_pattern)

    for idx, row_pat in enumerate(row_pattern):
      # row_pat["count"]: number of rows that share the same pattern
      # row_pat["pattern"]: pattern for a row. list of {count, pattern}
      # pattern is the read count for an output pixel

      outer_body = SequentialBlock()
      tag = self.annotation + f"_row_group{idx}"

      for inner_idx, pat in enumerate(row_pat["pattern"]):
        inner_loop = SimpleFor(pat["count"],
                               self._build_loop_body(pat["pattern"]),
                               f"{tag}_col_group{inner_idx}")
        outer_body.add(inner_loop)

      outer_loop = SimpleFor(row_pat["count"], outer_body,
                             f"{tag}_outer_loop(iterate row offset)")
      root.add(outer_loop)

    # 2. Read remaining pixels if any (tail loop)
    if self.remain > 0 and self.load_edge_info is not None:
      print(f"[DWConvBlock] node {getNodeID(self.call.call)} has remaining pixels to read: {self.remain}")
      tail_body = TextBlock(f"__builtin_IMCE_RECV({self.load_edge_info.fifo_id});")
      tail_loop = SimpleFor(self.remain * self.num_blocks, tail_body,
                            f"{self.annotation}_tail_loop")
      add_to_map(self.load_edge_info.owner,
                 RecvSendNum("recv", self.remain * self.num_blocks), is_send=False)
      root.add(tail_loop)

    return root

  def _render(self) -> str:
    return self.body.render()


class VecOpBlock(ImceCallCodeBlock):
  """
  VecOpBlock assembles a sequence of vec_ops (add, mult, bn, relu, minmax, etc.)
  into a single code block with proper variable chaining.

  This is the "anchor" block for imcflow.vecops and imcflow.preop-minmax composites,
  analogous to how ConvBlock anchors imcflow.qconv2d-with-postop composites.

  VecOpBlock handles internal edge RECV only:
  - First op's external input RECV: RecvSendWrapper handles this
  - Internal edges between ops: prev_op chaining (no RECV needed)
  - Mid-chain external inputs (e.g., converging patterns): internal RECV
  - Final output SEND: RecvSendWrapper handles this
  """

  def __init__(self, call: 'BuilderContext', ops: List['ImceCallCodeBlock'], annotation: str = ""):
    """Initialize VecOpBlock.

    Args:
        call: BuilderContext for the composite function call
        ops: List of vec_op blocks in execution order
        annotation: Debug annotation
    """
    self.call = call
    self.builder = call.builder
    self.ops = ops
    self.annotation = annotation
    self._original = self
    self.prev_op = None  # For compatibility with post_op chaining

    # Link ops via prev_op chain (same pattern as ConvBlock post_ops)
    prev = None
    for op in self.ops:
      op.prev_op = prev
      prev = op

    # Compute edge sets
    self._compute_edges()
    self.body = self._build_structure()

  def _compute_edges(self):
    """Classify edges as internal vs external."""
    all_in_edges = []
    all_out_edges = []
    for op in self.ops:
      all_in_edges.extend(op.in_edges)
      all_out_edges.extend(op.out_edges)

    self.in_edges = all_in_edges
    self.out_edges = all_out_edges
    self.internal_edges = set(all_out_edges) & set(all_in_edges)
    self.external_in_edges = [e for e in all_in_edges if e not in self.internal_edges]
    self.final_out_edges = [e for e in all_out_edges if e not in set(all_in_edges)]

  @property
  def num_blocks(self) -> int:
    """Number of input blocks (typically 4 for bit planes or channel groups)."""
    return self.ops[0].num_blocks if self.ops else 4

  @property
  def num_out_blocks(self) -> int:
    """Number of output blocks."""
    return self.ops[-1].num_out_blocks if self.ops else 4

  def get_send_block(self) -> 'ImceCallCodeBlock':
    """Return the block whose output variables should be used for SEND.

    For VecOpBlock, this is the last op in the chain (e.g., MinmaxQuantBlock).
    """
    return self.ops[-1] if self.ops else self

  def get_graph_node_id(self):
    return getNodeID(self.call.call)

  def _build_structure(self) -> CodeBlock:
    """Build the computation body.

    Assembles ops in order, adding RECV for mid-chain external inputs.
    First op's external inputs and final SEND are handled by RecvSendWrapper.
    """
    comp = SequentialBlock()
    first_op_edges = set(self.ops[0].in_edges) if self.ops else set()
    received_edges = set()

    for op in self.ops:
      # For non-first ops, RECV external inputs (converging pattern, etc.)
      for edge in op.in_edges:
        if edge in self.internal_edges:
          continue  # Output from prior op -> input to current (no RECV needed)
        if edge in first_op_edges:
          continue  # First op's external inputs handled by RecvSendWrapper
        if edge in received_edges:
          continue  # Already received

        # Mid-chain external input - generate RECV
        self._add_recv_for_edge(comp, edge, op)
        received_edges.add(edge)

      # Add the operation
      comp.add(op)

    return comp

  def _add_recv_for_edge(self, comp: SequentialBlock, edge: TensorEdge, op: 'ImceCallCodeBlock'):
    """Generate RECV code for an edge.

    Based on ConvBlock._build_loop_body() post_op external RECV logic.
    """
    if edge not in DevConfig().TensorEdgetoInfo:
      return

    te_info = DevConfig().TensorEdgetoInfo[edge]

    if te_info.fifo_id == TensorEdgeInfo.LOCAL_FIFO:
      return  # Local edge, no RECV needed

    # Skip constants (handled in INIT phase)
    try:
      arg_id = edge.src_id.graph_node_id
      if ConstPat.match(CustomIDToNode()[arg_id]):
        return
    except KeyError:
      pass

    if te_info.fifo_id != 0:
      # BUGFIX knob: knob=on (bugfix_off_mode()==False) reproduces a8af's
      # per-RECV SETFLAG(uuid)/STANDBY(sender,uuid)/SETFLAG(0) sync.
      if not bugfix_off_mode():
        for i in range(op.num_blocks):
          var_i = UniqueVar((edge, i))
          if var_i.static:
            continue
          annotation = f"{edge}, {te_info.node_info_str}"
          owner_edge = te_info.owner
          recv_code = IMCERecvBlock(str(var_i), te_info.fifo_id, owner_edge, annotation)
          comp.add(recv_code)
          if self.builder and hasattr(self.builder, 'pair_manager') and self.builder.pair_manager:
            pair = self.builder.pair_manager.get_pair(edge)
            if pair:
              sync_code = SequentialBlock()
              sync_code.add(f"// sync after IMCE recv: uuid={pair.uuid}")
              sync_code.add(f"__builtin_IMCE_SETFLAG({pair.uuid});")
              sync_code.add(f"__builtin_IMCE_STANDBY({pair.sender_node.value}, {pair.uuid});")
              sync_code.add(f"__builtin_IMCE_SETFLAG(0);")
              comp.add(sync_code)
        return

      # handcraft window sync: wrap ALL blocks' RECVs for this edge in one
      # window. imce<->imce = SETFLAG(1);RECV..;SETFLAG(0). inode data input =
      # SETFLAG(1);STANDBY(inode,1);SETFLAG(0);RECV.. (window closes before RECV).
      recv_blocks = []
      for i in range(op.num_blocks):
        var_i = UniqueVar((edge, i))
        if var_i.static:
          continue
        annotation = f"{edge}, {te_info.node_info_str}"
        owner_edge = te_info.owner
        recv_blocks.append(IMCERecvBlock(str(var_i), te_info.fifo_id, owner_edge, annotation))
      if not recv_blocks:
        return

      pre_lines, post_lines = (None, None)
      if self.builder and hasattr(self.builder, 'pair_manager') and self.builder.pair_manager:
        pre_lines, post_lines = self.builder.pair_manager.get_recv_window_sync(edge)

      if pre_lines:
        comp.add("\n".join(pre_lines))
      for rb in recv_blocks:
        comp.add(rb)
      if post_lines:
        comp.add("\n".join(post_lines))

  def _render(self) -> str:
    return self.body.render()


class BatchNormBlock(ImceCallCodeBlock):
  """
  BatchNormBlock for batch normalization operations.
  Only generates computation. RECV/SEND handled by wrapper or ConvBlock.
  """
  num_in_edges = 3

  def __init__(self, call: 'BuilderContext', annotation: str = ""):
    """ Code block for batch normalization """
    super().__init__(call, annotation)
    # real_blocks = blocks carrying ACTUAL batch_norm data. When num_blocks is
    # padded up to the producer's STEP-burst width (create_loop_from_call), only
    # the first real_blocks have valid scale/bias; the rest are drain packets
    # (their SENDs become dummy 0 in RecvSendWrapper). Defaults to num_blocks.
    self.real_blocks = self.num_blocks

  def _render(self) -> str:
    """Generate only computation, no RECV/SEND.

    If IMCFLOW_BUGFIX_OVERFLOW_SW=1, uses MultlSWFixOp for overflow bug fix.
    Otherwise, uses simple MULTL instruction.
    """
    code = TextBlock("")

    # Identify edges by tensor type
    data_edge = None
    scale_edge = None
    bias_edge = None
    for edge in self.in_edges:
      if edge.dst_id.tensor_type == "fused_scale":
        scale_edge = edge
      elif edge.dst_id.tensor_type == "fused_bias":
        bias_edge = edge
      elif edge.dst_id.tensor_type == "data":
        data_edge = edge

    # If inside composite, data may come from prev_op
    if data_edge is None and self.prev_op is not None:
      data_edge = self.prev_op

    # Compute only real_blocks; padded drain blocks (i >= real_blocks) have no
    # valid scale/bias and must not be computed (their SEND is a dummy 0).
    real_blocks = getattr(self, "real_blocks", self.num_blocks)
    print("[BatchNormBlock] num blocks:", self.num_blocks, "real:", real_blocks)
    for i in range(real_blocks):
      var_data = self._make_unique_input_var_for_post_op(data_edge, i)
      var_scale = UniqueVar((scale_edge, i)) if scale_edge else UniqueVar((self, i, "scale_placeholder"))
      var_bias = UniqueVar((bias_edge, i)) if bias_edge else UniqueVar((self, i, "bias_placeholder"))
      var_o = UniqueVar((self, i))

      if _is_multl_swfix_enabled():
        # Use composable MultlSWFixOp for overflow bug fix
        var_mult_result = UniqueVar((self, i, "mult_result"))
        code += MultlSWFixOp(var_data, var_scale, var_mult_result, (self, i))
        code += f"{var_o} = __builtin_IMCE_ADD({var_mult_result}, {var_bias}, 15);"
      else:
        # Use simple MULTL instruction
        code += f"{var_o} = __builtin_IMCE_MULTL({var_data}, {var_scale}, 15);"
        code += f"{var_o} = __builtin_IMCE_ADD({var_o}, {var_bias}, 15);"

    return code.render()


class RecvSendWrapper(ImceCodeBlock):
  """
  Wrapper that adds RECV and SEND operations around a computation block.
  """

  def __init__(self, body: CodeBlock, num_blocks: int, num_out_blocks: int, send_block: ImceCodeBlock,
               in_edges: List[TensorEdge], out_edges: List[TensorEdge], annotation: str = "", builder=None,
               skip_presend: bool = False, suppress_presend_only: bool = False,
               post_send_setflag_uuid: int = None):
    """Wrap a computation block with RECV/SEND operations.

    Args:
        body: The inner CodeBlock (usually a SequentialBlock or ImceCallCodeBlock)
        in_edges:
        out_edges:
        annotation: Optional annotation string
        builder: Optional builder reference for pair_manager access
        skip_presend: If True, suppress the pre-send STANDBY rendezvous AND emit
            dummy 0 SENDs. Used for zero-load (recv_count==0) col_groups whose SEND
            is pure zero-padding (handcraft emits a bare SEND with no STANDBY there).
        suppress_presend_only: If True, suppress ONLY the pre-send STANDBY but keep
            REAL data SENDs. Used by the Marker A imce_2_2 boundary reorder, where
            the pre-send STANDBY was already relocated into the LOAD window.
    """
    super().__init__(annotation)
    self.body = body
    self.num_blocks = num_blocks
    self.num_out_blocks = num_out_blocks
    self.in_edges = in_edges
    self.out_edges = out_edges
    self.send_block = send_block
    self.send_map = {}
    self.recv_map = {}
    self.builder = builder
    self.skip_presend = skip_presend
    self.suppress_presend_only = suppress_presend_only
    # Fix G producer side: SETFLAG(uuid) emitted AFTER the SEND loop, notifying the
    # sibling consumer (imce_3_2) that this node's psum for the boundary iteration
    # has been sent, so the consumer's boundary STEP may proceed. None -> no notify.
    self.post_send_setflag_uuid = post_send_setflag_uuid
  
  @classmethod
  def from_codeblock(cls, codeblock: ImceCallCodeBlock, annotation: str="", builder=None):
    # Instead of calling content(), we wrap the codeblock itself
    body = codeblock
    send_block = codeblock
    in_edges = codeblock.in_edges
    out_edges = codeblock.out_edges
    num_blocks = codeblock.num_blocks
    num_out_blocks = codeblock.num_out_blocks
    # Try to get builder from codeblock if not provided
    if builder is None and hasattr(codeblock, 'call') and hasattr(codeblock.call, 'builder'):
      builder = codeblock.call.builder

    return cls(body, num_blocks, num_out_blocks, send_block, in_edges, out_edges, annotation, builder=builder)

  def _render(self) -> str:
    """Generate RECV -> body -> SEND."""
    # BUGFIX knob: knob=on (bugfix_off_mode()==False) reproduces a8af's
    # per-RECV/per-SEND uuid sync (no window sync, no pre-send STANDBY, no
    # dummy-0 SEND / burst-pad).
    if not bugfix_off_mode():
      return self._render_a8af()

    code = TextBlock("")

    # --- 1. Generate RECVs ---
    # handcraft window sync: wrap ALL blocks' RECVs for a given edge in ONE
    # window (SETFLAG(1)...SETFLAG(0)) with no per-RECV STANDBY for imce<->imce
    # pipeline, or a STANDBY(inode,1) window before the RECV for inode data
    # input. So iterate edge-outer / block-inner.
    # Fix D: if this receiver is fed by >=2 inode data-input edges (region2
    # imce_1_3: add lhs from inode_0_0, rhs from inode_1_0), MERGE their
    # per-edge SETFLAG(1);STANDBY(inode,1);SETFLAG(0) windows into ONE window
    # that STANDBYs on every inode sender, closes once, then all the RECVs
    # follow (handcraft). A single flag reg/node (imce_ctrl.sv) means two
    # separate windows toggle 1->0->1->0 and desync from the flag=2 output
    # barrier -> deadlock. region1 has at most one inode data input per
    # receiver -> merge is None -> unchanged.
    merged_input_pre = None
    merged_input_edges = set()
    if self.in_edges and self.builder and hasattr(self.builder, 'pair_manager') and self.builder.pair_manager:
      pm = self.builder.pair_manager
      merged_input_pre = pm.get_merged_inode_input_window(self.in_edges)
      if merged_input_pre is not None:
        merged_input_edges = set(pm.collect_inode_data_input_edges(self.in_edges))
    merged_window_emitted = False

    if self.in_edges:
      for edge in self.in_edges:
        if edge in DevConfig().TensorEdgetoInfo:
          te_info = DevConfig().TensorEdgetoInfo[edge]
        else:
          src_graph_id = edge.src_id.graph_node_id
          dst_graph_id = edge.dst_id.graph_node_id
          assert len(src_graph_id) == 2 and len(dst_graph_id) == 2, "Graph node ID should be tuple of (outer_id, inner_id)"
          assert src_graph_id[0] == dst_graph_id[0], "If src and dst outer node id are different, this edge should be in DevConfig().TensorEdgetoInfo"
          te_info = None

        if te_info and te_info.fifo_id == TensorEdgeInfo.LOCAL_FIFO:
          continue # this edge's src and dst hw node is equal

        try:
          arg_id = edge.src_id.graph_node_id
          if ConstPat.match(CustomIDToNode()[arg_id]):
            continue
        except KeyError:
          pass
        if not te_info or te_info.fifo_id == 0:
          continue

        # Fix D merged path: this edge is one of >=2 inode data inputs -> emit
        # the single merged window once (before the FIRST such edge's RECVs),
        # then this edge's RECVs are bare (window already closed). Both edges'
        # RECVs thus follow the one merged SETFLAG(1);STANDBY..;SETFLAG(0).
        # Fix E (region3 imce_0_2): when num_blocks > 1 (>=2 words per producer
        # SEND-pair), the producer inodes re-arm their pre-send rendezvous flag
        # ONCE PER WORD (per SEND). A single window wrapping all num_blocks RECVs
        # (edge-outer) only raises the consumer flag once, so the producer's 2nd
        # SEND blocks forever on STANDBY(consumer,1) -> starve -> deadlock
        # (fsim-confirmed root cause). handcraft interleaves block-outer and
        # re-raises the window per block: for i: [window; RECV(lhs_i);
        # RECV(rhs_i)]. num_blocks==1 (region2 imce_1_3) collapses to the old
        # single-window behaviour, so region2 is unchanged.
        if merged_input_pre is not None and edge in merged_input_edges:
          if not merged_window_emitted:
            # Silicon-deadlock lever (IMCFLOW_MULTIBLOCK_FUSEDADD_BARE): for a
            # MULTI-BLOCK 2-inode fused-add consumer (region3 imce_1_1,
            # num_blocks==2) drop the per-block SETFLAG(1);STANDBY;SETFLAG(0)
            # window entirely and emit BARE RECVs (backpressure suffices; each
            # edge is a single count-matched RECV fifo). This removes the
            # 1->0->1->0 mid-iteration flag toggle + per-word producer re-arm
            # that races/wedges on real silicon. num_blocks==1 (region2
            # imce_1_3) never enters this branch -> unaffected. Default OFF ->
            # window emitted as before (byte-identical, RTL preserved).
            drop_merged_window = (multiblock_fusedadd_bare()
                                  and self.num_blocks > 1)
            # Silicon-SAFE lever (IMCFLOW_MULTIBLOCK_FUSEDADD_SAFE): REPLACE the
            # per-block SETFLAG(1);STANDBY(P0,1);STANDBY(P1,1);SETFLAG(0) window
            # (which re-arms the SAME value 1 each block -> a 1->0->1->0 toggle
            # collapsible on silicon) with a monotonic phase-token, interlocked,
            # order-independent handshake. For block i:
            #   SETFLAG(2i+1); STANDBY(P0,2i+1); STANDBY(P1,2i+1); SETFLAG(2i+2);
            #   RECV(lhs_i); RECV(rhs_i)
            # Distinct token per block (no repeated edge to alias) + the RECV is
            # the ack: C cannot loop to block i+1 (SETFLAG(2(i+1)+1)) until both
            # producers SENT block i, which requires they observed 2i+2 -> no
            # lost wakeup, any producer arrival order. Producers use the matching
            # token in _get_presend_sync_code_str (keyed on iter%num_blocks).
            # See multiblock_fusedadd_safe() / DESIGN_region3_fusedadd_redesign.
            safe_window = multiblock_fusedadd_safe()
            safe_senders = None
            if safe_window:
              # producer hw-node values feeding this consumer, ASCENDING (matches
              # the baseline STANDBY(P0,..);STANDBY(P1,..) order).
              sset = []
              for me in [e for e in self.in_edges if e in merged_input_edges]:
                p = pm.get_pair(me)
                if p is not None and p.sender_node.value not in sset:
                  sset.append(p.sender_node.value)
              safe_senders = sorted(sset)
              if len(safe_senders) < 2:
                safe_window = False  # only the 2-inode fused-add is retargeted
            # emit ALL merged edges here, block-outer, so the remaining merged
            # edges skip their own emission below.
            merged_edges_ordered = [e for e in self.in_edges if e in merged_input_edges]
            for i in range(self.num_blocks):
              any_recv = False
              block_lines = []
              for me in merged_edges_ordered:
                me_info = DevConfig().TensorEdgetoInfo.get(me)
                if me_info is None:
                  continue
                var_i = UniqueVar((me, i))
                if var_i.static:
                  continue
                any_recv = True
                block_lines.append(
                    f"{var_i} = __builtin_IMCE_RECV({me_info.fifo_id}); "
                    f"// {me}, {me_info.node_info_str}")
                add_to_map(me_info.owner, RecvSendNum("recv", 1), is_send=False)
              if not any_recv:
                continue
              if safe_window:
                # Token base = 3: skip the reserved consumer-flag values on this
                # node -- 0 (idle), 1 (legacy input invite), 2 (OUTPUT multicast
                # barrier watched by the downstream imce receivers, e.g.
                # STANDBY(imce_0_2,2)). Aliasing the output value 2 into an input
                # token would false-trigger a downstream RECV. base 3 -> block b
                # uses {3+2b, 3+2b+1}; num_blocks<=8 -> max 18 < 255 (SYNC_REG),
                # and the producer side uses the identical base 3 (kept in sync
                # by SAFE_TOKEN_BASE below).
                rdy, go = SAFE_TOKEN_BASE + 2 * i, SAFE_TOKEN_BASE + 2 * i + 1
                safe_pre = [f"__builtin_IMCE_SETFLAG({rdy}); // SAFE: invite block {i}"]
                for s in safe_senders:
                  safe_pre.append(f"__builtin_IMCE_STANDBY({s}, {rdy});")
                safe_pre.append(f"__builtin_IMCE_SETFLAG({go}); // SAFE: release block {i} (RECV=ack)")
                code += "\n".join(safe_pre)
              elif not drop_merged_window:
                code += "\n".join(merged_input_pre)
              for line in block_lines:
                code += line
            merged_window_emitted = True
          continue

        # Collect the non-static RECV lines for all blocks of this edge
        recv_lines = []
        for i in range(self.num_blocks):
          var_i = UniqueVar((edge, i))
          if var_i.static:
            continue
          annotation = f"{edge}, {te_info.node_info_str}"
          recv_lines.append(f"{var_i} = __builtin_IMCE_RECV({te_info.fifo_id}); // {annotation}")
          owner_edge = te_info.owner
          add_to_map(owner_edge, RecvSendNum("recv", 1), is_send=False)
        if not recv_lines:
          continue

        pre_lines, post_lines = (None, None)
        if self.builder and hasattr(self.builder, 'pair_manager') and self.builder.pair_manager:
          pre_lines, post_lines = self.builder.pair_manager.get_recv_window_sync(edge)

        if pre_lines:
          code += "\n".join(pre_lines)
        for line in recv_lines:
          code += line
        if post_lines:
          code += "\n".join(post_lines)

    # --- 2. Generate Body ---
    # Here we call content() on the child block(s)
    code += self.body

    # --- 3. Generate SENDs ---
    if self.out_edges:
      out_edge_src_ids = {edge.src_id for edge in self.out_edges}

      # if out edge is more than one -> split operation
      assert len(out_edge_src_ids) == 1, "out_edge_src_ids should have only one element"

      src_id = out_edge_src_ids.pop()
      te_out_infos = DevConfig().get_tensor_edge_info_with_id_dir(
          src_id, "out")

      # handle split ops in middle of functions. For example, qconv2d -> split
      output_edges=self.out_edges
      split_case = all(isinstance(dst_node, relay.Call) and dst_node.op == relay.op.get("split") for dst_node in [CustomIDToNode()[getInnerNodeID(edge.dst_id.graph_node_id)] for edge in self.out_edges])
      # if not te_out_infos: # local edge
      if split_case:
        assert len(self.out_edges) == 1, "In split case, there should be only one out_edge from this block"
        assert te_out_infos[0].fifo_id == TensorEdgeInfo.LOCAL_FIFO, "producer of split should have same HW node ID"
        # if not dst_node.op.name == "split":
        #   raise RuntimeError(f"Warning: no tensor edge info found for src_id {src_id}, dst_node op: {dst_node.op.name}")
        # else:
        # print(f"Info: no tensor edge info found for src_id {src_id}, dst_node is split, checking its output edges")
        split_node_graph_id = self.out_edges[0].dst_id.graph_node_id
        target_tensor_id = TensorID(split_node_graph_id, "odata")
        te_out_infos = DevConfig().get_tensor_edge_info_with_id_dir(target_tensor_id, "out")
        output_edges = [edge for edge in DevConfig().TensorEdgetoInfo.keys() if getInnerNodeID(edge.src_id.graph_node_id) == getInnerNodeID(target_tensor_id.graph_node_id)]
        print(f"Info: got {len(te_out_infos)} tensor edge infos from split dst edge")

      if not te_out_infos: raise RuntimeError(f"no tensor edge info found for src_id {src_id} even after checking split case")

      if split_case:
        assert len(te_out_infos) > 1, "In split case, there should be multiple tensor edge infos"
        addresses = {info.policy_info[0].address for info in te_out_infos}
        assert len(addresses) == 1, "In split case, all output addresses must be identical"
        fifo_ids = {info.fifo_id for info in te_out_infos}
        assert len(fifo_ids) == 1, "When merging same-address outputs, fifo_id must be identical"
        te_out_info = te_out_infos[0] # we need just one edge info due to multicast
      else:
        if len(te_out_infos) > 1:
          addresses = {info.policy_info[0].address for info in te_out_infos}
          assert len(addresses) == 1, "In split case, all output addresses must be identical"
          fifo_ids = {info.fifo_id for info in te_out_infos}
          assert len(fifo_ids) == 1, "When merging same-address outputs, fifo_id must be identical"
          te_out_info = te_out_infos[0] # we need just one edge info due to multicast
        else:
          te_out_info = te_out_infos[0]

      # Producer IMCE signals start of output (flag 0)
      # TEMPORARILY DISABLED: Testing UUID-based sync instead
      # code += f"__builtin_IMCE_SETFLAG(0);"

      # Get actual send block for variable reference
      # For VecOpBlock, use the last op (e.g., MinmaxQuantBlock)
      actual_send_block = self.send_block
      if isinstance(self.send_block, VecOpBlock):
        actual_send_block = self.send_block.get_send_block()

      # handcraft pre-send sync: emit ONE STANDBY(receiver, 1) BEFORE the SEND
      # loop when the receiver is an imce (pipeline). Output to an inode is bare.
      # Suppressed for zero-load col_groups (skip_presend) whose SEND is padding.
      if (te_out_info.fifo_id != TensorEdgeInfo.LOCAL_FIFO and not self.skip_presend
          and not self.suppress_presend_only):
        if self.builder and hasattr(self.builder, 'pair_manager') and self.builder.pair_manager and output_edges:
          pre_send = self.builder.pair_manager.get_pre_send_sync(output_edges[0])
          if pre_send:
            code += "\n".join(pre_send)

      for i in range(self.num_out_blocks):
        for te_out_info in [te_out_info]: #TODO: current version doesn't need it
          if te_out_info.fifo_id == TensorEdgeInfo.LOCAL_FIFO:
            continue # this edge's src and dst hw node is equal

          # Burst-pad drains (i >= real_blocks) send a dummy 0, not an undefined
          # compute var. Keeps SEND count == producer burst (4:1 lock-step) so
          # imce_3_1 -> inode_3_0 matches handcraft (1 real + 3 dummy).
          # skip_presend (pure-padding flush row feeding a fused consumer) emits
          # ALL dummy SENDs (no compute happened) -- handcraft imce_1_2 last row.
          _real = getattr(actual_send_block, "real_blocks", None)
          if self.skip_presend or (_real is not None and i >= _real):
            var_o = "0"
          else:
            var_o = UniqueVar((actual_send_block, i))
          if te_out_info:
            annotation = f"{','.join(map(str, self.out_edges))}, {te_out_info.node_info_str}"
            code += f"__builtin_IMCE_SEND({te_out_info.policy_info[0].address}, {var_o}, {te_out_info.fifo_id}, 0); // {annotation}"
            for out_edge in output_edges:
              add_to_map(out_edge, RecvSendNum("send", 1), is_send=True)

      # Fix G producer side: notify the sibling boundary-STEP consumer that this
      # node's psum SEND for the iteration is complete (paired with the consumer's
      # STANDBY(this_node, uuid) before its final boundary LOAD_LB).
      if self.post_send_setflag_uuid is not None:
        code += (f"//! notify psum send event to the sibling boundary consumer\n"
                 f"__builtin_IMCE_SETFLAG({self.post_send_setflag_uuid});")

      # Producer IMCE signals end of output (flag 2)
      # TEMPORARILY DISABLED: Testing UUID-based sync instead
      # code += f"__builtin_IMCE_SETFLAG(2);"

      # FIXME: the NOP count here is hardcoded
      # nops = " ".join([f"\"nop\\n\"" for _ in range(5)])
      # code += f"__asm__ volatile({nops});"

    return code.render()

  # ---- a8af (knob=on) fallback: per-RECV/per-SEND uuid sync, no windows ----
  def _render_a8af(self) -> str:
    """Generate RECV -> body -> SEND (a8af behavior)."""
    code = TextBlock("")

    # --- 1. Generate RECVs ---
    if self.in_edges:
      for i in range(self.num_blocks):
        for edge in self.in_edges:
          if edge in DevConfig().TensorEdgetoInfo:
            te_info = DevConfig().TensorEdgetoInfo[edge]
          else:
            src_graph_id = edge.src_id.graph_node_id
            dst_graph_id = edge.dst_id.graph_node_id
            assert len(src_graph_id) == 2 and len(dst_graph_id) == 2, "Graph node ID should be tuple of (outer_id, inner_id)"
            assert src_graph_id[0] == dst_graph_id[0], "If src and dst outer node id are different, this edge should be in DevConfig().TensorEdgetoInfo"
            te_info = None

          if te_info and te_info.fifo_id == TensorEdgeInfo.LOCAL_FIFO:
            continue # this edge's src and dst hw node is equal

          try:
            arg_id = edge.src_id.graph_node_id
            if ConstPat.match(CustomIDToNode()[arg_id]):
              continue
          except KeyError:
            pass
          var_i = UniqueVar((edge, i))
          if not te_info or var_i.static:
            continue
          if te_info.fifo_id == 0:
            continue
          annotation = f"{edge}, {te_info.node_info_str}"
          code += f"{var_i} = __builtin_IMCE_RECV({te_info.fifo_id}); // {annotation}"
          owner_edge = te_info.owner
          add_to_map(owner_edge, RecvSendNum("recv", 1), is_send=False)

          if self.builder and hasattr(self.builder, 'pair_manager') and self.builder.pair_manager:
            code = self._add_sync_after_recv(code, edge)

    # --- 2. Generate Body ---
    code += self.body

    # --- 3. Generate SENDs ---
    if self.out_edges:
      out_edge_src_ids = {edge.src_id for edge in self.out_edges}
      assert len(out_edge_src_ids) == 1, "out_edge_src_ids should have only one element"

      src_id = out_edge_src_ids.pop()
      te_out_infos = DevConfig().get_tensor_edge_info_with_id_dir(src_id, "out")

      output_edges = self.out_edges
      split_case = all(isinstance(dst_node, relay.Call) and dst_node.op == relay.op.get("split") for dst_node in [CustomIDToNode()[getInnerNodeID(edge.dst_id.graph_node_id)] for edge in self.out_edges])
      if split_case:
        assert len(self.out_edges) == 1, "In split case, there should be only one out_edge from this block"
        assert te_out_infos[0].fifo_id == TensorEdgeInfo.LOCAL_FIFO, "producer of split should have same HW node ID"
        split_node_graph_id = self.out_edges[0].dst_id.graph_node_id
        target_tensor_id = TensorID(split_node_graph_id, "odata")
        te_out_infos = DevConfig().get_tensor_edge_info_with_id_dir(target_tensor_id, "out")
        output_edges = [edge for edge in DevConfig().TensorEdgetoInfo.keys() if getInnerNodeID(edge.src_id.graph_node_id) == getInnerNodeID(target_tensor_id.graph_node_id)]
        print(f"Info: got {len(te_out_infos)} tensor edge infos from split dst edge")

      if not te_out_infos: raise RuntimeError(f"no tensor edge info found for src_id {src_id} even after checking split case")

      if split_case:
        assert len(te_out_infos) > 1, "In split case, there should be multiple tensor edge infos"
        addresses = {info.policy_info[0].address for info in te_out_infos}
        assert len(addresses) == 1, "In split case, all output addresses must be identical"
        fifo_ids = {info.fifo_id for info in te_out_infos}
        assert len(fifo_ids) == 1, "When merging same-address outputs, fifo_id must be identical"
        te_out_info = te_out_infos[0]
      else:
        if len(te_out_infos) > 1:
          addresses = {info.policy_info[0].address for info in te_out_infos}
          assert len(addresses) == 1, "In split case, all output addresses must be identical"
          fifo_ids = {info.fifo_id for info in te_out_infos}
          assert len(fifo_ids) == 1, "When merging same-address outputs, fifo_id must be identical"
          te_out_info = te_out_infos[0]
        else:
          te_out_info = te_out_infos[0]

      actual_send_block = self.send_block
      if isinstance(self.send_block, VecOpBlock):
        actual_send_block = self.send_block.get_send_block()

      # Max-throughput lever (IMCFLOW_DROP_PSUM): DON'T-CARE output. Drop the
      # per-pixel psum NoC SEND so the next pixel's feed/OUTPUT_HS issues earlier.
      # Only applies to a plain ConvBlock output (the qconv psum drain) to keep
      # every other SEND (weights/config/pipeline) byte-identical. eff bounded:
      # keep 1 SEND per drop_psum_keep_every() pixels (0 -> drop all). Gated by
      # env only; default OFF -> byte-identical.
      _drop_psum = (drop_psum_send()
                    and isinstance(actual_send_block, ConvBlock)
                    and not actual_send_block.post_ops)

      for i in range(self.num_out_blocks):
        for te_out_info in [te_out_info]:
          if te_out_info.fifo_id == TensorEdgeInfo.LOCAL_FIFO:
            continue

          if _drop_psum:
            # bounded-keep is handled at the whole-pixel granularity by the caller
            # (ConvBlock loop); here we simply omit the NoC SEND entirely.
            code += f"// [DROP_PSUM] omitted IMCE_SEND blk {i}\n"
            continue

          var_o = UniqueVar((actual_send_block, i))
          if te_out_info:
            annotation = f"{','.join(map(str, self.out_edges))}, {te_out_info.node_info_str}"
            code += f"__builtin_IMCE_SEND({te_out_info.policy_info[0].address}, {var_o}, {te_out_info.fifo_id}, 0); // {annotation}"
            for out_edge in output_edges:
              add_to_map(out_edge, RecvSendNum("send", 1), is_send=True)

            if self.builder and hasattr(self.builder, 'pair_manager') and self.builder.pair_manager:
              if output_edges:
                code = self._add_sync_after_send(code, output_edges[0])

    return code.render()

  def _add_sync_after_recv(self, code: TextBlock, edge: TensorEdge) -> CodeBlock:
    """Add synchronization after recv for a specific edge

    Returns the updated code block with sync instructions appended
    """
    pair = self.builder.pair_manager.get_pair(edge)
    if pair is None:
      return code

    # Get current node (receiver)
    try:
      dst_gid = edge.dst_id.graph_node_id
      if isinstance(dst_gid, tuple):
        dst_gid = dst_gid[-1]
      current_node = DevConfig().get_hw_node(dst_gid)
      if isinstance(current_node, tuple):
        current_node = current_node[0]
    except Exception:
      return code

    # Create a SequentialBlock for sync instructions
    sync_annotation = f"sync after IMCE recv: uuid={pair.uuid}, edge={edge}"

    sync_block = SequentialBlock()
    sync_block.add(f"// {sync_annotation}")
    sync_block.add(f"__builtin_IMCE_SETFLAG({pair.uuid});")

    # Receiver only needs to wait for sender (not other receivers in multicast)
    sync_block.add(f"__builtin_IMCE_STANDBY({pair.sender_node.value}, {pair.uuid});")

    sync_block.add(f"__builtin_IMCE_SETFLAG(0);")

    # Combine with existing code
    return code + sync_block

  def _add_sync_after_send(self, code: TextBlock, edge: TensorEdge) -> CodeBlock:
    """Add synchronization after send for a specific edge

    Returns the updated code block with sync instructions appended
    """
    pair = self.builder.pair_manager.get_pair(edge)
    if pair is None:
      return code

    # Get current node (sender)
    try:
      src_gid = edge.src_id.graph_node_id
      if isinstance(src_gid, tuple):
        src_gid = src_gid[-1]
      current_node = DevConfig().get_hw_node(src_gid)
    except Exception:
      return code

    # Create a SequentialBlock for sync instructions
    # SENDER pattern: STANDBY(receiver, uuid) → SETFLAG(uuid) → STANDBY(receiver, 0) → SETFLAG(0)
    sync_annotation = f"sync after IMCE send: uuid={pair.uuid}, edge={edge}"

    sync_block = SequentialBlock()
    sync_block.add(f"// {sync_annotation}")

    # First wait for receiver to be ready (receiver sets its flag first)
    for node in pair.all_nodes:
      if node != current_node:
        sync_block.add(f"__builtin_IMCE_STANDBY({node.value}, {pair.uuid});")

    # Then set sender flag to acknowledge
    sync_block.add(f"__builtin_IMCE_SETFLAG({pair.uuid});")

    # Wait for receiver to clear flag (receiver processed data)
    for node in pair.all_nodes:
      if node != current_node:
        sync_block.add(f"__builtin_IMCE_STANDBY({node.value}, 0);")

    # Clear sender flag
    sync_block.add(f"__builtin_IMCE_SETFLAG(0);")

    # Combine with existing code
    return code + sync_block


  def create_loop_from_call(self, call_ctx : 'BuilderContext', to_process_in_edges=None):
    """Wrap the content in a loop based on call's type_args.

    Returns a SimpleFor object.
    """

    sinple_out_edge = (len(self.out_edges) == 1)
    multicast = all(edge.src_id == self.out_edges[0].src_id for edge in self.out_edges) and (not sinple_out_edge)
    assert sinple_out_edge or multicast, "Only single output edge or multicast is supported in create_loop_from_call"

    call = call_ctx.call
    
    # FIXME: this is a quick fix for ruling out the constant edges using to_process_in_edges
    in_edges = to_process_in_edges or self.in_edges

    data_edge = next(
        edge for edge in in_edges if edge.dst_id.tensor_type in ["data", "rhs", "lhs"])
    out_edge = self.out_edges[0]

    datablock = DevConfig().CurrFuncMemLayout.get_data_block_by_edge(data_edge)

    arg_types = []
    ret_type  = None

    # get input edge size
    args = []
    if isinstance(call, relay.Call) and call.op == relay.op.get("concatenate"):
      args = call.args[0].fields  # for concat case 
    else:
      for idx, arg in enumerate(call.args):
        if ConstPat.match(arg):
          continue
        else:
          args.append(arg)

    real_ttype = None
    ch_size = None

    for arg in args:
      layout = DevConfig().LayoutMap[arg]
      vir_ttype = get_type(call_ctx.module, arg)
      arg_types.append(vir_ttype)
      ch_size = vir_ttype.shape[1]  # assuming NCHW
      real_ttype_ = apply_layout_to_type(vir_ttype, layout)
      if real_ttype and real_ttype != real_ttype_:
        raise RuntimeError("Mismatched real tensor types among input args")
      real_ttype = real_ttype_
    
    elem_count = math.prod(list(real_ttype.shape))
    if isinstance(elem_count, tvm.tir.expr.IntImm):
      elem_count = elem_count.value
    dtype = real_ttype.dtype
    if "int8" in dtype or "uint8" in dtype:
      bytes_per_elem = 1
    elif "int16" in dtype or "uint16" in dtype:
      bytes_per_elem = 2
    elif "int32" in dtype or "uint32" in dtype or "float32" in dtype:
      bytes_per_elem = 4
    else:
      raise RuntimeError(f"Unsupported dtype {dtype} in create_loop_from_call")
    in_total_bytes = elem_count * bytes_per_elem

    # get output edge size
    layout = DevConfig().LayoutMap[call_ctx.call]
    ret_type = get_type(call_ctx.module, call_ctx.call)
    real_ttype = apply_layout_to_type(ret_type, layout)
    
    elem_count = math.prod(list(real_ttype.shape))
    if isinstance(elem_count, tvm.tir.expr.IntImm):
      elem_count = elem_count.value
    dtype = real_ttype.dtype
    if "int8" in dtype or "uint8" in dtype:
      bytes_per_elem = 1
    elif "int16" in dtype or "uint16" in dtype:
      bytes_per_elem = 2
    elif "int32" in dtype or "uint32" in dtype or "float32" in dtype:
      bytes_per_elem = 4
    else:
      raise RuntimeError(f"Unsupported dtype {dtype} in create_loop_from_call")
    out_total_bytes = elem_count * bytes_per_elem

    assert in_total_bytes % 32 == 0, "Input total bytes must be multiple of 32"
    assert out_total_bytes % 32 == 0, "Output total bytes must be multiple of 32"
    num_blocks=None
    num_out_blocks=None
    count=None
    if out_total_bytes >= in_total_bytes:
      ratio = float(out_total_bytes) / float(in_total_bytes)
      assert ratio.is_integer(), "Output to input byte size ratio must be integer"
      num_blocks=1
      num_out_blocks=int(ratio)
      count=in_total_bytes//32
    else:
      ratio = float(in_total_bytes) / float(out_total_bytes)
      assert ratio.is_integer(), "Input to output byte size ratio must be integer"
      num_blocks=int(ratio)
      num_out_blocks=1
      count=out_total_bytes//32
    
    # Get actual send block (for VecOpBlock, use the last op in the chain)
    # This ensures proper handling when blocks like MinmaxQuantBlock, ConcatBlock,
    # or BatchNormBlock are wrapped inside a VecOpBlock
    actual_send_block = self.send_block
    if isinstance(self.send_block, VecOpBlock):
      actual_send_block = self.send_block.get_send_block()

    # Apply block-specific adjustments based on the actual send block type
    if isinstance(actual_send_block, MinmaxQuantBlock):
      # MinmaxQuant has minimum granularity of 4 (must recv all input channels first)
      if num_out_blocks <= 4:
        ratio = 4 // num_out_blocks
        num_blocks = num_blocks * ratio
        count = count // ratio
        num_out_blocks = 4
      else:
        ratio = num_out_blocks // 4
        num_out_blocks = 4
        count = count * ratio
        num_blocks = num_blocks // ratio
      actual_send_block._num_blocks = num_blocks
      arg_vtype = arg_types[0]
      N, IC, H, W = arg_vtype.shape
      N, IC, H, W = N.value, IC.value, H.value, W.value
      actual_send_block.channels = IC
    elif isinstance(actual_send_block, ConcatBlock):
      # For concat, num_out_blocks is determined by number of input edges
      # Each input edge has num_blocks blocks
      num_blocks = actual_send_block.num_blocks
      num_out_blocks = actual_send_block.num_out_blocks
      count = in_total_bytes // (32 * num_blocks)
    elif isinstance(actual_send_block, BatchNormBlock):
      # For standalone BatchNorm, use num_blocks from the block which is
      # calculated based on scale/bias edge size
      bn_num_blocks = actual_send_block.num_blocks
      if bn_num_blocks > 1 and num_blocks < bn_num_blocks:
        # Adjust count and num_blocks to match the scale/bias recv_count
        ratio = bn_num_blocks // num_blocks
        count = count // ratio
        num_blocks = bn_num_blocks
        num_out_blocks = bn_num_blocks
      # Burst-pad: the producer conv sends GRANULARITY (=4) values per STEP
      # (GET_CREG 0..3 -> SEND x4). A standalone BN with num_blocks < GRANULARITY
      # consumes fewer per window -> 1:GRANULARITY cadence mismatch vs producer
      # -> FIFO back-pressure -> imce_2_1 tx X-fatal (router_flow_rx[11]/[16]).
      # Pad num_blocks up to GRANULARITY, keep real_blocks = actual computed
      # blocks; extra RECVs drain, extra SENDs are dummy 0. Reproduces handcraft
      # imce_3_1 (loop/4, RECVx4, SENDx4 with 1 real + 3 dummy).
      # BUGFIX knob: burst-pad is part of the 934 sync work; a8af (knob=on) had
      # no burst-pad -> skip it.
      GRANULARITY = 4
      real_blocks = num_blocks
      if bugfix_off_mode() and num_out_blocks == real_blocks and real_blocks < GRANULARITY and (GRANULARITY % real_blocks == 0):
        ratio = GRANULARITY // real_blocks
        count = count // ratio
        num_blocks = GRANULARITY
        num_out_blocks = GRANULARITY
        actual_send_block.real_blocks = real_blocks
        actual_send_block._num_blocks = num_blocks
    elif isinstance(actual_send_block, VecBlock):
      # For standalone VecBlock (Add, Mult, Div), use channel-based num_blocks
      # The num_blocks should be channels / 16, not byte-ratio based
      vec_num_blocks = actual_send_block.num_blocks  # Already calculated from output shape
      if vec_num_blocks > 1:
        # Adjust count to account for more blocks
        ratio = vec_num_blocks // num_blocks if num_blocks > 0 else vec_num_blocks
        count = count // ratio if ratio > 1 else count
        num_blocks = vec_num_blocks
        num_out_blocks = vec_num_blocks
      actual_send_block._num_blocks = num_blocks
      actual_send_block._num_out_blocks = num_out_blocks

    # Create a new RecvSendWrapper that represents the inner logic
    inner = RecvSendWrapper(self.body, num_blocks, num_out_blocks,
                            self.send_block, self.in_edges, self.out_edges, self.annotation, builder=self.builder)

    return SimpleFor(count, inner, f"call_created_loop")

class ImceCodeBlockManager(NodeCodeBlockManager):
  """A class that manages and generates code blocks for imces."""

  def __init__(self, func_name: str):
    super().__init__()
    self.func_name = func_name

  @property
  def nodes(self) -> List[NodeID]:
    return NodeID.imces()

  @property
  def target(self) -> str:
    return "imce"

  def start_block(self) -> str:
    code = (
        "#include \"../common_decl.h\"\n"
        f"void {self.func_name}() {{\n"
        "  int hid = __builtin_IMCE_GET_CORE_HID();\n"
        "  int wid = __builtin_IMCE_GET_CORE_WID();\n"
        f"{indent(UniqueVar.get_decls_str(), '  ')}\n"
    )
    return code

  def end_block(self) -> str:
    return "}\n"


"""
  short16 test_builtins(short16 a, short16 b) {
  short16 var1 = __builtin_IMCE_ADD(a, b, 15);
  short16 var2 = __builtin_IMCE_SUB(a, var1, 15);
  short16 var3 = __builtin_IMCE_AND(a, var2, 15);
  short16 var4 = __builtin_IMCE_OR(a, var3, 15);
  short16 var5 = __builtin_IMCE_XOR(a, var4, 15);
  short16 var6 = __builtin_IMCE_SRL(a, var5, 15);
  short16 var7 = __builtin_IMCE_SLL(a, var6, 15);
  short16 var8 = __builtin_IMCE_SRA(a, var7, 15);
  short16 var9 = __builtin_IMCE_MAX(a, var8, 15);
  short16 var10 = __builtin_IMCE_MIN(a, var9, 15);
  short16 var11 = __builtin_IMCE_MULTL(a, var10, 15);
  short16 var12 = __builtin_IMCE_MULTH(a, var11, 15);

  short16 var14 = __builtin_IMCE_ADDI(var12, 1);
  short16 var15 = __builtin_IMCE_SUBI(var14, 1);
  short16 var16 = __builtin_IMCE_ANDI(var15, 1);
  short16 var17 = __builtin_IMCE_ORI(var16, 1);
  short16 var18 = __builtin_IMCE_XORI(var17, 1);
  short16 var19 = __builtin_IMCE_SRLI(var18, 1);
  short16 var20 = __builtin_IMCE_SLLI(var19, 1);
  short16 var21 = __builtin_IMCE_SRAI(var20, 1);
  short16 var22 = __builtin_IMCE_MAXI(var21, 1);
  short16 var23 = __builtin_IMCE_MINI(var22, 1);
  short16 var24 = __builtin_IMCE_MULTLI(var23, 1);
  short16 var25 = __builtin_IMCE_MULTHI(var24, 1);

  short16 var26 = __builtin_IMCE_DWCONV(var25, 1, 0, 1, 1);
  __builtin_IMCE_SEND(1, var26, 2, 3);
  short16 var27 = __builtin_IMCE_RECV(0);
  short16 var_min = __builtin_IMCE_RECV_MIN(0);
  short16 var_max = __builtin_IMCE_RECV_MAX(0);
  short16 var_cfg = __builtin_IMCE_RECV_CFG(0);
  short16 var_scan0 = __builtin_IMCE_RECV_SREG0(0);
  short16 var_scan1 = __builtin_IMCE_RECV_SREG1(0);
  __builtin_IMCE_SETFLAG(1);
  __builtin_IMCE_STANDBY(1, 2);

  short16 var28 = __builtin_IMCE_MAXPOOL(1, 2, 3);
  short16 var29 = __builtin_IMCE_AVGPOOL(1, 2, 3);

  __builtin_IMCE_ADDQ(var27, var28, 1, 2);
  __builtin_IMCE_SUBQ(a, var29, 1, 2);
  __builtin_IMCE_MULTLQ(a, var29, 1, 2);
  __builtin_IMCE_MULTHQ(a, var29, 1, 2);
  __builtin_IMCE_NU_QUANT(a, var29, 1, 2);
  __builtin_IMCE_MM_QUANT(a, 0, 15, 2);
  short16 var30 = __builtin_IMCE_GET_QREG(0);
  short16 var31 = __builtin_IMCE_GET_QREG(1);
  short16 var32 = __builtin_IMCE_GET_QREG(2);
  short16 var33 = __builtin_IMCE_GET_QREG(3);
  short16 var_0 = __builtin_IMCE_ADD(var30, var31, 15);
  short16 var_1 = __builtin_IMCE_ADD(var32, var33, 15);

  __builtin_IMCE_STEP();
  __builtin_IMCE_NOP();
  __builtin_IMCE_STOP();
  short16 var34 = __builtin_IMCE_GET_CREG(0);
  short16 var35 = __builtin_IMCE_GET_CREG(1);
  short16 var36 = __builtin_IMCE_GET_CREG(2);
  short16 var37 = __builtin_IMCE_GET_CREG(3);

  short16 var_2 = __builtin_IMCE_ADD(var34, var35, 15);
  short16 var_3 = __builtin_IMCE_ADD(var36, var37, 15);

  short16 var38 = __builtin_IMCE_SCAN_RW(a);

  short16 var_4 = __builtin_IMCE_ADD(var_0, var_1, 15);
  short16 var_5 = __builtin_IMCE_ADD(var_2, var_3, 15);
  short16 var_6 = __builtin_IMCE_ADD(var_4, var_5, 15);

  __builtin_IMCE_LOAD_LB(0);
"""
