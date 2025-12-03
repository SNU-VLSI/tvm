import re
import os
import tvm
from tvm import relay
from tvm.relay import op
from tvm.relay.frontend.common import infer_shape
from tvm.relay.dataflow_pattern import *
from tvm.contrib.imcflow import TensorID, TensorEdge
from tvm.relay.op.contrib.imcflow import CustomIDToNode
from tvm.relay.backend.contrib.imcflow import util
from tvm.relay.backend.contrib.imcflow import transform
from tvm.relay.backend.contrib.imcflow.transform import getNodeID
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from tvm.contrib.imcflow import CodegenContext
from tvm.relay.backend.contrib.imcflow.kernel_codegen import KernelCodegen
from tvm.relay.backend.contrib.imcflow.device_codegen import DeviceCodegen
from tvm.relay.backend.contrib.imcflow.codeblock import *
from tvm.relay.backend.contrib.imcflow.inode_codeblock import *
from tvm.relay.backend.contrib.imcflow.imce_codeblock import *
from tvm.relay.backend.contrib.imcflow.operation_handlers import get_handler_registry
import pdb

# Ensure external codegen registration side-effects are loaded.
from . import ext_codegen as _imcflow_ext_codegen  # noqa: F401
# Load operation handlers (imports trigger registration via decorators)
from . import imce_operation_handlers  # noqa: F401
from tvm.relay.backend.contrib.imcflow.imce_operation_handlers import IMCECodeBlockInfo

CompositePat = wildcard().has_attr({"Composite": "imcflow.qconv2d-with-postop"})(None)
TuplePat = is_tuple(None)
TupleGetItemPat = is_tuple_get_item(wildcard())
VarPat = is_var()
ConstPat = is_constant()


@util.create_imcflow_function_pass(opt_level=0)
class CodegenSuite:
  """A pass that generates/compiles code for IMCFlow functions"""

  def __init__(self, build_dir, module, host_isa="arm"):
    self.build_dir = build_dir
    self.host_isa = host_isa
    self.module = module
    if not os.path.exists(build_dir):
      os.makedirs(build_dir)

    common_decl = f"""
      typedef short short16 __attribute__((ext_vector_type(16)));
      __attribute__((noinline, used)) void __builtin_IMCE_STEP(void);
    """
    with open(f"{build_dir}/common_decl.h", "w") as file:
      file.write(common_decl)

  def transform_function(self, _, func):
    # Note: the function name strips off the "_impl" suffix to match the original funcion name
    # which is the parent func's global_symbol attribute (prior: func.attsr.global_symbol).
    func_name = func.attrs["Composite"].strip("_impl")

    # Set the codegen context for this function
    CodegenContext().set_func_name(func_name)

    # annotate edges between (non-composite) calls,
    # while translating vars into corresponding calls
    annotator = InternalEdgeAnnotator(func_name)
    annotator.visit(func)

    print(f"Annotated edges for function {func_name}:")
    for edge in annotator.edges:
      print(f"  {edge}")

    # clear IMCECodeBlockInfo before codegen
    IMCECodeBlockInfo().clear()

    # generate code blocks for each node
    builder = ImceCodeBlockBuilder(self.module, func_name, annotator.edges)
    builder.visit(func)

    # add stop block for active imces
    for hid in DevConfig().ActiveIMCEPerFunc[func_name]:
      block = CtrlBlock("STOP")
      builder.codeblocks.append(hid, block, CodePhase.END)

    DeviceCodegen("imce", self.build_dir, self.host_isa).handle_code_generation(
        func_name, builder.codeblocks)

    builder = InodeCodeBlockBuilder(func_name, annotator.edges)
    builder.visit(func)

    # add sync logic after INIT Phase
    builder.sync_inrt_clear(CodePhase.INIT)

    DeviceCodegen("inode", self.build_dir, self.host_isa).handle_code_generation(
        func_name, builder.codeblocks)

    PolicyTableCodegen(func_name, self.build_dir, self.host_isa).generate(func_name)

    # Clear the codegen context when done
    CodegenContext().clear()

    return func


class PolicyTableCodegen:
  """
  Write out a binary file for policy tables for each node.
  """

  def __init__(self, func_name, build_dir="/tmp", host_isa="arm"):
    super().__init__()
    self.build_dir = build_dir
    self.host_isa = host_isa
    self.func_dir = os.path.join(build_dir, func_name)

  def pack_to_bin(self, entry, endian):
    assert set(entry.keys()) == {
        'Local', 'North', 'East', 'South', 'West'}, "Invalid policy table entry"

    def get_bits(val, num_bits):
      return (val & ((1 << num_bits) - 1)) if val is not None else 0

    val = 0
    for direction in ['Local', 'North', 'East', 'South', 'West']:
      conf = entry[direction]
      val = (val << 1) | (1 if conf["enable"] else 0)
      val = (val << 6) | get_bits(conf["addr"], 6)
      if direction == 'Local':
        val = (val << 3) | 0b000
        val = (val << 6) | get_bits(conf["chunk_index"], 6)

    bin_data = bytearray()
    bin_data.extend(val.to_bytes(32, byteorder=endian, signed=False))
    return bytes(bin_data)

  def generate(self, func_name):
    for node_name, entries in transform.ImcflowDeviceConfig().PolicyTableDict.items():
      policytable_path = os.path.join(
          self.func_dir, f"{node_name.name}_policy")
      policytable_bin_file = f"{policytable_path}.bin"
      policytable_host_obj_file = f"{node_name.name}_policy.host.o"
      with open(policytable_bin_file, "wb") as file:
        for entry in entries:
          policytable_bin = self.pack_to_bin(entry, endian='little')
          file.write(policytable_bin)
      if ("inode" in node_name.name):
        DevCodegen = DeviceCodegen("inode", self.build_dir, self.host_isa)
        DevCodegen.func_dir = self.func_dir
        DevCodegen.create_host_object(
            f"{node_name.name}_policy.bin", policytable_host_obj_file)
      if ("imce" in node_name.name):
        DevCodegen = DeviceCodegen("inode", self.build_dir, self.host_isa)
        DevCodegen.func_dir = self.func_dir
        DevCodegen.create_host_object(
            f"{node_name.name}_policy.bin", policytable_host_obj_file)
    return


class InternalEdgeAnnotator(tvm.relay.ExprVisitor):
  def __init__(self, func_name):
    super().__init__()
    self.stack = [] # track composite call stacks
    # self.edges = set(TensorEdge._instances.values())
    self.edges = set(DevConfig().TensorEdgeListDict[func_name])
  
  @property
  def composite_call(self):
    return self.stack[-1] if self.stack else None

  def add_edge(self, dst_tid, arg, split_idx=None):
    # handle tuple using recursion
    if TuplePat.match(arg):
      for a in arg.fields:
        self.add_edge(dst_tid, a)
      return
    elif TupleGetItemPat.match(arg):
      self.add_edge(dst_tid, arg.tuple_value, split_idx=arg.index)
      return

    src_tid = self.get_tensor_id(arg, "odata")
    self.edges.add(TensorEdge(src_tid, dst_tid, split_idx))  # add edge to set

  def visit_call(self, call):
    if CompositePat.match(call):
      self.visit_composite_call(call)
    else:
      self.visit_regular_call(call)

  def visit_composite_call(self, call):
    self.stack.append(call)
    self.visit(call.op.body)
    self.stack.pop()
    for a in call.args:
      self.visit(a)

  def visit_regular_call(self, call):
    self.visit(call.op)

    # add edges for internal edges
    if self.composite_call:
      for idx, a in enumerate(call.args):
        if VarPat.match(a) or ConstPat.match(a) or not hasattr(call.op, "arguments"):
          continue
        dst_tag = call.op.arguments[idx].name
        dst_tid = self.get_tensor_id(call, dst_tag)
        self.add_edge(dst_tid, a)

    for a in call.args:
      self.visit(a)

  def get_tensor_id(self, call, tag):
    if self.composite_call:
      return TensorID((getNodeID(self.composite_call), getNodeID(call)), tag)
    else:
      return TensorID(getNodeID(call), tag)


class ImceCodeBlockBuilder(tvm.relay.ExprVisitor):
  """Visitor that generates IMCE code blocks from relay operations.

  This class uses a pluggable handler registry to process different operation types.
  New operations can be supported by creating handler classes and registering them
  with the @register_operation_handler decorator in imce_operation_handlers.py.

  Handlers receive a BuilderContext that wraps each call with helper methods.
  """

  def __init__(self, module, func_name, edges):
    super().__init__()
    # Shared state accessed by handlers through BuilderContext
    self.module = module
    self.edges = edges
    self.codeblocks = ImceCodeBlockManager(func_name)
    self.curr_composite_id = None
    self.curr_conv_block = None
    self.last_tuple_idx = None
    self._handler_registry = get_handler_registry()

  def visit_tuple(self, tup):
    for idx, x in enumerate(tup.fields):
      self.last_tuple_idx = idx
      self.visit(x)

  def visit_call(self, call):
    # Visit arguments first (post-order traversal)
    for idx, a in enumerate(call.args):
      self.visit(a)

    # Dispatch to handler registry (automatically wraps call in BuilderContext)
    handled = self._handler_registry.handle(call, self)

    # Fallback for unhandled operations
    if not handled:
      self.visit(call.op)


class InodeCodeBlockBuilder(tvm.relay.ExprVisitor):
  def __init__(self, func_name, edges):
    super().__init__()
    self.edges = edges
    self.codeblocks = InodeCodeBlockManager(func_name)
    # Track which hardware nodes already have an IMCE compute block added
    self._imce_compute_added = set()
    self.initialize()
    self.curr_composite_id = None
    self.finalize()
  
  def sync_inrt_clear(self, codephase: CodePhase):
    # standby and intrt
    inode_master = NodeID.inode_3_0
    inode_slaves = [node for node in NodeID.inodes() if node != inode_master]
    block = StandbyAndIntrtBlock(inode_slaves, "standby and intrt")
    self.codeblocks.append(inode_master, block, codephase)

    # set_flag
    block = SetFlagAndHaltBlock()
    for inode_slv in inode_slaves:
      self.codeblocks.append(inode_slv, block, codephase)

    # clear flag
    for inode in NodeID.inodes():
      block = ClearFlag("clear flag")
      self.codeblocks.append(inode, block, codephase)

  def initialize(self):
    # clear flag
    for inode in NodeID.inodes():
      block = ClearFlag("clear flag before policy update")
      self.codeblocks.append(inode, block, CodePhase.INIT)

    # policy update
    for inode in NodeID.inodes():
      block = PolicyUpdateBlock(inode, "policy update")
      self.codeblocks.append(inode, block, CodePhase.INIT)

    # sync and intrt and clear
    self.sync_inrt_clear(CodePhase.INIT)

    # imem write
    for imce, inst_edge in DevConfig().InstEdgeInfoDict.items():
      block = WriteIMEMBlock(inst_edge, f"imem write: {imce.name}")
      self.codeblocks.append(imce.master(), block, CodePhase.INIT)

    # imcu write
    for node in NodeID.inodes():
      block = WriteIMCUBlock(node, "imcu write")
      self.codeblocks.append(node, block, CodePhase.INIT)

    # imce compute
    active_imces = DevConfig().ActiveIMCEPerFunc[self.codeblocks.func_name]
    for imce, inst_edge in DevConfig().InstEdgeInfoDict.items():
      if imce in active_imces:
        policy_addr = inst_edge.policy_info[0].address # get first policy address
        block = IMCEComputeBlock(policy_addr, f"{imce.name} compute")
        self.codeblocks.append(imce.master(), block, CodePhase.INIT)
    
    # wait all enable of imce
    for inode in NodeID.inodes():
      block = SetFlag()
      self.codeblocks.append(inode, block, CodePhase.INIT)
      other_nodes = [n for n in NodeID.inodes() if n != inode]
      block = Standby(node_ids=other_nodes, annotation=f"standby for {inode.name}")
      self.codeblocks.append(inode, block, CodePhase.INIT)

      block = ClearFlag("clear flag after imce compute enable")
      self.codeblocks.append(inode, block, CodePhase.INIT)
    
    # send constant goes is taken care of by graph traverse


  def finalize(self):
    # standby and intrt
    # FIXME: hardcoded inode_3_0
    inode_master = NodeID.inode_3_0
    inode_slaves = [node for node in NodeID.inodes() if node != inode_master]
    block = StandbyAndIntrtBlock(inode_slaves, "standby and intrt")
    self.codeblocks.append(inode_master, block, CodePhase.END)

    # set_flag
    block = SetFlagAndHaltBlock()
    for inode_slv in inode_slaves:
      self.codeblocks.append(inode_slv, block, CodePhase.END)

  def visit_function(self, fn):
    # constant tensor tags except "weight" (IMCU weights are handled separately)
    param_edges = []
    const_edges = []
    output_edges = []
    for x in fn.params:
      # self.visit(x)
      param_id = getNodeID(x)
      # The input variable will go to the same router entry => only need one send block
      param_edge = self.get_output_edges_from_id(param_id)[0]
      param_edges.append(param_edge)
      # self.add_send_block(param_edge)
    #self.visit(fn.body)
    # traverse constant nodes

    for edge in self.edges:
      arg_id = edge.src_id.graph_node_id
      if ConstPat.match(CustomIDToNode()[arg_id]):
        if edge.src_id.tensor_type != "weight":
          const_edges.append(edge)
          # self.add_send_block(edge)

    # Add Recv Block
    fn_id = getNodeID(fn)
    fn_edges = self.get_input_edges_from_id(fn_id)
    for last_edge in fn_edges:
      output_edges.append(last_edge)
      # self.add_recv_block(last_edge)
    
    # send const edge interleaved
    #TODO: consider recv node order..

    # add send block for const edges based on imce const edge ordering
    imce_edges = IMCECodeBlockInfo()
    for hid in imce_edges.imce_const_edges.keys():
      const_edge_list = imce_edges.imce_const_edges[hid]
      for edge in const_edge_list:
        if edge in const_edges:
          self.add_send_block(edge, CodePhase.INIT)
        else:
          assert False, f"const edge {edge} in IMCECodeBlockInfo not found in function const edges"

    # send param edge interleaved
    #TODO: if edge count is more than one, interleave them
    if len(param_edges) == 1:
      self.add_send_block(param_edges[0], CodePhase.EXEC)
    else:
      self.add_send_block_interleaved(param_edges, CodePhase.EXEC)

    # recv output edge interleaved
    #TODO: if edge count is more than one, interleave them
    for edge in output_edges:
      self.add_recv_block(edge, CodePhase.EXEC)

  def add_send_block(self, edge, phase: CodePhase, db=None):
    out_edge_info = DevConfig().get_tensor_edge_info(edge)
    tid = edge.src_id
    hid = self.get_hid(tid)
    # split op handling => pass down to inner edges from input var node
    gid = edge.dst_id.graph_node_id
    if db is None:
      db = DevConfig().CurrFuncMemLayout.get_data_block_by_edge(edge)
      if db is None:
        pass
      assert db is not None, f"Data block not found for edge: {edge}"

    if CustomIDToNode()[gid].op.name == "split":
      inner_node = CustomIDToNode()[gid]
      for inner_edge in self.get_output_edges_from_id(getNodeID(inner_node)):
        self.add_send_block(inner_edge, phase, db)
        return

    annotation = f"send - {edge}, {out_edge_info.policy_info[0].router_id.name} -> {out_edge_info.policy_info[-1].router_id.name}"
    block = SendBlock(db, out_edge_info, annotation)
    self.codeblocks.append(hid, block, phase)
  
  def add_send_block_interleaved(self, edge_list, phase: CodePhase):
    dbs = []
    edge_infos = []
    # FIXME: change this if multiple inodes send params
    hids = [self.get_hid(edge.src_id) for edge in edge_list]
    assert all(hid == hids[0] for hid in hids), "all edges should have same starting inode for interleaved"
    hid = hids[0]

    def append_edgeinfo_and_db(edge: TensorEdge, edge_infos: List, dbs: List):
      out_edge_info = DevConfig().get_tensor_edge_info(edge)
      edge_infos.append(out_edge_info)
      dbs.append(db)
      annotation = f"send - {edge}, {out_edge_info.policy_info[0].router_id.name} -> {out_edge_info.policy_info[-1].router_id.name}, "
      return annotation


    annotation = ""
    for edge in edge_list:
      gid = edge.dst_id.graph_node_id
      db = DevConfig().CurrFuncMemLayout.get_data_block_by_edge(edge)

      # split handling
      if CustomIDToNode()[gid].op.name == "split":
        inner_node = CustomIDToNode()[gid]
        for inner_edge in self.get_output_edges_from_id(getNodeID(inner_node)):
          annotation += append_edgeinfo_and_db(inner_edge, edge_infos, dbs)
          break
      else:
        annotation += append_edgeinfo_and_db(edge, edge_infos, dbs)

    block = SendBlockInterleaved(dbs, edge_infos, annotation)
    self.codeblocks.append(hid, block, phase)

  def add_recv_block(self, edge, phase: CodePhase):
    in_edge_info = DevConfig().get_tensor_edge_info(edge)
    in_tid = edge.dst_id
    hid = self.get_hid(in_tid)
    db = DevConfig().CurrFuncMemLayout.get_data_block_by_edge(edge)

    block = RecvBlock(db, in_edge_info.fifo_id, f"recv: {in_tid}")
    self.codeblocks.append(hid, block, phase)
  
  def add_recv_block_interleaved(self, edge_list):
    pass

  def get_graph_node_id(self, call):
    if self.curr_composite_id:
      return (self.curr_composite_id, getNodeID(call))
    else:
      return getNodeID(call)

  def get_input_edges_from_id(self, id):
    return [edge for edge in self.edges if edge.dst_inner_gid_match(id)]

  def get_output_edges_from_id(self, id):
    return [edge for edge in self.edges if edge.src_inner_gid_match(id)]

  def get_hid(self, tensor_id):
    gid = tensor_id.graph_node_id
    if isinstance(gid, tuple):
      gid = gid[-1]
    return DevConfig().get_hw_node(gid)
