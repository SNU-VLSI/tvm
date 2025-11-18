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

CompositePat = wildcard().has_attr({"Composite": "imcflow.qconv2d-with-postop"})(None)
TuplePat = is_tuple(None)
TupleGetItemPat = is_tuple_get_item(wildcard())
VarPat = is_var()
ConstPat = is_constant()


@util.create_imcflow_function_pass(opt_level=0)
class CodegenSuite:
  """A pass that generates/compiles code for IMCFlow functions"""

  def __init__(self, build_dir, host_isa="arm"):
    self.build_dir = build_dir
    self.host_isa = host_isa
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

    # annotate edges between (non-composite) calls,
    # while translating vars into corresponding calls
    annotator = InternalEdgeAnnotator()
    annotator.visit(func)

    print(f"Annotated edges for function {func_name}:")
    for edge in annotator.edges:
      print(f"  {edge}")

    # generate code blocks for each node
    builder = ImceCodeBlockBuilder(func_name, annotator.edges)
    builder.visit(func)

    # add stop block for active imces
    for hid in DevConfig().ActiveIMCEPerFunc[func_name]:
      block = CtrlBlock("STOP")
      builder.codeblocks.append(hid, block, CodePhase.END)

    DeviceCodegen("imce", self.build_dir, self.host_isa).handle_code_generation(
        func_name, builder.codeblocks)

    builder = InodeCodeBlockBuilder(func_name, annotator.edges)
    builder.visit(func)
    DeviceCodegen("inode", self.build_dir, self.host_isa).handle_code_generation(
        func_name, builder.codeblocks)

    PolicyTableCodegen(func_name, self.build_dir, self.host_isa).generate(func_name)

    return func


class PolicyTableCodegen:
  """
  Write out a binary file for policy tables for each node.
  """

  def __init__(self, func_name, build_dir="/tmp", host_isa="arm"):
    super().__init__()
    self.func_name = func_name
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
  def __init__(self):
    super().__init__()
    self.composite_call = None
    self.stack = []
    self.edges = set(TensorEdge._instances.values())

  def add_edge(self, dst_tid, arg, split_idx=None):
    # pass arg in below cases
    if CompositePat.match(arg):
      self.stack.append(arg)
      self.add_edge(dst_tid, arg.op.body)
      self.stack.pop()
      return
    elif TuplePat.match(arg):
      for a in arg.fields:
        self.add_edge(dst_tid, a)
      return
    elif TupleGetItemPat.match(arg):
      self.add_edge(dst_tid, arg.tuple_value, split_idx=arg.index)
      return
    elif VarPat.match(arg) and self.composite_call:
      for idx, p in enumerate(self.composite_call.op.params):
        if p == arg:
          a = self.composite_call.args[idx]
          self.stack.append(None)
          self.add_edge(dst_tid, a)
          self.stack.pop()
      return

    src_composite = self.stack[-1] if self.stack else None

    # override src tag to const tag if dst tag is const tag
    const_tags = ["weight", "bias", "fused_scale", "fused_bias", "min", "max", "threshold", "scale", "config"]
    src_tag = "odata"
    if dst_tid.tensor_type in const_tags:
      src_tag = dst_tid.tensor_type

    src_tid = self.get_tensor_id(arg, src_tag, src_composite)
    # TODO: add split idx for split op
    self.edges.add(TensorEdge(src_tid, dst_tid, split_idx))  # add edge to set

  def visit_call(self, call):
    if CompositePat.match(call):
      self.visit_composite_call(call)
    else:
      self.visit_regular_call(call)

  def visit_composite_call(self, call):
    self.composite_call = call
    self.stack.append(call)
    self.visit(call.op.body)
    self.composite_call = None
    self.stack.pop()
    for a in call.args:
      self.visit(a)

  def visit_regular_call(self, call):
    self.visit(call.op)
    for idx, a in enumerate(call.args):
      if hasattr(call.op, "arguments"): # this is tvm primitive operations.
        dst_tag = call.op.arguments[idx].name
        dst_tid = self.get_tensor_id(call, dst_tag, self.composite_call)
        self.add_edge(dst_tid, a)
      self.visit(a)

  def get_tensor_id(self, call, tag, composite=None):
    if composite:
      return TensorID((getNodeID(composite), getNodeID(call)), tag)
    else:
      return TensorID(getNodeID(call), tag)


class ImceCodeBlockBuilder(tvm.relay.ExprVisitor):
  """Visitor that generates IMCE code blocks from relay operations.

  This class uses a pluggable handler registry to process different operation types.
  New operations can be supported by creating handler classes and registering them
  with the @register_operation_handler decorator in imce_operation_handlers.py.

  Handlers receive a BuilderContext that wraps each call with helper methods.
  """

  def __init__(self, func_name, edges):
    super().__init__()
    # Shared state accessed by handlers through BuilderContext
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

  def initialize(self):
    # clear flag
    for inode in NodeID.inodes():
      block = ClearFlag("clear flag before policy update")
      self.codeblocks.append(inode, block, CodePhase.INIT)

    # policy update
    for inode in NodeID.inodes():
      block = PolicyUpdateBlock(inode, "policy update")
      self.codeblocks.append(inode, block, CodePhase.INIT)

    # standby and intrt
    inode_master = NodeID.inode_3
    inode_slaves = [node for node in NodeID.inodes() if node != inode_master]
    block = StandbyAndIntrtBlock(inode_slaves, "standby and intrt")
    self.codeblocks.append(inode_master, block, CodePhase.INIT)

    # set_flag
    block = SetFlagAndHaltBlock()
    for inode_slv in inode_slaves:
      self.codeblocks.append(inode_slv, block, CodePhase.INIT)
    
    # clear flag
    for inode in NodeID.inodes():
      block = ClearFlag("clear flag before imem write")
      self.codeblocks.append(inode, block, CodePhase.INIT)

    # imem write
    for imce, inst_edge in DevConfig().InstEdgeInfoDict.items():
      block = WriteIMEMBlock(inst_edge, f"imem write: {imce.name}")
      self.codeblocks.append(imce.master(), block, CodePhase.INIT)

    # imcu write
    for node in NodeID.inodes():
      block = WriteIMCUBlock(node, "imcu write", self.codeblocks.func_name)
      self.codeblocks.append(node, block, CodePhase.INIT)

    # imce compute
    active_imces = DevConfig().ActiveIMCEPerFunc[self.codeblocks.func_name]
    for imce in active_imces:
      block = IMCEComputeBlock(f"{imce.name} compute")
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

  def finalize(self):
    # standby and intrt
    # FIXME: hardcoded inode_3
    inode_master = NodeID.inode_3
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
      if isinstance(arg_id, Tuple):
        arg_id = arg_id[1]
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
    for edge in const_edges:
      self.add_send_block(edge)

    # send param edge interleaved
    #TODO: if edge count is more than one, interleave them
    for edge in param_edges:
      self.add_send_block(edge)

    # recv output edge interleaved
    #TODO: if edge count is more than one, interleave them
    for edge in output_edges:
      self.add_recv_block(edge)

  def add_send_block(self, edge):
    out_edge_info = DevConfig().get_tensor_edge_info(edge)
    tid = edge.src_id
    hid = self.get_hid(tid)
    # split op handling => pass down to inner edges from input var node
    if hid == None and CustomIDToNode()[edge.dst_id.graph_node_id].op.name == "split":
      inner_node = CustomIDToNode()[edge.dst_id.graph_node_id]
      for inner_edge in self.get_output_edges_from_id(getNodeID(inner_node)):
        self.add_send_block(inner_edge)
      return

    db = DevConfig().MemLayout.get_data_block_by_id(edge)
    assert db is not None, f"Data block not found for edge: {edge}"

    block = SendBlock(db, out_edge_info.fifo_id, f"send: {edge}")
    self.codeblocks.append(hid, block, CodePhase.EXEC)
  
  def add_send_block_interleaved(self, edge_list):
    pass

  def add_recv_block(self, edge):
    in_edge_info = DevConfig().get_tensor_edge_info(edge)
    in_tid = edge.dst_id
    hid = self.get_hid(in_tid)
    db = DevConfig().MemLayout.get_data_block_by_id(edge)

    block = RecvBlock(db, in_edge_info.fifo_id, f"recv: {in_tid}")
    self.codeblocks.append(hid, block, CodePhase.EXEC)
  
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
    return DevConfig().get_hw_node(tensor_id.graph_node_id)
