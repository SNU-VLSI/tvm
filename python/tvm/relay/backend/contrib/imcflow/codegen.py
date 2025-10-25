import re
import os
import tvm
from tvm import relay
from tvm.relay import op
from tvm.relay.frontend.common import infer_shape
from tvm.relay.dataflow_pattern import *
from tvm.contrib.imcflow import TensorID, TensorEdge
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

CompositePat = wildcard().has_attr({"Composite": "imcflow.conv2d-with-postop"})(None)
TuplePat = is_tuple(None)
TupleGetItemPat = is_tuple_get_item(wildcard())
VarPat = is_var()

@util.create_imcflow_function_pass(opt_level=0)
class CodegenSuite:
  """A pass that generates/compiles code for IMCFlow functions"""

  def __init__(self, build_dir):
    self.build_dir = build_dir
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
    DeviceCodegen("imce", self.build_dir).handle_code_generation(func_name, builder.codeblocks)

    builder = InodeCodeBlockBuilder(func_name, annotator.edges)
    builder.visit(func)
    DeviceCodegen("inode", self.build_dir).handle_code_generation(func_name, builder.codeblocks)

    PolicyTableCodegen(func_name, self.build_dir).generate(func_name)

    return func

class PolicyTableCodegen:
  """
  Write out a binary file for policy tables for each node.
  """
  def __init__(self, func_name, build_dir="/tmp"):
    super().__init__()
    self.func_name = func_name
    self.build_dir = build_dir
    self.func_dir = os.path.join(build_dir, func_name)

  def pack_to_bin(self, entry, endian):
    assert set(entry.keys()) == {'Local', 'North', 'East', 'South', 'West'}, "Invalid policy table entry"

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
      policytable_path = os.path.join(self.func_dir, f"{node_name.name}_policy")
      policytable_bin_file = f"{policytable_path}.bin"
      policytable_host_obj_file = f"{node_name.name}_policy.host.o"
      with open(policytable_bin_file, "wb") as file:
        for entry in entries:
          policytable_bin = self.pack_to_bin(entry, endian='little')
          file.write(policytable_bin)
      if ("inode" in node_name.name):
        DevCodegen = DeviceCodegen("inode", self.build_dir)
        DevCodegen.func_dir = self.func_dir
        DevCodegen.create_host_object(f"{node_name.name}_policy.bin", policytable_host_obj_file)
      if ("imce" in node_name.name):
        DevCodegen = DeviceCodegen("inode", self.build_dir)
        DevCodegen.func_dir = self.func_dir
        DevCodegen.create_host_object(f"{node_name.name}_policy.bin", policytable_host_obj_file)
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
    const_tags = ["weight", "bias", "fused_scale", "fused_bias", "min", "max", "threshold", "scale"]
    src_tag = "odata"
    if dst_tid.tensor_type in const_tags:
      src_tag = dst_tid.tensor_type

    src_tid = self.get_tensor_id(arg, src_tag, src_composite)
    # TODO: add split idx for split op
    self.edges.add(TensorEdge(src_tid, dst_tid, split_idx)) # add edge to set

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
      if hasattr(call.op, "arguments"):
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
    self.initialize()
    self.curr_composite_id = None
    self.finalize()

  def initialize(self):
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

    # imem write
    for imce, inst_edge in DevConfig().InstEdgeInfoDict.items():
      block = WriteIMEMBlock(inst_edge, f"imem write: {imce.name}")
      self.codeblocks.append(imce.master(), block, CodePhase.INIT)

    # imcu write
    for node in NodeID.inodes():
      block = WriteIMCUBlock(node, "imcu write")
      self.codeblocks.append(node, block, CodePhase.INIT)

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

  def visit_call(self, call):
    IsComposite = isinstance(call.op, relay.Function) and \
        "Composite" in call.op.attrs
    for idx, a in enumerate(call.args):
      self.visit(a)
    if IsComposite:
      self.visit_composite_call(call)
    else:
      pass

  def visit_function(self, fn):
    for x in fn.params:
      self.visit(x)
    self.visit(fn.body)

    # Add Recv Block
    self.add_recv_block(fn)

  def add_send_block(self, node):
    out_edge = self.get_output_edges(node)[0]
    out_edge_info = DevConfig().get_tensor_edge_info(out_edge)
    tid = out_edge.src_id
    hid = self.get_hid(node)

    block = IMCEComputeBlock(f"imce compute start")
    self.codeblocks.append(hid, block, CodePhase.EXEC)

    db = DevConfig().MemLayout.get_data_block_by_id(tid)

    block = SendBlock(db, out_edge_info.fifo_id, "send")
    self.codeblocks.append(hid, block, CodePhase.EXEC)

  def add_recv_block(self, node):
    in_edge = self.get_input_edges(node)[0]
    in_edge_info = DevConfig().get_tensor_edge_info(in_edge)
    in_tid = in_edge.dst_id
    hid = self.get_hid(node)
    db = DevConfig().MemLayout.get_data_block_by_id(in_tid)

    block = RecvBlock(db, in_edge_info.fifo_id, "recv")
    self.codeblocks.append(hid, block, CodePhase.EXEC)

  def visit_var(self, var):
    self.add_send_block(var)

  def visit_constant(self, const):
    self.add_send_block(const)

  def visit_composite_call(self, call):
    self.curr_composite_id = getNodeID(call)
    self.visit(call.op.body)
    self.curr_composite_id = None
    for idx, a in enumerate(call.args):
      self.visit(a)

  def get_graph_node_id(self, call):
    if self.curr_composite_id:
      return (self.curr_composite_id, getNodeID(call))
    else:
      return getNodeID(call)

  def get_input_edges(self, call):
    return [edge for edge in self.edges if edge.dst_inner_gid_match(getNodeID(call))]

  def get_output_edges(self, call):
    return [edge for edge in self.edges if edge.src_inner_gid_match(getNodeID(call))]

  def get_hid(self, call):
    node_id = self.get_graph_node_id(call)
    return DevConfig().get_hw_node(node_id)
