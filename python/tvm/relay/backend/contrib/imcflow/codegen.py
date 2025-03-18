import re
import tvm
import logging
from tvm import relay
from tvm.relay import op
from tvm.relay.frontend.common import infer_shape
from tvm.relay.dataflow_pattern import *
from tvm.contrib.imcflow import TensorID, TensorEdge
from tvm.relay.backend.contrib.imcflow import util
from tvm.relay.backend.contrib.imcflow.transform import getNodeID
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from tvm.relay.backend.contrib.imcflow.kernel_codegen import KernelCodegen
from tvm.relay.backend.contrib.imcflow.device_codegen import DeviceCodegen
from tvm.relay.backend.contrib.imcflow.codeblock import *
from tvm.relay.backend.contrib.imcflow.inode_codeblock import *
from tvm.relay.backend.contrib.imcflow.imce_codeblock import *
import pdb

logger = logging.getLogger("IMCFLOW")

CompositePat = wildcard().has_attr({"Composite": "imcflow.conv2d-with-postop"})(None)
TuplePat = is_tuple(None)
TupleGetItemPat = is_tuple_get_item(wildcard())
VarPat = is_var()

@util.create_imcflow_function_pass(opt_level=0)
class CodegenSuite:
  """A pass that generates/compiles code for IMCFlow functions"""

  def transform_function(self, _, func):
    func_name = func.attrs.global_symbol

    # annotate edges between (non-composite) calls,
    # while translating vars into corresponding calls
    annotator = InternalEdgeAnnotator()
    annotator.visit(func)

    # generate code blocks for each node
    builder = ImceCodeBlockBuilder(func_name, annotator.edges)
    builder.visit(func)
    DeviceCodegen("imce", output_dir="./").handle_code_generation(func_name, builder.codeblocks)

    builder = InodeCodeBlockBuilder(func_name, annotator.edges)
    builder.visit(func)
    DeviceCodegen("inode", output_dir="./").handle_code_generation(func_name, builder.codeblocks)

class InternalEdgeAnnotator(tvm.relay.ExprVisitor):
  def __init__(self):
    super().__init__()
    self.composite_call = None
    self.stack = []
    self.edges = []

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
    self.edges.append(TensorEdge(src_tid, dst_tid, split_idx))

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
    for idx, a in enumerate(call.args):
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
  def __init__(self, func_name, edges):
    super().__init__()
    self.curr_composite_id = None
    self.curr_conv_block = None
    self.last_tuple_idx = None
    # FIXME: stores the tuple index when taking a field from tuple.
    # We use it to determine the qreg_start_idx for MinMaxQuantBlock but may not work
    # when there is more tuples in a composite call.
    self.edges = edges
    self.codeblocks = CodeBlocks(func_name, "imce")

  def visit_tuple(self, tup):
    for idx, x in enumerate(tup.fields):
      self.last_tuple_idx = idx
      self.visit(x)

  def visit_call(self, call):
    for idx, a in enumerate(call.args):
      self.visit(a)

    IsComposite = isinstance(call.op, relay.Function) and \
        "Composite" in call.op.attrs

    if IsComposite:
      self.visit_composite_call(call)
    elif call.op == op.get("nn.imcflow_qconv"):
      self.visit_conv_call(call)
    elif call.op == op.get("add"):
      self.visit_add_call(call)
    elif call.op == op.get("divide"):
      self.visit_divide_call(call)
    elif call.op == op.get("concatenate"):
      self.visit_concat_call(call)
    elif call.op == op.get("split"):
      self.visit_split_call(call)
    elif call.op == op.get("nn.bias_add"):
      # self.visit_bias_add_call(call)
      pass
    elif call.op == op.get("imcflow.fused_batch_norm"):
      # self.visit_batch_norm_call(call)
      pass
    elif call.op == op.get("nn.relu"):
      # self.visit_relu_call(call)
      pass
    elif call.op == op.get("qnn.imcflow_min_max_quantize"):
      self.visit_min_max_quantize_call(call)
    elif call.op == op.get("qnn.imcflow_nu_quantize"):
      pass
    else:
      self.visit(call.op)

  def visit_conv_call(self, call):
    args = self.get_arg_dict(call)
    shapes = self.get_arg_shape_dict(call)
    shapes["output"] = infer_shape(call)

    in_edges = self.get_input_edges(call)
    out_edge = self.get_output_edge(call)

    for edge in in_edges:
      if edge.src_id.tensor_type == "weight":
        w_edge = edge
    w_info = DevConfig().get_tensor_edge_info(w_edge)
    w_tid = w_edge.src_id
    hid = self.get_hid(call)

    # scan reg
    # TODO: add scan reg code block

    # config reg
    # TODO: add config reg code block

    # write weights using recv
    size = DevConfig().MemLayout.get_data_block_by_id(w_tid).size
    # TODO: change to write weight block
    block = LoadLBBlock(size, 1, w_info.fifo_id, "weight write")
    self.codeblocks.append(hid, block, CodePhase.INIT)

    block = ConvBlock(in_edges, out_edge, shapes, call.attrs, "conv exec")
    # FIXME: this assumes that convblock is called first... we don't want that
    self.curr_conv_block = block
    self.codeblocks.append(hid, block, CodePhase.EXEC)

  def visit_add_call(self, call):
    assert self.curr_composite_id, "Add must be inside a composite function"
    hid = self.get_hid(call)

    in_edges = self.get_input_edges(call)
    out_edge = self.get_output_edge(call)
    block = AddBlock(in_edges, out_edge, "add")
    self.curr_conv_block.add_post_op(block)

  def visit_divide_call(self, call):
    # TODO: divide block should be replaced later
    assert self.curr_composite_id, "Divide must be inside a composite function"
    hid = self.get_hid(call)

    in_edges = self.get_input_edges(call)
    out_edge = self.get_output_edge(call)
    block = DivBlock(in_edges, out_edge, "div")
    self.curr_conv_block.add_post_op(block)

  def visit_concat_call(self, call):
    hid = self.get_hid(call)
    conv_block = self.get_conv_block_by_hid(hid)

    in_edges = self.get_input_edges(call)
    out_edge = self.get_output_edge(call)

    block = ConcatBlock(in_edges, out_edge, "concat")
    conv_block.add_post_op(block)

  def visit_split_call(self, call):
    hid = self.get_hid(call)
    if hid.is_imce():
      conv_block = self.get_conv_block_by_hid(hid)

      in_edge = self.get_input_edge(call)
      out_edges = self.get_output_edges(call)

      block = SplitBlock(in_edge, out_edges, "split")
      conv_block.add_post_op(block)

  def visit_min_max_quantize_call(self, call):
    assert self.curr_composite_id, "MinMaxQuantize must be inside a composite function"
    hid = self.get_hid(call)

    for tag in ("min", "max"):
      edge = self.get_tensor_edge_from_tag(call, tag)
      # TODO: inode code block needs to put appropriate address for min/max reg.
      # TODO: two ways to set min/max reg. RecvConst vs. ADDI
      block = RecvConstBlock(edge, f"{tag} write")
      self.codeblocks.append(hid, block, CodePhase.INIT)

    # TODO: add reset qreg code block
    # _edge = TensorEdge(TensorID(-1, "zero"), TensorID(getNodeID(call), "data"))
    # block = RecvConstBlock(_edge, f"qreg reset")
    # self.codeblocks.append(hid, block, CodePhase.INIT)

    in_edges = self.get_input_edges(call)
    out_edge = self.get_output_edge(call)
    # set o_split_idx to 0 when last_tupe_idx is None
    block = MinmaxQuantBlock(in_edges, out_edge, self.last_tuple_idx or 0, "min_max_quantize")
    self.curr_conv_block.add_post_op(block)

  def visit_bias_add_call(self, call):
    assert self.curr_composite_id, "BiasAdd must be inside a composite function"
    hid = self.get_hid(call)

    bias_edge = self.get_tensor_edge_from_tag(call, "bias")
    block = RecvConstBlock(bias_edge, "bias write")
    self.codeblocks.append(hid, block, CodePhase.INIT)

    in_edges = self.get_input_edges(call)
    out_edge = self.get_output_edge(call)
    block = AddBlock(in_edges, out_edge, "add_bias")
    self.curr_conv_block.add_post_op(block)

  def visit_batch_norm_call(self, call):
    assert self.curr_composite_id, "BatchNorm must be inside a composite function"
    hid = self.get_hid(call)
    scale_edge = self.get_tensor_edge_from_tag(call, "fused_scale")
    bias_edge = self.get_tensor_edge_from_tag(call, "fused_bias")

    block = RecvConstBlock(scale_edge, "fused_scale write")
    self.codeblocks.append(hid, block, CodePhase.INIT)
    block = RecvConstBlock(bias_edge, "fused_bias write")
    self.codeblocks.append(hid, block, CodePhase.INIT)

    in_edges = self.get_input_edges(call)
    out_edge = self.get_output_edge(call)
    block = VecBlock(in_edges, out_edge, "batch_norm_scale")
    self.curr_conv_block.add_post_op(block)
    block = AddBlock(in_edges, out_edge, "batch_norm_bias")
    self.curr_conv_block.add_post_op(block)

  def visit_relu_call(self, call):
    assert self.curr_composite_id, "Relu must be inside a composite function"
    hid = self.get_hid(call)
    args = self.get_arg_dict(call)
    shapes = self.get_arg_shape_dict(call)
    edge = self.get_tensor_edge_from_tag(call, "odata")

  def visit_composite_call(self, call):
    self.curr_composite_id = getNodeID(call)
    self.visit(call.op.body)
    self.curr_composite_id = None
    for idx, a in enumerate(call.args):
      self.visit(a)

  def get_hid(self, call):
    node_id = self.curr_composite_id or getNodeID(call)
    return DevConfig().get_hw_node(node_id)

  def get_graph_node_id(self, call):
    if self.curr_composite_id:
      return (self.curr_composite_id, getNodeID(call))
    else:
      return getNodeID(call)

  def get_tensor_id(self, call, tag):
    return TensorID(self.get_graph_node_id(call), tag)

  def get_conv_block_by_hid(self, hid):
    for block in self.codeblocks.blocks[hid][CodePhase.EXEC]:
      if isinstance(block, ConvBlock):
        return block

  def get_input_edge(self, call):
    edges = self.get_input_edges(call)
    assert len(edges) == 1, "Input edge must be unique"
    return edges[0]

  def get_input_edges(self, call):
    return [edge for edge in self.edges if edge.dst_inner_gid_match(getNodeID(call))]

  def get_output_edge(self, call):
    edges = self.get_output_edges(call)
    assert len(edges) == 1, "Output edge must be unique"
    return edges[0]

  def get_output_edges(self, call):
    return [edge for edge in self.edges if edge.src_inner_gid_match(getNodeID(call))]

  def get_tensor_edge_from_tag(self, call, tag):
    tid = self.get_tensor_id(call, tag)
    te = DevConfig().get_tensor_edge(tid)
    return te

  def inspect(self, tmp, graph_node_id):
    if graph_node_id in DevConfig().HWNodeMap.keys():
      hw_node_id = DevConfig().HWNodeMap[graph_node_id]
    else:
      hw_node_id = None
    tid = DevConfig().get_tensor_ids_from_graph_node_id(graph_node_id)
    print(f"{tmp.__class__} graph_node_id: {graph_node_id}, hw_node_id: {hw_node_id}, tid: {tid}")
    print("----------------------")

  def get_gid(self, call):
    if hasattr(call, "op") and isinstance(call.op, relay.Function) and "Composite" in call.op.attrs and re.match(r"imcflow\..*", call.op.attrs["Composite"]):
      gid = (getNodeID(call), getNodeID(call.op.body))
    else:
      gid = getNodeID(call)
    return gid

  def get_arg_keys(self, call):
    return [arg.name for arg in call.op.arguments]

  def get_arg_dict(self, call):
    args = {}
    for idx, arg in enumerate(call.op.arguments):
      args[arg.name] = call.args[idx]
    return args

  def get_arg_shape_dict(self, call):
    args_shape = {}
    for idx, arg in enumerate(call.op.arguments):
      args_shape[arg.name] = tuple(call.type_args[idx].shape)
    return args_shape

  # def get_const_node(self, call):
  #   for arg in call.args:
  #     if isinstance(arg, relay.Constant):
  #       return arg


class InodeCodeBlockBuilder(tvm.relay.ExprVisitor):
  def __init__(self, func_name, edges):
    super().__init__()
    self.edges = edges
    self.codeblocks = CodeBlocks(func_name, "inode")
    self.initialize()
    self.curr_composite_id = None

  def initialize(self):
    for inode in NodeID.inodes():
      # policy update
      block = PolicyUpdateBlock(inode, "policy update")
      self.codeblocks.append(inode, block, CodePhase.INIT)
      # imem write
      # imcu write

    pass

  def visit_call(self, call):
    for idx, a in enumerate(call.args):
      self.visit(a)

    IsComposite = isinstance(call.op, relay.Function) and \
        "Composite" in call.op.attrs
    IsInode =  call.op == op.get("imcflow_unpacking") or \
      call.op == op.get("imcflow_packing")

    if IsComposite:
      self.visit_composite_call(call)
    elif IsInode:
      IsSend = False
      IsRecv = False

      # check call is in inode
      if DevConfig().get_hw_node(self.get_graph_node_id(call)).is_inode():
        # Determine Recv or Send and add Recv Block, send Block
        if call.op == op.get("imcflow_unpacking"):
          IsSend = True
        elif call.op == op.get("imcflow_packing"):
          IsRecv = True
        # elif call.op == op.get("split"):
        #   IsSend = True
        else:
          raise ValueError("wrong operation!")

      if IsSend:
        self.visit_send_call(call)
      elif IsRecv:
        self.visit_recv_call(call)
      else:
        self.visit(call.op)
    else:
      pass

  def visit_composite_call(self, call):
    self.curr_composite_id = getNodeID(call)
    self.visit(call.op.body)
    self.curr_composite_id = None
    for idx, a in enumerate(call.args):
      self.visit(a)

  def visit_send_call(self, call):
    out_edge = self.get_output_edges(call)[0]
    out_edge_info = DevConfig().get_tensor_edge_info(out_edge)
    out_tid = out_edge.src_id
    hid = self.get_hid(call)
    block = DevConfig().MemLayout.get_data_block_by_id(out_tid)

    dst_hw_node = DevConfig().get_hw_node(out_edge.dst_id.graph_node_id)
    if dst_hw_node is not None and dst_hw_node.is_inode():
      # The only available case that unpacking's dst hw node is inode is [unpacking -> split].
      # [unpacking -> split -> qconv], then both unpacking and split are inode.
      # In this case, no tensor edge exists in [unpacking -> split], so handle this case separately.

      # TODO: Need to add another blocks(control block, etc)
      block = SendBlock(block, 0, "send idata") # FIFO ID for input of qconv is always 0. Refer to transform.py/PolicyTableGenerator.add_EdgeInfo
      self.codeblocks.append(hid, block, CodePhase.EXEC)
    else:
      # TODO: Need to add another blocks(control block, etc)
      block = SendBlock(block, out_edge_info.fifo_id, "send")
      self.codeblocks.append(hid, block, CodePhase.EXEC)

    return

  def visit_recv_call(self, call):
    in_edge = self.get_input_edges(call)[0]
    in_edge_info = DevConfig().get_tensor_edge_info(in_edge)
    in_tid = in_edge.src_id
    hid = self.get_hid(call)
    db = DevConfig().MemLayout.get_data_block_by_id(in_tid)
    if db is None:
      logger.warning("DataBlock for %s is empty", in_tid)
      return

    # TODO: Need to add another blocks(control block, etc)
    block = RecvBlock(db, in_edge_info.fifo_id, "recv")
    self.codeblocks.append(hid, block, CodePhase.EXEC)

    return

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
