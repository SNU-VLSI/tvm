import re
import tvm
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


CompositePat = wildcard().has_attr({"Composite": "imcflow.conv2d-with-postop"})(None)
TuplePat = is_tuple(None)
TupleGetItemPat = is_tuple_get_item(wildcard())
VarPat = is_var()

@util.create_imcflow_function_pass(opt_level=0)
class CodegenSuite:
  """A pass that generates/compiles code for IMCFlow functions"""

  def transform_function(self, _, func):
    func_name = func.attrs.global_symbol
    annotator = InternalEdgeAnnotator()
    annotator.visit(func)
    # builder = ImceCodeBlockBuilder(func_name, annotator.edges)
    # builder.visit(func)
    pdb.set_trace()
    DeviceCodegen("imce", output_dir="./").handle_code_generation(func_name, builder.codeblocks)

    # builder = InodeCodeBlockBuilder(func_name).visit(func)
    # DeviceCodegen("inode").handle_code_generation(builder.codeblocks)

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
    src_tag = dst_tid.tensor_type if isinstance(arg, relay.Constant) else "odata"
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
    self.visit(call.op)
    for a in call.args:
      self.visit(a)
    self.composite_call = None
    self.stack.pop()

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
    elif call.op == op.get("concatenate"):
      self.visit_concat_call(call)
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

    w_tid = self.get_tensor_id(args["weight"], "weight")
    hid = self.get_hid(call)

    # scan reg
    # TODO: add scan reg code block

    # config reg
    # TODO: add config reg code block

    # write weights using recv
    size = DevConfig().MemLayout.get_data_block_by_id(w_tid).size
    # TODO: change to write weight block
    block = LoadLBBlock(size, 1, -1, "weight write")
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

  def visit_concat_call(self, call):
    assert self.curr_composite_id, "Concat must be inside a composite function"
    hid = self.get_hid(call)
    conv_block = self.get_conv_block_by_hid(hid)

    in_edges = self.get_input_edges(call)
    out_edge = self.get_output_edge(call)
    block = ConcatBlock(in_edges, out_edge, "concat")
    conv_block.add_post_op(block)

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

  def visit_min_max_quantize_call(self, call):
    assert self.curr_composite_id, "MinMaxQuantize must be inside a composite function"
    hid = self.get_hid(call)

    for tag in ("min", "max"):
      edge = self.get_tensor_edge_from_tag(call, tag)
      # TODO: inode code block needs to put appropriate address for min/max reg.
      # TODO: two ways to set min/max reg. RecvConst vs. ADDI
      block = RecvConstBlock(edge, f"{tag} write")
      self.codeblocks.append(hid, block, CodePhase.INIT)

    # set the qreg mask
    block = SetQregMaskBlock()
    self.codeblocks.append(hid, block, CodePhase.INIT)

    in_edges = self.get_input_edges(call)
    out_edge = self.get_output_edge(call)
    block = MinmaxQuantBlock(in_edges, out_edge, self.last_tuple_idx, "min_max_quantize")
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
    for idx, a in enumerate(call.args):
      self.visit(a)
    self.curr_composite_id = None

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

  def get_input_edge(self, call, tag):
    for edge in self.edges:
      if edge.dst_inner_gid_match(getNodeID(call)) and edge.dst_id.tensor_type == tag:
        return edge

  def get_input_edges(self, call):
    in_edges = []
    for edge in self.edges:
      if edge.dst_inner_gid_match(getNodeID(call)):
        in_edges.append(edge)

    return in_edges

  def get_output_edge(self, call):
    for edge in self.edges:
      if edge.src_inner_gid_match(getNodeID(call)):
        return edge

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
