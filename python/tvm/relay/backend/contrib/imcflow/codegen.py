import re
import tvm
from tvm import relay
from tvm.relay import op
from tvm.relay.frontend.common import infer_shape
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


@util.create_imcflow_function_pass(opt_level=0)
class CodegenSuite:
  """A pass that generates/compiles code for IMCFlow functions"""

  def transform_function(self, _, func):
    func_name = func.attrs.global_symbol
    builder = ImceCodeBlockBuilder(func_name)
    builder.visit(func)
    DeviceCodegen("imce").handle_code_generation(func_name, builder.codeblocks)

    # builder = InodeCodeBlockBuilder(func_name).visit(func)
    # DeviceCodegen("inode").handle_code_generation(builder.codeblocks)


class ImceCodeBlockBuilder(tvm.relay.ExprVisitor):
  def __init__(self, func_name):
    super().__init__()
    self.curr_composite_id = None
    self.edges = []
    self.post_process = []
    self.codeblocks = CodeBlocks(func_name, "imce")

  def add_edges(self, call, arg, idx):
    if not self.curr_composite_id:
      return
    if isinstance(arg, relay.Tuple):
      for a in arg.fields:
        self.add_edges(call, a, idx)
      return
    elif isinstance(arg, relay.TupleGetItem):
      self.add_edges(call, arg.tuple_value, idx)
      return
    elif isinstance(call.op, relay.Function):
      tag = call.op.params[idx].name_hint
    else:
      tag = call.op.arguments[idx].name
    src_tid = TensorID((self.curr_composite_id, getNodeID(arg)), "odata")
    dst_tid = TensorID((self.curr_composite_id, getNodeID(call)), tag)
    self.edges.append(TensorEdge(src_tid, dst_tid))

  def visit_call(self, call):
    for idx, a in enumerate(call.args):
      self.add_edges(call, a, idx)
      self.visit(a)

    IsComposite = isinstance(call.op, relay.Function) and \
        "Composite" in call.op.attrs

    if IsComposite:
      self.visit_composite_call(call)
    elif call.op == op.get("nn.conv2d"):
      self.visit_conv_call(call)
    elif call.op == op.get("add"):
      self.visit_add_call(call)
    elif call.op == op.get("nn.bias_add"):
      self.visit_bias_add_call(call)
    elif call.op == op.get("nn.batch_norm"):
      self.visit_batch_norm_call(call)
    elif call.op == op.get("nn.relu"):
      self.visit_relu_call(call)
    else:
      self.visit(call.op)

  def visit_conv_call(self, call):
    # pdb.set_trace()
    args = self.get_arg_dict(call)
    shapes = self.get_arg_shape_dict(call)
    shapes["output"] = infer_shape(call)

    w_tid = self.get_tensor_id(args["weight"], "weight")
    node_id = self.curr_composite_id or getNodeID(call)
    hid = DevConfig().get_hw_node(node_id)

    # scan reg
    # TODO: add scan reg code block

    # config reg
    # TODO: add config reg code block

    # write weights using recv
    size = DevConfig().MemLayout.get_data_block_by_id(w_tid).size
    # TODO: change to write weight block
    block = LoadLBBlock(size, 1, -1, "weight write")
    self.codeblocks.append(hid, block, CodePhase.INIT)

    block = ConvBlock(shapes, call.attrs, -1, -1, "conv exec")
    if self.curr_composite_id:
      block.add_op(self.post_process)
    self.codeblocks.append(hid, block, CodePhase.EXEC)

  def visit_add_call(self, call):
    assert self.curr_composite_id, "Add must be inside a composite function"
    hid = DevConfig().get_hw_node(self.curr_composite_id)

    in_edges = self.get_input_edges(call)
    out_edge = self.get_output_edge(call)
    block = AddBlock(in_edges, out_edge, "add")
    self.codeblocks.append(hid, block, CodePhase.EXEC)

  def visit_bias_add_call(self, call):
    assert self.curr_composite_id, "BiasAdd must be inside a composite function"
    args = self.get_arg_dict(call)
    shapes = self.get_arg_shape_dict(call)
    info = self.get_tensor_edge_info_from_tag(call, "bias")

  def visit_batch_norm_call(self, call):
    assert self.curr_composite_id, "BatchNorm must be inside a composite function"
    # pdb.set_trace()
    args = self.get_arg_dict(call)
    shapes = self.get_arg_shape_dict(call)
    scale_info = self.get_tensor_edge_info_from_tag(call, "scale")
    bias_info = self.get_tensor_edge_info_from_tag(call, "bias")

  def visit_relu_call(self, call):
    assert self.curr_composite_id, "Relu must be inside a composite function"
    args = self.get_arg_dict(call)
    shapes = self.get_arg_shape_dict(call)
    info = self.get_tensor_edge_info_from_tag(call, "odata")
    # block = ReLUBlock("relu")
    # self.post_process.append(block)

  def visit_composite_call(self, call):
    self.curr_composite_id = getNodeID(call)

    for idx, a in enumerate(call.args):
      self.visit(a)

    super().visit(call.op.body)
    self.curr_composite_id = None
    self.edges = []

  def get_tensor_id(self, call, tag):
    if self.curr_composite_id:
      tid = TensorID((self.curr_composite_id, getNodeID(call)), tag)
    else:
      tid = TensorID(getNodeID(call), tag)
    return tid

  def get_input_edges(self, call):
    in_edges = []
    for edge in self.edges:
      # TODO: hard coded to check for only the internal node id
      if edge.dst_id.graph_node_id[1] == getNodeID(call):
        in_edges.append(edge)
    return in_edges

  def get_output_edge(self, call):
    for edge in self.edges:
      # TODO: hard coded to check for only the internal node id
      if edge.src_id.graph_node_id[1] == getNodeID(call):
        return edge

  def get_tensor_edge_info_from_tag(self, call, tag):
    tid = self.get_tensor_id(call, tag)
    te = DevConfig().get_tensor_edge(tid)
    return DevConfig().get_tensor_edge_info(te)

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
