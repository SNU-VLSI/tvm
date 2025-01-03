import re
import tvm
from tvm import relay
from tvm.relay import op
from tvm.relay.frontend.common import infer_shape
from tvm.contrib.imcflow import TensorID
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
    self.post_process = []
    self.codeblocks = CodeBlocks(func_name, "imce")

  def visit_call(self, call):
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

    for a in call.args:
      self.visit(a)

  def visit_conv_call(self, call):
    args = self.get_arg_dict(call)
    shapes = self.get_arg_shape_dict(call)
    shapes["output"] = infer_shape(call)

    if self.curr_composite_id:
      hid = DevConfig().get_hw_node(self.curr_composite_id)
      w_tid = TensorID(
          (self.curr_composite_id, getNodeID(args["weight"])), "weight")
    else:
      hid = DevConfig().get_hw_node(getNodeID(call))
      w_tid = TensorID(getNodeID(args["weight"]), "weight")

    # scan reg
    # TODO: add scan reg code block

    # config reg
    # TODO: add config reg code block

    # write weights using recv
    size = DevConfig().MemLayout.get_data_block_by_id(
        w_tid).size  # TODO: this is rather long
    block = LoadLBBlock(size, 1, -1, hid, "weight write") # TODO: change to write weight block
    self.codeblocks.append(block)

    # load input
    block = ConvBlock(shapes, call.attrs, -1, -1, hid, "input load")
    if self.curr_composite_id:
      block.add_op(self.post_process)
    self.codeblocks.append(block)

  def visit_add_call(self, call):
    args = self.get_arg_dict(call)
    shapes = self.get_arg_shape_dict(call)
    for arg in call.args:
      if isinstance(arg, relay.Var) or isinstance(arg, relay.Constant):
        pass

    block = AddBlock("add")
    self.post_process.append(block)
    pdb.set_trace()

  def visit_bias_add_call(self, call):
    args = self.get_arg_dict(call)
    shapes = self.get_arg_shape_dict(call)
    pdb.set_trace()

  def visit_batch_norm_call(self, call):
    args = self.get_arg_dict(call)
    shapes = self.get_arg_shape_dict(call)
    pdb.set_trace()

  def visit_relu_call(self, call):
    args = self.get_arg_dict(call)
    shapes = self.get_arg_shape_dict(call)
    pdb.set_trace()


  def visit_composite_call(self, call):
    self.curr_composite_id = getNodeID(call)
    super().visit(call.op.body)
    self.curr_composite_id = None

  def visit_op(self, op):
    self.inspect(op, getNodeID(op))
    super().visit_op(op)

  def visit_var(self, var):
    self.inspect(var, getNodeID(var))
    super().visit_var(var)

  def visit_constant(self, const):
    self.inspect(const, getNodeID(const))
    super().visit_constant(const)

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
