import re
import tvm
from tvm import relay
from tvm.relay import op
from tvm.relay.frontend.common import infer_shape
from tvm.relay.backend.contrib.imcflow import util
from tvm.relay.backend.contrib.imcflow.transform import getNodeID
from tvm.contrib.imcflow import ImcflowDeviceConfig
from tvm.relay.expr import (Call, TupleGetItem, Tuple)
from tvm.relay.backend.contrib.imcflow.kernel_codegen import KernelCodegen
from tvm.relay.backend.contrib.imcflow.device_codegen import DeviceCodegen
from tvm.relay.backend.contrib.imcflow.codeblock import *
import pdb


@util.create_imcflow_function_pass(opt_level=0)
class CodegenSuite:
  """A pass that generates/compiles code for IMCFlow functions"""

  def transform_function(self, _, func):
    func_name = func.attrs.global_symbol
    _builder = ImceCodeBlockBuilder(func_name).visit(func)
    DeviceCodegen("imce").handle_code_generation(_builder.codeblocks)

    # _builder = InodeCodeBlockBuilder(func_name).visit(func)
    # DeviceCodegen("inode").handle_code_generation(_builder.codeblocks)


class ImceCodeBlockBuilder(tvm.relay.ExprVisitor):
  def __init__(self, func_name):
    super().__init__()
    self.func_name = func_name
    self.codeblocks = [
        CodeBlockStart(self.func_name, "imce"),
        CodeBlockEnd()
    ]

  def visit_call(self, call):
    IsComposite = isinstance(
        call.op, relay.Function) and "Composite" in call.op.attrs
    IsConv2d = call.op == op.get("nn.conv2d")

    if IsComposite:
      self.visit_composite_call(call)
    elif IsConv2d:
      self.visit_conv_call(call)
    else:
      self.visit(call.op)

    for a in call.args:
      self.visit(a)

  def visit_conv_call(self, call):
    gid = self.get_gid(call)
    hid = ImcflowDeviceConfig().get_hw_node(gid)
    pdb.set_trace()

    args = self.get_arg_dict(call)
    shapes = self.get_arg_shape_dict(call)
    shapes["output"] = infer_shape(call)

    # scan reg
    # TODO: add scan reg code block

    # config reg
    # TODO: add config reg code block

    # write weights using recv
    block = SimpleRecvCodeBlock(hid, "weight write")
    weight_tid = TensorID(getNodeID(args["weight"]), "weight")
    weight_size = ImcflowDeviceConfig().MemLayout.get_data_block_by_id(
        weight_tid).size  # TODO: this is rather long
    block.set_info(weight_size, fifo_id=-1)
    self.codeblocks.append(block)
    pdb.set_trace()

    # load input
    block = ConvCodeBlock(hid, "input load")
    block.set_info(shapes, call.attrs, fifo_id=-1)

  def visit_composite_call(self, call):
    # skip composite for now (the TensorID is not available yet)
    pass

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
    if graph_node_id in ImcflowDeviceConfig().HWNodeMap.keys():
      hw_node_id = ImcflowDeviceConfig().HWNodeMap[graph_node_id]
    else:
      hw_node_id = None
    tid = ImcflowDeviceConfig().get_tensor_ids_from_graph_node_id(graph_node_id)
    print(f"{tmp.__class__} graph_node_id: {graph_node_id}, hw_node_id: {hw_node_id}, tid: {tid}")
    print("----------------------")

  def get_gid(self, call):
    if hasattr(call, "op") and isinstance(call.op, relay.Function) and "Composite" in call.op.attrs and re.match(r"imcflow\..*", call.op.attrs["Composite"]):
      gid = (getNodeID(call), getNodeID(call.op.body))
    else:
      gid = getNodeID(call)
    return gid

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
