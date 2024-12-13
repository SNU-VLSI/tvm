import re
import tvm
from tvm import relay
from tvm.relay import op
from tvm.relay.backend.contrib.imcflow import util
from tvm.relay.backend.contrib.imcflow.transform import getNodeID
from tvm.contrib.imcflow import ImcflowDeviceConfig
from tvm.relay.expr import (Call, TupleGetItem, Tuple)
from tvm.relay.backend.contrib.imcflow.kernel_codegen import KernelCodegen
from tvm.relay.backend.contrib.imcflow.device_codegen import DeviceCodegen
from tvm.relay.backend.contrib.imcflow.codeblock import *

@util.create_imcflow_function_pass(opt_level=0)
class CodegenSuite:
  """A pass that generates/compiles code for IMCFlow functions"""

  def transform_function(self, _, func):
    func_name = func.attrs.global_symbol
    _builder = ImceCodeBlockBuilder(func_name).visit(func)
    import pdb; pdb.set_trace()

    DeviceCodegen("imce").handle_code_generation(_builder.codeblocks)

    # _builder = InodeCodeBlockBuilder(func_name).visit(func)
    # DeviceCodegen("inode").handle_code_generation(_builder.codeblocks)


import pdb
class ImceCodeBlockBuilder(tvm.relay.ExprVisitor):
  def __init__(self, func_name):
    super().__init__()
    self.func_name = func_name
    self.codeblocks = []
    self._init_codeblock()

  def _init_codeblock(self):
    self.codeblocks.append(CodeBlockStart(self.func_name))
    self.codeblocks.append(CodeBlockEnd())

  def visit_call(self, call):
    if call.op == op.get("nn.conv2d"):
      tid = ImcflowDeviceConfig().get_tensor_ids_from_graph_node_id(self.get_gid(call))
      self.codeblocks.append(RecvCodeBlock())
      import pdb; pdb.set_trace()

    IsComposite = isinstance(call.op, relay.Function) and "Composite" in call.op.attrs
    if IsComposite:
      self.visit_composite(call)
    else:
      self.visit(call.op)

    for a in call.args:
        self.visit(a)

  def visit_composite(self, call):
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
    if isinstance(call.op, relay.Function) and "Composite" in call.op.attrs and re.match(r"imcflow\..*", call.op.attrs["Composite"]):
      gid = (getNodeID(call), getNodeID(call.op.body))
    else:
      gid = getNodeID(call)
    return gid