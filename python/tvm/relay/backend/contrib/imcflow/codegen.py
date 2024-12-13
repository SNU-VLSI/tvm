import tvm
from tvm import relay
from tvm.relay.backend.contrib.imcflow import util
from tvm.contrib.imcflow import ImcflowDeviceConfig
from tvm.relay.expr import (Call, TupleGetItem, Tuple)
from tvm.relay.backend.contrib.imcflow.kernel_codegen import KernelCodegen
from tvm.relay.backend.contrib.imcflow.device_codegen import DeviceCodegen

@util.create_imcflow_function_pass(opt_level=0)
class CodegenSuite:
  """A pass that generates/compiles code for IMCFlow functions"""

  def __init__(self, MappingDict_2D, PolicyTable_2D):
    self.MappingDict_2D = MappingDict_2D
    self.PolicyTable_2D = PolicyTable_2D

  def transform_function(self, _, func):
    codeblock_builder = CodeBlockBuilder(func)
    kernel_codegen = KernelCodegen()
    kernel_codegen.visit(func)
    device_codegen = DeviceCodegen(output_dir="./output")
    device_codegen.handle_code_generation("example_op", [])

class CodeBlockBuilder(tvm.relay.ExprVisitor):
  def __init__(self, func):
    self.func = func
    self.codeblocks = []
    self.current_block = None
    self.current_block_id = 0
    self.current_block_name = None

  def visit_call(self, call):
    if isinstance(call.op, tvm.relay.op.Op):
      if call.op.name == "example_op":
        self.current_block = CodeBlock(self.current_block_id, "example_op")
        self.current_block_id += 1
        self.current_block_name = "example_op"
        self.codeblocks.append(self.current_block)
    self.generic_visit(call)