from abc import *
from typing import *
from copy import copy
import math
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from tvm.contrib.imcflow import NodeID, TensorID, TensorEdge
from tvm.relay.op.op_attrs import Conv2DAttrs
from tvm.relay.backend.contrib.imcflow.conv_util import ConvUtil
from tvm.relay.backend.contrib.imcflow.codeblock import *
from tvm.relay.dataflow_pattern import *
from textwrap import indent
import logging
import pdb

if TYPE_CHECKING:
  from .builder_context import BuilderContext

ConstPat = is_constant()


class ImceCodeBlock(CodeBlock):
  def __init__(self, annotation: str = ""):
    super().__init__()
    self.annotation = annotation

  def content(self) -> CodeBlock:
    if self.annotation:
      code = TextBlock("")
      code += f"// generate: {self.annotation}"
      code += copy(self._content())
      code += f"// endgenerate: {self.annotation}"
      return code
    else:
      return self._content()

  @abstractmethod
  def _content(self) -> CodeBlock:
    pass


class ImceCallCodeBlock(ImceCodeBlock):
  num_in_edges = None

  def __init__(self, call: 'BuilderContext', annotation: str = ""):
    super().__init__(annotation)
    self.call = call
    self.in_edges = call.get_input_edges()
    self.out_edges = call.get_output_edges()
    self.prev_op = None
    if self.num_in_edges is not None:
      assert len(self.in_edges) == self.num_in_edges


class LoadLBBlock(ImceCodeBlock):
  """ Code block for receiving data from given fifo id to the line buffer """

  def __init__(self, count: int, repeat: int, fifo_id: int, annotation: str = ""):
    super().__init__(annotation)
    self.count = count
    self.repeat = repeat
    self.fifo_id = fifo_id

  def _content(self) -> CodeBlock:
    code = TextBlock("")
    for _ in range(self.repeat):
      code += f"__builtin_IMCE_LOAD_LB({self.fifo_id});"
    return SimpleFor(self.count, code, "load_block")


class RecvConstBlock(ImceCodeBlock):
  """ Code block for receiving constant from given fifo id into a variable """
  # FIXME: Add support for initializing QREGs to zero
  num_in_edges = 1

  def __init__(self, in_edge: TensorEdge, annotation: str = ""):
    super().__init__(annotation)
    self.in_edge = in_edge

  def _content(self) -> CodeBlock:
    code = TextBlock("")
    te_info = DevConfig().get_tensor_edge_info_with_id_dir(
        self.in_edge.dst_id, "in")  # a hack to get the tensor edge info
    assert te_info, "Tensor edge info not found"

    size = DevConfig().MemLayout.get_data_block_by_id(self.in_edge.src_id).size
    base_addr = DevConfig().MemLayout.get_data_block_by_id(
        self.in_edge.src_id).base_address
    assert base_addr % 32 == 0, "Base address must be a multiple of 32"
    recv_count = math.ceil(size / 32.0)  # recv operates on 32-byte word

    for i in range(recv_count):
      var = UniqueVar((self.in_edge, i))
      var.set_static()
      if self.annotation == "min write":
        code += f"{var} = __builtin_IMCE_RECV_MIN({te_info.fifo_id});"
      elif self.annotation == "max write":
        code += f"{var} = __builtin_IMCE_RECV_MAX({te_info.fifo_id});"
      elif self.annotation == "config write":
        code += f"{var} = __builtin_IMCE_RECV_CFG({te_info.fifo_id});"
      elif self.annotation == "scan write":
        code += f"{var} = __builtin_IMCE_RECV_SREG{i}({te_info.fifo_id});"
        code += f"{var} = __builtin_IMCE_SCAN_RW({var});"
      else:
        code += f"{var} = __builtin_IMCE_RECV({te_info.fifo_id});"
    return code


class VecBlock(ImceCallCodeBlock):
  """
  VecBlock is base class for implementing R,I type vector operations.
  Only generates computation. RECV/SEND handled by wrapper or ConvBlock.
  """
  num_in_edges = 2

  def __init__(self, call: 'BuilderContext', annotation: str = ""):
    """ Code block for vector operations """
    super().__init__(call, annotation)
    self.op_name = self._op_name()
    self.imm_value = self._get_imm_value()

  @abstractmethod
  def _get_imm_value(self) -> int:
    pass

  @abstractmethod
  def _op_name(self) -> str:
    pass

  def _content(self) -> CodeBlock:
    """Generate only computation, no RECV/SEND."""
    code = TextBlock("")
    num_blocks = 4 if self.prev_op else 1

    for i in range(num_blocks):
      # put a tuple of (tensor edge, block index) as the key, giving a unique variable name
      var_ins = []
      for edge in self.in_edges:
        if self.prev_op and (edge in self.prev_op.out_edges):
          # replace the var with var_o of the prev_block if matches
          var_ins.append(UniqueVar((self.prev_op, i)))
        else:
          # else create a new variable using edge
          var_ins.append(UniqueVar((edge, i)))

      var_o = UniqueVar((self, i))
      var_in_str = ", ".join([f"{var_i}" for var_i in var_ins])
      # e.g. __builtin_IMCE_ADD(a, b, 15);
      code += f"{var_o} = __builtin_IMCE_{self.op_name}({var_in_str}, {self.imm_value});"

    return code


class AddBlock(VecBlock):
  def _get_imm_value(self) -> int:
    return 15  # src_mask

  def _op_name(self) -> str:
    return "ADD"


class DivBlock(VecBlock):
  def _get_imm_value(self) -> int:
    return 15  # src_mask

  def _op_name(self) -> str:
    return "DIV"


class MultlBlock(VecBlock):
  def _get_imm_value(self) -> int:
    return 15  # src_mask

  def _op_name(self) -> str:
    return "MULTL"


class MulthBlock(VecBlock):
  def _get_imm_value(self) -> int:
    return 15  # src_mask

  def _op_name(self) -> str:
    return "MULTH"


class ReLUBlock(VecBlock):
  num_in_edges = 1

  def _get_imm_value(self) -> int:
    return 0  # immediate value for MAXI

  def _op_name(self) -> str:
    return "MAXI"


class MinmaxQuantBlock(ImceCallCodeBlock):
  """
  MinmaxQuantBlock for min/max quantization operations.
  Only generates computation. RECV/SEND handled by wrapper or ConvBlock.
  """
  num_in_edges = 3

  def __init__(self, call: 'BuilderContext', o_split_idx: int, annotation: str = ""):
    """ Code block for min/max quantization """
    super().__init__(call, annotation)
    self.o_split_idx = o_split_idx

  def _content(self) -> CodeBlock:
    """Generate only computation, no RECV/SEND."""
    num_blocks = 4 if self.prev_op else 1
    src_mask = 15

    data_edge = None
    for edge in self.in_edges:
      if self.prev_op and (edge in self.prev_op.out_edges):
        data_edge = self.prev_op
      elif edge.dst_id.tensor_type == "data":
        data_edge = edge

    code = TextBlock("")

    for i in range(num_blocks):
      var_i = UniqueVar((data_edge, i))

      qreg_start_idx = i + 4 * self.o_split_idx
      # min max quantization does not require $rs2
      code += f"__builtin_IMCE_MM_QUANT({var_i}, 0, {src_mask}, {qreg_start_idx});"

      # Get QREG result for this block
      var_o = UniqueVar((self, i))
      code += f"{var_o} = __builtin_IMCE_GET_QREG({i});"

    return code


class ConcatBlock(ImceCallCodeBlock):
  min_in_edges = 2

  """
  Code block for concatenating multiple tensors
  FIXME: needs to look upon, since concat can happen not only in bitplanes...
  """

  def __init__(self, call: 'BuilderContext', annotation: str = ""):
    """ Code block for min/max quantization """
    super().__init__(call, annotation)
    assert len(
        self.in_edges) >= self.min_in_edges, "At least two input edges are required"

  def _content(self) -> CodeBlock:
    num_bitplanes = 4
    src_mask = 15

    code = TextBlock("")

    external_in_edges = [
        e for e in self.in_edges if e in DevConfig().TensorEdgetoInfo]
    internal_in_edge = (set(self.in_edges) - set(external_in_edges)).pop()

    for i in range(num_bitplanes):
      var_i = UniqueVar((internal_in_edge, i))
      var_o = UniqueVar((self, i))
      for ext_edge in external_in_edges:
        var_e = UniqueVar((ext_edge, i))
        fifo_id = DevConfig().get_tensor_edge_info(ext_edge).fifo_id

        code += f"{var_e} = __builtin_IMCE_RECV({fifo_id});"
        code += f"{var_o} = __builtin_IMCE_OR({var_i}, {var_e}, {src_mask});"

    return code


class SplitBlock(ImceCallCodeBlock):
  num_in_edges = 1

  """ Code block for splitting a tensor into multiple tensors """

  def __init__(self, call: 'BuilderContext', annotation: str = ""):
    super().__init__(call, annotation)
    first_policies = [DevConfig().get_tensor_edge_info(
        out_edge).policy_info[0] for out_edge in self.out_edges]
    fifo_ids = [DevConfig().get_tensor_edge_info(
        out_edge).fifo_id for out_edge in self.out_edges]
    assert all(policy == first_policies[0]
               for policy in first_policies), "All output edges must have the same first policy info"
    assert all(fid == fifo_ids[0]
               for fid in fifo_ids), "All output edges must have the same fifo id"

  def _content(self) -> CodeBlock:
    return TextBlock("")


class ConvBlock(ImceCallCodeBlock):
  """ Code block for receiving conv input data from given fifo id """
  num_in_edges = 3

  def __init__(self, call: 'BuilderContext', shapes: dict, conv_attrs: Conv2DAttrs,
               annotation: str = ""):
    super().__init__(call, annotation)
    self.conv = ConvUtil(shapes["data"][2], shapes["data"][3],
                         conv_attrs.padding[0], conv_attrs.strides[0],
                         conv_attrs.kernel_size[0], conv_attrs.kernel_size[1])
    self.post_op_chain = []

  def add_post_op(self, code: ImceCallCodeBlock):
    if self.post_op_chain:
      code.prev_op = self.post_op_chain[-1]
    else:
      code.prev_op = self
    self.post_op_chain.append(code)

  def _loop_body_content(self, recv_count: int) -> CodeBlock:
    num_blocks = 4  # FIXED in ConvBlock
    for edge in self.in_edges:
      if edge.dst_id.tensor_type == "data":
        data_edge = edge

    fifo_id_i = DevConfig().get_tensor_edge_info_with_id_dir(
        data_edge.dst_id, "in").fifo_id
    if fifo_id_i != 0:
      logging.warning(f"conv block data fifo_id_i is not 0, but {fifo_id_i}")

    last_out_edges = self.post_op_chain[-1].out_edges if self.post_op_chain else self.out_edges
    out_edge_infos = [DevConfig().get_tensor_edge_info_with_id_dir(edge.src_id, "out") for edge in last_out_edges]

    code = TextBlock("")
    code += LoadLBBlock(recv_count, num_blocks, fifo_id_i)
    code += "__builtin_IMCE_STEP();\n"

    for i in range(num_blocks):
      var_creg = UniqueVar((self, i))
      code += f"{var_creg} = __builtin_IMCE_GET_CREG((short){i});"

    for op in self.post_op_chain:
      code += "\n"
      code += copy(op)

    code += "\n"
    for i in range(num_blocks):
      # FIXME: we need the last_post_op's out_edge here (maybe deal with this in the wrapper?)
      # probably we can composite node's edges
      if self.post_op_chain:
        var_o = UniqueVar((self.post_op_chain[-1], i))
      else:
        var_o = UniqueVar((self, i))

      for te_out_info in out_edge_infos:
        if te_out_info:
          code += f"__builtin_IMCE_SEND({te_out_info.policy_info[0].address}, {var_o}, {te_out_info.fifo_id}, 0);"

    code += "\n"

    return code

  def _inner_loop_content(self, loop_count: int, recv_count: int) -> CodeBlock:
    return SimpleFor(loop_count, self._loop_body_content(recv_count), "inner_loop")

  def _outer_loop_content(self, loop_count: int, loop_pattern: dict) -> CodeBlock:
    code = TextBlock("")
    for pat in loop_pattern:
      code += self._inner_loop_content(pat["count"], pat["pattern"])

    return SimpleFor(loop_count, code, "outer_loop")

  def _content(self) -> CodeBlock:
    row_pattern = self.conv.extract_2d_pattern()
    code = TextBlock("")
    for row_pat in row_pattern:
      code += self._outer_loop_content(row_pat["count"], row_pat["pattern"])

    return code


class BatchNormBlock(ImceCallCodeBlock):
  """
  BatchNormBlock for batch normalization operations.
  Only generates computation. RECV/SEND handled by wrapper or ConvBlock.
  """
  num_in_edges = 3

  def __init__(self, call: 'BuilderContext', annotation: str = ""):
    """ Code block for batch normalization """
    super().__init__(call, annotation)

  def _content(self) -> CodeBlock:
    """Generate only computation, no RECV/SEND."""
    code = TextBlock("")

    # Identify edges by tensor type
    for edge in self.in_edges:
      if edge.dst_id.tensor_type == "fused_scale":
        scale_edge = edge
      elif edge.dst_id.tensor_type == "fused_bias":
        bias_edge = edge
      elif edge.dst_id.tensor_type == "data":
        data_edge = edge

    num_blocks = 4 if self.prev_op else 1

    for i in range(num_blocks):
      if self.prev_op and (data_edge in self.prev_op.out_edges):
        data_edge = self.prev_op
      var_data = UniqueVar((data_edge, i))
      var_scale = UniqueVar((scale_edge, i))
      var_bias = UniqueVar((bias_edge, i))
      var_o = UniqueVar((self, i))

      # e.g. __builtin_IMCE_MULTL(data, scale, 15);
      code += f"{var_o} = __builtin_IMCE_MULTL({var_data}, {var_scale}, 15);"
      # e.g. __builtin_IMCE_ADD(out, bias, 15);
      code += f"{var_o} = __builtin_IMCE_ADD({var_o}, {var_bias}, 15);"

    return code


class RecvSendWrapper(ImceCodeBlock):
  """
  Wrapper that adds RECV and SEND operations around a computation block.
  """

  def __init__(self, inner_block: ImceCallCodeBlock, annotation: str = ""):
    """Wrap a computation block with RECV/SEND operations.

    Args:
        inner_block: The computation block to wrap (VecBlock, BatchNormBlock, etc.)
        annotation: Optional annotation string
    """
    super().__init__(annotation)
    self.inner_block = inner_block
    self.call = inner_block.call.call
    self.in_edges = inner_block.in_edges
    self.out_edges = inner_block.out_edges

  def _content(self) -> CodeBlock:
    """Generate RECV, computation, and SEND for standalone operations."""
    # Constant tags that should not generate RECV/SEND
    const_tags = ["weight", "bias", "fused_scale", "fused_bias",
                  "min", "max", "threshold", "scale", "config"]

    code = TextBlock("")

    # Get tensor edge info for inputs and outputs
    te_in_infos = [DevConfig().get_tensor_edge_info_with_id_dir(
        edge.dst_id, "in") for edge in self.in_edges]
    te_out_infos = [DevConfig().get_tensor_edge_info_with_id_dir(
        edge.src_id, "out") for edge in self.out_edges]

    # Determine number of bitplanes to process based on inner block's prev_op flag
    # When prev_op exists, blocks process 4 bitplanes at once
    # When prev_op is None (standalone), blocks process 1 bitplane at a time
    # Note: inner_block should have prev_op=None since this wrapper is only for standalone ops
    # CHECK THIS!
    num_blocks = 4 if (hasattr(self.inner_block, 'prev_op')
                       and self.inner_block.prev_op) else 1

    # Generate RECV for non-constant input edges
    for i in range(num_blocks):
      for edge, te_info in zip(self.in_edges, te_in_infos):
        var_i = UniqueVar((edge, i))
        if te_info and not var_i.static and edge.dst_id.tensor_type not in const_tags:
          code += f"{var_i} = __builtin_IMCE_RECV({te_info.fifo_id});"

    # Add the inner block's computation
    code += copy(self.inner_block)

    # Generate SEND for all output edges
    for i in range(num_blocks):
      for te_out_info in te_out_infos:
        var_o = UniqueVar((self.inner_block, i))
        if te_out_info:
          code += f"__builtin_IMCE_SEND({te_out_info.policy_info[0].address}, {var_o}, {te_out_info.fifo_id}, 0);"

    # Wrap in a loop based on calls' type_args
    edge_shape = None
    for idx, arg in enumerate(self.call.args):
      if ConstPat.match(arg):
        continue
      else:
        if edge_shape is not None:
          assert (edge_shape == self.call.type_args[idx].shape), "all input args should have the same shape"
        else:
          edge_shape = self.call.type_args[idx].shape

    count = edge_shape[-1] * edge_shape[-2]

    return SimpleFor(count, code, f"{self.inner_block.__class__.__name__}_with_IO")


class ImceCodeBlockManager(NodeCodeBlockManager):
  """A class that manages and generates code blocks for imces."""

  def __init__(self, func_name: str):
    super().__init__()
    self.func_name = func_name

  @property
  def nodes(self) -> List[NodeID]:
    return NodeID.imces()

  @property
  def target(self) -> str:
    return "imce"

  def start_block(self) -> str:
    code = (
        "#include \"../common_decl.h\"\n"
        f"void {self.func_name}() {{\n"
        "  int hid = __builtin_IMCE_GET_CORE_HID();\n"
        "  int wid = __builtin_IMCE_GET_CORE_WID();\n"
        f"{indent(UniqueVar.get_decls_str(), '  ')}\n"
    )
    return code

  def end_block(self) -> str:
    return "}\n"


"""
  short16 test_builtins(short16 a, short16 b) {
  short16 var1 = __builtin_IMCE_ADD(a, b, 15);
  short16 var2 = __builtin_IMCE_SUB(a, var1, 15);
  short16 var3 = __builtin_IMCE_AND(a, var2, 15);
  short16 var4 = __builtin_IMCE_OR(a, var3, 15);
  short16 var5 = __builtin_IMCE_XOR(a, var4, 15);
  short16 var6 = __builtin_IMCE_SRL(a, var5, 15);
  short16 var7 = __builtin_IMCE_SLL(a, var6, 15);
  short16 var8 = __builtin_IMCE_SRA(a, var7, 15);
  short16 var9 = __builtin_IMCE_MAX(a, var8, 15);
  short16 var10 = __builtin_IMCE_MIN(a, var9, 15);
  short16 var11 = __builtin_IMCE_MULTL(a, var10, 15);
  short16 var12 = __builtin_IMCE_MULTH(a, var11, 15);

  short16 var14 = __builtin_IMCE_ADDI(var12, 1);
  short16 var15 = __builtin_IMCE_SUBI(var14, 1);
  short16 var16 = __builtin_IMCE_ANDI(var15, 1);
  short16 var17 = __builtin_IMCE_ORI(var16, 1);
  short16 var18 = __builtin_IMCE_XORI(var17, 1);
  short16 var19 = __builtin_IMCE_SRLI(var18, 1);
  short16 var20 = __builtin_IMCE_SLLI(var19, 1);
  short16 var21 = __builtin_IMCE_SRAI(var20, 1);
  short16 var22 = __builtin_IMCE_MAXI(var21, 1);
  short16 var23 = __builtin_IMCE_MINI(var22, 1);
  short16 var24 = __builtin_IMCE_MULTLI(var23, 1);
  short16 var25 = __builtin_IMCE_MULTHI(var24, 1);

  short16 var26 = __builtin_IMCE_DWCONV(var25, 1, 0, 1, 1);
  __builtin_IMCE_SEND(1, var26, 2, 3);
  short16 var27 = __builtin_IMCE_RECV(0);
  short16 var_min = __builtin_IMCE_RECV_MIN(0);
  short16 var_max = __builtin_IMCE_RECV_MAX(0);
  short16 var_cfg = __builtin_IMCE_RECV_CFG(0);
  short16 var_scan0 = __builtin_IMCE_RECV_SREG0(0);
  short16 var_scan1 = __builtin_IMCE_RECV_SREG1(0);
  __builtin_IMCE_SETFLAG(1);
  __builtin_IMCE_STANDBY(1, 2);

  short16 var28 = __builtin_IMCE_MAXPOOL(1, 2, 3);
  short16 var29 = __builtin_IMCE_AVGPOOL(1, 2, 3);

  __builtin_IMCE_ADDQ(var27, var28, 1, 2);
  __builtin_IMCE_SUBQ(a, var29, 1, 2);
  __builtin_IMCE_MULTLQ(a, var29, 1, 2);
  __builtin_IMCE_MULTHQ(a, var29, 1, 2);
  __builtin_IMCE_NU_QUANT(a, var29, 1, 2);
  __builtin_IMCE_MM_QUANT(a, 0, 15, 2);
  short16 var30 = __builtin_IMCE_GET_QREG(0);
  short16 var31 = __builtin_IMCE_GET_QREG(1);
  short16 var32 = __builtin_IMCE_GET_QREG(2);
  short16 var33 = __builtin_IMCE_GET_QREG(3);
  short16 var_0 = __builtin_IMCE_ADD(var30, var31, 15);
  short16 var_1 = __builtin_IMCE_ADD(var32, var33, 15);

  __builtin_IMCE_STEP();
  __builtin_IMCE_NOP();
  __builtin_IMCE_STOP();
  short16 var34 = __builtin_IMCE_GET_CREG(0);
  short16 var35 = __builtin_IMCE_GET_CREG(1);
  short16 var36 = __builtin_IMCE_GET_CREG(2);
  short16 var37 = __builtin_IMCE_GET_CREG(3);

  short16 var_2 = __builtin_IMCE_ADD(var34, var35, 15);
  short16 var_3 = __builtin_IMCE_ADD(var36, var37, 15);

  short16 var38 = __builtin_IMCE_SCAN_RW(a);

  short16 var_4 = __builtin_IMCE_ADD(var_0, var_1, 15);
  short16 var_5 = __builtin_IMCE_ADD(var_2, var_3, 15);
  short16 var_6 = __builtin_IMCE_ADD(var_4, var_5, 15);

  __builtin_IMCE_LOAD_LB(0);
"""
