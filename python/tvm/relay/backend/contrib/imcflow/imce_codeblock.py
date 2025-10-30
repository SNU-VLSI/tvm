from abc import *
from typing import *
from copy import copy
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from tvm.contrib.imcflow import NodeID, TensorID, TensorEdge
from tvm.relay.op.op_attrs import Conv2DAttrs
from tvm.relay.backend.contrib.imcflow.conv_util import ConvUtil
from tvm.relay.backend.contrib.imcflow.codeblock import *
from textwrap import indent
import logging
import pdb


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

  def __init__(self, in_edge: TensorEdge, annotation: str = ""):
    super().__init__(annotation)
    self.in_edge = in_edge

  def _content(self) -> CodeBlock:
    code = TextBlock("")
    te_info = DevConfig().get_tensor_edge_info_with_id_dir(
        self.in_edge.dst_id, "in")  # a hack to get the tensor edge info
    assert te_info, "Tensor edge info not found"

    size = DevConfig().MemLayout.get_data_block_by_id(self.in_edge.src_id).size
    assert size % 32 == 0, "Size must be a multiple of 32"
    recv_count = int(size / 32) # recv operates on 32-byte word

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


class VecBlock(ImceCodeBlock):
  """
  VecBlock is base class for implementing R,I type vector operations.
  """
  num_in_edges = 2

  def __init__(self, in_edges: List[TensorEdge], out_edge: TensorEdge, annotation: str = ""):
    """ Code block for adding two tensors """
    super().__init__(annotation)
    assert len(in_edges) == self.num_in_edges
    self.op_name = self._op_name()
    self.imm_value = self._get_imm_value()
    self.in_edges = in_edges
    self.out_edge = out_edge
    self.post_op = None

  @abstractmethod
  def _get_imm_value(self) -> int:
    pass

  @abstractmethod
  def _op_name(self) -> str:
    pass

  def _content(self) -> CodeBlock:

    code = TextBlock("")
    # a hack to get the tensor edge info for each input edge
    te_in_infos = [DevConfig().get_tensor_edge_info_with_id_dir(edge.dst_id, "in") for edge in self.in_edges]
    te_out_info = DevConfig().get_tensor_edge_info_with_id_dir(self.out_edge.src_id, "out")

    num_blocks = 4 if self.post_op else 1
    for i in range(num_blocks):
      # put a tuple of (tensor edge, block index) as the key, giving a unique variable name
      var_ins = [UniqueVar((edge, i)) for edge in self.in_edges]
      var_o = UniqueVar((self.out_edge, i))

      # te info is None for composite internal tensors
      for var_i, te_info in zip(var_ins, te_in_infos):
        if te_info and not var_i.static:
          code += f"{var_i} = __builtin_IMCE_RECV({te_info.fifo_id});"

      var_in_str = ", ".join([f"{var_i}" for var_i in var_ins])
      code += f"{var_o} = __builtin_IMCE_{self.op_name}({var_in_str}, {self.imm_value});" # e.g. __builtin_IMCE_ADD(a, b, 15);

      if te_out_info:
        code += f"__builtin_IMCE_SEND({te_out_info.policy_info[0].address}, {var_o}, {te_out_info.fifo_id}, 0);"

    if self.post_op:
      return code
    else:
      # FIXME: this can be problematic for >1 input edges
      count = te_in_infos[0].data_block.size // 32
      return SimpleFor(count, code, self.__class__.__name__)

class AddBlock(VecBlock):
  def _get_imm_value(self) -> int:
    return 15 # src_mask

  def _op_name(self) -> str:
    return "ADD"

class DivBlock(VecBlock):
  def _get_imm_value(self) -> int:
    return 15 # src_mask

  def _op_name(self) -> str:
    return "DIV"

class MultlBlock(VecBlock):
  def _get_imm_value(self) -> int:
    return 15 # src_mask

  def _op_name(self) -> str:
    return "MULTL"

class MulthBlock(VecBlock):
  def _get_imm_value(self) -> int:
    return 15 # src_mask

  def _op_name(self) -> str:
    return "MULTH"

class ReLUBlock(VecBlock):
  num_in_edges = 1
  def _get_imm_value(self) -> int:
    return 0 # immediate value for MAXI

  def _op_name(self) -> str:
    return "MAXI"

class MinmaxQuantBlock(ImceCodeBlock):
  num_in_edges = 3

  def __init__(self, in_edges: List[TensorEdge], out_edge: TensorEdge, o_split_idx: int, annotation: str = ""):
    """ Code block for min/max quantization """
    super().__init__(annotation)
    assert len(in_edges) == self.num_in_edges
    for edge in in_edges:
      if edge.dst_id.tensor_type == "data":
        self.in_edge = edge
    self.out_edge = out_edge
    self.o_split_idx = o_split_idx

  def _content(self) -> CodeBlock:
    num_blocks = 4
    src_mask = 15

    code = TextBlock("")

    for i in range(num_blocks):
      var_i = UniqueVar((self.in_edge, i))
      var_o = UniqueVar((self.out_edge, i))

      qreg_start_idx = i + 4 * self.o_split_idx
      # min max quantization does not require $rs2
      code += f"__builtin_IMCE_MM_QUANT({var_i}, 0, {src_mask}, {qreg_start_idx});"
      code += f"{var_o} = __builtin_IMCE_GET_QREG({i});"

    return code


class ConcatBlock(ImceCodeBlock):
  min_in_edges = 2

  """ Code block for concatenating multiple tensors """
  def __init__(self, in_edges: List[TensorEdge], out_edge: TensorEdge, annotation: str = ""):
    """ Code block for min/max quantization """
    super().__init__(annotation)
    assert len(in_edges) >= self.min_in_edges, "At least two input edges are required"
    self.in_edges = in_edges
    self.out_edge = out_edge

  def _content(self) -> CodeBlock:
    num_bitplanes = 4
    src_mask = 15

    code = TextBlock("")

    external_in_edges = [e for e in self.in_edges if e in DevConfig().TensorEdgetoInfo]
    internal_in_edge = (set(self.in_edges) - set(external_in_edges)).pop()

    for i in range(num_bitplanes):
      var_i = UniqueVar((internal_in_edge, i))
      var_o = UniqueVar((self.out_edge, i))
      for ext_edge in external_in_edges:
        var_e = UniqueVar((ext_edge, i))
        fifo_id = DevConfig().get_tensor_edge_info(ext_edge).fifo_id

        code += f"{var_e} = __builtin_IMCE_RECV({fifo_id});"
        code += f"{var_o} = __builtin_IMCE_OR({var_i}, {var_e}, {src_mask});"

    return code


class SplitBlock(ImceCodeBlock):
  num_in_edges = 1

  """ Code block for splitting a tensor into multiple tensors """
  def __init__(self, in_edge: TensorEdge, out_edges: List[TensorEdge], annotation: str = ""):
    super().__init__(annotation)
    self.in_edge = in_edge
    first_policies = [DevConfig().get_tensor_edge_info(out_edge).policy_info[0] for out_edge in out_edges]
    fifo_ids = [DevConfig().get_tensor_edge_info(out_edge).fifo_id for out_edge in out_edges]
    assert all(policy == first_policies[0] for policy in first_policies), "All output edges must have the same first policy info"
    assert all(fid == fifo_ids[0] for fid in fifo_ids), "All output edges must have the same fifo id"
    self.out_edge = out_edges[0]

  def _content(self) -> CodeBlock:
    return TextBlock("")


class ConvBlock(ImceCodeBlock):
  """ Code block for receiving conv input data from given fifo id """
  num_in_edges = 3

  def __init__(self, in_edges: List[TensorEdge], out_edge: TensorEdge, shapes: dict, conv_attrs: Conv2DAttrs,
               annotation: str = ""):
    super().__init__(annotation)
    assert len(in_edges) == self.num_in_edges
    for edge in in_edges:
      if edge.dst_id.tensor_type == "data":
        self.in_edge = edge
    self.out_edge = out_edge
    self.conv = ConvUtil(shapes["data"][2], shapes["data"][3],
                         conv_attrs.padding[0], conv_attrs.strides[0],
                         conv_attrs.kernel_size[0], conv_attrs.kernel_size[1])
    self.post_ops = []

  def add_post_op(self, code: CodeBlock):
    code.post_op = True
    self.post_ops.append(code)

  def _loop_body_content(self, recv_count: int) -> CodeBlock:
    num_blocks = 4
    fifo_id_i = DevConfig().get_tensor_edge_info_with_id_dir(
        self.in_edge.dst_id, "in").fifo_id
    if fifo_id_i != 0:
      logging.warning(f"conv block data fifo_id_i is not 0, but {fifo_id_i}")

    # hack to get the last tensor edge
    last_out_edge = self.post_ops[-1].out_edge if self.post_ops else self.out_edge
    out_edge_info = DevConfig().get_tensor_edge_info(last_out_edge)

    if out_edge_info:
      fifo_id_o = out_edge_info.fifo_id
      policy_addr_o = out_edge_info.policy_info[0].address
    else:
      logging.warning(f"Output edge info not found for {last_out_edge}")
      fifo_id_o = -1
      policy_addr_o = -1

    code = TextBlock("")
    code += LoadLBBlock(recv_count, num_blocks, fifo_id_i)
    code += "__builtin_IMCE_STEP();\n"

    for i in range(num_blocks):
      var_creg = UniqueVar((self.out_edge, i))
      code += f"{var_creg} = __builtin_IMCE_GET_CREG((short){i});"

    for op in self.post_ops:
      code += "\n"
      code += copy(op)

    code += "\n"
    for i in range(num_blocks):
      var_o = UniqueVar((last_out_edge, i))
      code += f"__builtin_IMCE_SEND({policy_addr_o}, {var_o}, {fifo_id_o}, 0);"

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

class BatchNormBlock(ImceCodeBlock):
  """
  BatchNormBlock is base class for implementing R,I type vector operations.
  """
  num_in_edges = 3

  def __init__(self, in_edges: List[TensorEdge], out_edge: TensorEdge, annotation: str = ""):
    """ Code block for adding two tensors """
    super().__init__(annotation)
    assert len(in_edges) == self.num_in_edges
    self.in_edges = in_edges
    self.out_edge = out_edge

  def _content(self) -> CodeBlock:

    code = TextBlock("")
    # a hack to get the tensor edge info for each input edge
    te_in_infos = [DevConfig().get_tensor_edge_info_with_id_dir(edge.dst_id, "in") for edge in self.in_edges]
    te_out_info = DevConfig().get_tensor_edge_info_with_id_dir(self.out_edge.src_id, "out")

    for edge in self.in_edges:
      if edge.dst_id.tensor_type == "fused_scale":
        scale_edge = edge
      elif edge.dst_id.tensor_type == "fused_bias":
        bias_edge = edge
      elif edge.dst_id.tensor_type == "data":
        data_edge = edge

    num_blocks = 4

    for i in range(num_blocks):
      # put a tuple of (tensor edge, block index) as the key, giving a unique variable name
      var_ins = [UniqueVar((edge, i)) for edge in self.in_edges]

      var_data = UniqueVar((data_edge, i))
      var_scale = UniqueVar((scale_edge, i))
      var_bias = UniqueVar((bias_edge, i))

      var_o = UniqueVar((self.out_edge, i))

      # te info is None for composite internal tensors
      for var_i, te_info in zip(var_ins, te_in_infos):
        if te_info and not var_i.static:
          code += f"{var_i} = __builtin_IMCE_RECV({te_info.fifo_id});"

      code += f"{var_o} = __builtin_IMCE_MULTL({var_data}, {var_scale}, 15);" # e.g. __builtin_IMCE_MULTL(data, scale, 15);
      code += f"{var_o} = __builtin_IMCE_ADD({var_o}, {var_bias}, 15);" # e.g. __builtin_IMCE_ADD(out, bias, 15);

      # TODO: SEND to output fifo may not be needed here, depending on the usage of BatchNorm
      if te_out_info:
        code += f"__builtin_IMCE_SEND({te_out_info.policy_info[0].address}, {var_o}, {te_out_info.fifo_id}, 0);"

    return code

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
