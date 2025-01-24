from abc import *
from typing import *
from copy import copy
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from tvm.contrib.imcflow import NodeID, TensorID, TensorEdge
from tvm.relay.op.op_attrs import Conv2DAttrs
from tvm.relay.backend.contrib.imcflow.conv_util import ConvUtil
from tvm.relay.backend.contrib.imcflow.codeblock import CodeBlock, TextBlock, SimpleFor, UniqueVar
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
      code += f"{var} = __builtin_IMCE_RECV({te_info.fifo_id});"
    return code


class VecBlock(ImceCodeBlock):
  num_in_edges = 2

  def __init__(self, in_edges: List[TensorEdge], out_edge: TensorEdge, annotation: str = ""):
    """ Code block for adding two tensors """
    super().__init__(annotation)
    assert len(in_edges) == self.num_in_edges
    self.op_name = self._op_name()
    self.in_edges = in_edges
    self.out_edge = out_edge

  @abstractmethod
  def _op_name(self) -> str:
    pass

  def _content(self) -> CodeBlock:
    num_blocks = 4
    src_mask = 15

    code = TextBlock("")
    te_info0 = DevConfig().get_tensor_edge_info_with_id_dir(
        self.in_edges[0].dst_id, "in")  # a hack to get the tensor edge info
    te_info1 = DevConfig().get_tensor_edge_info_with_id_dir(
        self.in_edges[1].dst_id, "in")

    for i in range(num_blocks):
      # put a tuple of (tensor edge, block index) as the key, giving a unique variable name
      var_o = UniqueVar((self.out_edge, i))
      var_i0 = UniqueVar((self.in_edges[0], i))
      var_i1 = UniqueVar((self.in_edges[1], i))

      if te_info0 and not var_i0.static:
        code += f"{var_i0} = __builtin_IMCE_RECV({te_info0.fifo_id});"
      if te_info1 and not var_i1.static:
        code += f"{var_i1} = __builtin_IMCE_RECV({te_info1.fifo_id});"

      code += f"{var_o} = __builtin_IMCE_{self.op_name}({var_i0}, {var_i1}, {src_mask});" # e.g. __builtin_IMCE_ADD(a, b, 15);

    return code

class AddBlock(VecBlock):
  def _op_name(self) -> str:
    return "ADD"

class DivBlock(VecBlock):
  def _op_name(self) -> str:
    return "DIV"

class MultlBlock(VecBlock):
  def _op_name(self) -> str:
    return "MULTL"

class MulthBlock(VecBlock):
  def _op_name(self) -> str:
    return "MULTH"

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
      var_o = UniqueVar((self.out_edge, i))
      var_i = UniqueVar((self.in_edge, i))

      qreg_start_idx = i + 4 * self.o_split_idx
      code += f"{var_o} = __builtin_IMCE_MM_QUANT({var_i}, 0, {src_mask}, {qreg_start_idx});"

    return code


class ConcatBlock(ImceCodeBlock):

  """ Code block for concatenating multiple tensors """
  def __init__(self, in_edges: List[TensorEdge], out_edge: TensorEdge, annotation: str = ""):
    """ Code block for min/max quantization """
    super().__init__(annotation)
    self.in_edges = in_edges
    self.out_edge = out_edge

  def _content(self) -> CodeBlock:
    num_bitplanes = 4
    src_mask = 15

    code = TextBlock("")

    external_in_edges = [e for e in self.in_edges if e in DevConfig().TensorEdgetoInfo]
    internal_in_edge = (set(self.in_edges) - set(external_in_edges)).pop()

    for i in range(num_bitplanes):
      var_o = UniqueVar((self.out_edge, i))
      var_i = UniqueVar((internal_in_edge, i))
      for ext_edge in external_in_edges:
        var_e = UniqueVar((ext_edge, i))
        fifo_id = DevConfig().get_tensor_edge_info(ext_edge).fifo_id

        code += f"{var_e} = __builtin_IMCE_RECV({fifo_id});"
        code += f"{var_o} = __builtin_IMCE_OR({var_i}, {var_e}, {src_mask});"

    return code


class ConvBlock(ImceCodeBlock):
  """ Code block for receiving conv input data from given fifo id """
  num_in_edges = 2

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
