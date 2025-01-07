from abc import *
from typing import *
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from tvm.contrib.imcflow import NodeID, TensorID, TensorEdge
from tvm.relay.op.op_attrs import Conv2DAttrs
from tvm.relay.backend.contrib.imcflow.conv_util import ConvUtil
from tvm.relay.backend.contrib.imcflow.codeblock import CodeBlock, SimpleFor, UniqueVar
import pdb


class ImceCodeBlock(CodeBlock):
  @abstractmethod
  def _content(self) -> Union[str, CodeBlock]:
    pass


class LoadLBBlock(ImceCodeBlock):
  """ Code block for receiving data from given fifo id to the line buffer """

  def __init__(self, count: int, repeat: int, fifo_id: int, annotation: str = ""):
    super().__init__(annotation)
    self.count = count
    self.repeat = repeat
    self.fifo_id = fifo_id

  def _content(self) -> Union[str, CodeBlock]:
    code = f"__builtin_IMCE_LOAD_LB({self.fifo_id});\n"
    return SimpleFor(self.count, code * self.repeat)


class AddBlock(ImceCodeBlock):
  num_in_edges = 2

  def __init__(self, in_edges: List[TensorEdge], out_edge: TensorEdge, annotation: str = ""):
    """ Code block for adding two tensors """
    super().__init__(annotation)
    assert len(in_edges) == self.num_in_edges
    self.in_edges = in_edges
    self.out_edge = out_edge

  def _content(self) -> Union[str, CodeBlock]:
    num_blocks = 4
    src_mask = 15

    var_o = UniqueVar(self.out_edge)
    var_i0 = UniqueVar(self.in_edges[0])
    var_i1 = UniqueVar(self.in_edges[1])

    code = ""
    pdb.set_trace()
    te_info0 = DevConfig().get_tensor_edge_info_with_id_dir(self.in_edges[0].dst_id, "in") # a hack to get the tensor edge info
    te_info1 = DevConfig().get_tensor_edge_info_with_id_dir(self.in_edges[1].dst_id, "in")

    for i in range(num_blocks):
      if te_info0:
        code += f"short16 {var_i0}_{i} = __builtin_IMCE_RECV({te_info0.fifo_id});\n"
      if te_info1:
        code += f"short16 {var_i1}_{i} = __builtin_IMCE_RECV({te_info1.fifo_id});\n"

      code += f"short16 {var_o}_{i} = __builtin_IMCE_ADD({var_i0}_{i}, {var_i1}_{i}, {src_mask});\n"

    return code

class ConvBlock(ImceCodeBlock):
  """ Code block for receiving conv input data from given fifo id """

  def __init__(self, in_edges: List[TensorEdge], out_edge: TensorEdge, shapes: dict, conv_attrs: Conv2DAttrs,
               annotation: str = ""):
    super().__init__(annotation)
    assert len(in_edges) == 2
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

  def _loop_body_content(self, recv_count: int) -> str:
    num_blocks = 4
    fifo_id_i = DevConfig().get_tensor_edge_info_with_id_dir(self.in_edge.dst_id, "in").fifo_id

    var_creg = UniqueVar(self.out_edge)
    # hack to get the last tensor edge
    out_edge = self.post_ops[-1].out_edge if self.post_ops else self.out_edge
    var_o = UniqueVar(out_edge)
    # fifo_id_o = DevConfig().get_tensor_edge_info_with_id_dir(out_edge.src_id, "out").fifo_id
    # policy_addr_o = DevConfig().get_tensor_edge_info_with_id_dir(out_edge.src_id, "out").policy_addr
    fifo_id_o = -1
    policy_addr_o = -1

    code = ""
    code += LoadLBBlock(recv_count, num_blocks, fifo_id_i)
    code += "__builtin_IMCE_STEP();\n"
    code += f"{var_creg}_0 = __builtin_IMCE_GET_CREG((short)0);\n"
    code += f"{var_creg}_1 = __builtin_IMCE_GET_CREG((short)1);\n"
    code += f"{var_creg}_2 = __builtin_IMCE_GET_CREG((short)2);\n"
    code += f"{var_creg}_3 = __builtin_IMCE_GET_CREG((short)3);\n"

    for op in self.post_ops:
      code += "\n"
      code += str(op)

    code += "\n"
    code += f"__builtin_IMCE_SEND({policy_addr_o}, {var_o}_0, {fifo_id_o}, 0);\n"
    code += f"__builtin_IMCE_SEND({policy_addr_o}, {var_o}_1, {fifo_id_o}, 0);\n"
    code += f"__builtin_IMCE_SEND({policy_addr_o}, {var_o}_2, {fifo_id_o}, 0);\n"
    code += f"__builtin_IMCE_SEND({policy_addr_o}, {var_o}_3, {fifo_id_o}, 0);\n"
    pdb.set_trace()

    # code += SendBlock(1, self.policy_addr)
    return code

  def _inner_loop_content(self, loop_count: int, recv_count: int) -> str:
    return SimpleFor(loop_count, self._loop_body_content(recv_count), "inner_loop")

  def _outer_loop_content(self, loop_count: int, loop_pattern: dict) -> str:
    code = ""
    for pat in loop_pattern:
      code += self._inner_loop_content(pat["count"], pat["pattern"])

    return SimpleFor(loop_count, code, "outer_loop")

  def _content(self) -> Union[str, CodeBlock]:
    row_pattern = self.conv.extract_2d_pattern()
    code = ""
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
  __builtin_IMCE_MM_QUANT(a, var29, 1, 2);
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
