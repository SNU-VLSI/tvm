from abc import *
from typing import *
from tvm.contrib.imcflow import NodeID
from tvm.relay.op.op_attrs import Conv2DAttrs
from tvm.relay.backend.contrib.imcflow.conv_util import ConvUtil
from tvm.relay.backend.contrib.imcflow.codeblock import CodeBlock, SimpleFor
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
  def __init__(self, fifo_id: int, annotation: str = ""):
    super().__init__(annotation)
    self.fifo_id = fifo_id

  def _content(self) -> Union[str, CodeBlock]:
    # code = f"{Var("y")} = __builtin_IMCE_RECV({self.fifo_id});\n"
    # code = f"{Var("x")} = __builtin_IMCE_ADD({Var("x")}, {Var("y")}, 15);\n"
    code = f"y = __builtin_IMCE_RECV({self.fifo_id});\n"
    code = f"x = __builtin_IMCE_ADD(x, y, 15);\n"
    return code

class ConvBlock(ImceCodeBlock):
  """ Code block for receiving conv input data from given fifo id """

  def __init__(self, shapes: dict, conv_attrs: Conv2DAttrs, fifo_id: int, policy_addr: int,
               annotation: str = ""):
    super().__init__(annotation)
    self.fifo_id = fifo_id
    self.policy_addr = policy_addr
    self.conv = ConvUtil(shapes["data"][2], shapes["data"][3],
                         conv_attrs.padding[0], conv_attrs.strides[0],
                         conv_attrs.kernel_size[0], conv_attrs.kernel_size[1])

  def add_tensor(self):
    # TODO: for adding another tensor
    pass

  def add_op(self, op: List[str]):
    # TODO: add support for vector ops with operands, etc.
    self.op = op

  def _loop_body_content(self, recv_count: int) -> str:
    code = ""
    code += LoadLBBlock(recv_count, 4, self.fifo_id)
    code += "__builtin_IMCE_STEP();\n"
    code += "c0 = __builtin_IMCE_GET_CREG((short)0);\n"
    code += "c1 = __builtin_IMCE_GET_CREG((short)1);\n"
    code += "c2 = __builtin_IMCE_GET_CREG((short)2);\n"
    code += "c3 = __builtin_IMCE_GET_CREG((short)3);\n"

    code += "c0 = __builtin_IMCE_ADD(c0, a0, 15);\n"
    code += "c1 = __builtin_IMCE_ADD(c1, a1, 15);\n"
    code += "c2 = __builtin_IMCE_ADD(c2, a2, 15);\n"
    code += "c3 = __builtin_IMCE_ADD(c3, a3, 15);\n"

    code += "c0 = __builtin_IMCE_MAX(c0, 0, 0);\n"
    code += "c1 = __builtin_IMCE_MAX(c1, 0, 0);\n"
    code += "c2 = __builtin_IMCE_MAX(c2, 0, 0);\n"
    code += "c3 = __builtin_IMCE_MAX(c3, 0, 0);\n"

    code += f"__builtin_IMCE_SEND({self.policy_addr}, c0, {self.fifo_id}, 0);\n"
    code += f"__builtin_IMCE_SEND({self.policy_addr}, c1, {self.fifo_id}, 0);\n"
    code += f"__builtin_IMCE_SEND({self.policy_addr}, c2, {self.fifo_id}, 0);\n"
    code += f"__builtin_IMCE_SEND({self.policy_addr}, c3, {self.fifo_id}, 0);\n"

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
