from abc import *
from tvm.contrib.imcflow import *
from tvm.relay.op.op_attrs import Conv2DAttrs


class CodeBlockBase(metaclass=ABCMeta):
  def __init__(self, annotation: str):
    self.annotation = annotation

  def _generate(self) -> str:
    pass

  def generate(self) -> str:
    code = f"// generate: {self.annotation}\n"
    code += self._generate()
    code = f"// endgenerate\n"
    return code

class ImceCodeBlockBase(CodeBlockBase):
  def __init__(self, node_id: NodeID, annotation: str):
    super().__init__(annotation)
    assert node_id.is_imce(), "ImceCodeBlockBase can only be used for imce"
    self.node_id = node_id

  @abstractmethod
  def _generate(self) -> str:
    pass


class InodeCodeBlockBase(CodeBlockBase):
  def __init__(self, node_id: NodeID, annotation: str):
    super().__init__(annotation)
    assert node_id.is_inode(), "InodeCodeBlockBase can only be used for inode"
    self.node_id = node_id

  @abstractmethod
  def _generate(self) -> str:
    pass

class CodeBlockStart(CodeBlockBase):
  def __init__(self, name: str, target: str):
    assert target in ["inode", "imce"], \
        "target must be either 'inode' or 'imce'"
    self.target = target
    self.func_name = name

  def generate(self) -> str:
    code = f"void {self.func_name}() {{"
    if self.target == "imce":
      code += f"  int hid = __builtin_IMCE_GET_CORE_HID();\n"
      code += f"  int wid = __builtin_IMCE_GET_CORE_WID();\n\n"
    else:
      code += f"  int hid = __builtin_INODE_GET_CORE_HID();\n\n"
    return code


class CodeBlockEnd(CodeBlockBase):
  def __init__(self):
    pass

  def generate(self) -> str:
    return "}\n"


class PolicyUpdateCodeBlock(InodeCodeBlockBase):
  """ Code block for updating policy table for given inode's hw node id  """

  def generate(self) -> str:
    assert self.node_id.is_inode(), "PolicyUpdateCodeBlock can only be used for inode"
    same_row_node_ids = [self.node_id] + self.node_id.slaves()
    code = "int policy_table_start_address;\n"
    for id in same_row_node_ids:
      db = ImcflowDeviceConfig().PolicyTableDict[id]
      code += f"\npolicy_table_start_address = {db.base_address};\n"
      for i in range(0, db.size, 32):
        code += f"__builtin_INODE_PU(policy_table_start_address, {i}, {int(i / 32)}, {id.to_coord(1)});\n"

    return code


class WriteIMEMCodeBlock(InodeCodeBlockBase):
  pass


class WriteIMCUCodeBlock(InodeCodeBlockBase):
  pass


class CtrlCodeBlock(CodeBlockBase):
  """
  DONE, HALT, INTRT, STANDBY, SET_ADDR_CNT, SET_FLAG
  NOP, STEP, STOP
  """
  pass


class RecvCodeBlock(InodeCodeBlockBase):
  """ Code block for receiving data from given fifo id """

  def set_info(self, block: DataBlock, fifo_id: int):
    self.block = block
    self.fifo_id = fifo_id

  def _generate(self) -> str:
    code += f"int recv_data_base_address = {self.block.base_address};\n"
    code += f"for (int i = 0; i < {self.block.size}; i += 32) {{\n"
    code += f"  __builtin_INODE_RECV(recv_data_base_address + i, 0, 0, {self.fifo_id});\n" # TODO: use imm?
    code += f"}}\n"

    return code

class SimpleRecvCodeBlock(ImceCodeBlockBase):
  """ Code block for receiving data from given fifo id """

  def set_info(self, size: int, fifo_id: int):
    self.size = size
    self.fifo_id = fifo_id

  def _generate(self) -> str:
    code += f"for (int i = 0; i < {self.size}; i += 32) {{\n"
    code += f"  __builtin_IMCE_RECV({self.fifo_id});\n"
    code += f"}}\n"

    return code

class InodeSendCodeBlock(InodeCodeBlockBase):
  pass


class SimpleSendCodeBlock(ImceCodeBlockBase):
  pass

class ConvCodeBlock(ImceCodeBlockBase):
  """ Code block for receiving conv input data from given fifo id """

  def add_tensor(self):
    # TODO: for adding another tensor
    pass

  def add_op(self, op: List[str]):
    # TODO: add support for vector ops with operands, etc.
    self.op = op

  def set_info(self, shapes: dict, conv_attr: Conv2DAttrs, fifo_id: int, policy_addr: int):
    self.kH, self.kW = conv_attr.kernel_size
    self.iH, self.iW = shapes["data"][2], shapes["data"][3]
    self.oH, self.oW = shapes["output"][2], shapes["output"][3]
    self.padding = conv_attr.padding
    self.dilation = conv_attr.dilation
    self.strides = conv_attr.strides
    self.fifo_id = fifo_id
    self.policy_addr = policy_addr

  def _get_linebuffer_length(self) -> int:
    length = (self.kH - 1) * self.iW + self.kW

  def _generate_loop_head(self) -> str:
    block = SimpleRecvCodeBlock(self.node_id, "loop_head")
    recv_size = self._get_linebuffer_length() * 32 # in bytes
    block.set_info(recv_size, self.fifo_id)
    return block._generate()


  def _generate_loop_body(self) -> str:
    pass

  def _generate_loop_tail(self) -> str:
    pass

  def _generate(self) -> str:
    code += self._generate_loop_head()
    code += self._generate_loop_body()
    code += self._generate_loop_tail()

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


  __builtin_INODE_SEND(1, 1, 1, 1);
  __builtin_INODE_RECV(1, 1, 1, 1);
  __builtin_INODE_LAYERINIT();
  __builtin_INODE_IMCE_COMPUTE(1);

  __builtin_INODE_WR_IMEM(1, 1, 1);
  __builtin_INODE_WR_IMCU(1, 1, 1);
  __builtin_INODE_WR_REG(1, 1, 1);
  __builtin_INODE_SET_ADDR_CNT(1);

  __builtin_INODE_SET_FLAG(1);
  __builtin_INODE_STANDBY(1, 1);

  __builtin_INODE_DONE();
  __builtin_INODE_HALT();
  __builtin_INODE_INTRT(1);

  __builtin_INODE_PU(1, 1, 1, 1);

  int a = __builtin_INODE_GET_CORE_HID();
  int b = __builtin_INODE_GET_CORE_WID();


"""