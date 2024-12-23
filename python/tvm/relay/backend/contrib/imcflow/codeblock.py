from abc import *
from tvm.contrib.imcflow import *
from tvm.relay.op.op_attrs import Conv2DAttrs


class CodeBlockBase(metaclass=ABCMeta):
  def __init__(self, node_id: NodeID, annotation: str):
    self.node_id = node_id
    self.builtin_prefix = "__builtin_INODE" if node_id.is_inode() else "__builtin_IMCE"
    self.annotation = annotation

  @abstractmethod
  def _generate(self) -> str:
    pass

  def generate(self) -> str:
    code = f"// generate: {self.annotation}\n"
    code += self._generate()
    code = f"// endgenerate\n"
    return code


class CodeBlockStart(CodeBlockBase):
  def __init__(self, name: str, target: str):
    assert target in ["inode", "imce"], \
        "target must be either 'inode' or 'imce'"
    self.target = target
    self.func_name = f"{name}_{target}"
    self.builtin_prefix = "__builtin_INODE" if target == "inode" else "__builtin_IMCE"

  def _generate(self) -> str:
    pass

  def generate(self) -> str:
    code = f"void {self.func_name}() {{"
    code += f"  int hid = {self.builtin_prefix}_GET_CORE_HID();\n\n"
    code += f"  int wid = {self.builtin_prefix}_GET_CORE_WID();\n\n"
    return code


class CodeBlockEnd(CodeBlockBase):
  def __init__(self):
    pass

  def _generate(self) -> str:
    pass

  def generate(self) -> str:
    return "}\n"


class PolicyUpdateCodeBlock(CodeBlockBase):
  """ Code block for updating policy table for given inode's hw node id  """

  def generate(self) -> str:
    assert self.node_id.is_inode(), "PolicyUpdateCodeBlock can only be used for inode"
    same_row_node_ids = [self.node_id] + self.node_id.slaves()
    code = "int policy_table_start_address;\n"
    for id in same_row_node_ids:
      db = ImcflowDeviceConfig().PolicyTableDict[id]
      code += f"\npolicy_table_start_address = {db.base_address};\n"
      for i in range(0, db.size, 32):
        code += f"{self.builtin_prefix}_PU(policy_table_start_address, {i}, {int(i / 32)}, {id.to_coord(1)});\n"

    return code


class WriteIMEMCodeBlock(CodeBlockBase):
  pass


class WriteIMCUCodeBlock(CodeBlockBase):
  pass


class CtrlCodeBlock(CodeBlockBase):
  pass


class RecvCodeBlock(CodeBlockBase):
  """ Code block for receiving data from given fifo id """

  def set_recv_info(self, size: int, fifo_id: int):
    self.size = size
    self.fifo_id = fifo_id

  def _generate(self) -> str:
    code = "int recv_start_address;\n"
    code += "int recv_size;\n"
    code += f"\nrecv_size = {self.size};\n"
    code += f"for (int i = 0; i < recv_size; i += 32) {{\n"
    code += f"  {self.builtin_prefix}_RECV(i, 0, 0, {self.fifo_id});\n"
    code += f"}}\n"

    return code

class ConvCodeBlock(CodeBlockBase):
  """ Code block for receiving conv input data from given fifo id """

  def add_tensor(self):
    # TODO: for adding another tensor
    pass

  def add_op(self):
    # TODO: add vector
    pass


  def set_recv_info(self, conv_attr: Conv2DAttrs, fifo_id: int):
    self.conv_attr = conv_attr
    self.fifo_id = fifo_id

  def _generate(self) -> str:
    code = "int recv_start_address;\n"
    code += "int recv_size;\n"
    code += f"\nrecv_size = {self.conv_attr};\n"
    code += f"for (int i = 0; i < recv_size; i += 32) {{\n"
    code += f"  {self.builtin_prefix}_RECV(i, 0, 0, {self.fifo_id});\n"
    code += f"}}\n"

    return code


class SendCodeBlock(CodeBlockBase):
  pass
