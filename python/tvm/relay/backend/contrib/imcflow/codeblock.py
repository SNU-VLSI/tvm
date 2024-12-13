from abc import *
from tvm.contrib.imcflow import *


class CodeBlockBase(metaclass=ABCMeta):
  def __init__(self, node_id: NodeID):
    self.node_id = node_id

  @abstractmethod
  def generate(self) -> str:
    pass

class CodeBlockStart(CodeBlockBase):
  def __init__(self, func_name: str):
    self.func_name = func_name

  def generate(self) -> str:
    code = f"void {self.func_name}() {{"
    code += "  int hid = __builtin_INODE_GET_CORE_HID();\n\n"
    return code

class CodeBlockEnd(CodeBlockBase):
  def __init__(self):
    pass

  def generate(self) -> str:
    return "}\n"

class PolicyUpdateCodeBlock(CodeBlockBase):
  """ Code block for updating policy table for given inode's hw node id  """
  def __init__(self, node_id: NodeID):
    assert node_id.is_inode(), "PolicyUpdateCodeBlock can only be used for inode"
    super().__init__(node_id)
    self.same_row_node_ids = [node_id] + node_id.slaves()

  def generate(self) -> str:
    code = "int policy_table_start_address;\n"
    for id in self.same_row_node_ids:
      db = ImcflowDeviceConfig().PolicyTableDict[id]
      code += f"\npolicy_table_start_address = {db.base_address};\n"
      for i in range(0, db.size, 32):
        code += f"__builtin_INODE_PU(policy_table_start_address, {i}, {int(i / 32)}, {id.to_coord(1)});\n"

    return code

class WriteIMEMCodeBlock(CodeBlockBase):
  pass

class WriteIMCUCodeBlock(CodeBlockBase):
  pass

class CtrlCodeBlock(CodeBlockBase):
  pass

class RecvCodeBlock(CodeBlockBase):
  def __init__(self, node_id: NodeID, tensor_edge: TensorEdge):
    super().__init__(node_id)
    self.tensor_edge = tensor_edge

  def generate(self) -> str:
    code = "int recv_start_address;\n"
    code += "int recv_size;\n"

    db = ImcflowDeviceConfig().get_tensor_edge_info(self.tensor_edge).data_block
    fifo_id = ImcflowDeviceConfig().get_tensor_edge_info(self.tensor_edge).fifo_id

    code += f"\nrecv_start_address = {db.base_address};\n"
    code += f"\nrecv_size = {db.size};\n"
    code += f"for (int i = 0; i < recv_size; i += 32) {{\n"
    code += f"  __builtin_INODE_RECV(i, 0, 0, {fifo_id});\n"
    code += f"}}\n"

    return code

class SendCodeBlock(CodeBlockBase):
  pass

# for op:
#   if op == "relu":
#     codeblocks.append(PolicyUpdateBlock(Node.inode_0))
#     codeblocks.append(CtrlBlock())
#     codeblocks.append(Conv2dBlock())

#     codeblocks.generate()
