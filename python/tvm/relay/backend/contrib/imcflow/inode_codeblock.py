from tvm.relay.backend.contrib.imcflow.codeblock import *
from tvm.contrib.imcflow import DataBlock
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from textwrap import indent
import pdb


class InodeCodeBlock(CodeBlock):
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
  def _content(self) -> Union[str, CodeBlock]:
    pass


class PolicyUpdateBlock(InodeCodeBlock):
  """ Code block for updating policy table for given inode's hw node id  """
  def __init__(self, node_id: NodeID, annotation: str = ""):
    super().__init__(annotation)
    assert node_id.is_inode(), "PolicyUpdateBlock can only be used for inode"
    self.node_id = node_id

  def _content(self) -> Union[str, CodeBlock]:
    assert self.node_id.is_inode(), "PolicyUpdateCodeBlock can only be used for inode"
    same_row_node_ids = [self.node_id] + self.node_id.slaves()
    code = "int policy_table_start_address;\n"
    for id in same_row_node_ids:
      db = DevConfig().PolicyTableDict[id]
      code += f"\npolicy_table_start_address = {db.base_address};\n"
      for i in range(0, db.size, 32):
        code += f"__builtin_INODE_PU(policy_table_start_address, {i}, {int(i / 32)}, {id.to_coord(1)});\n"

    return code


class WriteIMEMBlock(InodeCodeBlock):
  pass


class WriteIMCUBlock(InodeCodeBlock):
  pass


class RecvBlock(InodeCodeBlock):
  """ Code block for receiving data from given fifo id """

  def __init__(self, block: DataBlock, fifo_id: int, annotation: str = ""):
    super().__init__(annotation)
    self.block = block
    self.fifo_id = fifo_id

  def _content(self) -> Union[str, CodeBlock]:
    assert self.block.size % 32 == 0, "DataBlock size must be multiple of 32"
    recv_count = self.block.size // 32

    code = f"int recv_data_base_address = {self.block.base_address};\n"
    code += SimpleFor(recv_count,
                      f"__builtin_INODE_RECV(recv_data_base_address + i*32, 0, 0, {self.fifo_id});")

    return code


class SendBlock(InodeCodeBlock):
  pass


"""
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
