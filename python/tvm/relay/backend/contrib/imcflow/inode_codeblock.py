from tvm.relay.backend.contrib.imcflow.codeblock import *
from tvm.contrib.imcflow import DataBlock, InstEdgeInfo, TensorID
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
import math
import pdb


class InodeCodeBlock(CodeBlock):
  def __init__(self, annotation: str = ""):
    super().__init__()
    self.annotation = annotation

  def content(self) -> CodeBlock:
    if self.annotation:
      code = TextBlock("\n")
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
    same_row_node_ids.sort(key=lambda id: id.to_coord(1))  # Sort by id.to_coord(1)

    code = TextBlock("")
    for id in same_row_node_ids:
      db = DevConfig().MemLayout.get_data_block_by_id(f"{id.name}_policy")
      if db is None:
        continue
      var = UniqueVar("policy_table_start_address", dtype="int")
      code += f"{var} = {db.base_address};"
      for i in range(0, db.size, 32):
        code += f"__builtin_INODE_PU({var}, {i}, {int(i / 32)}, {id.to_coord(1)});"
    code += ""

    return code


class WriteIMEMBlock(InodeCodeBlock):
  """ Code block for writing IMEM given InstEdgeInfo """

  def __init__(self, edge_info: InstEdgeInfo, annotation: str = ""):
    super().__init__(annotation)
    self.edge_info = edge_info

  def _content(self) -> Union[str, CodeBlock]:
    code = TextBlock("")

    db = self.edge_info.data_block
    policy_addr = self.edge_info.policy_info[0].address # get first policy address

    var = UniqueVar("imem_start_address", dtype="int")
    code += f"{var} = {db.base_address};"
    code += SimpleFor(math.ceil(db.size / 32),
                      lambda iter: f"__builtin_INODE_WR_IMEM({var} + {iter}*32, 0, {policy_addr});")
                      # rs1, imm, policy
    code += ""

    return code


class WriteIMCUBlock(InodeCodeBlock):
  """ Code block for writing IMCU weights given the master inode's hid  """

  def __init__(self, node_id: NodeID, annotation: str = ""):
    super().__init__(annotation)
    assert node_id.is_inode(), "WriteIMCUBlock can only be used for inode"
    self.node_id = node_id

  def _content(self) -> Union[str, CodeBlock]:
    code = TextBlock("")
    region = DevConfig().MemLayout[f"{self.node_id.name}_data"]
    for db in region.blocks.values():
      if isinstance(db.id, TensorID) and "weight" == db.id.tensor_type:
        info = DevConfig().get_tensor_edge_info_with_id_dir(db.id, "out")
        assert info.fifo_id == 1, f"IMCU fifo id should be set to 1 (although not used), but got {info.fifo_id} for {db.id}"
        var = UniqueVar("imcu_start_address", dtype="int")
        code += f"{var} = {db.base_address};"
        code += SimpleFor(math.ceil(db.size / 32),
                          lambda iter: f"__builtin_INODE_WR_IMCU({var} + {iter}*32, 0, {info.policy_info[0].address});")
                          # rs1, imm, policy
        code += ""

    return code

class RecvBlock(InodeCodeBlock):
  """ Code block for receiving data from given fifo id """

  def __init__(self, block: DataBlock, fifo_id: int, annotation: str = ""):
    super().__init__(annotation)
    self.block = block
    self.fifo_id = fifo_id

  def _content(self) -> Union[str, CodeBlock]:
    assert self.block.size % 32 == 0, "DataBlock size must be multiple of 32"
    recv_count = self.block.size // 32
    code = TextBlock("")

    var = UniqueVar("recv_data_base_address", dtype="int")
    code += SimpleFor(recv_count,
                      lambda iter: f"__builtin_INODE_RECV({var} + {iter}*32, 0, 0, {self.fifo_id});")

    return code


class SendBlock(InodeCodeBlock):
  """ Code block for sending data from given fifo id """

  def __init__(self, block: DataBlock, fifo_id: int, annotation: str = ""):
    super().__init__(annotation)
    self.block = block
    self.fifo_id = fifo_id

  def _content(self) -> Union[str, CodeBlock]:
    assert self.block.size % 32 == 0, "DataBlock size must be multiple of 32"
    recv_count = self.block.size // 32
    code = TextBlock("")

    var = UniqueVar("send_data_base_address", dtype="int")
    code += SimpleFor(recv_count,
                      lambda iter: f"__builtin_INODE_SEND({var} + {iter}*32, 0, 0, {self.fifo_id});")

    return code



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
