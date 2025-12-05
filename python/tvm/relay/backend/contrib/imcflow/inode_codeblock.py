from tvm.relay.backend.contrib.imcflow.codeblock import *
from tvm.contrib.imcflow import DataBlock, InstEdgeInfo, TensorID, TensorEdge, TensorEdgeInfo
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from textwrap import indent
import math
import pdb


class InodeCodeBlock(CodeBlock):
  def __init__(self, annotation: str = ""):
    super().__init__(annotation)
    # Subclasses should build their structure into self.body in __init__
    self.body = SequentialBlock()

  def _render(self) -> str:
    return self.body.render()


class PolicyUpdateBlock(InodeCodeBlock):
  """ Code block for updating policy table for given inode's hw node id  """

  def __init__(self, node_id: NodeID, annotation: str = ""):
    super().__init__(annotation)
    assert node_id.is_inode(), "PolicyUpdateBlock can only be used for inode"
    self.node_id = node_id
    self._build()

  def _build(self):
    assert self.node_id.is_inode(), "PolicyUpdateCodeBlock can only be used for inode"
    same_row_node_ids = [self.node_id] + self.node_id.slaves()
    same_row_node_ids.sort(key=lambda id: id.to_coord(1))

    for id in same_row_node_ids:
      db = DevConfig().CurrFuncMemLayout.get_data_block_by_edge(f"{id.name}_policy")
      if db is None:
        continue
      var = UniqueVar("policy_table_start_address", dtype="int")
      loop_count = math.ceil(db.size / 32)

      self.body.add(TextBlock(f"{var} = {db.offset};"))
      
      # FIXME: maybe we should leave the loop optimization to llvm?
      if loop_count > 5:
        # Using lambda for SimpleFor body to inject 'iter' variable
        self.body.add(SimpleFor(loop_count, 
            lambda iter, wid=id.to_coord(1): f"__builtin_INODE_PU({var} + {iter}*32, 0, {iter}, {wid});"))
      else:
        for i in range(loop_count):
          self.body.add(TextBlock(f"__builtin_INODE_PU({var}, {i*32}, {i}, {id.to_coord(1)});"))


class WriteIMEMBlock(InodeCodeBlock):
  """ Code block for writing IMEM given InstEdgeInfo """

  def __init__(self, edge_info: InstEdgeInfo, annotation: str = ""):
    super().__init__(annotation)
    self.edge_info = edge_info
    self._build()

  def _build(self):
    db = self.edge_info.data_block
    policy_addr = self.edge_info.policy_info[0].address

    var = UniqueVar("imem_start_address", dtype="int")
    self.body.add(TextBlock(f"{var} = {db.offset};"))
    self.body.add(TextBlock(f"__builtin_INODE_SET_ADDR_CNT(0);"))

    self.body.add(SimpleFor(math.ceil(db.size / 32),
                      lambda iter: f"__builtin_INODE_WR_IMEM({var} + {iter}*32, 0, {policy_addr});"))


class WriteIMCUBlock(InodeCodeBlock):
  """ Code block for writing IMCU weights given the master inode's hid  """

  def __init__(self, node_id: NodeID, annotation: str = ""):
    super().__init__(annotation)
    assert node_id.is_inode(), "WriteIMCUBlock can only be used for inode"
    self.node_id = node_id
    self._build()

  def _build(self):
    region = DevConfig().CurrFuncMemLayout[f"{self.node_id.name}_data"]
    for db in region.blocks.values():
      if isinstance(db.id, TensorEdge) and "weight" == db.id.src_id.tensor_type:
        info = DevConfig().get_tensor_edge_info(db.id)
        assert info.fifo_id == 1, f"IMCU fifo id should be set to 1 (although not used), but got {info.fifo_id} for {db.id}"
        var = UniqueVar("imcu_start_address", dtype="int")
        
        self.body.add(TextBlock(f"{var} = {db.offset};"))
        self.body.add(TextBlock(f"__builtin_INODE_SET_ADDR_CNT(0);"))
        self.body.add(SimpleFor(math.ceil(db.size / 32),
                          lambda iter: f"__builtin_INODE_WR_IMCU({var} + {iter}*32, 0, {info.policy_info[0].address});"))


class RecvBlock(InodeCodeBlock):
  """ Code block for receiving data from given fifo id """

  def __init__(self, block: DataBlock, fifo_id: int, annotation: str = ""):
    super().__init__(annotation)
    self.block = block
    self.fifo_id = fifo_id
    self._build()

  def _build(self):
    recv_count = math.ceil(self.block.size / 32)
    var = UniqueVar("recv_data_base_address", dtype="int")
    
    self.body.add(TextBlock(f"{var} = {self.block.offset};"))
    self.body.add(SimpleFor(recv_count,
                      lambda iter: f"__builtin_INODE_RECV({var} + {iter}*32, 0, 0, {self.fifo_id});"))


class SendBlock(InodeCodeBlock):
  """ Code block for sending data from given fifo id """

  def __init__(self, block: DataBlock, edge_info: TensorEdgeInfo, annotation: str = ""):
    super().__init__(annotation)
    self.block = block
    self.edge_info = edge_info
    self._build()

  def _build(self):
    recv_count = math.ceil(self.block.size / 32)
    fifo_id = self.edge_info.fifo_id
    assert fifo_id >= 0, "fifo id should be assigned to a positive id"
    next_policy_addr = self.edge_info.policy_info[0].address

    var = UniqueVar("send_data_base_address", dtype="int")
    self.body.add(TextBlock(f"{var} = {self.block.offset};"))
    self.body.add(SimpleFor(recv_count,
                      lambda iter: f"__builtin_INODE_SEND({var} + {iter}*32, 0, {next_policy_addr}, {fifo_id});"))


class SendBlockInterleaved(InodeCodeBlock):
  """ Code block for sending data from given fifo id """

  def __init__(self, blocks: List[DataBlock], edge_infos: List[TensorEdgeInfo], annotation: str = ""):
    super().__init__(annotation)
    assert len(blocks) == len(edge_infos), "# of blocks and fifo_ids must be equal"
    self.blocks = blocks
    self.edge_infos = edge_infos
    self._build()

  def _build(self):
    recv_count = math.ceil(self.blocks[0].size / 32)
    inst_info = []
    for idx, (block, edge_info) in enumerate(zip(self.blocks, self.edge_infos)):
      assert recv_count == math.ceil(block.size / 32), "blocks should have the same recv_counts"
      next_policy_addr = edge_info.policy_info[0].address
      fifo_id = edge_info.fifo_id
      inst_info.append((block.offset, next_policy_addr, fifo_id))

    self.body.add(SimpleFor(recv_count,
                      lambda iter: "\n".join([
                        f"__builtin_INODE_SEND({iter}*32, {offset}, {policy_addr}, {fid});"
                        for offset, policy_addr, fid in inst_info
                      ])))


class IMCEComputeBlock(InodeCodeBlock):
  """ Code block for sending data from given fifo id """

  def __init__(self, policy_addr, annotation: str = ""):
    super().__init__(annotation)
    self.policy_addr = policy_addr
    self._build()

  def _build(self):
    self.body.add(TextBlock(f"__builtin_INODE_IMCE_COMPUTE(0, {self.policy_addr});"))


class StandbyAndIntrtBlock(InodeCodeBlock):
  def __init__(self, node_ids: List[NodeID], annotation: str = ""):
    super().__init__(annotation)
    self.node_ids = node_ids
    self._build()

  def _build(self):
    for node in self.node_ids:
      self.body.add(TextBlock(f"__builtin_INODE_STANDBY({node.value}, 1);"))
    self.body.add(TextBlock(f"__builtin_INODE_DONE();"))
    self.body.add(TextBlock(f"__builtin_INODE_INTRT(0);"))
    self.body.add(TextBlock(f"__builtin_INODE_HALT();"))


class Standby(InodeCodeBlock):
  def __init__(self, node_ids: List[NodeID], annotation: str = ""):
    super().__init__(annotation)
    self.node_ids = node_ids
    self._build()

  def _build(self):
    for node in self.node_ids:
      self.body.add(TextBlock(f"__builtin_INODE_STANDBY({node.value}, 1);"))


class SetFlagAndHaltBlock(InodeCodeBlock):
  def __init__(self, annotation: str = ""):
    super().__init__(annotation)
    self._build()

  def _build(self):
    self.body.add(TextBlock(f"__builtin_INODE_SET_FLAG(1);"))
    self.body.add(TextBlock(f"__builtin_INODE_HALT();"))


class SetFlag(InodeCodeBlock):
  def __init__(self, annotation: str = ""):
    super().__init__(annotation)
    self._build()

  def _build(self):
    self.body.add(TextBlock(f"__builtin_INODE_SET_FLAG(1);"))


class ClearFlag(InodeCodeBlock):
  def __init__(self, annotation: str = ""):
    super().__init__(annotation)
    self._build()

  def _build(self):
    self.body.add(TextBlock(f"__builtin_INODE_SET_FLAG(0);"))


class InodeCodeBlockManager(NodeCodeBlockManager):
  """A class that manages and generates code blocks for inodes."""

  def __init__(self, func_name: str):
    super().__init__()
    self.func_name = func_name

  @property
  def nodes(self) -> List[NodeID]:
    return NodeID.inodes()

  @property
  def target(self) -> str:
    return "inode"

  def start_block(self) -> str:
    code = (
      "#include \"../common_decl.h\"\n"
      f"void {self.func_name}() {{\n"
      "  int hid = __builtin_INODE_GET_CORE_HID();\n"
      "  int wid = 0;\n"
      f"{indent(UniqueVar.get_decls_str(), '  ')}\n"
    )
    return code

  def end_block(self) -> str:
    return "}\n"



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

  __builtin_INODE_PU(addr, imm, rs, slv_node_id);

  int a = __builtin_INODE_GET_CORE_HID();
  int b = __builtin_INODE_GET_CORE_WID();
"""
