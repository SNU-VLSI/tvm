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
                          lambda iter, addr=info.policy_info[0].address: f"__builtin_INODE_WR_IMCU({var} + {iter}*32, 0, {addr});"))


class RecvBlock(InodeCodeBlock):
  """ Code block for receiving data from given fifo id """

  def __init__(self, builder, block: DataBlock, fifo_id: int, annotation: str = ""):
    super().__init__(annotation)
    self.builder = builder
    self.block = block
    self.fifo_id = fifo_id
    if self.block.tiling_info is not None:
      self._build_tiled()
    else:
      self._build()

  def _build(self):
    recv_count = math.ceil(self.block.size / 32)
    var = UniqueVar("recv_data_base_address", dtype="int")

    self.body.add(TextBlock(f"{var} = {self.block.offset};"))
    self.body.add(SimpleFor(recv_count,
                      lambda iter: f"__builtin_INODE_RECV({var} + {iter}*32, 0, 0, {self.fifo_id});"))

  def _build_tiled(self):
    fifo_id = self.fifo_id
    if isinstance(self.block.id, list):
      block_id = self.block.id[0]
    else:
      block_id = self.block.id
    assert isinstance(block_id, TensorEdge), "Tiled recv block must be a tensor edge"
    target_edge = block_id

    cnt_addr_var = UniqueVar(f"{target_edge.simple_name()}_recv_data_base_address", dtype="int", pointer_type=True)
    _recv_cnt_address_block = DevConfig().MemLayout[self.builder.func_name].get_data_block_by_id(f"{target_edge.simple_name()}_cnt_base_addr")
    _recv_cnt_address = _recv_cnt_address_block.offset
    self.body.add(TextBlock(f"{cnt_addr_var} = (int*)({_recv_cnt_address});"))

    loop_cnt_var = UniqueVar(f"{target_edge.simple_name()}_tile_loop_count", dtype="int")
    base_var = UniqueVar("recv_data_base_address", dtype="int")
    self.body.add(TextBlock(f"{base_var} = {self.block.offset};"))

    self.body.add(TextBlock(f"{loop_cnt_var} = {cnt_addr_var}[0];"))
    self.body.add(TextBlock(f"__asm__ volatile(\"nop\");")) # BUGFIX_LOAD_USE_HAZARD
    # Unroll 4x with stride of 128 bytes (4 * 32)
    def unrolled_recv_body(iter, base_addr_var=base_var, fid=fifo_id):
      return (f"__builtin_INODE_RECV({base_addr_var} + {iter}*128, 0, 0, {fid});\n"
              f"__builtin_INODE_RECV({base_addr_var} + {iter}*128 + 32, 0, 0, {fid});\n"
              f"__builtin_INODE_RECV({base_addr_var} + {iter}*128 + 64, 0, 0, {fid});\n"
              f"__builtin_INODE_RECV({base_addr_var} + {iter}*128 + 96, 0, 0, {fid});")
    self.body.add(SimpleFor(loop_cnt_var, unrolled_recv_body))

    # Consumer INODE waits for producer IMCE to signal completion
    # Find producer IMCE ID from the tensor edge
    SYNC_OUTPUT_FLAG = 2
    producer_imce_ids = set()

    # Get tensor edge info to find the producer
    if target_edge in DevConfig().TensorEdgetoInfo:
      te_info = DevConfig().TensorEdgetoInfo[target_edge]
      if te_info.policy_info:
        # The first router in policy_info is the source (producer)
        producer_router_id = te_info.policy_info[0].router_id
        if producer_router_id.is_imce():
          producer_imce_ids.add(producer_router_id.value)

    # Add STANDBY calls for each producer IMCE
    for producer_imce_id in sorted(producer_imce_ids):
      # Find producer name for annotation
      producer_name = None
      if target_edge in DevConfig().TensorEdgetoInfo:
        te_info = DevConfig().TensorEdgetoInfo[target_edge]
        if te_info.policy_info and te_info.policy_info[0].router_id.value == producer_imce_id:
          producer_name = te_info.policy_info[0].router_id.name

      if producer_name:
        self.body.add(TextBlock(f"__builtin_INODE_STANDBY({producer_imce_id}, {SYNC_OUTPUT_FLAG}); // {producer_name}"))
      else:
        self.body.add(TextBlock(f"__builtin_INODE_STANDBY({producer_imce_id}, {SYNC_OUTPUT_FLAG});"))


class RecvBlockInterleaved(InodeCodeBlock):
  """ Code block for receiving data from given fifo id interleaved """

  def __init__(self, blocks: List[DataBlock], fifo_ids: List[int], annotation: str = ""):
    super().__init__(annotation)
    assert len(blocks) == len(fifo_ids), "# of blocks and fifo_ids must be equal"
    self.blocks = blocks
    self.fifo_ids = fifo_ids
    self._build()

  def _build(self):
    # Collect block info
    info_list = []
    for block, fifo_id in zip(self.blocks, self.fifo_ids):
      recv_count = math.ceil(block.size / 32)
      info_list.append({
          'recv_count': recv_count,
          'offset': block.offset,
          'fid': fifo_id
      })

    # Sort unique recv_counts to define intervals
    counts = sorted(list(set(x['recv_count'] for x in info_list)))
    
    current_base = 0
    for limit in counts:
      duration = limit - current_base
      if duration <= 0:
        continue
        
      # Identify blocks active in this interval
      active_infos = [x for x in info_list if x['recv_count'] > current_base]
      
      # Generate loop for this interval
      for x in active_infos:
        var = UniqueVar("recv_offset_address", dtype="int")
        self.body.add(TextBlock(f"{var} = {x['offset']};"))
        self.body.add(SimpleFor(duration,
            lambda iter, base=current_base, offset_var=var, fid=x['fid']: 
              f"__builtin_INODE_RECV({offset_var} + ({f'({base} + {iter})' if base > 0 else iter})*32, 0, 0, {fid});"))

      current_base = limit


class SendBlock(InodeCodeBlock):
  """ Code block for sending data from given fifo id """

  def __init__(self, builder, block: DataBlock, edge_info: TensorEdgeInfo, annotation: str = ""):
    super().__init__(annotation)
    self.builder = builder
    self.block = block
    self.edge_info = edge_info
    if self.block.tiling_info is not None:
      self._build_tiled()
    else:
      self._build()

  def _build(self):
    recv_count = math.ceil(self.block.size / 32)
    fifo_id = self.edge_info.fifo_id
    assert fifo_id >= 0, "fifo id should be assigned to a positive id"
    next_policy_addr = self.edge_info.policy_info[0].address

    var = UniqueVar("send_data_base_address", dtype="int")
    self.body.add(TextBlock(f"{var} = {self.block.offset};"))
    self.body.add(SimpleFor(recv_count,
                      lambda iter, var=var, next_policy_addr=next_policy_addr, fifo_id=fifo_id: 
                        f"__builtin_INODE_SEND({var} + {iter}*32, 0, {next_policy_addr}, {fifo_id});"))

  def _build_tiled(self):
    # tiling_info = self.block.tiling_info
    fifo_id = self.edge_info.fifo_id
    assert fifo_id >= 0, "fifo id should be assigned to a positive id"
    next_policy_addr = self.edge_info.policy_info[0].address
    if isinstance(self.block.id, list):
      block_id = self.block.id[0]
    else:
      block_id = self.block.id
    assert isinstance(block_id, TensorEdge), "Tiled send block must be a tensor edge"
    target_edge = block_id

    cnt_addr_var = UniqueVar(f"{target_edge.simple_name()}_send_data_base_address", dtype="int", pointer_type=True)
    _send_cnt_address_block = DevConfig().MemLayout[self.builder.func_name].get_data_block_by_id(f"{target_edge.simple_name()}_cnt_base_addr")
    _send_cnt_address = _send_cnt_address_block.offset
    self.body.add(TextBlock(f"{cnt_addr_var} = (int*)({_send_cnt_address});"))

    loop_cnt_var = UniqueVar(f"{target_edge.simple_name()}_tile_loop_count", dtype="int")
    base_var = UniqueVar("send_data_base_address", dtype="int")
    self.body.add(TextBlock(f"{base_var} = {self.block.offset};"))

    self.body.add(TextBlock(f"{loop_cnt_var} = {cnt_addr_var}[0];"))
    self.body.add(TextBlock(f"__asm__ volatile(\"nop\");")) # BUGFIX_LOAD_USE_HAZARD
    # Unroll 4x with stride of 128 bytes (4 * 32)
    def unrolled_send_body(iter, base_addr_var=base_var, policy_addr=next_policy_addr, fid=fifo_id):
      return (f"__builtin_INODE_SEND({base_addr_var} + {iter}*128, 0, {policy_addr}, {fid});\n"
              f"__builtin_INODE_SEND({base_addr_var} + {iter}*128 + 32, 0, {policy_addr}, {fid});\n"
              f"__builtin_INODE_SEND({base_addr_var} + {iter}*128 + 64, 0, {policy_addr}, {fid});\n"
              f"__builtin_INODE_SEND({base_addr_var} + {iter}*128 + 96, 0, {policy_addr}, {fid});")
    self.body.add(SimpleFor(loop_cnt_var, unrolled_send_body))

class SendBlockInterleaved(InodeCodeBlock):
  """ Code block for sending data from given fifo id """

  def __init__(self, blocks: List[DataBlock], edge_infos: List[TensorEdgeInfo], annotation: str = ""):
    super().__init__(annotation)
    assert len(blocks) == len(edge_infos), "# of blocks and fifo_ids must be equal"
    self.blocks = blocks
    self.edge_infos = edge_infos
    self._build()

  def _build(self):
    # Collect block info
    info_list = []
    for block, edge_info in zip(self.blocks, self.edge_infos):
      recv_count = math.ceil(block.size / 32)
      next_policy_addr = edge_info.policy_info[0].address
      fifo_id = edge_info.fifo_id
      info_list.append({
          'owner': block,
          'recv_count': recv_count,
          'offset': block.offset,
          'policy': next_policy_addr,
          'fid': fifo_id
      })

    # Sort unique recv_counts to define intervals
    counts = sorted(list(set(x['recv_count'] for x in info_list)))
    
    current_base = 0
    for limit in counts:
      duration = limit - current_base
      if duration <= 0:
        continue
        
      # Identify blocks active in this interval
      active_infos = [x for x in info_list if x['recv_count'] > current_base]
      
      # Generate loop for this interval
      for x in active_infos:
        # var = UniqueVar("send_offset_address", dtype="int")
        var = UniqueVar(x['owner'], dtype="int")
        self.body.add(TextBlock(f"{var} = {x['offset']};"))
        self.body.add(SimpleFor(duration,
            lambda iter, base=current_base, offset_var=var, policy=x['policy'], fid=x['fid']: 
              f"__builtin_INODE_SEND({offset_var} + ({f'{base} + {iter}' if base > 0 else iter})*32, 0, {policy}, {fid});"))

      current_base = limit


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

    nops = " ".join([f"\"nop\\n\"" for _ in range(len(self.node_ids))])
    self.body.add(TextBlock(f"__asm__ volatile({nops});"))

    self.body.add(TextBlock(f"__builtin_INODE_DONE();"))
    self.body.add(TextBlock(f"__builtin_INODE_INTRT(0);"))
    self.body.add(TextBlock(f"__builtin_INODE_HALT();"))

class DoneAndIntrtBlock(InodeCodeBlock):
  def __init__(self, annotation: str = ""):
    super().__init__(annotation)
    self._build()

  def _build(self):
    self.body.add(TextBlock(f"__builtin_INODE_DONE();"))
    self.body.add(TextBlock(f"__builtin_INODE_INTRT(0);"))

class SyncAllINodes(InodeCodeBlock):
  def __init__(self, node_id : NodeID, annotation: str = ""):
    super().__init__(annotation)
    self.node_id = node_id
    self._build()

  def _build(self):
    self.body.add(TextBlock(f"__builtin_INODE_SET_FLAG(1);"))
    for node in NodeID.inodes():
      if node != self.node_id:
        self.body.add(TextBlock(f"__builtin_INODE_STANDBY({node.value}, 1);"))

    nops = " ".join([f"\"nop\\n\"" for _ in range(len(NodeID.inodes()))])
    self.body.add(TextBlock(f"__asm__ volatile({nops});"))
    self.body.add(TextBlock(f"__builtin_INODE_SET_FLAG(0);"))

class Standby(InodeCodeBlock):
  def __init__(self, node_ids: List[NodeID], annotation: str = ""):
    super().__init__(annotation)
    self.node_ids = node_ids
    self._build()

  def _build(self):
    for node in self.node_ids:
      self.body.add(TextBlock(f"__builtin_INODE_STANDBY({node.value}, 1);"))

    nops = " ".join([f"\"nop\\n\"" for _ in range(len(self.node_ids))])
    self.body.add(TextBlock(f"__asm__ volatile({nops});"))


class SetFlagAndHaltBlock(InodeCodeBlock):
  def __init__(self, annotation: str = ""):
    super().__init__(annotation)
    self._build()

  def _build(self):
    self.body.add(TextBlock(f"__builtin_INODE_SET_FLAG(1);"))
    self.body.add(TextBlock(f"__builtin_INODE_HALT();"))

class HaltBlock(InodeCodeBlock):
  def __init__(self, annotation: str = ""):
    super().__init__(annotation)
    self._build()

  def _build(self):
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
    UniqueVar.reset()
    self.blocks = {key: {CodePhase.INIT: [], CodePhase.EXEC: [], CodePhase.EXEC_TILE: [], CodePhase.END: []}
                   for key in self.nodes}
    self.func_name = func_name

  @property
  def nodes(self) -> List[NodeID]:
    return NodeID.inodes()

  @property
  def target(self) -> str:
    return "inode"

  def render_phase(self, node: NodeID, phase: CodePhase) -> str:
    blocks = self.blocks[node][phase]
    if len(blocks) == 0:
      return ""

    if phase is CodePhase.EXEC_TILE:
      # Tiling phase: wrap in for loop
      seq_block = SequentialBlock()
      for codeblock in blocks:
        seq_block.add(codeblock)
      code = f"{indent(SimpleFor(DevConfig().ImcflowFuncMap[self.func_name].tiling_factor, seq_block).render(), '  ')}\n"
    else:
      # None tiling phase: just render
      code = ""
      for codeblock in blocks:
        code += f"{indent(codeblock.render(), '  ')}\n"

    return code

  def generate_body(self) -> str:
    code = ""
    first = True
    for node in self.nodes:
      condition = f"if" if first else f"else if"
      code += f"{condition} (hid == {node.to_coord(0)} && wid == {node.to_coord(1)}) {{ // {node.name}\n"
      code += self.render_phase(node, CodePhase.INIT)
      code += self.render_phase(node, CodePhase.EXEC)
      code += self.render_phase(node, CodePhase.EXEC_TILE)
      code += self.render_phase(node, CodePhase.END)
      code += "}\n"
      first = False
    return code

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
