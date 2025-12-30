# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from typing import Tuple, List, Dict, Union
from enum import Enum
from collections import defaultdict, UserDict
import copy

import re
import math

SMALL_DEBUG = 0


class NodeID(Enum):
  inode_0_0 = 0
  imce_0_1 = 1
  imce_0_2 = 2
  imce_0_3 = 3
  imce_0_4 = 4
  inode_1_0 = 5
  imce_1_1 = 6
  imce_1_2 = 7
  imce_1_3 = 8
  imce_1_4 = 9
  inode_2_0 = 10
  imce_2_1 = 11
  imce_2_2 = 12
  imce_2_3 = 13
  imce_2_4 = 14
  inode_3_0 = 15
  imce_3_1 = 16
  imce_3_2 = 17
  imce_3_3 = 18
  imce_3_4 = 19

  @staticmethod
  def from_coord(x: int, y: int) -> 'NodeID':
    """Returns the NodeID corresponding to a 2D coordinate."""
    value = x * ImcflowDeviceConfig.NODE_COL_NUM + y
    for node in NodeID:
      if node.value == value:
        return node
    raise ValueError(f"No Node found for coordinate ({x}, {y})")

  @staticmethod
  def from_inode_coord(x: int) -> 'NodeID':
    assert x >= 0 and x < ImcflowDeviceConfig.INODE_NUM, "inode coord is out of range"
    return NodeID(ImcflowDeviceConfig.NODE_COL_NUM*x)

  @staticmethod
  def from_imce_coord(x: int, y: Union[None | int] = None) -> 'NodeID':
    if y is None:
      ImceHeight = x//ImcflowDeviceConfig.IMCE_W_NUM
      ImceWidth = x % ImcflowDeviceConfig.IMCE_W_NUM
      return NodeID(ImcflowDeviceConfig.NODE_COL_NUM*ImceHeight + (ImceWidth+1))
    else:
      return NodeID(ImcflowDeviceConfig.NODE_COL_NUM*x + (y+1))

  @staticmethod
  def inodes() -> List['NodeID']:
    """Returns a list of all inode nodes."""
    return [node for node in NodeID if node.is_inode()]

  @staticmethod
  def imces() -> List['NodeID']:
    """Returns a list of all imce nodes."""
    return [node for node in NodeID if node.is_imce()]

  def is_inode(self) -> bool:
    return self.value % ImcflowDeviceConfig.NODE_COL_NUM == 0

  def is_imce(self) -> bool:
    return not self.is_inode()

  def to_coord(self, *args) -> Union[tuple, int]:
    """Converts this node to its 2D coordinate."""
    coord = divmod(self.value, ImcflowDeviceConfig.NODE_COL_NUM)
    if len(args) == 1 and args[0] == 0:
      return coord[0]
    elif len(args) == 1 and args[0] == 1:
      return coord[1]
    elif len(args) == 0:
      return coord
    else:
      raise ValueError("Invalid number of arguments")

  def slaves(self) -> List['NodeID']:
    """Returns a list of imces that are slaved to this inode."""
    assert self.is_inode(), "Only inode nodes have slaves"
    return [NodeID(self.value + i) for i in range(1, ImcflowDeviceConfig.NODE_COL_NUM)]

  def master(self) -> 'NodeID':
    """Returns the inode that is master to this imce."""
    assert self.is_imce(), "Only imce nodes have master"
    return NodeID(self.value - self.value % ImcflowDeviceConfig.NODE_COL_NUM)


class TensorID:
  _instances = {}

  def __new__(cls, graph_node_id: Union[int, Tuple], tensor_type: str):
    key = (graph_node_id, tensor_type)
    if tensor_type not in {"data", "odata", "weight",
                           "bias", "fused_scale", "fused_bias", "lhs", "rhs",
                           "min", "max", "threshold", "zero","config", "var", "func_out"}:
                           print("Invalid tensor type")
    if key not in cls._instances:
      instance = super(TensorID, cls).__new__(cls)
      cls._instances[(graph_node_id, tensor_type)] = instance
      instance.graph_node_id = graph_node_id
      instance.tensor_type = tensor_type

    return cls._instances[key]

  def inner_gid_match(self, graph_node_id: Union[int, Tuple]):
    import tvm
    if isinstance(self.graph_node_id, (int, tvm.tir.expr.IntImm)):
      return self.graph_node_id == graph_node_id
    if isinstance(self.graph_node_id, tuple):
      return graph_node_id == self.graph_node_id[1]
    print("Error in inner_gid_match")
    return False

  def __str__(self):
    return f"TensorID({self.graph_node_id}, {self.tensor_type})"

  def __repr__(self):
    return self.__str__()


class TensorEdge:
  _instances = {}

  def __new__(cls, src_id: TensorID, dst_id: TensorID, split_idx: Union[None, int] = None):
    key = (src_id, dst_id, split_idx)
    if key not in cls._instances:
      instance = super(TensorEdge, cls).__new__(cls)
      cls._instances[(src_id, dst_id, split_idx)] = instance
      instance.src_id = src_id
      instance.dst_id = dst_id
      instance.split_idx = split_idx

    return cls._instances[key]

  def src_inner_gid_match(self, graph_node_id: Union[int, Tuple]):
    return self.src_id.inner_gid_match(graph_node_id)

  def dst_inner_gid_match(self, graph_node_id: Union[int, Tuple]):
    return self.dst_id.inner_gid_match(graph_node_id)
  
  def simple_name(self):
    return ""

  def __str__(self):
    if self.split_idx is None:
      return f"TensorEdge(({self.src_id.graph_node_id}, {self.src_id.tensor_type}), ({self.dst_id.graph_node_id}, {self.dst_id.tensor_type}))"
    else:
      return f"TensorEdge(({self.src_id.graph_node_id}, {self.src_id.tensor_type}), ({self.dst_id.graph_node_id}, {self.dst_id.tensor_type}), {self.split_idx})"

  def __repr__(self):
    return self.__str__()


class FunctionInfo:
  """Information about an IMCFlow function"""
  def __init__(self, func_node, tiling_factor=1):
    self.func_node = func_node          # relay.Function object
    self.tiling_factor = tiling_factor  # int: tiling factor for memory optimization
    self.const_name_map = {} # node id : name

  def __repr__(self):
    return f"FunctionInfo(tiling_factor={self.tiling_factor})"

class BlockTileInfo:
  """Tiling information for a data block"""
  def __init__(self):
    self.height_base_coords  = [] # for tiling
    self.height_sizes        = [] # for tiling
    self.pkt_cnts            = [] # for tiling
    self.c_input_var_offsets = [] # for tiling
    self.c_input_var_sizes   = [] # for tiling


class DataBlock:
  def __init__(self, id: Union[str, TensorEdge, List[TensorEdge]], size: int):
    """
    id can be either:
      str, used for "policy", "imem", etc
      TensorEdge, corresponding to a single edge (one dst)
      List[TensorEdge], corresponding to multiple edges (1< dst)
    kept internally as edges
    """
    self.edges = [id] if not isinstance(id, List) else id
    self.size = size
    self.offset = -1  # offset in the region
    self.base_address = -1  # base address in the device memory
    self.tiling_info = None
  
  @property
  def id(self):
    if len(self.edges) == 1:
      return self.edges[0]
    else:
      return self.edges

  def set_size(self, size: int):
    if isinstance(size, float):
      assert size % 1 == 0, "Size should be an integer"
    self.size = int(size)

  def set_offset(self, offset: int):
    if isinstance(offset, float):
      assert offset % 1 == 0, "Offset should be an integer"
    self.offset = int(offset)

  def set_base_address(self, address: int):
    if isinstance(address, float):
      assert address % 1 == 0, "Address should be an integer"
    self.base_address = int(address)

  def __str__(self):
    return (f"DataBlock({self.id}, {self.size}, {self.base_address})")

  def __repr__(self):
    return self.__str__()


class MemoryRegionEntry:
  def __init__(self, name: str, size: int):
    self.name = name
    self.size = size
    self.blocks = {}  # {data_block_name : DataBlock}
    self.base_address = -1  # offset in the device memory
    self._last_offset = 0  # last offset in the region

  def get_data_block_by_edge(self, edge):
    """
    Gets data_block by edge.
    """
    for block in self.blocks.values():
      if edge in block.edges:
        return block
    return None
  
  def _already_exists(self, block: DataBlock):
    if block.id in [x.id for x in self.blocks.values()]:
      print(f"Trying allocate {block} but skipped (already exists)")
      return True

    # extend existing Datablock's id when same src_id found (same data, different destinations)
    if isinstance(block.id, TensorEdge) and block.id.split_idx is None:
      for existing_block in self.blocks.values():
        # Check if any edge has the same src_id
        for edge in existing_block.edges:
          if isinstance(edge, TensorEdge) and edge.src_id == block.id.src_id:
            existing_block.edges.append(block.id)
            print(f"Extended existing block with {block.id}, now block.id = {existing_block.id}")
            return True
    return False

  def allocate(self, block: DataBlock) -> bool:
    """
    Allocate a data block in the region sequentially, assuming they are not deallocated
    returns True (when allocated, or already allocated) or False (exceeds region size)
    """
    if self._already_exists(block):
      return True

    print(f"Trying allocate {block}")
    # find first 32B aligned free offset
    aligned_offset = math.ceil((self.base_address + self._last_offset) / 32) * 32 - self.base_address
    if block.size + aligned_offset <= self.size:
      block.set_offset(aligned_offset)
      block.set_base_address(aligned_offset + self.base_address)
      self._last_offset = aligned_offset + block.size
      self.blocks[block.id] = block
      return True
    else:
      print(f"Data block size exceeds region size. {block.size} + {aligned_offset} > {self.size}")
      return False

  def set_base_address(self, address: int):
    self.base_address = int(address)

  def __str__(self):
    if not self.blocks:
      return f"MemoryRegionEntry({self.name}, {self.size}, {self.base_address}, blocks=[])"
    blocks_str = ",\n      ".join(str(block) for block in self.blocks.values())
    return (f"MemoryRegionEntry({self.name}, {self.size}, {self.base_address}, "
            f"blocks=[\n      {blocks_str}\n    ])")

  def __repr__(self):
    return self.__str__()


class MemoryRegion(UserDict):
  """
    MemoryRegion is a dict of MemoryRegionEntry accessible using key (idx).
    Automatically creates new MemoryRegionEntry instances from the template when accessing non-existent keys.
  """
  def __init__(self, region: MemoryRegionEntry):
    super().__init__()
    self._template_region = region

  @property
  def blocks(self):
    """Aggregate all blocks from all entries. Returns dict {block_id: DataBlock}"""
    aggregated = {}
    for idx, region in self.data.items():
      if region.blocks:  # Only include entries with blocks
        aggregated.update(region.blocks)
    return aggregated
  
  def allocate(self, block: DataBlock, phase: str):
    """Allocate block by finding first region of the phase with available space."""
    i = 0
    while not self[(phase, i)].allocate(block):
      i += 1


  def __getitem__(self, key) -> 'MemoryRegionEntry':
    """auto-create from template"""
    if key not in self.data:
      return self.__missing__(key)
    return self.data[key]

  def __missing__(self, key) -> 'MemoryRegionEntry':
    """
    Called when a key is not found. Creates a new MemoryRegionEntry from the template.
    """
    # Create a deep copy of the template region
    new_region = copy.deepcopy(self._template_region)
    self[key] = new_region
    return new_region

  def __str__(self):
    if not self.data:
      return f"MemoryRegion('{self._template_region.name}', empty)"

    lines = [f"MemoryRegion('{self._template_region.name}', ["]
    for idx, region in self.data.items():
      lines.append(f"  {idx}: {region},")
    lines.append("])")
    return "\n".join(lines)

  def __repr__(self):
    return self.__str__()



class FuncMemoryLayout(UserDict):
  def __init__(self, *regions: MemoryRegionEntry):
    super().__init__()
    _last_end_address = 0

    for region in regions:
      self.data[region.name] = MemoryRegion(region)
      region.set_base_address(_last_end_address)
      _last_end_address += region.size

  @property
  def blocks(self):
    """Aggregate all blocks from all regions and entries. Returns dict {block_id: DataBlock}"""
    aggregated = {}
    for region_name, region_dict in self.data.items():
      region_blocks = region_dict.blocks
      if region_blocks:  # Only include regions with blocks
        aggregated.update(region_blocks)
    return aggregated

  def __getitem__(self, key: str) -> 'MemoryRegion':
    """Get a memory region by name (e.g., 'inode_0_0_data')"""
    return self.data[key]

  def get_data_block_by_edge(self, edge: Union[str, TensorEdge]):
    """
    Gets data_block by id from aggregated blocks
    """
    for block in self.blocks.values():
      if edge in block.edges:
        return block
    return None

  def __str__(self):
    regions_str = ",\n  ".join(str(region) for region in self.data.values())
    return f"FuncMemoryLayout(regions=[\n  {regions_str}\n])"

  def __repr__(self):
    return self.__str__()

class MemoryLayout(UserDict):
  """
    MemoryLayout is a dict of FuncMemoryLayout accessible using key (func_name).
    Automatically creates new FuncMemoryLayout instances from the template when accessing non-existent keys.
  """
  def __init__(self, *regions: MemoryRegionEntry):
    super().__init__()
    self._layout = FuncMemoryLayout(*regions)

  @property
  def blocks(self):
    """Aggregate all blocks from all functions. Returns dict {func_name: {region_name: {idx: {block_id: DataBlock}}}}"""
    aggregated = {}
    for func_name, layout in self.data.items():
      layout_blocks = layout.blocks
      if layout_blocks:  # Only include functions with blocks
        aggregated[func_name] = layout_blocks
    return aggregated

  def __getitem__(self, key: str) -> 'FuncMemoryLayout':
    """auto-create from template"""
    if key not in self.data:
      return self.__missing__(key)
    return self.data[key]

  def __missing__(self, key: str) -> 'FuncMemoryLayout':
    """
    Called when a key is not found. Creates a new FuncMemoryLayout from the template.
    """
    # Create a deep copy of the template layout
    new_layout = copy.deepcopy(self._layout)
    self[key] = new_layout
    return new_layout

  def __str__(self):
    if not self.data:
      return "MemoryLayout(empty)"

    lines = ["MemoryLayout("]
    for func_name, layout in self.data.items():
      lines.append(f"  {func_name!r}: {layout},")
    lines.append(")")
    return "\n".join(lines)

  def __repr__(self):
    return self.__str__()


class RouterEntry:
  def __init__(self, router_id: int, address: int, data: Dict):
    self.router_id = router_id
    self.address = address
    self.data = data

  def __str__(self):
    return f"RouterEntry({self.router_id}, {self.address}, {self.data})"

  def __repr__(self):
    return self.__str__()

  def __eq__(self, other):
    is_equal = self.router_id == other.router_id and self.address == other.address
    if is_equal:
      assert self.data == other.data, f"Data mismatch: {self.data} != {other.data} in same router entry"
    return is_equal


class EdgeInfo:
  """ stores the list of router entries and memory block info for a data movement (edge). """

  def __init__(self, policy_info: List[RouterEntry], data_block: Union[DataBlock, None] = None):
    self.policy_info = policy_info
    self.data_block = data_block

  def set_policy_info(self, policy_info: List[RouterEntry]):
    self.policy_info = policy_info

  def append_policy_info(self, entry: RouterEntry):
    self.policy_info.append(entry)

  def set_data_block(self, data_block: DataBlock):
    self.data_block = data_block


class InstEdgeInfo(EdgeInfo):
  def __str__(self):
    policy_info_str = ", ".join(str(entry) for entry in self.policy_info)
    return f"InstEdgeInfo([{policy_info_str}], {self.data_block})"

  def __repr__(self):
    return self.__str__()


class TensorEdgeInfo(EdgeInfo):
  LOCAL_FIFO = -2
  def __init__(self, policy_info: List[RouterEntry] = None, data_block: Union[DataBlock, None] = None, fifo_id: int = -1):
    super().__init__(policy_info, data_block)
    self.fifo_id = fifo_id
    self.owner = None

  def set_fifo_id(self, fifo_id):
    self.fifo_id = fifo_id

  @property
  def node_info_str(self):
    return f"{self.policy_info[0].router_id.name} -> {self.policy_info[-1].router_id.name}"
  
  def get_height_base_coords(self):
    return self.height_base_coords
  
  def get_height_sizes(self):
    return self.height_sizes
  
  def get_pkt_cnts(self):
    return self.pkt_cnts

  def __str__(self):
    policy_info_str = ", ".join(str(entry) for entry in self.policy_info) if self.policy_info else "[]"
    return f"TensorEdgeInfo([{policy_info_str}], {self.data_block}, {self.fifo_id})"

  def __repr__(self):
    return self.__str__()


class CodegenContext:
  """Singleton context for codegen that tracks the current function being processed"""

  def __new__(cls):
    if not hasattr(cls, "instance"):
      cls.instance = super(CodegenContext, cls).__new__(cls)
      cls.instance._initialize()
    return cls.instance

  def _initialize(self):
    self._func_name = None

  def set_func_name(self, func_name: str):
    """Set the current function name being processed"""
    self._func_name = func_name

  def get_func_name(self) -> str:
    """Get the current function name"""
    if self._func_name is None:
      raise RuntimeError("No function context set. Call set_func_name() first in codegen.")
    return self._func_name

  @property
  def func_name(self) -> str:
    """Property access to current function name"""
    return self.get_func_name()

  def clear(self):
    """Clear the current function context"""
    self._func_name = None


class ImcflowDeviceConfig:
  """Imcflow config class"""
  if SMALL_DEBUG:
    NODE_COL_NUM = 3
    INODE_NUM = 4
    IMCE_H_NUM = 4
    IMCE_W_NUM = 2
    IMCE_NUM = 8
  else:
    NODE_COL_NUM = 5
    INODE_NUM = 4
    IMCE_H_NUM = 4
    IMCE_W_NUM = 4
    IMCE_NUM = 16

  IMCU_ROW_NUM = 256
  INODE_MMREG_SIZE = 128
  INODE_DATA_MEM_SIZE = 65536
  # INODE_DATA_MEM_SIZE = 131072
  INODE_INST_MEM_SIZE = 1024
  IMCE_INST_MEM_SIZE = 1024
  IMCFLOW_ADDR_SIZE = 266368 # 128 + 4*(65536+1024) == 260.125KB
  HOST_OS = "baremetal"
  HOST_ISA = "x86"

  INODE0_IMEM_BASE_ADDR = 128
  INODE0_DMEM_BASE_ADDR = INODE0_IMEM_BASE_ADDR + INODE_INST_MEM_SIZE

  INODE1_IMEM_BASE_ADDR = INODE0_DMEM_BASE_ADDR + INODE_DATA_MEM_SIZE
  INODE1_DMEM_BASE_ADDR = INODE1_IMEM_BASE_ADDR + INODE_INST_MEM_SIZE

  INODE2_IMEM_BASE_ADDR = INODE1_DMEM_BASE_ADDR + INODE_DATA_MEM_SIZE
  INODE2_DMEM_BASE_ADDR = INODE2_IMEM_BASE_ADDR + INODE_INST_MEM_SIZE

  INODE3_IMEM_BASE_ADDR = INODE2_DMEM_BASE_ADDR + INODE_DATA_MEM_SIZE # 0x30c80
  INODE3_DMEM_BASE_ADDR = INODE3_IMEM_BASE_ADDR + INODE_INST_MEM_SIZE

  SUPPORTED_OPS = ["nn.imcflow_qconv", "nn.imcflow_qdwconv", "nn.bias_add", "imcflow.fused_batch_norm",
                   "nn.relu", "add", "split", "concatenate", "qnn.imcflow_min_max_quantize",
                   "qnn.imcflow_nu_quantize", "divide", "imcflow_packing", "imcflow_unpacking",
                   "nn.conv2d", "nn.batch_norm","multiply"]
  NO_COST_OPS = ["split", "concatenate", "imcflow_packing", "imcflow_unpacking"]
  QAUNT_OPS = ["qnn.imcflow_min_max_quantize", "qnn.imcflow_nu_quantize"]

  def __new__(cls):
    if not hasattr(cls, "instance"):
      cls.instance = super(ImcflowDeviceConfig, cls).__new__(cls)
      cls.instance._initialize()
    return cls.instance

  def _initialize(self):
    self.HWNodeMap = {}
    self.TensorIDtoEdge = {}
    self.TensorEdgetoInfo = {}
    self.TensorEdgeList = []
    self.TensorEdgeListDict = {}
    self.PolicyTableDict = {}
    self.InstEdgeInfoDict = {}
    self.MemLayout = MemoryLayout(
        MemoryRegionEntry("state_regs", ImcflowDeviceConfig.INODE_MMREG_SIZE),
        MemoryRegionEntry("inode_0_0_inst", ImcflowDeviceConfig.INODE_INST_MEM_SIZE),
        MemoryRegionEntry("inode_0_0_data", ImcflowDeviceConfig.INODE_DATA_MEM_SIZE),
        MemoryRegionEntry("inode_1_0_inst", ImcflowDeviceConfig.INODE_INST_MEM_SIZE),
        MemoryRegionEntry("inode_1_0_data", ImcflowDeviceConfig.INODE_DATA_MEM_SIZE),
        MemoryRegionEntry("inode_2_0_inst", ImcflowDeviceConfig.INODE_INST_MEM_SIZE),
        MemoryRegionEntry("inode_2_0_data", ImcflowDeviceConfig.INODE_DATA_MEM_SIZE),
        MemoryRegionEntry("inode_3_0_inst", ImcflowDeviceConfig.INODE_INST_MEM_SIZE),
        MemoryRegionEntry("inode_3_0_data", ImcflowDeviceConfig.INODE_DATA_MEM_SIZE),
    )
    self.ActiveIMCEPerFunc = {}
    self.NoCPaths = {}
    self.DataBlocks = {}
    self.ImcflowFuncMap = {}  # {func_name: FunctionInfo}
    self.use_def_chain = {}   # {func_name: use-def mapping}
    self.LayoutMap={}
    self.FIFOConflictTable = {}
    self.NoCDeadlockTable = {}

  def clear(self):
    self._initialize()

  @ staticmethod
  def is_supported_kernel(KH, KW):
    return (KH, KW) in {(1, 1), (3, 3), (5, 5), (7, 7)}

  def add_hw_node(self, graph_node_id: Union[int, Tuple], hwnode_id: int):
    self.HWNodeMap[graph_node_id] = hwnode_id

  def get_hw_node(self, graph_node_id: Union[int, Tuple]):
    return self.HWNodeMap.get(graph_node_id, None)

  def add_tensor_edge(self, tensor_id: TensorID, tensor_edge: TensorEdge):
    self.TensorIDtoEdge[tensor_id] = tensor_edge

  def get_tensor_edge(self, tensor_id: TensorID):
    return self.TensorIDtoEdge.get(tensor_id, None)

  def add_tensor_edge_info(self, tensor_edge: TensorEdge, tensor_edge_info: TensorEdgeInfo):
    tensor_edge_info.owner = tensor_edge
    self.TensorEdgetoInfo[tensor_edge] = tensor_edge_info

  def get_tensor_edge_info(self, tensor_edge: TensorEdge):
    return self.TensorEdgetoInfo.get(tensor_edge, None)

  def get_tensor_edge_info_with_id_dir(self, tensor_id: TensorID, dir: str) -> List[TensorEdgeInfo]:
    edge_infos = []
    if dir == "in":
      for edge in self.TensorEdgetoInfo.keys():
        if edge.dst_id == tensor_id:
          edge_infos.append(self.TensorEdgetoInfo[edge])
      return edge_infos
    elif dir == "out":
      for edge in self.TensorEdgetoInfo.keys():
        if edge.src_id == tensor_id:
          edge_infos.append(self.TensorEdgetoInfo[edge])
      return edge_infos
    else:
      raise ValueError("Invalid direction")

  def get_tensor_ids_from_graph_node_id(self, graph_node_id: Union[int, Tuple]):
    tids = []
    for tid in self.TensorIDtoEdge.keys():
      if tid.graph_node_id == graph_node_id:
        tids.append(tid)
    return tids

  def get_tensor_edges_from_graph_node_id(self, graph_node_id: Union[int, Tuple]):
    for tid in self.get_tensor_edges_from_graph_node_id(graph_node_id):
      yield self.get_tensor_edge(tid)

  def add_inst_edge_info(self, func_name: str, imce_id: NodeID, inst_edge_info: InstEdgeInfo):
    assert imce_id.is_imce(), "Only imce nodes have inst edge info"
    self.InstEdgeInfoDict.setdefault(func_name, {})[imce_id] = inst_edge_info

  def get_inst_edge_info(self, func_name: str, imce_id: NodeID):
    assert imce_id.is_imce(), "Only imce nodes have inst edge info"
    return self.InstEdgeInfoDict.get(func_name, {}).get(imce_id, None)

  @property
  def CurrFuncMemLayout(self):
    """
    Get the memory layout for the current function in CodegenContext.

    This is a convenience property that allows accessing the current function's
    memory layout without explicitly passing func_name around:
    """
    return self.MemLayout[CodegenContext().func_name]

  def format_policy_table(self):
    """
    Format the PolicyTableDict into a hierarchical string representation.

    Returns:
      str: Formatted policy table with function names, node IDs, and compact policy entries.

    Example format:
      tvmgen_default_tvmgen_default_imcflow_main_47_round_imcflow_region1_main_0
        <NodeID.inode_0_0: 0>
          0:  L:(0, 0, 0), N:(1, 31), E:(0, 50), S:(1, 61), W:(0, 37)
          1:  L:(0, 0, 0), N:(1, 31), E:(0, 50), S:(1, 61), W:(0, 37)
    """
    lines = []

    for func_name, nodes in self.PolicyTableDict.items():
      lines.append(f"{func_name}")

      for node_id, policies in nodes.items():
        lines.append(f"  {node_id}")

        for idx, policy in enumerate(policies):
          # Extract direction info
          local = policy.get('Local', {})
          north = policy.get('North', {})
          east = policy.get('East', {})
          south = policy.get('South', {})
          west = policy.get('West', {})

          # Format Local: (enable, addr, chunk_index)
          local_str = ""
          if local:
            enable = int(local.get('enable', False))
            addr = local.get('addr', 0)
            chunk_index = local.get('chunk_index', 0)
            ksel = local.get('ksel', 0)
            local_str = f"L:({enable}, {addr}, {ksel}, {chunk_index})"

          # Format other directions: (enable, addr)
          def format_dir(dir_dict, name):
            if dir_dict:
              enable = int(dir_dict.get('enable', False))
              addr = dir_dict.get('addr', 0)
              return f"{name}:({enable}, {addr})"
            return None

          parts = []
          if local_str:
            parts.append(local_str)
          for dir_data, dir_name in [(north, 'N'), (east, 'E'), (south, 'S'), (west, 'W')]:
            formatted = format_dir(dir_data, dir_name)
            if formatted:
              parts.append(formatted)

          lines.append(f"    {idx}:  {', '.join(parts)}")

      lines.append("")  # Empty line between functions

    return "\n".join(lines)

  def get_data_block_dict(self, func, func_name, input_node_ids=None, output_node_id=None, const_node_ids=None):
    from tvm.relay.backend.contrib.imcflow import transform as imcflow_transform
    compiled_blocks, input_data_blocks, output_data_blocks, const_data_blocks = [], [], [], []

    for memory_region in ImcflowDeviceConfig().MemLayout[func_name].values():
      for block_name, block in memory_region.blocks.items():
        # get compiled data blocks
        if isinstance(block_name, str):
          compiled_blocks.append(block)
        # get input & output data blocks
        if isinstance(block_name, TensorEdge):
          if isinstance(block_name.src_id.graph_node_id, Tuple):
            src_gid = block_name.src_id.graph_node_id[1]
          else:
            src_gid = block_name.src_id.graph_node_id

          is_input_block = False
          is_const_block = False
          is_output_block = False
          # get input data blocks
          if any([input_node_id == imcflow_transform.getInnerNodeID(src_gid) for input_node_id in input_node_ids]):
            input_data_blocks.append(block)
            is_input_block = True

          # get const data blocks
          if any([const_node_id == imcflow_transform.getInnerNodeID(src_gid) for const_node_id in const_node_ids]):
            const_data_blocks.append(block)
            is_const_block = True

          # get output data blocks
          if isinstance(block_name.dst_id.graph_node_id, Tuple):
            dst_gid = block_name.dst_id.graph_node_id[1]
          else:
            dst_gid = block_name.dst_id.graph_node_id
          if output_node_id == imcflow_transform.getInnerNodeID(dst_gid):
            output_data_blocks.append(block)
            is_output_block = True

          if sum([is_input_block, is_const_block, is_output_block]) == 0:
            print(f"Warning: DataBlock {block} is neither input, output, nor const block for function {func_name}")
            # TODO: add the exception again after dealing with is_input_block with split node
            # raise ValueError("DataBlock type identification error")
          elif sum([is_input_block, is_const_block, is_output_block]) > 1:
            print(f"Warning: DataBlock {block} is multiple types of blocks for function {func_name}")
            raise ValueError("DataBlock type identification error")


    self.DataBlocks[func_name] = {
        "compiled": compiled_blocks,
        "input": input_data_blocks,
        "output": output_data_blocks,
        "const" : const_data_blocks
    }
