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
from collections import defaultdict

import re
import math

SMALL_DEBUG = 0


class NodeID(Enum):
  inode_0 = 0
  imce_0 = 1
  imce_1 = 2
  imce_2 = 3
  imce_3 = 4
  inode_1 = 5
  imce_4 = 6
  imce_5 = 7
  imce_6 = 8
  imce_7 = 9
  inode_2 = 10
  imce_8 = 11
  imce_9 = 12
  imce_10 = 13
  imce_11 = 14
  inode_3 = 15
  imce_12 = 16
  imce_13 = 17
  imce_14 = 18
  imce_15 = 19

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
                           "min", "max", "threshold", "zero","config"}:
                           print("Invalid tensor type")
    if key not in cls._instances:
      instance = super(TensorID, cls).__new__(cls)
      cls._instances[(graph_node_id, tensor_type)] = instance
      instance.graph_node_id = graph_node_id
      instance.tensor_type = tensor_type

    return cls._instances[key]

  def inner_gid_match(self, graph_node_id: Union[int, Tuple]):
    if isinstance(self.graph_node_id, int):
      return self.graph_node_id == graph_node_id
    if isinstance(self.graph_node_id, tuple):
      return graph_node_id == self.graph_node_id[1]

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

  def __str__(self):
    if self.split_idx is None:
      return f"TensorEdge(({self.src_id.graph_node_id}, {self.src_id.tensor_type}), ({self.dst_id.graph_node_id}, {self.dst_id.tensor_type}))"
    else:
      return f"TensorEdge(({self.src_id.graph_node_id}, {self.src_id.tensor_type}), ({self.dst_id.graph_node_id}, {self.dst_id.tensor_type}), {self.split_idx})"

  def __repr__(self):
    return self.__str__()


class DataBlock:
  def __init__(self, id: Union[str, TensorID], size: int):
    self.id = id
    self.size = size
    self.offset = -1  # offset in the region
    self.base_address = -1  # base address in the device memory

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


class MemoryRegion:
  def __init__(self, name: str, size: int):
    self.name = name
    self.size = size
    self.blocks = {} # {function : {data_block_name : DataBlock}}
    self.base_address = -1  # offset in the device memory
    self._last_offset = defaultdict(int)  # {function_name : last_offset}
    self.weight_offset = defaultdict(int)  # {function_name : weight_offset}
    self.weight_allocated = defaultdict(bool)  # {function_name : weight_allocated}

  def __getitem__(self, id: Union[str, TensorID]):
    return self.blocks.get(id, None)

  def allocate(self, function_name, block: DataBlock):
    """Allocate a data block in the region sequentially, assuming they are not delocated"""
    if function_name not in self.blocks:
      self.blocks[function_name] = {}

    # find first 32B aligned free offset
    if block.id in [x.id for x in self.blocks[function_name].values()]:
      print(f"Trying allocate {block} but skipped @ {function_name}")
      return

    print(f"Trying allocate {block} @ {function_name}")
    aligned_offset = math.ceil((self.base_address + self._last_offset[function_name]) / 32) * 32 - self.base_address
    try:
      assert block.size + aligned_offset <= self.size
    except:
      print(f"Data block size exceeds region size. {block.size} + {aligned_offset} > {self.size}")
      print(self)
      exit(0)
    block.set_offset(aligned_offset)
    block.set_base_address(aligned_offset + self.base_address)
    self._last_offset[function_name] = aligned_offset + block.size
    self.blocks[function_name][block.id] = block

  def allocate_allow_overlap(self, function_name, block: DataBlock):
    """Allocate a data block in the region, allowing overlapping in case of weight params."""
    if function_name not in self.blocks:
      self.blocks[function_name] = {}

    if block.id in [x.id for x in self.blocks[function_name].values()]:
      print(f"Trying allocate_overlap {block} but skipped @ {function_name}")
      return

    print(f"Trying allocate_overlap {block}")
    if self.weight_allocated[function_name] is False:
      self.weight_allocated[function_name] = True
      # Align weight_offset to 32B boundary
      self.weight_offset[function_name] = math.ceil((self.base_address) / 32) * 32 - self.base_address
    
    # Align current weight_offset to 32B boundary
    aligned_offset = self.weight_offset[function_name]
    try:
      assert block.size + aligned_offset <= self.size
    except:
      print("overlap Data block size exceeds region size")
      print(self)
      exit(0)
    
    block.set_offset(aligned_offset)
    block.set_base_address(aligned_offset + self.base_address)
    self.blocks[function_name][block.id] = block

  def set_base_address(self, address: int):
    self.base_address = int(address)

  def __str__(self):
    if not self.blocks:
      return f"MemoryRegion({self.name}, {self.size}, {self.base_address}, blocks=[])"
    blocks_str=""
    for function_name, blocks in self.blocks.items():
      blocks_str += f"\n{function_name}:\n"
      blocks_str += f"----------------------------------------------------------\n"
      blocks_str += ",\n      ".join(str(block) for block in blocks.values())
    return (f"MemoryRegion({self.name}, {self.size}, {self.base_address}, "
            f"blocks=[\n      {blocks_str}\n    ])")

  def __repr__(self):
    return self.__str__()


class MemoryLayout:
  def __init__(self, *regions: MemoryRegion):
    self.regions = {}
    _last_end_address = 0

    for region in regions:
      self.regions[region.name] = region
      region.set_base_address(_last_end_address)
      _last_end_address += region.size

  def get_data_block_by_id(self, id: Union[str, TensorID]):
    for region in self.regions.values():
      block = region[id]
      if block is not None:
        return block
    return None

  def __getitem__(self, region_name: str):
    return self.regions.get(region_name, None)

  def __str__(self):
    regions_str = ",\n  ".join(str(region) for region in self.regions.values())
    return f"MemoryLayout(regions=[\n  {regions_str}\n])"

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
  def __init__(self, policy_info: List[RouterEntry] = None, data_block: Union[DataBlock, None] = None, fifo_id: int = -1):
    super().__init__(policy_info, data_block)
    self.fifo_id = fifo_id

  def set_fifo_id(self, fifo_id):
    self.fifo_id = fifo_id

  def __str__(self):
    policy_info_str = ", ".join(str(entry) for entry in self.policy_info) if self.policy_info else "[]"
    return f"TensorEdgeInfo([{policy_info_str}], {self.data_block}, {self.fifo_id})"

  def __repr__(self):
    return self.__str__()


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
        MemoryRegion("state_regs", ImcflowDeviceConfig.INODE_MMREG_SIZE),
        MemoryRegion("inode_0_inst", ImcflowDeviceConfig.INODE_INST_MEM_SIZE),
        MemoryRegion("inode_0_data", ImcflowDeviceConfig.INODE_DATA_MEM_SIZE),
        MemoryRegion("inode_1_inst", ImcflowDeviceConfig.INODE_INST_MEM_SIZE),
        MemoryRegion("inode_1_data", ImcflowDeviceConfig.INODE_DATA_MEM_SIZE),
        MemoryRegion("inode_2_inst", ImcflowDeviceConfig.INODE_INST_MEM_SIZE),
        MemoryRegion("inode_2_data", ImcflowDeviceConfig.INODE_DATA_MEM_SIZE),
        MemoryRegion("inode_3_inst", ImcflowDeviceConfig.INODE_INST_MEM_SIZE),
        MemoryRegion("inode_3_data", ImcflowDeviceConfig.INODE_DATA_MEM_SIZE),
    )
    self.ActiveIMCEPerFunc = {}
    self.NoCPaths = {}
    self.DataBlocks = {}
    self.ImcflowFuncMap = {}

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
    self.TensorEdgetoInfo[tensor_edge] = tensor_edge_info

  def get_tensor_edge_info(self, tensor_edge: TensorEdge):
    return self.TensorEdgetoInfo.get(tensor_edge, None)

  def get_tensor_edge_info_with_id_dir(self, tensor_id: TensorID, dir: str):
    if dir == "in":
      for edge in self.TensorEdgetoInfo.keys():
        if edge.dst_id == tensor_id:
          return self.TensorEdgetoInfo[edge]
    elif dir == "out":
      for edge in self.TensorEdgetoInfo.keys():
        if edge.src_id == tensor_id:
          return self.TensorEdgetoInfo[edge]
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

  def add_inst_edge_info(self, imce_id: NodeID, inst_edge_info: InstEdgeInfo):
    assert imce_id.is_imce(), "Only imce nodes have inst edge info"
    self.InstEdgeInfoDict[imce_id] = inst_edge_info

  def get_inst_edge_info(self, imce_id: NodeID):
    assert imce_id.is_imce(), "Only imce nodes have inst edge info"
    return self.InstEdgeInfoDict.get(imce_id, None)

  def get_data_block_dict(self, func):
    from tvm.relay.backend.contrib.imcflow import transform as imcflow_transform
    compiled_blocks, input_data_blocks, output_data_blocks = [], [], []

    # get input/output node ID
    input_node_ids = [imcflow_transform.getNodeID(
        n) for n in imcflow_transform.getInputNodesOfFunc(func)]
    output_node_id = imcflow_transform.getNodeID(func)

    for memory_region in ImcflowDeviceConfig().MemLayout.regions.values():
      for block_name, block in memory_region.blocks.items():
        # get compiled data blocks
        if isinstance(block_name, str):
          compiled_blocks.append(block)
        # get input & output data blocks
        if isinstance(block_name, TensorID):
          # get input data blocks
          if any([input_node_id == imcflow_transform.getInnerNodeID(block_name.graph_node_id) for input_node_id in input_node_ids]):
            input_data_blocks.append(block)
          # get output data blocks
          if output_node_id == imcflow_transform.getInnerNodeID(block_name.graph_node_id):
            output_data_blocks.append(block)

    self.DataBlocks = {
        "compiled": compiled_blocks,
        "input": input_data_blocks,
        "output": output_data_blocks,
    }