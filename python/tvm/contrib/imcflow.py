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

class TensorID:
  def __init__(self, graph_node_id: Union[int, Tuple], tensor_type: str):
    assert tensor_type in {"idata", "odata", "weight",
                           "bias", "scale", "idata0", "idata1"}, "Invalid tensor type"
    self.graph_node_id = graph_node_id
    self.tensor_type = tensor_type

  def __str__(self):
    return f"TensorID({self.graph_node_id}, {self.tensor_type})"


class TensorEdge:
  def __init__(self, src_id: TensorID, dst_id: TensorID, split_idx: Union[None, int] = None):
    self.src_id = src_id
    self.dst_id = dst_id
    self.split_idx = split_idx

  def __str__(self):
    return f"TensorEdge({self.src_id}, {self.dst_id}, {self.split_idx})"


class MultiCastTensorEdge:
  def __init__(self, src_id: TensorID, dst_ids: List[TensorID], split_idx: List[Union[None, int]]):
    self.src_id = src_id
    self.dst_ids = dst_ids
    self.split_idx = split_idx

  def __str__(self):
    dst_id_strs = ", ".join(str(dst_id) for dst_id in self.dst_ids)
    return f"MultiCastTensorEdge({self.src_id}, [{dst_id_strs}], {self.split_idx})"


class DataBlock:
  def __init__(self, id: Union[str, TensorID], size: int):
    self.id = id
    self.size = size
    self.offset = -1  # offset in the region
    self.base_address = -1  # base address in the device memory

  def set_offset(self, offset: int):
    self.offset = offset

  def set_base_address(self, address: int):
    self.base_address = address

  def __str__(self):
    return (f"DataBlock({self.id}, {self.size}, {self.base_address})")


class MemoryRegion:
  def __init__(self, name: str, size: int):
    self.name = name
    self.size = size
    self.blocks = {}
    self.base_address = -1  # offset in the device memory
    self._last_offset = 0

  def __getitem__(self, block_name: str):
    return self.blocks.get(block_name, None)

  def allocate(self, block: DataBlock):
    """Allocate a data block in the region sequentially, assuming they are not delocated"""
    assert block.size + self._last_offset <= self.size, "Data block size exceeds region size"
    block.set_offset(self._last_offset)
    block.set_base_address(self._last_offset + self.base_address)
    self._last_offset += block.size
    self.blocks[block.id] = block

  def set_base_address(self, address: int):
    self.base_address = address

  def __str__(self):
    if not self.blocks:
      return f"MemoryRegion({self.name}, {self.size}, {self.base_address}, blocks=[])"
    blocks_str = ",\n      ".join(str(block) for block in self.blocks.values())
    return (f"MemoryRegion({self.name}, {self.size}, {self.base_address}, "
            f"blocks=[\n      {blocks_str}\n    ])")


class MemoryLayout:
  def __init__(self, *regions: MemoryRegion):
    self.regions = {}
    _last_end_address = 0

    for region in regions:
      self.regions[region.name] = region
      region.set_base_address(_last_end_address)
      _last_end_address += region.size

  def __getitem__(self, region_name: str):
    return self.regions.get(region_name, None)

  def __str__(self):
    regions_str = ",\n  ".join(str(region) for region in self.regions.values())
    return f"MemoryLayout(regions=[\n  {regions_str}\n])"


class RouterEntry:
  def __init__(self, router_id: int, address: int, data: Dict):
    self.router_id = router_id
    self.address = address
    self.data = data

  def __str__(self):
    return f"RouterEntry({self.router_id}, {self.address}, {self.data})"


class EdgeInfo:
  """ stores the list of router entries and memory block info for a data movement (edge). """

  def __init__(self, policy_info: List[RouterEntry], data_block: Union[DataBlock, None] = None):
    self.policy_info = policy_info
    self.data_block = data_block

  def append_policy_info(self, entry: RouterEntry):
    self.policy_info.append(entry)

  def set_data_block(self, data_block: DataBlock):
    self.data_block = data_block


class InstEdgeInfo(EdgeInfo):
  def __str__(self):
    policy_info_str = ", ".join(str(entry) for entry in self.policy_info)
    return f"InstEdgeInfo([{policy_info_str}], {self.data_block}, {self.fifo_id})"


class TensorEdgeInfo(EdgeInfo):
  def __init__(self, policy_info: List[RouterEntry], data_block: Union[DataBlock, None] = None, fifo_id: int = -1):
    super().__init__(policy_info, data_block)
    self.fifo_id = fifo_id

  def set_fifo_id(self, fifo_id):
    self.fifo_id = fifo_id

  def __str__(self):
    policy_info_str = ", ".join(str(entry) for entry in self.policy_info)
    return f"TensorEdgeInfo([{policy_info_str}], {self.data_block}, {self.fifo_id})"


class ImcflowDeviceConfig:
  """Imcflow config class"""
  INODE_NUM = 4
  IMCE_H_NUM = 4
  IMCE_W_NUM = 4
  IMCE_NUM = 16
  INODE_MMREG_SIZE = 128
  INODE_DATA_MEM_SIZE = 65536
  INODE_INST_MEM_SIZE = 1024
  IMCE_INST_MEM_SIZE = 1024

  def __new__(cls, *args, **kwargs):
    if not hasattr(cls, "instance"):
      cls.instance = super(ImcflowDeviceConfig, cls).__new__(cls)
      cls.instance.HWNodeMap = {}  # maps graph_node_id to hw_node_id
      cls.instance.TensorIDtoEdge = {}  # maps tensor_id to tensor_edge
      cls.instance.TensorEdgetoInfo = {}  # maps tensor_edge to tensor_edge_info
      cls.instance.TensorEdgeList = []   # list of tensor edges
      cls.instance.TensorEdgeListDict = {}  # maps func to list of tensor edges
      cls.instance.PolicyTableDict = {}  # maps hw_node_id to policy table entries
      cls.instance.InstEdgeDict = {}  # maps hw_node_id to inst_edge_info
      cls.instance.MemLayout = MemoryLayout(
          MemoryRegion("state_regs", ImcflowDeviceConfig.INODE_MMREG_SIZE),
          MemoryRegion("inode0_inst", ImcflowDeviceConfig.INODE_INST_MEM_SIZE),
          MemoryRegion("inode0_data", ImcflowDeviceConfig.INODE_DATA_MEM_SIZE),
          MemoryRegion("inode1_inst", ImcflowDeviceConfig.INODE_INST_MEM_SIZE),
          MemoryRegion("inode1_data", ImcflowDeviceConfig.INODE_DATA_MEM_SIZE),
          MemoryRegion("inode2_inst", ImcflowDeviceConfig.INODE_INST_MEM_SIZE),
          MemoryRegion("inode2_data", ImcflowDeviceConfig.INODE_DATA_MEM_SIZE),
          MemoryRegion("inode3_inst", ImcflowDeviceConfig.INODE_INST_MEM_SIZE),
          MemoryRegion("inode3_data", ImcflowDeviceConfig.INODE_DATA_MEM_SIZE),
      )
    return cls.instance

  def __init__(self):
    pass

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
