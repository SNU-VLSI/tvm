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

from typing import Tuple


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
    if cls._instance is None:
      cls._instance = super(ImcflowDeviceConfig, cls).__new__(
          cls, *args, **kwargs)
      cls._instance.mem_layout = MemoryLayout(
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
    return cls._instance

  def __init__(self):
    pass

  @staticmethod
  def is_supported_kernel(KH, KW):
    return (KH, KW) in {(1, 1), (3, 3), (5, 5), (7, 7)}


class DataBlock:
  def __init__(self, name: str, size: int):
    self.name = name
    self.size = size
    self.offset = -1  # offset in the region
    self.base_address = -1  # base address in the device memory

  def set_offset(self, offset: int):
    self.offset = offset

  def set_base_address(self, address: int):
    self.base_address = address

  def __str__(self):
    return (f"DataBlock({self.name}, {self.size}, {self.base_address})")


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
    self.blocks[block.name] = block

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
