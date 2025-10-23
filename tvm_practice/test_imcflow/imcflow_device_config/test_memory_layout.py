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

import tvm
import tvm.testing
from tvm.contrib.imcflow import MemoryLayout, MemoryRegion, DataBlock

# Test Code
def test_memory_layout():
  my_layout = MemoryLayout(
    MemoryRegion("state_regs", 128),
    MemoryRegion("inode_0_inst", 1024),
    MemoryRegion("inode_0_data", 65536),
    MemoryRegion("inode_1_inst", 1024),
    MemoryRegion("inode_1_data", 65536),
    MemoryRegion("inode_2_inst", 1024),
    MemoryRegion("inode_2_data", 65536),
    MemoryRegion("inode_3_inst", 1024),
    MemoryRegion("inode_3_data", 65536),
  )

  state_regs = my_layout["state_regs"]
  inode_0_data = my_layout["inode_0_data"]

  state_regs.allocate(DataBlock("block_0", 64))
  inode_0_data.allocate(DataBlock("block_0", 1024))
  inode_0_data.allocate(DataBlock("block_1", 256))

  # Test addresses
  address = my_layout["inode_0_data"]["block_1"].base_address
  assert address == 128 + 1024 + 1024, f"Expected {128 + 1024 + 1024}, got {address}"
  print(my_layout)

  print("All tests passed!")

if __name__ == "__main__":
    tvm.testing.main()