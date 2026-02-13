"""Scan Register Codegen for IMCFlow.

This module generates code for scan register programming on IMCFlow hardware.
It follows the same architecture as codegen.py and ext_codegen.py:

* Receives PolicyTable_2D and NoCPaths from scan_reg_policy_gen.py
* Generates imce.cpp using ImceCodeBlockManager
* Generates inode.cpp using InodeCodeBlockManager
* Generates kernel wrapper code using KernelCodeGenerator pattern
* Memory layout (addresses/sizes) integrated with existing memory management

Architecture:
-----------
1. ScanRegMemoryLayout: Manages memory allocation for scan register data
2. ScanImceCodeBlockBuilder: Generates IMCE code blocks for recv + scan_rw
3. ScanInodeCodeBlockBuilder: Generates INode code blocks for sending scan data
4. ScanPolicyTableCodegen: Generates policy table binaries
5. ScanKernelCodegen: Generates host-side kernel wrapper code

Usage:
------
    from scan_reg_policy_gen import ScanRegPolicyGenerator
    from scan_codegen import ScanCodegenSuite

    # Generate policy table and NoC paths
    policy_gen = ScanRegPolicyGenerator()
    policy_gen.construct_noc_path()
    policy_gen.gen_policy_table()

    # Generate code
    codegen = ScanCodegenSuite(
        func_name="scan_programming",
        build_dir="./build",
        policy_table=policy_gen.PolicyTable_2D,
        noc_paths=policy_gen.NoCPaths,
        scan_packet_count=128  # Number of scan register writes
    )
    codegen.generate()
"""

from __future__ import annotations

import os
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

# TVM imports
from tvm.contrib.imcflow import NodeID, InstEdgeInfo, ImcflowDeviceConfig as DevConfig, DataBlock, CodegenContext
from tvm.relay.backend.contrib.imcflow.codeblock import (
    CodePhase, TextBlock, SequentialBlock, CodeBlock, SimpleFor, UniqueVar
)
from tvm.relay.backend.contrib.imcflow.imce_codeblock import ImceCodeBlockManager
from tvm.relay.backend.contrib.imcflow.inode_codeblock import (
    InodeCodeBlockManager,
    InodeCodeBlock,
    HaltBlock,
    SyncAllINodes,
    DoneAndIntrtBlock,
    ClearFlag,
    IMCEComputeBlock,
)
from tvm.relay.backend.contrib.imcflow.device_codegen import DeviceCodegen

if (os.getenv("IMCFLOW_HOST_OS") == "baremetal"):
  print("IMCFLOW_HOST_OS: baremetal")
  IMCFLOW_ADDR = 0x80000000
  IMCFLOW_LEN = DevConfig.IMCFLOW_ADDR_SIZE
  INT_ACK_GEN_ADDR = 0
  INT_ACK_GEN_LEN = 0
elif (os.getenv("IMCFLOW_HOST_OS") == "linux"):
  print("IMCFLOW_HOST_OS: linux")
  IMCFLOW_ADDR = os.environ["IMCFLOW_ADDR"]
  IMCFLOW_LEN = os.environ["IMCFLOW_LEN"]
  INT_ACK_GEN_ADDR = os.environ["INT_ACK_GEN_ADDR"]
  INT_ACK_GEN_LEN = os.environ["INT_ACK_GEN_LEN"]
else:
  raise ValueError(f"Unsupported IMCFLOW_HOST_OS: {os.getenv('IMCFLOW_HOST_OS')}")


# ============================================================================
# Helper Functions for Memory Layout Access
# ============================================================================
class WriteSCANIMEMBlock(InodeCodeBlock):
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

def get_scan_data_block(func_name: str) -> Optional[DataBlock]:
    """Get scan data block from DevConfig memory layout.

    Args:
        func_name: Function name (e.g., "scan_reg")

    Returns:
        DataBlock for scan data, or None if not found
    """
    mem_layout = DevConfig().MemLayout.get(func_name)
    if not mem_layout:
        return None

    # Find the scan data block (allocated by scan_reg_policy_gen.py)
    # MemLayout[func_name] is a dictionary-like object mapping to MemoryRegion objects
    for inode_data_name in mem_layout.keys():
        if isinstance(inode_data_name, str) and inode_data_name.endswith("_data"):
            inode_data = mem_layout[inode_data_name]
            # MemoryRegion.blocks returns dict {block_id: DataBlock}
            for block in inode_data.blocks.values():
                # DataBlock.id returns the block identifier (string or TensorEdge)
                if isinstance(block.id, str) and "scan_data" in block.id:
                    return block
    return None


def get_policy_block_address(func_name: str, node_id: NodeID) -> Optional[int]:
    """Get policy table block base address for a node.

    Args:
        func_name: Function name (e.g., "scan_reg")
        node_id: Node ID

    Returns:
        Base address of the policy table block, or None if not found
    """
    mem_layout = DevConfig().MemLayout.get(func_name)
    if not mem_layout:
        return None

    # Get master inode for IMCE nodes
    master_node = node_id.master() if node_id.is_imce() else node_id
    inode_data_name = f"{master_node.name}_data"

    if inode_data_name not in mem_layout:
        return None

    inode_data = mem_layout[inode_data_name]
    # Look for policy table block - MemoryRegion.blocks returns dict {block_id: DataBlock}
    for block in inode_data.blocks.values():
        # DataBlock.id returns the block identifier (string or TensorEdge)
        if isinstance(block.id, str) and f"{node_id.name}_scan_reg_policy" == block.id:
            return block.offset

    return None


# ============================================================================
# Custom Code Blocks for Scan Register Programming
# ============================================================================

class ScanPolicyUpdateBlock(InodeCodeBlock):
    """Policy update block for scan register programming.

    Uses policy tables from DevConfig().PolicyTableDict[func_name].
    """

    def __init__(self, node_id: NodeID, func_name: str, annotation: str = ""):
        super().__init__(annotation)
        assert node_id.is_inode(), "ScanPolicyUpdateBlock can only be used for inode"
        self.node_id = node_id
        self.func_name = func_name
        self._build()

    def _build(self):
        same_row_node_ids = [self.node_id] + self.node_id.slaves()
        same_row_node_ids.sort(key=lambda id: id.to_coord(1))

        policy_table_dict = DevConfig().PolicyTableDict.get(self.func_name, {})

        for id in same_row_node_ids:
            # Get policy table entries for this node
            policy_entries = policy_table_dict.get(id, [])
            if len(policy_entries) <= 1:  # Skip nodes with only zero entry
                continue

            # Get offset from DevConfig memory layout
            offset = get_policy_block_address(self.func_name, id)
            if offset is None:
                continue

            var = UniqueVar("policy_table_start_address", dtype="int")
            loop_count = len(policy_entries)

            self.body.add(TextBlock(f"{var} = {offset};"))

            if loop_count > 5:
                # Using lambda for SimpleFor body to inject 'iter' variable
                self.body.add(SimpleFor(loop_count,
                    lambda iter, wid=id.to_coord(1): f"__builtin_INODE_PU({var} + {iter}*32, 0, {iter}, {wid});"))
            else:
                for i in range(loop_count):
                    self.body.add(TextBlock(f"__builtin_INODE_PU({var}, {i*32}, {i}, {id.to_coord(1)});"))


# ============================================================================
# IMCE Code Block Builder for Scan Registers
# ============================================================================

class ScanImceCodeBlockBuilder:
    """Generates IMCE code blocks for scan register programming.

    For each IMCE node:
    1. RECV scan register packets from master inode
    2. Execute __builtin_IMCE_SCAN_RW() for each packet
    """

    def __init__(self, func_name: str, per_imce_packets: int = 2):
        self.func_name = func_name
        self.per_imce_packets = per_imce_packets
        self.codeblocks = ImceCodeBlockManager(func_name)
        # Get NoC paths from DevConfig
        self.noc_paths = DevConfig().NoCPaths.get(func_name, {})

    def build(self, fifo_id: int = 0) -> None:
        """Generate IMCE code blocks for all IMCE nodes."""
        # Each IMCE receives scan packets and writes them
        for imce_node in NodeID.imces():
            if imce_node not in self.noc_paths:
                continue

            source_inode, dest_imce, split_idx = self.noc_paths[imce_node]

            # Generate recv + scan_rw for each packet
            # Each IMCE receives per_imce_packets (e.g., 2 packets = 64 bytes = 2 scan registers)
            for i in range(self.per_imce_packets):
                # RECV packet
                self.codeblocks.blocks[imce_node][CodePhase.INIT].append(
                    TextBlock(f"short16 scan_pkt_{i} = __builtin_IMCE_RECV({fifo_id}); // scan reg {i}")
                )
            for i in range(self.per_imce_packets):
                # SCAN_RW to write the scan register
                self.codeblocks.blocks[imce_node][CodePhase.INIT].append(
                    TextBlock(f"scan_pkt_{i} = __builtin_IMCE_SCAN_RW(scan_pkt_{i}); // write scan reg {i}")
                )

        # Add STOP blocks for all active IMCE nodes
        for imce_node in self.noc_paths.keys():
            self.codeblocks.blocks[imce_node][CodePhase.END].append(
                TextBlock("__builtin_IMCE_STOP(); // STOP")
            )

    def generate(self) -> str:
        """Generate complete IMCE C++ code."""
        return self.codeblocks.generate()


# ============================================================================
# INode Code Block Builder for Scan Registers
# ============================================================================

class ScanInodeCodeBlockBuilder:
    """Generates INode code blocks for scan register programming.

    Master inode for each IMCE sends scan register packets to that IMCE.
    """

    def __init__(self, func_name: str):
        self.func_name = func_name
        self.codeblocks = InodeCodeBlockManager(func_name)
        # Get NoC paths and policy table from DevConfig
        self.noc_paths = DevConfig().NoCPaths.get(func_name, {})
        self.policy_table = DevConfig().PolicyTableDict.get(func_name, {})

    def build_initialization(self) -> None:
        """Build initialization phase: clear flags, update policy."""
        # Clear flags
        for inode in NodeID.inodes():
            block = ClearFlag("clear flag before scan policy update")
            self.codeblocks.append(inode, block, CodePhase.INIT)

        # Policy update - gets policy table from DevConfig
        for inode in NodeID.inodes():
            block = ScanPolicyUpdateBlock(inode, self.func_name, "scan register policy update")
            self.codeblocks.append(inode, block, CodePhase.INIT)

        # Sync and interrupt
        self.sync_and_halt(CodePhase.INIT)

        # IMEM write (same pattern as codegen.py)
        # This writes the IMCE instruction binaries (e.g., {imce}_imem.bin) into IMCE IMEM,
        # using InstEdgeInfoDict populated by scan_reg_policy_gen.add_edge_info().
        inst_edges = DevConfig().InstEdgeInfoDict.get(self.func_name, {})
        if not inst_edges:
            raise RuntimeError(
                f"InstEdgeInfoDict['{self.func_name}'] is empty. "
                "Run scan_reg_policy_gen.add_edge_info(func_name=...) before codegen."
            )

        for imce, inst_edge in sorted(inst_edges.items(), key=lambda x: x[0].name):
            block = WriteSCANIMEMBlock(inst_edge, f"imem write: {imce.name}")
            self.codeblocks.append(imce.master(), block, CodePhase.INIT)

        self.build_imce_compute()

        self.sync_and_halt(CodePhase.INIT)

    def build_scan_send(self, scan_packet_count: int, fifo_id: int = 0, per_imce_packets: int = 2) -> None:
        """Build scan register send operations.

        Each master inode sends scan packets to its IMCE(s) from its own data region.

        Args:
            scan_packet_count: Total number of packets (e.g., 32 = 16 IMCEs × 2 packets)
            fifo_id: FIFO ID for communication
            per_imce_packets: Number of packets each IMCE receives (default: 2)
        """
        mem_layout = DevConfig().MemLayout[self.func_name]

        # Group IMCE nodes by their master inode
        master_to_imces: Dict[NodeID, List[NodeID]] = {}
        for imce_node, (source_inode, dest_imce, split_idx) in self.noc_paths.items():
            if source_inode not in master_to_imces:
                master_to_imces[source_inode] = []
            master_to_imces[source_inode].append(dest_imce)

        # Calculate packet index for each IMCE (based on IMCE's position in grid)
        # IMCE grid: h=0-3, w=1-4 → IMCE indices 0-15
        def get_imce_index(imce_node: NodeID) -> int:
            """Get linear index for IMCE (0-15)."""
            # Parse node name like "imce_0_1" to extract h=0, w=1
            parts = imce_node.name.split('_')
            h, w = int(parts[1]), int(parts[2])
            return h * 4 + (w - 1)  # w is 1-4, convert to 0-3

        # Generate send blocks for each master inode
        for master_inode, imce_list in sorted(master_to_imces.items(), key=lambda x: x[0].name):
            # Get this INode's scan_data block from its own data region
            inode_data_name = f"{master_inode.name}_data"
            scan_data_block = None
            for block in mem_layout[inode_data_name].blocks.values():
                if f"{master_inode.name}_scan_data" in str(block.id):
                    scan_data_block = block
                    break

            if scan_data_block is None:
                raise RuntimeError(f"Scan data block not found for {master_inode.name}")

            # Get policy address for broadcast (all IMCEs in group)
            # policy_entries = self.policy_table.get(master_inode, [])
            # policy_addr = 1 if len(policy_entries) > 1 else 0  # Skip zero entry at index 0

            # Send packets for each IMCE in this group
            # Use local packet index within this INode's data block
            local_pkt_idx = 0
            for imce_node in sorted(imce_list, key=lambda n: n.name):
                # Each IMCE gets per_imce_packets consecutive packets
                for pkt_idx in range(per_imce_packets):
                    # Local packet offset within this INode's scan_data block
                    packet_offset = local_pkt_idx * 32  # 32 bytes per packet
                    packet_addr = scan_data_block.offset + packet_offset

                    # policy address setting
                    policy_addr = imce_node.to_coord(1)

                    # INODE_SEND(addr, imm, policy, fifo_id)
                    self.codeblocks.blocks[master_inode][CodePhase.EXEC].append(
                        TextBlock(
                            f"__builtin_INODE_SEND({packet_addr}, 0, {policy_addr}, {fifo_id}); "
                            f"// scan packet {pkt_idx} to {imce_node.name}"
                        )
                    )
                    local_pkt_idx += 1

    def build_finalization(self) -> None:
        """Build finalization phase: sync, done, halt."""
        self.sync_and_halt(CodePhase.END)
    
    def build_imce_compute(self) -> None:
        """Build IMCE compute phase."""
        for imce, inst_edge in sorted(DevConfig().InstEdgeInfoDict[self.func_name].items(), key=lambda x: x[0].name):
            policy_addr = inst_edge.policy_info[0].address # get first policy address
            block = IMCEComputeBlock(policy_addr, f"{imce.name} compute")
            self.codeblocks.append(imce.master(), block, CodePhase.INIT)

    def sync_and_halt(self, phase: CodePhase) -> None:
        """Sync all inodes and halt."""
        inode_master = NodeID.inode_3_0
        inode_slaves = [node for node in NodeID.inodes() if node != inode_master]

        # Sync all inodes
        for inode in NodeID.inodes():
            block = SyncAllINodes(inode, "sync all inodes")
            self.codeblocks.append(inode, block, phase)

        # Halt slaves
        for inode_slv in inode_slaves:
            block = HaltBlock("halt slave inode")
            self.codeblocks.append(inode_slv, block, phase)

        # Done and halt master
        block = DoneAndIntrtBlock("done and intrt for master inode")
        self.codeblocks.append(inode_master, block, phase)
        block = HaltBlock("halt master inode")
        self.codeblocks.append(inode_master, block, phase)

    def generate(self) -> str:
        """Generate complete INode C++ code."""
        return self.codeblocks.generate()


# ============================================================================
# Policy Table Codegen for Scan Registers
# ============================================================================

class ScanPolicyTableCodegen:
    """Generates policy table binary files for scan register paths.

    Similar to PolicyTableCodegen in codegen.py but for scan register paths.
    """

    def __init__(self, func_name: str, build_dir: str, host_isa: str = "arm"):
        self.func_name = func_name
        self.build_dir = build_dir
        self.host_isa = host_isa
        self.func_dir = os.path.join(build_dir, func_name)
        os.makedirs(self.func_dir, exist_ok=True)

    def pack_to_bin(self, entry: Dict, endian: str = 'little') -> bytes:
        """Pack a policy table entry to 32-byte binary format."""
        assert set(entry.keys()) == {
            'Local', 'North', 'East', 'South', 'West'
        }, "Invalid policy table entry"

        def get_bits(val, num_bits):
            return (val & ((1 << num_bits) - 1)) if val is not None else 0

        val = 0
        for direction in ['Local', 'North', 'East', 'South', 'West']:
            conf = entry[direction]
            val = (val << 1) | (1 if conf["enable"] else 0)
            val = (val << 6) | get_bits(conf["addr"], 6)
            if direction == 'Local':
                val = (val << 3) | get_bits(conf["ksel"], 3)
                val = (val << 6) | get_bits(conf["chunk_index"], 6)

        bin_data = bytearray()
        bin_data.extend(val.to_bytes(32, byteorder=endian, signed=False))
        return bytes(bin_data)

    def generate(self, policy_table: Dict) -> List[Path]:
        """Generate policy table binary files for all nodes.

        Args:
            policy_table: PolicyTable_2D from ScanRegPolicyGenerator

        Returns:
            List of generated file paths
        """
        generated_files = []

        for node_id, entries in sorted(policy_table.items(), key=lambda x: x[0].name):
            if len(entries) <= 1:  # Skip nodes with only zero entry
                continue

            # Generate binary file
            # Note: When ld -r -b binary converts the file, it creates symbols based on:
            # _binary_<dir_name>_<file_name>_bin_<start|end|size>
            # Since we're in the {func_name} directory and want symbols like:
            # _binary_{func_name}_{node_id}_scan_reg_policy_bin_start
            # The filename should be: {node_id}_scan_reg_policy.bin
            bin_filename = f"{node_id.name}_scan_reg_policy.bin"
            bin_path = os.path.join(self.func_dir, bin_filename)

            with open(bin_path, "wb") as f:
                for entry in entries:
                    bin_data = self.pack_to_bin(entry)
                    f.write(bin_data)

            generated_files.append(Path(bin_path))

            # Generate host object file
            host_obj_filename = f"{node_id.name}_scan_reg_policy.host.o"
            dev_codegen = DeviceCodegen("inode", self.build_dir, self.host_isa)
            dev_codegen.func_dir = self.func_dir
            dev_codegen.create_host_object(bin_filename, host_obj_filename)

            print(f"Generated policy table: {bin_path} ({len(entries)} entries)")

        return generated_files


# ============================================================================
# Kernel Code Generation for Scan Registers
# ============================================================================

class ScanKernelCodegen:
    """Generates host-side kernel wrapper code for scan register programming.

    Similar to KernelCodeGenerator in ext_codegen.py but simplified for scan registers.
    """

    def __init__(
        self,
        func_name: str,
        scan_packet_count: int,
        host_os: str = "linux",
        use_polling: bool = True,
    ):
        self.func_name = func_name
        self.scan_packet_count = scan_packet_count
        self.host_os = host_os
        self.use_polling = use_polling

        # Initialize base address macros (similar to ext_codegen.py)
        self.base_address_macros = {
            "IMCFLOW_ADDR": IMCFLOW_ADDR,
            "IMCFLOW_LEN": IMCFLOW_LEN,
            "INT_ACK_GEN_ADDR": INT_ACK_GEN_ADDR,
            "INT_ACK_GEN_LEN": INT_ACK_GEN_LEN,
            "IMCFLOW_DEVICE": '"/dev/uio5"',
            "INT_ACK_GEN_DEVICE": '"/dev/uio4"',
            "STATE_REG_IDX": 0,
            "PC_REG_IDX": 2,
            "INTR_DONE_REG_IDX": 7,
            "SET_IDLE_CODE": 0,
            "SET_RUN_CODE": 1,
            "SET_PROGRAM_CODE": 2,
            "INODE_PC_START_P0_ENUM_VAL": 0,
            "INODE_PC_START_EXTERN_ENUM_VAL": 1,
            "INODE_NUM": DevConfig().INODE_NUM,
        }

    def generate_header(self) -> str:
        """Generate C header includes."""
        # Note: we keep Linux-only headers unconditionally for now since
        # this generator primarily targets the existing Linux host runner.
        # The OS-specific behavior is handled in generated C code blocks.
        return """
#include <stdlib.h>
#include <string.h>
#include <string>
#include <stdio.h>
#include <stdint.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cnpy.h>
#include <vector>
"""

    def generate_base_addr_macros(self) -> str:
        """Generate base address macro definitions from environment variables."""
        lines = []
        for key, value in self.base_address_macros.items():
            lines.append(f"#define {key} {value}")
        lines.append("")
        lines.append("// short16 type definition")
        lines.append("typedef short short16 __attribute__((ext_vector_type(16)));")
        lines.append("")
        return '\n'.join(lines)

    def generate_extern_declarations(self) -> str:
        """Generate extern declarations for policy and instruction binary files."""
        lines = ['extern "C" {']

        mem_layout = DevConfig().MemLayout[self.func_name]

        # Collect all policy and imem blocks
        binary_blocks = []
        for region_name in mem_layout.keys():
            # Check both data regions (for policy tables) and instruction regions (for imem)
            if isinstance(region_name, str) and (region_name.endswith("_data") or region_name.endswith("_inst")):
                region = mem_layout[region_name]
                for block in region.blocks.values():
                    block_id_str = str(block.id)
                    # Include policy tables (contains "policy") and imem blocks (contains "_imem")
                    if "policy" in block_id_str or "imem" in block_id_str:
                        binary_blocks.append((block_id_str, block))

        # Generate extern declarations for each block
        for block_id, block in sorted(binary_blocks, key=lambda x: x[0]):
            # Generate binary object file name: _binary_{func_name}_{block_id}_bin
            bin_name = f"_binary_{self.func_name}_{block_id}_bin"
            lines.append(f"  extern const int32_t {bin_name}_start[];")
            lines.append(f"  extern const int32_t {bin_name}_end[];")

        lines.append('}')
        lines.append('')
        return '\n'.join(lines)

    def generate_policy_and_imem_transfers(self) -> str:
        """Generate code to transfer policy tables and instruction memory from binary files."""
        lines = []
        lines.append("    // Transfer policy tables and instruction memory from binary files")
        lines.append("    fprintf(stderr, \"Transferring policy tables and instruction memory to NPU:\\n\");")

        mem_layout = DevConfig().MemLayout[self.func_name]

        # Collect all policy and imem blocks
        binary_blocks = []
        for region_name in mem_layout.keys():
            # Check both data regions (for policy tables) and instruction regions (for imem)
            if isinstance(region_name, str) and (region_name.endswith("_data") or region_name.endswith("_inst")):
                region = mem_layout[region_name]
                for block in region.blocks.values():
                    block_id_str = str(block.id)
                    # Include policy tables (contains "policy") and imem blocks (contains "_imem")
                    if "policy" in block_id_str or "_imem" in block_id_str or "imem" in block_id_str:
                        binary_blocks.append((block_id_str, block))

        # Generate transfer code for each block
        for block_id, block in sorted(binary_blocks, key=lambda x: x[1].base_address):
            bin_name = f"_binary_{self.func_name}_{block_id}_bin"
            base_addr = block.base_address
            size = block.size

            lines.append(f"    // Transfer {block_id} ({size} bytes at 0x{base_addr:x})")
            lines.append(f"    {{")
            lines.append(f"        const uint32_t* src = (const uint32_t*){bin_name}_start;")
            lines.append(f"        size_t len = (size_t)({bin_name}_end - {bin_name}_start);")
            lines.append(f"        for (size_t i = 0; i < len; i++) {{")
            lines.append(f"            npu_ptr[({base_addr} / 4) + i] = src[i];")
            lines.append(f"        }}")
            lines.append(f"        fprintf(stderr, \"  {block_id}: 0x%x (%zu bytes)\\n\", {base_addr}, len * 4);")
            lines.append(f"    }}")

        lines.append("")
        return '\n'.join(lines)

    def generate_utilities(self) -> str:
        """Generate utility functions."""
        return """
static inline void enable_imcflow_interrupt(int fd)
{
  uint32_t info = 1;
  ssize_t nb = write(fd, &info, sizeof(info));
  if (nb != (ssize_t)sizeof(info)) {
    perror("write failed");
    close(fd);
    exit(1);
  }
}

static inline void wait_imcflow_interrupt(int fd)
{
  uint32_t info;
  ssize_t nb = read(fd, &info, sizeof(info));
}

static inline void generate_ack(uint32_t* int_ack_gen)
{
  int_ack_gen[0] = 0b1;
}

// Poll until ImcFlow returns to IDLE state
static void wait_for_idle(volatile uint32_t* npu_pointer)
{
  while (npu_pointer[STATE_REG_IDX] != SET_IDLE_CODE) {
    // Busy wait
  }
}
"""

    def _generate_device_pointer_setup(self) -> str:
        """Generate OS-specific device open + mmap setup (ext_codegen.py style)."""
        if self.host_os == "linux":
            # Match ext_codegen.py behavior: open both UIO devices and mmap.
            return """
    int npu_fd = open(IMCFLOW_DEVICE, O_RDWR);
    if (npu_fd < 0) {
        perror("npu UIO cannot be opened");
        free(scan_data);
        return -1;
    }

    int int_ack_gen_fd = open(INT_ACK_GEN_DEVICE, O_RDWR);
    if (int_ack_gen_fd < 0) {
        perror("interrupt ack gen UIO cannot be opened");
        close(npu_fd);
        free(scan_data);
        return -1;
    }

    size_t npu_len = (size_t)IMCFLOW_LEN;
    uint32_t* npu_ptr = (uint32_t*)mmap(NULL, npu_len, PROT_WRITE | PROT_READ, MAP_SHARED, npu_fd, 0);
    if (npu_ptr == MAP_FAILED) {
        perror("npu_ptr mmap error");
        close(npu_fd);
        close(int_ack_gen_fd);
        free(scan_data);
        return -1;
    }

    size_t int_ack_gen_len = (size_t)INT_ACK_GEN_LEN;
    uint32_t* int_ack_ptr = (uint32_t*)mmap(NULL, int_ack_gen_len, PROT_WRITE | PROT_READ, MAP_SHARED, int_ack_gen_fd, 0);
    if (int_ack_ptr == MAP_FAILED) {
        perror("int_ack_ptr mmap error");
        munmap(npu_ptr, npu_len);
        close(npu_fd);
        close(int_ack_gen_fd);
        free(scan_data);
        return -1;
    }
"""
        if self.host_os == "baremetal":
            return """
    uint32_t* npu_ptr = (uint32_t*)IMCFLOW_ADDR;
    uint32_t* int_ack_ptr = (uint32_t*)INT_ACK_GEN_ADDR;
"""
        raise ValueError(f"Unsupported host_os: {self.host_os}")

    def _generate_device_pointer_cleanup(self) -> str:
        """Generate OS-specific device cleanup (ext_codegen.py style)."""
        if self.host_os == "linux":
            return """
    // Cleanup device pointer
    munmap(npu_ptr, (size_t)IMCFLOW_LEN);
    close(npu_fd);
    munmap(int_ack_ptr, (size_t)INT_ACK_GEN_LEN);
    close(int_ack_gen_fd);
"""
        if self.host_os == "baremetal":
            return """
    // baremetal: no cleanup
"""
        raise ValueError(f"Unsupported host_os: {self.host_os}")

    def _generate_wait_completion(self) -> str:
        """Generate wait-for-completion code based on OS and polling config."""
        if self.host_os == "linux":
            return "wait_imcflow_interrupt(npu_fd);\n    generate_ack(int_ack_ptr);\n"
        # baremetal
        if self.use_polling:
            return "wait_for_idle(npu_ptr);\n"
        # If polling disabled on baremetal, there's currently no interrupt path.
        return "// NOTE: baremetal without polling is not supported; falling back to polling\n    wait_for_idle(npu_ptr);\n"

    def _generate_enable_interrupt(self) -> str:
        if self.host_os == "linux":
            return "enable_imcflow_interrupt(npu_fd);\n"
        return ""

    def generate_scan_data_array(self, scan_values: List[List[int]]) -> str:
        """Generate C array containing scan register values.

        Args:
            scan_values: List of scan register values (each is short16)
        """
        lines = [
            f"// Scan register values ({len(scan_values)} packets, 32 bytes each)",
            f"static const short16 {self.func_name}_scan_data[] = {{"
        ]

        for i, values in enumerate(scan_values):
            if len(values) != 16:
                raise ValueError(f"Scan packet {i} must have exactly 16 int16 values")

            vals_str = ", ".join(str(int(v)) for v in values)
            lines.append(f"    {{ {vals_str} }},  // packet {i}")

        lines.append("};")
        return "\n".join(lines)

    def generate_kernel_function(self) -> str:
        """Generate main kernel function."""
        mem_layout = DevConfig().MemLayout[self.func_name]

        # Get scan_data blocks for all INodes
        scan_data_blocks = []
        for inode in NodeID.inodes():
            inode_data_name = f"{inode.name}_data"
            for block in mem_layout[inode_data_name].blocks.values():
                if f"{inode.name}_scan_data" in str(block.id):
                    scan_data_blocks.append((inode, block))
                    break

        if len(scan_data_blocks) != 4:
            raise RuntimeError(
                f"Expected 4 scan_data blocks (one per INode), found {len(scan_data_blocks)}"
            )

        # Generate transfer code for each INode's scan_data block
        transfer_code = []
        packets_per_inode = 8  # 4 IMCEs × 2 packets each
        for inode_idx, (inode, block) in enumerate(scan_data_blocks):
            packet_start = inode_idx * packets_per_inode
            transfer_code.append(
                f"""
    // Transfer scan data for {inode.name} (packets {packet_start}-{packet_start + packets_per_inode - 1})
    fprintf(stderr, "  {inode.name}: 0x%x ({packets_per_inode} packets)\\n", {block.base_address});
    npu_scan_base = &npu_ptr[{block.base_address} / 4];
    for (int i = {packet_start}; i < {packet_start + packets_per_inode}; i++) {{
        // Each packet is 32 bytes = 8 uint32_t
        for (int j = 0; j < 8; j++) {{
            npu_scan_base[(i - {packet_start}) * 8 + j] = scan_data_ptr[i * 8 + j];
        }}
    }}"""
            )

        transfer_code_str = "".join(transfer_code)

        device_setup = self._generate_device_pointer_setup()
        device_cleanup = self._generate_device_pointer_cleanup()
        enable_intr = self._generate_enable_interrupt()
        wait_done = self._generate_wait_completion()

        return f"""
#ifdef __cplusplus
extern "C"
#endif
int32_t program_scan_reg(const char* scan_file_path) {{
    // Check if scan_file_path is provided
    if (scan_file_path == NULL || scan_file_path[0] == '\\0') {{
        fprintf(stderr, "Error: NPZ directory path is required for program_scan_reg\\n");
        fprintf(stderr, "Usage: program_scan_reg <npz_directory_path>\\n");
        return -1;
    }}

    // Strip surrounding quotes from scan_file_path if present
    std::string scan_file_path_str(scan_file_path);
    if (scan_file_path_str.size() >= 2 && scan_file_path_str.front() == '"' && scan_file_path_str.back() == '"') {{
        scan_file_path_str = scan_file_path_str.substr(1, scan_file_path_str.size() - 2);
    }}
    const char* actual_scan_file_path = scan_file_path_str.c_str();

    fprintf(stderr, "Starting program_scan_reg from directory: %s\\n", actual_scan_file_path);

    // Allocate memory for scan data (32 packets × 32 bytes = 1024 bytes)
    const int num_packets = {self.scan_packet_count};
    const int bytes_per_packet = 32;
    uint32_t* scan_data = (uint32_t*)malloc(num_packets * bytes_per_packet);
    if (!scan_data) {{
        fprintf(stderr, "Error: Failed to allocate memory for scan data\\n");
        return -1;
    }}

    // Convert NPZ files to scan packets
    // Each IMCE has its own NPZ file with 64 bytes → 2 packets (32 bytes each)
    // 16 IMCEs × 2 packets = 32 packets total
    const int imce_count = 16;
    const int bytes_per_imce = 64;

    // Convert NPZ bytes to scan packets following the same logic as load_scan_values_from_npz
    // For each IMCE: load NPZ file, 64 bytes → 512 bits → bit-reverse → split into reg0/reg1 → 2 packets
    fprintf(stderr, "Loading and converting NPZ data with bit reversal...\\n");
    
    for (int imce_idx = 0; imce_idx < imce_count; imce_idx++) {{
        // Calculate IMCE coordinates: h=0-3, w=1-4
        int h = imce_idx / 4;
        int w = imce_idx % 4 + 1;
        std::string imce_filename = std::string(actual_scan_file_path) + "/imce_" + std::to_string(h) + "_" + std::to_string(w) + ".npz";
        
        // Load IMCE NPZ file
        cnpy::npz_t imce_npz;
        try {{
            imce_npz = cnpy::npz_load(imce_filename);
        }} catch (const std::exception& e) {{
            fprintf(stderr, "Error loading NPZ file %s: %s\\n", imce_filename.c_str(), e.what());
            free(scan_data);
            return -1;
        }}

        if (imce_npz.find("arr_0") == imce_npz.end()) {{
            fprintf(stderr, "Error: 'arr_0' not found in NPZ file %s\\n", imce_filename.c_str());
            free(scan_data);
            return -1;
        }}

        cnpy::NpyArray imce_arr = imce_npz["arr_0"];
        uint8_t* imce_bytes = imce_arr.data<uint8_t>();
        size_t imce_bytes_size = imce_arr.num_bytes();

        if (imce_bytes_size != bytes_per_imce) {{
            fprintf(stderr, "Error: Expected %d bytes in NPZ file %s, got %zu bytes\\n", 
                    bytes_per_imce, imce_filename.c_str(), imce_bytes_size);
            free(scan_data);
            return -1;
        }}
        
        // Step 1: Convert 64 bytes to 512-bit string
        char bit_str[513];  // 512 bits + null terminator
        for (int i = 0; i < bytes_per_imce; i++) {{
            for (int b = 0; b < 8; b++) {{
                bit_str[i * 8 + b] = ((imce_bytes[i] >> (7 - b)) & 1) ? '1' : '0';
            }}
        }}
        bit_str[512] = '\\0';
        
        // Step 2: Reverse the entire bit string
        char rev_bit_str[513];
        for (int i = 0; i < 512; i++) {{
            rev_bit_str[i] = bit_str[511 - i];
        }}
        rev_bit_str[512] = '\\0';
        
        // Step 3: Extract reg1 (bits 0-256) and reg0 (bits 256-512)
        // reg1 corresponds to packet 1 (bytes 0-31 after reversal)
        // reg0 corresponds to packet 0 (bytes 32-63 after reversal)
        char reg1_bits[257], reg0_bits[257];
        memcpy(reg1_bits, &rev_bit_str[0], 256);
        reg1_bits[256] = '\\0';
        memcpy(reg0_bits, &rev_bit_str[256], 256);
        reg0_bits[256] = '\\0';
        
        // Step 4: Convert bit strings to int16 packets
        // Each packet has 16 int16 values
        int16_t* packet_0 = (int16_t*)&scan_data[(imce_idx * 2 + 0) * 8];  // 32 bytes = 8 uint32_t = 16 int16
        int16_t* packet_1 = (int16_t*)&scan_data[(imce_idx * 2 + 1) * 8];
        
        // Convert reg0_bits to packet_0
        for (int i = 0; i < 16; i++) {{
            // Extract 16 bits for this int16
            char bits_16[17];
            memcpy(bits_16, &reg0_bits[i * 16], 16);
            bits_16[16] = '\\0';
            
            // Reverse bits for little-endian int16 interpretation
            char bits_16_rev[17];
            for (int b = 0; b < 16; b++) {{
                bits_16_rev[b] = bits_16[15 - b];
            }}
            bits_16_rev[16] = '\\0';
            
            // Convert binary string to int16 (treat leftmost char as MSB)
            int32_t val = 0;
            for (int b = 0; b < 16; b++) {{
                if (bits_16_rev[b] == '1') {{
                    val |= (1 << (15 - b));  // Leftmost char is MSB (bit 15)
                }}
            }}
            
            // Handle signed conversion (2's complement)
            if (val >= 32768) {{
                val -= 65536;
            }}
            packet_0[i] = (int16_t)val;
        }}
        
        // Convert reg1_bits to packet_1
        for (int i = 0; i < 16; i++) {{
            // Extract 16 bits for this int16
            char bits_16[17];
            memcpy(bits_16, &reg1_bits[i * 16], 16);
            bits_16[16] = '\\0';
            
            // Reverse bits for little-endian int16 interpretation
            char bits_16_rev[17];
            for (int b = 0; b < 16; b++) {{
                bits_16_rev[b] = bits_16[15 - b];
            }}
            bits_16_rev[16] = '\\0';
            
            // Convert binary string to int16 (treat leftmost char as MSB)
            int32_t val = 0;
            for (int b = 0; b < 16; b++) {{
                if (bits_16_rev[b] == '1') {{
                    val |= (1 << (15 - b));  // Leftmost char is MSB (bit 15)
                }}
            }}
            
            // Handle signed conversion (2's complement)
            if (val >= 32768) {{
                val -= 65536;
            }}
            packet_1[i] = (int16_t)val;
        }}
    }}
    
    fprintf(stderr, "Loaded and converted %d scan packets from NPZ files\\n", num_packets);

{device_setup}

{self.generate_policy_and_imem_transfers()}

    // Transfer scan data to each INode's memory region
    fprintf(stderr, "Transferring scan data to NPU:\\n");
    uint32_t* scan_data_ptr = scan_data;
    uint32_t* npu_scan_base;{transfer_code_str}

    // Set PC and execute policy update phase
    fprintf(stderr, "Set the inode pc to 0 and run.\\n");
    for (int i = 0; i < INODE_NUM; i++) {{
        npu_ptr[PC_REG_IDX + i] = (INODE_PC_START_EXTERN_ENUM_VAL << 30);
    }}
    {enable_intr}    npu_ptr[STATE_REG_IDX] = SET_PROGRAM_CODE;
    {wait_done}    npu_ptr[INTR_DONE_REG_IDX] = 1;
    for (int i = 0; i < INODE_NUM; i++) {{
        npu_ptr[PC_REG_IDX + i] = (INODE_PC_START_P0_ENUM_VAL << 30);
    }}
    {enable_intr}    npu_ptr[STATE_REG_IDX] = SET_RUN_CODE;
    {wait_done}    npu_ptr[INTR_DONE_REG_IDX] = 1;

    // Execute scan register programming phase
    fprintf(stderr, "Executing scan register programming phase\\n");
    for (int i = 0; i < INODE_NUM; i++) {{
        npu_ptr[PC_REG_IDX + i] = (INODE_PC_START_P0_ENUM_VAL << 30);
    }}
    {enable_intr}    npu_ptr[STATE_REG_IDX] = SET_RUN_CODE;
    {wait_done}    npu_ptr[INTR_DONE_REG_IDX] = 1;

    // Cleanup
    free(scan_data);
{device_cleanup}

    fprintf(stderr, "program_scan_reg completed successfully\\n");
    return 0;
}}

int main(int argc, char* argv[]) {{
    if (argc != 2) {{
        fprintf(stderr, "Usage: %s <scan_file_path>\\n", argv[0]);
        fprintf(stderr, "  scan_file_path: Directory containing NPZ files (imce_<h>_<w>.npz)\\n");
        return 1;
    }}

    const char* scan_file_path = argv[1];

    program_scan_reg(scan_file_path);

    return 0;
}}
"""

    def generate(self, scan_values: List[List[int]]) -> str:
        """Generate complete kernel code.

        Args:
            scan_values: List of scan register values to program
        """
        code_parts = [
            self.generate_header(),
            self.generate_base_addr_macros(),
            self.generate_extern_declarations(),
            self.generate_utilities(),
            # self.generate_scan_data_array(scan_values),
            self.generate_kernel_function()
        ]
        return "\n".join(code_parts)


# ============================================================================
# Main Codegen Suite
# ============================================================================

class ScanCodegenSuite:
    """Main code generation suite for scan register programming.

    Orchestrates all code generation steps:
    1. IMCE code generation
    2. INode code generation
    3. Policy table generation
    4. Kernel wrapper generation

    Note: Memory layout, InstEdgeInfo, PolicyTableDict, and NoCPaths must be set up
    by scan_reg_policy_gen.py before calling this class.

    Usage:
        # First run policy generation
        policy_gen = ScanRegPolicyGenerator()
        policy_gen.construct_noc_path()
        policy_gen.gen_policy_table()
        policy_gen.add_edge_info(func_name="scan_reg")
        policy_gen.allocate(func_name="scan_reg")

        # Then generate code (no need to pass policy_table and noc_paths)
        suite = ScanCodegenSuite(
            func_name="scan_reg",
            build_dir="./build",
            scan_packet_count=128
        )
        suite.generate(scan_values)
    """

    def __init__(
        self,
        func_name: str,
        build_dir: str,
        scan_packet_count: int
    ):
        self.func_name = func_name
        self.build_dir = build_dir
        self.scan_packet_count = scan_packet_count
        self.host_os = os.getenv("IMCFLOW_HOST_OS", "linux")
        self.host_isa = os.getenv("IMCFLOW_HOST_ISA", "arm")

        # Create build directory
        os.makedirs(build_dir, exist_ok=True)
        self.func_dir = os.path.join(build_dir, func_name)
        os.makedirs(self.func_dir, exist_ok=True)

        # Create common_decl.h file (required for IMCE/INode compilation)
        common_decl = """
typedef short short16 __attribute__((ext_vector_type(16)));
__attribute__((noinline, used)) void __builtin_IMCE_STEP(void);
"""
        common_decl_path = os.path.join(build_dir, "common_decl.h")
        with open(common_decl_path, "w") as f:
            f.write(common_decl)

        # Verify that memory layout, InstEdgeInfo, PolicyTableDict, and NoCPaths are set up
        if func_name not in DevConfig().MemLayout:
            raise RuntimeError(
                f"Memory layout for '{func_name}' not found in DevConfig. "
                "Run scan_reg_policy_gen.allocate() first."
            )
        if func_name not in DevConfig().InstEdgeInfoDict:
            raise RuntimeError(
                f"InstEdgeInfo for '{func_name}' not found in DevConfig. "
                "Run scan_reg_policy_gen.add_edge_info() first."
            )
        if func_name not in DevConfig().PolicyTableDict:
            raise RuntimeError(
                f"PolicyTableDict for '{func_name}' not found in DevConfig. "
                "Run scan_reg_policy_gen.allocate() first."
            )
        if func_name not in DevConfig().NoCPaths:
            raise RuntimeError(
                f"NoCPaths for '{func_name}' not found in DevConfig. "
                "Run scan_reg_policy_gen.allocate() first."
            )

        # Get policy table and noc paths from DevConfig
        self.policy_table = DevConfig().PolicyTableDict[func_name]
        self.noc_paths = DevConfig().NoCPaths[func_name]

        # Allocate scan_data block if not already allocated
        self._ensure_scan_data_allocated()

        # Print memory layout summary
        print(self._get_memory_layout_summary())

    def _ensure_scan_data_allocated(self) -> None:
        """Ensure scan_data blocks are allocated in memory layout for each INode.

        Each INode needs its own scan_data block in its own data region.
        """
        mem_layout = DevConfig().MemLayout[self.func_name]

        # Allocate scan_data block for each INode in its own data region
        # Each INode sends to 4 IMCEs, each IMCE gets 2 packets = 8 packets per INode
        packets_per_inode = 8
        scan_data_size_per_inode = packets_per_inode * 32  # 256 bytes per INode

        for inode in NodeID.inodes():
            inode_data_name = f"{inode.name}_data"
            block_name = f"{inode.name}_scan_data"

            if inode_data_name not in mem_layout:
                raise RuntimeError(
                    f"Memory layout for {inode_data_name} not found. "
                    f"Run scan_reg_policy_gen.allocate() first."
                )

            # Check if block already exists
            existing_blocks = mem_layout[inode_data_name].blocks
            if any(block_name in str(block.id) for block in existing_blocks.values()):
                print(f"Scan data block for {inode.name} already allocated")
                continue

            # Allocate scan_data block for this INode
            scan_data_block = DataBlock(block_name, scan_data_size_per_inode)
            mem_layout[inode_data_name].allocate(scan_data_block, phase="init")
            print(f"Allocated {block_name}: {scan_data_size_per_inode} bytes in {inode_data_name}")

    def _get_memory_layout_summary(self) -> str:
        """Get a summary of the memory layout from DevConfig."""
        lines = [
            f"Scan Register Memory Layout for {self.func_name}",
            "=" * 60,
        ]

        mem_layout = DevConfig().MemLayout[self.func_name]
        total_size = 0

        # Collect all data blocks
        all_blocks = []
        for inode_data_name in mem_layout.keys():
            if isinstance(inode_data_name, str) and inode_data_name.endswith("_data"):
                inode_data = mem_layout[inode_data_name]
                # MemoryRegion.blocks returns dict {block_id: DataBlock}
                for block in inode_data.blocks.values():
                    all_blocks.append(block)
                    total_size += block.size

        lines.append(f"Total blocks: {len(all_blocks)}")
        lines.append(f"Total size: {total_size} bytes")
        lines.append("")
        lines.append("Blocks:")
        for block in sorted(all_blocks, key=lambda b: b.base_address):
            block_id_str = str(block.id) if not isinstance(block.id, str) else block.id
            lines.append(f"  {block_id_str:30s} @ {hex(block.base_address):8s} size={block.size:5d} bytes")

        return "\n".join(lines)

    def generate(self, scan_values: List[List[int]], fifo_id: int = 0) -> Dict[str, Path]:
        """Generate all code artifacts.

        Args:
            scan_values: List of scan register values (each is list of 16 int16)
            fifo_id: FIFO ID to use for communication

        Returns:
            Dict mapping artifact name to file path
        """
        if len(scan_values) != self.scan_packet_count:
            raise ValueError(
                f"Expected {self.scan_packet_count} scan packets, "
                f"got {len(scan_values)}"
            )

        # Set the codegen context for this function
        CodegenContext().set_func_name(self.func_name)

        try:
            generated_files = {}

            print("\n" + "="*60)
            print(f"GENERATING CODE FOR {self.func_name}")
            print("="*60)

            # 1. Generate IMCE code
            print("\n[1/5] Generating IMCE code...")
            # Each IMCE receives 2 packets (64 bytes = 2 scan registers)
            per_imce_packets = 2
            imce_builder = ScanImceCodeBlockBuilder(
                self.func_name, per_imce_packets
            )
            imce_builder.build(fifo_id)
            imce_code = imce_builder.generate()

            imce_path = Path(self.func_dir) / "imce.cpp"
            imce_path.write_text(imce_code)
            generated_files["imce.cpp"] = imce_path
            print(f"   Written: {imce_path}")

            DeviceCodegen("imce", self.build_dir, self.host_isa).handle_code_generation(
                self.func_name, imce_builder.codeblocks
            )

            # 2. Generate INode code
            print("\n[2/5] Generating INode code...")
            inode_builder = ScanInodeCodeBlockBuilder(self.func_name)
            inode_builder.build_initialization()
            inode_builder.build_scan_send(self.scan_packet_count, fifo_id, per_imce_packets)
            inode_builder.build_finalization()
            inode_code = inode_builder.generate()

            inode_path = Path(self.func_dir) / "inode.cpp"
            inode_path.write_text(inode_code)
            generated_files["inode.cpp"] = inode_path
            print(f"   Written: {inode_path}")

            # 3. Compile device code
            print("\n[3/5] Compiling device code...")
            DeviceCodegen("inode", self.build_dir, self.host_isa).handle_code_generation(
                self.func_name, inode_builder.codeblocks
            )
            print("   Device code compiled")

            # 4. Generate policy tables
            print("\n[4/5] Generating policy tables...")
            policy_codegen = ScanPolicyTableCodegen(
                self.func_name, self.build_dir, self.host_isa
            )
            policy_files = policy_codegen.generate(self.policy_table)
            for pf in policy_files:
                generated_files[pf.name] = pf

            # 5. Generate kernel wrapper
            print("\n[5/5] Generating kernel wrapper...")
            kernel_codegen = ScanKernelCodegen(
                self.func_name,
                self.scan_packet_count,
                host_os=self.host_os,
                use_polling=True,
            )
            kernel_code = kernel_codegen.generate(scan_values)

            kernel_path = Path(self.func_dir) / f"{self.func_name}_kernel.cc"
            kernel_path.write_text(kernel_code)
            generated_files["kernel.cc"] = kernel_path
            print(f"   Written: {kernel_path}")

            print("\n" + "="*60)
            print("CODE GENERATION COMPLETE")
            print("="*60)
            print(f"Output directory: {self.func_dir}")
            print(f"Generated {len(generated_files)} files")

            return generated_files

        finally:
            # Clear the codegen context when done
            CodegenContext().clear()


# ============================================================================
# Convenience Functions
# ============================================================================

def generate_scan_code_from_policy_gen(
    scan_values: List[List[int]],
    func_name: str = "scan_programming",
    build_dir: str = "./build",
    **kwargs
) -> Dict[str, Path]:
    """Generate scan register code from DevConfig.

    Note: scan_reg_policy_gen.allocate() must have already been called to register:
    - PolicyTableDict[func_name]
    - NoCPaths[func_name]
    - MemLayout[func_name]
    - InstEdgeInfoDict[func_name]

    Args:
        scan_values: List of scan register values
        func_name: Name for generated function (must match the one used in policy generator)
        build_dir: Build directory
        **kwargs: Additional arguments for ScanCodegenSuite

    Returns:
        Dict mapping artifact name to file path
    """
    codegen = ScanCodegenSuite(
        func_name=func_name,
        build_dir=build_dir,
        scan_packet_count=len(scan_values),
        **kwargs
    )
    return codegen.generate(scan_values)


def generate_default_scan_values(count: int) -> List[List[int]]:
    """Generate default scan register values for testing.

    Args:
        count: Number of scan packets to generate

    Returns:
        List of scan register values (each is list of 16 int16)
    """
    return [[i % 256] * 16 for i in range(count)]


def load_scan_values_from_npz(npz_files: Union[str, List[str]], imce_count: int = 16) -> List[List[int]]:
    """Load scan register values from NPZ file(s).

    Follows the same decoding logic as acim.ScanData:
    - 64 bytes → 512 bits → bit-reversed → split into reg0 (bits 256-512) and reg1 (bits 0-256)
    - reg0 corresponds to packet 0 (bytes 32-63, bit-reversed)
    - reg1 corresponds to packet 1 (bytes 0-31, bit-reversed)

    Args:
        npz_files: Either:
            - Single NPZ file path (same scan values for all IMCEs)
            - List of NPZ file paths (one per IMCE, must have exactly imce_count files)
        imce_count: Number of IMCEs (default: 16)

    Returns:
        List of scan packets (imce_count * 2 packets, each is list of 16 int16)
    """
    import numpy as np

    # Handle single file vs list of files
    if isinstance(npz_files, str):
        # Single file: use for all IMCEs
        npz_list = [npz_files] * imce_count
    else:
        npz_list = npz_files
        if len(npz_list) != imce_count:
            raise ValueError(
                f"Expected {imce_count} NPZ files (one per IMCE), got {len(npz_list)}"
            )

    all_packets = []

    for imce_idx, npz_file in enumerate(npz_list):
        # Load NPZ file
        if not os.path.exists(npz_file):
            raise FileNotFoundError(f"NPZ file not found: {npz_file}")

        data = np.load(npz_file, allow_pickle=True)
        if "arr_0" not in data:
            raise ValueError(f"NPZ file {npz_file} must contain 'arr_0' key")

        scan_bytes = data["arr_0"].astype(np.uint8)
        if scan_bytes.shape != (64,):
            raise ValueError(
                f"NPZ file {npz_file} arr_0 must be 64 bytes, got shape {scan_bytes.shape}"
            )

        # Follow ScanData.get_reg() logic:
        # 1. Convert bytes to bit string
        bit_list = [f'{byte:08b}' for byte in scan_bytes]
        bit_str = ''.join(bit_list)  # 512 bits total

        # 2. Reverse the entire bit string
        rev_bit_str = bit_str[::-1]

        # 3. Extract reg0 (bits 256-512) and reg1 (bits 0-256)
        # Note: reg0 comes from bytes 32-63, reg1 comes from bytes 0-31 (after reversal)
        reg1_bits = rev_bit_str[0:256]    # First 256 bits (bytes 0-31 reversed)
        reg0_bits = rev_bit_str[256:512]  # Last 256 bits (bytes 32-63 reversed)

        # 4. Convert bit strings to short16 packets (16 int16 values each)
        # Each int16 is 16 bits, little-endian within each 16-bit value
        def bits_to_short16(bits: str) -> List[int]:
            """Convert 256-bit string to 16 int16 values."""
            values = []
            for i in range(16):
                # Extract 16 bits for this int16 (little-endian bit order within the int16)
                start = i * 16
                end = start + 16
                bits_16 = bits[start:end]

                # Reverse bits for little-endian int16 interpretation
                bits_16_rev = bits_16[::-1]

                # Convert to int16 (signed)
                val = int(bits_16_rev, 2)
                # Handle signed conversion (2's complement for negative values)
                if val >= 32768:
                    val -= 65536
                values.append(val)
            return values

        # packet_0 from reg0 (bytes 32-63)
        # packet_1 from reg1 (bytes 0-31)
        packet_0 = bits_to_short16(reg0_bits)
        packet_1 = bits_to_short16(reg1_bits)

        all_packets.append(packet_0)
        all_packets.append(packet_1)

    return all_packets


def load_scan_values_from_directory(
    npz_dir: str,
    imce_ids: Optional[List[str]] = None
) -> List[List[int]]:
    """Load scan values from a directory of NPZ files.

    Args:
        npz_dir: Directory containing NPZ files
        imce_ids: List of IMCE IDs (e.g., ["imce_0_1", "imce_0_2", ...])
                  If None, uses default naming: imce_{h}_{w}

    Returns:
        List of scan packets (imce_count * 2 packets)
    """
    if imce_ids is None:
        # Default IMCE IDs for 4x4 grid
        imce_ids = [
            f"imce_{h}_{w}"
            for h in range(4)
            for w in range(1, 5)
        ]

    npz_files = []
    for imce_id in imce_ids:
        npz_path = os.path.join(npz_dir, f"{imce_id}.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ file not found for {imce_id}: {npz_path}")
        npz_files.append(npz_path)

    return load_scan_values_from_npz(npz_files, imce_count=len(imce_ids))


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Command-line interface for scan register code generation."""
    import argparse
    import sys
    import os

    # Add TVM to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../python"))

    parser = argparse.ArgumentParser(
        description="Generate scan register programming code for IMCFlow"
    )
    parser.add_argument(
        "--func-name",
        default="program_scan_reg",
        help="Function name (default: program_scan_reg)"
    )
    parser.add_argument(
        "--build-dir",
        default="./build",
        help="Build directory (default: ./build)"
    )
    parser.add_argument(
        "--scan-count",
        type=int,
        default=2,
        help="Number of scan register packets (default: 2)"
    )
    parser.add_argument(
        "--run-policy-gen",
        action="store_true",
        help="Run policy generation first"
    )
    parser.add_argument(
        "--scan-npz",
        type=str,
        help="NPZ file(s) for scan values. Either:\n"
             "  - Single NPZ file (same for all IMCEs)\n"
             "  - Directory containing {imce_id}.npz files\n"
             "  - Comma-separated list of 16 NPZ files (one per IMCE)"
    )
    parser.add_argument(
        "--scan-npz-dir",
        type=str,
        help="Directory containing NPZ files named {imce_id}.npz"
    )

    args = parser.parse_args()

    # Import policy generator (use relative import since we're in the same directory)
    try:
        from scan_reg_policy_gen import ScanRegPolicyGenerator
    except ImportError:
        # Fallback: add current directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from scan_reg_policy_gen import ScanRegPolicyGenerator

    # Generate or load policy
    if args.run_policy_gen:
        print("Running policy generation...")
        policy_gen = ScanRegPolicyGenerator()
        policy_gen.construct_noc_path()
        policy_gen.gen_policy_table()
        policy_gen.add_edge_info(func_name=args.func_name)
        policy_gen.allocate(func_name=args.func_name)
    else:
        print("Error: Policy generation must be run. Use --run-policy-gen")
        sys.exit(1)

    # Generate or load scan values
    if args.scan_npz_dir:
        # Load from directory
        print(f"Loading scan values from directory: {args.scan_npz_dir}")
        scan_values = load_scan_values_from_directory(args.scan_npz_dir)
        print(f"Loaded {len(scan_values)} scan packets from NPZ files")
    elif args.scan_npz:
        # Load from NPZ file(s)
        if os.path.isdir(args.scan_npz):
            # It's a directory
            print(f"Loading scan values from directory: {args.scan_npz}")
            scan_values = load_scan_values_from_directory(args.scan_npz)
        elif "," in args.scan_npz:
            # Comma-separated list of files
            npz_files = [f.strip() for f in args.scan_npz.split(",")]
            print(f"Loading scan values from {len(npz_files)} NPZ files")
            scan_values = load_scan_values_from_npz(npz_files)
        else:
            # Single file
            print(f"Loading scan values from NPZ file: {args.scan_npz}")
            scan_values = load_scan_values_from_npz(args.scan_npz)
        print(f"Loaded {len(scan_values)} scan packets from NPZ files")
    else:
        # Generate default values
        print(f"Generating default scan values ({args.scan_count} packets)")
        scan_values = generate_default_scan_values(args.scan_count)

    # Generate code (policy_gen data is now in DevConfig)
    files = generate_scan_code_from_policy_gen(
        scan_values=scan_values,
        func_name=args.func_name,
        build_dir=args.build_dir
    )

    print("\nGenerated files:")
    for name, path in sorted(files.items()):
        print(f"  {name:20s} -> {path}")

    # Automatically build the executable
    print("\n" + "="*60)
    print("BUILDING EXECUTABLE")
    print("="*60)

    import subprocess

    # Get the host ISA from environment
    host_isa = os.getenv("IMCFLOW_HOST_ISA", "arm").lower()

    # Path to scan_executable_make directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scan_executable_make_dir = os.path.join(current_dir, "scan_executable_make")
    build_script = os.path.join(scan_executable_make_dir, "build.sh")

    if not os.path.exists(build_script):
        print(f"Warning: Build script not found at {build_script}")
        print("Skipping executable build")
        return 0

    # Run build.sh with the host ISA
    print(f"Running build script for {host_isa.upper()} architecture...")
    try:
        result = subprocess.run(
            [build_script, host_isa],
            cwd=scan_executable_make_dir,
            check=True,
            capture_output=False
        )
        print("\n" + "="*60)
        print("EXECUTABLE BUILD COMPLETE")
        print("="*60)
        print(f"Executable: scan_executable_make/build/program_scan_reg")
    except subprocess.CalledProcessError as e:
        print(f"\nError: Build failed with exit code {e.returncode}")
        print("You can manually build by running:")
        print(f"  cd {scan_executable_make_dir}")
        print(f"  ./build.sh {host_isa}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
