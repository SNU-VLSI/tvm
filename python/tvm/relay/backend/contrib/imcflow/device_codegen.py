import os
import tempfile
import subprocess
import logging
from tvm.contrib.imcflow import NodeID, DataBlock
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from tvm.relay.backend.contrib.imcflow.codeblock import *
import pdb


class DeviceCodegen:
  def __init__(self, target, build_dir="/tmp"):
    assert target in ["inode", "imce"], f"Unknown target: {target}"
    self.target = target
    self.build_dir = build_dir
    self.compile_options = f"-O1 --target={target.upper()} -c -fPIC -mllvm=-force-hardware-loops -mllvm=-force-nested-hardware-loop"
    self.objcopy_options = "-O binary --only-section=.text"
    self.lld_options = "-e 0 -Ttext 0x0"
    self.ld_options = "-r -b binary"
    self.func_dir = None
    logging.basicConfig(level=logging.INFO)

  def handle_code_generation(self, func_name, codeblock_manager: NodeCodeBlockManager):
    """
    The main entry point for DeviceCodegen.
    Handles code generation, saving to file, compilation, linking, and host object creation.
    """
    self.func_dir = os.path.join(self.build_dir, func_name)
    if not os.path.exists(self.func_dir):
      os.makedirs(self.func_dir)
    logging.info(
        f"Generating {self.target} code for function: {func_name} in {self.func_dir}")

    code = codeblock_manager.generate()
    cpp_name = self.save_target_code_to_file(code)
    obj_map = self.compile_target_code(cpp_name)
    self.update_device_config_with_obj_info(obj_map)

  def save_target_code_to_file(self, code: str):
    cpp_name = f"{self.target}.cpp"
    with open(os.path.join(self.func_dir, cpp_name), "w") as file:
      file.write(code)
    return cpp_name

  def compile_target_code(self, cpp_name: str):
    obj_map = {}
    if not cpp_name.endswith(".cpp"):
      raise ValueError(f"Invalid cpp_name: {cpp_name}")

    nodes = NodeID.inodes() if self.target == "inode" else NodeID.imces()
    for node in nodes:
      file_name = f"{node.name}_imem"
      obj_file = f"{file_name}.o"
      out_file = f"{file_name}.out"
      bin_file = f"{file_name}.bin"
      host_obj_file = f"{file_name}.host.o"
      self.compile_cpp_to_object(cpp_name, obj_file, node)
      self.link_object_to_binary(obj_file, out_file)
      self.extract_text_section(out_file, bin_file)

      # replace with padded binary (padded to 32-byte boundary)
      if self.target == "inode":
        self.pad_inode_bin_inplace(bin_file, stride=32)
      # replace with padded binary (padded to 32-byte boundary)
      if self.target == "imce":
        self.pad_imce_bin_inplace(bin_file, inst_size=4, stride=32)

      # flip byte-order of imce binary (big endian to little endian)
      # FIXME: remove this after llvm generates correct little endian binary.
      # currently we're using the imcflow_bigendian branch of the llvm_project
      self.flip_byte_order(bin_file)

      self.create_host_object(bin_file, host_obj_file)
      obj_map[node] = host_obj_file

    return obj_map

  def compile_cpp_to_object(self, cpp_name: str, obj_file: str, node: NodeID):
    # FIXME: change the INODE_hid/INODE_wid to lowercase in llvm for consistency
    hid_str = "imce_hid" if self.target == "imce" else "INODE_hid"
    wid_str = "imce_wid" if self.target == "imce" else "INODE_wid"

    command = [
        "clang",
        *self.compile_options.split(),
        f"-mllvm=-{hid_str}={node.to_coord(0)}",
        f"-mllvm=-{wid_str}={node.to_coord(1)}",
        "-o", obj_file,
        cpp_name
    ]
    subprocess.run(command, cwd=self.func_dir, check=True)

  def link_object_to_binary(self, obj_file: str, out_file: str):
    command = ["ld.lld", *self.lld_options.split(), "-o", out_file, obj_file]
    subprocess.run(command, cwd=self.func_dir, check=True)

  def extract_text_section(self, out_file: str, bin_file: str):
    command = ["llvm-objcopy", *self.objcopy_options.split(), out_file,
               bin_file]
    subprocess.run(command, cwd=self.func_dir, check=True)

  def pad_inode_bin_inplace(self, bin_file: str, stride=32):
    """Pad each instruction to stride(32)-byte boundaries, overwriting input file"""
    # Read all data first
    bin_path = os.path.join(self.func_dir, bin_file)
    with open(bin_path, 'rb') as infile:
      data = infile.read()

    # Check if data length is multiple of 4
    if len(data) % 4 != 0:
      raise ValueError("Input file size must be multiple of 4 bytes")

    # Create padded data
    padded_data = bytearray()
    for i in range(0, len(data) // stride):
      instruction = data[i:i+stride]
      padded_data.extend(instruction)

    # Add remaining data
    padded_data.extend(data[len(data) // stride * stride:])
    padded_data.extend(b'\x00' * (stride - (len(data) % stride)))

    # Write back to same file
    with open(bin_path, 'wb') as outfile:
      outfile.write(padded_data)

  def pad_imce_bin_inplace(self, bin_file: str, inst_size=4, stride=32):
    """Pad each instruction to stride(32)-byte boundaries, overwriting input file"""
    # Read all data first
    bin_path = os.path.join(self.func_dir, bin_file)
    with open(bin_path, 'rb') as infile:
      data = infile.read()

    # Check if data length is multiple of 4
    if len(data) % 4 != 0:
      raise ValueError("Input file size must be multiple of 4 bytes")

    # Create padded data
    padded_data = bytearray()
    for i in range(0, len(data), inst_size):
      instruction = data[i:i+inst_size]
      padded_data.extend(instruction)
      padded_data.extend(b'\x00' * (stride - inst_size))

    # Write back to same file
    with open(bin_path, 'wb') as outfile:
      outfile.write(padded_data)

  def flip_byte_order(self, bin_file: str):
    """Convert big endian binary to little endian by swapping byte order within each 4-byte word"""
    bin_path = os.path.join(self.func_dir, bin_file)

    # Read the binary file
    with open(bin_path, 'rb') as f:
      data = bytearray(f.read())

    # Check if data length is multiple of 4 (32-bit instructions)
    if len(data) % 4 != 0:
      raise ValueError(f"Binary file {bin_file} size ({len(data)}) is not multiple of 4.")

    # Convert big endian to little endian by swapping bytes within each 4-byte word
    for i in range(0, len(data), 4):
      if i + 3 < len(data):
        data[i], data[i+1], data[i+2], data[i+3] = data[i+3], data[i+2], data[i+1], data[i]

    # Write the converted data back to the file
    with open(bin_path, 'wb') as f:
      f.write(data)

  def create_host_object(self, bin_file: str, host_obj_file: str):
    command = ["aarch64-linux-gnu-ld", *
               self.ld_options.split(), "-o", host_obj_file, bin_file]
    subprocess.run(command, cwd=self.func_dir, check=True)

  def get_object_size(self, obj_file: str, key: str = "text"):
    command = ["llvm-size", obj_file]
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, cwd=self.func_dir)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
      print(f"Error executing llvm-size: {stderr}")
      return None

    try:
      # Parse the output to extract the size
      std_out_lines = stdout.splitlines()
      keys = std_out_lines[0].split()   # Get the first line
      sizes = std_out_lines[1].split()  # Get the second line
      if key not in keys:
        print(f"Key '{key}' not found in llvm-size output: {stdout}")
        return None
      index = keys.index(key)  # Find the index of the key
      size = int(sizes[index])  # Get the corresponding size
      return size
    except (IndexError, ValueError):
      print(f"Error parsing llvm-size output: {stdout}")
      return None

  def update_device_config_with_obj_info(self, obj_map: dict[NodeID, str]):
    for node, obj_file in obj_map.items():
      size = self.get_object_size(obj_file, key="data")
      if size is not None:
        db = DataBlock(f"{node.name}_imem", size)
        if self.target == "inode":
          self.allocate_db(db, f"{node.name}_inst")
        else:
          self.allocate_db(db, f"{node.master().name}_data")
          self.insert_db_to_inst_edge_info(db, node)
      else:
        print(f"Failed to allocate imem for {obj_file}")
    print(DevConfig().MemLayout)

  def allocate_db(self, data_block: DataBlock, region: str):
    DevConfig().MemLayout[region].allocate(data_block)

  def insert_db_to_inst_edge_info(self, db: DataBlock, node: NodeID):
    edge_info = DevConfig().get_inst_edge_info(node)
    edge_info.set_data_block(db)
