import os
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
    logging.basicConfig(level=logging.INFO)

  def handle_code_generation(self, func_name, codeblocks: CodeBlocks):
    """
    The main entry point for DeviceCodegen.
    Handles code generation, saving to file, compilation, linking, and host object creation.
    """
    logging.info(f"Generating {self.target} code for function: {func_name}")
    code = codeblocks.generate()
    cpp_name = self.save_target_code_to_file(code, func_name)
    obj_map = self.compile_target_code(cpp_name)
    self.allocate_imem(obj_map)

  def save_target_code_to_file(self, code: str, func_name: str):
    cpp_name = os.path.join(self.build_dir, f"{func_name}_{self.target}.cpp")
    with open(cpp_name, "w") as file:
      file.write(code)
    return cpp_name

  def compile_target_code(self, cpp_name: str):
    obj_map = {}
    if not cpp_name.endswith(".cpp"):
      raise ValueError(f"Invalid cpp_name: {cpp_name}")

    base_name = cpp_name[:-4]
    nodes = NodeID.inodes() if self.target == "inode" else NodeID.imces()
    if self.target == "inode":
      return obj_map
    for node in nodes:
      file_name = f"{base_name}_{node.name}"
      obj_file = f"{file_name}.o"
      out_file = f"{file_name}.out"
      bin_file = f"{file_name}.bin"
      host_obj_file = f"{file_name}.host.o"
      self.compile_cpp_to_object(cpp_name, obj_file, node)
      self.link_object_to_binary(obj_file, out_file)
      self.extract_text_section(out_file, bin_file)
      self.create_host_object(bin_file, host_obj_file)
      obj_map[node] = obj_file

    return obj_map

  def compile_cpp_to_object(self, cpp_name: str, obj_file: str, node: NodeID):
    command = [
        "clang",
        *self.compile_options.split(),
        f"-mllvm=-{self.target}_hid={node.to_coord(0)}",
        f"-mllvm=-{self.target}_wid={node.to_coord(1)}",
        "-o", obj_file,
        cpp_name
    ]
    subprocess.run(command, check=True)

  def link_object_to_binary(self, obj_file: str, out_file: str):
    command = ["ld.lld", *self.lld_options.split(), "-o", out_file, obj_file]
    subprocess.run(command, check=True)

  def extract_text_section(self, out_file: str, bin_file: str):
    command = ["llvm-objcopy", *self.objcopy_options.split(), out_file,
               bin_file]
    subprocess.run(command, check=True)

  def create_host_object(self, bin_file: str , host_obj_file: str):
    command = ["ld", *self.ld_options.split(), "-o", host_obj_file, bin_file]
    subprocess.run(command, check=True)

  def get_object_size(self, obj_file: str):
    command = ["llvm-size", obj_file]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
      print(f"Error executing llvm-size: {stderr}")
      return None

    try:
      # Parse the output to extract the size
      size_line = stdout.splitlines()[1]  # Get the second line
      size = int(size_line.split()[0])  # Extract the first number
      return size
    except (IndexError, ValueError):
      print(f"Error parsing llvm-size output: {stdout}")
      return None

  def allocate_imem(self, obj_map: dict[NodeID, str]):
    for node, obj_file in obj_map.items():
      size = self.get_object_size(obj_file)
      db = DataBlock(f"{node.name}_imem", size)
      region = f"{node.master().name}_data"
      if size is not None:
        DevConfig().MemLayout[region].allocate(db)
      else:
        print(f"Failed to allocate imem for {obj_file}")
    print(DevConfig().MemLayout)