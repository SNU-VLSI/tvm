import os
import subprocess
import logging
from tvm.contrib.imcflow import NodeID
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from tvm.relay.backend.contrib.imcflow.codeblock import *
import pdb

class DeviceCodegen:
  def __init__(self, target, output_dir="/tmp"):
    assert target in ["inode", "imce"], f"Unknown target: {target}"
    self.target = target
    self.output_dir = output_dir
    self.compile_options = f"-O1 --target={target} -c -fPIC -mllvm=-force-hardware-loops -mllvm=-force-nested-hardware-loop"
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
    pdb.set_trace()
    code = codeblocks.generate()
    pdb.set_trace()
    cpp_name = self.save_target_code_to_file(code, func_name)
    self.compile_target_code(cpp_name)

  def save_target_code_to_file(self, code, func_name):
    cpp_name = os.path.join(self.output_dir, f"{func_name}_{self.target}.cpp")
    with open(cpp_name, "w") as file:
      file.write(code)
    return cpp_name

  def compile_target_code(self, cpp_name):
    if not cpp_name.endswith(".cpp"):
      raise ValueError(f"Invalid cpp_name: {cpp_name}")

    base_name = cpp_name[:-4]
    nodes = NodeID.inodes() if self.target == "inode" else NodeID.imces()
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

  def compile_cpp_to_object(self, cpp_name, obj_file, node):
    command = [
        "clang",
        *self.compile_options.split(),
        f"-mllvm=-{self.target}_hid={node.to_coord(0)}",
        f"-mllvm=-{self.target}_wid={node.to_coord(1)}",
        "-o", obj_file,
        cpp_name
    ]
    subprocess.run(command, check=True)

  def link_object_to_binary(self, obj_file, out_file):
    command = ["ld.lld", *self.lld_options.split(), "-o", out_file, obj_file]
    subprocess.run(command, check=True)

  def extract_text_section(self, out_file, bin_file):
    command = ["llvm-objcopy", *self.objcopy_options.split(), out_file,
               bin_file]
    subprocess.run(command, check=True)

  def create_host_object(self, bin_file, host_obj_file):
    command = ["ld", *self.ld_options.split(), "-o", host_obj_file, bin_file]
    subprocess.run(command, check=True)