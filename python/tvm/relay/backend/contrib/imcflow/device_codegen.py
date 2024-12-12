import os
import logging

class DeviceCodegen:
  def __init__(self, output_dir="/tmp"):
    self.output_dir = output_dir
    self.compile_options = "-O1 --target=INODE -c -fPIC -mllvm=-force-hardware-loops -mllvm=-force-nested-hardware-loop"
    self.objcopy_options = "-O binary --only-section=.text"
    self.lld_options = "-e 0 -Ttext 0x0"
    self.ld_options = "-r -b binary"
    logging.basicConfig(level=logging.INFO)

  def handle_code_generation(self, op_name, args):
    logging.info(f"Generating inode code for operator: {op_name}")
    inode_code = self.generate_target_code(op_name, args, "inode")
    cpp_name = self.save_target_code_to_file(inode_code, op_name, "inode")
    self.compile_target_code(cpp_name, "inode")

    logging.info(f"Generating imce code not implemented yet for: {op_name}")

  def generate_target_code(self, op_name, args, target):
    if target == "inode":
      code = f"// Generated code for {op_name} with args {args} targeting {target}\n"
    elif target == "imce":
      raise NotImplementedError("IMCE codegen not implemented yet.")
    else:
      raise ValueError(f"Unknown target: {target}")

    return code

  def save_target_code_to_file(self, code, op_name, target):
    cpp_name = os.path.join(self.output_dir, f"{op_name}_{target}.cpp")
    with open(cpp_name, "w") as file:
      file.write(code)
    return cpp_name

  def compile_target_code(self, cpp_name, target):
    if not cpp_name.endswith(".cpp"):
      raise ValueError(f"Invalid cpp_name: {cpp_name}")

    base_name = cpp_name[:-4]
    if target == "inode":
      hids = list(range(ImcflowDeviceConfig.INODE_NUM))
      wids = [0]
    elif target == "imce":
      hids = list(range(ImcflowDeviceConfig.IMCE_H_NUM))
      wids = list(range(1, 1 + ImcflowDeviceConfig.IMCE_W_NUM))
    else:
      raise ValueError(f"Unknown target: {target}")

    for hid in hids:
      for wid in wids:
        hid_str = str(hid)
        wid_str = str(wid)
        obj_file = f"{base_name}_{hid_str}.o"
        out_file = f"{base_name}_{hid_str}.out"
        bin_file = f"{base_name}_{hid_str}.bin"
        host_obj_file = f"{base_name}_{hid_str}.host.o"

        self.compile_cpp_to_object(cpp_name, obj_file, hid_str, wid_str)
        self.link_object_to_binary(obj_file, out_file)
        self.extract_text_section(out_file, bin_file)
        self.create_host_object(bin_file, host_obj_file)

  def compile_cpp_to_object(self, cpp_name, obj_file, hid, wid):
    command = [
        "clang",
        *self.compile_options.split(),
        f"-mllvm=-INODE_hid={hid}",
        f"-mllvm=-INODE_wid={wid}",
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