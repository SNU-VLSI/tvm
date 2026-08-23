import os
import tempfile
import subprocess
import logging
from tvm.contrib.imcflow import NodeID, DataBlock
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from tvm.relay.backend.contrib.imcflow.codeblock import *
import pdb


# --------------------------------------------------------------------------
# Size-aware BN/minmax packing 2-pass de-fuse -- pass-1 IMEM overflow COLLECT
# mode (see python/tvm/relay/op/contrib/imcflow.py and
# transform.stamp_pack_atomic_keys).
#
# _check_imem_capacity normally HARD-RAISES on an IMCE program that overflows
# the wrap-around IMEM. During pass 1 of the 2-pass compile the driver wants
# codegen to COMPLETE (collecting EVERY overflowing node's stable atomic key in
# one shot) instead of aborting on the first overflow. The driver installs a
# fresh list via set_imem_overflow_collect([]) before pass-1 codegen; while that
# list is not None the guard APPENDS the overflowing node's atomic key to it and
# returns (no raise). When it is None (default) the guard raises exactly as
# before -> stock/lever-OFF codegen is byte-identical.
_IMEM_OVERFLOW_COLLECT = None  # None => raise (default); a list => collect+continue


def set_imem_overflow_collect(list_or_None):
  """Install (or clear) the pass-1 IMEM-overflow collect sink.

  Pass a fresh ``list`` to enter collect mode: subsequent IMEM overflows append
  their atomic key ``(orig_conv_name, oc_id, ic_id)`` to it and codegen keeps
  going. Pass ``None`` to restore the default hard-raise behavior. Returns the
  previously installed sink so callers can restore it.
  """
  global _IMEM_OVERFLOW_COLLECT
  prev = _IMEM_OVERFLOW_COLLECT
  _IMEM_OVERFLOW_COLLECT = list_or_None
  return prev


def get_imem_overflow_collect():
  """Return the current collect sink (a list) or None if collect mode is off."""
  return _IMEM_OVERFLOW_COLLECT


def _resolve_overflow_atomic_key(func_name, node):
  """Recover the stable atomic key of the IMCE node that overflowed IMEM.

  ``node`` is a PHYSICAL imce NodeID (from NodeID.imces()); ``func_name`` scopes
  it to one imcflow function. Recipe (confirmed against
  transform.constructActiveIMCEDict / constructNoCPathDict, which do the same
  gid->HWNodeMap->NodeID lookup):

    1. CustomIDInFunc()[func_name] -> the list of graph gids in this function.
    2. HWNodeMap maps gid -> physical NodeID. Invert it, restricted to this
       function's gids, to find the candidate composite gid(s) placed on `node`.
    3. CustomIDToNode()[gid] -> the composite Call. Its op is the composite
       Function stamped by stamp_pack_atomic_keys; read
       op.attrs["imcflow_atomic_key"] and parse "name|oc|ic" back to
       (orig_conv_name, oc_id, ic_id).

  Returns the key tuple, or None if it cannot be resolved (unstamped node,
  missing mapping, etc.) -- the caller then collects nothing for that node.
  """
  from tvm.relay.op.contrib.imcflow import CustomIDInFunc, CustomIDToNode

  cfg = DevConfig()
  hw_map = cfg.HWNodeMap
  id_in_func = CustomIDInFunc()
  id_to_node = CustomIDToNode()

  func_gids = id_in_func.get(func_name)
  if not func_gids:
    return None

  # Candidate gids in THIS function that HWNodeMap places on the physical `node`.
  candidates = []
  for gid in func_gids:
    mapped = hw_map.get(gid) if gid in hw_map else None
    if isinstance(mapped, NodeID) and mapped == node:
      candidates.append(gid)

  for gid in candidates:
    try:
      call = id_to_node[gid]
    except (KeyError, TypeError):
      continue
    op = getattr(call, "op", None)
    attrs = getattr(op, "attrs", None)
    if attrs is None:
      continue
    key_str = None
    try:
      if "imcflow_atomic_key" in attrs:
        key_str = attrs["imcflow_atomic_key"]
    except (TypeError, KeyError):
      key_str = None
    if not key_str:
      continue
    parts = str(key_str).split("|")
    if len(parts) != 3:
      continue
    try:
      return (parts[0], int(parts[1]), int(parts[2]))
    except ValueError:
      continue
  return None


class DeviceCodegen:
  def __init__(self, target, build_dir="/tmp", host_isa="arm"):
    assert target in ["inode", "imce"], f"Unknown target: {target}"
    self.target = target
    self.build_dir = build_dir
    # self.inode_compile_options = f"-O1 --target={target.upper()} -c -fPIC -mllvm=-force-nested-hardware-loop -mllvm=-debug -mllvm=--debug-pass=Details -mllvm=-print-after-all"
    self.inode_compile_options = f"-O1 --target={target.upper()} -c -fPIC -mllvm=-force-nested-hardware-loop"
    self.imce_compile_options = f"-O1 --target={target.upper()} -c -fPIC -mllvm=-force-hardware-loops -mllvm=-force-nested-hardware-loop"
    self.objcopy_options = "-O binary --only-section=.text"
    self.lld_options = "-e 0 -Ttext 0x0"
    self.ld_options = "-r -b binary"
    self.func_dir = None
    self.host_isa = host_isa
    logging.basicConfig(level=logging.INFO)

  def handle_code_generation(self, func_name, codeblock_manager: NodeCodeBlockManager):
    """
    The main entry point for DeviceCodegen.
    Handles code generation, saving to file, compilation, linking, and host object creation.
    """
    self.func_dir = os.path.join(self.build_dir, func_name)
    os.makedirs(self.func_dir, exist_ok=True)

    logging.info(
        f"Generating {self.target} code for function: {func_name} in {self.func_dir}")

    # C1b (C) wave-launch: for a multi-wave IMCE region, emit + compile one
    # program per (core, wave) so each wave has its own IMEM blob (the inode
    # re-WR_IMEMs the core per wave under conv-cap=1, weights unchanged). The
    # inode target is always single-program (policy/weights static), and any
    # single-wave region (every non-merged region) takes the stock path below ->
    # byte-identical.
    waves = codeblock_manager.wave_indices() if self.target == "imce" else [0]
    if self.target == "imce" and len(waves) > 1:
      obj_map = self._handle_wave_code_generation(func_name, codeblock_manager, waves)
      self.update_device_config_with_obj_info(func_name, obj_map, wave_aware=True)
      return

    code = codeblock_manager.generate()
    cpp_name = self.save_target_code_to_file(code)
    obj_map = self.compile_target_code(cpp_name)
    self.update_device_config_with_obj_info(func_name, obj_map)

  def _handle_wave_code_generation(self, func_name, codeblock_manager, waves):
    """C1b (C): generate + compile per-(core, wave) IMCE programs. Returns
    obj_map keyed by (NodeID, wave) -> host object file. Each wave gets its own
    imce_wave{k}.cpp (wave-filtered body) compiled per core, so a core reused
    across waves yields a DISTINCT IMEM blob per wave."""
    obj_map = {}
    for k in waves:
      code = codeblock_manager.generate(wave=k)
      cpp_name = f"{self.target}_wave{k}.cpp"
      with open(os.path.join(self.func_dir, cpp_name), "w") as f:
        f.write(code)
      for node in NodeID.imces():
        # Only cores that actually have wave-k code produce a blob for wave k.
        # Detect via a nonempty per-core wave-k body (cheap: check tagged blocks).
        if not self._core_has_wave(codeblock_manager, node, k):
          continue
        stem = f"{node.name}_imem_wave{k}"
        obj_file = f"{stem}.o"
        out_file = f"{stem}.out"
        bin_file = f"{stem}.bin"
        host_obj_file = f"{stem}.host.o"
        self.compile_cpp_to_object(cpp_name, obj_file, node)
        self.link_object_to_binary(obj_file, out_file)
        self.extract_text_section(out_file, bin_file)
        self.pad_imce_bin_inplace(bin_file, inst_size=4, stride=32)
        self.flip_byte_order(bin_file)
        self.create_host_object(bin_file, host_obj_file)
        obj_map[(node, k)] = host_obj_file
    return obj_map

  @staticmethod
  def _core_has_wave(codeblock_manager, node, wave):
    for phase in codeblock_manager.blocks[node]:
      for cb in codeblock_manager.blocks[node][phase]:
        if getattr(cb, "_wave", 0) == wave:
          return True
    return False

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
        *(self.imce_compile_options if self.target == "imce" else self.inode_compile_options).split(),
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
    for i in range(0, len(data), stride):
      if (len(data) - i) < stride:
        break
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
    # Extract function name from func_dir to use as symbol prefix
    func_name = os.path.basename(self.func_dir)
    temp_obj_file = f"{host_obj_file}.tmp"

    # First, create the object file with default symbols
    if self.host_isa == "arm":
      command = ["aarch64-linux-gnu-ld", *
                self.ld_options.split(), "-o", temp_obj_file, bin_file]
    elif self.host_isa == "x86":
      command = ["ld", *self.ld_options.split(),
                 "-o", temp_obj_file, bin_file]
    else:
      raise ValueError(f"Unknown host ISA: {self.host_isa}")
    subprocess.run(command, cwd=self.func_dir, check=True)

    # Generate symbol name from binary filename (ld converts filename to symbol)
    # e.g., "inode_0_0_imem.bin" -> "_binary_inode_0_0_imem_bin"
    bin_name_base = bin_file.replace('.', '_')
    old_symbol_prefix = f"_binary_{bin_name_base}"
    new_symbol_prefix = f"_binary_{func_name}_{bin_name_base}"

    # Create a redefine-sym file for objcopy
    # redefine_file = os.path.join(self.func_dir, f"{host_obj_file}.redefine")
    redefine_file = f"{host_obj_file}.redefine"
    with open(os.path.join(self.func_dir, redefine_file), 'w') as f:
      f.write(f"{old_symbol_prefix}_start {new_symbol_prefix}_start\n")
      f.write(f"{old_symbol_prefix}_end {new_symbol_prefix}_end\n")
      f.write(f"{old_symbol_prefix}_size {new_symbol_prefix}_size\n")

    # Use objcopy to rename the symbols
    objcopy_cmd = "aarch64-linux-gnu-objcopy" if self.host_isa == "arm" else "objcopy"
    rename_command = [objcopy_cmd, f"--redefine-syms={redefine_file}", temp_obj_file, host_obj_file]
    subprocess.run(rename_command, cwd=self.func_dir, check=True)

    # Remove temporary files
    os.remove(os.path.join(self.func_dir, temp_obj_file))
    os.remove(os.path.join(self.func_dir, redefine_file))

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

  # IMCE instruction memory depth (words) from imcflow RTL parameters.yaml
  # (IMCE_IMEM_DEPTH: 256). The PC is clog2(256)=8 bits, so it WRAPS at 256:
  # a program whose .text exceeds 256 words silently overwrites its own entry
  # (word 256 lands on addr 0), corrupting recv_cfg/prolog -> the core resumes
  # mid-body reading un-latched CREG -> X-propagation fatal on BUGFIX-off RTL.
  # We detect that here (the ONLY place the true compiled program size is known)
  # and fail loudly instead of emitting a silently-corrupt image. This check is
  # inert for every program that fits (all lever-OFF / stock builds), so codegen
  # output is byte-identical unless a program would actually overflow.
  # Default 256 matches the stock RTL (parameters.yaml IMCE_IMEM_DEPTH: 256).
  # For the IMEM-enlargement diagnostic experiment, the RTL is rebuilt with a
  # bumped depth (e.g. 512) and IMCFLOW_IMCE_IMEM_DEPTH is set to the SAME value
  # so this compile-time guard stays honest against the actual built IMEM depth
  # (otherwise it would falsely fire at 297>256 and block the build). Default is
  # 256 -> byte-identical to stock behavior.
  # BIG_IMEM builds enlarge the IMCE imem to 1024 words (params.svh ifdef);
  # default the guard accordingly so packed conv+bn programs >256w compile.
  # IMCFLOW_IMCE_IMEM_DEPTH still overrides for one-off experiments.
  _BIG = os.environ.get("IMCFLOW_BIG_IMEM", "0") == "1"
  IMCE_IMEM_DEPTH_WORDS = int(os.environ.get("IMCFLOW_IMCE_IMEM_DEPTH", "1024" if _BIG else "256"))
  # The imem host object embeds one instruction per fixed-width slot; the emitted
  # binary blob (queried below via key="data") is words * IMEM_SLOT_BYTES.
  IMCE_IMEM_SLOT_BYTES = 32

  def _check_imem_capacity(self, node, obj_file, data_bytes, func_name=None):
    """Hard-fail if an imce program overflows the wrap-around IMEM.

    data_bytes is the imem host-object blob size (words * IMCE_IMEM_SLOT_BYTES),
    already read by the caller; we reuse it so we do not shell out to llvm-size
    a second time.

    In pass-1 COLLECT mode (set_imem_overflow_collect installed a list), an
    overflow does NOT raise: instead the overflowing node's stable atomic key is
    appended to the collect list and codegen CONTINUES, so pass 1 records EVERY
    overflowing node in one shot. In the default mode (collect sink is None) the
    original hard-raise is preserved, so lever-OFF codegen is byte-identical.
    """
    if self.target != "imce" or data_bytes is None:
      return
    cap_bytes = self.IMCE_IMEM_DEPTH_WORDS * self.IMCE_IMEM_SLOT_BYTES
    if data_bytes > cap_bytes:
      words = data_bytes // self.IMCE_IMEM_SLOT_BYTES
      collect = get_imem_overflow_collect()
      if collect is not None:
        key = _resolve_overflow_atomic_key(func_name, node)
        logging.warning(
          f"[pack-bn-minmax pass1] IMCE IMEM overflow on node {node.name} "
          f"({words} words > {self.IMCE_IMEM_DEPTH_WORDS}); collecting atomic "
          f"key {key} for pass-2 exclusion (obj={obj_file})."
        )
        if key is not None and key not in collect:
          collect.append(key)
        # Continue codegen so all overflowing nodes are collected in one pass.
        return
      raise RuntimeError(
        f"IMCE IMEM overflow: node {node.name} program is {words} words "
        f"> IMEM depth {self.IMCE_IMEM_DEPTH_WORDS} words. The "
        f"{self.IMCE_IMEM_DEPTH_WORDS}-word PC wraps, overwriting the program "
        f"entry and producing X-propagation at runtime. This is typically an "
        f"over-fused node (e.g. IMCFLOW_PACK_BN_MINMAX folding BN/minmax onto a "
        f"large spatial conv): exclude this conv from packing so its BN/minmax "
        f"render on a separate IMCE. obj={obj_file}"
      )

  def update_device_config_with_obj_info(self, func_name, obj_map: dict[NodeID, str],
                                         wave_aware: bool = False):
    if wave_aware:
      # C1b (C): obj_map keyed by (NodeID, wave). Allocate a DISTINCT IMEM
      # DataBlock per (core, wave) into the owning inode's data region and record
      # it on the core's InstEdgeInfo per-wave map. All waves' blobs coexist in
      # the inode data region (additive) -- MERGE=2 region2 inode budget checked
      # (~33KB << 64KB). Policy/weights stay single (loaded once).
      for (node, wave), obj_file in obj_map.items():
        size = self.get_object_size(obj_file, key="data")
        self._check_imem_capacity(node, obj_file, size, func_name=func_name)
        if size is not None:
          db = DataBlock(f"{node.name}_imem_wave{wave}", size)
          self.allocate_db(db, f"{node.master().name}_data", "init")
          self.insert_wave_db_to_inst_edge_info(func_name, db, node, wave)
        else:
          print(f"Failed to allocate imem for {obj_file}")
      print(DevConfig().CurrFuncMemLayout)
      return

    for node, obj_file in obj_map.items():
      size = self.get_object_size(obj_file, key="data")
      self._check_imem_capacity(node, obj_file, size, func_name=func_name)
      if size is not None:
        db = DataBlock(f"{node.name}_imem", size)
        if self.target == "inode":
          self.allocate_db(db, f"{node.name}_inst", "init")
        else:
          self.allocate_db(db, f"{node.master().name}_data", "init")
          self.insert_db_to_inst_edge_info(func_name, db, node)
      else:
        print(f"Failed to allocate imem for {obj_file}")
    print(DevConfig().CurrFuncMemLayout)

  def allocate_db(self, data_block: DataBlock, region: str, phase: str):
    DevConfig().CurrFuncMemLayout[region].allocate(data_block, phase)

  def insert_db_to_inst_edge_info(self, func_name:str , db: DataBlock, node: NodeID):
    edge_info = DevConfig().get_inst_edge_info(func_name, node)
    edge_info.set_data_block(db)

  def insert_wave_db_to_inst_edge_info(self, func_name: str, db: DataBlock,
                                       node: NodeID, wave: int):
    """C1b (C): attach a per-wave IMEM DataBlock to the core's InstEdgeInfo.
    Also seed the legacy single data_block from wave 0 so any un-migrated reader
    (which expects .data_block) still sees a valid blob."""
    edge_info = DevConfig().get_inst_edge_info(func_name, node)
    edge_info.set_wave_data_block(wave, db)
    if edge_info.data_block is None or wave == 0:
      edge_info.set_data_block(db)
