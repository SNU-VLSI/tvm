import re
import os
import tvm
from tvm import relay
from tvm.relay import op
from tvm.relay.frontend.common import infer_shape
from tvm.relay.dataflow_pattern import *
from tvm.contrib.imcflow import TensorID, TensorEdge
from tvm.relay.op.contrib.imcflow import CustomIDToNode
from tvm.relay.backend.contrib.imcflow import util
from tvm.relay.backend.contrib.imcflow import transform
from tvm.relay.backend.contrib.imcflow.transform import getNodeID
from tvm.relay.backend.contrib.imcflow.transform_utils import getNodeDebugID
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from tvm.contrib.imcflow import CodegenContext
from tvm.contrib.imcflow import bugfix_off_mode
from tvm.contrib.imcflow import pack_bn_minmax_mode
from tvm.contrib.imcflow import serialize_imcu_load
from tvm.contrib.imcflow import drop_psum_send, drop_psum_keep_every
from tvm.contrib.imcflow import step_freerun_n, step_freerun_factors
from tvm.relay.backend.contrib.imcflow.kernel_codegen import KernelCodegen
from tvm.relay.backend.contrib.imcflow.device_codegen import DeviceCodegen
from tvm.relay.backend.contrib.imcflow.codeblock import *
from tvm.relay.backend.contrib.imcflow.inode_codeblock import *
from tvm.relay.backend.contrib.imcflow.imce_codeblock import *
from tvm.relay.backend.contrib.imcflow import imce_codeblock
from tvm.relay.backend.contrib.imcflow.operation_handlers import get_handler_registry
from tvm.relay.backend.contrib.imcflow.transform_utils import UseDefChainBuilder, UseDefChainParser
import pdb
import pprint

# Ensure external codegen registration side-effects are loaded.
from . import ext_codegen as _imcflow_ext_codegen  # noqa: F401
# Load operation handlers (imports trigger registration via decorators)
from . import imce_operation_handlers  # noqa: F401
from tvm.relay.backend.contrib.imcflow.imce_operation_handlers import IMCECodeBlockInfo
from tvm.relay.backend.contrib.imcflow.send_recv_sync import SendRecvPairManager

CompositePat = wildcard().has_attr({"Composite": "imcflow.qconv2d-with-postop"})(None) | \
               wildcard().has_attr({"Composite": "imcflow.qconv2d-split-concat"})(None) | \
               wildcard().has_attr({"Composite": "imcflow.qdwconv2d-with-postop"})(None) | \
               wildcard().has_attr({"Composite": "imcflow.qdwconv2d-split-concat"})(None)
TuplePat = is_tuple(None)
TupleGetItemPat = is_tuple_get_item(wildcard())
VarPat = is_var()
ConstPat = is_constant()


def _is_func_out_psum_edge(edge) -> bool:
  """True iff this tensor edge is a conv psum drain to an inode func_out collector
  (dst tensor_type starts with 'func_out'). Used to scope the IMCFLOW_DROP_PSUM
  K-keep recv-count division to exactly the edge the imce/inode K-keep affects."""
  try:
    dst_tt = getattr(edge.dst_id, "tensor_type", "")
    return isinstance(dst_tt, str) and dst_tt.startswith("func_out")
  except Exception:
    return False


@util.create_imcflow_function_pass(opt_level=0)
class CodegenSuite:
  """A pass that generates/compiles code for IMCFlow functions"""

  def __init__(self, model_dir, module, host_isa="arm", rebuild_modified_cpp=False):
    self.model_dir = model_dir
    self.build_dir = f"{model_dir}/build"
    self.host_isa = host_isa
    self.module = module
    self.rebuild_modified_cpp = rebuild_modified_cpp
    os.makedirs(self.build_dir, exist_ok=True)

    common_decl = f"""
      typedef short short16 __attribute__((ext_vector_type(16)));
      __attribute__((noinline, used)) void __builtin_IMCE_STEP(void);
    """
    with open(f"{self.build_dir}/common_decl.h", "w") as file:
      file.write(common_decl)

  def _select_cpp_file(self, func_dir, base):
    """Select .patched.cpp if enabled and exists, otherwise .cpp"""
    if DevConfig().use_patched_cpp:
      patched = f"{base}.patched.cpp"
      if os.path.exists(os.path.join(func_dir, patched)):
        return patched
    return f"{base}.cpp"

  def validate_recv_send_consistency(self, func_name, imce_builder, inode_builder, model_dir):
    """
    Validate that send and recv counts match for each edge.
    For each edge: inode_send + imce_send == inode_recv + imce_recv
    """
    output_lines = []
    output_lines.append("="*40)
    output_lines.append(f"Validating recv/send consistency for function {func_name}")

    all_edges = set()
    all_edges.update(imce_builder.send_map.keys())
    all_edges.update(imce_builder.recv_map.keys())
    all_edges.update(inode_builder.send_map.keys())
    all_edges.update(inode_builder.recv_map.keys())

    inconsistencies = []
    consistencies = []
    for edge in sorted(list(all_edges), key=lambda x: str(x)):
      imce_send  = imce_builder.send_map.get(edge, 0)
      imce_recv  = imce_builder.recv_map.get(edge, 0)
      inode_send = inode_builder.send_map.get(edge, 0)
      inode_recv = inode_builder.recv_map.get(edge, 0)

      try:
        imce_send = int(imce_send)
      except:
        imce_send = imce_send.value

      try:
        imce_recv = int(imce_recv)
      except:
        imce_recv = imce_recv.value

      try:
        inode_send = int(inode_send)
      except:
        inode_send = inode_send.value

      try:
        inode_recv = int(inode_recv)
      except:
        inode_recv = inode_recv.value

      total_send = int(imce_send) + int(inode_send)
      total_recv = int(imce_recv) + int(inode_recv)

      if total_send != total_recv:
        inconsistencies.append({
          'edge': edge,
          'imce_send': imce_send,
          'imce_recv': imce_recv,
          'inode_send': inode_send,
          'inode_recv': inode_recv,
          'total_send': total_send,
          'total_recv': total_recv
        })
      else:
        consistencies.append({
          'edge': edge,
          'imce_send': imce_send,
          'imce_recv': imce_recv,
          'inode_send': inode_send,
          'inode_recv': inode_recv,
          'total_send': total_send,
          'total_recv': total_recv
        })


    if inconsistencies:
      output_lines.append(f"\nFound {len(inconsistencies)} inconsistencies:")
      for item in inconsistencies:
        output_lines.append(f"\n  Edge: {item['edge']}")
        output_lines.append(f"    IMCE  - Send: {item['imce_send']}, Recv: {item['imce_recv']}")
        output_lines.append(f"    Inode - Send: {item['inode_send']}, Recv: {item['inode_recv']}")
        output_lines.append(f"    Total - Send: {item['total_send']}, Recv: {item['total_recv']}")
        output_lines.append(f"    Mismatch: {item['total_send']} sends vs {item['total_recv']} recvs")
      output_lines.append("\n" + "="*40)
    else:
      output_lines.append(f"✓ All edges have consistent send/recv counts")
      output_lines.append("="*40)
      for item in consistencies:
        output_lines.append(f"\n  Edge: {item['edge']}")
        output_lines.append(f"    IMCE  - Send: {item['imce_send']}, Recv: {item['imce_recv']}")
        output_lines.append(f"    Inode - Send: {item['inode_send']}, Recv: {item['inode_recv']}")
        output_lines.append(f"    Total - Send: {item['total_send']}, Recv: {item['total_recv']}")

    # Print to console
    for line in output_lines:
      print(line)

    # Write to file (append mode to collect all functions)
    output_file = os.path.join(model_dir, "recv_send_consistency.txt")
    with open(output_file, "a") as f:
      f.write("\n".join(output_lines) + "\n")

    # P2 (DESIGN §2.4): fail at COMPILE TIME on a fifo-count mismatch instead of
    # letting it surface as a 20000-poll RTL deadlock (state 0x1). ΣSEND==ΣRECV
    # per edge is a hardware invariant (fifo occupancy conservation): if a
    # producer sends more than the consumer receives, the producer wedges on a
    # full fifo forever; fewer, the consumer wedges on empty. ResNet8 is already
    # fully consistent, so this never fires there. dwconv granularity mismatches
    # (the P3/P4 work) now show up here as an assert with the exact offending
    # edges, not as an opaque hang. Opt out with IMCFLOW_SKIP_SYNC_ASSERT=1.
    # BUGFIX knob: the P2 fifo-count assert is part of the 934+P0-P3 sync work.
    # knob=on (bugfix_off_mode()==False) restores a8af, which did NOT raise here.
    # DROP_PSUM (Design A, DON'T-CARE output) INTENTIONALLY creates a func_out
    # send/recv mismatch (imce SEND + inode RECV + host read all dropped as a
    # matched set). That would "deadlock in RTL" but on chip the un-written
    # inode_3_0 port is simply never read, so it is safe. Auto-bypass the assert
    # for func_out* edges when DROP_PSUM is on (so the deploy doesn't need to also
    # remember IMCFLOW_SKIP_SYNC_ASSERT=1). Non-func_out mismatches still raise.
    if drop_psum_send():
      inconsistencies = [
        it for it in inconsistencies
        if "func_out" not in str(it.get("edge", ""))]
    if bugfix_off_mode() and inconsistencies and os.environ.get("IMCFLOW_SKIP_SYNC_ASSERT", "0") != "1":
      detail = "; ".join(
        f"{it['edge']}: send {it['total_send']} vs recv {it['total_recv']}"
        for it in inconsistencies)
      raise AssertionError(
        f"[recv/send fifo-count mismatch] function {func_name}: "
        f"{len(inconsistencies)} edge(s) would deadlock in RTL. {detail}. "
        f"(see {output_file}; set IMCFLOW_SKIP_SYNC_ASSERT=1 to bypass)")
    print(f"Recv/send consistency appended to: {output_file}")

    # P4 (DESIGN §2.4, invariant II): flag-rendezvous handshake balance. Unlike
    # (I) fifo-count (a byte-conservation invariant), (II) checks that a
    # producer's SETFLAG barrier count equals EACH consumer's window count for a
    # flag-rendezvous edge. A mismatch means one side raises/waits the wrong
    # number of times on the single per-node flag slot -> lost-wakeup / 20000-poll
    # hang (exactly the dwconv middle-stage deadlock P4 fixes). Populated only for
    # edges the codeblocks explicitly contract (dwconv output); the map is empty
    # for ResNet8, so this never fires there.
    # BUGFIX knob: the entire P4 flag-rendezvous handshake validation is part of
    # the 934+P4 sync work. knob=on (bugfix_off_mode()==False) restores a8af,
    # which had no such validation. (Under knob=on handshake_num_map is empty
    # anyway because the imce-side emission that populates it is gated, but skip
    # the whole block explicitly for a8af parity.)
    if bugfix_off_mode():
      from tvm.relay.backend.contrib.imcflow.imce_codeblock import handshake_num_map
      if handshake_num_map:
        print(f"[FLAG rendezvous] {func_name} handshake balance: {handshake_num_map}")
      flag_mismatches = []
      for uuid, hs in handshake_num_map.items():
        prod = hs.get("producer", 0)
        for cnode, cwin in hs.get("consumer", {}).items():
          if prod != cwin:
            flag_mismatches.append((uuid, cnode, prod, cwin))
      if flag_mismatches:
        print("="*40)
        print(f"[FLAG rendezvous] function {func_name}: {len(flag_mismatches)} mismatch(es)")
        for uuid, cnode, prod, cwin in flag_mismatches:
          print(f"  pair uuid={uuid}: producer_handshakes={prod} vs consumer(node {cnode}) windows={cwin}")
        with open(output_file, "a") as f:
          f.write(f"\n[FLAG rendezvous] {len(flag_mismatches)} mismatch(es): "
                  + "; ".join(f"uuid={u} prod={p} vs node{c}={w}"
                              for u, c, p, w in flag_mismatches) + "\n")
      if flag_mismatches and os.environ.get("IMCFLOW_SKIP_SYNC_ASSERT", "0") != "1":
        detail = "; ".join(
          f"uuid={u}: producer {p} vs consumer(node {c}) {w}"
          for u, c, p, w in flag_mismatches)
        raise AssertionError(
          f"[flag-rendezvous handshake mismatch] function {func_name}: "
          f"{len(flag_mismatches)} edge(s) would lost-wakeup deadlock in RTL. {detail}. "
          f"(set IMCFLOW_SKIP_SYNC_ASSERT=1 to bypass)")

      # Reset the flag-handshake map for the next function (per-function scope, like
      # the codegen context). fifo maps are handled by construct_recv_send_map.
      handshake_num_map.clear()

  def transform_function(self, _, func):
    # Note: the function name strips off the "_impl" suffix to match the original funcion name
    # which is the parent func's global_symbol attribute (prior: func.attsr.global_symbol).
    func_name = func.attrs["Composite"].strip("_impl")

    # Set the codegen context for this function
    CodegenContext().set_func_name(func_name)

    # Handle rebuild_modified_cpp case
    if self.rebuild_modified_cpp:
      print(f"\n--- Skipping codegen for function {func_name} (rebuild_modified_cpp=True) ---")

      # Load DevConfig state if it hasn't been loaded yet
      # We only need to load once since DevConfig is a singleton
      if not hasattr(DevConfig(), '_state_loaded_for_rebuild'):
        devconfig_state_path = os.path.join(self.model_dir, "devconfig_state.pkl")
        if os.path.exists(devconfig_state_path):
          print(f"Loading DevConfig state from: {devconfig_state_path}")
          DevConfig().load_state(devconfig_state_path)
          DevConfig()._state_loaded_for_rebuild = True
        else:
          raise FileNotFoundError(
            f"DevConfig state file not found: {devconfig_state_path}\n"
            f"Please run full compilation (rebuild_modified_cpp=False) first to generate the state file."
          )

      for base in ["imce", "inode"]:
        device_codegen = DeviceCodegen(target=base, build_dir=".", host_isa=self.host_isa)
        device_codegen.func_dir = os.path.join(self.build_dir, func_name)
        file = self._select_cpp_file(device_codegen.func_dir, base)
        obj_map = device_codegen.compile_target_code(file)
        device_codegen.update_device_config_with_obj_info(func_name, obj_map)

      CodegenContext().clear()

      return func

    # annotate edges between (non-composite) calls,
    # while translating vars into corresponding calls
    annotator = InternalEdgeAnnotator(func_name)
    annotator.visit(func)

    print(f"Annotated edges for function {func_name}:")
    sorted_edges = sorted(list(annotator.edges), key=lambda x: str(x))
    for edge in sorted_edges:
      print(f"  {edge}")

    # Create send-recv pair manager for synchronization.
    # BUGFIX knob: knob=off (bugfix_off_mode) keeps ALL inter-node pairs
    # (filter_contention=False) so the 934 pipeline sync set is complete; knob=on
    # restores a8af's contention-only filtering (filter_contention=True default).
    _filter_contention = bugfix_off_mode() is False  # False => keep all pairs (934); True => a8af
    pair_manager = SendRecvPairManager(sorted_edges, exclude_const=True, filter_contention=_filter_contention)
    print(f"SendRecvPairManager created: {len(pair_manager.pairs)} send-recv pairs")

    # get use def chain
    use_def_chains = {}
    for gv, func_ in self.module.functions.items():
      if util.is_imcflow_func(func_):
        target_func = DevConfig().ImcflowFuncMap[gv.name_hint]
        parser = UseDefChainParser()
        parser.visit(target_func.func_node)
        use_def_chains[gv.name_hint] = parser

    # clear IMCECodeBlockInfo before codegen
    IMCECodeBlockInfo().clear()

    # generate code blocks for each node
    imce_builder = ImceCodeBlockBuilder(self.module, func_name, sorted_edges, use_def_chains)
    imce_builder.pair_manager = pair_manager  # Add pair manager for sync support
    imce_builder.visit(func)

    # add stop block for active imces
    for hid in DevConfig().ActiveIMCEPerFunc[func_name]:
      block = CtrlBlock("STOP")
      imce_builder.codeblocks.append(hid, block, CodePhase.END)
    
    # dump block structures
    imce_builder.dump_block_structure(self.model_dir, func_name)
    
    DeviceCodegen("imce", self.build_dir, self.host_isa).handle_code_generation(
        func_name, imce_builder.codeblocks)

    # dump recv/send map and check consistency
    imce_builder.construct_recv_send_map()
    print("-"*40)
    print(f"Function {func_name} IMCE Recv Map:")
    for edge, recv_info in imce_builder.recv_map.items():
      print(f"  {edge} : {recv_info}")
    print(f"Function {func_name} IMCE Send Map:")
    for edge, send_info in imce_builder.send_map.items():
      print(f"  {edge} : {send_info}")
    print("-"*40)

    inode_builder = InodeCodeBlockBuilder(self.module, func, func_name, sorted_edges)
    inode_builder.pair_manager = pair_manager  # Add pair manager for sync support
    inode_builder.initialize()
    inode_builder.visit(func)
    inode_builder.finalize()

    # add sync logic after INIT Phase
    inode_builder.sync_inrt_clear(CodePhase.INIT)

    # dump block structures
    inode_builder.dump_block_structure(self.model_dir, func_name)

    DeviceCodegen("inode", self.build_dir, self.host_isa).handle_code_generation(
        func_name, inode_builder.codeblocks)

    PolicyTableCodegen(func_name, self.build_dir, self.host_isa).generate(func_name)

    CntBaseAddrCodegen(func_name, self.build_dir, self.host_isa).generate(func_name)

    # dump recv/send map and check consistency
    inode_builder.construct_recv_send_map()
    print("-"*40)
    print(f"Function {func_name} Inode Recv Map:")
    for edge, recv_count in inode_builder.recv_map.items():
      print(f"  {edge} : {recv_count}")
    print(f"Function {func_name} Inode Send Map:")
    for edge, send_count in inode_builder.send_map.items():
      print(f"  {edge} : {send_count}")
    print("-"*40)

    # Validate recv/send consistency
    self.validate_recv_send_consistency(func_name, imce_builder, inode_builder, self.model_dir)

    # Clear the codegen context when done
    CodegenContext().clear()

    with open(f"{self.model_dir}/final_imcflow_config_memory_map_with_inst.txt", "w") as f:
      print(f"----------------------- memory_map_with_inst ------------------------", file=f)
      for key, value in DevConfig().MemLayout.items():
        pprint.pprint(f"{key} : {value}", stream=f)

    return func


class PolicyTableCodegen:
  """
  Write out a binary file for policy tables for each node.
  """

  def __init__(self, func_name, build_dir="/tmp", host_isa="arm"):
    super().__init__()
    self.build_dir = build_dir
    self.host_isa = host_isa
    self.func_dir = os.path.join(build_dir, func_name)

  def pack_to_bin(self, entry, endian):
    assert set(entry.keys()) == {
        'Local', 'North', 'East', 'South', 'West'}, "Invalid policy table entry"

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

  def generate(self, func_name):
    for node_name, entries in sorted(transform.ImcflowDeviceConfig().PolicyTableDict[func_name].items(), key=lambda x: x[0].name):
      policytable_path = os.path.join(
          self.func_dir, f"{node_name.name}_policy")
      policytable_bin_file = f"{policytable_path}.bin"
      policytable_host_obj_file = f"{node_name.name}_policy.host.o"
      with open(policytable_bin_file, "wb") as file:
        for entry in entries:
          policytable_bin = self.pack_to_bin(entry, endian='little')
          file.write(policytable_bin)
      if ("inode" in node_name.name):
        DevCodegen = DeviceCodegen("inode", self.build_dir, self.host_isa)
        DevCodegen.func_dir = self.func_dir
        DevCodegen.create_host_object(
            f"{node_name.name}_policy.bin", policytable_host_obj_file)
      if ("imce" in node_name.name):
        DevCodegen = DeviceCodegen("inode", self.build_dir, self.host_isa)
        DevCodegen.func_dir = self.func_dir
        DevCodegen.create_host_object(
            f"{node_name.name}_policy.bin", policytable_host_obj_file)
    return


class CntBaseAddrCodegen:
  """
  Write out a binary file for counter base address blocks.
  These are 32-byte blocks used for tiling send/recv operations.
  The binary contains pkt_cnts from BlockTilingInfo, packed as 32-bit integers.
  """

  def __init__(self, func_name, build_dir="/tmp", host_isa="arm"):
    super().__init__()
    self.build_dir = build_dir
    self.host_isa = host_isa
    self.func_dir = os.path.join(build_dir, func_name)

  def write_readable_file(self, block_id, edge_simple_name, pkt_cnts, bin_file):
    """Write a human-readable text file corresponding to the binary file."""
    txt_file = f"{block_id}.txt"
    txt_path = os.path.join(self.func_dir, txt_file)

    with open(txt_path, "w") as file:
      file.write(f"Counter Base Address Block: {block_id}\n")
      file.write(f"Edge: {edge_simple_name}\n")
      file.write(f"Binary file: {bin_file}\n")
      file.write("=" * 60 + "\n\n")

      file.write(f"Packet Counts ({len(pkt_cnts)} entries):\n")
      file.write("-" * 60 + "\n")
      for i, cnt in enumerate(pkt_cnts):
        # Show decimal, hex, and binary offset
        cnt_val = int(cnt)
        file.write(f"  [{i:2d}] offset 0x{i*4:02x}: {cnt_val:10d} (0x{cnt_val:08x})\n")

      # Show padding info
      written_bytes = len(pkt_cnts) * 4
      if written_bytes < 32:
        padding_bytes = 32 - written_bytes
        file.write(f"\nPadding: {padding_bytes} bytes (0x00) to reach 32-byte alignment\n")

      file.write("\n" + "=" * 60 + "\n")
      file.write(f"Total binary size: {max(written_bytes, 32)} bytes\n")

  def generate(self, func_name):
    # Get all data blocks from the memory layout for this function
    mem_layout = DevConfig().MemLayout[func_name]

    # Find all cnt_base_addr blocks and their corresponding data blocks
    for block_id, block in mem_layout.blocks.items():
      if not isinstance(block_id, str) or not block_id.endswith("_cnt_base_addr"):
        continue

      # Extract edge simple name from block_id (remove "_cnt_base_addr" suffix)
      edge_simple_name = block_id[:-len("_cnt_base_addr")]

      # Find the corresponding data block with tiling_info
      data_block = None
      for db_id, db in mem_layout.blocks.items():
        if hasattr(db, 'tiling_info') and db.tiling_info is not None:
          # Check if this data block's edge matches our cnt_base_addr block
          for edge in db.edges:
            if edge.simple_name() == edge_simple_name:
              data_block = db
              break
          if data_block:
            break
      if data_block is None:
        raise RuntimeError(f"Could not find data block for cnt_base_addr block {block_id}")

      # Create binary file
      bin_file = f"{block_id}.bin"
      bin_path = os.path.join(self.func_dir, bin_file)
      host_obj_file = f"{block_id}.host.o"

      # Pack pkt_cnts as 32-bit little-endian integers
      pkt_cnts = None
      with open(bin_path, "wb") as file:
        if data_block and data_block.tiling_info and data_block.tiling_info.pkt_cnts:
          pkt_cnts = data_block.tiling_info.pkt_cnts
          for cnt in pkt_cnts:
            if cnt < 0:
              print(f"Warning: Negative packet count {cnt} in block {block_id}")
              raise RuntimeError(f"Negative packet count {cnt} in block {block_id}")
            file.write(int(cnt).to_bytes(4, byteorder='little', signed=False))
          # Pad to 32 bytes if needed
          written_bytes = len(pkt_cnts) * 4
          if written_bytes < 32:
            file.write(b'\x00' * (32 - written_bytes))
        else:
          raise RuntimeError(f"No tiling info found for data block corresponding to {block_id}")

      # Write human-readable text file
      self.write_readable_file(block_id, edge_simple_name, pkt_cnts, bin_file)

      # Create host object file using DeviceCodegen
      DevCodegen = DeviceCodegen("inode", self.build_dir, self.host_isa)
      DevCodegen.func_dir = self.func_dir
      DevCodegen.create_host_object(bin_file, host_obj_file)

    return


class InternalEdgeAnnotator(tvm.relay.ExprVisitor):
  def __init__(self, func_name):
    super().__init__()
    self.stack = [] # track composite call stacks
    # self.edges = set(TensorEdge._instances.values())
    self.edges = set(DevConfig().TensorEdgeListDict[func_name])
  
  @property
  def composite_call(self):
    return self.stack[-1] if self.stack else None

  def add_edge(self, dst_tid, arg, split_idx=None):
    # skip relay Var. arg can be var from tuple path
    if isinstance(arg, tvm.relay.Var):
      return

    # handle tuple using recursion
    if TuplePat.match(arg):
      if split_idx is not None:
        self.add_edge(dst_tid, arg.fields[split_idx])
      else:
        for a in arg.fields:
          self.add_edge(dst_tid, a)
      return
    elif TupleGetItemPat.match(arg):
      self.add_edge(dst_tid, arg.tuple_value, split_idx=arg.index)
      return

    src_tid = self.get_tensor_id(arg, "odata")
    self.edges.add(TensorEdge(src_tid, dst_tid, split_idx))  # add edge to set

  def visit_call(self, call):
    if CompositePat.match(call):
      self.visit_composite_call(call)
    else:
      self.visit_regular_call(call)

  def visit_composite_call(self, call):
    self.stack.append(call)
    self.visit(call.op.body)
    self.stack.pop()
    for a in call.args:
      self.visit(a)

  def visit_regular_call(self, call):
    self.visit(call.op)

    # add edges for internal edges
    if self.composite_call:
      for idx, a in enumerate(call.args):
        if VarPat.match(a) or ConstPat.match(a) or not hasattr(call.op, "arguments"):
          continue
        dst_tag = call.op.arguments[idx].name
        dst_tid = self.get_tensor_id(call, dst_tag)
        self.add_edge(dst_tid, a)

    for a in call.args:
      self.visit(a)

  def get_tensor_id(self, call, tag):
    if self.composite_call:
      return TensorID((getNodeID(self.composite_call), getNodeID(call)), tag)
    else:
      return TensorID(getNodeID(call), tag)


def _collect_block_lines(block, output_lines, indent=0):
  """Helper function to recursively collect block information for dumping.

  This is shared between ImceCodeBlockBuilder and InodeCodeBlockBuilder.
  """
  # Skip if block is a lambda function (used in SimpleFor body)
  if callable(block) and not isinstance(block, CodeBlock):
    prefix = " " * indent
    output_lines.append(f"{prefix}- Lambda function")
    return

  prefix = " " * indent
  block_type = type(block).__name__
  try:
    graph_node_id = block.get_graph_node_id()
  except:
    graph_node_id = "N/A"
  annotation = getattr(block, "annotation", "")
  annotation += f" (graph_node_id: {graph_node_id})"
  annotation += f", {block.dump()}"
  output_lines.append(f"{prefix}- {block_type} : {annotation}")

  # Handle nested blocks
  if isinstance(block, SequentialBlock):
    for child in block.blocks:
      _collect_block_lines(child, output_lines, indent + 2)
  elif isinstance(block, SimpleFor):
    _collect_block_lines(block.body, output_lines, indent + 2)
  elif isinstance(block, LoadLBBlock):
    _collect_block_lines(block.body, output_lines, indent + 2)
  elif isinstance(block, RecvSendWrapper):
    _collect_block_lines(block.body, output_lines, indent + 2)
  elif isinstance(block, ConvBlock):
    _collect_block_lines(block.body, output_lines, indent + 2)
  elif isinstance(block, InodeCodeBlock):
    # InodeCodeBlock has a body attribute that is a SequentialBlock
    if hasattr(block, 'body') and isinstance(block.body, SequentialBlock):
      _collect_block_lines(block.body, output_lines, indent + 2)


class ImceCodeBlockBuilder(tvm.relay.ExprVisitor):
  """Visitor that generates IMCE code blocks from relay operations.

  This class uses a pluggable handler registry to process different operation types.
  New operations can be supported by creating handler classes and registering them
  with the @register_operation_handler decorator in imce_operation_handlers.py.

  Handlers receive a BuilderContext that wraps each call with helper methods.
  """

  def __init__(self, module, func_name, edges, use_def_chains):
    super().__init__()
    # Shared state accessed by handlers through BuilderContext
    self.module = module
    self.func_name = func_name
    self.use_def_chains = use_def_chains
    self.edges = edges
    self.codeblocks = ImceCodeBlockManager(func_name)
    self.curr_composite_id = None
    self.curr_conv_block = None
    self.last_tuple_idx = None
    self.post_op_stack = None
    self.conv_pending_info = None
    self.vec_op_stack = None
    self._handler_registry = get_handler_registry()
    self.send_map = {}
    self.recv_map = {}

  def visit_tuple(self, tup):
    consumers = self.use_def_chains[self.func_name].get_users(tup)
    for idx, x in enumerate(tup.fields):
      if len(consumers) == 1 and isinstance(consumers[0], tvm.relay.Call) and consumers[0].op.name == "concatenate":
        self.last_tuple_idx = idx
      self.visit(x)

  def visit_call(self, call):
    # Visit arguments first (post-order traversal)
    for idx, a in enumerate(call.args):
      self.visit(a)

    # Dispatch to handler registry (automatically wraps call in BuilderContext)
    print("[IMCE CODE BUILDER] Visit call:", getNodeID(call), getNodeDebugID(call))
    handled = self._handler_registry.handle(call, self)

    # Fallback for unhandled operations
    if not handled:
      print("[IMCE CODE BUILDER] FAIL Visit call:", getNodeID(call), getNodeDebugID(call))
      self.visit(call.op)
  
  def visit_var(self, var):
    super().visit_var(var)
    print("[IMCE CODE BUILDER] Visited var:", getNodeID(var), getNodeDebugID(var))
  
  def visit_tuple_getitem(self, t):
    super().visit_tuple_getitem(t)
    print("[IMCE CODE BUILDER] Visited tuple_getitem:", getNodeID(t), getNodeDebugID(t))
  
  def visit_function(self, fn):
    imce_codeblock.send_num_map = {}
    imce_codeblock.recv_num_map = {}
    super().visit_function(fn)
    print("[IMCE CODE BUILDER] Visited function")

  def dump_block_structure(self, dir_name, func_name):
    output_lines = []
    output_lines.append("="*40)
    output_lines.append(f"[IMCE CODE BUILDER] Dumping block structure for {func_name}")

    # Sort hids for deterministic output
    sorted_hids = sorted(self.codeblocks.blocks.keys(), key=lambda x: str(x))

    for hid in sorted_hids:
      phases = self.codeblocks.blocks[hid]
      output_lines.append(f"Node: {hid}")
      for phase in sorted(phases.keys(), key=lambda x: ["INIT", "EXEC", "END"].index(x.name)):
        blocks = phases[phase]
        if not blocks: continue
        output_lines.append(f"  Phase: {phase}")
        for block in blocks:
          _collect_block_lines(block, output_lines, indent=4)
    output_lines.append("="*40)

    # Print to console
    for line in output_lines:
      print(line)

    # Write to file (append mode to collect all functions)
    output_file = os.path.join(dir_name, "block_structure.txt")
    with open(output_file, "a") as f:
      f.write("\n".join(output_lines) + "\n")
    print(f"Block structure appended to: {output_file}")
  
  def construct_recv_send_map(self):
    """
    Iterate code blocks and find recv_send_wrapper and RecvConstBlock block.
    recv_send_wrapper block has local recv, send map. Aggregate it.
    """
    print("-"*40)
    print("[IMCE CODE BUILDER] Constructing recv/send map")
    self.recv_map.update(imce_codeblock.recv_num_map)
    self.send_map.update(imce_codeblock.send_num_map)

class InodeCodeBlockBuilder(tvm.relay.ExprVisitor):
  def __init__(self, mod, func, func_name, edges):
    super().__init__()
    self.mod = mod
    self.func = func
    self.func_name = func_name
    self.is_tiled = (DevConfig().ImcflowFuncMap[func_name].tiling_factor > 1)
    self.edges = edges
    self.codeblocks = InodeCodeBlockManager(func_name)
    # Track which hardware nodes already have an IMCE compute block added
    self._imce_compute_added = set()
    self.send_map = {}
    self.recv_map = {}
    self.curr_composite_id = None
  
  def sync_inrt_clear(self, codephase: CodePhase):
    inode_master = NodeID.inode_3_0
    inode_slaves = [node for node in NodeID.inodes() if node != inode_master]

    # sync all inodes
    # Under packing, use a sense-reversing barrier (alternating 254/255, no
    # clear) so inode arrival skew can't cause a lost-wakeup (see SyncAllINodes).
    # One sense per logical barrier -> all 4 per-inode instances share it.
    _sense = SyncAllINodes.next_sense() if pack_bn_minmax_mode() else None
    for inode in NodeID.inodes():
      block = SyncAllINodes(inode, "sync all inodes", sense=_sense)
      self.codeblocks.append(inode, block, codephase)
    
    # halt for slave inodes
    for inode_slv in inode_slaves:
      block = HaltBlock("halt for slave inodes")
      self.codeblocks.append(inode_slv, block, codephase)
    
    # done and interrupt for master inode. and halt
    block = DoneAndIntrtBlock("done and intrt for master inode")
    self.codeblocks.append(inode_master, block, codephase)
    block = HaltBlock("halt for master inode after done and intrt")
    self.codeblocks.append(inode_master, block, codephase)

  def initialize(self):
    # Reset the sense-reversing barrier counter at the start of each region
    # function so every region's inode programs use a consistent 254/255
    # alternation from the same starting parity. No-op when packing is off.
    SyncAllINodes.reset_sense()

    # clear flag
    for inode in NodeID.inodes():
      block = ClearFlag("clear flag before policy update")
      self.codeblocks.append(inode, block, CodePhase.INIT)

    # policy update
    for inode in NodeID.inodes():
      block = PolicyUpdateBlock(inode, "policy update")
      self.codeblocks.append(inode, block, CodePhase.INIT)

    # sync and intrt and clear
    self.sync_inrt_clear(CodePhase.INIT)

    # imem write
    func_name = self.codeblocks.func_name
    for imce, inst_edge in sorted(DevConfig().InstEdgeInfoDict[func_name].items(), key=lambda x: x[0].name):
      block = WriteIMEMBlock(inst_edge, f"imem write: {imce.name}")
      self.codeblocks.append(imce.master(), block, CodePhase.INIT)

    # imcu write
    # Silicon-SAFE serialization lever (IMCFLOW_SERIALIZE_IMCU, default OFF):
    # ResNet8 region3 wedges the real B2 chip while the BUGFIX-off RTL passes.
    # Chip-ladder root cause (all warmup-ON, IMCFLOW_BUGFIX=off, iter_003):
    #   - subset21: only inode_2_0 streams WR_IMCU (2 bursts, no concurrency) -> PASS
    #   - region2 (passes): all 4 inodes stream WR_IMCU, but each exactly 1 burst -> PASS
    #   - region3 (wedges): all 4 inodes concurrent AND inode_3_0 streams 2
    #     consecutive bursts (512 words) -> WEDGE at region3 kernel entry.
    # So the trigger is a DOUBLE (consecutive) WR_IMCU burst in one inode running
    # concurrently with the other inodes' bursts (neither concurrency alone nor a
    # double-burst alone wedges). When ON, serialize the IMCU-write phase with a
    # 255 barrier (SyncAllINodes) BEFORE each inode's IMCU write in a fixed inode
    # order so only one inode streams weight bursts at a time.
    #
    # GUARD: apply ONLY to functions where some inode has >=2 weight blocks
    # (i.e. region3). Functions with <=1 weight block per inode (region1/2/4) are
    # left byte-identical -- inserting the barrier there previously WEDGED region2
    # on silicon (its clean 4x single-burst pattern does not need serialization).
    def _inode_has_double_imcu_burst():
      try:
        for node in NodeID.inodes():
          region = DevConfig().CurrFuncMemLayout[f"{node.name}_data"]
          n_wt = sum(
              1 for db in region.blocks.values()
              if isinstance(db.id, TensorEdge) and "weight" == db.id.src_id.tensor_type)
          if n_wt >= 2:
            return True
      except Exception:
        return False
      return False

    if serialize_imcu_load() and _inode_has_double_imcu_burst():
      for node in NodeID.inodes():
        # Barrier so the previous inode's IMCU burst fully drains before this
        # inode starts streaming (all inodes rendezvous, then only `node`
        # proceeds to WR_IMCU while the others wait at the NEXT gate / the
        # step-6 "sync before compute enable" barrier).
        _sense = SyncAllINodes.next_sense() if pack_bn_minmax_mode() else None
        for inode in NodeID.inodes():
          bar = SyncAllINodes(inode, f"serialize imcu: gate before {node.name}", sense=_sense)
          self.codeblocks.append(inode, bar, CodePhase.INIT)
        block = WriteIMCUBlock(node, "imcu write")
        self.codeblocks.append(node, block, CodePhase.INIT)
    else:
      for node in NodeID.inodes():
        block = WriteIMCUBlock(node, "imcu write")
        self.codeblocks.append(node, block, CodePhase.INIT)

    # # sync before imce compute
    _sense = SyncAllINodes.next_sense() if pack_bn_minmax_mode() else None
    for inode in NodeID.inodes():
      block = SyncAllINodes(inode, "sync before compute enable", sense=_sense)
      self.codeblocks.append(inode, block, CodePhase.INIT)

    # imce compute
    active_imces = DevConfig().ActiveIMCEPerFunc[func_name]
    for imce, inst_edge in sorted(DevConfig().InstEdgeInfoDict[func_name].items(), key=lambda x: x[0].name):
      if imce in active_imces:
        policy_addr = inst_edge.policy_info[0].address # get first policy address
        block = IMCEComputeBlock(policy_addr, f"{imce.name} compute")
        self.codeblocks.append(imce.master(), block, CodePhase.INIT)
    
    # wait all enable of imce
    _sense = SyncAllINodes.next_sense() if pack_bn_minmax_mode() else None
    for inode in NodeID.inodes():
      block = SyncAllINodes(inode, "wait all imce compute enable", sense=_sense)
      self.codeblocks.append(inode, block, CodePhase.INIT)
      # block = SetFlag()
      # self.codeblocks.append(inode, block, CodePhase.INIT)
      # other_nodes = [n for n in NodeID.inodes() if n != inode]
      # block = Standby(node_ids=other_nodes, annotation=f"standby for {inode.name}")
      # self.codeblocks.append(inode, block, CodePhase.INIT)

      # block = ClearFlag("clear flag after imce compute enable")
      # self.codeblocks.append(inode, block, CodePhase.INIT)
    
    # send constant goes is taken care of by graph traverse


  def finalize(self):
    self.sync_inrt_clear(CodePhase.EXEC_TILE if self.is_tiled else CodePhase.END)
    # # standby and intrt
    # # FIXME: hardcoded inode_3_0
    # inode_master = NodeID.inode_3_0
    # inode_slaves = [node for node in NodeID.inodes() if node != inode_master]
    # block = StandbyAndIntrtBlock(inode_slaves, "standby and intrt")
    # self.codeblocks.append(inode_master, block, CodePhase.EXEC_TILE if self.is_tiled else CodePhase.END)

    # # set_flag
    # block = SetFlagAndHaltBlock()
    # for inode_slv in inode_slaves:
    #   self.codeblocks.append(inode_slv, block, CodePhase.EXEC_TILE if self.is_tiled else CodePhase.END)

  def dump_block_structure(self, dir_name, func_name):
    output_lines = []
    output_lines.append("="*40)
    output_lines.append(f"[INODE CODE BUILDER] Dumping block structure for {func_name}")

    # Sort hids for deterministic output
    sorted_hids = sorted(self.codeblocks.blocks.keys(), key=lambda x: str(x))

    for hid in sorted_hids:
      phases = self.codeblocks.blocks[hid]
      output_lines.append(f"Node: {hid}")
      for phase in sorted(phases.keys(), key=lambda x: ["INIT", "EXEC", "EXEC_TILE", "END"].index(x.name)):
        blocks = phases[phase]
        if not blocks: continue
        output_lines.append(f"  Phase: {phase}")
        for block in blocks:
          _collect_block_lines(block, output_lines, indent=4)
    output_lines.append("="*40)

    # Print to console
    for line in output_lines:
      print(line)

    # Write to file (append mode to collect all functions)
    output_file = os.path.join(dir_name, "block_structure.txt")
    with open(output_file, "a") as f:
      f.write("\n".join(output_lines) + "\n")
    print(f"Block structure appended to: {output_file}")

  def visit_function(self, fn):
    # constant tensor tags except "weight" (IMCU weights are handled separately)
    param_edges = []
    const_edges = []
    output_edges = []
    for x in fn.params:
      # self.visit(x)
      param_id = getNodeID(x)
      # The input variable will go to the same router entry => only need one send block
      param_edge = self.get_output_edges_from_id(param_id)[0]
      param_edges.append(param_edge)
      # self.add_send_block(param_edge)
    #self.visit(fn.body)
    # traverse constant nodes

    def is_dwconv_weight(edge: TensorEdge) -> bool:
      """Check if this weight edge goes to a qdwconv node.

      DW conv weights need to be sent via FIFO (unlike standard conv weights
      which are loaded into IMCU). This function identifies weight edges that
      should be included in const_edges for INODE SEND block generation.
      """
      if edge.src_id.tensor_type != "weight":
        return False
      dst_node_id = getInnerNodeID(edge.dst_id.graph_node_id)
      dst_node = CustomIDToNode()[dst_node_id]
      if isinstance(dst_node, relay.Call) and isinstance(dst_node.op, tvm.ir.Op):
        return dst_node.op == op.get("nn.imcflow_qdwconv")
      return False

    for edge in self.edges:
      arg_id = edge.src_id.graph_node_id
      if ConstPat.match(CustomIDToNode()[arg_id]):
        # Include DW conv weight (not standard conv weight which goes to IMCU)
        if edge.src_id.tensor_type != "weight" or is_dwconv_weight(edge):
          const_edges.append(edge)
          # self.add_send_block(edge)

    # Add Recv Block
    fn_id = getNodeID(fn)
    fn_edges = self.get_input_edges_from_id(fn_id)
    for last_edge in fn_edges:
      output_edges.append(last_edge)
    
    # send const edge interleaved
    #TODO: consider recv node order..

    # add send block for const edges based on imce const edge ordering
    imce_edges = IMCECodeBlockInfo()
    for hid in sorted(imce_edges.imce_const_edges.keys(), key=lambda x: x.name):
      const_edge_list = imce_edges.imce_const_edges[hid]
      for edge in const_edge_list:
        if edge in const_edges:
          self.add_send_block(edge, CodePhase.INIT)
        else:
          assert False, f"const edge {edge} in IMCECodeBlockInfo not found in function const edges"

    # send param edge interleaved
    # Group param edges by source HID (inode)
    param_edges_by_hid = {}
    for edge in param_edges:
      hid = self.get_hid(edge.src_id)
      if hid not in param_edges_by_hid:
        param_edges_by_hid[hid] = []
      param_edges_by_hid[hid].append(edge)

    # Generate send blocks for each HID group
    for hid, edges in param_edges_by_hid.items():
      if len(edges) == 1:
        self.add_send_block(edges[0], CodePhase.EXEC_TILE if self.is_tiled else CodePhase.EXEC)
      else:
        self.add_send_block_interleaved(edges, CodePhase.EXEC_TILE if self.is_tiled else CodePhase.EXEC)

    # recv output edge interleaved
    # Group output edges by destination HID (inode)
    output_edges_by_hid = {}
    for edge in output_edges:
      # NOTE: updated to handle tuple hwnodeid
      hid = self.get_hid(edge.dst_id, edge.split_idx)
      if hid not in output_edges_by_hid:
        output_edges_by_hid[hid] = []
      output_edges_by_hid[hid].append(edge)

    for hid, edges in output_edges_by_hid.items():
      if len(edges) == 1:
        self.add_recv_block(edges[0], CodePhase.EXEC_TILE if self.is_tiled else CodePhase.EXEC)
      else:
        self.add_recv_block_interleaved(edges, CodePhase.EXEC_TILE if self.is_tiled else CodePhase.EXEC)

  def construct_recv_send_map(self):
    """
    Iterate code blocks and find SendBlock, SendBlockInterleaved, and RecvBlock.
    Aggregate send and recv counts per edge by counting actual loop iterations.
    """
    import math
    def add_to_map(map, edges, count):
      if isinstance(edges, list):
        for edge in edges:
          if edge not in map:
            map[edge] = 0
          map[edge] += count
      else:
        edge = edges
        if edge not in map:
          map[edge] = 0
        map[edge] += count
    
    def _add_send_map(send_map, block, edge):
      if isinstance(edge, list):
        for e in edge:
          _add_send_map(send_map, block, e)
      else:
        dst_node_graph_id = edge.dst_id.graph_node_id
        dst_node = CustomIDToNode()[getInnerNodeID(dst_node_graph_id)]
        if dst_node.op.name == "split":
          split_out_tensor_id = TensorID(dst_node_graph_id, "odata")
          output_edges = [edge for edge in DevConfig().TensorEdgetoInfo.keys() if getInnerNodeID(edge.src_id.graph_node_id) == getInnerNodeID(split_out_tensor_id.graph_node_id)]
          for output_edge in output_edges:
            if block.tiling_info:
              send_count = sum(block.tiling_info.pkt_cnts)
            else:
              send_count = math.ceil(block.size / 32)
            add_to_map(send_map, output_edge, send_count)
        else:
          if block.tiling_info:
            send_count = sum(block.tiling_info.pkt_cnts)
          else:
            send_count = math.ceil(block.size / 32)
          # STEP_FREERUN: the inode activation feed is wrapped in a runtime (1+N)x
          # hardware loop (inode_codeblock.py SendBlock) to match the imce ConvBlock's
          # (1+N)x LOAD_LB, whose recorded recv IS scaled by the loop stack. The
          # inode send_count here is size-derived (loop-agnostic), so scale it by the
          # SAME (1+N) for the activation feed edge so recv/send stays balanced (both
          # base*(1+N)). Only the qconv/qdwconv "data" feed; never weight/const.
          _fr = step_freerun_n()
          if (_fr > 0
              and getattr(edge.dst_id, "tensor_type", None) == "data"
              and dst_node.op.name in ("nn.imcflow_qconv", "nn.imcflow_qdwconv")):
            # Use the ACTUAL nested-loop factor product (may slightly exceed 1+N),
            # matching the imce recv scaling (count_stack = product of the same
            # factors) so recv/send stays exactly balanced.
            _prod = 1
            for _f in step_freerun_factors(1 + _fr):
              _prod *= _f
            send_count *= _prod
          add_to_map(send_map, edge, send_count)


    all_blocks = self.codeblocks.get_blocks()
    for block in all_blocks:
      if isinstance(block, SendBlock):
        edge = block.block.id
        _add_send_map(self.send_map, block.block, edge)
      elif isinstance(block, SendBlockInterleaved):
        # For interleaved sends, each block sends in parallel in the same loop
        for db in block.blocks:
          edge = db.id
          _add_send_map(self.send_map, db, edge)
      elif isinstance(block, RecvBlockInterleaved):
        for db in block.blocks:
          edge = db.id
          # recv_count = math.ceil(db.size / 32)
          if db.tiling_info:
            recv_count = sum(db.tiling_info.pkt_cnts)
          else:
            recv_count = math.ceil(db.size / 32)
          add_to_map(self.recv_map, edge, recv_count)

      elif isinstance(block, RecvBlock):
        edge = block.block.id
        # Calculate actual number of recv operations (loop count)
        # recv_count = math.ceil(block.block.size / 32)
        if block.block.tiling_info:
          recv_count = sum(block.block.tiling_info.pkt_cnts)
        else:
          recv_count = math.ceil(block.block.size / 32)
        # Max-throughput lever (IMCFLOW_DROP_PSUM + IMCFLOW_DROP_PSUM_KEEP=K): the
        # producing imce keeps only 1 psum drain per K pixels (structural keep/skip
        # tiers in ConvBlock._build_structure), and the inode RECV loop
        # (RecvBlock._build_tiled) receives only kept_pkts == recv_count // K. Record
        # the SAME kept count here so validate_recv_send_consistency matches the
        # imce kept-SEND count (else 32 vs 256 false mismatch). K=0 -> whole RECV
        # loop omitted (drop-all) -> recv_count stays as-is on both sides (imce also
        # omits all sends -> both 0 net for the runtime, map still balances). Only
        # the tiled func_out psum drain is affected (the edge the inode dropped).
        if (drop_psum_send() and drop_psum_keep_every() > 0
            and block.block.tiling_info
            and _is_func_out_psum_edge(edge)):
          _k = drop_psum_keep_every()
          if recv_count % _k == 0:
            recv_count = recv_count // _k
          # STEP_FREERUN composes multiplicatively: the imce repeats the whole conv
          # (1+N)x, so it emits (1+N)x as many KEPT psum SENDs, and the inode RECVs
          # kept//K * (1+N). The imce func_out send in the map is already scaled by
          # the loop stack (row loop x(1+N)); scale this recorded RECV by the SAME
          # factor product so validate_recv_send_consistency matches (else N=100,K=8
          # gives 3232 send vs 32 recv). keep-tier (spatial /K) and freerun
          # (temporal x(1+N)) are orthogonal and both apply.
          _fr = step_freerun_n()
          if _fr > 0:
            _prod = 1
            for _f in step_freerun_factors(1 + _fr):
              _prod *= _f
            recv_count *= _prod
        add_to_map(self.recv_map, edge, recv_count)

  def add_send_block(self, edge, phase: CodePhase, db=None):
    out_edge_info = DevConfig().get_tensor_edge_info(edge)
    tid = edge.src_id
    hid = self.get_hid(tid)
    # split op handling => pass down to inner edges from input var node
    gid = edge.dst_id.graph_node_id
    if db is None:
      db = DevConfig().CurrFuncMemLayout.get_data_block_by_edge(edge)
      if db is None:
        pass
      assert db is not None, f"Data block not found for edge: {edge}"

    if CustomIDToNode()[gid].op.name == "split":
      inner_node = CustomIDToNode()[gid]
      split_id = getNodeID(inner_node)

      # Check is_multi_cast from SplitInfo
      # - True (regular conv): multicast, policy table handles routing, process first edge only
      # - False (DW conv): unicast, need to send to each split output individually
      split_info = DevConfig().SplitInfo.get(self.func_name, {}).get(split_id, {})
      is_multicast = split_info.get('is_multi_cast', True)

      for inner_edge in self.get_output_edges_from_id(split_id):
        self.add_send_block(inner_edge, phase, db)
        if is_multicast:
          # multicast: policy table handles routing, only need first edge
          return
      # non-multicast (DW conv): all edges processed, return
      return

    annotation = f"send - {edge}, {out_edge_info.policy_info[0].router_id.name} -> {out_edge_info.policy_info[-1].router_id.name}"
    block = SendBlock(self, db, out_edge_info, annotation)
    self.codeblocks.append(hid, block, phase)
  
  def add_send_block_interleaved(self, edge_list, phase: CodePhase):
    dbs = []
    edge_infos = []
    # FIXME: change this if multiple inodes send params
    hids = [self.get_hid(edge.src_id) for edge in edge_list]
    assert all(hid == hids[0] for hid in hids), "all edges should have same starting inode for interleaved"
    hid = hids[0]

    def append_edgeinfo_and_db(edge: TensorEdge, edge_infos: List, dbs: List):
      out_edge_info = DevConfig().get_tensor_edge_info(edge)
      edge_infos.append(out_edge_info)
      dbs.append(db)
      annotation = f"send - {edge}, {out_edge_info.policy_info[0].router_id.name} -> {out_edge_info.policy_info[-1].router_id.name}, "
      return annotation


    annotation = ""
    for edge in edge_list:
      gid = edge.dst_id.graph_node_id
      db = DevConfig().CurrFuncMemLayout.get_data_block_by_edge(edge)

      # split handling
      if CustomIDToNode()[gid].op.name == "split":
        inner_node = CustomIDToNode()[gid]
        split_id = getNodeID(inner_node)

        # Check is_multi_cast from SplitInfo
        split_info = DevConfig().SplitInfo.get(self.func_name, {}).get(split_id, {})
        is_multicast = split_info.get('is_multi_cast', True)

        for inner_edge in self.get_output_edges_from_id(split_id):
          annotation += append_edgeinfo_and_db(inner_edge, edge_infos, dbs)
          if is_multicast:
            # multicast: only need first edge
            break
        # non-multicast (DW conv): all edges processed
      else:
        annotation += append_edgeinfo_and_db(edge, edge_infos, dbs)

    block = SendBlockInterleaved(self, dbs, edge_infos, annotation)
    self.codeblocks.append(hid, block, phase)

  def add_recv_block(self, edge, phase: CodePhase):
    in_edge_info = DevConfig().get_tensor_edge_info(edge)
    in_tid = edge.dst_id
    hid = self.get_hid(in_tid, edge.split_idx)
    db = DevConfig().CurrFuncMemLayout.get_data_block_by_edge(edge)

    block = RecvBlock(self, db, in_edge_info.fifo_id, f"recv: {in_tid}")
    self.codeblocks.append(hid, block, phase)
  
  def add_recv_block_interleaved(self, edge_list, phase: CodePhase):
    dbs = []
    fifo_ids = []
    # Group by HID
    hids = [self.get_hid(edge.dst_id, edge.split_idx) for edge in edge_list]
    assert all(hid == hids[0] for hid in hids), "all edges should have same destination inode for interleaved"
    hid = hids[0]

    annotation = ""
    for edge in edge_list:
      in_edge_info = DevConfig().get_tensor_edge_info(edge)
      db = DevConfig().CurrFuncMemLayout.get_data_block_by_edge(edge)
      assert db is not None
      dbs.append(db)
      fifo_ids.append(in_edge_info.fifo_id)
      annotation += f"recv - {edge}, {in_edge_info.fifo_id}, "

    block = RecvBlockInterleaved(self, dbs, fifo_ids, annotation)
    self.codeblocks.append(hid, block, phase)

  def get_graph_node_id(self, call):
    if self.curr_composite_id:
      return (self.curr_composite_id, getNodeID(call))
    else:
      return getNodeID(call)

  def get_input_edges_from_id(self, id):
    return [edge for edge in self.edges if edge.dst_inner_gid_match(id)]

  def get_output_edges_from_id(self, id):
    return [edge for edge in self.edges if edge.src_inner_gid_match(id)]

  def get_hid(self, tensor_id, tuple_idx=None):
    gid = tensor_id.graph_node_id
    if isinstance(gid, tuple):
      gid = gid[-1]
    hid = DevConfig().get_hw_node(gid)
    if isinstance(hid, tuple):
      assert tuple_idx is not None, f"tuple index must be provided for tuple hw node id: {hid}"
      hid = hid[tuple_idx]
    return hid
