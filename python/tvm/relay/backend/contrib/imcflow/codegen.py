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
from tvm.relay.op.contrib.imcflow import residual_in_region_mode
from tvm.contrib.imcflow import serialize_imcu_load
from tvm.contrib.imcflow import drop_psum_send
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
    # C1b (C) wave-launch: each per-(core,wave) program needs its OWN terminator.
    # For a core reused across waves, its non-final wave program ends with a
    # completion SETFLAG (ImceWaveDoneBlock) THEN STOP -- the inode STANDBYs on
    # that flag before re-WR_IMEM (emit_wave_launches) so the previous wave has
    # fully run before its IMEM is overwritten (fixes the WR_IMEM-vs-in-flight-
    # fetch race). The final wave (and every single-wave core) ends with a bare
    # STOP == the stock behavior (byte-identical off merge).
    wave_map = DevConfig().NodeToWavePerFunc.get(func_name, {}) or {}
    all_waves = sorted({w for ws in wave_map.values() for w in ws})
    n_waves = (max(all_waves) + 1) if all_waves else 1
    for hid in DevConfig().ActiveIMCEPerFunc[func_name]:
      core_waves = sorted(wave_map.get(hid, {0}))
      if n_waves <= 1 or len(core_waves) <= 1:
        # single-wave core: bare STOP in its (only) wave -> stock byte-identical
        block = CtrlBlock("STOP")
        imce_builder.codeblocks.current_wave = core_waves[0] if core_waves else 0
        imce_builder.codeblocks.append(hid, block, CodePhase.END)
      else:
        # reused core: per-wave terminator. Non-final waves: SETFLAG(done)+STOP;
        # final wave: bare STOP.
        last_wave = core_waves[-1]
        for w in core_waves:
          if w != last_wave:
            imce_builder.codeblocks.current_wave = w
            imce_builder.codeblocks.append(
                hid, ImceWaveDoneBlock(f"{hid.name} wave{w} done"), CodePhase.END)
          imce_builder.codeblocks.current_wave = w
          imce_builder.codeblocks.append(hid, CtrlBlock("STOP"), CodePhase.END)
    imce_builder.codeblocks.current_wave = 0
    
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
    # C1b (C) wave-launch: after wave-0 streaming, emit waves 1+ launch segments
    # (re-WR_IMEM + COMPUTE-enable + per-wave streaming) on the EXEC path. No-op
    # for single-wave regions -> byte-identical.
    inode_builder.emit_wave_launches()
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

    # Merge-mode DMEM alias guard: under IMCFLOW_REGION_MERGE the per-(core,wave)
    # IMEM blobs (init) and the region input / cnt_base_addr (exec) share the inode
    # DMEM; a phase-overlap silently CLOBBERS a wave IMEM blob at runtime (garbage
    # IMEM -> imce X). The single-arena allocator (imcflow.MemoryRegion.allocate)
    # prevents it; this assert catches any regression at compile time. Per physical
    # DMEM region, no two DISTINCT DataBlocks may share a byte. Gated to merge so
    # the non-merged (phase-overlapping-by-luck) layout is untouched/byte-identical.
    from tvm.relay.op.contrib.imcflow import region_merge_mode as _rmm
    if _rmm() > 1:
      for _fn, _mmap in DevConfig().MemLayout.items():
        for _rname, _region in _mmap.data.items():
          spans = []  # (lo, hi, name)
          for _entry in _region.data.values():
            for _b in _entry.blocks.values():
              a = _b.base_address
              if a is None or a < 0:
                continue
              spans.append((a, a + _b.size, str(_b.id)))
          spans.sort()
          for _i in range(len(spans) - 1):
            lo0, hi0, n0 = spans[_i]
            lo1, hi1, n1 = spans[_i + 1]
            if hi0 > lo1 and n0 != n1:
              raise RuntimeError(
                  f"[merge DMEM alias] region '{_rname}' (func {_fn}): blocks "
                  f"OVERLAP -> {n0} [{lo0},{hi0}) & {n1} [{lo1},{hi1}). A staged "
                  f"block would clobber another (e.g. input/cnt over a wave IMEM "
                  f"blob). Fix the allocator arena.")

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

    # C1b (C) wave-launch: stamp the manager's current launch wave for this call
    # so blocks appended by the handler are tagged with the right wave (for
    # per-(core,wave) IMEM). Composite handlers refine this from curr_composite_id
    # once they enter the composite; here we set the standalone-call default.
    # Wave 0 for every non-merged region -> inert (byte-identical).
    self.codeblocks.current_wave = DevConfig().GraphNodeToWavePerFunc.get(
        self.func_name, {}).get(getNodeID(call), 0)

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
    # C1b (C) wave-launch realization: derive this region's wave (launch-round)
    # structure from the PnR wave map. `n_waves`==1 for every non-merged region
    # (and the un-merged region1 under MERGE=2), so all wave-branching below is a
    # no-op there -> byte-identical. `wave_cores[k]` = physical IMCE NodeIDs whose
    # program runs in wave k; a core may appear in >1 wave (it is re-WR_IMEM'd per
    # wave, weights unchanged under conv-cap=1). `core_first_wave[core]` = the
    # earliest wave a core is used (its WR_IMEM lands there; later waves reuse the
    # resident program only if the SAME node -- but co-placed nodes differ per
    # wave, so each wave re-writes that core's IMEM).
    self._init_wave_structure(func_name)
    # Track which hardware nodes already have an IMCE compute block added
    self._imce_compute_added = set()
    self.send_map = {}
    self.recv_map = {}
    self.curr_composite_id = None
    # C1b (C) Stage 4 -- streaming wave partition. In a multi-wave region we DEFER
    # every inode Send/RecvBlock into a per-wave bucket instead of appending it to
    # EXEC immediately; emit_wave_launches() then flushes each wave's streaming
    # AFTER that wave's WR_IMEM+COMPUTE segment, giving the correct EXEC order
    # [wave0 stream] -> [wave1 program+compute -> wave1 stream] -> ... A SEND is
    # never placed before its wave's cores are programmed (would corrupt); a RECV
    # placed at its consumer wave only stalls until its data arrives. Off merge
    # (n_waves==1) _defer_streams is False -> blocks append immediately, exactly
    # as before -> byte-identical.
    self._defer_streams = (self.n_waves > 1)
    self._wave_streams = {k: [] for k in range(self.n_waves)}
    # Direct edges whose producer wave != consumer wave (Stage 3 will reroute
    # these through inode DMEM via func_out/func_in). Recorded here for diagnosis;
    # Stage 4 places them at the consumer (later) wave conservatively.
    self._cross_wave_edges = []

  def _init_wave_structure(self, func_name):
    """Derive per-wave launch structure for this region from the PnR wave map.

    Sets:
      self.n_waves        : number of launch waves (1 for every non-merged region)
      self.wave_cores     : {wave_k: set(imce NodeID)} cores active in wave k
      self.core_waves     : {imce NodeID: sorted[waves]} inverse
    Only populated (>1 wave) under region_merge_mode(); otherwise n_waves==1 and
    every wave-branch downstream collapses to the stock single-launch path.

    STOP-REPORT guard: wave-launch (P1 re-invoke as the wave axis) reuses the
    tiling re-invoke machinery, so a region that is BOTH wave-split (>1 wave) AND
    spatially tiled (tiling_factor>1) would need nested wave x tile loops, which
    is explicitly OUT of scope. Assert it cannot happen (subset31: the merged
    region2 has tiling_factor==1; the tiled region1 has 1 wave).
    """
    wave_map = DevConfig().NodeToWavePerFunc.get(func_name, {}) or {}
    self.core_waves = {core: sorted(ws) for core, ws in wave_map.items()}
    all_waves = sorted({w for ws in wave_map.values() for w in ws})
    self.n_waves = (max(all_waves) + 1) if all_waves else 1
    self.wave_cores = {k: set() for k in range(self.n_waves)}
    for core, ws in wave_map.items():
      for w in ws:
        self.wave_cores[w].add(core)
    if self.n_waves > 1 and self.is_tiled:
      raise NotImplementedError(
        f"[wave-launch] region {func_name} is BOTH wave-split (n_waves="
        f"{self.n_waves}) AND spatially tiled (tiling_factor="
        f"{DevConfig().ImcflowFuncMap[func_name].tiling_factor}). Nested "
        f"wave x tile launch is out of scope for C1b (C). STOP-REPORT.")

  def sync_inrt_clear(self, codephase: CodePhase):
    inode_master = NodeID.inode_3_0
    inode_slaves = [node for node in NodeID.inodes() if node != inode_master]

    # sync all inodes
    # Under packing, use a sense-reversing barrier (alternating 254/255, no
    # clear) so inode arrival skew can't cause a lost-wakeup (see SyncAllINodes).
    # One sense per logical barrier -> all 4 per-inode instances share it.
    # Under MERGE (n_waves>1) use the skew-ROBUST two-phase barrier instead: the
    # per-wave re-WR_IMEM/COMPUTE barrier pairs break the single-phase barrier's
    # <=1-skew assumption (v9 wedge). merge-gated -> non-merged byte-identical.
    _tp = self.n_waves > 1
    _sense = SyncAllINodes.next_sense() if (pack_bn_minmax_mode() or residual_in_region_mode()) else None
    for inode in NodeID.inodes():
      block = SyncAllINodes(inode, "sync all inodes", sense=_sense, two_phase=_tp)
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
    if self.n_waves > 1:
      # C1b (C) wave-launch: with >1 wave, EVERY wave's WR_IMEM+COMPUTE (including
      # wave 0) is emitted in its own EXEC launch segment by emit_wave_launches(),
      # NOT in INIT. Reason (RTL-proven): IMCE STOP is launch-terminal, and the INIT
      # phase is terminated by its OWN DONE (sync_inrt_clear at INIT end) before any
      # EXEC streaming runs. If wave-0 COMPUTE were enabled in INIT, those imces STOP
      # at the INIT-end DONE and the wave-0 data stream (an EXEC segment = a LATER
      # host RUN) then arrives at STOPPED cores -> they never compute -> X/wedge
      # (region2: wave0 COMPUTE @ end of GO#1, wave0 stream in GO#2 -> imce_1_1 X).
      # So INIT does ONLY policy + weights (WR_IMCU, resident); each wave's
      # WR_IMEM+COMPUTE+stream stay together in one segment. No WR_IMEM here.
      pass
    else:
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
        _sense = SyncAllINodes.next_sense() if (pack_bn_minmax_mode() or residual_in_region_mode()) else None
        for inode in NodeID.inodes():
          bar = SyncAllINodes(inode, f"serialize imcu: gate before {node.name}", sense=_sense,
                              two_phase=(self.n_waves > 1))
          self.codeblocks.append(inode, bar, CodePhase.INIT)
        block = WriteIMCUBlock(node, "imcu write")
        self.codeblocks.append(node, block, CodePhase.INIT)
    else:
      for node in NodeID.inodes():
        block = WriteIMCUBlock(node, "imcu write")
        self.codeblocks.append(node, block, CodePhase.INIT)

    # # sync before imce compute
    _sense = SyncAllINodes.next_sense() if (pack_bn_minmax_mode() or residual_in_region_mode()) else None
    for inode in NodeID.inodes():
      block = SyncAllINodes(inode, "sync before compute enable", sense=_sense,
                            two_phase=(self.n_waves > 1))
      self.codeblocks.append(inode, block, CodePhase.INIT)

    # imce compute
    active_imces = DevConfig().ActiveIMCEPerFunc[func_name]
    for imce, inst_edge in sorted(DevConfig().InstEdgeInfoDict[func_name].items(), key=lambda x: x[0].name):
      if imce in active_imces:
        # C1b (C) wave-launch: with >1 wave, NO COMPUTE is enabled in INIT -- every
        # wave (incl. wave 0) enables its cores in its own EXEC launch segment so
        # COMPUTE and the wave's data stream share ONE host RUN (STOP is
        # launch-terminal; INIT ends with its own DONE). Single-wave regions enable
        # all active cores here (stock, byte-identical).
        if self.n_waves > 1:
          continue
        policy_addr = inst_edge.policy_info[0].address # get first policy address
        block = IMCEComputeBlock(policy_addr, f"{imce.name} compute")
        self.codeblocks.append(imce.master(), block, CodePhase.INIT)
    
    # wait all enable of imce
    _sense = SyncAllINodes.next_sense() if (pack_bn_minmax_mode() or residual_in_region_mode()) else None
    for inode in NodeID.inodes():
      block = SyncAllINodes(inode, "wait all imce compute enable", sense=_sense,
                            two_phase=(self.n_waves > 1))
      self.codeblocks.append(inode, block, CodePhase.INIT)
      # block = SetFlag()
      # self.codeblocks.append(inode, block, CodePhase.INIT)
      # other_nodes = [n for n in NodeID.inodes() if n != inode]
      # block = Standby(node_ids=other_nodes, annotation=f"standby for {inode.name}")
      # self.codeblocks.append(inode, block, CodePhase.INIT)

      # block = ClearFlag("clear flag after imce compute enable")
      # self.codeblocks.append(inode, block, CodePhase.INIT)
    
    # send constant goes is taken care of by graph traverse

  def emit_wave_launches(self):
    """C1b (C) wave-launch: emit waves 1..N-1 as EXEC-phase launch segments.

    Structure per wave k>=1 (inode-internal, sequenced after wave k-1's stream
    drains): all-inode barrier -> WR_IMEM(wave-k cores, the core's wave-k IMEM
    blob) -> barrier -> IMCE_COMPUTE(wave-k cores) -> barrier. Wave-0's WR_IMEM/
    COMPUTE were emitted in initialize() (INIT). Per-wave data STREAMING is
    partitioned in the next stage; here we lay down the re-program + compute
    enable so each wave's IMEM is loaded onto its (possibly reused) cores.

    No-op for single-wave regions (n_waves==1) -> byte-identical.
    """
    if self.n_waves <= 1:
      return
    func_name = self.codeblocks.func_name
    inst_dict = DevConfig().InstEdgeInfoDict[func_name]
    active_imces = DevConfig().ActiveIMCEPerFunc[func_name]
    print(f"[wave-stream] {func_name}: per-wave deferred stream counts = "
          f"{ {k: len(v) for k, v in self._wave_streams.items()} }; "
          f"inode cross-wave edges = {len(self._cross_wave_edges)}")

    def _barrier(annot):
      # emit_wave_launches only runs for merge (n_waves>1) -> always the skew-robust
      # two-phase barrier (the per-wave re-WR_IMEM/COMPUTE seams need it).
      _sense = SyncAllINodes.next_sense() if (pack_bn_minmax_mode() or residual_in_region_mode()) else None
      for inode in NodeID.inodes():
        self.codeblocks.append(inode, SyncAllINodes(inode, annot, sense=_sense,
                               two_phase=(self.n_waves > 1)), CodePhase.EXEC)

    def _flush_wave_stream(w):
      # C1b (C) Stage 4: append wave-w's deferred streaming blocks to EXEC, in
      # their original per-wave emission order (already the correct producer/
      # consumer order within a wave). These were collected by _emit_stream during
      # visit_function; flushing here places them AFTER wave-w's WR_IMEM+COMPUTE.
      for hid, phase, block in self._wave_streams.get(w, []):
        self.codeblocks.append(hid, block, phase)

    # ===== (A) per-wave HOST RE-INVOKE model (IMCE STOP = launch-terminal) =====
    # RTL proved an IMCE will NOT re-enter compute after OP_STOP within one P1
    # launch, so (B) in-launch wave sequencing wedged. (A): each wave is its own
    # host RUN (P1) invocation. The inode PC uses PC_INCR on RUN, so it RESUMES at
    # the instruction after each wave's DONE/INTRT/HALT -> the next wave's segment.
    # Thus the inode EXEC phase is a linear sequence of wave segments, each ending
    # with sync_inrt_clear (DONE/INTRT/HALT); the host re-invokes RUN once per wave
    # (ext_codegen invoke count = n_waves). A fresh RUN re-launches the wave-k
    # IMCE COMPUTE from its wave-k IMEM (STOP-terminal avoided). Policy + weights
    # loaded ONCE in the PROGRAM (INIT) launch; only WR_IMEM(wave k) re-runs here.
    #
    # EVERY wave (including wave 0) emits its WR_IMEM + COMPUTE-enable + streaming
    # in ONE EXEC launch segment, so a wave's COMPUTE and the data stream it
    # consumes share a single host RUN. This is REQUIRED because IMCE STOP is
    # launch-terminal AND the INIT phase is closed by its own DONE before any EXEC
    # runs: if wave-0 COMPUTE were enabled in INIT (the old code), those cores STOP
    # at the INIT-end DONE and wave-0's stream (a later RUN) hits stopped cores -> X.
    # INIT now does only policy + weights (WR_IMCU). Each wave segment ends with
    # sync_inrt_clear (DONE/INTRT/HALT); the host re-invokes RUN once per wave
    # (ext_codegen invoke count == n_waves). The LAST wave's terminator is emitted
    # by finalize() (END phase), matching the stock single-wave region's single END.
    for k in range(0, self.n_waves):
      wcores = self.wave_cores.get(k, set())
      if not wcores:
        _flush_wave_stream(k)
        if k != self.n_waves - 1:
          self.sync_inrt_clear(CodePhase.EXEC)
        continue
      # (re-)program IMEM for this wave's cores with their wave-k blob. Runs on the
      # wave-k RUN (PC resumes past wave k-1's HALT, or past INIT's DONE for k==0).
      for imce, inst_edge in sorted(inst_dict.items(), key=lambda x: x[0].name):
        if imce in wcores:
          self.codeblocks.append(
              imce.master(),
              WriteIMEMBlock(inst_edge, f"imem write w{k}: {imce.name}", wave=k),
              CodePhase.EXEC)
      # sync so all inodes finish (re-)WR_IMEM before enabling compute.
      _barrier(f"wave {k}: sync after re-WR_IMEM, before COMPUTE")
      # enable compute for this wave's active cores (fresh launch -> re-entrant).
      for imce, inst_edge in sorted(inst_dict.items(), key=lambda x: x[0].name):
        if imce in wcores and imce in active_imces:
          policy_addr = inst_edge.policy_info[0].address
          self.codeblocks.append(
              imce.master(),
              IMCEComputeBlock(policy_addr, f"{imce.name} compute w{k}"),
              CodePhase.EXEC)
      _barrier(f"wave {k}: sync after COMPUTE enable")
      # wave-k streaming.
      _flush_wave_stream(k)
      # wave-k terminator (DONE/INTRT/HALT) so the host RUN returns and can
      # re-invoke for wave k+1; the final wave is terminated by finalize().
      if k != self.n_waves - 1:
        self.sync_inrt_clear(CodePhase.EXEC)

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
    from tvm.relay.op.contrib.imcflow import residual_in_region_mode
    from tvm.contrib.imcflow import residual_add_outer_gids
    # Only a fanout that actually feeds an in-region residual ADD gets the
    # per-edge emission (and downstream the fanout-lead schedule + policy split
    # + bare consumers). An ordinary >1-consumer param fanout (e.g. subset18
    # region1: conv head + single-operand skip-EXPORT vecops, the add being in
    # the NEXT region) keeps the OFF-identical [0]-only single multicast send --
    # its consumers keep their windows, and a bare fanout-lead against a
    # windowed consumer wedges (see residual_add_outer_gids).
    _resid_outers = (residual_add_outer_gids(self.edges)
                     if residual_in_region_mode() else set())
    def _feeds_residual_add(es):
      for _e in es:
        _d = getattr(_e.dst_id, "graph_node_id", None)
        if (isinstance(_d, tuple) and _d[0] in _resid_outers
            and getattr(_e.dst_id, "tensor_type", None) == "data"):
          return True
      return False
    for x in fn.params:
      # self.visit(x)
      param_id = getNodeID(x)
      # The input variable normally goes to a single router entry (or a same-entry
      # multicast the router fans out), so one send block suffices -> take [0].
      # BUT under IMCFLOW_RESIDUAL_IN_REGION a merged region can consume one model
      # input at TWO distinct receivers on DIFFERENT fifos (e.g. b1 entry minmax on
      # fifo2 AND the residual add's skip on fifo3). Those are separate router
      # entries -> each needs its own send block, else the un-sent receiver's RECV
      # blocks forever. Emit every output edge in that case; the downstream
      # param_edges_by_hid grouping then interleaves them. OFF / single-consumer /
      # non-residual fanout params keep the [0]-only behavior -> byte-identical.
      out_edges = self.get_output_edges_from_id(param_id)
      if (residual_in_region_mode() and len(out_edges) > 1
          and _feeds_residual_add(out_edges)):
        param_edges.extend(out_edges)
      else:
        param_edges.append(out_edges[0])
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
      # Task #10: a RESBUF (INODE_BUFFER) hop endpoint has no relay node. Its
      # resend SEND (src tensor_type "resbuf_out") is emitted by
      # _add_resid_buffer_blocks(), NOT the const-edge path -> skip it here.
      _src_node = CustomIDToNode().get(getInnerNodeID(arg_id))
      if _src_node is None:
        continue
      if ConstPat.match(_src_node):
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
    # Under IMCFLOW_RESIDUAL_IN_REGION a single model-input param can fan out to
    # TWO DISTINCT receivers (different graph nodes / fifos) from the same inode.
    # add_send_block_interleaved is for a split/same-data multicast stream and would
    # bill each edge the full size (2x -> send 8192 vs recv 4096). Those distinct
    # unicast fan-outs must instead be emitted as SEPARATE per-edge send blocks
    # (each 4096, matching its receiver). Pull them out before the HID grouping.
    # residual_fanout_groups: {src_id: [edgeA, edgeB, ...]} for a model-input param
    # that fans out to >1 distinct converging consumer. These are emitted as ONE
    # word-INTERLEAVED send block per source (not N sequential per-edge SendBlocks):
    # both streams reach the same in-region residual add, so a serialized full-A-
    # then-full-B burst starves the converge and wedges the first tile (region1
    # 20000-poll deadlock). Interleaving alternates SEND-to-A / SEND-to-B per word,
    # each keeping its own fifo/policy/rendezvous, so neither consumer starves.
    residual_fanout_groups = {}
    grouped_param_edges = param_edges
    if residual_in_region_mode():
      from collections import defaultdict as _dd
      _by_src = _dd(list)
      for e in param_edges:
        _by_src[e.src_id].append(e)
      residual_fanout_groups = {src: es for src, es in _by_src.items() if len(es) > 1}
      _fanout_edges = {e for es in residual_fanout_groups.values() for e in es}
      grouped_param_edges = [e for e in param_edges if e not in _fanout_edges]

    for src, edges in residual_fanout_groups.items():
      self.add_send_block_residual_fanout_interleaved(
          edges, CodePhase.EXEC_TILE if self.is_tiled else CodePhase.EXEC)

    param_edges_by_hid = {}
    for edge in grouped_param_edges:
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

    # Task #8: func_out edges that will be emitted INTERLEAVED with a RESBUF resend
    # (the add-imce this inode both feeds via resend AND collects func_out from) must
    # NOT also be emitted as a standalone recv block below -- that plain collector
    # RECV, placed before the resend, is exactly the ordering that deadlocks. Pull
    # them out here; _add_resid_buffer_blocks() emits the interleaved version.
    interleave_funcout = self._resbuf_funcout_interleave_edges()
    output_edges = [e for e in output_edges if e not in interleave_funcout]

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

    # Task #10: in-region residual-skip buffer (RESBUF). For each RESBUF assigned
    # to an inode in THIS function, emit the two hops on that inode:
    #   collector: INODE_RECV hop A (producer imce -> inode dmem buffer)
    #   resend   : INODE_SEND hop B (inode dmem buffer -> add imce)
    # The producer imce SEND (hop A) and the add imce RECV (hop B) are emitted by
    # the imce codegen (unchanged in shape). Mirrors the func_out collector +
    # a simple in-order resend loop (pixel-timing is task #9).
    self._add_resid_buffer_blocks()

  def _resbuf_funcout_interleave_edges(self):
    """Task #8: return {func_out_edge: resbuf_gid} for every RESBUF on THIS inode
    whose fed add-imce is ALSO the producer of a func_out this inode collects.

    Those func_out collector RECVs must be interleaved with the RESBUF resend (see
    ResidResendFuncoutInterleavedBlock) rather than emitted as a standalone leading
    recv (which deadlocks: the add can't produce func_out until resent). No-op /
    empty unless the residual inode-buffer sub-lever fired -> byte-identical."""
    result = {}
    from tvm.relay.op.contrib.imcflow import residual_inode_buffer_mode
    if not residual_inode_buffer_mode():
      return result
    resid_info = getattr(DevConfig(), "ResidBufferInfo", None)
    if not resid_info:
      return result
    def _outer(gid):
      # graph_node_id may be a (outer, inner) composite tuple; the func_out edge
      # (src inner 79) and the resbuf hopB dst (inner 76) are DIFFERENT internal
      # calls of the SAME add composite (outer 105) -> match on the OUTER id.
      return gid[0] if isinstance(gid, tuple) else gid
    for resbuf_gid, info in resid_info.items():
      if info.get("func_name") != self.func_name:
        continue
      hopB = info["hopB"]  # (RESBUF,resbuf_out) -> (add,data)
      add_outer = _outer(hopB.dst_id.graph_node_id)
      resend_hid = self.get_hid(hopB.src_id)
      # the func_out this inode collects whose producer is the same add-imce
      fn_id = getNodeID(self.func)
      for out_edge in self.get_input_edges_from_id(fn_id):
        if "func_out" not in str(getattr(out_edge.dst_id, "tensor_type", "")):
          continue
        if _outer(out_edge.src_id.graph_node_id) != add_outer:
          continue
        if self.get_hid(out_edge.dst_id, out_edge.split_idx) != resend_hid:
          continue
        result[out_edge] = resbuf_gid
    return result

  def _auto_fill_lead_groups(self, hopA, hopB, total_words, group, ngroups):
    """Derive the RESBUF fill-lead LAG from graph geometry (no hand tuning).

    Receptive-field arithmetic (see resid_fill_lead_groups docstring for the
    deadlock analysis whose window this computes):
      N*      = diverge pixels the MAIN path must consume before the add's
                first lhs pixel arrives (per-conv "output px k needs input
                through px k_in" composed along the chain; passthrough ops
                are 1:1; max over data+rhs operand paths).
      f_skip  = skip-chain words produced by diverge px N* (same arithmetic
                forward, x producer words/px = total_words / producer out-px).
      LAG_lb  = ceil((f_skip - FIFO_SLACK) / group)   [prime + fifo must
                absorb f_skip while funcout(0) blocks]
      LAG     = min(LAG_lb + 1, ngroups - 1)          [+1 margin; the upper
                bound (skip self-saturation ~ f_skip) keeps LAG*group <=
                f_skip-ish, which lb+1 respects]
    FIFO_SLACK is the inode recv-fifo depth (INODE_RECV_FIFO_DEPTH=2 words
    in params.svh) -- NOT 16. The original 16 assumption wedged subset31_orig
    region3 (join 104->105: need f_skip=64w, absorbed 52 prime + 2 fifo + ~7
    router/send-fifo in-flight = 61w, deadlocked 3w short; fsim fifo3 showed
    push54/pop52 = full at 2). Router in-flight capacity is left as free
    margin, never counted. Upper bound (do not prime too much): once the
    prime exceeds ~(N* - rf_fill) groups the MAIN path emits funcout words
    the inode is not yet receiving, the funcout fifo (also depth 2) fills,
    the diverge node blocks, and -- since the fill chain hangs off the same
    diverge -- the prime itself deadlocks. lb+1 stays far below that.
    Anchors: subset18 b2.res (N*=28px, f_skip=32w, 16 groups) -> 9,
    RTL bit-exact; subset31_orig region3 join (f_skip=64w, 64 groups) -> 17
    (13 from the old slack=16 formula wedged). Returns None if the walk
    fails; the consuming block then raises unless IMCFLOW_RESID_FILL_LEAD
    is set (no magic fallback -- the window is geometry-dependent)."""
    FIFO_SLACK = 2  # INODE_RECV_FIFO_DEPTH (params.svh); words, per fifo
    edges = DevConfig().TensorEdgeListDict.get(self.func_name, [])

    def _outer(gid):
      return gid[0] if isinstance(gid, tuple) else gid

    resbuf_gid = _outer(hopA.dst_id.graph_node_id)
    # C1b (C) Stage 3: OTHER cross-wave RESBUFs may have spliced (producer ->
    # RESBUF -> consumer) hops into this edge list. The residual fill-lead
    # receptive-field walk must see the ORIGINAL producer->consumer edge (a
    # synthetic RESBUF node has no relay op -> _conv_of returns None -> walk
    # aborts). Collapse each cross-wave RESBUF's two hops back into one transparent
    # producer->consumer edge for this walk only (does not mutate the real list).
    _xw_gids = getattr(DevConfig(), "CrossWaveResbufGids", set())
    _xw_src = {}   # xw resbuf gid -> its producer src TensorID
    for e in edges:
      if _outer(e.dst_id.graph_node_id) in _xw_gids:
        _xw_src[_outer(e.dst_id.graph_node_id)] = e.src_id
    _collapsed = []
    for e in edges:
      sg = _outer(e.src_id.graph_node_id)
      if sg in _xw_gids and sg != resbuf_gid and sg in _xw_src:
        # resbuf_out -> consumer : rewrite src to the RESBUF's real producer
        _collapsed.append(TensorEdge(_xw_src[sg], e.dst_id, e.split_idx))
      elif _outer(e.dst_id.graph_node_id) in _xw_gids:
        continue  # drop producer -> RESBUF hop (folded into the collapsed edge)
      else:
        _collapsed.append(e)
    in_edges = {}
    for e in _collapsed:
      if getattr(e.dst_id, "tensor_type", None) not in ("data", "rhs"):
        continue
      if _outer(e.src_id.graph_node_id) == resbuf_gid:
        continue  # the RESBUF resend operand is what we're scheduling
      in_edges.setdefault(_outer(e.dst_id.graph_node_id), []).append(e)

    def _conv_of(gid_inner):
      try:
        n = CustomIDToNode()[getInnerNodeID(gid_inner)]
      except Exception:
        return None
      if (isinstance(n, relay.expr.Call) and isinstance(n.op, tvm.ir.Op)
          and n.op.name in ("nn.imcflow_qconv", "nn.imcflow_qdwconv")):
        try:
          sh = [int(x) for x in n.args[0].checked_type.shape]
        except Exception:
          try:
            sh = [int(x) for x in infer_shape(n.args[0])]
          except Exception:
            return None
        k = int(n.attrs.kernel_size[0])
        p = int(n.attrs.padding[0])
        s = int(n.attrs.strides[0]) if n.attrs.strides else 1
        return (k, p, s, sh[2], sh[3])
      return None

    def _in_px(conv, k):
      """Input px count needed for output raster px index k of `conv`."""
      kk, pp, ss, H, W = conv
      Wo = (W + 2 * pp - kk) // ss + 1
      r, c = k // Wo, k % Wo
      rin = min(max(r * ss + (kk - 1) - pp, 0), H - 1)
      cin = min(max(c * ss + (kk - 1) - pp, 0), W - 1)
      return rin * W + cin

    # Skip back-chain: producer -> ... -> stream source (linear).
    prod_gid = _outer(hopA.src_id.graph_node_id)
    skip_nodes, skip_index = [], {}
    cur = prod_gid
    for _ in range(64):
      es = in_edges.get(cur, [])
      conv = _conv_of(es[0].dst_id.graph_node_id) if es else None
      skip_index[cur] = len(skip_nodes)
      skip_nodes.append((cur, conv))
      if not es:
        break
      cur = _outer(es[0].src_id.graph_node_id)
    if cur not in skip_index:
      skip_index[cur] = len(skip_nodes)
      skip_nodes.append((cur, None))

    # Main walk: max diverge-px need over operand paths reaching the skip chain.
    # Memoized on (gid, k): diamond fan-in patterns (split -> parallel convs ->
    # add) otherwise explode the path count exponentially (25-min codegen hang
    # on subset18 region2 without this).
    add_outer = _outer(hopB.dst_id.graph_node_id)
    _memo = {}
    def _need(gid, k, depth=0):
      if gid in skip_index:
        return (k + 1, gid)
      if depth > 32:
        return None
      key = (gid, k)
      if key in _memo:
        return _memo[key]
      _memo[key] = None  # cycle guard
      best = None
      for e in in_edges.get(gid, []):
        conv = _conv_of(e.dst_id.graph_node_id)
        k_in = _in_px(conv, k) if conv else k
        r = _need(_outer(e.src_id.graph_node_id), k_in, depth + 1)
        if r is not None and (best is None or r[0] > best[0]):
          best = r
      _memo[key] = best
      return best

    res = _need(add_outer, 0)
    if res is None:
      return None
    nstar, hit = res

    # Skip forward: producible producer out-px given nstar px at the diverge.
    n = nstar
    for j in range(skip_index[hit] - 1, -1, -1):
      _, conv = skip_nodes[j]
      if conv is None:
        continue
      kk, pp, ss, H, W = conv
      Wo = (W + 2 * pp - kk) // ss + 1
      Ho = (H + 2 * pp - kk) // ss + 1
      cnt = 0
      for kpx in range(Ho * Wo):
        if _in_px(conv, kpx) + 1 <= n:
          cnt += 1
        else:
          break
      n = cnt
    prod_conv = skip_nodes[0][1]
    if prod_conv is not None:
      kk, pp, ss, H, W = prod_conv
      out_px = ((H + 2 * pp - kk) // ss + 1) * ((W + 2 * pp - kk) // ss + 1)
    else:
      out_px = max(1, ngroups)
    wordsppx = max(1, total_words // max(1, out_px))
    f_words = n * wordsppx
    lag_lb = max(1, -(-max(f_words - FIFO_SLACK, 0) // group))
    lag = min(lag_lb + 1, max(1, ngroups - 1))
    print(f"[resid-auto-lag] N*={nstar}px @diverge {hit}, f_skip={f_words}w "
          f"({n}px x {wordsppx}w), lb={lag_lb} -> LAG={lag} "
          f"(groups={ngroups}; IMCFLOW_RESID_FILL_LEAD overrides)")
    return lag

  def _add_resid_buffer_blocks(self):
    """Task #10: emit the RESBUF collector (RECV hop A) + resend (SEND hop B) on
    the assigned inode. No-op unless the residual inode-buffer sub-lever fired.

    Fill-lead LAG is AUTO-DERIVED per RESBUF from graph geometry (see
    _auto_fill_lead_groups); IMCFLOW_RESID_FILL_LEAD overrides.

    Task #8: when this inode ALSO collects the fed add-imce's func_out, the resend
    (hop B) and that func_out collector are emitted INTERLEAVED (per output group)
    so the add's fed-then-produce dependency does not deadlock. Otherwise the plain
    in-order resend is used."""
    from tvm.relay.op.contrib.imcflow import residual_inode_buffer_mode
    if not residual_inode_buffer_mode():
      return
    resid_info = getattr(DevConfig(), "ResidBufferInfo", None)
    if not resid_info:
      return
    phase = CodePhase.EXEC_TILE if self.is_tiled else CodePhase.EXEC
    # func_out edges routed through the interleaved block, by resbuf gid.
    interleave_by_gid = {gid: e for e, gid in
                         self._resbuf_funcout_interleave_edges().items()}
    for resbuf_gid, info in resid_info.items():
      if info.get("func_name") != self.func_name:
        continue
      hopA = info["hopA"]  # (producer,odata) -> (RESBUF,resbuf)   collector RECV
      hopB = info["hopB"]  # (RESBUF,resbuf_out) -> (add,data)      resend  SEND
      db = DevConfig().CurrFuncMemLayout.get_data_block_by_edge(hopA)
      assert db is not None, f"RESBUF DataBlock not found for {hopA}"
      recv_info = DevConfig().get_tensor_edge_info(hopA)
      recv_hid = self.get_hid(hopA.dst_id)
      # resend: INODE_SEND hop B out of the buffer.
      send_info = DevConfig().get_tensor_edge_info(hopB)
      send_hid = self.get_hid(hopB.src_id)
      fout_edge = interleave_by_gid.get(resbuf_gid)
      # C1b (C) Stage 3: a CROSS-WAVE RESBUF is a plain store-and-forward across a
      # launch boundary, NOT a residual-add convergence -- it has no fill-lead
      # pixel-timing (its consumer is a next-wave conv, not an add fed by two
      # in-flight branches). Force the plain in-order resend path (fout_edge=None)
      # so the residual-only _auto_fill_lead_groups geometry walk is skipped.
      if resbuf_gid in getattr(DevConfig(), "CrossWaveResbufGids", set()):
        fout_edge = None
      # Task #8 iter3: fold the collector fill (hop A) into the per-group interleave
      # only when it is co-located on the SAME inode as the resend (hop B) -- i.e.
      # the resend/funcout interleave path fires. A standalone 256-word collector
      # drain BEFORE the resend wedges (the skip producer only trickles words as the
      # input streams; demanding all 256 up-front starves the fed add and
      # backpressures the region -- iter2 stalled at collector word 64/256).
      fold_fill = fout_edge is not None and recv_hid == send_hid
      # Stage 4 wave placement: collector RECV (hop A) belongs to the PRODUCER's
      # wave (it drains that producer's skip output); resend/interleave (hop B)
      # belongs to the fed add-imce's (CONSUMER) wave. If they differ this RESBUF
      # bridges waves -> a Stage-3 cross-wave case; a RECV placed at the later
      # wave only stalls (safe), so we place the collector at the producer wave
      # and the resend at the consumer wave, and flag the mismatch.
      _wA = self._tensor_wave(hopA.src_id)
      _wB = self._tensor_wave(hopB.dst_id)
      _wA = _wA if _wA is not None else 0
      _wB = _wB if _wB is not None else 0
      if _wA != _wB:
        self._cross_wave_edges.append((hopA, _wA, _wB))
      if not fold_fill:
        # collector: standalone INODE_RECV hop A into the buffer (plain-resend path).
        recv_block = RecvBlock(self, db, recv_info.fifo_id,
                               f"resbuf collector recv: {hopA}")
        self._emit_stream(recv_hid, recv_block, phase, _wA)
      if fout_edge is not None:
        # Task #8: interleave the resend with the add's func_out collector, and
        # (iter3) fold the collector fill in per-group when co-located.
        fout_info = DevConfig().get_tensor_edge_info(fout_edge)
        fout_db = DevConfig().CurrFuncMemLayout.get_data_block_by_edge(fout_edge)
        assert fout_db is not None, f"func_out DataBlock not found for {fout_edge}"
        _total = -(-db.size // 32)
        _auto_lag = self._auto_fill_lead_groups(hopA, hopB, _total, 4, _total // 4)
        il_block = ResidResendFuncoutInterleavedBlock(
            self, db, send_info, fout_db, fout_info.fifo_id, group=4,
            collector_fifo_id=(recv_info.fifo_id if fold_fill else None),
            auto_lag=_auto_lag,
            annotation=f"resbuf resend/funcout interleave: {hopB} || {fout_edge}")
        # stash for consistency counting (bill resend hopB + func_out edge, and the
        # collector hop A when folded in).
        il_block._resbuf_resend_edge = hopB
        il_block._funcout_edge = fout_edge
        il_block._resbuf_collector_edge = hopA if fold_fill else None
        # resend/funcout interleave runs in the fed add-imce's (consumer) wave.
        self._emit_stream(send_hid, il_block, phase, _wB)
        print(f"[resid-inode-buffer] codegen RESBUF {resbuf_gid}: "
              f"{'fill+' if fold_fill else 'standalone-collector RECV @ %s + ' % recv_hid}"
              f"resend/funcout INTERLEAVE @ {send_hid} "
              f"(collector fifo {recv_info.fifo_id}, funcout fifo {fout_info.fifo_id})")
        continue
      # plain in-order resend (no co-located func_out collector on this inode).
      # C1b (C) Stage 3: a MULTICAST cross-wave RESBUF collects the shared stream
      # ONCE (hopA above) and re-sends it UNICAST to EACH cross-wave consumer;
      # info["hopBs"] holds all resend edges (>=1). A residual / unicast RESBUF
      # has a single hopB. Emit one SendBlock per resend, each billed to its own
      # edge for consistency; the collector RECV (hopA) already emitted once above.
      hopBs = info.get("hopBs", [hopB])
      for _hb in hopBs:
        _send_info = DevConfig().get_tensor_edge_info(_hb)
        _send_hid = self.get_hid(_hb.src_id)
        _wBk = self._tensor_wave(_hb.dst_id)
        _wBk = _wBk if _wBk is not None else _wB
        send_block = SendBlock(self, db, _send_info,
                               f"resbuf resend send: {_hb}, "
                               f"{_send_info.policy_info[0].router_id.name} -> "
                               f"{_send_info.policy_info[-1].router_id.name}")
        self._emit_stream(_send_hid, send_block, phase, _wBk)
      print(f"[resid-inode-buffer] codegen RESBUF {resbuf_gid}: "
            f"collector RECV @ {recv_hid} (fifo {recv_info.fifo_id}) + "
            f"{len(hopBs)} resend SEND(s)")

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


    def _resbuf_owner_edge(edges, tensor_types):
      """Task #10: a RESBUF block's DataBlock lists BOTH hops. Count only the hop
      matching this block's direction (resbuf_out for the resend SEND; resbuf for
      the collector RECV) so each hop is billed exactly once."""
      if not isinstance(edges, list):
        return None
      for e in edges:
        if getattr(e.src_id, "tensor_type", None) in tensor_types or \
           getattr(e.dst_id, "tensor_type", None) in tensor_types:
          return e
      return None

    all_blocks = self.codeblocks.get_blocks()
    # IMCFLOW_RESIDUAL_IN_REGION: SendBlocks per shared DataBlock. A residual
    # per-edge fanout emits N SendBlocks over ONE DataBlock (owner-narrowed
    # billing below); a single multicast SendBlock over a shared DataBlock (the
    # OFF-style [0]-only param send whose one SEND reaches every edge of the
    # multicast) must bill the WHOLE edge list -- owner-narrowing it leaves the
    # sibling edge at send 0 (false "send 0 vs recv N" deadlock report).
    _sendblocks_per_db = {}
    if residual_in_region_mode():
      for _b in all_blocks:
        if isinstance(_b, SendBlock):
          _k = id(_b.block)
          _sendblocks_per_db[_k] = _sendblocks_per_db.get(_k, 0) + 1
    for block in all_blocks:
      if isinstance(block, SendBlock):
        edge = block.block.id
        # Task #10: RESBUF resend SEND -> count only hop B (src tensor_type
        # "resbuf_out"); the collector RECV hop A (in the same shared DataBlock)
        # is billed by the RecvBlock branch below.
        _resbuf_e = _resbuf_owner_edge(edge, ("resbuf_out",))
        if _resbuf_e is not None:
          _add_send_map(self.send_map, block.block, _resbuf_e)
          continue
        # IMCFLOW_RESIDUAL_IN_REGION: a model-input multicast fan-out emits a
        # SEPARATE per-edge SendBlock per receiver, but all of them share ONE
        # DataBlock whose .id lists BOTH edges. Counting block.id (the list) for
        # each per-edge block bills every edge once PER fan-out block (2 blocks x
        # 2 edges -> each edge counted 2x -> send 8192 vs recv 4096 false
        # deadlock). Count only THIS block's own edge (edge_info.owner) so each
        # edge is billed exactly once. Applies ONLY when the DataBlock really
        # has multiple SendBlocks; a lone multicast SendBlock bills every edge
        # (one physical SEND reaches all receivers). OFF / non-fanout ->
        # unchanged.
        if (residual_in_region_mode() and isinstance(edge, list)
            and _sendblocks_per_db.get(id(block.block), 0) > 1):
          own = getattr(block.edge_info, "owner", None)
          if own is not None and own in edge:
            edge = own
        _add_send_map(self.send_map, block.block, edge)
      elif isinstance(block, ResidResendFuncoutInterleavedBlock):
        # Task #8: RESBUF resend (hop B) interleaved with the add-imce's func_out
        # collector. Bill BOTH halves exactly once: the resend edge as a SEND
        # (mirrors the plain SendBlock resbuf_out branch) and the func_out edge as
        # a RECV (mirrors the standalone collector we suppressed). Counts are the
        # RESBUF word count (both hops of the same add, equal length). Lever OFF ->
        # class never constructed -> byte-identical.
        _resend_e = getattr(block, "_resbuf_resend_edge", None)
        _fout_e = getattr(block, "_funcout_edge", None)
        _coll_e = getattr(block, "_resbuf_collector_edge", None)
        if _resend_e is not None:
          _add_send_map(self.send_map, block.resbuf_db, _resend_e)
        if _fout_e is not None:
          if block.funcout_db.tiling_info:
            _rc = sum(block.funcout_db.tiling_info.pkt_cnts)
          else:
            _rc = math.ceil(block.funcout_db.size / 32)
          add_to_map(self.recv_map, _fout_e, _rc)
        # Task #8 iter3: when the collector fill (hop A) is folded into this block
        # (co-located inode), bill its RECV here (the standalone RecvBlock that used
        # to bill it is suppressed). Count = resbuf word count. When NOT folded, hop
        # A is billed by the standalone RecvBlock as usual (byte-identical).
        if _coll_e is not None:
          _cc = math.ceil(block.resbuf_db.size / 32)
          add_to_map(self.recv_map, _coll_e, _cc)
      elif isinstance(block, SendBlockResidualFanoutInterleaved):
        # IMCFLOW_RESIDUAL_IN_REGION: word-interleaved model-input fan-out. All
        # edges SHARE ONE DataBlock whose .id lists every fan-out edge, so counting
        # db.id (the shared list) would bill every edge once PER db (2 dbs x 2 edges
        # -> each edge 2x -> send 8192 vs recv 4096 false deadlock, the original
        # 1ee91d854 bug in interleaved form). Bill only each edge's OWN owner
        # (edge_info.owner) so each edge is counted exactly once. Lever OFF ->
        # class never constructed -> byte-identical.
        for db, ei in zip(block.blocks, block.edge_infos):
          own = getattr(ei, "owner", None)
          edge = own if own is not None else db.id
          _add_send_map(self.send_map, db, edge)
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
        # Task #10: RESBUF collector RECV -> count only hop A (dst tensor_type
        # "resbuf"); the resend SEND hop B is billed by the SendBlock branch.
        _resbuf_e = _resbuf_owner_edge(edge, ("resbuf",))
        if _resbuf_e is not None:
          edge = _resbuf_e
        # Calculate actual number of recv operations (loop count)
        # recv_count = math.ceil(block.block.size / 32)
        if block.block.tiling_info:
          recv_count = sum(block.block.tiling_info.pkt_cnts)
        else:
          recv_count = math.ceil(block.block.size / 32)
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
    self._emit_stream(hid, block, phase, self._edge_wave(edge))

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
    # Stage 4: a multicast/interleaved SEND must not precede ANY consumer's wave;
    # place at the latest consumer wave (conservative, deadlock-safe).
    _w = max((self._edge_wave(e) for e in edge_list), default=0)
    self._emit_stream(hid, block, phase, _w)

  def add_send_block_residual_fanout_interleaved(self, edge_list, phase: CodePhase):
    """IMCFLOW_RESIDUAL_IN_REGION: emit the model-input residual fan-out (one inode
    source -> TWO converging consumers) as ONE word-interleaved send loop instead of
    N sequential per-edge SendBlocks. Serializing them (full burst A, then full burst
    B) starves the converging consumer and wedges the first tile (region1 20000-poll
    deadlock). Each edge keeps its own fifo/policy and per-word rendezvous; only the
    emission is interleaved. Each edge carries its OWN edge_info (owner) so the
    downstream helper SendBlock derives the correct per-consumer rendezvous."""
    hids = [self.get_hid(edge.src_id) for edge in edge_list]
    assert all(hid == hids[0] for hid in hids), \
        "all residual-fanout edges must share the source inode for interleaving"
    hid = hids[0]

    dbs = []
    edge_infos = []
    annotation = ""
    for edge in edge_list:
      out_edge_info = DevConfig().get_tensor_edge_info(edge)
      db = DevConfig().CurrFuncMemLayout.get_data_block_by_edge(edge)
      assert db is not None, f"Data block not found for residual-fanout edge: {edge}"
      dbs.append(db)
      edge_infos.append(out_edge_info)
      annotation += (f"resid-fanout send - {edge}, "
                     f"{out_edge_info.policy_info[0].router_id.name} -> "
                     f"{out_edge_info.policy_info[-1].router_id.name}, ")

    block = SendBlockResidualFanoutInterleaved(self, dbs, edge_infos, annotation)
    # Stage 4: residual fan-out feeds two converging consumers (possibly in
    # different waves); place at the later consumer wave (SEND never before a
    # consumer's cores are enabled). If they differ it is a cross-wave case.
    _w = max((self._edge_wave(e) for e in edge_list), default=0)
    self._emit_stream(hid, block, phase, _w)

  def add_recv_block(self, edge, phase: CodePhase):
    in_edge_info = DevConfig().get_tensor_edge_info(edge)
    in_tid = edge.dst_id
    hid = self.get_hid(in_tid, edge.split_idx)
    db = DevConfig().CurrFuncMemLayout.get_data_block_by_edge(edge)

    block = RecvBlock(self, db, in_edge_info.fifo_id, f"recv: {in_tid}")
    self._emit_stream(hid, block, phase, self._edge_wave(edge))
  
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
    # Stage 4: collector RECV over several producers; place at the latest wave so
    # the inode has been (re)programmed for all of them. A RECV only stalls.
    _w = max((self._edge_wave(e) for e in edge_list), default=0)
    self._emit_stream(hid, block, phase, _w)

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

  # ---- C1b (C) Stage 4: streaming wave resolution & deferral --------------
  def _tensor_wave(self, tensor_id):
    """Wave of a tensor endpoint via GraphNodeToWavePerFunc (IMCE nodes only).
    Returns None for inode endpoints (var/const/func_out/RESBUF -- not in the
    wave map) so the caller falls back to the other (IMCE) endpoint. Mirrors
    get_hid's tuple(composite) fallback."""
    gmap = DevConfig().GraphNodeToWavePerFunc.get(self.func_name, {})
    gid = tensor_id.graph_node_id
    if gid in gmap:
      return gmap[gid]
    if isinstance(gid, tuple):
      if gid[-1] in gmap:
        return gmap[gid[-1]]
      if gid[0] in gmap:
        return gmap[gid[0]]
    return None

  def _edge_wave(self, edge):
    """Launch wave for a data edge = its IMCE endpoint's wave. Consumer (recv)
    wave takes priority (const/param inode->imce, and the general case); producer
    wave is the fallback (func_out imce->inode). A direct edge whose producer and
    consumer waves DIFFER is a cross-wave edge (Stage 3 reroutes it); Stage 4
    classifies it by the CONSUMER (later) wave so a SEND is never placed before
    its consumer's cores are enabled. Defaults to 0 (single-wave / unmapped)."""
    w_dst = self._tensor_wave(edge.dst_id)
    w_src = self._tensor_wave(edge.src_id)
    if w_dst is not None and w_src is not None and w_dst != w_src:
      self._cross_wave_edges.append((edge, w_src, w_dst))
      return max(w_src, w_dst)
    if w_dst is not None:
      return w_dst
    if w_src is not None:
      return w_src
    return 0

  def _emit_stream(self, hid, block, phase, wave):
    """Append a streaming block, deferring it to its wave bucket in multi-wave
    regions (flushed in emit_wave_launches). Off merge -> immediate append =
    byte-identical.

    EXEC/EXEC_TILE (data-streaming) blocks are wave-deferred by their edge wave.

    INIT-phase const/config SENDs are ALSO deferred to their edge's wave segment
    under merge, for EVERY wave (including wave 0). Reason: with the wave-launch
    segment structure, INIT does ONLY policy + weights (WR_IMCU) -- NO imce is
    COMPUTE-enabled during INIT; each wave's WR_IMEM+COMPUTE+stream live together in
    that wave's own EXEC launch segment (see initialize()/emit_wave_launches). The
    imce codegen stamps each node's RecvConstBlock into its wave-k IMEM
    (current_wave, visit_call), so the matching const RECV only runs after wave-k's
    COMPUTE. A const SEND left in INIT would issue its (pack-const) rendezvous
    STANDBY(imce, ...) against a core that is not yet enabled -> the inode wedges in
    the WR_IMCU/INIT segment with no consumer to drain it (region2: inode_3_0/0_0
    stuck at STANDBY(1) in GO#1 while all imces IDLE). Deferring the const SEND to
    its wave segment (as EXEC, flushed after that wave's WR_IMEM+COMPUTE) matches the
    imce-side RecvConstBlock, for wave 0 and wave k alike.

    Off merge (n_waves==1, _defer_streams False) -> immediate append =
    byte-identical (INIT const sends stay in the INIT preamble as before)."""
    if self._defer_streams and phase in (CodePhase.EXEC, CodePhase.EXEC_TILE):
      self._wave_streams.setdefault(wave, []).append((hid, phase, block))
    elif self._defer_streams and phase == CodePhase.INIT:
      # Merged region: NO imce is COMPUTE-enabled in INIT (weights-only). Emit this
      # const/config SEND inside its wave's launch segment (as EXEC) so it lands
      # AFTER that wave's WR_IMEM+COMPUTE, matching the imce wave-k RecvConstBlock.
      self._wave_streams.setdefault(wave, []).append((hid, CodePhase.EXEC, block))
    else:
      self.codeblocks.append(hid, block, phase)
