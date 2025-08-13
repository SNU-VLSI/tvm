import tvm
from tvm import relay
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from tvm.contrib.imcflow import DataBlock
from . import transform as imcflow_transform


@tvm._ffi.register_func("relay.ext.imcflow")
def imcflow_external_codegen(func: relay.Function):
  # Obtain the function name (global symbol) assigned by the partitioning pass
  func_name = func.attrs["global_symbol"] if hasattr(func, "attrs") and "global_symbol" in func.attrs else "imcflow_subgraph"

  # Ensure instruction/data blocks required by the existing kernel generator are present.
  # This mirrors the allocation used in tvm_practice/test_imcflow/codegen/test.py
  # so that makeKernelStartCode can discover blocks via getInstructionBlocks/getDataBlocks.
  try:
    # Allocate inode instruction placeholders
    for i in range(4):
      inode_inst = DataBlock(f"{func_name}_inst_inode{i}", 4)
      inode_inst.set_base_address(8 + i * 4)
      DevConfig().MemLayout[f"inode_{i}_inst"].allocate(inode_inst)

    # Allocate imce instruction placeholders (placed in corresponding inode data region)
    for i in range(DevConfig().IMCE_NUM):
      imce_inst = DataBlock(f"{func_name}_inst_imce{i}", 4)
      imce_inst.set_base_address(8 + 4 * 4 + i * 4)
      inode_idx = i % 4
      DevConfig().MemLayout[f"inode_{inode_idx}_data"].allocate(imce_inst)
  except Exception:
    # If MemLayout not initialized here, continue; the generator can still produce code,
    # though some transfer loops may be empty.
    pass

  # Reuse existing kernel code generator
  code = imcflow_transform.makeKernelStartCode(func_name, func)

  # Wrap as a CSourceModule so TVM can compile/link it with the rest of the MLF
  # Note: returning a CSourceModule is the standard for BYOC Python codegen.
  return tvm.runtime._ffi_api.CSourceModuleCreate(code, "cc", [func_name], None)


@tvm._ffi.register_func("relay.ext.imcflow.constant_updater")
def imcflow_constant_updater(expr, symbol):  # pylint: disable=unused-argument
  # Keep ownership of constants inside the external module
  return dict()