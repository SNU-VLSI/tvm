from typing import List
import tvm
from tvm import relay
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from tvm.contrib.imcflow import DataBlock
from tvm.relay.ty import TensorType, TupleType
from . import transform as imcflow_transform
from tvm.contrib.imcflow import ImcflowDeviceConfig, TensorID, DataBlock, TensorEdge
from tvm.relay.op.contrib.imcflow import CustomIDToNode
from tvm.relay.expr import (Var, Constant)
from tvm.runtime import String
import math
import os


IMCFLOW_ADDR = 0x80000000
IMCFLOW_LEN = DevConfig.IMCFLOW_ADDR_SIZE
INT_ACK_GEN_ADDR = 0
INT_ACK_GEN_LEN = 0

# Device paths
IMCFLOW_DEVICE = "/dev/uio5"
INT_ACK_GEN_DEVICE = "/dev/uio4"
# Code generation constants
CONST_TAGS = ["weight", "bias", "fused_scale", "fused_bias", "min", "max", "threshold", "scale", "config"]
ALIGNMENT_BYTES = 32

def getInnerNodeID(graph_node_id):
  if isinstance(graph_node_id, tuple):
    return graph_node_id[1]
  else:
    return graph_node_id

def align_to_n_bytes(size, n_bytes):
  if (size % n_bytes) != 0:
    size = (size // n_bytes + 1) * n_bytes
  return size


def dtype_to_cpp(dtype: str) -> str:
  mapping = {
      "float64": "double",
      "float32": "float",
      "float": "float",
      "uint32": "uint32_t",
      "int32": "int32_t",
      "uint16": "uint16_t",
      "int16": "int16_t",
      "uint8": "uint8_t",
      "int8": "int8_t",
  }

  # if dtype not in mapping: print(dtype)
  return mapping.get(dtype, "unknown_type")


class CodeWriter:
  def __init__(self, indent_str="  "):
    self.lines = []
    self.indent_str = indent_str
    self.indent_level = 0

  def nextIndent(self):
    self.indent_level += 1

  def prevIndent(self):
    self.indent_level -= 1

  def write(self, line=""):
    for line_ in line.split("\n"):
      if len(line_) > 0:
        self.lines.append(
            f"{self.indent_str * self.indent_level}{line_}")

  def get_code(self):
    return "\n".join(self.lines)

  def __str__(self):
    return self.get_code()

  def __add__(self, other):
    if isinstance(other, CodeWriter):
      # Apply current indentation level to incoming lines
      if self.indent_level > 0:
        indent_prefix = self.indent_str * self.indent_level
        indented_lines = [indent_prefix + line for line in other.lines]
        self.lines.extend(indented_lines)
      else:
        self.lines.extend(other.lines)
      return self
    elif isinstance(other, str):
      self.write(other)
      return self

def makeBaseAddrName(block):
  if isinstance(block.id, str):
    return f"{block.id.upper()}_BASE_ADDR"

  # Get first TensorEdge from edges list
  if isinstance(block.id, TensorEdge) or isinstance(block.id, List):
    edge = block.edges[0]  # Use first edge for naming
    graph_node_id = imcflow_transform.getInnerNodeID(edge.src_id.graph_node_id)
    node_id_str = str(graph_node_id).replace("-", "m")
    if edge.src_id.tensor_type in CONST_TAGS:
      return f"{edge.src_id.tensor_type.upper()}_{node_id_str}_BASE_ADDR"
    else:
      return f"{edge.dst_id.tensor_type.upper()}_{node_id_str}_BASE_ADDR"

  raise ValueError("Wrong data block type!")

def makeConstArrayDecl(func, func_name, target_func):
  params = {}
  class ConstantCollector(tvm.relay.ExprVisitor):
      def __init__(self):
          super().__init__()
          self.cnt=0

      def visit_constant(self, const):
          # constant 이름 생성 (symbol 기반)
          node_id = getInnerNodeID(imcflow_transform.getNodeID(const))
          name = f"imcflow_{func_name}_const_{self.cnt}"
          params[String(name)] = const.data
          DevConfig().ImcflowFuncMap[func_name].const_name_map[node_id] = String(name)
          self.cnt += 1
          super().visit_constant(const)

  collector = ConstantCollector()
  collector.visit(target_func)

  code = CodeWriter()
  for const_name, array in params.items():
    dtype = dtype_to_cpp(array.dtype)
    shape = array.shape
    size = 1
    for dim in shape:
      size *= dim
    code += f"static const {dtype} {const_name}[] __attribute__((aligned(16))) = {{"
    array_values = array.asnumpy().flatten()
    for i, val in enumerate(array_values):
      if i % 16 == 0:
        code += "\n  "
      code += f"{val}, "
    code += "\n};\n\n"

  return code


def getConstantIdx(func, node_id):
  """
  Get the index of the constant node in the function by its inner node ID.
  parameters:
    func    : relay.Function
    node_id : inner node ID of the constant node
  """
  node_id_to_constant_id = {}

  class _Visitor(tvm.relay.ExprVisitor):
    def __init__(self):
      super().__init__()
      self.Cnt = 0

    def visit_constant(self, const):
      node_id = getInnerNodeID(imcflow_transform.getNodeID(const))
      node_id_to_constant_id[node_id] = self.Cnt
      self.Cnt = self.Cnt + 1
      super().visit_constant(const)

  _Visitor().visit(func)
  return node_id_to_constant_id[node_id]


def getCInputVarName(func_name, data_block):
  node_map = CustomIDToNode()

  # Get first edge from edges list (handles both single TensorEdge and List[TensorEdge])
  assert data_block.edges, "data_block must have at least one TensorEdge to get C input var name"

  edge = data_block.edges[0]
  graph_node_inner_id = imcflow_transform.getInnerNodeID(edge.src_id.graph_node_id)

  node_type = node_map[graph_node_inner_id]
  if isinstance(node_type, Var):
    return node_type.name_hint
  elif isinstance(node_type, Constant):
    data_type = dtype_to_cpp(node_type.checked_type.dtype)
    node_id = getInnerNodeID(imcflow_transform.getNodeID(node_type))
    const_name = DevConfig().ImcflowFuncMap[func_name].const_name_map[node_id]
    return f"(({data_type}*)({const_name}))"
  else:
    raise ValueError(f"Invalid node_type!: {node_type}")

def getObjectFileName(data_block, func_name=None):
  assert isinstance(data_block.id, str), "data_block.id must be string to get object file name"
  if func_name:
    return f"_binary_{func_name}_{data_block.id}_bin"
  else:
    return f"_binary_{data_block.id}_bin"


class KernelCodeGenerator:
  """Code generator for IMCFlow kernel functions."""

  def __init__(self, func_name, func, os="linux"):
    """
    Initialize the kernel code generator.

    Parameters:
      func_name: Name of the function to generate code for
      func: Relay function (wrap function)
      os: Target OS ("linux" or "baremetal")
    """
    self.func_name = func_name
    self.func = func
    self.os = os

    # Get target function info
    self.target_func_info = DevConfig().ImcflowFuncMap.get(func_name, None)
    if self.target_func_info is None:
      raise ValueError(f"Function {func_name} not found in ImcflowFuncMap")
    self.target_func = self.target_func_info.func_node

    # Get data blocks
    self.compiled_blocks = ImcflowDeviceConfig().DataBlocks[func_name]["compiled"]
    self.compiled_per_tile_blocks = ImcflowDeviceConfig().DataBlocks[func_name]["compiled_per_tile"]
    self.const_blocks = ImcflowDeviceConfig().DataBlocks[func_name]["const"]
    self.input_blocks = ImcflowDeviceConfig().DataBlocks[func_name]["input"]
    self.output_blocks = ImcflowDeviceConfig().DataBlocks[func_name]["output"]

    # Initialize base address macros
    self.base_address_macros = {
        "IMCFLOW_ADDR": IMCFLOW_ADDR,
        "IMCFLOW_LEN": IMCFLOW_LEN,
        "INT_ACK_GEN_ADDR": INT_ACK_GEN_ADDR,
        "INT_ACK_GEN_LEN": INT_ACK_GEN_LEN,
        "IMCFLOW_DEVICE": f'"{IMCFLOW_DEVICE}"',
        "INT_ACK_GEN_DEVICE": f'"{INT_ACK_GEN_DEVICE}"',
        "SET_IDLE_CODE": 0,
        "SET_RUN_CODE": 1,
        "SET_PROGRAM_CODE": 2,
        "STATE_REG_IDX": 0,
        "PC_REG_IDX": 2,
        "INTR_DONE_REG_IDX": 7,
        "INODE_PC_START_P1_ENUM_VAL": 0,
        "INODE_PC_START_EXTERN_ENUM_VAL": 1,
        "INODE_PC_START_P0_ENUM_VAL": 2,
        "INODE_NUM": ImcflowDeviceConfig().INODE_NUM,
    }

    # Get input/output node types
    self.input_nodes = [n for n in imcflow_transform.getInputNodesOfFunc(func)]
    self.input_node_types = [n.checked_type.dtype for n in self.input_nodes]
    self.output_node = imcflow_transform.getOutputNodeOfFunc(func)
    self.output_node_types = self._node_types_to_list(self.output_node)

  def _node_types_to_list(self, node):
    """Extract dtype(s) from a node's checked_type into a list."""
    checked_type = node.checked_type
    if isinstance(checked_type, TupleType):
      return [field.dtype for field in checked_type.fields]
    elif isinstance(checked_type, TensorType):
      return [checked_type.dtype]
    else:
      raise TypeError(f"unsupported node type {checked_type.__class__}")

  def generateHeader(self):
    """Generate C header includes."""
    return ("""
#include <stdlib.h>
#include <string.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/c_backend_api.h>
#include <dlpack/dlpack.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>
""")

  def generateInterruptUtilities(self):
    """Generate interrupt handling utility functions."""
    return ("""
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
""")

  def generateDevicePointerSetup(self):
    """Generate device pointer setup code based on OS."""
    if self.os == "linux":
      return ("""
  int npu_fd = open(IMCFLOW_DEVICE, O_RDWR);
  if (npu_fd < 0) {
    perror("npu UIO cannot be opened");
    exit(1);
  }

  int int_ack_gen_fd = open(INT_ACK_GEN_DEVICE, O_RDWR);
  if (int_ack_gen_fd < 0) {
    perror("interrupt ack gen UIO cannot be opened");
    close(npu_fd);
    exit(1);
  }

  size_t npu_len = (size_t) IMCFLOW_LEN;
  uint32_t *npu_pointer = (uint32_t *) mmap(NULL, npu_len, PROT_WRITE | PROT_READ, MAP_SHARED, npu_fd, 0);
  if (npu_pointer == MAP_FAILED) {
    perror("npu_pointer mmap error");
    close(npu_fd);
    close(int_ack_gen_fd);
    exit(1);
  }

  size_t int_ack_gen_len = (size_t)INT_ACK_GEN_LEN;
  uint32_t *int_ack_gen_pointer = (uint32_t*) mmap(NULL, int_ack_gen_len, PROT_WRITE | PROT_READ, MAP_SHARED, int_ack_gen_fd, 0);
  if (int_ack_gen_pointer == MAP_FAILED) {
    perror("int_ack_gen_pointer mmap error");
    munmap(npu_pointer, npu_len);
    close(npu_fd);
    close(int_ack_gen_fd);
    exit(1);
  }
  """)
    elif self.os == "baremetal":
      return (f"""
    uint32_t* npu_pointer = (uint32_t*)IMCFLOW_ADDR;
    uint32_t* int_ack_gen_pointer = (uint32_t*)INT_ACK_GEN_ADDR;
""")
    else:
      raise ValueError("Unsupported OS type for device pointer setup!")

  def generatePolicyUpdateCode(self):
    """Generate policy update code."""
    out = [
      "// Set the inode pc to 0 and run.",
      "for(int i=0; i<INODE_NUM; i++) {",
      "  npu_pointer[(PC_REG_IDX + i)] = (INODE_PC_START_EXTERN_ENUM_VAL << 30 + 0);",
      "}",
      "enable_imcflow_interrupt(npu_fd);" if self.os == "linux" else "",
      " npu_pointer[STATE_REG_IDX] = SET_PROGRAM_CODE;",
      "wait_imcflow_interrupt(npu_fd);" if self.os == "linux" else "",
      "generate_ack(int_ack_gen_pointer);" if self.os == "linux" else "",
      "npu_pointer[INTR_DONE_REG_IDX] = 1;",
    ]
    return "\n".join(out) + "\n"

  def generateInvokeCode(self):
    """Generate NPU invoke code."""
    out = [
      "for(int i=0; i<INODE_NUM; i++) {",
      "  npu_pointer[(PC_REG_IDX + i)] = (INODE_PC_START_P1_ENUM_VAL << 30 + 0);",
      "}",
      "enable_imcflow_interrupt(npu_fd);" if self.os == "linux" else "",
      "npu_pointer[STATE_REG_IDX] = SET_RUN_CODE;",
      "wait_imcflow_interrupt(npu_fd);" if self.os == "linux" else "",
      "generate_ack(int_ack_gen_pointer);" if self.os == "linux" else "",
        "npu_pointer[INTR_DONE_REG_IDX] = 1;"
    ]
    return "\n".join(out) + "\n"

  def generateDevicePointerCleanup(self):
    """Generate device pointer cleanup code."""
    if self.os == "linux":
      return ("""
  // Cleanup device pointer
  munmap(npu_pointer, npu_len);
  close(npu_fd);
  munmap(int_ack_gen_pointer, int_ack_gen_len);
  close(int_ack_gen_fd);
  """)
    else:
      return ""

  def _get_transfer_loop_params(self, block, tile_idx):
    """
    Calculate loop parameters for data transfer.
    Returns (loop_start, loop_end).
    """
    if block.tiling_info is not None:
      # Tiled transfer
      tiling_info = block.tiling_info
      loop_start = tiling_info.c_var_offsets[tile_idx]
      loop_end = loop_start + tiling_info.c_var_sizes[tile_idx]
    else:
      # Regular transfer
      size = align_to_n_bytes(block.size, ALIGNMENT_BYTES)
      loop_start = 0
      loop_end = math.ceil(size/4)
    return loop_start, loop_end

  def generateToNpuTransferCode(self, blocks, tile_idx=None):
    """Generate code to transfer data to NPU memory."""

    def _appendLoopForObjectFileTransfer(code, block, base_address_name, func_name, tile_idx=None):
      # Binary object file transfer
      var_prefix = getObjectFileName(block, func_name)
      if tile_idx is None:
        loop_start = 0
        loop_end = f"(size_t)({var_prefix}_end-{var_prefix}_start)"
        src_var = f"{var_prefix}_start"
        code += f"for(int i={loop_start}; i<{loop_end}; i++){{\n"
        code += f"  npu_pointer[({base_address_name} / 4) + i] = ((uint32_t*){src_var})[i];\n"
        code += f"}}\n"
      else:
        src_var = f"{var_prefix}_start"
        code += f"npu_pointer[({base_address_name} / 4)] = ((uint32_t*){src_var})[{tile_idx}];\n"
      return code

    def _appendLoopForCVarTransfer(code, block, base_address_name, func_name, tile_idx=None):
      # C Var transfer
      loop_start, loop_end = self._get_transfer_loop_params(block, tile_idx)
      src_var = getCInputVarName(func_name, block)

      code += f"for(int i=0; i<{loop_end-loop_start}; i++){{\n"
      code += f"  npu_pointer[({base_address_name} / 4) + i] = ((uint32_t*){src_var})[i + {loop_start}];\n"
      code += f"}}\n"
      return code

    code = CodeWriter()
    code += "// Transfer data into NPU memory\n"
    for block in blocks:
      base_address = block.base_address
      base_address_name = makeBaseAddrName(block)
      self.base_address_macros.update({base_address_name: base_address})

      # Add tiling comment if applicable
      if block.tiling_info is not None:
        code += f"// Transfer data [TILE:{tile_idx}]\n"

      # Determine source variable and loop parameters based on block type
      if isinstance(block.id, str):
        code = _appendLoopForObjectFileTransfer(code, block, base_address_name, self.func_name, tile_idx)
      else:
        code = _appendLoopForCVarTransfer(code, block, base_address_name, self.func_name, tile_idx)

    return code

  def generateFromNpuTransferCode(self, blocks, tile_idx=None):
    """Generate code to transfer data from NPU memory."""
    code = CodeWriter()
    code += "// Transfer data from NPU memory\n"
    for block in blocks:
      assert "func_out" in block.id.dst_id.tensor_type, "output data block must have 'func_out' in tensor_type"
      idx = block.id.dst_id.tensor_type.replace("func_out", "")
      base_address = block.base_address
      base_address_name = makeBaseAddrName(block)
      self.base_address_macros.update({base_address_name: base_address})

      # Add tiling comment if applicable
      if block.tiling_info is not None:
        code += f"// Transfer data [TILE:{tile_idx}]\n"

      # Get loop parameters
      loop_start, loop_end = self._get_transfer_loop_params(block, tile_idx)

      # Generate loop code
      code += f"for(int i=0; i<{loop_end-loop_start}; i++){{\n"
      code += f"  ((uint32_t*)out{idx})[i + {loop_start}] = npu_pointer[({base_address_name} / 4) + i];\n"
      code += f"}}\n"
    return code

  def generateBaseAddrMacros(self):
    """Generate base address macro definitions."""
    code = CodeWriter()
    for key, value in self.base_address_macros.items():
      code += f"#define {key} {value}\n"
    code += "\n"
    return code

  def generateExternLink(self):
    """Generate extern C linkage declarations for compiled blocks."""
    code = CodeWriter()
    code += 'extern "C" { \n'
    for block in (self.compiled_blocks + self.compiled_per_tile_blocks):
      if isinstance(block.id, str):
        filename = f"_binary_{self.func_name}_{block.id}_bin"
        code += f'  extern const int32_t {filename}_start[];\n'
        code += f'  extern const int32_t {filename}_end[];\n'
    code += '}\n'
    return code

  def generatePackedFuncWrapper(self):
    """Generate PackedFunc wrapper for CRT."""
    code = CodeWriter()
    code += "#ifdef __cplusplus\n"
    code += "extern \"C\"\n"
    code += "#endif\n"
    code += f"TVM_DLL int32_t {self.func_name}(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {{\n"
    code.nextIndent()
    code += "(void)resource_handle;\n"
    code += "if (num_args < 2) return -1;\n"

    num_in_args = len(self.input_node_types)
    num_out_args = len(self.output_node_types)

    # get input and output data pointers
    for idx in range(num_in_args):
      code += f"void* _in{idx} = (((TVMValue*)args)[{idx}].v_handle);\n"
      code += f"DLTensor* in{idx} = (DLTensor*)_in{idx};\n"
    for idx in range(num_out_args):
      code += f"void* _out{idx} = (((TVMValue*)args)[{idx + num_in_args}].v_handle);\n"
      code += f"DLTensor* out{idx} = (DLTensor*)_out{idx};\n"

    # call kernel function
    args_list = []
    for idx, i_type in enumerate(self.input_node_types):
      args_list.append(f"({dtype_to_cpp(i_type)}*)in{idx}->data")
    for idx, o_type in enumerate(self.output_node_types):
      args_list.append(f"({dtype_to_cpp(o_type)}*)out{idx}->data")
    code += f"{self.func_name}_kernel({', '.join(args_list)});\n"

    code += "(void)out_ret_value;\n"
    code += "if (out_ret_tcode) { *out_ret_tcode = kTVMArgInt; }\n"
    code += "return 0;\n"
    code.prevIndent()
    code += "}\n"
    return code

  def makeKernelDef(self):
    """Generate the complete kernel definition."""
    # Build function prototype
    proto_list = []
    for i, param in enumerate(self.func.params):
      impl_param = self.target_func.params[i]
      param_name = impl_param.name_hint if impl_param.name_hint else f"arg{i}"
      dtype = "float32"
      if hasattr(param, "checked_type") and isinstance(param.checked_type, TensorType):
        dtype = param.checked_type.dtype
        cpp_type = dtype_to_cpp(dtype)
        proto_list.append(f"{cpp_type}* {param_name}")

    for idx, node_type in enumerate(self.output_node_types):
      proto_list.append(f"{dtype_to_cpp(node_type)}* out{idx}")

    args_proto_type = ", ".join(proto_list)

    code = CodeWriter()
    code += self.generateHeader()
    code += self.generateExternLink()
    code += makeConstArrayDecl(self.func, self.func_name, self.target_func)
    code += self.generateInterruptUtilities()

    # Kernel function prototype and definition (C)
    code += f"void {self.func_name}_kernel({args_proto_type}) {{\n"
    code.nextIndent()
    code += f"printf(\"{self.func_name}_kernel called\\n\");\n"
    code += self.generateDevicePointerSetup()
    code += self.generateToNpuTransferCode(self.compiled_blocks) # inode instrunction + policy
    code += self.generateToNpuTransferCode(self.const_blocks) # constant
    code += self.generatePolicyUpdateCode() # start from pc 0, up to halt
    code += self.generateInvokeCode() # proceed up to halt

    # kernel tiling factor
    tile_factor = self.target_func_info.tiling_factor
    for t_idx in range(tile_factor):
      code += self.generateToNpuTransferCode(self.compiled_per_tile_blocks, t_idx) # per-tile: cnt_base_addr
      code += self.generateToNpuTransferCode(self.input_blocks, t_idx) # input
      code += self.generateInvokeCode() # end of exec
      code += self.generateFromNpuTransferCode(self.output_blocks, t_idx) # output
    code += self.generateDevicePointerCleanup()
    code.prevIndent()
    code += '}\n'

    code += self.generatePackedFuncWrapper()
    code = self.generateBaseAddrMacros() + code

    return code


def makeKernelStartCode(func_name, func, os="linux"):
  """Generate kernel start code using KernelCodeGenerator class."""
  generator = KernelCodeGenerator(func_name, func, os)
  code = generator.makeKernelDef()
  return str(code)

def generate_invoke_code_for_subgraphs(mod):
  invoke_code_map = {}
  for func_name_var in mod.functions:
    func = mod[func_name_var.name_hint]
    if func.attrs and func.attrs.get("Compiler") == "imcflow":
      func_name = func_name_var.name_hint
      code = makeKernelStartCode(func_name, func)
      invoke_code_map[func_name] = code

  for fn, code in invoke_code_map.items():
    with open(f"{fn}.cc", "w") as f:
      f.write(code)

  return invoke_code_map

@tvm._ffi.register_func("relay.ext.imcflow.constant_updater")
def imcflow_constant_updater(expr, symbol):
    """
    Relay function에서 constant를 추출하여 반환
    """
    return dict()

@tvm._ffi.register_func("relay.ext.imcflow")
def imcflow_external_codegen(func: relay.Function):
  # Obtain the function name (global symbol) assigned by the partitioning pass
  func_name = func.attrs["global_symbol"] if hasattr(
      func, "attrs") and "global_symbol" in func.attrs else "imcflow_subgraph"

  # const_vars = list(DevConfig().ImcflowFuncMap[func_name].const_name_map.values())

  # Reuse existing kernel code generator
  code = makeKernelStartCode(func_name, func, DevConfig.HOST_OS)

  # Wrap as a CSourceModule so TVM can compile/link it with the rest of the MLF
  # Note: returning a CSourceModule is the standard for BYOC Python codegen.
  return tvm.runtime._ffi_api.CSourceModuleCreate(code, "cc", [String(func_name)], None)