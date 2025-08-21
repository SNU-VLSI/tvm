import tvm
from tvm import relay
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from tvm.contrib.imcflow import DataBlock
from tvm.relay.ty import TensorType
from . import transform as imcflow_transform
from tvm.contrib.imcflow import ImcflowDeviceConfig, TensorID, DataBlock
from tvm.relay.op.contrib.imcflow import CustomIDToNode
from tvm.relay.expr import (Var, Constant)
import math

def align_to_n_bytes(size, n_bytes):
  if (size % n_bytes) != 0:
    size = (size // n_bytes + 1) * n_bytes
  return size


def dtype_to_cpp(dtype: str) -> str:
  mapping = {
      "float32": "float",
      "float": "float",
      "int32": "int32_t",
      "int16": "int16_t",
      "int8": "int8_t",
      "uint8": "uint8_t",
      "float64": "double",
  }

  # if dtype not in mapping: print(dtype)
  return mapping.get(dtype, "unknown_type")


class CodeWriter:
  indent_level = 0

  def __init__(self, indent_str="  "):
    self.lines = []
    self.indent_str = indent_str

  def nextIndent(self):
    CodeWriter.indent_level += 1

  def prevIndent(self):
    CodeWriter.indent_level -= 1

  def write(self, line=""):
    for line_ in line.split("\n"):
      if len(line_) > 0:
        self.lines.append(
            f"{self.indent_str * CodeWriter.indent_level}{line_}")

  def get_code(self):
    return "\n".join(self.lines)

  def __str__(self):
    return self.get_code()

  def __add__(self, other):
    if isinstance(other, CodeWriter):
      self.lines.extend(other.lines)
      return self
    elif isinstance(other, str):
      self.write(other)
      return self

def makeBaseAddrName(block):
  if isinstance(block.id, TensorID):
    return f"{block.id.tensor_type.upper()}_{imcflow_transform.getInnerNodeID(block.id.graph_node_id)}_BASE_ADDR"
  elif isinstance(block.id, str):
    return f"{block.id.upper()}_BASE_ADDR"
  else:
    raise ValueError("Wrong data block type!")


def getConstantIdx(func, node_id):
  node_id_to_constant_id = {}

  class _Visitor(tvm.relay.ExprVisitor):
    def __init__(self):
      super().__init__()
      self.Cnt = 0

    def visit_constant(self, const):
      node_id_to_constant_id[imcflow_transform.getNodeID(const)] = self.Cnt
      self.Cnt = self.Cnt + 1
      super().visit_constant(const)

  _Visitor().visit(func)
  return node_id_to_constant_id[node_id]


def getCInputVarName(func, func_name, data_block):
  node_map = CustomIDToNode()

  if isinstance(data_block.id, TensorID):
    graph_node_inner_id = imcflow_transform.getInnerNodeID(
        data_block.id.graph_node_id)
    if isinstance(node_map[graph_node_inner_id], Var):
      return node_map[graph_node_inner_id].name_hint
    elif isinstance(node_map[graph_node_inner_id], Constant):
      data_type = dtype_to_cpp(
          node_map[graph_node_inner_id].checked_type.dtype)
      return f"(({data_type}*)({func_name}_consts[{getConstantIdx(func, graph_node_inner_id)}]->data))"
    else:
      print(data_block)
      raise ValueError("Wrong data block type!")
  elif isinstance(data_block.id, str):
    filename = f"_binary_{data_block.id}_bin_start"
    return filename
  else:
    print(data_block)
    raise ValueError("Wrong data block type!")


def generateToNpuTransferCode(func, func_name, blocks, address_macros):
  code = CodeWriter()
  code += "// Transfer data into NPU memory\n"
  for block in blocks:
    size = align_to_n_bytes(block.size, 32)  # 32bytes alignment
    base_address = block.base_address
    base_address_name = makeBaseAddrName(block)
    address_macros.update({base_address_name: base_address})
    numel = math.ceil(size/4)
    code += f"for(int i=0; i<{numel}; i++){{\n"
    code += f"  *(npu_pointer + ({base_address_name} / 4) + i) = {getCInputVarName(func, func_name, block)}[i];\n"
    code += f"}}\n"
  return code


def generateFromNpuTransferCode(data_blocks, address_macros):
  code = CodeWriter()
  code += "// Transfer data from NPU memory\n"
  for idx, block in enumerate(data_blocks):
    size = align_to_n_bytes(block.size, 32)  # 32bytes alignment
    base_address = block.base_address
    base_address_name = makeBaseAddrName(block)
    address_macros.update({base_address_name: base_address})
    numel = math.ceil(size/4)
    code += f"for(int i=0; i<{numel}; i++){{\n"
    code += f"  out{idx}[i] = *(npu_pointer + ({base_address_name} / 4) + i);\n"
    code += f"}}\n"
  return code


def generateBaseAddrMacros(base_address_macros):
  code = CodeWriter()
  for key, value in base_address_macros.items():
    code += f"#define {key} {value}\n"
  code += "\n"
  return code

def generateExternLink(func_name, compiled_blocks):
  code = CodeWriter()
  code += 'extern "C" { \n'
  for block in compiled_blocks:
    if isinstance(block.id, str):
      filename = f"_binary_{block.id}_bin"
      code += f'  extern const int32_t {filename}_start[];\n'
  code += '}\n'
  return code


def generatePackedFuncWrapper(func_name, input_node_types, output_node_type):
  code = CodeWriter()
  # PackedFunc wrapper for CRT
  code += "#ifdef __cplusplus\n"
  code += "extern \"C\"\n"
  code += "#endif\n"
  code += f"TVM_DLL int32_t {func_name}(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {{\n"
  code.nextIndent()
  code += "(void)resource_handle;\n"
  code += "if (num_args < 2) return -1;\n"

  # get input and output data pointers
  for idx in range(len(input_node_types)):
    code += f"void* _in{idx} = (((TVMValue*)args)[{idx}].v_handle);\n"
    code += f"DLTensor* in{idx} = (DLTensor*)_in{idx};\n"
  code += f"void* _out0 = (((TVMValue*)args)[{len(input_node_types)}].v_handle);\n"
  code += f"DLTensor* out0 = (DLTensor*)_out0;\n"

  # call kernel function
  args_list = []
  for idx in range(len(input_node_types)):
    args_list.append(
        f"({dtype_to_cpp(input_node_types[idx].dtype)}*)in{idx}->data")
  args_list.append(f"({dtype_to_cpp(output_node_type.dtype)}*)out0->data")
  code += f"{func_name}_kernel({', '.join(args_list)});\n"

  code += "(void)out_ret_value;\n"
  code += "if (out_ret_tcode) { *out_ret_tcode = kTVMArgInt; }\n"
  code += "return 0;\n"
  code.prevIndent()
  code += "}\n"
  return code


def makeKernelDef(func_name, func, compiled_blocks, data_blocks):
  base_address_macros = {
      "IMCFLOW_ADDR": 0xa0000000,
      "IMCFLOW_LEN": 0x100000,
      "INT_ACK_GEN_ADDR": 0xa0110000,
      "INT_ACK_GEN_LEN": 0x10000,
      "IMCFLOW_DEVICE": "\"/dev/uio5\"",
      "INT_ACK_GEN_DEVICE": "\"/dev/uio4\"",
      "SET_IDLE_CODE": 0,
      "SET_RUN_CODE": 1,
      "SET_PROGRAM_CODE": 2,
      "STATE_REG_IDX": 0,
      "PC_REG_IDX": 2,
      "INODE_PC_START_P1_ENUM_VAL": 0,
      "INODE_PC_START_EXTERN_ENUM_VAL": 1,
      "INODE_PC_START_P0_ENUM_VAL": 2,
      "INODE_NUM": ImcflowDeviceConfig().INODE_NUM,
  }
  proto_list = []
  for i, param in enumerate(func.params):
    param_name = param.name_hint if param.name_hint else f"arg{i}"
    dtype = "float32"
    if hasattr(param, "checked_type") and isinstance(param.checked_type, TensorType):
      dtype = param.checked_type.dtype
      cpp_type = dtype_to_cpp(dtype)
      proto_list.append(f"{cpp_type}* {param_name}")

  input_nodes = [n for n in imcflow_transform.getInputNodesOfFunc(func)]
  input_node_types = [n.checked_type for n in input_nodes]
  output_node = imcflow_transform.getOutputNodesOfFunc(func)
  output_node_type = output_node.checked_type
  proto_list.append(f"{dtype_to_cpp(output_node_type.dtype)}* out0")

  args_proto_type = ", ".join(proto_list)

  code = CodeWriter()
  code += generateHeader()
  code += generateExternLink(func_name, compiled_blocks)
  code += generateInterruptUtilities()

  # Kernel function prototype and definition (C)
  code += f"void {func_name}_kernel({args_proto_type}) {{\n"
  code.nextIndent()
  code += generateDevicePointerSetup()
  code += generateToNpuTransferCode(func, func_name,
                                    compiled_blocks, base_address_macros)
  code += generateToNpuTransferCode(func, func_name,
                                    data_blocks[0], base_address_macros)
  code += generateInvokeCode()
  code += generateFromNpuTransferCode(data_blocks[1], base_address_macros)
  code += generateDevicePointerCleanup()
  code.prevIndent()
  code += '}\n'

  code += generatePackedFuncWrapper(func_name,
                                    input_node_types, output_node_type)
  code = generateBaseAddrMacros(base_address_macros) + code

  return code


def makeKernelStartCode(func_name, func):
  compiled_blocks = ImcflowDeviceConfig().DataBlocks["compiled"]
  data_blocks = ImcflowDeviceConfig().DataBlocks["input"], ImcflowDeviceConfig().DataBlocks["output"]
  kernel_def = makeKernelDef(func_name, func, compiled_blocks, data_blocks)
  code = kernel_def

  return str(code)


def generateHeader():
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


def generateInterruptUtilities():
  return ("""
void enable_imcflow_interrupt(int fd)
{
  uint32_t info = 1;
  ssize_t nb = write(fd, &info, sizeof(info));
  if (nb != (ssize_t)sizeof(info)) {
    perror("write failed");
    close(fd);
    exit(1);
  }
}

void wait_imcflow_interrupt(int fd)
{
  uint32_t info;
  ssize_t nb = read(fd, &info, sizeof(info));
}

void generate_ack(uint32_t* int_ack_gen)
{
  int_ack_gen[0] = 0b1;
}
""")


def generateDevicePointerSetup():
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


def generateInvokeCode():
  return ("""
// Set the inode pc to 0 and run.
for(int i=0; i<INODE_NUM; i++) {
  *(npu_pointer + (PC_REG_IDX + i)) = (INODE_PC_START_EXTERN_ENUM_VAL << 30 + 0);
}
enable_imcflow_interrupt(npu_fd);
*(npu_pointer + STATE_REG_IDX) = SET_RUN_CODE;

wait_imcflow_interrupt(npu_fd);
generate_ack(int_ack_gen_pointer);
npu_pointer[7] = 1;
""")


def generateDevicePointerCleanup():
  return ("""
// Cleanup device pointer
munmap(npu_pointer, npu_len);
close(npu_fd);
munmap(int_ack_gen_pointer, int_ack_gen_len);
close(int_ack_gen_fd);
""")



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


@tvm._ffi.register_func("relay.ext.imcflow")
def imcflow_external_codegen(func: relay.Function):
  # Obtain the function name (global symbol) assigned by the partitioning pass
  func_name = func.attrs["global_symbol"] if hasattr(
      func, "attrs") and "global_symbol" in func.attrs else "imcflow_subgraph"

  # Reuse existing kernel code generator
  code = makeKernelStartCode(func_name, func)

  # Wrap as a CSourceModule so TVM can compile/link it with the rest of the MLF
  # Note: returning a CSourceModule is the standard for BYOC Python codegen.
  return tvm.runtime._ffi_api.CSourceModuleCreate(code, "cc", [func_name], None)


@tvm._ffi.register_func("relay.ext.imcflow.constant_updater")
def imcflow_constant_updater(expr, symbol):  # pylint: disable=unused-argument
  # Keep ownership of constants inside the external module
  return dict()
