import tvm
from tvm import relay
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from tvm.contrib.imcflow import DataBlock
from tvm.relay.ty import TensorType
from . import transform as imcflow_transform
from tvm.contrib.imcflow import ImcflowDeviceConfig, TensorID, DataBlock
from tvm.relay.op.contrib.imcflow import CustomIDToNode
from tvm.relay.expr import (Var, Constant)

import re
import math

def getInstructionBlocks(func_name, func):
    instruction_blocks = []

    for key, memory_region in ImcflowDeviceConfig().MemLayout.regions.items():
      if re.match(r"inode_\d+_inst", key):
        for block_name, block in memory_region.blocks.items():
          if "imem" in block_name:
            instruction_blocks.append(block)

    return instruction_blocks

def getDataBlocks(func_name, func):
    input_data_blocks = []
    output_data_blocks = []

    # get input/output node ID
    input_node_ids = [imcflow_transform.getNodeID(n) for n in imcflow_transform.getInputNodesOfFunc(func)]
    output_node_id = imcflow_transform.getNodeID(imcflow_transform.getOutputNodesOfFunc(func))

    # get input data blocks
    for key, memory_region in ImcflowDeviceConfig().MemLayout.regions.items():
      if re.match(r"inode_\d+_data", key):
        for block_name, block in memory_region.blocks.items():
          current_func_input_data = isinstance(block.id, TensorID) and any([input_node_id == imcflow_transform.getInnerNodeID(block_name.graph_node_id) for input_node_id in input_node_ids])
          current_func_inst = isinstance(block.id, str) and "imem" in block.id
          if current_func_input_data or current_func_inst:
            input_data_blocks.append(block)

    # get output data blocks
    #TODO : odata ??
    for key, memory_region in ImcflowDeviceConfig().MemLayout.regions.items():
      if re.match(r"inode_\d+_data", key):
        for block_name, block in memory_region.blocks.items():
          if isinstance(block_name, TensorID) and output_node_id == imcflow_transform.getInnerNodeID(block_name.graph_node_id):
            output_data_blocks.append(block)

    return input_data_blocks, output_data_blocks

def dtype_to_cpp(dtype: str) -> str:
    mapping = {
        "float32": "float",
        "float": "float",
        "int32": "int32_t",
        "int16" : "int16_t",
        "int8": "int8_t",
        "uint8": "uint8_t",
        "float64": "double",
    }

    # if dtype not in mapping: print(dtype)
    return mapping.get(dtype, "unknown_type")

class CodeWriter:
    def __init__(self, indent_str="  "):
        self.lines = []
        self.indent_str = indent_str
        self.indent_level = 0

    def getIndent(self):
      return self.indent_level

    def setIndent(self, indent_level):
      self.indent_level = indent_level

    def applyIndent(self, indent_level):
      for idx, line in enumerate(self.lines):
        line_ = indent_level * self.indent_str + line.lsstrip()
        self.lines[idx] = line_

    def nextIndent(self):
      self.indent_level += 1
      return self

    def prevIndent(self):
      self.indent_level -= 1
      return self

    def write(self, line=""):
        for line_ in line.split("\n"):
          if len(line_) > 0:
            self.lines.append(f"{self.indent_str * self.indent_level}{line_}")

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

def generateHeader():
  code = CodeWriter()
  code += "#include <stdint.h>\n"
  code += "#include <stdlib.h>\n"
  code += "#include <string.h>\n"
  code += "#include <tvm/runtime/c_runtime_api.h>\n"
  code += "#include <tvm/runtime/c_backend_api.h>\n"
  code += "#include <dlpack/dlpack.h>\n"
  code += "#include <sys/types.h>\n"
  code += "#include <sys/stat.h>\n"
  code += "#include <fcntl.h>\n"
  code += "#include <stdio.h>\n"
  code += "#include <sys/mman.h>\n"
  code += "#include <unistd.h>\n"
  return code

def generateInterruptRelatedCode():
  code = CodeWriter()
  # code += "volatile uint32_t npu_done = 0;\n"
  code += '#define WAIT_NPU_INTERRUPT \\\n'
  code += 'nb = read(npu_fd, &info, sizeof(info));\\\n'
  code += 'if (nb == (ssize_t)sizeof(info)) {\\\n'
  code += '  printf("Interrupt #%u!\\n", info);\\\n'
  code += '}\\\n'
  code += 'int_ack_gen_pointer[0] = 1;\\\n'
  code += 'npu_pointer[7] = 1;\n'
  code += '#define ENABLE_NPU_INTERRUPT \\\n'
  code += "info = 1; /* unmask interrupt pin*/\\\n"
  code += "nb = write(npu_fd, &info, sizeof(info));\\\n"
  code += "if (nb != (ssize_t)sizeof(info)) {\\\n"
  code += '  perror("interrupt unmasking failed");\\\n'
  code += "  close(npu_fd);\\\n"
  code += "  exit(EXIT_FAILURE);\\\n"
  code += "}\n"

  return code

def generateInstructionDef():
    code = CodeWriter()
    for key, memory_region in ImcflowDeviceConfig().MemLayout.regions.items():
      for block_name, block in memory_region.blocks.items():
        if isinstance(block.id, str) and "imem" in block.id:
          code += f"int32_t {block.id}[{math.ceil(block.size/4)}] = {{0,}};\n"

    return code

def generateFpgaPointerDef():
  code = CodeWriter()
  # code += "volatile uint32_t *NPU_BASE_ADDR = (uint32_t *)0x40000000;\n"
  code += "#define NPU_BASE_ADDR 0x40000000\n"
  code += "#define NPU_ADDR_RANGE 0x1000\n"
  code += "#define INT_ACK_GEN_ADDR 0xa0110000\n"
  code += "#define INT_ACK_GEN_LEN 0x10000\n"
  code += 'int npu_fd = open("/dev/uio5", O_RDWR);\n'
  code += "if (npu_fd < 0) {\n"
  code += '  perror("open");\n'
  code += "  exit(EXIT_FAILURE);\n"
  code += "}\n"
  code += "uint32_t info = 1; /* unmask interrupt pin*/\n"
  code += "ssize_t nb = write(npu_fd, &info, sizeof(info));\n"
  code += "if (nb != (ssize_t)sizeof(info)) {\n"
  code += '  perror("interrupt unmasking failed");\n'
  code += "  close(npu_fd);\n"
  code += "  exit(EXIT_FAILURE);\n"
  code += "}\n"
  code += "int int_ack_gen_fd = open(\"/dev/uio4\", O_RDWR);\n"
  code += "if (int_ack_gen_fd < 0) {\n"
  code += '  perror("open int_ack_gen_fd");\n'
  code += "  exit(EXIT_FAILURE);\n"
  code += "}\n"
  code += "size_t npu_len = (size_t) NPU_ADDR_RANGE;\n"
  code += "uint32_t *npu_pointer = (uint32_t *) mmap(NULL, npu_len, PROT_WRITE | PROT_READ, MAP_SHARED, npu_fd, 0);\n"
  code += "if (npu_pointer == MAP_FAILED) {\n"
  code += '  perror("npu_pointer mmap error");\n'
  code += "  exit(1);\n"
  code += "}\n"
  code += "size_t int_ack_gen_len = (size_t)INT_ACK_GEN_LEN;\n"
  code += "uint32_t *int_ack_gen_pointer = (uint32_t*) mmap(NULL, int_ack_gen_len, PROT_WRITE | PROT_READ, MAP_SHARED, int_ack_gen_fd, 0);\n"
  code += "if (int_ack_gen_pointer == MAP_FAILED) {\n"
  code += '  perror("int_ack_gen_pointer mmap error");\n'
  code += "  exit(1);\n"
  code += "}\n"

  return code

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
    graph_node_inner_id = imcflow_transform.getInnerNodeID(data_block.id.graph_node_id)
    if isinstance(node_map[graph_node_inner_id], Var):
      return node_map[graph_node_inner_id].name_hint
    elif isinstance(node_map[graph_node_inner_id], Constant):
      data_type = dtype_to_cpp(node_map[graph_node_inner_id].checked_type.dtype)
      return f"(({data_type}*)({func_name}_consts[{getConstantIdx(func, graph_node_inner_id)}]->data))"
    else:
      print(data_block)
      raise ValueError("Wrong data block type!")
  elif isinstance(data_block.id, str):
    return data_block.id
  else:
    print(data_block)
    raise ValueError("Wrong data block type!")

def generateInstructionTransferCode(func, func_name, instruction_blocks, address_macros):
    code = CodeWriter()
    code += "// Transfer instruction data into NPU memory\n"
    bitwidth_per_transfer = 32
    for block in instruction_blocks:
      size = block.size
      base_address = block.base_address
      base_address_name = makeBaseAddrName(block)
      address_macros.update({base_address_name: base_address})
      iteration_bound = math.ceil(size/bitwidth_per_transfer)
      code += f"for(int i=0; i<{iteration_bound}; i++){{\n"
      code += f"  *(npu_pointer + {base_address_name} + i * {bitwidth_per_transfer//8}) = {getCInputVarName(func, func_name, block)}[i];\n"
      code += f"}}\n"
    return code

def generateInputDataTransferCode(func, func_name, data_blocks, address_macros):
    code = CodeWriter()
    bitwidth_per_transfer = 32
    for block in data_blocks:
      size = block.size
      base_address = block.base_address
      base_address_name = makeBaseAddrName(block)
      address_macros.update({base_address_name : base_address})
      iteration_bound = math.ceil(size/bitwidth_per_transfer)
      code += f"for(int i=0; i<{iteration_bound}; i++){{\n"
      code += f"  *(npu_pointer + {base_address_name} + i * {bitwidth_per_transfer//8}) = {getCInputVarName(func, func_name, block)}[i];\n"
      code += f"}}\n"
    return code

def generateOutputDataTransferCode(data_blocks, address_macros):
    code = CodeWriter()
    bitwidth_per_transfer = 32
    for idx, block in enumerate(data_blocks):
      size = block.size
      base_address = block.base_address
      base_address_name = makeBaseAddrName(block)
      address_macros.update({base_address_name : base_address})
      iteration_bound = math.ceil(size/bitwidth_per_transfer)
      code += f"for(int i=0; i<{iteration_bound}; i++){{\n"
      code += f"  out{idx}[i] = *(npu_pointer + {base_address_name} + i * {bitwidth_per_transfer//8});\n"
      code += f"}}\n"
    return code

def generateWaitForInterruptCode():
    code =  (
              "WAIT_NPU_INTERRUPT;\n"
            )
              # "while(1) {\n"
              # "  if(npu_done == 1) {\n"
              # "    npu_done = 0;\n"
              # "    break;\n"
              # "  }\n"
              # "  std::this_thread::sleep_for(std::chrono::milliseconds(1));\n"
              # "}\n"
    return code

def generateEnableForInterruptCode():
    code =  (
              "ENABLE_NPU_INTERRUPT;\n"
            )
    return code

def generateInvokeCode():
    IDLE_CODE = 0
    RUN_CODE = 1
    PROGRAM_CODE = 2

    INODE_PC_START_P1_ENUM_VAL = 0
    INODE_PC_START_EXTERN_ENUM_VAL = 1
    INODE_PC_START_P0_ENUM_VAL = 2

    INODE_NUM = ImcflowDeviceConfig().INODE_NUM

    STATE_REG_IDX = 0
    PC_REG_IDX = 2

    code = (
      "// Invoke with policy update mode\n"
      f"for(int i=0; i<{INODE_NUM}; i++) {{\n"
      f"  *(npu_pointer + ({PC_REG_IDX} + i)*4) = ({INODE_PC_START_EXTERN_ENUM_VAL} << 30 + 0);\n"
      "}\n"
      f"{generateEnableForInterruptCode()}\n"
      f"*(npu_pointer + {STATE_REG_IDX}) = {PROGRAM_CODE};\n"
      f"{generateWaitForInterruptCode()}\n"
      "// Invoke with compute mode\n"
      f"for(int i=0; i<{INODE_NUM}; i++) {{\n"
      f"  *(npu_pointer + ({PC_REG_IDX} + i)*4) = ({INODE_PC_START_P1_ENUM_VAL} << 30 + 0);\n"
      "}\n"
      f"{generateEnableForInterruptCode()}\n"
      f"*(npu_pointer + {STATE_REG_IDX}) = {RUN_CODE};\n"
      f"{generateWaitForInterruptCode()}\n"
    )
    return code

def generateBaseAddrMacros(base_address_macros):
  code = CodeWriter()
  for key, value in base_address_macros.items():
    code += f"#define {key} {value}\n"
  return code

def generateLoadBinaryCode(func_name):
  code = CodeWriter()
  code += "// Load binary files into buffers\n"
  for key, memory_region in ImcflowDeviceConfig().MemLayout.regions.items():
    for block_name, block in memory_region.blocks.items():
      if isinstance(block.id, str) and "imem" in block.id:
        node_name, node_number = block.id.split("_")[0], block.id.split("_")[1]
        
        filename = f"./build/{func_name}_{node_name}_{node_name}_{node_number}.bin"

        size = math.ceil(block.size / 4)

        code += (
            f'{{\n'
            f'FILE *fp = fopen("{filename}", "rb");\n'
            f'if (!fp) {{ perror("fopen failed"); exit(1); }}\n'
            f'fread(&{block.id}, sizeof(int32_t), {size}, fp);\n'
            f'fclose(fp);\n'
            f'}}\n'
        )
  return code

def makeKernelDef(func_name, func, instruction_blocks, data_blocks):
    base_address_macros = {}
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

    code += generateInterruptRelatedCode()

    code += generateInstructionDef()

    # Kernel function prototype and definition (C)
    code += f"void {func_name}_kernel({args_proto_type});\n"
    code += f"void {func_name}_kernel({args_proto_type}) {{\n"
    code.nextIndent()
    code += generateFpgaPointerDef()
    # Todo. load binary files to buffer
    code += f'{generateLoadBinaryCode(func_name)}'
    code += f'{generateInstructionTransferCode(func, func_name, instruction_blocks, base_address_macros)}'
    code += f'{generateInputDataTransferCode(func, func_name, data_blocks[0], base_address_macros)}'
    code += f'{generateInvokeCode()}'
    code += f'{generateOutputDataTransferCode(data_blocks[1], base_address_macros)}'
    code.prevIndent()
    code += '}\n'

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
      args_list.append(f"({dtype_to_cpp(input_node_types[idx].dtype)}*)in{idx}->data")
    args_list.append(f"({dtype_to_cpp(output_node_type.dtype)}*)out0->data")
    code += f"{func_name}_kernel({', '.join(args_list)});\n"

    code += "(void)out_ret_value;\n"
    code += "if (out_ret_tcode) { *out_ret_tcode = kTVMArgInt; }\n"
    code += "return 0;\n"
    code.prevIndent()
    code += "}\n"

    # no explicit registration needed: system lib references the symbol directly

    code = generateBaseAddrMacros(base_address_macros) + code

    return code


def makeKernelStartCode(func_name, func):
    instruction_blocks = getInstructionBlocks(func_name, func)
    data_blocks = getDataBlocks(func_name, func)
    kernel_def = makeKernelDef(func_name, func, instruction_blocks, data_blocks)
    code = kernel_def

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

@tvm._ffi.register_func("relay.ext.imcflow")
def imcflow_external_codegen(func: relay.Function):
  # Obtain the function name (global symbol) assigned by the partitioning pass
  func_name = func.attrs["global_symbol"] if hasattr(func, "attrs") and "global_symbol" in func.attrs else "imcflow_subgraph"

  # Reuse existing kernel code generator
  code = makeKernelStartCode(func_name, func)

  # Wrap as a CSourceModule so TVM can compile/link it with the rest of the MLF
  # Note: returning a CSourceModule is the standard for BYOC Python codegen.
  return tvm.runtime._ffi_api.CSourceModuleCreate(code, "cc", [func_name], None)


@tvm._ffi.register_func("relay.ext.imcflow.constant_updater")
def imcflow_constant_updater(expr, symbol):  # pylint: disable=unused-argument
  # Keep ownership of constants inside the external module
  return dict()