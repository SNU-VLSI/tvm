import logging
from pathlib import Path


class KernelCodegen:
  def __init__(self, output_dir="/tmp"):
    self.output_dir = Path(output_dir)
    self.header_file = self.output_dir / "imcflow_kernel.h"
    self.source_file = self.output_dir / "imcflow_kernel.cc"
    self.header_lines_from_end = 6
    self.source_lines_from_end = 4
    self.header_init = """#ifndef TVM_RUNTIME_CONTRIB_IMCFLOW_IMCFLOW_KERNEL_H_
#define TVM_RUNTIME_CONTRIB_IMCFLOW_IMCFLOW_KERNEL_H_

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>

#include <vector>

namespace tvm {
namespace runtime {
namespace contrib {

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_IMCFLOW_IMCFLOW_KERNEL_H_
"""
    self.source_init = """#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "imcflow_kernel.h"

namespace tvm {
namespace runtime {
namespace contrib {

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
"""

  def handle_code_generation(self, func_name, args):
    logging.info(f"Generating kernel code for func: {func_name}")

    header_code = self.generate_header(func_name, args)
    source_code = self.generate_source(func_name, args)

    logging.info(f"Header code: {header_code}")
    logging.info(f"Source code: {source_code}")

    self.initialize_file(self.header_file, self.header_init)
    self.insert_code_to_file(
        self.header_file, header_code, self.header_lines_from_end)

    self.initialize_file(self.source_file, self.source_init)
    self.insert_code_to_file(
        self.source_file, source_code, self.source_lines_from_end)

  def initialize_file(self, file_path, file_content):
    if not file_path.exists():
      file_path.parent.mkdir(parents=True, exist_ok=True)
      with open(file_path, "w") as file:
        file.write(file_content)

  def generate_header(self, func_name, args):
    return f"extern \"C\" TVM_DLL void {func_name}_kernel(float* data, float* out);\n"

  def generate_source(self, func_name, args):
    return f"""extern \"C\" TVM_DLL void {func_name}_kernel(float* data, float* out) {{
    out = data;
}}
"""

  def insert_code_to_file(self, file_path, code, lines_from_end):
    with open(file_path, "r+") as file:
      lines = file.readlines()
      insertion_point = len(lines) - lines_from_end
      updated_lines = lines[:insertion_point] + \
          [code + "\n"] + lines[insertion_point:]
      file.seek(0)
      file.writelines(updated_lines)
      file.truncate()
