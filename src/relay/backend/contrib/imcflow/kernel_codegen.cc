#include "kernel_codegen.h"

#include <tvm/runtime/logging.h>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace tvm {
namespace relay {
namespace contrib {

void KernelCodegen::HandleCodeGeneration(const std::string& func_name,
                                         const std::vector<std::string>& args) {
  LOG(INFO) << "Generating kernel code for func: " << func_name;
  std::string header_code = GenerateHeader(func_name, args);
  std::string source_code = GenerateSource(func_name, args);
  LOG(INFO) << "header_code: " << header_code << std::endl;
  LOG(INFO) << "source_code: " << source_code << std::endl;

  std::string header_file = output_dir_ + "/" + header_file_;
  InitializeFile(header_file, header_init_);
  InsertCodeToFile(header_file, header_code, header_lines_from_end_);

  std::string source_file = output_dir_ + "/" + source_file_;
  InitializeFile(source_file, source_init_);
  InsertCodeToFile(source_file, source_code, source_lines_from_end_);
}

void KernelCodegen::InitializeFile(const std::string& file_name, const std::string& file_content) {
  // check if source_file_ exists and if not create it
  if (!std::filesystem::exists(file_name)) {
    std::ofstream file(file_name, std::ios::app);
    file << file_content;
    file.close();
  }
}

std::string KernelCodegen::GenerateHeader(const std::string& func_name,
                                          const std::vector<std::string>& args) const {
  std::ostringstream code_stream;
  code_stream << "extern \"C\" TVM_DLL void " << func_name << "_kernel(";
  code_stream << "float* data, float* out";
  code_stream << ");";

  return code_stream.str();
}

std::string KernelCodegen::GenerateSource(const std::string& func_name,
                                          const std::vector<std::string>& args) const {
  std::ostringstream code_stream;
  code_stream << "extern \"C\" TVM_DLL void " << func_name << "_kernel(";
  code_stream << "float* data, float* out";
  code_stream << ") {\n";
  code_stream << "  out = data;\n";
  // code_stream << "  // TODO: Implement kernel code\n";
  code_stream << "}";

  return code_stream.str();
}

void KernelCodegen::InsertCodeToFile(const std::string& file_name, const std::string& code,
                                     size_t lines_from_end) {
  std::fstream file(file_name, std::ios::in | std::ios::out);
  if (!file.is_open()) {
    throw std::ios_base::failure("Failed to open file: " + file_name);
  }

  // Locate the insertion point
  std::streampos insertion_point = FindInsertionPoint(file, lines_from_end);

  // Read remaining content from the insertion point
  std::string remaining_content = ReadRemainingContent(file, insertion_point);

  // Write new content and append the remaining content
  WriteContent(file, insertion_point, code, remaining_content);

  file.close();
}

std::streampos KernelCodegen::FindInsertionPoint(std::fstream& file, size_t lines_from_end) const {
  file.seekg(0, std::ios::end);
  std::streamoff file_size = file.tellg();

  size_t lines_counted = 0;
  std::vector<std::streampos> line_positions;

  for (std::streamoff pos = file_size - 1; pos >= 0; --pos) {
    file.seekg(pos);
    char ch;
    file.get(ch);

    if (ch == '\n' || pos == 0) {
      line_positions.push_back(pos + 1);
      ++lines_counted;
      if (lines_counted > lines_from_end) {
        break;
      }
    }
  }

  return line_positions[lines_from_end];
}

// Reads the remaining content from the insertion point to the end of the file
std::string KernelCodegen::ReadRemainingContent(std::fstream& file,
                                                std::streampos insertion_point) const {
  file.seekg(insertion_point, std::ios::beg);
  return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

// Writes the content up to the insertion point, inserts new content, and appends the remaining
// content
void KernelCodegen::WriteContent(std::fstream& file, std::streampos insertion_point,
                                 const std::string& file_content,
                                 const std::string& remaining_content) const {
  file.seekp(insertion_point, std::ios::beg);
  file << file_content << '\n';  // Insert new content
  file << remaining_content;     // Append the rest of the file
}

}  // namespace contrib
}  // namespace relay
}  // namespace tvm