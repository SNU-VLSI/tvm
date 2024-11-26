#include "device_codegen.h"

#include <tvm/runtime/logging.h>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../../utils.h"
#include "inode_codegen.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

std::string DeviceCodegen::HandleDeviceCodeGeneration(const std::string& op_name,
                                                      const std::vector<std::string>& args) {
  LOG(INFO) << "Generating inode code for operator: " << op_name;
  std::string inode_code = GenerateTargetCode(op_name, args, "inode");
  std::string file_name = SaveCodeToFile(inode_code, op_name, "inode");
  return CompileDeviceCode(file_name);
}

std::string DeviceCodegen::GenerateTargetCode(const std::string& op_name,
                                              const std::vector<std::string>& args,
                                              const std::string& target) {
  std::ostringstream code_stream;
  if (target == "inode") {
    InodeCodegen inode_codegen("imcflow.json");
    code_stream << inode_codegen.GenerateCode(op_name);  // TODO: pass args?
  } else if (target == "imce") {
    LOG(FATAL) << "IMCE codegen not implemented yet.";
  } else {
    LOG(FATAL) << "Unknown target: " << target;
  }

  return code_stream.str();
}

std::string DeviceCodegen::SaveCodeToFile(const std::string& code, const std::string& op_name,
                                          const std::string& target) {
  std::string file_name = output_dir_ + "/" + op_name + "_" + target + ".cpp";
  std::ofstream file(file_name);
  ICHECK(file.is_open()) << "Failed to open file for writing: " << file_name;
  file << code;
  file.close();
  return file_name;
}

std::string DeviceCodegen::CompileDeviceCode(const std::string& file_name) {
  std::string shared_library = file_name + ".so";
  std::string command = "clang " + compile_options_ + " -o " + shared_library + " " + file_name;
  int ret = system(command.c_str());
  ICHECK_EQ(ret, 0) << "Compilation failed for " << file_name;
  return shared_library;
}

}  // namespace contrib
}  // namespace relay
}  // namespace tvm