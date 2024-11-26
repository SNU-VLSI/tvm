#include "device_codegen.h"

#include <tvm/runtime/logging.h>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "inode_codegen.h"

namespace tvm {
namespace relay {
namespace contrib {

std::string DeviceCodegen::HandleDeviceCodeGeneration(const std::string& op_name,
                                                      const std::vector<std::string>& args) {
  LOG(INFO) << "Generating inode code for operator: " << op_name;
  std::string inode_code = GenerateTargetCode(op_name, args, "inode");
  std::string cpp_name = SaveCodeToFile(inode_code, op_name, "inode");
  return CompileDeviceCode(cpp_name);
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
  std::string cpp_name = output_dir_ + "/" + op_name + "_" + target + ".cpp";
  std::ofstream file(cpp_name);
  ICHECK(file.is_open()) << "Failed to open file for writing: " << cpp_name;
  file << code;
  file.close();
  return cpp_name;
}

std::string DeviceCodegen::CompileDeviceCode(const std::string& cpp_name) {
  // truncate .cpp from cpp_name
  std::string file_name_no_ext = cpp_name.substr(0, cpp_name.size() - 4);
  std::string lib_name = file_name_no_ext + ".o";
  std::string out_name = file_name_no_ext + ".out";
  std::string bin_name = file_name_no_ext + ".bin";
  std::string host_lib_name = file_name_no_ext + ".host.o";

  // clang: compile into *.o
  std::string command = "clang " + compile_options_ + " -o " + lib_name + " " + cpp_name;
  int ret = system(command.c_str());
  ICHECK_EQ(ret, 0) << "clang failed for " << cpp_name;

  // ld.lld: link and resolve relocation into *.out
  command = "ld.lld " + lld_options_ + " -o " + out_name + " " + lib_name;
  ret = system(command.c_str());
  ICHECK_EQ(ret, 0) << "ld.lld failed for " << lib_name;

  // llvm-objcopy: copy .text section into *.bin
  command = "llvm-objcopy " + objcopy_options_ + out_name + " " + bin_name;
  ret = system(command.c_str());
  ICHECK_EQ(ret, 0) << "objcopy failed for " << out_name;

  // ld: re-link the library for the host to include the binary *.host.o
  command = "ld " + ld_options_ + " -o " + host_lib_name + " " + bin_name;
  ret = system(command.c_str());
  ICHECK_EQ(ret, 0) << "ld failed for " << bin_name;

  return host_lib_name;
}

}  // namespace contrib
}  // namespace relay
}  // namespace tvm