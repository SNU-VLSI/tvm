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

void DeviceCodegen::HandleCodeGeneration(const std::string& op_name,
                                         const std::vector<std::string>& args) {
  LOG(INFO) << "Generating inode code for operator: " << op_name;
  std::string inode_code = GenerateTargetCode(op_name, args, "inode");
  std::string cpp_name = SaveTargetCodeToFile(inode_code, op_name, "inode");
  CompileTargetCode(cpp_name, "inode");
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

std::string DeviceCodegen::SaveTargetCodeToFile(const std::string& code, const std::string& op_name,
                                                const std::string& target) {
  std::string cpp_name = output_dir_ + "/" + op_name + "_" + target + ".cpp";
  std::ofstream file(cpp_name);
  ICHECK(file.is_open()) << "Failed to open file for writing: " << cpp_name;
  file << code;
  file.close();
  return cpp_name;
}

void DeviceCodegen::CompileTargetCode(const std::string& cpp_name, const std::string& target) {
  if (cpp_name.size() < 4 || cpp_name.substr(cpp_name.size() - 4) != ".cpp") {
    LOG(FATAL) << "Invalid cpp_name: " << cpp_name;
  }

  std::string base_name = cpp_name.substr(0, cpp_name.size() - 4);
  std::vector<int> hids = {0, 1, 2, 3};
  std::vector<int> wids;

  if (target == "inode") {
    wids = {0};
  } else if (target == "imce") {
    wids = {1, 2, 3, 4};
  } else {
    LOG(FATAL) << "Unknown target: " << target;
  }

  for (auto hid : hids) {
    for (auto wid : wids) {
      std::string hid_str = std::to_string(hid);
      std::string wid_str = std::to_string(wid);
      std::string obj_file = base_name + "_" + hid_str + ".o";
      std::string out_file = base_name + "_" + hid_str + ".out";
      std::string bin_file = base_name + "_" + hid_str + ".bin";
      std::string host_obj_file = base_name + "_" + hid_str + ".host.o";

      // Compile into object file
      CompileCppToObject(cpp_name, obj_file, hid_str, wid_str);

      // Link object file into an output binary
      LinkObjectToBinary(obj_file, out_file);

      // Extract the .text section into a binary file
      ExtractTextSection(out_file, bin_file);

      // Create a host-compatible object file including the binary
      CreateHostObject(bin_file, host_obj_file);
    }
  }
}

void DeviceCodegen::CompileCppToObject(const std::string& cpp_name, const std::string& obj_file,
                                       const std::string& hid, const std::string& wid) {
  std::string command = "clang " + compile_options_ + " -mllvm=\"-INODE_hid=" + hid + "\" " +
                        "-mllvm=\"-INODE_wid=" + wid + "\" -o " + obj_file + " " + cpp_name;
  int ret = system(command.c_str());
  ICHECK_EQ(ret, 0) << "clang failed for " << cpp_name << " (hid=" << hid << ")";
}

void DeviceCodegen::LinkObjectToBinary(const std::string& obj_file, const std::string& out_file) {
  std::string command = "ld.lld " + lld_options_ + " -o " + out_file + " " + obj_file;
  int ret = system(command.c_str());
  ICHECK_EQ(ret, 0) << "ld.lld failed for " << obj_file;
}

void DeviceCodegen::ExtractTextSection(const std::string& out_file, const std::string& bin_file) {
  std::string command = "llvm-objcopy " + objcopy_options_ + " " + out_file + " " + bin_file;
  int ret = system(command.c_str());
  ICHECK_EQ(ret, 0) << "llvm-objcopy failed for " << out_file;
}

void DeviceCodegen::CreateHostObject(const std::string& bin_file,
                                     const std::string& host_obj_file) {
  std::string command = "ld " + ld_options_ + " -o " + host_obj_file + " " + bin_file;
  int ret = system(command.c_str());
  ICHECK_EQ(ret, 0) << "ld failed for " << bin_file;
}

}  // namespace contrib
}  // namespace relay
}  // namespace tvm