#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <tvm/runtime/logging.h>

namespace tvm {
namespace relay {
namespace contrib {

/*!
 * \brief A class to handle device code generation, file management, and compilation.
 */
class DeviceCodegen {
 public:
  /*!
   * \brief Constructor to initialize DeviceCodegen with an output directory.
   * \param output_dir The directory where generated code and binaries will be stored.
   */
  explicit DeviceCodegen(const std::string& output_dir = "/tmp") : output_dir_(output_dir) {}

  /*!
   * \brief Generate device code for a specific operator.
   * \param op_name The name of the operator (e.g., "nn.conv2d").
   * \param args The arguments for the operator.
   * \return The generated code as a string.
   */
  std::string GenerateImceCode(const std::string& op_name, const std::vector<std::string>& args) {
    std::ostringstream code_stream;
    code_stream << "#include <common_decl.h>\n";
    code_stream << "void " << op_name << "() {\n";
    code_stream << ") {\n";
    code_stream << "    // Generated code for " << op_name << "\n";
    code_stream << "    for (int i = 0; i < 16; i++) {\n";
    code_stream << "        short16 result = __builtin_IMCE_RECV(0);\n";
    code_stream << "        short16 relu_result = __builtin_IMCE_MAX(result, 0, 0);\n";
    code_stream << "        __builtin_IMCE_SEND(1, relu_result, 0, 0);\n";
    code_stream << "    }\n";
    code_stream << "}\n";
    return code_stream.str();
  }

  std::string GenerateInodeCode(const std::string& op_name, const std::vector<std::string>& args) {
    std::ostringstream code_stream;
    code_stream << "#include <common_decl.h>\n";
    code_stream << "void " << op_name << "() {\n";
    code_stream << ") {\n";
    code_stream << "    // Generated code for " << op_name << "\n";
    code_stream << "    for (int i = 0; i < 16; i++) {\n";
    code_stream << "        short16 result = __builtin_IMCE_RECV(0);\n";
    code_stream << "        short16 relu_result = __builtin_IMCE_MAX(result, 0, 0);\n";
    code_stream << "        __builtin_IMCE_SEND(0, relu_result, 0, 0);\n";
    code_stream << "    }\n";
    code_stream << "}\n";
    return code_stream.str();
  }

  /*!
   * \brief Save generated code to a file.
   * \param code The generated code as a string.
   * \param op_name The name of the operator or function for naming the file.
   * \return The file path where the code was saved.
   */
  std::string SaveCodeToFile(const std::string& code, const std::string& op_name) {
    std::string file_name = output_dir_ + "/" + op_name + ".cpp";
    std::ofstream file(file_name);
    ICHECK(file.is_open()) << "Failed to open file for writing: " << file_name;
    file << code;
    file.close();
    return file_name;
  }

  /*!
   * \brief Compile the device code using clang.
   * \param file_name The file path of the code to compile.
   * \return The file path of the compiled shared library.
   */
  std::string CompileDeviceCode(const std::string& file_name) {
    std::string shared_library = file_name + ".so";
    std::string command = "clang " + compile_options_ + " -o " + shared_library + " " + file_name;
    int ret = system(command.c_str());
    ICHECK_EQ(ret, 0) << "Compilation failed for " << file_name;
    return shared_library;
  }

  /*!
   * \brief Generate, save, and compile device code for an operator.
   * \param op_name The name of the operator.
   * \param args The arguments for the operator.
   * \return The file path of the compiled shared library.
   */
  std::string HandleDeviceCodeGeneration(const std::string& op_name, const std::vector<std::string>& args) {
    LOG(INFO) << "Generating device code for operator: " << op_name;
    std::string device_code = GenerateImceCode(op_name, args);
    std::string file_name = SaveCodeToFile(device_code, op_name);
    return CompileDeviceCode(file_name);
  }

 private:
  std::string output_dir_;  /*!< The directory for generated code and binaries. */
  const std::string compile_options_ = "\
    -O1 --target=IMCE -mllvm=\"-force-hardware-loops\" -mllvm=\"-force-nested-hardware-loop\" -shared -fPIC\
    ";  /*!< The compile options for clang. */
};

}  // namespace contrib
}  // namespace relay
}  // namespace tvm