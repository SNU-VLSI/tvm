#ifndef TVM_RELAY_CONTRIB_DEVICE_CODEGEN_H_
#define TVM_RELAY_CONTRIB_DEVICE_CODEGEN_H_

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

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
   * \brief Generate, save, and compile target code for an operator.
   * \param op_name The name of the operator.
   * \param args The arguments for the operator.
   * \return The file path of the compiled shared library.
   */
  void HandleCodeGeneration(const std::string& op_name, const std::vector<std::string>& args);

 private:
  /*!
   * \brief Generate target device code for a specific operator.
   * \param op_name The name of the operator (e.g., "nn.conv2d").
   * \param args The arguments for the operator.
   * \return The generated code as a string.
   */
  std::string GenerateTargetCode(const std::string& op_name, const std::vector<std::string>& args,
                                 const std::string& target);

  /*!
   * \brief Save generated code to a file.
   * \param code The generated code as a string.
   * \param op_name The name of the operator or function for naming the file.
   * \return The file path where the code was saved.
   */
  std::string SaveTargetCodeToFile(const std::string& code, const std::string& op_name,
                                   const std::string& target);
  /*!
   * \brief Compile the target code using clang.
   * \param cpp_name The file path of the code to compile.
   * \return The file path of the compiled shared library.
   */
  void CompileTargetCode(const std::string& cpp_name, const std::string& target);
  void CompileCppToObject(const std::string& cpp_name, const std::string& obj_file,
                          const std::string& hid, const std::string& wid);
  void LinkObjectToBinary(const std::string& obj_file, const std::string& out_file);
  void ExtractTextSection(const std::string& out_file, const std::string& bin_file);
  void CreateHostObject(const std::string& bin_file, const std::string& host_obj_file);

  const int NUM_H_NODES = 4;
  const int NUM_W_NODES = 5;

  std::string output_dir_; /*!< The directory for generated code and binaries. */
  const std::string compile_options_ =
      "\
    -O1 --target=INODE -c -fPIC -mllvm=\"-force-hardware-loops\" -mllvm=\"-force-nested-hardware-loop\"\
    "; /*!< The options for clang. */
  // -O1 --target=IMCE -mllvm=\"-force-hardware-loops\" -mllvm=\"-force-nested-hardware-loop\"
  // -shared -fPIC
  const std::string objcopy_options_ =
      "\
    -O binary --only-section=.text\
    "; /*!< The options for llvm-objcopy. */
  const std::string lld_options_ =
      "\
    -e 0 -Ttext 0x0\
    "; /*!< The options for ld. */
  const std::string ld_options_ =
      "\
    -r -b binary\
    "; /*!< The options for ld. */
};

}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_CONTRIB_DEVICE_CODEGEN_H_
