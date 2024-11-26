#ifndef TVM_RELAY_CONTRIB_DEVICE_CODEGEN_H_
#define TVM_RELAY_CONTRIB_DEVICE_CODEGEN_H_

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <tvm/runtime/logging.h>
#include "../../utils.h"
#include "inode_codegen.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

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
  std::string HandleDeviceCodeGeneration(const std::string& op_name, const std::vector<std::string>& args);

 private:
  /*!
   * \brief Generate target device code for a specific operator.
   * \param op_name The name of the operator (e.g., "nn.conv2d").
   * \param args The arguments for the operator.
   * \return The generated code as a string.
   */
  std::string GenerateTargetCode(const std::string& op_name, const std::vector<std::string>& args, const std::string& target);

  /*!
   * \brief Save generated code to a file.
   * \param code The generated code as a string.
   * \param op_name The name of the operator or function for naming the file.
   * \return The file path where the code was saved.
   */
  std::string SaveCodeToFile(const std::string& code, const std::string& op_name, const std::string& target);
  /*!
   * \brief Compile the target code using clang.
   * \param file_name The file path of the code to compile.
   * \return The file path of the compiled shared library.
   */
  std::string CompileDeviceCode(const std::string& file_name);

  std::string output_dir_;  /*!< The directory for generated code and binaries. */
  const std::string compile_options_ = "\
    -O1 --target=INODE -c -mllvm=\"-force-hardware-loops\" -mllvm=\"-force-nested-hardware-loop\"\
    ";  /*!< The compile options for clang. */
    // -O1 --target=IMCE -mllvm=\"-force-hardware-loops\" -mllvm=\"-force-nested-hardware-loop\" -shared -fPIC
};

}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif // TVM_RELAY_CONTRIB_DEVICE_CODEGEN_H_