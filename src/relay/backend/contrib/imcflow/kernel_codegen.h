#ifndef TVM_RELAY_CONTRIB_KERNEL_CODEGEN_H_
#define TVM_RELAY_CONTRIB_KERNEL_CODEGEN_H_

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "inode_codegen.h"
#include "kernel_code_templates.h"

namespace tvm {
namespace relay {
namespace contrib {

/*!
 * \brief A class to handle kernel code generation.
 */
class KernelCodegen {
 public:
  /*!
   * \brief Constructor to initialize KernelCodegen with an output directory.
   * \param output_dir The directory where generated code are stored.
   */
  explicit KernelCodegen(const std::string& output_dir = "/tmp") : output_dir_(output_dir) {};

  /*!
   * \brief Generate, and append kernel code to kernel header and body.
   * \param op_name The name of the kernel.
   * \param args The arguments for the function.
   * \return The file path of the compiled shared library.
   */
  void HandleCodeGeneration(const std::string& func_name, const std::vector<std::string>& args);

 private:
  /*!
   * \brief Generate skeleton for file if it doesn't exist.
   */
  void InitializeFile(const std::string& file_name, const std::string& file_content);

  /*!
   * \brief Generate kernel code (Header and Source) for a specific function.
   * \param func_name The name of the function
   * \param args The arguments for the function.
   * \return The generated code as a string.
   */
  std::string GenerateHeader(const std::string& func_name,
                             const std::vector<std::string>& args) const;
  std::string GenerateSource(const std::string& func_name,
                             const std::vector<std::string>& args) const;

  /*!
   * \brief Insert generated code to file
   * \param file_name The file name to insert code into.
   * \param code The generated code as a string.
   */
  void InsertCodeToFile(const std::string& file_name, const std::string& code,
                        size_t lines_from_end);

  /*!
   * \brief find the insertion point for appending code
   */
  std::streampos FindInsertionPoint(std::fstream& file, size_t lines_from_end) const;

  /*!
   * \brief Read the remaining content from the insertion point to the end of the file.
   */
  std::string ReadRemainingContent(std::fstream& file, std::streampos insertion_point) const;

  /*!
   * \brief Write the content up to the insertion point, insert new content, and append the
   * remaining content.
   */
  void WriteContent(std::fstream& file, std::streampos insertion_point,
                    const std::string& file_content, const std::string& remaining_content) const;

  std::string output_dir_; /*!< The directory for generated code and binaries. */
  const std::string header_file_ = "imcflow_kernel.h";  /*!< The header file for kernel code. */
  const std::string source_file_ = "imcflow_kernel.cc"; /*!< The src file for kernel code. */
  const int header_lines_from_end_ = 6; /*!< The number of lines from the end of the header file. */
  const int source_lines_from_end_ = 4; /*!< The number of lines from the end of the source file. */
};

}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_CONTRIB_KERNEL_CODEGEN_H_
