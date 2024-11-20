#ifndef TVM_RELAY_CONTRIB_INODE_CODEGEN_H_
#define TVM_RELAY_CONTRIB_INODE_CODEGEN_H_

#include <string>
#include <vector>
#include <map>
#include <tvm/runtime/logging.h>
#include <nlohmann/json.hpp>

namespace tvm {
namespace relay {
namespace contrib {

using json = nlohmann::json;

/*!
 * \brief A class to handle complex codegen for Inode.
 */
class InodeCodegen {
 public:
  /*!
   * \brief Constructor to initialize DeviceCodegen with an output directory.
   */
  InodeCodegen() = default;

  /*!
   * \brief Generates the full code for a given inode's operation.
   * \param inode_metadata Metadata for the current inode (hid, wid).
   * \return A string containing the generated C++ code.
   */
  std::string GenerateCode(const std::map<std::string, int>& inode_metadata);

  /*!
   * \brief Load configuration from a JSON file.
   * \param json_path Path to the JSON file.
   */
  void LoadConfig(const std::string& json_path);

 private:
  /*!
   * \brief Parse the inodes and their components.
   */
  void ParseInodes(const json& jf);

  /*!
   * \brief Parse the memory layout details.
   */
  void ParseMemoryLayout(const json& layout);

  /*!
   * \brief Parse the code block details.
   */
  void ParseCodeBlocks(const json& blocks);

  /*!
   * \brief Parse an entry of a code block.
   */
  void ParseEntry(const json& entry);

  // Configuration data
  std::vector<std::map<std::string, int>> inodes_;
  std::map<std::string, std::string> memory_blocks_;
};

}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_CONTRIB_INODE_CODEGEN_H_
