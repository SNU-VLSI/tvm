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
  InodeCodegen(const std::string& json_path);

  /*!
   * \brief Generates the full code for a given function for all inodes.
   */
  std::string GenerateCode(const std::string& op_name);

 private:
  /*!
   * \brief Generates the code for a given inode's hid/wid. Also parses the data block layout.
   */
  std::string GenerateNodeCode(const json& inode);

  /*!
   * \brief Parses the memory layout for a given inode.
   */
  void ParseMemoryLayout(const json& memory_layout);

  /*!
   * \brief Parses the data block layout for a given data block.
   */
  void ParseDataBlockLayout(const json& data_block);

  /*!
   * \brief Validates the block offset and size.
   */
  void ValidateBlock(int offset, int size);



  /*!
   * \brief Generates the code for a given block of code.
   * \param code_block The json containing block entry to generate.
   * \return A string containing the partial C++ code.
   */
  std::string GenerateBlockCode(const json& code_block);

  /*!
   * \brief GeneratesBlockCode implementation for a given code block and data block layout.
   */
  std::string GeneratePolicyUpdateCode(const json& code_block);
  std::string GenerateWriteIMCUCode(const json& code_block);
  std::string GenerateWriteIMEMCode(const json& code_block);
  std::string GenerateRecvCode(const json& code_block);
  std::string GenerateSendCode(const json& code_block);
  std::string GenerateCtrlCode(const json& code_block);

  // json object containing the parsed JSON file
  json jf_;

  // map containing the block offset and size for each data block
  std::map<std::string, std::pair<int, int>> dbl_;

};

}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_CONTRIB_INODE_CODEGEN_H_
