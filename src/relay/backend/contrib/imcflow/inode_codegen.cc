#include "inode_codegen.h"
#include <fstream>
#include <sstream>

namespace tvm {
namespace relay {
namespace contrib {

using json = nlohmann::json;

// Load the configuration from a JSON file
void InodeCodegen::LoadConfig(const std::string& json_path) {
  std::ifstream ifs(json_path);
  json jf = json::parse(ifs);

  ParseInodes(jf);

  LOG(INFO) << "Configuration loaded successfully.";
}

// Parse the inodes and their components
void InodeCodegen::ParseInodes(const json& jf) {
  if (!jf.contains("inodes")) {
    LOG(FATAL) << "No 'inodes' key found in the JSON configuration.";
    return;
  }

  for (const auto& inode : jf["inodes"]) {
    std::map<std::string, int> inode_metadata;

    // Parse metadata
    if (inode.contains("metadata")) {
      for (const auto& [key, value] : inode["metadata"].items()) {
        inode_metadata[key] = value.get<int>();
      }
      LOG(INFO) << "Metadata parsed successfully.";
    }

    // Parse memory layout
    if (inode.contains("memory_layout")) {
      ParseMemoryLayout(inode["memory_layout"]);
      LOG(INFO) << "Memory layout parsed successfully.";
    }

    // Parse code blocks
    if (inode.contains("code_blocks")) {
      ParseCodeBlocks(inode["code_blocks"]);
      LOG(INFO) << "Code blocks parsed successfully.";
    }

    inodes_.push_back(inode_metadata);
  }
}

// Parse the memory layout of an inode
void InodeCodegen::ParseMemoryLayout(const nlohmann::json& layout) {
  for (const auto& region : layout) {
    std::string region_name = region["region"];
    memory_blocks_["region"] = region_name;

    if (region.contains("blocks")) {
      for (const auto& block : region["blocks"]) {
        std::string block_name = block["name"];
        memory_blocks_["block_name"] = block_name;
      }
    } else {
      LOG(WARNING) << "No 'blocks' found in memory layout region.";
    }
  }
}

// Parse the code blocks of an inode
void InodeCodegen::ParseCodeBlocks(const nlohmann::json& blocks) {
  for (const auto& block : blocks) {
    std::string block_type = block["type"];
    LOG(INFO) << "Processing block type: " << block_type;

    if (block.contains("entries")) {
      for (const auto& entry : block["entries"]) {
        ParseEntry(entry);
      }
    } else {
      LOG(WARNING) << "No 'entries' found for block type: " << block_type;
    }
  }
}

// Parse an entry of a code block
void InodeCodegen::ParseEntry(const nlohmann::json& entry) {
  int entry_id = entry.value("id", -1);
  LOG(INFO) << "Processing entry id: " << entry_id;
  std::string data_block_name = entry.value("data_block", "");
  int policy_addr = entry.value("policy_addr", -1);
  int fifo_id = entry.value("fifo_id", -1);
}

// Generate the full code for an inode
std::string InodeCodegen::GenerateCode(const std::map<std::string, int>& inode_metadata) {
  std::ostringstream code;
  code << "#include <builtin.h>\n\n";
  code << "int main() {\n";

  // Generate code based on metadata and parsed blocks
  for (const auto& [key, value] : inode_metadata) {
    code << "  // Metadata: " << key << " = " << value << "\n";
  }

  for (const auto& [block_name, block_value] : memory_blocks_) {
    code << "  // Memory Block: " << block_name << " = " << block_value << "\n";
  }

  code << "  __builtin_INODE_SET_FLAG(1);\n";
  code << "  __builtin_INODE_HALT();\n";
  code << "  return 0;\n";
  code << "}\n";

  return code.str();
}

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
