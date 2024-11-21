#include "inode_codegen.h"

#include <fstream>
#include <sstream>

namespace tvm {
namespace relay {
namespace contrib {

using json = nlohmann::json;

// Constructor to initialize DeviceCodegen with an output directory.
InodeCodegen::InodeCodegen(const std::string& json_path) {
  std::ifstream file(json_path);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + json_path);
  }
  file >> jf_;
  dbl_ = {};
}

std::string IndentMultilineString(const std::string& input, int indent_level = 2,
                                  char indent_char = ' ') {
  std::istringstream input_stream(input);
  std::ostringstream output_stream;
  std::string line;
  std::string indent(indent_level, indent_char);

  while (std::getline(input_stream, line)) {
    output_stream << indent << line << '\n';
  }

  return output_stream.str();
}

// Generate the full code for an inode
std::string InodeCodegen::GenerateCode(const std::string& func_name) {
  std::ostringstream code;
  code << "int " << func_name << "() {\n";
  code << "  int hid = __builtin_INODE_GET_CORE_HID();\n\n";

  // Generate code for each inode
  for (const auto& inode : jf_["inodes"]) {
    LOG(INFO) << "Generating code for hid: " << inode["metadata"]["hid"];
    code << IndentMultilineString(GenerateNodeCode(inode));
  }
  code << "}\n";

  return code.str();
}

// Generate the code for a given inode and parse data block layout
std::string InodeCodegen::GenerateNodeCode(const json& inode) {
  std::ostringstream code;
  int hid = inode["metadata"]["hid"];
  code << "if (hid == " << hid << ") {\n";

  // Parse memory layout and populate dbl_
  LOG(INFO) << "Parsing memory layout for hid: " << hid;
  ParseMemoryLayout(inode["memory_layout"]);

  // Generate code for each code block
  LOG(INFO) << "Generating code for hid: " << inode["metadata"]["hid"];
  for (const auto& code_block : inode["code_blocks"]) {
    code << IndentMultilineString(GenerateBlockCode(code_block));
  }

  code << "}\n";

  return code.str();
}

// Parse memory layout and populate dbl_
void InodeCodegen::ParseMemoryLayout(const json& memory_layout) {
  for (const auto& region : memory_layout) {
    if (region["region"] == "Data") {
      ParseDataBlockLayout(region["blocks"]);
    }
  }
}

// Parse data blocks and populate dbl_
void InodeCodegen::ParseDataBlockLayout(const json& blocks) {
  for (const auto& block : blocks) {
    std::string name = block["name"];
    int offset = block["block_offset"];
    int size = block["block_size"];
    ValidateBlock(offset, size);
    dbl_[name] = std::make_pair(offset, size);
  }
}

// Validate block offset and size
void InodeCodegen::ValidateBlock(int offset, int size) {
  if (offset % 32 != 0 || size % 32 != 0) {
    LOG(FATAL) << "Block offset and size must be a multiple of 32 (bytes)";
  }
}

// Generate code for a given block
std::string InodeCodegen::GenerateBlockCode(const json& code_block) {
  std::ostringstream code;
  std::string type = code_block["type"];

  LOG(INFO) << "Generating code for block type: " << type;
  code << "/*generate: " << type << "*/\n";

  if (type == "AddPolicyUpdate") {
    code << GeneratePolicyUpdateCode(code_block);
  } else if (type == "AddWriteIMCU") {
    code << GenerateWriteIMCUCode(code_block);
  } else if (type == "AddWriteIMEM") {
    code << GenerateWriteIMEMCode(code_block);
  } else if (type == "AddRecv") {
    code << GenerateRecvCode(code_block);
  } else if (type == "AddSend") {
    code << GenerateSendCode(code_block);
  } else if (type == "AddCtrl") {
    code << GenerateCtrlCode(code_block);
  } else {
    LOG(FATAL) << "Unknown block type: " << type;
  }
  code << "/*endgenerate: " << type << "*/\n";
  code << "\n";

  return code.str();
}

// Generate code for policy update
std::string InodeCodegen::GeneratePolicyUpdateCode(const json& code_block) {
  std::ostringstream code;

  code << "int policy_table_start_address;\n";

  for (const auto& entry : code_block["entries"]) {
    std::string db_name = entry["data_block"];
    int address = dbl_[db_name].first;
    int size = dbl_[db_name].second;

    code << "\npolicy_table_start_address = " << address << ";\n";
    for (int i = 0; i < size; i += 32) {
      code << "__builtin_INODE_PU(policy_table_start_address, " << i << ", "
           << int(i / 32) << ", " << entry["col_id"] << ");\n";
    }
  }

  return code.str();
}

// Generate code for writing to IMCU
std::string InodeCodegen::GenerateWriteIMCUCode(const json& code_block) {
  std::ostringstream code;

  return code.str();
}

// Generate code for writing to IMEM
std::string InodeCodegen::GenerateWriteIMEMCode(const json& code_block) {
  std::ostringstream code;

  code << "int imem_size;\n";
  code << "int imem_start_address;\n";

  for (const auto& entry : code_block["entries"]) {
    std::string db_name = entry["data_block"];
    int address = dbl_[db_name].first;
    int size = dbl_[db_name].second;

    // TODO: we can unroll some of the loop
    code << "\nimem_start_address = " << address << ";\n";
    code << "for (int i = 0; i < imem_size; i += 32) {\n";
    code << "  __builtin_INODE_WR_IMEM(i, 0, " << entry["policy_addr"] << ");\n";
    code << "}\n";
  }

  return code.str();
}

// Generate code for receiving data
std::string InodeCodegen::GenerateRecvCode(const json& code_block) {
  std::ostringstream code;

  code << "int recv_start_address;\n";
  code << "int recv_size;\n";

  for (const auto& entry : code_block["entries"]) {
    std::string db_name = entry["data_block"];
    int address = dbl_[db_name].first;
    int size = dbl_[db_name].second;

    // TODO: we can unroll some of the loop
    // TODO: remove policy from RECV
    code << "\nrecv_start_address = " << address << ";\n";
    code << "for (int i = 0; i < recv_size; i += 32) {\n";
    code << "  __builtin_INODE_RECV(i, 0, 0," << entry["fifo_id"] <<");\n";
    code << "}\n";
  }

  return code.str();
}

// Generate code for sending data
std::string InodeCodegen::GenerateSendCode(const json& code_block) {
  std::ostringstream code;

  code << "int send_start_address;\n";
  code << "int send_size;\n";

  for (const auto& entry : code_block["entries"]) {
    std::string db_name = entry["data_block"];
    int address = dbl_[db_name].first;
    int size = dbl_[db_name].second;

    // TODO: we can unroll some of the loop
    code << "\nsend_start_address = " << address << ";\n";
    code << "for (int i = 0; i < send_size; i += 32) {\n";
    code << "  __builtin_INODE_SEND(i, 0, " << entry["policy_addr"] << "," << entry["fifo_id"] <<");\n";
    code << "}\n";
  }

  return code.str();
}

// Generate code for control block
std::string InodeCodegen::GenerateCtrlCode(const json& code_block) {
  std::ostringstream code;

  // TODO: what if there are multiple sync ids?
  for (const auto& entry: code_block["entries"]) {
    std::string op = entry["operation"];
    if (op == "setflag") {
      code << "__builtin_INODE_SET_FLAG(" << entry["flag_value"] << ");\n";
    } else if (op == "standby") {
      code << "__builtin_INODE_STANDBY(" << entry["target_id"] << ", " << entry["flag_value"] << ");\n";
    } else if (op == "halt") {
      code << "__builtin_INODE_HALT();\n";
    } else if (op == "interrupt") {
      code << "__builtin_INODE_INTERRUPT();\n";
    } else if (op == "done") {
      code << "__builtin_INODE_DONE();\n";
    } else {
      LOG(FATAL) << "Unknown control operation: " << op;
    }

  }

  return code.str();
}

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
