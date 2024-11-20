#include "inode_codegen.h"
#include <iostream>

int main(int argc, char** argv) {
  // Check if JSON file is provided as an argument
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <path to JSON file>\n";
    return 1;
  }

  std::string json_path = argv[1];
  tvm::relay::contrib::InodeCodegen codegen;

  codegen.LoadConfig(json_path);

  std::map<std::string, int> metadata = {{"hid", 0}, {"wid", 0}};
  std::string test_code = codegen.GenerateCode(metadata);
  std::cout << test_code;

  return 0;
}