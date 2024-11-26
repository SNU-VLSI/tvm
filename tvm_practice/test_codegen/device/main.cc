#include "device_codegen.h"
#include <iostream>

int main(int argc, char** argv) {
  const std::vector<std::string> args = {"arg1", "arg2"};
  tvm::relay::contrib::DeviceCodegen codegen("./");

  std::string so_name = codegen.HandleDeviceCodeGeneration("relu", args);
  std::cout << so_name << std::endl;

  return 0;
}