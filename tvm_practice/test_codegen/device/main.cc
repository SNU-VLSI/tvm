#include "device_codegen.h"
#include <iostream>

int main(int argc, char** argv) {
  const std::vector<std::string> args = {"arg1", "arg2"};
  tvm::relay::contrib::DeviceCodegen codegen("./");

  codegen.HandleDeviceCodeGeneration("relu", args);

  return 0;
}