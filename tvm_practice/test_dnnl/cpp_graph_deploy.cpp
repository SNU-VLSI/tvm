#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cstdio>

void DeployGraphExecutor() {
  LOG(INFO) << "Running graph executor...";
  // load in the library
  DLDevice dev{kDLCPU, 0};
  tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile("lib.so");
  // create the graph executor module
  tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
  tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
  tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
  tvm::runtime::PackedFunc run = gmod.GetFunction("run");

  // Use the C++ API
  tvm::runtime::NDArray x1 = tvm::runtime::NDArray::Empty({1, 32, 56, 56}, DLDataType{kDLFloat, 32, 1}, dev);
  tvm::runtime::NDArray x2 = tvm::runtime::NDArray::Empty({1, 32, 56, 56}, DLDataType{kDLFloat, 32, 1}, dev);
  tvm::runtime::NDArray y = tvm::runtime::NDArray::Empty({1, 32, 56, 56}, DLDataType{kDLFloat, 32, 1}, dev);

  for(int i = 0; i < 32; ++i) {
    for(int j = 0; j < 56; ++j) {
      for(int k = 0; k < 56; ++k) {
        static_cast<float*>(x1->data)[i * 56 * 56 + j * 56 + k] = i * 56 * 56 + j * 56 + k;
        static_cast<float*>(x2->data)[i * 56 * 56 + j * 56 + k] = i * 56 * 56 + j * 56 + k;
      }
    }
  }

  // set the right input
  set_input("x1", x1);
  set_input("x2", x2);

  // run the code
  run();

  // get the output
  get_output(0, y);

  for(int i = 0; i < 32; ++i) {
    for(int j = 0; j < 56; ++j) {
      for(int k = 0; k < 56; ++k) {
        printf("y[%d][%d][%d] = %f\n", i, j, k, static_cast<float*>(y->data)[i * 56 * 56 + j * 56 + k]);
      }
    }
  }
}

int main(void) {
  DeployGraphExecutor();
  return 0;
}
