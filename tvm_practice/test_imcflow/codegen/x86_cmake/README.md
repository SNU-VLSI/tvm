Build TVM CRT + model for x86_64 from an MLF tar.

Prerequisites:
- Generate MLF with build script in parent dir (Graph Executor, system-lib)
- Standard x86_64 toolchain (gcc/g++)

Configure & build:
  mkdir -p build && cd build
  cmake .. -C ../cmake/config_host.cmake
  cmake --build . -j

Outputs:
- Static libraries: libtvm_model.a, libtvm_x86.a and CRT component libs in build tree.
- Executable: tvm_host_runner (if TVM_BUILD_HOST_RUNNER is ON)