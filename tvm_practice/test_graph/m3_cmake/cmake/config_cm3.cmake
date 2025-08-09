# CM3 initial cache (persistent)
# Usage:
#   cmake -S ${SOURCE_DIR} -B ${BINARY_DIR} \
#         -C ${SOURCE_DIR}/cmake/config_cm3.cmake \
#         -DCMAKE_TOOLCHAIN_FILE=/root/project/tvm/tvm_practice/test_imcflow/codegen/arm-cortex-m3.cmake

set(MLF_TAR "/root/project/tvm/tvm_practice/test_graph/lib_m3_graph_system-lib.tar" CACHE FILEPATH "Path to MLF tarball")
set(TVM_BUILD_HOST_RUNNER OFF CACHE BOOL "Build host tvm_m3_runner")
set(TVM_BUILD_M3_RUNNER ON CACHE BOOL "Build CM3 tvm_m3_runner")
set(CMAKE_TOOLCHAIN_FILE "/root/project/tvm/tvm_practice/test_imcflow/codegen/arm-cortex-m3.cmake" CACHE FILEPATH "Path to toolchain file")