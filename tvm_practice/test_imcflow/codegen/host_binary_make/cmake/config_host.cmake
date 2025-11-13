# Host initial cache (persistent)
# Usage:
#   cmake -S ${SOURCE_DIR} -B ${BINARY_DIR} -C ${SOURCE_DIR}/cmake/config_host.cmake

set(MLF_TAR "/root/project/tvm/tvm_practice/test_graph/lib_graph_system-lib.tar" CACHE FILEPATH "Path to MLF tarball")
set(TVM_BUILD_HOST_RUNNER ON CACHE BOOL "Build host runner")
set(TVM_BUILD_ARM_RUNNER OFF CACHE BOOL "Build ARM runner")