# A53 initial cache (persistent)
# Usage:
#   cmake -S ${SOURCE_DIR} -B ${BINARY_DIR} \
#         -C ${SOURCE_DIR}/cmake/config_a53.cmake

set(MLF_TAR "${CMAKE_CURRENT_LIST_DIR}/../../small_ref/lib_graph_system-lib.tar" CACHE FILEPATH "Path to MLF tarball")
set(TVM_BUILD_HOST_RUNNER OFF CACHE BOOL "Build host runner")
set(TVM_BUILD_ARM_RUNNER ON CACHE BOOL "Build ARM runner")
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_DIR}/arm-cortex-a53.cmake" CACHE FILEPATH "Path to toolchain file")