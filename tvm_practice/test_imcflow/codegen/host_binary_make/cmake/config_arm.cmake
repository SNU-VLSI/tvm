# ARM Build Configuration
# Usage:
#   cmake -S ${SOURCE_DIR} -B ${BINARY_DIR} -C ${SOURCE_DIR}/cmake/config_arm.cmake

set(ISA "ARM" CACHE STRING "Target ISA")
set(TVM_BUILD_HOST_RUNNER ON CACHE BOOL "Build host runner")
# set(TVM_BUILD_ARM_RUNNER ON CACHE BOOL "Build ARM runner")
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_DIR}/arm-cortex-a53.cmake" CACHE FILEPATH "Path to toolchain file")

set(MLF_TAR "${CMAKE_CURRENT_LIST_DIR}/../../one_relu_evl/lib_graph_system-lib.tar" CACHE FILEPATH "Path to MLF tarball")
set(H_OBJ_PATH "${CMAKE_CURRENT_LIST_DIR}/../../one_relu_evl/build/tvmgen_default_tvmgen_default_imcflow_main_0_round_imcflow_region1_main_0" CACHE PATH "Path to host object files")
set(MAIN_SCRIPT "relu.c" CACHE STRING "Path to main script (not needed for x86)")
