# A53 initial cache (persistent)
# Usage:
#   cmake -S ${SOURCE_DIR} -B ${BINARY_DIR} \
#         -C ${SOURCE_DIR}/cmake/config_a53.cmake

set(MLF_TAR "${CMAKE_CURRENT_LIST_DIR}/../../one_relu_evl/lib_graph_system-lib.tar" CACHE FILEPATH "Path to MLF tarball")
set(TVM_BUILD_HOST_RUNNER ON CACHE BOOL "Build host runner")
set(TVM_BUILD_ARM_RUNNER OFF CACHE BOOL "Build ARM runner")
set(H_OBJ_PATH "${CMAKE_CURRENT_LIST_DIR}/../../one_relu_evl/build/tvmgen_default_tvmgen_default_imcflow_main_0_round_imcflow_region1_main_0" CACHE PATH "Path to Host object files")