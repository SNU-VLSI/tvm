# inputs
set(ISA "X86" CACHE STRING "Target ISA")
set(MAIN_TEST_FOLDER "one_relu_evl" CACHE STRING "Path to main script (not needed for x86)")
set(MAIN_SCRIPT "relu.c" CACHE STRING "Path to main script (not needed for x86)")

set(TVM_BUILD_HOST_RUNNER ON CACHE BOOL "Build host runner")
set(MLF_TAR "${CMAKE_CURRENT_LIST_DIR}/../../${MAIN_TEST_FOLDER}/lib_graph_system-lib.tar" CACHE FILEPATH "Path to MLF tarball")
set(H_OBJ_PATH "${CMAKE_CURRENT_LIST_DIR}/../../${MAIN_TEST_FOLDER}/build/tvmgen_default_tvmgen_default_imcflow_main_0_round_imcflow_region1_main_0" CACHE PATH "Path to host object files")

if(${ISA} STREQUAL "ARM")
  include("${CMAKE_CURRENT_LIST_DIR}/config_arm.cmake")
else()
  include("${CMAKE_CURRENT_LIST_DIR}/config_x86.cmake")
endif()
