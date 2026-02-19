# inputs
set(ISA "X86" CACHE STRING "Target ISA")
set(MAIN_TEST_FOLDER "one_relu_evl" CACHE STRING "Path to main script (not needed for x86)")
set(MAIN_SCRIPT "execute_graph.c" CACHE STRING "Path to main script")

set(TVM_BUILD_HOST_RUNNER ON CACHE BOOL "Build host runner")
# Paths are relative to the test directory (when host_binary_make is copied to {test}_evl/host_binary_make/)
set(MLF_TAR "${CMAKE_CURRENT_LIST_DIR}/../../lib_graph_system-lib.tar" CACHE FILEPATH "Path to MLF tarball")
# H_OBJ_PATH now points to the build directory, CMake will recursively find all .host.o files
set(H_OBJ_PATH "${CMAKE_CURRENT_LIST_DIR}/../../build" CACHE PATH "Path to host object files")

if(${ISA} STREQUAL "ARM")
  include("${CMAKE_CURRENT_LIST_DIR}/config_arm.cmake")
else()
  include("${CMAKE_CURRENT_LIST_DIR}/config_x86.cmake")
endif()
