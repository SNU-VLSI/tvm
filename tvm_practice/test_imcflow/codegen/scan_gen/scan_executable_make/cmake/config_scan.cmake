# Configuration for scan register executable build
set(ISA "X86" CACHE STRING "Target ISA")

# Path to program_scan_reg build directory (contains kernel.cc and .host.o files)
set(SCAN_KERNEL_PATH "${CMAKE_CURRENT_LIST_DIR}/../../build/program_scan_reg" CACHE PATH "Path to program_scan_reg build directory")

if(${ISA} STREQUAL "ARM")
  include("${CMAKE_CURRENT_LIST_DIR}/config_arm.cmake")
else()
  include("${CMAKE_CURRENT_LIST_DIR}/config_x86.cmake")
endif()
