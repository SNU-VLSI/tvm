cmake_minimum_required(VERSION 3.18)

# ============================================================================
# ISA Selection
# ============================================================================
set(ISA "X86" CACHE STRING "Target ISA: ARM or X86")
set_property(CACHE ISA PROPERTY STRINGS "ARM" "X86")

# Validate ISA
if(NOT ISA MATCHES "^(ARM|X86)$")
  message(FATAL_ERROR "Invalid ISA: ${ISA}. Must be ARM or X86")
endif()

message(STATUS "Building for ISA: ${ISA}")

# ============================================================================
# Project Setup
# ============================================================================
if(ISA STREQUAL "ARM")
  project(scan_reg_arm C CXX)
else()
  project(scan_reg_x86 C CXX)
endif()

set(TARGET_EXE_NAME program_scan_reg)

# Options
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# ============================================================================
# Input Validation
# ============================================================================
# Scan kernel source file
if(NOT DEFINED SCAN_KERNEL_PATH)
  set(SCAN_KERNEL_PATH "${CMAKE_CURRENT_LIST_DIR}/../build/program_scan_reg" CACHE PATH "Path to program_scan_reg build directory")
endif()

set(SCAN_KERNEL_SOURCE "${SCAN_KERNEL_PATH}/program_scan_reg_kernel.cc")
if(NOT EXISTS "${SCAN_KERNEL_SOURCE}")
  message(FATAL_ERROR "Scan kernel source not found: ${SCAN_KERNEL_SOURCE}\nGenerate it with scan_codegen.py first")
endif()

# Host object files (contains binary data for policy tables, IMEM, etc.)
file(GLOB_RECURSE HOST_OBJ_FILES "${SCAN_KERNEL_PATH}/*.host.o")

if(NOT HOST_OBJ_FILES)
  message(WARNING "No host object files (*.host.o) found in: ${SCAN_KERNEL_PATH}")
  message(WARNING "The executable may fail to link without these binary data files")
else()
  list(LENGTH HOST_OBJ_FILES HOST_OBJ_COUNT)
  message(STATUS "Found ${HOST_OBJ_COUNT} host object files:")
  foreach(obj_file ${HOST_OBJ_FILES})
    message(STATUS "  - ${obj_file}")
  endforeach()
endif()

# ============================================================================
# TVM Root Setup (for cnpy)
# ============================================================================
if(NOT DEFINED TVM_ROOT)
  set(TVM_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../../.." CACHE PATH "Path to TVM root")
endif()
if(NOT EXISTS "${TVM_ROOT}/3rdparty/cnpy")
  if(DEFINED ENV{TVM_ROOT})
    set(TVM_ROOT "$ENV{TVM_ROOT}" CACHE PATH "Path to TVM root" FORCE)
  elseif(DEFINED ENV{TVM_HOME})
    set(TVM_ROOT "$ENV{TVM_HOME}" CACHE PATH "Path to TVM root" FORCE)
  endif()
endif()

if(NOT EXISTS "${TVM_ROOT}/3rdparty/cnpy")
  message(FATAL_ERROR "TVM 3rdparty/cnpy not found at: ${TVM_ROOT}/3rdparty/cnpy\nSet -DTVM_ROOT=/path/to/tvm (or export TVM_ROOT/TVM_HOME)")
endif()

# ============================================================================
# Build cnpy Library (from TVM 3rdparty)
# ============================================================================
# Find static zlib library for static linking
if(ISA STREQUAL "ARM")
    find_library(ZLIB_STATIC_LIB
        NAMES libz.a
        PATHS /usr/lib/aarch64-linux-gnu
        NO_DEFAULT_PATH
    )
else()
    # X86
    find_library(ZLIB_STATIC_LIB
        NAMES libz.a
        PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/lib64
    )
endif()

if(ZLIB_STATIC_LIB)
    set(ZLIB_LIBRARIES ${ZLIB_STATIC_LIB})
    message(STATUS "Found static zlib: ${ZLIB_LIBRARIES}")
else()
    message(WARNING "Static zlib not found. Trying find_package...")
    find_package(ZLIB REQUIRED)
    set(ZLIB_LIBRARIES ${ZLIB_LIBRARIES})
endif()

set(CNPY_DIR "${TVM_ROOT}/3rdparty/cnpy")

# Build cnpy as a static library
add_library(cnpy STATIC "${CNPY_DIR}/cnpy.cpp")
target_include_directories(cnpy PUBLIC "${CNPY_DIR}")
target_link_libraries(cnpy PUBLIC ${ZLIB_LIBRARIES})
target_compile_features(cnpy PUBLIC cxx_std_11)

message(STATUS "Building cnpy from: ${CNPY_DIR}")
message(STATUS "Using zlib: ${ZLIB_LIBRARIES}")

# ============================================================================
# Build Scan Register Executable
# ============================================================================
add_executable(${TARGET_EXE_NAME} ${SCAN_KERNEL_SOURCE})

# Add host object files if they exist
if(HOST_OBJ_FILES)
  target_sources(${TARGET_EXE_NAME} PRIVATE ${HOST_OBJ_FILES})
  message(STATUS "Linking ${HOST_OBJ_COUNT} host object files to ${TARGET_EXE_NAME}")
endif()

# Link libraries
target_link_libraries(${TARGET_EXE_NAME} PRIVATE cnpy)

# Include directories
target_include_directories(${TARGET_EXE_NAME} PRIVATE
  "${CNPY_DIR}"
)

# Compile definitions
target_compile_definitions(${TARGET_EXE_NAME} PRIVATE
  DMLC_LOG_STACK_TRACE=0
)

# Optimize for gem5 simulation: static libgcc/libstdc++ + no PIE for faster initialization
# This reduces dynamic linker overhead and mprotect syscalls in gem5
target_compile_options(${TARGET_EXE_NAME} PRIVATE -fno-pie)
target_link_options(${TARGET_EXE_NAME} PRIVATE
  -no-pie
  -static
  -static-libgcc
  -static-libstdc++
)

# ============================================================================
# Create Host Object Files from Binary Data
# ============================================================================
if(HOST_OBJ_FILES)
  message(STATUS "Setting up host object file creation")

  add_custom_target(create_host_objects ALL
    COMMENT "Creating/updating host object files"
  )

  set(HOST_OBJ_TARGETS "")
  foreach(host_obj_file ${HOST_OBJ_FILES})
    # Extract binary file name
    get_filename_component(host_obj_name ${host_obj_file} NAME_WE)
    string(REPLACE ".host" "" base_name ${host_obj_name})
    set(bin_file_name "${base_name}.bin")

    get_filename_component(host_obj_dir ${host_obj_file} DIRECTORY)
    set(bin_file_path "${host_obj_dir}/${bin_file_name}")

    # Create unique target name using full path relative to SCAN_KERNEL_PATH
    file(RELATIVE_PATH rel_obj_path "${SCAN_KERNEL_PATH}" "${host_obj_file}")
    string(REPLACE "/" "_" host_obj_target_name "${rel_obj_path}")
    string(REPLACE ".host.o" "" host_obj_target_name "${host_obj_target_name}")
    string(REPLACE "." "_" host_obj_target_name "${host_obj_target_name}")
    set(host_obj_target "create_${host_obj_target_name}")

    # Select linker based on ISA
    if(ISA STREQUAL "ARM" AND DEFINED AARCH64_LINKER)
      set(LINKER_CMD ${AARCH64_LINKER})
      message(STATUS "Using ARM linker: ${AARCH64_LINKER}")
    else()
      set(LINKER_CMD ld)
    endif()

    add_custom_command(
      OUTPUT ${host_obj_file}
      COMMAND ${CMAKE_COMMAND} -E echo "Processing ${host_obj_file}"
      COMMAND ${LINKER_CMD} -r -b binary -o ${host_obj_file} ${bin_file_name}
      DEPENDS ${bin_file_path}
      COMMENT "Creating host object: ${host_obj_file}"
      WORKING_DIRECTORY ${host_obj_dir}
    )

    add_custom_target(${host_obj_target} DEPENDS ${host_obj_file})
    list(APPEND HOST_OBJ_TARGETS ${host_obj_target})
  endforeach()

  if(HOST_OBJ_TARGETS)
    add_dependencies(create_host_objects ${HOST_OBJ_TARGETS})
  endif()

  add_dependencies(${TARGET_EXE_NAME} create_host_objects)
endif()

# ============================================================================
# Status Messages
# ============================================================================
message(STATUS "========================================")
message(STATUS "Scan Register Executable Build Config")
message(STATUS "========================================")
message(STATUS "ISA:                ${ISA}")
message(STATUS "Executable Name:    ${TARGET_EXE_NAME}")
message(STATUS "TVM Root:           ${TVM_ROOT}")
message(STATUS "Scan Kernel Path:   ${SCAN_KERNEL_PATH}")
message(STATUS "Scan Kernel Source: ${SCAN_KERNEL_SOURCE}")
message(STATUS "Host Object Files:  ${HOST_OBJ_COUNT}")
message(STATUS "========================================")
