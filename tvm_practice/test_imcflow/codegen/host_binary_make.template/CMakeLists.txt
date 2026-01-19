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
  project(tvm_arm_lib C CXX)
  set(TARGET_LIB_NAME tvm_arm)
  set(TARGET_DUMMY_FILE "${CMAKE_BINARY_DIR}/tvm_arm_dummy.c")
  set(TARGET_DUMMY_FUNC "void tvm_arm_dummy(void){}\n")
else()
  project(tvm_x86_lib C CXX)
  set(TARGET_LIB_NAME tvm_x86)
  set(TARGET_DUMMY_FILE "${CMAKE_BINARY_DIR}/tvm_x86_dummy.c")
  set(TARGET_DUMMY_FUNC "void tvm_x86_dummy(void){}\n")
endif()

# Options
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# ============================================================================
# Input Validation
# ============================================================================
# MLF TAR file
if(NOT EXISTS "${MLF_TAR}")
  message(FATAL_ERROR "MLF_TAR not found: ${MLF_TAR}\nGenerate it with build script")
endif()

# Host object files
file(GLOB_RECURSE HOST_OBJ_FILES "${H_OBJ_PATH}/*.host.o")

if(ISA STREQUAL "ARM")
  # ARM: HOST_OBJ_FILES optional
  if(HOST_OBJ_FILES)
    message(STATUS "Found host object files for ARM")
    foreach(obj_file ${HOST_OBJ_FILES})
      message(STATUS "  - ${obj_file}")
    endforeach()
  else()
    message(STATUS "No host object files found (optional for ARM)")
  endif()
else()
  # X86: HOST_OBJ_FILES optional
  if(NOT HOST_OBJ_FILES)
    message(STATUS "No host object files (*.host.o) found in: ${H_OBJ_PATH} (optional)")
  else()
    list(LENGTH HOST_OBJ_FILES HOST_OBJ_COUNT)
    message(STATUS "Found ${HOST_OBJ_COUNT} host object files:")
    foreach(obj_file ${HOST_OBJ_FILES})
      message(STATUS "  - ${obj_file}")
    endforeach()
  endif()
endif()

# ============================================================================
# TVM Root Setup
# ============================================================================
if(NOT DEFINED TVM_ROOT)
  set(TVM_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." CACHE PATH "Path to TVM root")
endif()
if(NOT EXISTS "${TVM_ROOT}/src/runtime/crt")
  if(DEFINED ENV{TVM_ROOT})
    set(TVM_ROOT "$ENV{TVM_ROOT}" CACHE PATH "Path to TVM root" FORCE)
  elseif(DEFINED ENV{TVM_HOME})
    set(TVM_ROOT "$ENV{TVM_HOME}" CACHE PATH "Path to TVM root" FORCE)
  endif()
endif()

set(CRT_LIB_BASE "${TVM_ROOT}/src/runtime/crt")
if(NOT EXISTS "${CRT_LIB_BASE}")
  message(FATAL_ERROR "TVM CRT directory not found at: ${CRT_LIB_BASE}\nSet -DTVM_ROOT=/path/to/tvm (or export TVM_ROOT/TVM_HOME)")
endif()

# ============================================================================
# Build cnpy Library (from TVM 3rdparty)
# ============================================================================
# Find static zlib library for static linking
set(ZLIB_USE_STATIC_LIBS ON)
find_package(ZLIB REQUIRED)

# If dynamic zlib was found, try to find static version manually
if(ZLIB_LIBRARIES MATCHES "\\.so$")
  message(STATUS "Dynamic zlib found, searching for static version...")
  find_library(ZLIB_STATIC_LIB libz.a PATHS /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu)
  if(ZLIB_STATIC_LIB)
    set(ZLIB_LIBRARIES ${ZLIB_STATIC_LIB})
    message(STATUS "Using static zlib: ${ZLIB_LIBRARIES}")
  else()
    message(WARNING "Static zlib not found, will use dynamic (may cause linking issues with -static)")
  endif()
endif()

set(CNPY_DIR "${TVM_ROOT}/3rdparty/cnpy")
if(NOT EXISTS "${CNPY_DIR}")
  message(FATAL_ERROR "cnpy not found at: ${CNPY_DIR}\nEnsure TVM's 3rdparty/cnpy exists")
endif()

# Build cnpy as a static library
add_library(cnpy STATIC "${CNPY_DIR}/cnpy.cpp")
target_include_directories(cnpy PUBLIC "${CNPY_DIR}")
target_link_libraries(cnpy PUBLIC ${ZLIB_LIBRARIES})
target_compile_features(cnpy PUBLIC cxx_std_11)

message(STATUS "Building cnpy from: ${CNPY_DIR}")
message(STATUS "Using zlib: ${ZLIB_LIBRARIES}")

# ============================================================================
# Extract MLF
# ============================================================================
set(MLF_DIR "${CMAKE_BINARY_DIR}/mlf")
file(MAKE_DIRECTORY "${MLF_DIR}")
execute_process(
  COMMAND ${CMAKE_COMMAND} -E tar xvf "${MLF_TAR}"
  WORKING_DIRECTORY "${MLF_DIR}"
  RESULT_VARIABLE TAR_RESULT
)
if(NOT TAR_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to extract MLF tar (${MLF_TAR})")
endif()

# Collect MLF sources/headers
file(GLOB_RECURSE TVM_MODEL_SRCS
  "${MLF_DIR}/codegen/host/src/*.c"
  "${MLF_DIR}/codegen/host/src/*.cc"
)
file(GLOB_RECURSE TVM_MODEL_OBJS
  "${MLF_DIR}/codegen/host/lib/*.o"
)
set(TVM_MODEL_INCLUDE "${MLF_DIR}/include" "${MLF_DIR}/runtime/include")

# crt_config.h for this target
set(CRT_CONFIG_INCLUDE "${CMAKE_CURRENT_LIST_DIR}/crt_config")

# ============================================================================
# Build CRT Component Libraries
# ============================================================================
set(CRT_LIBS
  graph_executor_module
  graph_executor
  common
  memory
)

foreach(crt_lib_name ${CRT_LIBS})
  file(GLOB_RECURSE crt_lib_srcs
    "${CRT_LIB_BASE}/${crt_lib_name}/*.c"
    "${CRT_LIB_BASE}/${crt_lib_name}/*.cc"
  )
  if(NOT crt_lib_srcs)
    message(FATAL_ERROR "No sources found for CRT component '${crt_lib_name}' in ${CRT_LIB_BASE}/${crt_lib_name}")
  endif()

  add_library(${crt_lib_name} STATIC ${crt_lib_srcs})
  target_include_directories(${crt_lib_name} PRIVATE
    "${CRT_LIB_BASE}/include"
    "${TVM_ROOT}/include"
    "${TVM_ROOT}/3rdparty/dlpack/include"
    "${TVM_ROOT}/3rdparty/dmlc-core/include"
    ${TVM_MODEL_INCLUDE}
    "${CRT_CONFIG_INCLUDE}"
  )
  target_compile_definitions(${crt_lib_name} PRIVATE DMLC_LOG_STACK_TRACE=0 TVM_CRT_DEBUG=1)
endforeach()

# ============================================================================
# Build Model Library
# ============================================================================
add_library(tvm_model STATIC)
set_source_files_properties(${TVM_MODEL_SRCS} PROPERTIES LANGUAGE CXX)
target_sources(tvm_model PRIVATE ${TVM_MODEL_SRCS} ${TVM_MODEL_OBJS} ${HOST_OBJ_FILES})

target_include_directories(tvm_model PRIVATE
  ${TVM_MODEL_INCLUDE}
  "${CRT_CONFIG_INCLUDE}"
  "${CRT_LIB_BASE}/include"
  "${TVM_ROOT}/include"
  "${TVM_ROOT}/3rdparty/dlpack/include"
  "${TVM_ROOT}/3rdparty/dmlc-core/include"
)

target_compile_definitions(tvm_model PRIVATE DMLC_LOG_STACK_TRACE=0)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(tvm_model PRIVATE
    -Wno-unused-variable
    -Wno-missing-braces
    -Wno-unused-const-variable
  )
endif()

# ============================================================================
# Build Target Library (tvm_arm or tvm_x86)
# ============================================================================
file(WRITE "${TARGET_DUMMY_FILE}" "${TARGET_DUMMY_FUNC}")

add_library(${TARGET_LIB_NAME} STATIC "${TARGET_DUMMY_FILE}")
target_link_libraries(${TARGET_LIB_NAME} PUBLIC tvm_model ${CRT_LIBS})

target_include_directories(${TARGET_LIB_NAME} PUBLIC
  ${TVM_MODEL_INCLUDE}
  "${CRT_LIB_BASE}/include"
  "${CRT_CONFIG_INCLUDE}"
  "${TVM_ROOT}/include"
  "${TVM_ROOT}/3rdparty/dlpack/include"
  "${TVM_ROOT}/3rdparty/dmlc-core/include"
)

target_compile_definitions(${TARGET_LIB_NAME} PUBLIC DMLC_LOG_STACK_TRACE=0)

# ============================================================================
# Build Host Runner Executable (Common)
# ============================================================================
option(TVM_BUILD_HOST_RUNNER "Build a host runner executable" OFF)

if(TVM_BUILD_HOST_RUNNER)
  add_executable(tvm_host_runner)
  
  # Select source file
  if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/tests/${MAIN_SCRIPT}")
    target_sources(tvm_host_runner PRIVATE ${CMAKE_CURRENT_LIST_DIR}/tests/${MAIN_SCRIPT})
    message(STATUS "Using ${MAIN_SCRIPT} for host runner")
  else()
    message(FATAL_ERROR "No source file found for host runner: ${MAIN_SCRIPT}")
  endif() 
  
  # Add program_scan_reg_kernel.cc if it exists
  if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/tests/program_scan_reg_kernel.cc")
    target_sources(tvm_host_runner PRIVATE ${CMAKE_CURRENT_LIST_DIR}/tests/program_scan_reg_kernel.cc)
    message(STATUS "Adding program_scan_reg_kernel.cc to host runner")
  endif()
  
  # Common settings
  target_link_libraries(tvm_host_runner PRIVATE ${TARGET_LIB_NAME} tvm_model ${CRT_LIBS} m stdc++ cnpy)
  target_include_directories(tvm_host_runner PRIVATE
    "${CRT_LIB_BASE}/include"
    "${TVM_ROOT}/include"
    "${TVM_ROOT}/3rdparty/dlpack/include"
    "${TVM_ROOT}/3rdparty/dmlc-core/include"
    "${CRT_CONFIG_INCLUDE}"
    ${TVM_MODEL_INCLUDE}
    "${CNPY_DIR}"
  )
  target_sources(tvm_host_runner PRIVATE "${TVM_ROOT}/src/runtime/crt/host/platform.cc")
  target_compile_definitions(tvm_host_runner PRIVATE
    DMLC_LOG_STACK_TRACE=0
    TVM_WORKSPACE_SIZE_BYTES=8388608
    TVM_CRT_LOG_LEVEL=3
  )

  # Optimize for gem5 simulation: static libgcc/libstdc++ + no PIE for faster initialization
  # This reduces dynamic linker overhead and mprotect syscalls in gem5
  # Note: We don't use -static to allow dynamic linking of system libraries like zlib
  target_compile_options(tvm_host_runner PRIVATE -fno-pie)
  target_link_options(tvm_host_runner PRIVATE
    -no-pie
    -static-libgcc
    -static-libstdc++
  )
  
  message(STATUS "Host runner executable will be built")
endif()

# ============================================================================
# Build ARM-specific Bare-metal Runner
# ============================================================================
if(ISA STREQUAL "ARM")
  option(TVM_BUILD_ARM_RUNNER "Build ARM-Cortex runner executable (bare metal)" OFF)
  
  if(TVM_BUILD_ARM_RUNNER AND CMAKE_CROSSCOMPILING)
    if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/arm_runner.c")
      add_executable(tvm_arm_runner ${CMAKE_CURRENT_LIST_DIR}/arm_runner.c)
      
      # Group static libraries to avoid order-sensitive resolution
      if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_link_options(tvm_arm_runner PRIVATE -Wl,--start-group)
      endif()
      
      target_link_libraries(tvm_arm_runner PRIVATE 
        ${TARGET_LIB_NAME} tvm_model ${CRT_LIBS} m stdc++
      )
      
      if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_link_options(tvm_arm_runner PRIVATE -Wl,--end-group)
      endif()
      
      target_include_directories(tvm_arm_runner PRIVATE
        "${CRT_LIB_BASE}/include"
        "${TVM_ROOT}/include"
        "${TVM_ROOT}/3rdparty/dlpack/include"
        "${TVM_ROOT}/3rdparty/dmlc-core/include"
        "${CRT_CONFIG_INCLUDE}"
        ${TVM_MODEL_INCLUDE}
      )
      
      target_compile_definitions(tvm_arm_runner PRIVATE 
        DMLC_LOG_STACK_TRACE=0 
        TVM_CRT_LOG_LEVEL=0 
        TVM_ARM_LINUX=1
      )
      
      message(STATUS "ARM bare-metal runner will be built")
    else()
      message(WARNING "arm_runner.c not found, skipping ARM runner")
    endif()
  elseif(TVM_BUILD_ARM_RUNNER AND NOT CMAKE_CROSSCOMPILING)
    message(WARNING "TVM_BUILD_ARM_RUNNER requested but not cross-compiling; skipping")
  endif()
endif()

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

    # Create unique target name using full path relative to H_OBJ_PATH
    file(RELATIVE_PATH rel_obj_path "${H_OBJ_PATH}" "${host_obj_file}")
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

  add_dependencies(tvm_model create_host_objects)
endif()

# ============================================================================
# Status Messages
# ============================================================================
message(STATUS "========================================")
message(STATUS "TVM CRT Build Configuration")
message(STATUS "========================================")
message(STATUS "ISA:                ${ISA}")
message(STATUS "Target Library:     ${TARGET_LIB_NAME}")
message(STATUS "TVM Root:           ${TVM_ROOT}")
message(STATUS "MLF Extracted to:   ${MLF_DIR}")
message(STATUS "CRT Libraries:      ${CRT_LIBS}")
message(STATUS "Host Runner:        ${TVM_BUILD_HOST_RUNNER}")
if(ISA STREQUAL "ARM")
  message(STATUS "ARM Bare Runner:    ${TVM_BUILD_ARM_RUNNER}")
endif()
message(STATUS "========================================")
