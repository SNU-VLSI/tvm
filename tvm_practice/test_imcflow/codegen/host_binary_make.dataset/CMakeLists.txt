cmake_minimum_required(VERSION 3.18)

# ============================================================================
# Dataset Evaluation Build System
# Builds execute_graph_for_dataset for evaluating models on datasets (e.g., CIFAR-10)
# ============================================================================

# ============================================================================
# ISA Selection
# ============================================================================
set(ISA "X86" CACHE STRING "Target ISA: ARM or X86")
set_property(CACHE ISA PROPERTY STRINGS "ARM" "X86")

if(NOT ISA MATCHES "^(ARM|X86)$")
  message(FATAL_ERROR "Invalid ISA: ${ISA}. Must be ARM or X86")
endif()

message(STATUS "Building for ISA: ${ISA}")

# ============================================================================
# Project Setup
# ============================================================================
if(ISA STREQUAL "ARM")
  project(dataset_eval_arm C CXX)
else()
  project(dataset_eval_x86 C CXX)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# ============================================================================
# Input Validation
# ============================================================================
if(NOT EXISTS "${MLF_TAR}")
  message(FATAL_ERROR "MLF_TAR not found: ${MLF_TAR}\nProvide path to lib_graph_system-lib.tar")
endif()

# Host object files (optional)
file(GLOB_RECURSE HOST_OBJ_FILES "${H_OBJ_PATH}/*.host.o")
if(HOST_OBJ_FILES)
  list(LENGTH HOST_OBJ_FILES HOST_OBJ_COUNT)
  message(STATUS "Found ${HOST_OBJ_COUNT} host object files")
else()
  message(STATUS "No host object files found (optional)")
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
  message(FATAL_ERROR "TVM CRT directory not found at: ${CRT_LIB_BASE}\nSet -DTVM_ROOT=/path/to/tvm")
endif()

# DMM C API path (TVM_HOME only; no fallback)
set(DMM_CAPI_DIR "$ENV{TVM_HOME}/3rdparty/measurement_utils/capi")

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

# Collect MLF sources
file(GLOB_RECURSE TVM_MODEL_SRCS
  "${MLF_DIR}/codegen/host/src/*.c"
  "${MLF_DIR}/codegen/host/src/*.cc"
)
file(GLOB_RECURSE TVM_MODEL_OBJS
  "${MLF_DIR}/codegen/host/lib/*.o"
)
set(TVM_MODEL_INCLUDE "${MLF_DIR}/include" "${MLF_DIR}/runtime/include")
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
    message(FATAL_ERROR "No sources found for CRT component '${crt_lib_name}'")
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

# Add DMM power measurement support to tvm_model
if("$ENV{IMCFLOW_HOST_OS}" STREQUAL "linux" AND "$ENV{IMCFLOW_MEASURE_POWER}" STREQUAL "1")
  if(NOT EXISTS "${DMM_CAPI_DIR}/dmm_measure.c")
    message(FATAL_ERROR "DMM C API not found: ${DMM_CAPI_DIR}/dmm_measure.c\nSet TVM_HOME to your TVM repo root")
  endif()
  target_sources(tvm_model PRIVATE "${DMM_CAPI_DIR}/dmm_measure.c")
  set_source_files_properties("${DMM_CAPI_DIR}/dmm_measure.c" PROPERTIES LANGUAGE C)
  target_include_directories(tvm_model PRIVATE "${DMM_CAPI_DIR}")
  message(STATUS "Adding DMM measurement support to tvm_model")
endif()

target_compile_definitions(tvm_model PRIVATE DMLC_LOG_STACK_TRACE=0)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(tvm_model PRIVATE
    -Wno-unused-variable
    -Wno-missing-braces
    -Wno-unused-const-variable
  )
endif()

# ============================================================================
# Build Dataset Evaluation Executable
# ============================================================================
add_executable(execute_graph_for_dataset
  ${CMAKE_CURRENT_LIST_DIR}/src/execute_graph_for_dataset.c
)

target_link_libraries(execute_graph_for_dataset PRIVATE
  tvm_model ${CRT_LIBS} m stdc++
)

target_include_directories(execute_graph_for_dataset PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/src"
  "${CRT_LIB_BASE}/include"
  "${TVM_ROOT}/include"
  "${TVM_ROOT}/3rdparty/dlpack/include"
  "${TVM_ROOT}/3rdparty/dmlc-core/include"
  "${CRT_CONFIG_INCLUDE}"
  ${TVM_MODEL_INCLUDE}
)

target_sources(execute_graph_for_dataset PRIVATE
  "${TVM_ROOT}/src/runtime/crt/host/platform.cc"
)

# Add DMM power measurement support (linux + IMCFLOW_MEASURE_POWER=1)
if("$ENV{IMCFLOW_HOST_OS}" STREQUAL "linux" AND "$ENV{IMCFLOW_MEASURE_POWER}" STREQUAL "1")
  if(NOT EXISTS "${DMM_CAPI_DIR}/dmm_measure.c")
    message(FATAL_ERROR "DMM C API not found: ${DMM_CAPI_DIR}/dmm_measure.c\nSet TVM_HOME to your TVM repo root")
  endif()
  target_sources(execute_graph_for_dataset PRIVATE "${DMM_CAPI_DIR}/dmm_measure.c")
  target_include_directories(execute_graph_for_dataset PRIVATE "${DMM_CAPI_DIR}")
  message(STATUS "Adding DMM measurement support to dataset executable")
endif()

target_compile_definitions(execute_graph_for_dataset PRIVATE
  DMLC_LOG_STACK_TRACE=0
  TVM_WORKSPACE_SIZE_BYTES=8388608
  TVM_CRT_LOG_LEVEL=3
)

# Static linking for portability
target_compile_options(execute_graph_for_dataset PRIVATE -fno-pie)
target_link_options(execute_graph_for_dataset PRIVATE
  -no-pie
  -static
  -static-libgcc
  -static-libstdc++
)

# ============================================================================
# Host Object File Creation (if needed)
# ============================================================================
if(HOST_OBJ_FILES)
  add_custom_target(create_host_objects ALL
    COMMENT "Creating/updating host object files"
  )

  set(HOST_OBJ_TARGETS "")
  foreach(host_obj_file ${HOST_OBJ_FILES})
    get_filename_component(host_obj_name ${host_obj_file} NAME_WE)
    string(REPLACE ".host" "" base_name ${host_obj_name})
    set(bin_file_name "${base_name}.bin")

    get_filename_component(host_obj_dir ${host_obj_file} DIRECTORY)
    set(bin_file_path "${host_obj_dir}/${bin_file_name}")

    file(RELATIVE_PATH rel_obj_path "${H_OBJ_PATH}" "${host_obj_file}")
    string(REPLACE "/" "_" host_obj_target_name "${rel_obj_path}")
    string(REPLACE ".host.o" "" host_obj_target_name "${host_obj_target_name}")
    string(REPLACE "." "_" host_obj_target_name "${host_obj_target_name}")
    set(host_obj_target "create_${host_obj_target_name}")

    if(ISA STREQUAL "ARM" AND DEFINED AARCH64_LINKER)
      set(LINKER_CMD ${AARCH64_LINKER})
    else()
      set(LINKER_CMD ld)
    endif()

    add_custom_command(
      OUTPUT ${host_obj_file}
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
message(STATUS "Dataset Evaluation Build Configuration")
message(STATUS "========================================")
message(STATUS "ISA:              ${ISA}")
message(STATUS "TVM Root:         ${TVM_ROOT}")
message(STATUS "MLF Extracted:    ${MLF_DIR}")
message(STATUS "Executable:       execute_graph_for_dataset")
message(STATUS "========================================")
