cmake_minimum_required(VERSION 3.18)
project(tvm_arm_lib C CXX)

# Options
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Input: path to the Model Library Format tar (produced by build_arm.py)
# Defaults to the file emitted next to this CMakeLists.
if(NOT EXISTS "${MLF_TAR}")
  message(FATAL_ERROR "MLF_TAR not found: ${MLF_TAR}\nGenerate it with build_arm.py -e graph")
endif()

file(GLOB_RECURSE HOST_OBJ_FILES "${H_OBJ_PATH}/*.host.o")

# Locate TVM root (defaults to 4 directories up from this file), allow overrides
# and fall back to TVM_ROOT/TVM_HOME env vars if provided.
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
  message(FATAL_ERROR "TVM CRT directory not found at: ${CRT_LIB_BASE}\nSet -DTVM_ROOT=/path/to/tvm (or export TVM_ROOT/TVM_HOME) so that ${CRT_LIB_BASE} exists.")
endif()

# Extract MLF during configure
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
# Some MLFs include prebuilt objects under lib/
file(GLOB_RECURSE TVM_MODEL_OBJS
  "${MLF_DIR}/codegen/host/lib/*.o"
)
set(TVM_MODEL_INCLUDE "${MLF_DIR}/include" "${MLF_DIR}/runtime/include")

# crt_config.h for this target
set(CRT_CONFIG_INCLUDE "${CMAKE_CURRENT_LIST_DIR}/crt_config")

# Build CRT component libraries
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
    message(FATAL_ERROR "No sources found for CRT component '${crt_lib_name}' in ${CRT_LIB_BASE}/${crt_lib_name}.\nResolved TVM_ROOT: ${TVM_ROOT}")
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

# Build model as a static library
add_library(tvm_model STATIC)
# Note: object files (if any) are added as sources; CMake will link them in.
# Treat C++ sources as C++
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
# Quiet some benign warnings from generated code
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(tvm_model PRIVATE
    -Wno-unused-variable
    -Wno-missing-braces
    -Wno-unused-const-variable
  )
endif()

# Add a tiny source for tvm_arm to satisfy CMake
file(WRITE "${CMAKE_BINARY_DIR}/tvm_arm_dummy.c" "void tvm_arm_dummy(void){}\n")

# Aggregate library that depends on CRT and model
add_library(tvm_arm STATIC "${CMAKE_BINARY_DIR}/tvm_arm_dummy.c")
# Link implementation libraries
# Note: Consumers can link 'tvm_arm' and CMake will propagate needed libs
# (or link 'tvm_model' and listed CRT libs directly).
target_link_libraries(tvm_arm PUBLIC tvm_model ${CRT_LIBS})

target_include_directories(tvm_arm PUBLIC
  ${TVM_MODEL_INCLUDE}
  "${CRT_LIB_BASE}/include"
  "${CRT_CONFIG_INCLUDE}"
  "${TVM_ROOT}/include"
  "${TVM_ROOT}/3rdparty/dlpack/include"
  "${TVM_ROOT}/3rdparty/dmlc-core/include"
)

target_compile_definitions(tvm_arm PUBLIC DMLC_LOG_STACK_TRACE=0)

# Option to build a small host runner that embeds graph+params and runs once
option(TVM_BUILD_HOST_RUNNER "Build a host runner executable for testing CRT Graph Executor" OFF)

# Only build the host runner when not cross-compiling
if(TVM_BUILD_HOST_RUNNER AND NOT CMAKE_CROSSCOMPILING)
  add_executable(tvm_host_runner)
  target_sources(tvm_host_runner PRIVATE ${CMAKE_CURRENT_LIST_DIR}/main.c)
  target_link_libraries(tvm_host_runner PRIVATE tvm_arm)
  target_include_directories(tvm_host_runner PRIVATE
    "${CRT_LIB_BASE}/include"
    "${TVM_ROOT}/include"
    "${TVM_ROOT}/3rdparty/dlpack/include"
    "${TVM_ROOT}/3rdparty/dmlc-core/include"
    "${CRT_CONFIG_INCLUDE}"
    ${TVM_MODEL_INCLUDE}
  )
  target_sources(tvm_host_runner PRIVATE "${TVM_ROOT}/src/runtime/crt/host/platform.cc")
  target_compile_definitions(tvm_host_runner PRIVATE DMLC_LOG_STACK_TRACE=0 TVM_WORKSPACE_SIZE_BYTES=8388608 TVM_CRT_LOG_LEVEL=3)
elseif(TVM_BUILD_HOST_RUNNER AND CMAKE_CROSSCOMPILING)
  message(WARNING "TVM_BUILD_HOST_RUNNER requested but cross-compiling; skipping runner")
endif()

# Build a Cortex bare-metal runner when cross-compiling
option(TVM_BUILD_ARM_RUNNER "Build a ARM-Cortex runner executable (bare metal)" OFF)
if(TVM_BUILD_ARM_RUNNER AND CMAKE_CROSSCOMPILING)
  add_executable(tvm_arm_runner ${CMAKE_CURRENT_LIST_DIR}/arm_runner.c)
  # Group static libraries to avoid order-sensitive resolution and ensure libm satisfies references in tvm_model
  if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(tvm_arm_runner PRIVATE -Wl,--start-group)
  endif()
  target_link_libraries(tvm_arm_runner PRIVATE tvm_arm tvm_model graph_executor_module graph_executor common memory m stdc++)
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
  target_compile_definitions(tvm_arm_runner PRIVATE DMLC_LOG_STACK_TRACE=0 TVM_CRT_LOG_LEVEL=0 TVM_ARM_LINUX=1)
elseif(TVM_BUILD_ARM_RUNNER AND NOT CMAKE_CROSSCOMPILING)
  message(WARNING "TVM_BUILD_ARM_RUNNER requested but not cross-compiling; skipping runner")
endif()

# Print brief status
message(STATUS "Using TVM root: ${TVM_ROOT}")
message(STATUS "Extracted MLF to: ${MLF_DIR}")
message(STATUS "Built libraries: tvm_model, ${CRT_LIBS}")

# Custom target to create/update host object files
# This ensures host.o files are compatible with the current linker
if(DEFINED AARCH64_LINKER AND HOST_OBJ_FILES)
  message(STATUS "Setting up host object creation with linker: ${AARCH64_LINKER}")



  # Create custom target for host object creation
  add_custom_target(create_host_objects ALL
    COMMENT "Creating/updating host object files with ${AARCH64_LINKER}"
  )

    # Add commands to create each host object
  set(HOST_OBJ_TARGETS "")
  foreach(host_obj_file ${HOST_OBJ_FILES})
    # Extract the binary file name from host object path
    get_filename_component(host_obj_name ${host_obj_file} NAME_WE)
    string(REPLACE ".host" "" base_name ${host_obj_name})
    set(bin_file_name "${base_name}.bin")

    # Look for binary file in the same directory
    get_filename_component(host_obj_dir ${host_obj_file} DIRECTORY)
    set(bin_file_path "${host_obj_dir}/${bin_file_name}")

    # Create a unique target name for this host object
    string(REPLACE "/" "_" host_obj_target_name ${host_obj_name})
    set(host_obj_target "create_${host_obj_target_name}")

    # Add custom command to create this host object
    add_custom_command(
      OUTPUT ${host_obj_file}
      COMMAND ${CMAKE_COMMAND} -E echo "Processing ${host_obj_file}"
      COMMAND ${AARCH64_LINKER} -r -b binary -o ${host_obj_file} ${bin_file_name}
      DEPENDS ${bin_file_path}
      COMMENT "Creating host object: ${host_obj_file}"
      WORKING_DIRECTORY ${host_obj_dir}
    )

    # Create a custom target for this host object
    add_custom_target(${host_obj_target} DEPENDS ${host_obj_file})
    list(APPEND HOST_OBJ_TARGETS ${host_obj_target})
  endforeach()

  # Make the main target depend on all individual host object targets
  if(HOST_OBJ_TARGETS)
    add_dependencies(create_host_objects ${HOST_OBJ_TARGETS})
  endif()

  # Make tvm_model depend on host object creation
  add_dependencies(tvm_model create_host_objects)
endif()