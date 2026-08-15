# Toolchain for Cortex-A53 (AArch32) bare-metal cross compilation
# Uses arm-none-eabi toolchain on host to target another ARM machine.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR "aarch64")
if(NOT CMAKE_C_COMPILER)
  find_program(AARCH64_XILINX_GCC aarch64-xilinx-linux-gcc)
  find_program(AARCH64_GNU_GCC    aarch64-linux-gnu-gcc)
  if(AARCH64_XILINX_GCC)
    set(CMAKE_C_COMPILER ${AARCH64_XILINX_GCC})
  elseif(AARCH64_GNU_GCC)
    set(CMAKE_C_COMPILER ${AARCH64_GNU_GCC})
  else()
    message(FATAL_ERROR "No AArch64 cross GCC found")
  endif()
endif()
if(NOT CMAKE_CXX_COMPILER)
  if(CMAKE_C_COMPILER MATCHES "aarch64-xilinx-linux-gcc")
    set(CMAKE_CXX_COMPILER aarch64-xilinx-linux-g++)
  elseif(CMAKE_C_COMPILER MATCHES "aarch64-linux-gnu-gcc")
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
  else()
    message(FATAL_ERROR "No matching CXX for ${CMAKE_C_COMPILER}")
  endif()
endif()

# Detect and set the appropriate linker
if(NOT CMAKE_C_LINKER)
  if(CMAKE_C_COMPILER MATCHES "aarch64-xilinx-linux-gcc")
    find_program(AARCH64_XILINX_LD aarch64-xilinx-linux-ld)
    if(AARCH64_XILINX_LD)
      set(CMAKE_C_LINKER ${AARCH64_XILINX_LD})
    else()
      message(FATAL_ERROR "No aarch64-xilinx-linux-ld found")
    endif()
  elseif(CMAKE_C_COMPILER MATCHES "aarch64-linux-gnu-gcc")
    find_program(AARCH64_GNU_LD aarch64-linux-gnu-ld)
    if(AARCH64_GNU_LD)
      set(CMAKE_C_LINKER ${AARCH64_GNU_LD})
    else()
      message(FATAL_ERROR "No aarch64-linux-gnu-ld found")
    endif()
  else()
    message(FATAL_ERROR "No matching linker for ${CMAKE_C_COMPILER}")
  endif()
endif()

# Export linker information for use in CMakeLists.txt
set(AARCH64_LINKER ${CMAKE_C_LINKER} CACHE STRING "AArch64 linker executable" FORCE)
message(STATUS "Using AArch64 linker: ${AARCH64_LINKER}")

# Disable compiler tests for bare metal
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# CPU/FPU tuning (AArch32 on Cortex-A53)
# Adjust float-abi if your newlib/libgcc is built for hard-float
add_compile_options(
  -mcpu=cortex-a53
  -march=armv8-a
)

# Optional: printf format specifiers (bare metal convenience)
add_compile_definitions(
    PRId64=\"ld\"
    PRIu64=\"lu\"
    PRId32=\"d\"
    PRIu32=\"u\"
    PRId16=\"d\"
    PRIu16=\"u\"
    PRId8=\"d\"
    PRIu8=\"u\"
)

# Restrict host finding
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)