# Toolchain for Cortex-A53 (AArch32) bare-metal cross compilation
# Uses arm-none-eabi toolchain on host to target another ARM machine.

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR "cortex-a53")
set(CMAKE_C_COMPILER   "arm-none-eabi-gcc")
set(CMAKE_CXX_COMPILER "arm-none-eabi-g++")

# Disable compiler tests for bare metal
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# CPU/FPU tuning (AArch32 on Cortex-A53)
# Adjust float-abi if your newlib/libgcc is built for hard-float
add_compile_options(
  -mcpu=cortex-a53
  -mthumb
  -mfloat-abi=softfp
  -mfpu=neon-vfpv4
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