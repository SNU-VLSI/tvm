# Toolchain for Cortex-M3 (AArch32) bare-metal cross compilation
# Uses arm-none-eabi toolchain on host to target another ARM machine.

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR "cortex-m3")
set(CMAKE_C_COMPILER "arm-none-eabi-gcc")
set(CMAKE_CXX_COMPILER "arm-none-eabi-g++")

# Disable compiler tests for bare metal
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# ARM-specific flags
add_compile_options(-mcpu=cortex-m3 -mthumb -mfloat-abi=soft)

# Add format specifier definitions for ARM bare metal
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

# Disable host-specific features
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)