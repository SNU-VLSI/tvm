message("Configuring for ARM Cortex-A53 target")
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_DIR}/arm-cortex-a53.cmake" CACHE FILEPATH "Path to toolchain file")
