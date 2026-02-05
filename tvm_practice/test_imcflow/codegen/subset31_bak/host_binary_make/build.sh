#!/bin/bash
# Build script for ${ISA} target
MAIN_SCRIPT=${1:-"execute_graph.c"}
MAIN_TEST_FOLDER=${2:-"one_relu_evl"}
ISA=${3:-"x86"}

#check ISA validity
if [ "$ISA" != "x86" ] && [ "$ISA" != "arm" ]; then
    echo "Error: Unsupported ISA '$ISA'. Supported ISAs are 'x86' and 'arm'."
    exit 1
fi

# Check if we're in build directory
CURRENT_DIR=$(basename "$PWD")
if [ "$CURRENT_DIR" != "build" ]; then
    echo "Warning: Recommended to run from build directory"
    echo "Creating build directory..."
    mkdir -p build
    cd build
fi

# Clean previous build
rm -rf *

# Configure with ${ISA} settings
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DMAIN_SCRIPT="$MAIN_SCRIPT" \
    -DMAIN_TEST_FOLDER="$MAIN_TEST_FOLDER" \
    -DISA=${ISA^^} \
    -C ../cmake/config_imcflow.cmake

# Build
if [ $? -eq 0 ]; then
    echo "Configuration successful, building..."
    cmake --build . -j$(nproc)
else
    echo "Configuration failed!"
    exit 1
fi
