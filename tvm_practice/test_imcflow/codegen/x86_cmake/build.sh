#!/bin/bash

# Check if current directory is 'build'
CURRENT_DIR=$(basename "$PWD")
if [ "$CURRENT_DIR" != "build" ]; then
    echo "Error: This script must be run from the 'build' directory"
    echo "Current directory: $PWD"
    echo "Please run: mkdir -p build && cd build && ../build.sh"
    exit 1
fi

rm -rf *
cmake .. -DCMAKE_BUILD_TYPE=Debug -C ../cmake/config_imcflow_relu_a53.cmake; cmake --build .