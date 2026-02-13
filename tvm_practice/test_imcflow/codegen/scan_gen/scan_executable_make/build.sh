#!/bin/bash
# Build script for scan register executable
ISA=${1:-"x86"}

# Check ISA validity
if [ "$ISA" != "x86" ] && [ "$ISA" != "arm" ]; then
    echo "Error: Unsupported ISA '$ISA'. Supported ISAs are 'x86' and 'arm'."
    exit 1
fi

# Convert ISA to uppercase for CMake
ISA_UPPER=$(echo "$ISA" | tr '[:lower:]' '[:upper:]')

# Check if we're in build directory
CURRENT_DIR=$(basename "$PWD")
if [ "$CURRENT_DIR" != "build" ]; then
    echo "Creating build directory..."
    mkdir -p build
    cd build
fi

# Clean previous build
echo "Cleaning previous build..."
rm -rf *

# Configure with ${ISA} settings
echo "Configuring for ${ISA_UPPER}..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DISA=${ISA_UPPER} \
    -C ../cmake/config_scan.cmake

# Build
if [ $? -eq 0 ]; then
    echo "Configuration successful, building..."
    cmake --build . -j$(nproc)

    if [ $? -eq 0 ]; then
        echo ""
        echo "========================================"
        echo "Build successful!"
        echo "Executable: program_scan_reg"
        echo "========================================"
        echo ""
        echo "Usage: ./program_scan_reg <scan_file_path>"
        echo "  scan_file_path: Directory containing NPZ files (imce_<h>_<w>.npz)"
    else
        echo "Build failed!"
        exit 1
    fi
else
    echo "Configuration failed!"
    exit 1
fi
