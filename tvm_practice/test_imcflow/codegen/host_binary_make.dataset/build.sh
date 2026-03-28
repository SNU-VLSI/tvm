#!/bin/bash
# Build script for execute_graph_for_dataset
# Usage: ./build.sh <test_evl_dir> [ISA] [RETRY_DISABLE] [MAX_RETRY_COUNT]
#
# Example:
#   ./build.sh /root/project/tvm/tvm_practice/test_imcflow/codegen/resnet8_subset31_pretrained_orig_evl x86
#   ./build.sh /root/project/tvm/tvm_practice/test_imcflow/codegen/resnet8_subset31_pretrained_orig_evl x86 1
#   ./build.sh /root/project/tvm/tvm_practice/test_imcflow/codegen/resnet8_subset31_pretrained_orig_evl x86 0 5

set -e

# Arguments
TEST_EVL_DIR=${1:-""}
ISA=${2:-""}
RETRY_DISABLE=${3:-""}
MAX_RETRY_COUNT=${4:-""}

# Auto-detect ISA from IMCFLOW_HOST_ISA environment variable if not specified
if [ -z "$ISA" ]; then
    if [ -n "$IMCFLOW_HOST_ISA" ]; then
        ISA="$IMCFLOW_HOST_ISA"
        echo "Auto-detected ISA from IMCFLOW_HOST_ISA: $ISA"
    else
        ISA="x86"
        echo "No ISA specified and IMCFLOW_HOST_ISA not set, defaulting to: $ISA"
    fi
fi

# Validate arguments
if [ -z "$TEST_EVL_DIR" ]; then
    echo "Usage: $0 <test_evl_dir> [ISA]"
    echo ""
    echo "Arguments:"
    echo "  test_evl_dir     Path to the test evaluation directory (e.g., resnet8_*_evl)"
    echo "  ISA              Target ISA: x86 or arm (default: x86)"
    echo "  RETRY_DISABLE    Set to 1 to disable retry on timeout (default: retry enabled)"
    echo "  MAX_RETRY_COUNT  Maximum retry count (default: 3, ignored if RETRY_DISABLE=1)"
    echo ""
    echo "Available test directories:"
    find "$(dirname "$0")/../eval_dir" -maxdepth 1 -name "*_evl*" -type d 2>/dev/null | while read dir; do
        if [ -f "$dir/lib_graph_system-lib.tar" ]; then
            echo "  $dir"
        fi
    done
    exit 1
fi

# Convert to absolute path
TEST_EVL_DIR=$(realpath "$TEST_EVL_DIR")

# Validate ISA
ISA_UPPER=$(echo "$ISA" | tr '[:lower:]' '[:upper:]')
if [ "$ISA_UPPER" != "X86" ] && [ "$ISA_UPPER" != "ARM" ]; then
    echo "Error: Unsupported ISA '$ISA'. Supported ISAs are 'x86' and 'arm'."
    exit 1
fi

# Validate test directory
MLF_TAR="$TEST_EVL_DIR/lib_graph_system-lib.tar"
H_OBJ_PATH="$TEST_EVL_DIR/build"

if [ ! -f "$MLF_TAR" ]; then
    echo "Error: MLF tar not found: $MLF_TAR"
    exit 1
fi

if [ ! -d "$H_OBJ_PATH" ]; then
    echo "Warning: H_OBJ_PATH not found: $H_OBJ_PATH"
    echo "Creating empty directory (no host objects)"
    mkdir -p "$H_OBJ_PATH"
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "Building execute_graph_for_dataset"
echo "========================================"
echo "Test Dir:        $TEST_EVL_DIR"
echo "MLF Tar:         $MLF_TAR"
echo "H_OBJ_PATH:      $H_OBJ_PATH"
echo "ISA:             $ISA_UPPER"
echo "RETRY_DISABLE:   ${RETRY_DISABLE:-0}"
echo "MAX_RETRY_COUNT: ${MAX_RETRY_COUNT:-"3 (default)"}"
echo "========================================"

# Create and enter build directory
BUILD_DIR="$SCRIPT_DIR/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Clean previous build
rm -rf *

# Configure
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Debug
    -DISA="$ISA_UPPER"
    -DMLF_TAR="$MLF_TAR"
    -DH_OBJ_PATH="$H_OBJ_PATH"
)

if [ "$RETRY_DISABLE" = "1" ]; then
    CMAKE_ARGS+=(-DRETRY_DISABLE=ON)
fi

if [ -n "$MAX_RETRY_COUNT" ]; then
    CMAKE_ARGS+=(-DMAX_RETRY_COUNT="$MAX_RETRY_COUNT")
fi

if [ "$ISA_UPPER" = "ARM" ]; then
    TOOLCHAIN_FILE="$SCRIPT_DIR/cmake/arm-cortex-a53.cmake"
    if [ ! -f "$TOOLCHAIN_FILE" ]; then
        echo "Error: ARM toolchain file not found: $TOOLCHAIN_FILE"
        exit 1
    fi
    CMAKE_ARGS+=(-DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE")
    echo "Using ARM toolchain: $TOOLCHAIN_FILE"
fi

# Pass DEBUG_EXE to cmake if set
if [ "${DEBUG_EXE}" = "1" ]; then
    CMAKE_ARGS+=(-DDEBUG_EXE=1)
fi

cmake "$SCRIPT_DIR" "${CMAKE_ARGS[@]}"

# Build
if [ $? -eq 0 ]; then
    echo ""
    echo "Configuration successful, building..."
    cmake --build . -j$(nproc)

    if [ $? -eq 0 ]; then
        echo ""
        echo "========================================"
        echo "Build successful!"
        echo "========================================"
        echo "Executable: $BUILD_DIR/execute_graph_for_dataset"
        echo ""
        echo "Usage:"
        echo "  $BUILD_DIR/execute_graph_for_dataset \\"
        echo "      $BUILD_DIR/mlf/executor-config/graph/default.graph \\"
        echo "      $BUILD_DIR/mlf/parameters/default.params \\"
        echo "      /path/to/images.npy \\"
        echo "      /path/to/labels.npy \\"
        echo "      [num_samples]"
        echo ""
    else
        echo "Build failed!"
        exit 1
    fi
else
    echo "Configuration failed!"
    exit 1
fi
