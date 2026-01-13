#!/bin/bash
#
# Rebuild a single device node (inode or imce) from modified C++ source
#
# Usage: ./rebuild_device_node.sh <test_name> <node_type> <hid> <wid>
# Example: ./rebuild_device_node.sh one_mmquant_evl inode 0 0
#

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print usage
usage() {
    echo "Usage: $0 <test_name> [node_type] [hid wid]"
    echo ""
    echo "Arguments:"
    echo "  test_name  : Test name (e.g., one_mmquant_evl) [REQUIRED]"
    echo "  node_type  : Node type ('inode' or 'imce') [OPTIONAL]"
    echo "  hid        : Hardware ID (row coordinate) [OPTIONAL, requires wid]"
    echo "  wid        : Width ID (column coordinate) [OPTIONAL, requires hid]"
    echo ""
    echo "Examples:"
    echo "  $0 one_mmquant_evl              # Build all nodes (inode 0-3,0 and imce 0-3,1-4)"
    echo "  $0 one_mmquant_evl inode        # Build all inode nodes (0-3,0)"
    echo "  $0 one_mmquant_evl imce         # Build all imce nodes (0-3,1-4)"
    echo "  $0 one_mmquant_evl inode 0 0    # Build specific inode"
    echo "  $0 one_mmquant_evl imce 3 4     # Build specific imce"
    echo ""
    echo "Node ranges (fixed):"
    echo "  inode: hid 0-3, wid 0"
    echo "  imce:  hid 0-3, wid 1-4"
    exit 1
}

# Check arguments
if [ $# -lt 1 ] || [ $# -eq 3 ] || [ $# -gt 4 ]; then
    echo -e "${RED}Error: Invalid number of arguments${NC}"
    usage
fi

TEST_NAME=$1
NODE_TYPE=""
HID=""
WID=""

# Parse arguments based on count
if [ $# -eq 1 ]; then
    # Build all nodes
    MODE="all"
elif [ $# -eq 2 ]; then
    # Build all nodes of a specific type
    NODE_TYPE=$2
    MODE="type"
elif [ $# -eq 4 ]; then
    # Build specific node
    NODE_TYPE=$2
    HID=$3
    WID=$4
    MODE="specific"
fi

# Validate node type if specified
if [ -n "$NODE_TYPE" ]; then
    if [ "$NODE_TYPE" != "inode" ] && [ "$NODE_TYPE" != "imce" ]; then
        echo -e "${RED}Error: node_type must be 'inode' or 'imce'${NC}"
        usage
    fi
fi

# Base directory
CODEGEN_DIR="/root/project/tvm/tvm_practice/test_imcflow/codegen"
TEST_DIR="${CODEGEN_DIR}/${TEST_NAME}"

# Find the build directory (contains generated C++ code)
BUILD_DIR=$(find "${TEST_DIR}/build" -type d -name "tvmgen_default*" | head -1)

if [ -z "$BUILD_DIR" ]; then
    echo -e "${RED}Error: Could not find build directory in ${TEST_DIR}/build${NC}"
    echo "Make sure you have run the test at least once to generate code."
    exit 1
fi

# Function to build a single node
build_single_node() {
    local node_type=$1
    local hid=$2
    local wid=$3

    echo -e "${BLUE}=== Building ${node_type}_${hid}_${wid} ===${NC}"

    # Set source file and target based on node type
    if [ "$node_type" == "inode" ]; then
        SOURCE_FILE="${BUILD_DIR}/inode.cpp"
        TARGET_UPPER="INODE"
        TARGET_STR="INODE"
    else
        SOURCE_FILE="${BUILD_DIR}/imce.cpp"
        TARGET_UPPER="IMCE"
        TARGET_STR="imce"
    fi

    # Check if source file exists
    if [ ! -f "$SOURCE_FILE" ]; then
        echo -e "${RED}Error: Source file not found: ${SOURCE_FILE}${NC}"
        return 1
    fi

    # Output file names
    NODE_NAME="${node_type}_${hid}_${wid}"
    OBJ_FILE="${BUILD_DIR}/${NODE_NAME}_imem.o"
    OUT_FILE="${BUILD_DIR}/${NODE_NAME}_imem.out"
    BIN_FILE="${BUILD_DIR}/${NODE_NAME}_imem.bin"
    HOST_OBJ_FILE="${BUILD_DIR}/${NODE_NAME}_imem.host.o"

    cd "$BUILD_DIR"

    # Step 1: Compile C++ to object file
    echo -e "${GREEN}[1/7]${NC} Compiling ${SOURCE_FILE##*/} to ${OBJ_FILE##*/}..."
    clang -O1 --target=${TARGET_UPPER} -c -fPIC \
        $([ "$TARGET_UPPER" == "IMCE" ] && echo "-mllvm=-force-hardware-loops") \
        -mllvm=-force-nested-hardware-loop \
        -mllvm=-${TARGET_STR}_hid=${hid} \
        -mllvm=-${TARGET_STR}_wid=${wid} \
        -o "${OBJ_FILE}" "${SOURCE_FILE}"

    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Compilation failed${NC}"
        return 1
    fi

    # Step 2: Link to ELF executable
    echo -e "${GREEN}[2/7]${NC} Linking to ${OUT_FILE##*/}..."
    ld.lld -e 0 -Ttext 0x0 -o "${OUT_FILE}" "${OBJ_FILE}"

    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Linking failed${NC}"
        return 1
    fi

    # Step 3: Extract .text section to binary
    echo -e "${GREEN}[3/7]${NC} Extracting .text section to ${BIN_FILE##*/}..."
    llvm-objcopy -O binary --only-section=.text "${OUT_FILE}" "${BIN_FILE}"

    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Binary extraction failed${NC}"
        return 1
    fi

    # Step 4: Pad to 32-byte boundaries
    echo -e "${GREEN}[4/7]${NC} Padding binary to 32-byte boundaries..."
    python3 << EOF
def pad_bin_inplace(bin_file, stride=32):
    with open(bin_file, 'rb') as f:
        data = bytearray(f.read())

    # Ensure data is multiple of stride
    remainder = len(data) % stride
    if remainder != 0:
        data.extend(b'\x00' * (stride - remainder))

    with open(bin_file, 'wb') as f:
        f.write(data)

pad_bin_inplace('${BIN_FILE}', 32)
print(f"Padded to {len(open('${BIN_FILE}', 'rb').read())} bytes")
EOF

    # Step 5: Flip byte order (big endian → little endian)
    echo -e "${GREEN}[5/7]${NC} Flipping byte order..."
    python3 << EOF
def flip_byte_order(bin_file):
    with open(bin_file, 'rb') as f:
        data = bytearray(f.read())

    # Flip every 4-byte word
    for i in range(0, len(data), 4):
        if i + 3 < len(data):
            data[i], data[i+1], data[i+2], data[i+3] = \
                data[i+3], data[i+2], data[i+1], data[i]

    with open(bin_file, 'wb') as f:
        f.write(data)

flip_byte_order('${BIN_FILE}')
print("Byte order flipped")
EOF

    # Step 6: Create host-linkable object file
    echo -e "${GREEN}[6/7]${NC} Creating host object ${HOST_OBJ_FILE##*/}..."
    ld -r -b binary -o "${HOST_OBJ_FILE}" "${BIN_FILE}"

    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Host object creation failed${NC}"
        return 1
    fi

    # Step 7: Verify output
    echo -e "${GREEN}[7/7]${NC} Verifying outputs..."
    if [ -f "${HOST_OBJ_FILE}" ]; then
        SIZE=$(stat -c%s "${HOST_OBJ_FILE}")
        echo -e "${GREEN}✓${NC} ${NODE_NAME} successfully built (${SIZE} bytes)"
    else
        echo -e "${RED}✗${NC} Host object not created"
        return 1
    fi

    echo ""
    return 0
}

# Main build logic
echo -e "${BLUE}=== Rebuilding Device Nodes ===${NC}"
echo -e "Test: ${YELLOW}${TEST_NAME}${NC}"
echo -e "Build dir: ${BUILD_DIR}"
echo ""

# Counters for summary
TOTAL_NODES=0
SUCCESS_NODES=0
FAILED_NODES=0

# Build based on mode
if [ "$MODE" == "specific" ]; then
    # Build single specific node
    TOTAL_NODES=1
    build_single_node "$NODE_TYPE" "$HID" "$WID"
    if [ $? -eq 0 ]; then
        SUCCESS_NODES=1
    else
        FAILED_NODES=1
    fi
elif [ "$MODE" == "type" ]; then
    # Build all nodes of a specific type
    if [ "$NODE_TYPE" == "inode" ]; then
        echo -e "Building all inode nodes (hid 0-3, wid 0)..."
        echo ""
        for hid in {0..3}; do
            TOTAL_NODES=$((TOTAL_NODES + 1))
            build_single_node "inode" $hid 0
            if [ $? -eq 0 ]; then
                SUCCESS_NODES=$((SUCCESS_NODES + 1))
            else
                FAILED_NODES=$((FAILED_NODES + 1))
            fi
        done
    else  # imce
        echo -e "Building all imce nodes (hid 0-3, wid 1-4)..."
        echo ""
        for hid in {0..3}; do
            for wid in {1..4}; do
                TOTAL_NODES=$((TOTAL_NODES + 1))
                build_single_node "imce" $hid $wid
                if [ $? -eq 0 ]; then
                    SUCCESS_NODES=$((SUCCESS_NODES + 1))
                else
                    FAILED_NODES=$((FAILED_NODES + 1))
                fi
            done
        done
    fi
elif [ "$MODE" == "all" ]; then
    # Build all nodes (inode and imce)
    echo -e "Building all nodes (inode 0-3,0 and imce 0-3,1-4)..."
    echo ""

    # Build all inode nodes
    for hid in {0..3}; do
        TOTAL_NODES=$((TOTAL_NODES + 1))
        build_single_node "inode" $hid 0
        if [ $? -eq 0 ]; then
            SUCCESS_NODES=$((SUCCESS_NODES + 1))
        else
            FAILED_NODES=$((FAILED_NODES + 1))
        fi
    done

    # Build all imce nodes
    for hid in {0..3}; do
        for wid in {1..4}; do
            TOTAL_NODES=$((TOTAL_NODES + 1))
            build_single_node "imce" $hid $wid
            if [ $? -eq 0 ]; then
                SUCCESS_NODES=$((SUCCESS_NODES + 1))
            else
                FAILED_NODES=$((FAILED_NODES + 1))
            fi
        done
    done
fi

# Print summary
echo -e "${GREEN}=== Build Summary ===${NC}"
echo -e "Total nodes: ${TOTAL_NODES}"
echo -e "${GREEN}Successful: ${SUCCESS_NODES}${NC}"
if [ $FAILED_NODES -gt 0 ]; then
    echo -e "${RED}Failed: ${FAILED_NODES}${NC}"
fi
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Rebuild host binary: cd ../${TEST_NAME}/host_binary_make/build && direnv exec . ../build.sh execute_graph.c . x86"
echo "  2. Run RTL test: cd /root/project/imcflow/pmap/ISA_sim/gem5/tests/imcflow/rtl_runner && ./run.sh tvm_host_runner no ${TEST_NAME}"
echo ""
echo "Or use: ${YELLOW}./rebuild_and_test.sh ${TEST_NAME}${NC}"

# Exit with error if any builds failed
if [ $FAILED_NODES -gt 0 ]; then
    exit 1
fi
