#!/bin/bash
#
# Full rebuild and RTL test automation
#
# Usage: ./rebuild_and_test.sh <test_name> [skip_compile]
# Example: ./rebuild_and_test.sh one_mmquant_evl
#          ./rebuild_and_test.sh one_mmquant_evl skip_compile
#

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print usage
usage() {
    echo "Usage: $0 [OPTIONS] <test_name>"
    echo ""
    echo "Arguments:"
    echo "  test_name         : Test name (e.g., one_mmquant_evl) [REQUIRED]"
    echo ""
    echo "Options:"
    echo "  -h, --help        : Show this help message"
    echo "  -d, --device      : Rebuild device code before host rebuild"
    echo "  -s, --skip-compile: Skip VCS compilation (faster, uses existing binary)"
    echo ""
    echo "Examples:"
    echo "  $0 one_mmquant_evl"
    echo "  $0 --device one_mmquant_evl"
    echo "  $0 one_mmquant_evl --skip-compile"
    echo "  $0 --device one_mmquant_evl --skip-compile"
    echo ""
    echo "This script will:"
    echo "  1. [Optional] Rebuild device nodes (with --device flag)"
    echo "  2. Rebuild the host binary with updated device code"
    echo "  3. Copy binaries to RTL runner directory"
    echo "  4. [Optional] Compile VCS simulation (unless --skip-compile)"
    echo "  5. Run RTL simulation with gem5 + VCS"
    echo "  6. Display results and logs"
    exit 1
}

# Parse arguments
TEST_NAME=""
SKIP_COMPILE=""
REBUILD_DEVICE=""

while [ $# -gt 0 ]; do
    case $1 in
        -h|--help)
            usage
            ;;
        -d|--device)
            REBUILD_DEVICE="yes"
            shift
            ;;
        -s|--skip-compile|skip_compile)
            SKIP_COMPILE="skip_compile"
            shift
            ;;
        -*)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            usage
            ;;
        *)
            if [ -z "$TEST_NAME" ]; then
                TEST_NAME=$1
            else
                echo -e "${RED}Error: Multiple test names specified${NC}"
                usage
            fi
            shift
            ;;
    esac
done

# Check if test name was provided
if [ -z "$TEST_NAME" ]; then
    echo -e "${RED}Error: Test name is required${NC}"
    echo ""
    usage
fi

# Directory paths
CODEGEN_DIR="/root/project/tvm/tvm_practice/test_imcflow/codegen"
TEST_DIR="${CODEGEN_DIR}/${TEST_NAME}"
HOST_BINARY_DIR="${TEST_DIR}/host_binary_make"
TEMPLATE_DIR="${CODEGEN_DIR}/host_binary_make.template"
RTL_RUNNER_DIR="/root/project/imcflow/pmap/ISA_sim/gem5/tests/imcflow/rtl_runner"

# Check if test directory exists
if [ ! -d "$TEST_DIR" ]; then
    echo -e "${RED}Error: Test directory not found: ${TEST_DIR}${NC}"
    echo "Available tests:"
    ls -1d "${CODEGEN_DIR}"/*_evl 2>/dev/null | xargs -n1 basename | sort
    exit 1
fi

echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  TVM ImcFlow: Rebuild and RTL Test                        ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Test:${NC} ${YELLOW}${TEST_NAME}${NC}"
echo -e "${BLUE}Date:${NC} $(date '+%Y-%m-%d %H:%M:%S')"
if [ -n "$REBUILD_DEVICE" ]; then
    echo -e "${BLUE}Mode:${NC} Device + Host rebuild"
else
    echo -e "${BLUE}Mode:${NC} Host rebuild only"
fi
echo ""

# Determine total steps
if [ -n "$REBUILD_DEVICE" ]; then
    TOTAL_STEPS=5
    STEP_OFFSET=1
else
    TOTAL_STEPS=4
    STEP_OFFSET=0
fi

# Step 0 (optional): Rebuild device nodes
if [ -n "$REBUILD_DEVICE" ]; then
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}[Step 1/${TOTAL_STEPS}]${NC} Rebuilding device nodes..."
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    DEVICE_SCRIPT="${CODEGEN_DIR}/utils/rebuild_device_node.sh"

    if [ ! -f "$DEVICE_SCRIPT" ]; then
        echo -e "${RED}✗${NC} Device rebuild script not found: ${DEVICE_SCRIPT}"
        exit 1
    fi

    echo -e "${BLUE}→${NC} Running: ${DEVICE_SCRIPT} ${TEST_NAME}"
    mkdir -p "${TEST_DIR}/logs"
    if bash "${DEVICE_SCRIPT}" "${TEST_NAME}" > "${TEST_DIR}/logs/device_rebuild.log" 2>&1; then
        echo -e "${GREEN}✓${NC} Device nodes rebuilt successfully"
    else
        echo -e "${RED}✗${NC} Device rebuild failed"
        echo -e "${YELLOW}Check logs:${NC} ${TEST_DIR}/logs/device_rebuild.log"
        tail -20 "${TEST_DIR}/logs/device_rebuild.log"
        exit 1
    fi

    echo ""
fi

# Step 1: Rebuild host binary
CURRENT_STEP=$((1 + STEP_OFFSET))
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}[Step ${CURRENT_STEP}/${TOTAL_STEPS}]${NC} Rebuilding host binary..."
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Copy template if it doesn't exist
if [ ! -d "${HOST_BINARY_DIR}" ]; then
    echo -e "${BLUE}→${NC} Copying host_binary_make template to test directory..."
    cp -r "${TEMPLATE_DIR}" "${HOST_BINARY_DIR}"
    echo -e "${GREEN}✓${NC} Template copied"
fi

# Create build directory if it doesn't exist
mkdir -p "${HOST_BINARY_DIR}/build"
cd "${HOST_BINARY_DIR}/build"

# Run build script (use "." for test directory since we're building in-place)
echo -e "${BLUE}→${NC} Running: direnv exec . ../build.sh execute_graph.c . x86"
if direnv exec . ../build.sh execute_graph.c "." x86 > "${TEST_DIR}/logs/rebuild.log" 2>&1; then
    echo -e "${GREEN}✓${NC} Host binary rebuilt successfully"
else
    echo -e "${RED}✗${NC} Host binary build failed"
    echo -e "${YELLOW}Check logs:${NC} ${TEST_DIR}/logs/rebuild.log"
    tail -20 "${TEST_DIR}/logs/rebuild.log"
    exit 1
fi

# Verify binary exists
if [ -f "tvm_host_runner" ]; then
    SIZE=$(stat -c%s "tvm_host_runner")
    echo -e "${GREEN}✓${NC} Binary: tvm_host_runner ($(numfmt --to=iec-i --suffix=B $SIZE))"
else
    echo -e "${RED}✗${NC} Binary not found: tvm_host_runner"
    exit 1
fi

echo ""

# Step 2: Copy binaries to RTL runner
CURRENT_STEP=$((2 + STEP_OFFSET))
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}[Step ${CURRENT_STEP}/${TOTAL_STEPS}]${NC} Copying binaries to RTL runner..."
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

cd "${RTL_RUNNER_DIR}"

# Create binaries directory if it doesn't exist
mkdir -p binaries

# Copy binary
echo -e "${BLUE}→${NC} Copying tvm_host_runner..."
cp "${HOST_BINARY_DIR}/build/tvm_host_runner" binaries/
echo -e "${GREEN}✓${NC} Binary copied"

# Copy MLF directory
echo -e "${BLUE}→${NC} Copying MLF directory..."
if [ -d "${HOST_BINARY_DIR}/build/mlf" ]; then
    rm -rf mlf
    cp -r "${HOST_BINARY_DIR}/build/mlf" .
    echo -e "${GREEN}✓${NC} MLF directory copied"
else
    echo -e "${YELLOW}⚠${NC}  MLF directory not found (may use existing)"
fi

echo ""

# Step 3: Compile VCS simulation (if needed)
CURRENT_STEP=$((3 + STEP_OFFSET))
if [ "$SKIP_COMPILE" != "skip_compile" ]; then
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}[Step ${CURRENT_STEP}/${TOTAL_STEPS}]${NC} Compiling VCS simulation..."
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    if [ -f "build/simv_imcflow_gem5" ]; then
        echo -e "${YELLOW}⚠${NC}  VCS binary exists, skipping compilation"
        echo -e "${YELLOW}→${NC} Use '${0} ${TEST_NAME} --skip-compile' to skip this check"
    else
        echo -e "${BLUE}→${NC} Running: make compile"
        if make compile > logs/vcs_compile.log 2>&1; then
            echo -e "${GREEN}✓${NC} VCS compilation successful"
        else
            echo -e "${RED}✗${NC} VCS compilation failed"
            echo -e "${YELLOW}Check logs:${NC} logs/vcs_compile.log"
            tail -20 logs/vcs_compile.log
            exit 1
        fi
    fi
else
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}[Step ${CURRENT_STEP}/${TOTAL_STEPS}]${NC} Skipping VCS compilation"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}⚠${NC}  Using existing VCS binary"
fi

echo ""

# Step 4: Run RTL simulation
CURRENT_STEP=$((4 + STEP_OFFSET))
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}[Step ${CURRENT_STEP}/${TOTAL_STEPS}]${NC} Running RTL simulation..."
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo -e "${BLUE}→${NC} Running: ./run.sh tvm_host_runner no ${TEST_NAME}"
echo -e "${YELLOW}Note:${NC} This may take several minutes..."
echo ""

START_TIME=$(date +%s)

# Run simulation
if ./run.sh tvm_host_runner no "${TEST_NAME}"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo ""
    echo -e "${GREEN}✓${NC} RTL simulation completed in ${DURATION}s"
else
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo ""
    echo -e "${RED}✗${NC} RTL simulation failed after ${DURATION}s"
    echo -e "${YELLOW}Check logs:${NC}"
    echo "  - gem5 output: logs/gem5_output.log"
    echo "  - VCS output:  logs/vcs_sim.log"
    exit 1
fi

echo ""

# Step 5: Display results summary
echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Test Results Summary                                      ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check for output file
OUTPUT_FILE="${TEST_DIR}/test_outputs/rtl_runner/output.npy"
if [ -f "$OUTPUT_FILE" ]; then
    SIZE=$(stat -c%s "$OUTPUT_FILE")
    echo -e "${GREEN}✓${NC} Output file created: ${OUTPUT_FILE##*/} (${SIZE} bytes)"
else
    echo -e "${YELLOW}⚠${NC}  Output file not found: ${OUTPUT_FILE}"
fi

# Check for key log files
echo ""
echo -e "${BLUE}Log files:${NC}"
if [ -f "logs/fsim_logs/gem5_output.log" ]; then
    echo -e "  ${GREEN}✓${NC} gem5 log: logs/fsim_logs/gem5_output.log"
else
    echo -e "  ${YELLOW}⚠${NC}  gem5 log not found"
fi

if [ -f "logs/fsim_logs/vcs_sim.log" ]; then
    echo -e "  ${GREEN}✓${NC} VCS log:  logs/fsim_logs/vcs_sim.log"
else
    echo -e "  ${YELLOW}⚠${NC}  VCS log not found"
fi

# Count signal logs
SIGNAL_LOGS=$(find logs/fsim_logs -name "*.log" 2>/dev/null | wc -l)
if [ $SIGNAL_LOGS -gt 0 ]; then
    echo -e "  ${GREEN}✓${NC} Signal logs: ${SIGNAL_LOGS} files in logs/fsim_logs/"
fi

# Check for FSDB waveform
if [ -f "logs/fsim_logs/imcflow_gem5.fsdb" ]; then
    FSDB_SIZE=$(stat -c%s "logs/fsim_logs/imcflow_gem5.fsdb")
    echo -e "  ${GREEN}✓${NC} Waveform: imcflow_gem5.fsdb ($(numfmt --to=iec-i --suffix=B $FSDB_SIZE))"
fi

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ All steps completed successfully!${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${BLUE}View logs in:${NC} ${RTL_RUNNER_DIR}/logs/fsim_logs/"
echo -e "${BLUE}View output:${NC} ${OUTPUT_FILE}"
echo ""
