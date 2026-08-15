#!/bin/bash

# Script to run chip test: generate test, transfer files and execute on remote chip
# Usage: ./run_chiptest.sh [options] <test_name> <input_setting>

# Function to display help message
show_help() {
    echo "Usage: $0 [options] <test_folder_evl.linux> <input_setting>"
    echo ""
    echo "Automated chip test workflow that:"
    echo "  1. Runs test.py to generate test folder"
    echo "  2. Transfers test folder to remote server"
    echo "  3. Transfers scan_reg_files to remote server"
    echo "  4. Transfers scan_executable to remote server"
    echo "  5. Programs scan registers on remote chip"
    echo "  6. Executes test on remote chip"
    echo ""
    echo "Options:"
    echo "  -s, --skip LIST  Comma-separated step numbers to skip (e.g., 1,3)"
    echo "  --power-config FILE  Enable tagged whole-run power measurement"
    echo "  -h, --help       Show this help message"
    echo ""
    echo "Arguments:"
    echo "  test_folder_evl.linux  Exact test folder name ending with '_evl.linux'"
    echo "  input_setting          Input setting for the test (e.g., 'ones', 'random', 'incremental')"
    echo ""
    echo "Examples:"
    echo "  $0 one_relu_evl.linux ones"
    echo "  $0 one_conv_small_evl.linux random"
    echo "  $0 -s 1,3 resnet8_subset31_pretrained_orig_evl.linux random"
    echo ""
    echo "Remote configuration is loaded from .env file."
    echo "Note: Host binary should be built before running this script"
    exit 0
}

# Parse options
SKIP_STEP1=false
SKIP_STEP2=false
SKIP_STEP3=false
SKIP_STEP4=false
SKIP_STEP5=false
SKIP_STEP6=false
SKIP_LIST=""
POWER_CONFIG="${IMCFLOW_POWER_CONFIG:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -s|--skip)
            if [[ -z "$2" ]]; then
                echo "Error: Missing value for $1"
                exit 1
            fi
            SKIP_LIST="$2"
            shift 2
            ;;
        --power-config)
            if [[ -z "$2" ]]; then
                echo "Error: Missing value for $1"
                exit 1
            fi
            POWER_CONFIG="$2"
            shift 2
            ;;
        --skip1|--skip-1)
            SKIP_STEP1=true
            shift
            ;;
        --skip2|--skip-2)
            SKIP_STEP2=true
            shift
            ;;
        --skip3|--skip-3)
            SKIP_STEP3=true
            shift
            ;;
        --skip4|--skip-4)
            SKIP_STEP4=true
            shift
            ;;
        --skip5|--skip-5)
            SKIP_STEP5=true
            shift
            ;;
        --skip6|--skip-6)
            SKIP_STEP6=true
            shift
            ;;
        --)
            shift
            break
            ;;
        -* )
            echo "Error: Unknown option $1"
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

if [[ -n "$SKIP_LIST" ]]; then
    IFS=',' read -ra SKIP_ARR <<< "$SKIP_LIST"
    for step in "${SKIP_ARR[@]}"; do
        step="${step//[[:space:]]/}"
        case "$step" in
            1) SKIP_STEP1=true ;;
            2) SKIP_STEP2=true ;;
            3) SKIP_STEP3=true ;;
            4) SKIP_STEP4=true ;;
            5) SKIP_STEP5=true ;;
            6) SKIP_STEP6=true ;;
            "") ;;
            *)
                echo "Error: Invalid step in skip list: $step"
                exit 1
                ;;
        esac
    done
fi

# Check for insufficient arguments
if [[ $# -lt 2 ]]; then
    show_help
fi

# Configuration
TEST_FOLDER="$1"
INPUT_SETTING="$2"

# Validate test folder name format: must be *_evl.linux
if [[ "$TEST_FOLDER" != *_evl.linux ]]; then
    echo "Error: Test folder must end with '_evl.linux' (got: $TEST_FOLDER)"
    echo ""
    echo "Example:"
    echo "  $0 resnet8_subset31_pretrained_orig_evl.linux random"
    echo ""
    echo "This is required because chip test runs on ARM Linux platform."
    exit 1
fi

# Extract model name from test folder (remove _evl.linux suffix)
TEST_NAME="${TEST_FOLDER%_evl.linux}.linux"

DEFAULT_GRAPH_PATH="mlf/executor-config/graph/default.graph"
DEFAULT_PARAMS_PATH="mlf/parameters/default.params"
DEFAULT_RUNNER_NAME="."
NPZ_FILE_PATH="scan_reg_files"

# Select executable based on DEBUG_EXE
if [[ "${DEBUG_EXE}" == "1" ]]; then
    EXEC_NAME="debug_execute_graph"
else
    EXEC_NAME="execute_graph"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scan_steps.sh"
load_env
source "$SCRIPT_DIR/power_steps.sh"

echo "=========================================="
echo "Running chip test for: $TEST_NAME"
echo "Test folder: $TEST_FOLDER"
echo "Input setting: $INPUT_SETTING"
echo "=========================================="
echo ""

# Step 1: Run test.py to generate test folder
if [[ "$SKIP_STEP1" == true ]]; then
    echo "Step 1: Skipped."
    echo ""
else
    echo "Step 1: Running main.py to generate $TEST_FOLDER..."
    echo ""
    CMD=(python main.py -p "$INPUT_SETTING" -m "$TEST_NAME" --with-patch)
    "${CMD[@]}"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to generate test folder"
        exit 1
    fi
    echo ""
fi

# Acquire chip lock before any remote transfer/execution
chip_lock_acquire "run_chiptest.sh"
trap 'chip_lock_release' EXIT

# Step 2: Transfer test folder to remote
if [[ "$SKIP_STEP2" == true ]]; then
    echo "Step 2: Skipped."
    echo ""
else
    echo "Step 2: Transferring $TEST_FOLDER to remote server..."
    echo ""
    echo "y" | ./transfer_evl.sh --host "$REMOTE_HOST" "$TEST_FOLDER"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to transfer $TEST_FOLDER"
        exit 1
    fi
    echo ""
fi

# Steps 3-5: Scan programming (shared with run_dataset_eval.sh)
scan_transfer_reg_files 3 "$SKIP_STEP3"
scan_transfer_executable 4 "$SKIP_STEP4"

scan_program_registers 5 "$SKIP_STEP5"

# Step 6: Execute on remote chip
if [[ "$SKIP_STEP6" == true ]]; then
    echo "Step 6: Skipped."
    echo ""
else
    POWER_REMOTE_BINARY="$REMOTE_BASE_PATH/eval_dir/$TEST_FOLDER/host_binary_make/build/$EXEC_NAME"
    power_prepare "$POWER_CONFIG" "$TEST_NAME" "eval_dir/$TEST_FOLDER" \
        "runner=run_chiptest" \
        "model=$TEST_NAME" \
        "board=${BOARD:-unknown}" \
        "chip=${IMCFLOW_CHIP:-unknown}" \
        "checkpoint=${CKPT:-}" || exit 1
    POWER_REMOTE_ENV="$(power_remote_environment)"
    echo "Step 6: Executing on remote chip (timeout: 300s)..."
    echo ""
    scan_ssh "source ~/.bashrc && source /home/root/.venv/bin/activate && \
                cd $REMOTE_BASE_PATH/eval_dir/$TEST_FOLDER/host_binary_make/build && ${POWER_REMOTE_ENV}timeout 300 ./$EXEC_NAME \
                eval_dir/$TEST_FOLDER $REMOTE_BASE_PATH $DEFAULT_GRAPH_PATH $DEFAULT_PARAMS_PATH $DEFAULT_RUNNER_NAME $REMOTE_BASE_PATH/$NPZ_FILE_PATH; \
                tvm_status=\$?; cd /home/root/imcflow/xilinx/petalinux-csrc && make clear_time && make warmup > /dev/null 2>&1; \
                exit \$tvm_status"
    EVAL_STATUS=$?
    power_finalize_run "$EVAL_STATUS"
    FINAL_STATUS=$?

    if [ $FINAL_STATUS -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "Chip test completed successfully!"
        echo "=========================================="
    else
        echo ""
        echo "=========================================="
        echo "Chip test failed!"
        echo "=========================================="
        exit 1
    fi
fi
