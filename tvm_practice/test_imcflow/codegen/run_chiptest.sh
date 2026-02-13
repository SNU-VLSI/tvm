#!/bin/bash

# Script to run chip test: generate test, transfer files and execute on remote chip
# Usage: ./run_chiptest.sh [options] <test_name> <input_setting>

# Function to display help message
show_help() {
    echo "Usage: $0 [options] <test_name> <input_setting>"
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
    echo "  -r              Pass -r to main.py in step 1"
    echo "  -s, --skip LIST  Comma-separated step numbers to skip (e.g., 1,3)"
    echo "  -h, --help       Show this help message"
    echo ""
    echo "Arguments:"
    echo "  test_name       Name of the test (e.g., 'one_relu', 'one_conv_small')"
    echo "  input_setting   Input setting for the test (e.g., 'ones', 'random', 'incremental')"
    echo ""
    echo "Examples:"
    echo "  $0 one_relu ones"
    echo "  $0 one_conv_small random"
    echo "  $0 -r one_conv_small random"
    echo "  $0 -s 2,3 one_conv_small random"
    echo "  $0 -h                                    # Show this help message"
    echo ""
    echo "Remote Configuration:"
    echo "  Host: 147.46.117.99"
    echo "  Port: 1326"
    echo "  User: root"
    echo "  Path: /home/root/tvm/tvm_practice/test_imcflow/codegen"
    echo ""
    echo "Note: Host binary should be built before running this script"
    exit 0
}

# Parse options
RERUN_FLAG=false
SKIP_STEP1=false
SKIP_STEP2=false
SKIP_STEP3=false
SKIP_STEP4=false
SKIP_STEP5=false
SKIP_STEP6=false
SKIP_LIST=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -r)
            RERUN_FLAG=true
            shift
            ;;
        -s|--skip)
            if [[ -z "$2" ]]; then
                echo "Error: Missing value for $1"
                exit 1
            fi
            SKIP_LIST="$2"
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
TEST_NAME="$1"
INPUT_SETTING="$2"
TEST_FOLDER="${TEST_NAME}_evl"

REMOTE_HOST="147.46.117.99"
REMOTE_PORT="1326"
REMOTE_USER="root"
REMOTE_PASSWORD="root"
REMOTE_BASE_PATH="/home/root/tvm/tvm_practice/test_imcflow/codegen"
DEFAULT_GRAPH_PATH="mlf/executor-config/graph/default.graph"
DEFAULT_PARAMS_PATH="mlf/parameters/default.params"
DEFAULT_RUNNER_NAME="."
NPZ_FILE_PATH="scan_reg_files"

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
    CMD=(python main.py -p "$INPUT_SETTING" -m "$TEST_NAME")
    if [[ "$RERUN_FLAG" == true ]]; then
        CMD+=("-r")
    fi
    "${CMD[@]}"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to generate test folder"
        exit 1
    fi
    echo ""
fi

# Step 2: Transfer test folder to remote
if [[ "$SKIP_STEP2" == true ]]; then
    echo "Step 2: Skipped."
    echo ""
else
    echo "Step 2: Transferring $TEST_FOLDER to remote server..."
    echo ""
    echo "y" | ./transfer_evl.sh "$TEST_FOLDER"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to transfer $TEST_FOLDER"
        exit 1
    fi
    echo ""
fi

# Step 3: Transfer scan_reg_files to remote
if [[ "$SKIP_STEP3" == true ]]; then
    echo "Step 3: Skipped."
    echo ""
else
    echo "Step 3: Transferring scan_reg_files to remote server..."
    echo "y" | ./transfer_evl.sh --path "scan_gen/$NPZ_FILE_PATH"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to transfer $NPZ_FILE_PATH"
        exit 1
    fi
    echo ""
fi

# Step 4: Transfer scan_executable to remote
if [[ "$SKIP_STEP4" == true ]]; then
    echo "Step 4: Skipped."
    echo ""
else
    echo "Step 4: Transferring program_scan_reg to remote server..."
    echo "y" | ./transfer_evl.sh --path "scan_gen/scan_executable_make"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to transfer scan_executable_make"
        exit 1
    fi
    echo ""
fi

# Step 5: Execute scan program on remote chip
if [[ "$SKIP_STEP5" == true ]]; then
    echo "Step 5: Skipped."
    echo ""
else
    echo "Step 5: Executing scan program on remote chip (timeout: 0.5s)..."
    echo ""
    sshpass -p "$REMOTE_PASSWORD" ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST \
               "cd $REMOTE_BASE_PATH/scan_executable_make/build && timeout -s INT 0.5s ./program_scan_reg \
                $REMOTE_BASE_PATH/$NPZ_FILE_PATH; \
                cd /home/root/imcflow/xilinx/petalinux-csrc && make clear_time && make warmup > /dev/null 2>&1 && \
                tvm_status=\$?; exit \$tvm_status"

    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "SCAN programming completed successfully!"
        echo "=========================================="
    else
        echo ""
        echo "=========================================="
        echo "SCAN programming failed!"
        echo "=========================================="
        exit 1
    fi
fi

# Step 6: Execute on remote chip
if [[ "$SKIP_STEP6" == true ]]; then
    echo "Step 6: Skipped."
    echo ""
else
    echo "Step 6: Executing on remote chip (timeout: 0.5s)..."
    echo ""
    sshpass -p "$REMOTE_PASSWORD" ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST \
               "cd $REMOTE_BASE_PATH/$TEST_FOLDER/host_binary_make/build && timeout -s INT 0.5s ./tvm_host_runner \
                $TEST_FOLDER $REMOTE_BASE_PATH $DEFAULT_GRAPH_PATH $DEFAULT_PARAMS_PATH $DEFAULT_RUNNER_NAME $REMOTE_BASE_PATH/$NPZ_FILE_PATH; \
                cd /home/root/imcflow/xilinx/petalinux-csrc && make clear_time && make warmup > /dev/null 2>&1 && \
                tvm_status=\$?; exit \$tvm_status"

    if [ $? -eq 0 ]; then
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