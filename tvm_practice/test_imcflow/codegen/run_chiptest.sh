#!/bin/bash

# Script to run chip test: generate test, transfer files and execute on remote chip
# Usage: ./run_chiptest.sh <test_name> <input_setting>

# Function to display help message
show_help() {
    echo "Usage: $0 <test_name> <input_setting>"
    echo ""
    echo "Automated chip test workflow that:"
    echo "  1. Runs test.py to generate test folder"
    echo "  2. Transfers test folder to remote server"
    echo "  3. Transfers host_binary_make to remote server"
    echo "  4. Executes test on remote chip"
    echo ""
    echo "Arguments:"
    echo "  test_name       Name of the test (e.g., 'one_relu', 'one_conv_small')"
    echo "  input_setting   Input setting for the test (e.g., 'ones', 'random', 'incremental')"
    echo ""
    echo "Examples:"
    echo "  $0 one_relu ones"
    echo "  $0 one_conv_small random"
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

# Check for help flag or insufficient arguments
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]] || [[ $# -lt 2 ]]; then
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
DEFAULT_RUNNER_NAME="chiptest"
NPZ_FILE_PATH="scan_reg_files"

echo "=========================================="
echo "Running chip test for: $TEST_NAME"
echo "Test folder: $TEST_FOLDER"
echo "Input setting: $INPUT_SETTING"
echo "=========================================="
echo ""

# Step 1: Run test.py to generate test folder
echo "Step 1: Running test.py to generate $TEST_FOLDER..."
echo ""
python test.py -k "$TEST_NAME and $INPUT_SETTING" -s
if [ $? -ne 0 ]; then
    echo "Error: Failed to generate test folder"
    exit 1
fi
echo ""

# Step 2: Transfer test folder to remote
echo "Step 2: Transferring $TEST_FOLDER to remote server..."
echo ""
echo "y" | ./transfer_evl.sh "$TEST_FOLDER"
if [ $? -ne 0 ]; then
    echo "Error: Failed to transfer $TEST_FOLDER"
    exit 1
fi
echo ""

## Step 3: Transfer utils to remote
#echo "Step 3: Transferring host_binary_make to remote server..."
#echo ""
#echo "y" | ./transfer_evl.sh --path utils/$NPZ_FILE_PATH
#if [ $? -ne 0 ]; then
#    echo "Error: Failed to transfer utils"
#    exit 1
#fi
#echo ""

# Step 4: Execute on remote chip
echo "Step 4: Executing on remote chip..."
echo ""
sshpass -p "$REMOTE_PASSWORD" ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST \
           "cd $REMOTE_BASE_PATH/$TEST_FOLDER/host_binary_make/build && ./tvm_host_runner \
            $TEST_FOLDER $REMOTE_BASE_PATH $DEFAULT_GRAPH_PATH $DEFAULT_PARAMS_PATH $DEFAULT_RUNNER_NAME $REMOTE_BASE_PATH/$NPZ_FILE_PATH"

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
