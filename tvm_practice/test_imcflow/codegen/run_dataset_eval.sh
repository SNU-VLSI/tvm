#!/bin/bash

# Script to run dataset evaluation on remote chip
# Usage: ./run_dataset_eval.sh [options] [num_samples]

# Default configuration
REMOTE_HOST="147.46.117.99"
REMOTE_PORT="1326"
REMOTE_USER="root"
REMOTE_PASSWORD="root"
REMOTE_BASE_PATH="/home/root/tvm/tvm_practice/test_imcflow/codegen"

# Paths
BINARY_DIR="host_binary_make.dataset"
GRAPH_PATH="$BINARY_DIR/build/mlf/executor-config/graph/default.graph"
PARAMS_PATH="$BINARY_DIR/build/mlf/parameters/default.params"
DATASET_DIR="dataset"
IMAGES_PATH="$DATASET_DIR/cifar10/images.npy"
LABELS_PATH="$DATASET_DIR/cifar10/labels.npy"

# Function to display help message
show_help() {
    echo "Usage: $0 [options] [num_samples]"
    echo ""
    echo "Run dataset evaluation on remote chip"
    echo ""
    echo "Steps:"
    echo "  1. Transfer host_binary_make.dataset to remote"
    echo "  2. Transfer dataset to remote"
    echo "  3. Execute evaluation on remote chip"
    echo ""
    echo "Options:"
    echo "  -s, --skip LIST  Comma-separated step numbers to skip (e.g., 1,2)"
    echo "  -h, --help       Show this help message"
    echo ""
    echo "Arguments:"
    echo "  num_samples    Number of samples to evaluate (default: 20)"
    echo ""
    echo "Examples:"
    echo "  $0              # Run all steps, evaluate 20 samples"
    echo "  $0 100          # Run all steps, evaluate 100 samples"
    echo "  $0 -s 1,2 50    # Skip transfer steps, evaluate 50 samples"
    echo ""
    echo "Remote Configuration:"
    echo "  Host: $REMOTE_HOST"
    echo "  Port: $REMOTE_PORT"
    echo "  User: $REMOTE_USER"
    exit 0
}

# Parse options
SKIP_STEP1=false
SKIP_STEP2=false
SKIP_STEP3=false
SKIP_LIST=""

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
        --)
            shift
            break
            ;;
        -*)
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
            "") ;;
            *)
                echo "Error: Invalid step in skip list: $step"
                exit 1
                ;;
        esac
    done
fi

# Default number of samples
NUM_SAMPLES="${1:-20}"

echo "=========================================="
echo "Running dataset evaluation on remote chip"
echo "Number of samples: $NUM_SAMPLES"
echo "=========================================="
echo ""

# Step 1: Transfer host_binary_make.dataset to remote
if [[ "$SKIP_STEP1" == true ]]; then
    echo "Step 1: Skipped."
    echo ""
else
    echo "Step 1: Transferring $BINARY_DIR to remote server..."
    echo ""
    echo "[CMD] echo \"y\" | ./transfer_evl.sh --path \"$BINARY_DIR\""
    echo "y" | ./transfer_evl.sh --path "$BINARY_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to transfer $BINARY_DIR"
        exit 1
    fi
    echo ""
fi

# Step 2: Transfer dataset to remote
if [[ "$SKIP_STEP2" == true ]]; then
    echo "Step 2: Skipped."
    echo ""
else
    echo "Step 2: Transferring $DATASET_DIR to remote server..."
    echo ""
    echo "[CMD] ./dataset/transfer_dataset.sh"
    ./dataset/transfer_dataset.sh
    if [ $? -ne 0 ]; then
        echo "Error: Failed to transfer $DATASET_DIR"
        exit 1
    fi
    echo ""
fi

# Step 3: Execute on remote chip
if [[ "$SKIP_STEP3" == true ]]; then
    echo "Step 3: Skipped."
    echo ""
else
    echo "Step 3: Executing on remote chip..."
    echo ""
    REMOTE_CMD="cd $REMOTE_BASE_PATH && $BINARY_DIR/build/execute_graph_for_dataset \
$GRAPH_PATH \
$PARAMS_PATH \
$IMAGES_PATH \
$LABELS_PATH \
$NUM_SAMPLES; \
cd /home/root/imcflow/xilinx/petalinux-csrc && make clear_time && make warmup > /dev/null 2>&1"
    echo "[CMD] sshpass -p \"$REMOTE_PASSWORD\" ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST \"$REMOTE_CMD\""
    sshpass -p "$REMOTE_PASSWORD" ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST \
        "$REMOTE_CMD; tvm_status=\$?; exit \$tvm_status"

    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "Dataset evaluation completed successfully!"
        echo "=========================================="
    else
        echo ""
        echo "=========================================="
        echo "Dataset evaluation failed!"
        echo "=========================================="
        exit 1
    fi
fi
