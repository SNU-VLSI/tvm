#!/bin/bash

# Script to run dataset evaluation on remote chip
# Usage: ./run_dataset_eval.sh [options] [num_samples] [remote_host]

NPZ_FILE_PATH="scan_reg_files"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scan_steps.sh"
load_env

# Paths
BINARY_DIR="host_binary_make.dataset"
GRAPH_PATH="$BINARY_DIR/build/mlf/executor-config/graph/default.graph"
PARAMS_PATH="$BINARY_DIR/build/mlf/parameters/default.params"
DATASET_DIR="dataset"
IMAGES_PATH="$DATASET_DIR/cifar10/images.npy"
LABELS_PATH="$DATASET_DIR/cifar10/labels.npy"
REMOTE_RESULT_PATH="/tmp/tvm_dataset_results.txt"
LOCAL_RESULT_DIR="eval_results"

# Function to display help message
show_help() {
    echo "Usage: $0 [options] [num_samples]"
    echo ""
    echo "Run dataset evaluation on remote chip"
    echo ""
    echo "Steps:"
    echo "  1. Build dataset binary (host_binary_make.dataset/build.sh)"
    echo "  2. Transfer host_binary_make.dataset + dataset/ to remote"
    echo "  3. Transfer scan_reg_files to remote"
    echo "  4. Transfer scan_executable to remote"
    echo "  5. Program scan registers on remote chip"
    echo "  6. Execute evaluation on remote chip"
    echo "  7. Fetch result file from remote to local"
    echo ""
    echo "Options:"
    echo "  -m, --model DIR    Evl dir name for build step (default: resnet8_subset31_pretrained_orig_evl.linux)"
    echo "  -s, --skip LIST    Comma-separated step numbers to skip (e.g., 1,2,7)"
    echo "  -i, --indices LIST Comma-separated sample indices (e.g., 0,5,10,15). Overrides num_samples."
    echo "  -q, --quiet        Quiet mode: suppress remote stdout during evaluation"
    echo "  -o, --output DIR   Local directory to save result file (default: eval_results)"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Arguments:"
    echo "  num_samples    Number of samples to evaluate (default: 20)"
    echo ""
    echo "Examples:"
    echo "  $0                       # Run all steps, evaluate 20 samples"
    echo "  $0 100                   # Run all steps, evaluate 100 samples"
    echo "  $0 -s 1,2,3,4,5 50      # Skip to evaluation only, 50 samples"
    echo "  $0 -q 100               # Quiet mode, 100 samples"
    echo "  $0 -m my_model_evl.linux 20  # Use different evl dir for build"
    echo "  $0 -i 0,5,10,15,20      # Evaluate specific sample indices"
    echo ""
    echo "Remote configuration is loaded from .env file."
    exit 0
}

# Parse options
SKIP_STEP1=false
SKIP_STEP2=false
SKIP_STEP3=false
SKIP_STEP4=false
SKIP_STEP5=false
SKIP_STEP6=false
SKIP_STEP7=false
SKIP_LIST=""
QUIET_MODE=false
SAMPLE_INDICES=""
MODEL_EVL_DIR="resnet8_subset31_pretrained_orig_evl.linux"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -m|--model)
            if [[ -z "$2" ]]; then
                echo "Error: Missing value for $1"
                exit 1
            fi
            MODEL_EVL_DIR="$2"
            shift 2
            ;;
        -s|--skip)
            if [[ -z "$2" ]]; then
                echo "Error: Missing value for $1"
                exit 1
            fi
            SKIP_LIST="$2"
            shift 2
            ;;
        -i|--indices)
            if [[ -z "$2" ]]; then
                echo "Error: Missing value for $1"
                exit 1
            fi
            SAMPLE_INDICES="$2"
            shift 2
            ;;
        -q|--quiet)
            QUIET_MODE=true
            shift
            ;;
        -o|--output)
            if [[ -z "$2" ]]; then
                echo "Error: Missing value for $1"
                exit 1
            fi
            LOCAL_RESULT_DIR="$2"
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
            4) SKIP_STEP4=true ;;
            5) SKIP_STEP5=true ;;
            6) SKIP_STEP6=true ;;
            7) SKIP_STEP7=true ;;
            "") ;;
            *)
                echo "Error: Invalid step in skip list: $step"
                exit 1
                ;;
        esac
    done
fi

# Positional arguments
NUM_SAMPLES="${1:-20}"

# Determine the samples argument for the binary
if [[ -n "$SAMPLE_INDICES" ]]; then
    SAMPLES_ARG="$SAMPLE_INDICES"
    SAMPLES_DISPLAY="indices: $SAMPLE_INDICES"
else
    SAMPLES_ARG="$NUM_SAMPLES"
    SAMPLES_DISPLAY="num_samples: $NUM_SAMPLES"
fi

echo "=========================================="
echo "Running dataset evaluation on remote chip"
echo "Model (evl dir): $MODEL_EVL_DIR"
echo "Samples: $SAMPLES_DISPLAY"
echo "Remote host: $REMOTE_HOST"
echo "Quiet mode: $QUIET_MODE"
echo "Result file (remote): $REMOTE_RESULT_PATH"
echo "Result dir (local):   $LOCAL_RESULT_DIR/"
echo "=========================================="
echo ""

# Step 1: Build dataset binary
if [[ "$SKIP_STEP1" == true ]]; then
    echo "Step 1: Skipped."
    echo ""
else
    echo "Step 1: Building dataset binary..."
    echo ""
    echo "[CMD] cd $BINARY_DIR && ./build.sh ../eval_dir/$MODEL_EVL_DIR arm"
    (cd "$BINARY_DIR" && ./build.sh "../eval_dir/$MODEL_EVL_DIR" arm)
    if [ $? -ne 0 ]; then
        echo "Error: Failed to build dataset binary"
        exit 1
    fi
    echo ""
fi

# Step 2: Transfer host_binary_make.dataset + dataset to remote
if [[ "$SKIP_STEP2" == true ]]; then
    echo "Step 2: Skipped."
    echo ""
else
    echo "Step 2: Transferring $BINARY_DIR to remote server..."
    echo ""
    echo "y" | ./transfer_evl.sh --host "$REMOTE_HOST" --path "$BINARY_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to transfer $BINARY_DIR"
        exit 1
    fi

    echo "Step 2: Transferring $DATASET_DIR to remote server..."
    echo ""
    ./dataset/transfer_dataset.sh "$REMOTE_HOST"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to transfer $DATASET_DIR"
        exit 1
    fi
    echo ""
fi

# Steps 3-5: Scan programming (shared with run_chiptest.sh)
scan_transfer_reg_files 3 "$SKIP_STEP3"
scan_transfer_executable 4 "$SKIP_STEP4"
scan_program_registers 5 "$SKIP_STEP5"

# Step 6: Execute on remote chip
if [[ "$SKIP_STEP6" == true ]]; then
    echo "Step 6: Skipped."
    echo ""
else
    echo "Step 6: Executing on remote chip..."
    echo ""
    REMOTE_CMD="cd $REMOTE_BASE_PATH && $BINARY_DIR/build/execute_graph_for_dataset \
$GRAPH_PATH \
$PARAMS_PATH \
$IMAGES_PATH \
$LABELS_PATH \
$SAMPLES_ARG \
$REMOTE_RESULT_PATH; \
cd /home/root/imcflow/xilinx/petalinux-csrc && make clear_time && make warmup > /dev/null 2>&1"
    echo "[CMD] sshpass -p \"$REMOTE_PASSWORD\" ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST \"$REMOTE_CMD\""

    if [[ "$QUIET_MODE" == true ]]; then
        echo "(Quiet mode: remote output suppressed, results saved to $REMOTE_RESULT_PATH)"
        sshpass -p "$REMOTE_PASSWORD" ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST \
            "$REMOTE_CMD; tvm_status=\$?; exit \$tvm_status" > /dev/null 2>&1
    else
        sshpass -p "$REMOTE_PASSWORD" ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST \
            "$REMOTE_CMD; tvm_status=\$?; exit \$tvm_status"
    fi

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

# Step 7: Fetch result file from remote
if [[ "$SKIP_STEP7" == true ]]; then
    echo "Step 7: Skipped."
    echo ""
else
    echo ""
    echo "Step 7: Fetching result file from remote..."
    echo ""
    mkdir -p "$LOCAL_RESULT_DIR"
    LOCAL_RESULT_FILE="$LOCAL_RESULT_DIR/dataset_results_$(date +%Y%m%d_%H%M%S).txt"
    echo "[CMD] sshpass -p \"$REMOTE_PASSWORD\" scp -P $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST:$REMOTE_RESULT_PATH $LOCAL_RESULT_FILE"
    sshpass -p "$REMOTE_PASSWORD" scp -P $REMOTE_PORT \
        "$REMOTE_USER@$REMOTE_HOST:$REMOTE_RESULT_PATH" "$LOCAL_RESULT_FILE"

    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "Result file saved to: $LOCAL_RESULT_FILE"
        echo "=========================================="
        echo ""
        echo "--- Result Summary ---"
        # Print only the FINAL RESULTS section
        sed -n '/FINAL RESULTS/,/^$/p' "$LOCAL_RESULT_FILE"
    else
        echo ""
        echo "=========================================="
        echo "Warning: Failed to fetch result file from remote"
        echo "=========================================="
    fi
fi
