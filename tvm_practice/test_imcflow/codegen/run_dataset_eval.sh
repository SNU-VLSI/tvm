#!/bin/bash

# Script to run dataset evaluation on remote chip
# Usage: ./run_dataset_eval.sh [options] [num_samples] [remote_host]

NPZ_FILE_PATH="scan_reg_files"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scan_steps.sh"
load_env
source "$SCRIPT_DIR/power_steps.sh"

# Paths (BINARY_DIR may be overridden by --binary-dir option)
BINARY_DIR="host_binary_make.dataset"
GRAPH_PATH="$BINARY_DIR/build/mlf/executor-config/graph/default.graph"
PARAMS_PATH="$BINARY_DIR/build/mlf/parameters/default.params"

# Select executable based on DEBUG_EXE
if [[ "${DEBUG_EXE}" == "1" ]]; then
    DATASET_EXEC_NAME="debug_execute_graph_for_dataset"
else
    DATASET_EXEC_NAME="execute_graph_for_dataset"
fi
DATASET_DIR="dataset"
DATASET_NAME="${DATASET_NAME:-cifar10}"
IMAGES_PATH="$DATASET_DIR/$DATASET_NAME/images.npy"
LABELS_PATH="$DATASET_DIR/$DATASET_NAME/labels.npy"
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
    echo "  -b, --binary-dir DIR  Binary directory (default: host_binary_make.dataset)"
    echo "  -m, --model DIR       Evl dir name for build step (default: resnet8_subset31_pretrained_orig_evl.linux)"
    echo "  -d, --dataset NAME    Dataset subdir under dataset/ (default: cifar10)"
    echo "  -s, --skip LIST       Comma-separated step numbers to skip (e.g., 1,2,7)"
    echo "  -i, --indices LIST    Comma-separated sample indices (e.g., 0,5,10,15). Overrides num_samples."
    echo "  -l, --log-level LVL   Console log level: DEBUG (verbose, default) or INFO (progress bar)"
    echo "  -q, --quiet           Quiet mode: suppress remote stdout during evaluation"
    echo "  -o, --output DIR      Local directory to save result file (default: eval_results)"
    echo "  --power-config FILE   Enable tagged whole-run power measurement"
    echo "  -h, --help            Show this help message"
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
    echo "  $0 -b host_binary_make.dataset.v1 20  # Use tagged binary directory"
    echo "  $0 -l INFO 100               # Progress bar only (less verbose)"
    echo "  CONSOLE_LOG_LEVEL=INFO $0 100 # Same via env var"
    echo ""
    echo "Environment variables:"
    echo "  CKPT           Checkpoint alias for debug dump subdir (debugging/fpga/<CKPT>)"
    echo "  DEBUG_EXE=1    Enable debug node dump fetch from remote"
    echo "  PRESERVE_DEBUG_DUMPS=1  Keep existing local sample_* and remote debug_nodes after fetch"
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
CONSOLE_LOG_LEVEL="${CONSOLE_LOG_LEVEL:-DEBUG}"
MODEL_EVL_DIR="resnet8_subset31_pretrained_orig_evl.linux"
CKPT="${CKPT:-}"
POWER_CONFIG="${IMCFLOW_POWER_CONFIG:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -b|--binary-dir)
            if [[ -z "$2" ]]; then
                echo "Error: Missing value for $1"
                exit 1
            fi
            BINARY_DIR="$2"
            shift 2
            ;;
        -m|--model)
            if [[ -z "$2" ]]; then
                echo "Error: Missing value for $1"
                exit 1
            fi
            MODEL_EVL_DIR="$2"
            shift 2
            ;;
        -d|--dataset)
            if [[ -z "$2" ]]; then
                echo "Error: Missing value for $1"
                exit 1
            fi
            DATASET_NAME="$2"
            IMAGES_PATH="$DATASET_DIR/$DATASET_NAME/images.npy"
            LABELS_PATH="$DATASET_DIR/$DATASET_NAME/labels.npy"
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
        -l|--log-level)
            if [[ -z "$2" ]]; then
                echo "Error: Missing value for $1"
                exit 1
            fi
            CONSOLE_LOG_LEVEL="$(echo "$2" | tr '[:lower:]' '[:upper:]')"
            if [[ "$CONSOLE_LOG_LEVEL" != "DEBUG" && "$CONSOLE_LOG_LEVEL" != "INFO" ]]; then
                echo "Error: Invalid log level '$2'. Must be DEBUG or INFO."
                exit 1
            fi
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
        --power-config)
            if [[ -z "$2" ]]; then
                echo "Error: Missing value for $1"
                exit 1
            fi
            POWER_CONFIG="$2"
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

# Recompute MLF paths against the (possibly -b overridden) BINARY_DIR. These were
# initialised at load time from the default BINARY_DIR, before option parsing, so
# without this a `-b host_binary_make.dataset.<model>` run would execute that dir's
# binary against the DEFAULT dir's graph/params (wrong MLF -> wrong/garbage model).
GRAPH_PATH="$BINARY_DIR/build/mlf/executor-config/graph/default.graph"
PARAMS_PATH="$BINARY_DIR/build/mlf/parameters/default.params"

# Positional arguments
NUM_SAMPLES="${1:-20}"

# Extract TAG from BINARY_DIR for result filename
# e.g., host_binary_make.dataset.v1 -> TAG="v1"
DEFAULT_BINARY_DIR="host_binary_make.dataset"
if [[ "$BINARY_DIR" == "${DEFAULT_BINARY_DIR}."* ]]; then
    BINARY_TAG="${BINARY_DIR#${DEFAULT_BINARY_DIR}.}"
else
    BINARY_TAG=""
fi

# Determine the samples argument for the binary
if [[ -n "$SAMPLE_INDICES" ]]; then
    # Ensure comma is present so C parser detects indices mode (vs num_samples)
    if [[ "$SAMPLE_INDICES" != *","* ]]; then
        SAMPLES_ARG="${SAMPLE_INDICES},"
    else
        SAMPLES_ARG="$SAMPLE_INDICES"
    fi
    SAMPLES_DISPLAY="indices: $SAMPLE_INDICES"
else
    SAMPLES_ARG="$NUM_SAMPLES"
    SAMPLES_DISPLAY="num_samples: $NUM_SAMPLES"
fi

# Stage only the samples this run evaluates, instead of shipping the whole
# dataset dir to the chip. transfer_dataset.sh scp's -r the entire dataset/,
# which for a deep model (VWW: 1.2GB / 10961 samples) sends the full array every
# run even though the loop touches ~100 of them. We slice images.npy/labels.npy
# down to the requested rows (num_samples -> first K; -i indices -> those rows),
# 0-based remapped into dataset/<name>/_staged, and point the run at that subset.
# sample_map.json there records staged->original rows. Set NO_STAGE=1 to keep the
# old whole-dataset behavior (e.g. to evaluate the full set on-chip).
STAGED_DIR="$DATASET_DIR/$DATASET_NAME/_staged"
if [[ "${NO_STAGE:-0}" != "1" ]]; then
    if [[ -n "$SAMPLE_INDICES" ]]; then
        STAGE_SEL=(--indices "$SAMPLE_INDICES")
    else
        STAGE_SEL=(--num-samples "$NUM_SAMPLES")
    fi
    echo "Staging subset -> $STAGED_DIR ($SAMPLES_DISPLAY)"
    STAGED_COUNT="$(python3 "$DATASET_DIR/stage_subset.py" \
        --src-dir "$DATASET_DIR/$DATASET_NAME" \
        --out-dir "$STAGED_DIR" \
        "${STAGE_SEL[@]}")"
    if [[ $? -ne 0 || -z "$STAGED_COUNT" ]]; then
        echo "Error: failed to stage sample subset"
        exit 1
    fi
    # Staged file holds the selected rows as 0..K-1, so the C binary runs in
    # num_samples mode over the whole staged file (its absolute rows == 0..K-1).
    IMAGES_PATH="$STAGED_DIR/images.npy"
    LABELS_PATH="$STAGED_DIR/labels.npy"
    SAMPLES_ARG="$STAGED_COUNT"
    SAMPLES_DISPLAY="$SAMPLES_DISPLAY (staged $STAGED_COUNT rows)"
fi

echo "=========================================="
echo "Running dataset evaluation on remote chip"
echo "Binary dir:       $BINARY_DIR"
echo "Model (evl dir): $MODEL_EVL_DIR"
echo "Dataset:          $DATASET_NAME"
echo "Samples: $SAMPLES_DISPLAY"
echo "Remote host: $REMOTE_HOST"
echo "Ckpt alias:       ${CKPT:-<none>}"
echo "Log level: $CONSOLE_LOG_LEVEL"
echo "Quiet mode: $QUIET_MODE"
echo "Preserve debug dumps: ${PRESERVE_DEBUG_DUMPS:-0}"
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

# Acquire chip lock before any remote transfer/execution
chip_lock_acquire "run_dataset_eval.sh"
trap 'chip_lock_release' EXIT

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

    if [[ "${NO_STAGE:-0}" != "1" ]]; then
        # Ship only the staged subset (dataset/<name>/_staged), not the whole
        # dataset/ dir. Keeps a deep model's full 1.2GB array off the wire.
        echo "Step 2: Transferring staged subset ($DATASET_NAME/_staged) to remote server..."
        echo ""
        ./dataset/transfer_dataset.sh "$REMOTE_HOST" "$DATASET_NAME/_staged"
    else
        echo "Step 2: Transferring $DATASET_DIR to remote server..."
        echo ""
        ./dataset/transfer_dataset.sh "$REMOTE_HOST"
    fi
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
    power_prepare "$POWER_CONFIG" "$MODEL_EVL_DIR" "eval_dir/$MODEL_EVL_DIR" \
        "runner=run_dataset_eval" \
        "model=$MODEL_EVL_DIR" \
        "dataset=$DATASET_NAME" \
        "sample_selection=$SAMPLES_DISPLAY" \
        "board=${BOARD:-unknown}" \
        "chip=${IMCFLOW_CHIP:-unknown}" \
        "checkpoint=${CKPT:-}" || exit 1
    POWER_REMOTE_ENV="$(power_remote_environment)"
    echo "Step 6: Executing on remote chip..."
    echo ""
    # Write per-node debug dumps + heartbeat to tmpfs (RAM), NOT the SD card.
    # A deep model (VWW: ~900 tiny .npy per sample × 100 samples ≈ 90k files)
    # of create/unlink churn on the SD ext4 was corrupting its directory htree
    # ("Bad message"). tmpfs absorbs it with zero SD wear; fetched + freed below.
    # Overridable via CHIP_DEBUG_DUMP_DIR / CHIP_HEARTBEAT_PATH.
    CHIP_DEBUG_DUMP_DIR="${CHIP_DEBUG_DUMP_DIR:-/var/volatile/debug_nodes}"
    CHIP_HEARTBEAT_PATH="${CHIP_HEARTBEAT_PATH:-/var/volatile/imcflow_chip_heartbeat.txt}"
    # Pin the eval binary to a single, otherwise-idle CPU core (taskset). The ZynqMP
    # A53 has 4 cores; CPU0 carries the eth0/mmc/i2c IRQ handlers. When the eval was
    # left to migrate across cores, bouncing onto CPU0 mid-way through the tight
    # per-kernel MMIO + UIO-interrupt handshake raced with those IRQ handlers and
    # eventually WEDGED the whole SoC (SSH-dead, RAM/tmpfs free) around sample ~46 of
    # a deep-model (VWW, 47 kernels) run. Pinning the eval to an idle core removes the
    # migration + CPU0 contention: verified a full 100-sample run then completes with
    # 0 failures where every un-pinned run died by ~46-52. Overridable via CHIP_EVAL_CPU.
    CHIP_EVAL_CPU="${CHIP_EVAL_CPU:-3}"
    # Optional per-sample timing breakdown (hw vs cpu vs setin). Passed through to
    # the eval binary which emits [TIMING] lines to stderr + a summary to the
    # result file. Off unless IMCFLOW_TIMING is set in this driver's environment.
    TIMING_ENV=""
    if [ -n "${IMCFLOW_TIMING:-}" ]; then
        TIMING_ENV="IMCFLOW_TIMING=$IMCFLOW_TIMING "
    fi
    REMOTE_CMD="cd $REMOTE_BASE_PATH && ${POWER_REMOTE_ENV}${TIMING_ENV}IMCFLOW_DEBUG_DUMP_DIR=$CHIP_DEBUG_DUMP_DIR IMCFLOW_HEARTBEAT_PATH=$CHIP_HEARTBEAT_PATH taskset -c $CHIP_EVAL_CPU $BINARY_DIR/build/$DATASET_EXEC_NAME \
$GRAPH_PATH \
$PARAMS_PATH \
$IMAGES_PATH \
$LABELS_PATH \
$SAMPLES_ARG \
$REMOTE_RESULT_PATH; \
tvm_status=\$?; cd /home/root/imcflow/xilinx/petalinux-csrc && make clear_time && make warmup > /dev/null 2>&1; \
exit \$tvm_status"
    echo "[CMD] $(scan_ssh_display) \"$REMOTE_CMD\""

    if [[ "$QUIET_MODE" == true ]]; then
        echo "(Quiet mode: remote output suppressed, results saved to $REMOTE_RESULT_PATH)"
        scan_ssh "$REMOTE_CMD; tvm_status=\$?; exit \$tvm_status" > /dev/null 2>&1
        EVAL_STATUS=$?
    elif [[ "$CONSOLE_LOG_LEVEL" == "INFO" ]]; then
        echo "(INFO mode: showing progress bar only)"
        scan_ssh "$REMOTE_CMD; tvm_status=\$?; exit \$tvm_status" 2>&1 | "$SCRIPT_DIR/scripts/filter_eval_output.sh"
        EVAL_STATUS=${PIPESTATUS[0]}
    else
        scan_ssh "$REMOTE_CMD; tvm_status=\$?; exit \$tvm_status"
        EVAL_STATUS=$?
    fi
    power_finalize_run "$EVAL_STATUS"
    FINAL_STATUS=$?
    if [ $FINAL_STATUS -eq 0 ]; then
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

# Step 6.5: Fetch debug_nodes from remote and cleanup (only when DEBUG_EXE=1)
if [[ "${DEBUG_EXE}" == "1" ]] && [[ "$SKIP_STEP6" != true ]]; then
    # Dumps now live in tmpfs on the chip (see Step 6). Fetch from there; the
    # cleanup below then frees that RAM. Keep in sync with CHIP_DEBUG_DUMP_DIR.
    REMOTE_DEBUG_DIR="${CHIP_DEBUG_DUMP_DIR:-/var/volatile/debug_nodes}"
    if [[ -n "$CKPT" ]]; then
        LOCAL_DEBUG_DIR="debugging/fpga/$CKPT"
    else
        LOCAL_DEBUG_DIR="debugging/fpga"
    fi

    echo ""
    echo "Step 6.5: Fetching debug_nodes from remote (DEBUG_EXE=1)..."
    echo ""

    # Clean only sample_* from target directory, preserving run_* from repeat_dataset_eval.
    mkdir -p "$LOCAL_DEBUG_DIR"
    if [[ "${PRESERVE_DEBUG_DUMPS:-0}" == "1" ]]; then
        echo "Preserving existing local sample_* in $LOCAL_DEBUG_DIR/ ..."
    elif compgen -G "$LOCAL_DEBUG_DIR/sample_*" > /dev/null; then
        echo "Removing stale sample_* in $LOCAL_DEBUG_DIR/ ..."
        rm -rf "$LOCAL_DEBUG_DIR"/sample_*
    fi

    # scp -r all sample folders from FPGA to host
    echo "[CMD] scp -r remote:$REMOTE_DEBUG_DIR/* -> $LOCAL_DEBUG_DIR/"
    scan_scp -r "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DEBUG_DIR/*" "$LOCAL_DEBUG_DIR/"

    if [ $? -eq 0 ]; then
        echo "✅ debug_nodes fetched to $LOCAL_DEBUG_DIR/"

        # ALWAYS clean the remote dumps: they now live in tmpfs (RAM), so leaving
        # them accumulates across runs and can exhaust chip memory. PRESERVE_DEBUG_DUMPS
        # only governs the LOCAL copy (already fetched above); it must NOT keep the
        # tmpfs copy alive.
        echo "Freeing remote tmpfs debug_nodes..."
        scan_ssh "rm -rf $REMOTE_DEBUG_DIR"
        echo "✅ Remote tmpfs debug_nodes freed"
    else
        echo "⚠️  Failed to fetch debug_nodes (remote may not have any)"
    fi
    echo ""
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
    if [[ -n "$BINARY_TAG" ]]; then
        LOCAL_RESULT_FILE="$LOCAL_RESULT_DIR/dataset_results_${BINARY_TAG}_$(date +%Y%m%d_%H%M%S).txt"
    else
        LOCAL_RESULT_FILE="$LOCAL_RESULT_DIR/dataset_results_$(date +%Y%m%d_%H%M%S).txt"
    fi
    echo "[CMD] $(scan_scp_display) $REMOTE_USER@$REMOTE_HOST:$REMOTE_RESULT_PATH $LOCAL_RESULT_FILE"
    scan_scp "$REMOTE_USER@$REMOTE_HOST:$REMOTE_RESULT_PATH" "$LOCAL_RESULT_FILE"

    if [ $? -eq 0 ]; then
        # Prepend build configuration metadata to result file
        METADATA_FILE="eval_dir/${MODEL_EVL_DIR}/build_metadata.json"
        SCAN_REG_LINK="scan_gen/${NPZ_FILE_PATH}"

        if [[ -f "$METADATA_FILE" ]]; then
            # Resolve scan_reg_files symlink target
            SCAN_REG_TARGET=""
            if [[ -L "$SCAN_REG_LINK" ]]; then
                SCAN_REG_TARGET="$(readlink -f "$SCAN_REG_LINK")"
            elif [[ -d "$SCAN_REG_LINK" ]]; then
                SCAN_REG_TARGET="$(cd "$SCAN_REG_LINK" && pwd)"
            fi

            {
                echo "========================================"
                echo "  BUILD CONFIGURATION"
                echo "========================================"
                python3 -c "
import json, sys
m = json.load(open(sys.argv[1]))
for k, v in m.items():
    if v is None:
        v = '-'
    print(f'{k:.<30s} {v}')
" "$METADATA_FILE"
                if [[ -n "$SCAN_REG_TARGET" ]]; then
                    printf "%-30s %s\n" "scan_reg_files................" "$SCAN_REG_TARGET"
                fi
                echo "========================================"
                echo ""
                cat "$LOCAL_RESULT_FILE"
            } > "${LOCAL_RESULT_FILE}.tmp"
            mv "${LOCAL_RESULT_FILE}.tmp" "$LOCAL_RESULT_FILE"
        fi

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
