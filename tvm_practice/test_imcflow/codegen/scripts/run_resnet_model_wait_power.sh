#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TVM_VENV="${TVM_VENV:-$SCRIPT_DIR/../../tvm_env}"
POWER_CONFIG="${IMCFLOW_POWER_CONFIG:-$SCRIPT_DIR/power_configs/model_wait_min.json}"

if [[ ! -f "$TVM_VENV/bin/activate" ]]; then
    echo "Error: TVM virtual environment not found: $TVM_VENV" >&2
    exit 1
fi
if [[ ! -f "$POWER_CONFIG" ]]; then
    echo "Error: MODEL wait power config not found: $POWER_CONFIG" >&2
    exit 1
fi

# Equivalent to the interactive `activate` alias on the master server.
source "$TVM_VENV/bin/activate"
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/imcflow-linux.sh"

export POWER_MEASUREMENT_HOST="${POWER_MEASUREMENT_HOST:-147.46.117.49}"
export POWER_MEASUREMENT_PORT="${POWER_MEASUREMENT_PORT:-9910}"

CKPT="${CKPT:-chip3_run4_ft_e80_iter003}"
MODEL="${MODEL:-resnet8_subset31_pretrained_orig}"
MODEL_EVL="${MODEL_EVL:-resnet8_subset31_pretrained_orig_evl.linux.bugfixoff}"
MODEL_PROFILE="${MODEL_PROFILE:-resnet8}"
DATASET_NAME="${DATASET_NAME:-cifar10}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
IMCFLOW_BUGFIX="${IMCFLOW_BUGFIX:-off}"
ACC_MASK="${ACC_MASK:-1}"
IMCFLOW_NO_PERKERNEL_WARMUP="${IMCFLOW_NO_PERKERNEL_WARMUP:-0}"
IMCFLOW_MMIO_BARRIER="${IMCFLOW_MMIO_BARRIER:-100}"

echo "[1/3] Compiling $MODEL with MODEL power event tags"
CKPT="$CKPT" \
MODEL_PROFILE="$MODEL_PROFILE" \
DATASET_NAME="$DATASET_NAME" \
IMCFLOW_BUGFIX="$IMCFLOW_BUGFIX" \
ACC_MASK="$ACC_MASK" \
IMCFLOW_NO_PERKERNEL_WARMUP="$IMCFLOW_NO_PERKERNEL_WARMUP" \
IMCFLOW_MMIO_BARRIER="$IMCFLOW_MMIO_BARRIER" \
python3 -u main.py \
    --model "$MODEL" \
    --acc-mask "$ACC_MASK" \
    --ref-models transformed \
    --random-seed 42 \
    --dataset "$DATASET_NAME" \
    --sample "$SAMPLE_INDEX" \
    --stop-at compile

echo "[2/3] Building the ARM dataset executable"
(
    export DEBUG_EXE=0
    cd host_binary_make.dataset
    ./build.sh "../eval_dir/$MODEL_EVL" arm 1
)

echo "[3/3] Running one MODEL/wait/loop-off power capture"
CKPT="$CKPT" \
DATASET_NAME="$DATASET_NAME" \
DEBUG_EXE=0 \
CONSOLE_LOG_LEVEL="${CONSOLE_LOG_LEVEL:-INFO}" \
IMCFLOW_BUGFIX="$IMCFLOW_BUGFIX" \
./run_dataset_eval.sh \
    -s 1 \
    -i "$SAMPLE_INDEX" \
    -m "$MODEL_EVL" \
    --power-config "$POWER_CONFIG"

