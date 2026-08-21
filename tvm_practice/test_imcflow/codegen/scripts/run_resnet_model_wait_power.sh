#!/usr/bin/env bash

set -euo pipefail

log_stage() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$*"
}

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

export IMCFLOW_DIR="${IMCFLOW_DIR:-/root/project/imcflow}"
export IMCFLOW_HOME="${IMCFLOW_HOME:-$IMCFLOW_DIR}"
export IMCFLOW_ADDR="${IMCFLOW_ADDR:-0xa0000000}"
export IMCFLOW_LEN="${IMCFLOW_LEN:-0x100000}"
export INT_ACK_GEN_ADDR="${INT_ACK_GEN_ADDR:-0xa0110000}"
export INT_ACK_GEN_LEN="${INT_ACK_GEN_LEN:-0x10000}"
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
export IMCFLOW_PRE_RUN_WARMUP="${IMCFLOW_PRE_RUN_WARMUP:-1}"
export SCAN_SSH_CONNECT_TIMEOUT_SECONDS="${SCAN_SSH_CONNECT_TIMEOUT_SECONDS:-10}"
export SCAN_SSH_SERVER_ALIVE_INTERVAL_SECONDS="${SCAN_SSH_SERVER_ALIVE_INTERVAL_SECONDS:-5}"
export SCAN_SSH_SERVER_ALIVE_COUNT_MAX="${SCAN_SSH_SERVER_ALIVE_COUNT_MAX:-3}"
CHIP_RUN_TIMEOUT_SECONDS="${CHIP_RUN_TIMEOUT_SECONDS:-360}"

if [[ ! "$CHIP_RUN_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: CHIP_RUN_TIMEOUT_SECONDS must be a positive integer" >&2
    exit 1
fi

log_stage "[1/3] Compiling $MODEL with MODEL power event tags"
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

log_stage "[2/3] Building the ARM dataset executable"
(
    export DEBUG_EXE=0
    cd host_binary_make.dataset
    ./build.sh "../eval_dir/$MODEL_EVL" arm 1
)

log_stage "[3/3] Running one MODEL/wait/loop-off power capture"
echo "  pre-run warmup: $IMCFLOW_PRE_RUN_WARMUP"
echo "  chip-run timeout: ${CHIP_RUN_TIMEOUT_SECONDS}s"
echo "  SSH liveness: connect=${SCAN_SSH_CONNECT_TIMEOUT_SECONDS}s, keepalive=${SCAN_SSH_SERVER_ALIVE_INTERVAL_SECONDS}s x ${SCAN_SSH_SERVER_ALIVE_COUNT_MAX}"
set +e
CKPT="$CKPT" \
DATASET_NAME="$DATASET_NAME" \
DEBUG_EXE=0 \
CONSOLE_LOG_LEVEL="${CONSOLE_LOG_LEVEL:-INFO}" \
IMCFLOW_BUGFIX="$IMCFLOW_BUGFIX" \
timeout --signal=TERM --kill-after=20s "${CHIP_RUN_TIMEOUT_SECONDS}s" ./run_dataset_eval.sh \
    -s 1 \
    -i "$SAMPLE_INDEX" \
    -m "$MODEL_EVL" \
    --power-config "$POWER_CONFIG"
run_status=$?
set -e

if [[ $run_status -eq 124 || $run_status -eq 137 ]]; then
    echo "Error: chip run exceeded ${CHIP_RUN_TIMEOUT_SECONDS}s; stopped instead of waiting indefinitely" >&2
    exit "$run_status"
fi
if [[ $run_status -ne 0 ]]; then
    echo "Error: dataset power run failed with status $run_status" >&2
    exit "$run_status"
fi
log_stage "MODEL wait power run completed"
