#!/usr/bin/env bash
# Reproduce the CIFAR-10 ResNet-8 iter_006 B1 compile and build a dedicated
# non-debug ARM dataset binary. This stops before FPGA transfer/execution.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEGEN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TVM_ROOT="$(cd "$CODEGEN_DIR/../../.." && pwd)"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/root/project/CIM/runs/cifar.run2/resnet8_chip_noise_loop/iter_006/deploy/2026-Jul-29-08-08-57/checkpoint.pth.tar}"
COLUMN_DISABLE_CONFIG="${COLUMN_DISABLE_CONFIG:-/root/project/CIM/noise/noise_df/B1_out_0_1/N32/disabled.json}"
CHECKPOINT_ALIAS="${CHECKPOINT_ALIAS:-resnet8_chip_noise_loop_iter_006}"
MODEL="${MODEL:-resnet8_subset31_pretrained_orig}"
MODEL_EVL_DIR="${MODEL_EVL_DIR:-resnet8_subset31_pretrained_orig_evl.linux}"

# Keep each compiled model in its own host-binary directory.
MODEL_POSTFIX="${MODEL_POSTFIX:-resnet8_iter006}"
BASE_BINARY_DIR="$CODEGEN_DIR/host_binary_make.dataset"
BINARY_DIR="$CODEGEN_DIR/host_binary_make.dataset.$MODEL_POSTFIX"
EVAL_DIR="$CODEGEN_DIR/eval_dir/$MODEL_EVL_DIR"
BUILD_DIR="$BINARY_DIR/build"
BUILD_TYPE="${BUILD_TYPE:-Release}"

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint not found: $CHECKPOINT_PATH" >&2
  exit 1
fi
if [[ ! -f "$COLUMN_DISABLE_CONFIG" ]]; then
  echo "Column-disable config not found: $COLUMN_DISABLE_CONFIG" >&2
  exit 1
fi
if [[ ! -d "$BASE_BINARY_DIR" ]]; then
  echo "Base dataset-binary directory not found: $BASE_BINARY_DIR" >&2
  exit 1
fi

cd "$CODEGEN_DIR"

source "$TVM_ROOT/tvm_practice/tvm_env/bin/activate"
eval "$(direnv export bash)"
source "$CODEGEN_DIR/imcflow-linux.sh"

export PYTHONPATH="$TVM_ROOT/python:$TVM_ROOT/vta/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$TVM_ROOT/build:${LD_LIBRARY_PATH:-}"

# .envrc currently defaults DEBUG_EXE to 1. Override it after loading direnv so
# both compilation and the dataset CMake build consistently use the normal
# execute_graph_for_dataset runner.
export DEBUG_EXE=0

if [[ "${SKIP_COMPILE:-0}" != "1" ]]; then
  echo "[1/2] Compiling $MODEL from $CHECKPOINT_ALIAS"
  BOARD=B1 \
  CKPT="$CHECKPOINT_ALIAS" \
  CKPT_PATH="$CHECKPOINT_PATH" \
  ACC_MASK=15 \
  MODEL_PROFILE=resnet8 \
  DATASET_NAME=cifar10 \
  DEBUG_EXE=0 \
  python3 main.py \
    --model "$MODEL" \
    --acc-mask 15 \
    --driver-v2 \
    --ref-models transformed \
    --fixed-imce-core 0,1 \
    --num-disable-columns 32 \
    --column-disable-config "$COLUMN_DISABLE_CONFIG" \
    --random-seed 42 \
    --stop-at compile
else
  echo "[1/2] Reusing existing compile output: $EVAL_DIR"
fi

if [[ ! -f "$EVAL_DIR/lib_graph_system-lib.tar" ]]; then
  echo "Compiled MLF archive not found: $EVAL_DIR/lib_graph_system-lib.tar" >&2
  exit 1
fi
if [[ ! -d "$EVAL_DIR/build" ]]; then
  echo "Compiled host-object directory not found: $EVAL_DIR/build" >&2
  exit 1
fi

# Avoid silently mixing a previous model/build configuration into this binary.
if [[ -e "$BINARY_DIR" ]]; then
  echo "Binary directory already exists: $BINARY_DIR" >&2
  echo "Choose a new MODEL_POSTFIX (for example, resnet8_iter006_v2)." >&2
  exit 1
fi

echo "[2/2] Creating and building host_binary_make.dataset.$MODEL_POSTFIX"
mkdir -p "$BINARY_DIR"
rsync -a --exclude 'build/' "$BASE_BINARY_DIR/" "$BINARY_DIR/"
mkdir -p "$BUILD_DIR"

cmake \
  -S "$BINARY_DIR" \
  -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DISA=ARM \
  -DDEBUG_EXE=0 \
  -DMLF_TAR="$EVAL_DIR/lib_graph_system-lib.tar" \
  -DH_OBJ_PATH="$EVAL_DIR/build" \
  -DCMAKE_TOOLCHAIN_FILE="$BINARY_DIR/cmake/arm-cortex-a53.cmake"

cmake --build "$BUILD_DIR" --parallel "${JOBS:-$(nproc)}"

EXECUTABLE="$BUILD_DIR/execute_graph_for_dataset"
if [[ ! -x "$EXECUTABLE" ]]; then
  echo "Expected executable was not created: $EXECUTABLE" >&2
  exit 1
fi

echo
echo "Build complete; no FPGA transfer or host execution was performed."
echo "Binary directory: $BINARY_DIR"
echo "Executable:       $EXECUTABLE"
echo "Graph:            $BUILD_DIR/mlf/executor-config/graph/default.graph"
echo "Parameters:       $BUILD_DIR/mlf/parameters/default.params"
echo
echo "A later run can select this model-specific directory with:"
echo "  DEBUG_EXE=0 ./run_dataset_eval.sh -b host_binary_make.dataset.$MODEL_POSTFIX -m $MODEL_EVL_DIR -d cifar10 ..."
