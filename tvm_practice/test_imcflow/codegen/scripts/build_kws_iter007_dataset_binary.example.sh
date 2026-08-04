#!/usr/bin/env bash
# Reproduce the KWS DS-CNN iter_007 B1 compile and build a model-specific
# non-debug ARM dataset binary. This stops after the local binary is built;
# it does not transfer files to the FPGA or execute the host binary.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEGEN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TVM_ROOT="$(cd "$CODEGEN_DIR/../../.." && pwd)"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/root/project/CIM/runs/kws.run5/kws_dscnn_chip_noise_loop/iter_007/deploy/2026-Aug-04-08-06-49/checkpoint.pth.tar}"
COLUMN_DISABLE_CONFIG="${COLUMN_DISABLE_CONFIG:-/root/project/CIM/noise/noise_df/B1_out_refine_fixed_full_v1_partial/N32/disabled.json}"
CHECKPOINT_ALIAS="${CHECKPOINT_ALIAS:-kws_dscnn_chip_noise_loop_iter_007}"
MODEL="${MODEL:-ds_cnn_full_pretrained}"
MODEL_EVL_DIR="${MODEL_EVL_DIR:-ds_cnn_full_pretrained_evl.linux}"

# Per-model binary convention used by the demo and run_dataset_eval.sh -b.
MODEL_POSTFIX="${MODEL_POSTFIX:-kws}"
BASE_BINARY_DIR="$CODEGEN_DIR/host_binary_make.dataset"
BINARY_DIR="$CODEGEN_DIR/host_binary_make.dataset.$MODEL_POSTFIX"
EVAL_DIR="$CODEGEN_DIR/eval_dir/$MODEL_EVL_DIR"
BUILD_DIR="$BINARY_DIR/build"

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

if [[ "${SKIP_COMPILE:-0}" != "1" ]]; then
  echo "[1/2] Compiling $MODEL from $CHECKPOINT_ALIAS"
  BOARD=B1 \
  CKPT="$CHECKPOINT_ALIAS" \
  CKPT_PATH="$CHECKPOINT_PATH" \
  ACC_MASK=1 \
  MODEL_PROFILE=kws_dscnn \
  DATASET_NAME=kws_sc \
  python3 main.py \
    --model "$MODEL" \
    --acc-mask 1 \
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

# Refuse to mix this model with a pre-existing per-model binary. Choose another
# MODEL_POSTFIX or move the existing directory aside before rerunning.
if [[ -e "$BINARY_DIR" ]]; then
  echo "Binary directory already exists: $BINARY_DIR" >&2
  echo "Choose a new MODEL_POSTFIX (for example, kws_iter007)." >&2
  exit 1
fi

echo "[2/2] Creating and building host_binary_make.dataset.$MODEL_POSTFIX"
mkdir -p "$BINARY_DIR"
rsync -a --exclude 'build/' "$BASE_BINARY_DIR/" "$BINARY_DIR/"
mkdir -p "$BUILD_DIR"

# Explicitly select the normal runner. CMake creates execute_graph_for_dataset;
# no per-node debug executable is built.
export DEBUG_EXE=0

cmake \
  -S "$BINARY_DIR" \
  -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Debug \
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
echo "  ./run_dataset_eval.sh -b host_binary_make.dataset.$MODEL_POSTFIX -m $MODEL_EVL_DIR -d kws_sc ..."
