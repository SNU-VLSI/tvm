#!/usr/bin/env bash
set -euo pipefail

# Example wrapper for:
#   1. Run py_runner with the same ADC noise CSV/layout used by CIM inference.
#   2. Compare py_runner dumps against debug_model.with_noise.pkl via compare_all_convs.py.
#
# Edit the variables in this block first. Everything can also be overridden from
# the environment, e.g.:
#   SAMPLES="0 1 2" RUN_SIM=0 ./scripts/run_compare_all_convs_example.sh
#   RUN_COMPILE=1 SAMPLES="0 1 2" ./scripts/run_compare_all_convs_example.sh

CODEGEN_DIR="${CODEGEN_DIR:-/root/project/tvm/tvm_practice/test_imcflow/codegen}"
MODEL="${MODEL:-resnet8_subset31_pretrained_orig}"
CKPT="${CKPT:-n0_signed_sample_fixed}"
SAMPLES="${SAMPLES:-0}"

# These must match the CIM inference.py run that produced debug_model.with_noise.pkl.
# n0_signed_sample_fixed was trained/evaluated with signed-weight ref noise.
NOISE_CSV="${NOISE_CSV:-/root/project/CIM/noise/noise_df/B2_signed_weight_ref_out/N0/B2_noise_matrix_per_ch_concat_signed_weight_ref.csv}"
NOISE_LAYOUT_JSON="${NOISE_LAYOUT_JSON:-/root/project/CIM/noise/noise_df/B2_signed_weight_ref_out/N0/concat_per_core.json}"
NOISE_TABLE_FORMAT="${NOISE_TABLE_FORMAT:-ref}"
NOISE_GRANULARITY="${NOISE_GRANULARITY:-input_bitplane}"
NOISE_SEED="${NOISE_SEED:-42}"

# Use "greedy" for deterministic exact dump comparison. "sample"/"alias" follows
# the same empirical distribution but is not bitwise-aligned with CIM's torch RNG.
NOISE_MODE="${NOISE_MODE:-greedy}"

# Set RUN_COMPILE=1 to regenerate the eval_dir before simulation. Compile runs
# once before the sample loop. Set RUN_SIM=0 to reuse existing py_runner dumps.
# Set RUN_COMPARE=0 to only regenerate dumps.
RUN_COMPILE="${RUN_COMPILE:-1}"
RUN_SIM="${RUN_SIM:-1}"
RUN_COMPARE="${RUN_COMPARE:-1}"

# COL
NUM_DISABLE="${NUM_DISABLE:-0}  "

# Keep these aligned with the compile metadata for n0_signed_sample_fixed.
MAIN_ARGS=(
  --model "${MODEL}"
  --driver-v2
  --ref-models transformed
  --fixed-imce-core 0,1
  --num-disable-columns ${NUM_DISABLE}
  --column-disable-config /root/project/CIM/noise/noise_df/B2_out_refine_fixed_full_v1_partial/N32/disabled.json
  --random-seed 42
)

SIM_ARGS=(
  --dataset cifar10
  --start-at simulate
  --stop-at simulate
)

COMPILE_ARGS=(
  --stop-at compile
)

cd "${CODEGEN_DIR}"

source imcflow-baremetal.sh

# compare_all_convs.py currently hardcodes this eval dir and pkl location:
#   eval_dir/resnet8_subset31_pretrained_orig_evl.baremetal
#   debugging/${CKPT}/debug_model.with_noise.pkl
# If you use another eval dir, update compare_all_convs.py or add an eval-dir
# argument there before using this wrapper.
PKL_PATH="${CODEGEN_DIR}/debugging/${CKPT}/debug_model.with_noise.pkl"
if [[ "${RUN_COMPARE}" == "1" && ! -f "${PKL_PATH}" ]]; then
  echo "Missing pkl: ${PKL_PATH}" >&2
  echo "Put debug_model.with_noise.pkl there, or set CKPT to the directory name under debugging/." >&2
  exit 1
fi

if [[ "${RUN_SIM}" == "1" ]]; then
  [[ -f "${NOISE_CSV}" ]] || { echo "Missing NOISE_CSV: ${NOISE_CSV}" >&2; exit 1; }
  [[ -f "${NOISE_LAYOUT_JSON}" ]] || { echo "Missing NOISE_LAYOUT_JSON: ${NOISE_LAYOUT_JSON}" >&2; exit 1; }

  first_noise_key="$(sed -n '3p' "${NOISE_CSV}" | cut -d, -f1)"
  if [[ "${NOISE_TABLE_FORMAT}" == "ref" ]]; then
    if [[ "${first_noise_key}" == *_* ]]; then
      echo "NOISE_CSV looks like wpattern_ref, but NOISE_TABLE_FORMAT=ref: ${NOISE_CSV}" >&2
      echo "Expected numeric signed-ref row keys on line 3, got '${first_noise_key}'." >&2
      exit 1
    fi
  elif [[ "${NOISE_TABLE_FORMAT}" == "wpattern_ref" ]]; then
    if [[ "${first_noise_key}" != *_* ]]; then
      echo "NOISE_CSV looks like signed-ref, but NOISE_TABLE_FORMAT=wpattern_ref: ${NOISE_CSV}" >&2
      echo "Expected row keys like <wpattern>_<ref> on line 3, got '${first_noise_key}'." >&2
      exit 1
    fi
  fi
fi

export IMCFLOW_RUNNER="${IMCFLOW_RUNNER:-py_runner}"

if [[ "${RUN_COMPILE}" == "1" ]]; then
  echo
  echo "================================================================================"
  echo "compile ckpt=${CKPT}"
  echo "================================================================================"
  CKPT="${CKPT}" python main.py "${MAIN_ARGS[@]}" "${COMPILE_ARGS[@]}"
else
  echo "Skipping compile (RUN_COMPILE=0)"
fi

for SAMPLE in ${SAMPLES}; do
  echo
  echo "================================================================================"
  echo "sample=${SAMPLE} ckpt=${CKPT}"
  echo "================================================================================"

  if [[ "${RUN_SIM}" == "1" ]]; then
    echo "[1/2] Running py_runner dump"
    CKPT="${CKPT}" python main.py \
      "${MAIN_ARGS[@]}" \
      "${SIM_ARGS[@]}" \
      --sample "${SAMPLE}" \
      --noise-csv "${NOISE_CSV}" \
      --noise-layout-json "${NOISE_LAYOUT_JSON}" \
      --noise-mode "${NOISE_MODE}" \
      --noise-table-format "${NOISE_TABLE_FORMAT}" \
      --noise-granularity "${NOISE_GRANULARITY}" \
      --noise-seed "${NOISE_SEED}"
  else
    echo "[1/2] Skipping py_runner dump (RUN_SIM=0)"
  fi

  if [[ "${RUN_COMPARE}" == "1" ]]; then
    echo "[2/2] Comparing py_runner dumps against CIM pkl"
    python scripts/compare_all_convs.py --sample "${SAMPLE}" --ckpt "${CKPT}"
  else
    echo "[2/2] Skipping compare_all_convs.py (RUN_COMPARE=0)"
  fi
done
