#!/bin/bash
# Build a CSV-reconstructed aggregated noise table from debug dumps.
#
# Usage:
#   CKPT=n32_signed_sample ./scripts/B2/build_csv_reconstructed_noise_table.sh [run_dir] [samples]
#
# Environment:
#   CKPT                 required checkpoint alias
#   CSV                  noise CSV path or filename under NOISE_DIR
#   NOISE_TABLE_FORMAT   auto, wpattern_ref, or ref (default: auto)
#   REF_RECONSTRUCTION_GRANULARITY
#                        input_bitplane or output for ref CSVs (default: input_bitplane)
#   NOISE_DIR            directory containing concat_per_core.json
#   LAYOUT_JSON          explicit concat_per_core.json path
#   NPZ_PATH             psum_imcu_column_map.npz path
#   OUTPUT_DIR           output directory
#   OUTPUT_NPZ           explicit output NPZ path
#   MATCH_NPZ            existing aggregated npz whose bins should be reused
#   N_MC_TRIALS          MC trials per output element (default: 50)
#   REF_BINS             number of ref bins if MATCH_NPZ is unset (default: 200)
#   NOISE_BINS           number of noise bins if MATCH_NPZ is unset (default: 200)
#   DEVICE               cpu or cuda (default: cpu)
#   SEED                 RNG seed (default: 42)
#   ACC_MASK             4-bit acc mask used by the compiled chip model (default: 0)
#   NOISE_MODEL_PROFILE  diagnostics profile: resnet8 or kws_dscnn (default: resnet8)

set -euo pipefail

CKPT="${CKPT:?ERROR: CKPT env var must be set}"
DUMP_BASE="debugging/fpga/${CKPT}"

if [[ -n "${1:-}" ]]; then
    RUN_DIR="${DUMP_BASE}/${1}"
else
    RUN_DIR=$(find "$DUMP_BASE" -maxdepth 1 -type d -name 'run_*' 2>/dev/null | sort | tail -1 || true)
    if [[ -z "$RUN_DIR" ]]; then
        if find "$DUMP_BASE" -maxdepth 1 -type d -name 'sample_*' -print -quit 2>/dev/null | grep -q .; then
            RUN_DIR="$DUMP_BASE"
        else
            echo "ERROR: No run_XX or sample_XX directories found in ${DUMP_BASE}/"
            exit 1
        fi
    fi
fi

SAMPLES="${2:-0-100}"
NOISE_DIR="${NOISE_DIR:-/root/project/CIM/noise/noise_df/B2_out/N32}"
LAYOUT_JSON="${LAYOUT_JSON:-${NOISE_DIR}/concat_per_core.json}"
NPZ_PATH="${NPZ_PATH:-eval_dir/resnet8_subset31_pretrained_orig_evl.linux/psum_imcu_column_map.npz}"
CSV="${CSV:-B2_noise_matrix_per_ch_concat.csv}"
NOISE_TABLE_FORMAT="${NOISE_TABLE_FORMAT:-auto}"
REF_RECONSTRUCTION_GRANULARITY="${REF_RECONSTRUCTION_GRANULARITY:-input_bitplane}"
N_MC_TRIALS="${N_MC_TRIALS:-50}"
REF_BINS="${REF_BINS:-200}"
NOISE_BINS="${NOISE_BINS:-200}"
DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-42}"
ACC_MASK="${ACC_MASK:-0}"
NOISE_MODEL_PROFILE="${NOISE_MODEL_PROFILE:-resnet8}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/project/CIM/noise/noise_df/B2_chip_inference/N32/${CKPT}}"
OUTPUT_NPZ="${OUTPUT_NPZ:-${OUTPUT_DIR}/csv_reconstructed_noise_table.npz}"

CKPT_PATH="${CKPT_PATH:-}"
if [[ -z "$CKPT_PATH" ]]; then
    CKPT_PATH=$(python3 -c "
import sys; sys.path.insert(0, '/root/project/CIM')
from checkpoints import resolve
print(resolve('B2', 'half', '${CKPT}')[0])
")
fi

mkdir -p "$OUTPUT_DIR"
if [[ ! -e "$OUTPUT_DIR/concat_per_core.json" ]]; then
    ln -s "$LAYOUT_JSON" "$OUTPUT_DIR/concat_per_core.json"
fi

EXTRA_ARGS=()
if [[ -n "${MATCH_NPZ:-}" ]]; then
    EXTRA_ARGS+=(--match-npz "$MATCH_NPZ")
fi

echo "=========================================="
echo "Build CSV-reconstructed noise table"
echo "  CKPT:       $CKPT"
echo "  Dump dir:   $RUN_DIR"
echo "  Samples:    $SAMPLES"
echo "  CSV:        $CSV"
echo "  Format:     $NOISE_TABLE_FORMAT"
echo "  Ref recon:  $REF_RECONSTRUCTION_GRANULARITY"
echo "  MC trials:  $N_MC_TRIALS"
echo "  Acc mask:   $ACC_MASK"
echo "  Profile:    $NOISE_MODEL_PROFILE"
echo "  Checkpoint: $CKPT_PATH"
echo "  Noise dir:  $NOISE_DIR"
echo "  Layout:     $LAYOUT_JSON"
echo "  NPZ path:   $NPZ_PATH"
echo "  Output dir: $OUTPUT_DIR"
echo "  Output NPZ: $OUTPUT_NPZ"
if [[ -n "${MATCH_NPZ:-}" ]]; then
    echo "  Match NPZ:  $MATCH_NPZ"
else
    echo "  Ref bins:   $REF_BINS"
    echo "  Noise bins: $NOISE_BINS"
fi
echo "=========================================="

python3 scripts/build_csv_reconstructed_noise_table.py \
    --dump-dir "$RUN_DIR" \
    --samples "$SAMPLES" \
    --checkpoint "$CKPT_PATH" \
    --csv "$CSV" \
    --noise-table-format "$NOISE_TABLE_FORMAT" \
    --ref-reconstruction-granularity "$REF_RECONSTRUCTION_GRANULARITY" \
    --noise-dir "$NOISE_DIR" \
    --layout-json "$LAYOUT_JSON" \
    --npz-path "$NPZ_PATH" \
    --output "$OUTPUT_NPZ" \
    --n-mc-trials "$N_MC_TRIALS" \
    --n-ref-bins "$REF_BINS" \
    --n-noise-bins "$NOISE_BINS" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --acc-mask "$ACC_MASK" \
    --model-profile "$NOISE_MODEL_PROFILE" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "Done. Files:"
ls -lh "$OUTPUT_NPZ"
