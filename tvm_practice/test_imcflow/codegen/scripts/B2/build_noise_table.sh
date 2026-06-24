#!/bin/bash
# Build aggregated noise table from chip debug dumps.
#
# Usage:
#   ./scripts/B2/build_noise_table.sh [run_dir] [samples]
#
# Examples:
#   # Default: latest run, samples 0-100
#   CKPT=uqat_tmp02_refine_ndis32_0_78 ./scripts/B2/build_noise_table.sh
#
#   # Specific run and sample range
#   CKPT=uqat_tmp02_refine_ndis32_0_78 ./scripts/B2/build_noise_table.sh run_00 0-199
#
# Environment:
#   CKPT          (required) checkpoint alias
#   REF_BINS      number of ref bins (default: 200)
#   NOISE_BINS    number of noise bins (default: 200)
#   DEVICE        cpu or cuda (default: cpu)
#   NOISE_DIR     directory containing concat_per_core.json
#   LAYOUT_JSON   explicit concat_per_core.json path
#   NPZ_PATH      psum_imcu_column_map.npz path
#   OUTPUT_DIR    directory for aggregated_noise_table.npz
#   WRITE_CSV     set to 1 to also emit CIM ref-only training CSV
#   CSV_OUTPUT    explicit CSV output path
#   CSV_LAYOUT_OUTPUT explicit concat_per_core.json output path
#   CSV_METADATA_OUTPUT explicit CSV metadata output path
#   CSV_MIN_COUNT minimum observations for a CSV (pseudo_ch, ref) row
#   ACC_MASK      4-bit acc mask used by the compiled chip model (default: 0)

set -euo pipefail

CKPT="${CKPT:?ERROR: CKPT env var must be set}"
DUMP_BASE="debugging/fpga/${CKPT}"

# Determine run directory
if [[ -n "${1:-}" ]]; then
    RUN_DIR="${DUMP_BASE}/${1}"
else
    # Find latest run_XX directory
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
REF_BINS="${REF_BINS:-200}"
NOISE_BINS="${NOISE_BINS:-200}"
DEVICE="${DEVICE:-cpu}"
ACC_MASK="${ACC_MASK:-0}"
NOISE_DIR="${NOISE_DIR:-/root/project/CIM/noise/noise_df/B2_out/N32}"
LAYOUT_JSON="${LAYOUT_JSON:-${NOISE_DIR}/concat_per_core.json}"
NPZ_PATH="${NPZ_PATH:-eval_dir/resnet8_subset31_pretrained_orig_evl.linux/psum_imcu_column_map.npz}"

# Resolve checkpoint path
CKPT_PATH="${CKPT_PATH:-}"
if [[ -z "$CKPT_PATH" ]]; then
    CKPT_PATH=$(python3 -c "
import sys; sys.path.insert(0, '/root/project/CIM')
from checkpoints import resolve
print(resolve('B2', 'half', '${CKPT}')[0])
")
fi

# Output paths
CIM_NOISE_DIR="${OUTPUT_DIR:-/root/project/CIM/noise/noise_df/B2_chip_inference/N32/${CKPT}}"
mkdir -p "$CIM_NOISE_DIR"
WRITE_CSV="${WRITE_CSV:-0}"
CSV_OUTPUT="${CSV_OUTPUT:-}"
CSV_LAYOUT_OUTPUT="${CSV_LAYOUT_OUTPUT:-}"
CSV_METADATA_OUTPUT="${CSV_METADATA_OUTPUT:-}"
CSV_MIN_COUNT="${CSV_MIN_COUNT:-1}"

if [[ "$WRITE_CSV" == "1" && -z "$CSV_OUTPUT" ]]; then
    CSV_OUTPUT="$CIM_NOISE_DIR/B2_noise_matrix_per_ch_concat_signed_weight_ref.csv"
fi
if [[ -n "$CSV_OUTPUT" && -z "$CSV_LAYOUT_OUTPUT" ]]; then
    CSV_LAYOUT_OUTPUT="$(dirname "$CSV_OUTPUT")/concat_per_core.json"
fi
if [[ -n "$CSV_OUTPUT" && -z "$CSV_METADATA_OUTPUT" ]]; then
    CSV_METADATA_OUTPUT="$(dirname "$CSV_OUTPUT")/chip_ref_noise_metadata.json"
fi

# Symlink concat_per_core.json if not present
if [[ ! -e "$CIM_NOISE_DIR/concat_per_core.json" ]]; then
    ln -s "$LAYOUT_JSON" "$CIM_NOISE_DIR/concat_per_core.json"
fi

echo "=========================================="
echo "Build aggregated noise table"
echo "  CKPT:       $CKPT"
echo "  Dump dir:   $RUN_DIR"
echo "  Samples:    $SAMPLES"
echo "  Ref bins:   $REF_BINS"
echo "  Noise bins: $NOISE_BINS"
echo "  Acc mask:   $ACC_MASK"
echo "  Checkpoint: $CKPT_PATH"
echo "  Noise dir:  $NOISE_DIR"
echo "  Layout:     $LAYOUT_JSON"
echo "  NPZ path:   $NPZ_PATH"
echo "  Output dir: $CIM_NOISE_DIR"
if [[ -n "$CSV_OUTPUT" ]]; then
echo "  CSV output: $CSV_OUTPUT"
echo "  CSV layout: $CSV_LAYOUT_OUTPUT"
echo "  CSV meta:   $CSV_METADATA_OUTPUT"
fi
echo "=========================================="

ARGS=(
    scripts/build_aggregated_noise_table.py
    --dump-dir "$RUN_DIR" \
    --samples "$SAMPLES" \
    --checkpoint "$CKPT_PATH" \
    --output "$CIM_NOISE_DIR/aggregated_noise_table.npz" \
    --noise-dir "$NOISE_DIR" \
    --layout-json "$LAYOUT_JSON" \
    --npz-path "$NPZ_PATH" \
    --n-ref-bins "$REF_BINS" \
    --n-noise-bins "$NOISE_BINS" \
    --acc-mask "$ACC_MASK" \
    --device "$DEVICE"
)

if [[ -n "$CSV_OUTPUT" ]]; then
    ARGS+=(
        --csv-output "$CSV_OUTPUT"
        --csv-layout-output "$CSV_LAYOUT_OUTPUT"
        --csv-metadata-output "$CSV_METADATA_OUTPUT"
        --csv-min-count "$CSV_MIN_COUNT"
    )
fi

python3 "${ARGS[@]}"

if [[ -n "$CSV_OUTPUT" ]]; then
    cp "$NPZ_PATH" "$(dirname "$CSV_OUTPUT")/psum_imcu_column_map.npz"
fi

echo ""
echo "Done. Files:"
ls -lh "$CIM_NOISE_DIR"/aggregated_noise_table*.npz
if [[ -n "$CSV_OUTPUT" ]]; then
    ls -lh "$CSV_OUTPUT" "$CSV_LAYOUT_OUTPUT" "$(dirname "$CSV_OUTPUT")/psum_imcu_column_map.npz"
fi
