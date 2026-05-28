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

set -euo pipefail

CKPT="${CKPT:?ERROR: CKPT env var must be set}"
DUMP_BASE="debugging/fpga/${CKPT}"

# Determine run directory
if [[ -n "${1:-}" ]]; then
    RUN_DIR="${DUMP_BASE}/${1}"
else
    # Find latest run_XX directory
    RUN_DIR=$(ls -d "${DUMP_BASE}"/run_* 2>/dev/null | sort | tail -1)
    if [[ -z "$RUN_DIR" ]]; then
        echo "ERROR: No run_XX directories found in ${DUMP_BASE}/"
        exit 1
    fi
fi

SAMPLES="${2:-0-100}"
REF_BINS="${REF_BINS:-200}"
NOISE_BINS="${NOISE_BINS:-200}"
DEVICE="${DEVICE:-cpu}"

# Resolve checkpoint path
CKPT_PATH=$(python3 -c "
import sys; sys.path.insert(0, '/root/project/CIM')
from checkpoints import resolve
print(resolve('B2', 'half', '${CKPT}')[0])
")

# Output paths
CIM_NOISE_DIR="/root/project/CIM/noise/noise_df/B2_chip_inference/N32/${CKPT}"
mkdir -p "$CIM_NOISE_DIR"

# Symlink concat_per_core.json if not present
if [[ ! -e "$CIM_NOISE_DIR/concat_per_core.json" ]]; then
    ln -s ../../../B2_out/N32/concat_per_core.json "$CIM_NOISE_DIR/concat_per_core.json"
fi

echo "=========================================="
echo "Build aggregated noise table"
echo "  CKPT:       $CKPT"
echo "  Dump dir:   $RUN_DIR"
echo "  Samples:    $SAMPLES"
echo "  Ref bins:   $REF_BINS"
echo "  Noise bins: $NOISE_BINS"
echo "  Checkpoint: $CKPT_PATH"
echo "  Output dir: $CIM_NOISE_DIR"
echo "=========================================="

python scripts/build_aggregated_noise_table.py \
    --dump-dir "$RUN_DIR" \
    --samples "$SAMPLES" \
    --checkpoint "$CKPT_PATH" \
    --output "$CIM_NOISE_DIR/aggregated_noise_table.npz" \
    --n-ref-bins "$REF_BINS" \
    --n-noise-bins "$NOISE_BINS" \
    --device "$DEVICE"

echo ""
echo "Done. Files:"
ls -lh "$CIM_NOISE_DIR"/aggregated_noise_table*.npz
