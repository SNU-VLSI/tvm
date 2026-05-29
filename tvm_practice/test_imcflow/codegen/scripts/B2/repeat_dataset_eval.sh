#!/bin/bash
# Repeat dataset evaluation N times, saving debug dumps and results per run.
#
# Usage:
#   CKPT=uqat_tmp02_refine_ndis32_0_78 DEBUG_EXE=1 CONSOLE_LOG_LEVEL=INFO \
#       ./scripts/B2/repeat_dataset_eval.sh [num_repeats] [num_samples]
#
# Defaults: 10 repeats, 200 samples
# Steps 1-5 run once, then steps 6+6.5+7 repeat N times.
# Debug dumps:  debugging/fpga/<CKPT>/run_00/ ... run_09/
# Eval results: eval_results/dataset_results_<CKPT>_run00_<timestamp>.txt ...

set -euo pipefail

NUM_REPEATS="${1:-10}"
NUM_SAMPLES="${2:-200}"
CKPT="${CKPT:?ERROR: CKPT env var must be set}"

# --- Auto-detect completed runs ---
BASE_DEBUG_DIR="debugging/fpga/${CKPT}"
BASE_RESULT_DIR="eval_results"

START_RUN=0
for ((j=0; j<NUM_REPEATS; j++)); do
    RUN_TAG=$(printf "run_%02d" "$j")
    # Consider a run complete if its debug dir exists (with samples) or a result file exists
    if [[ -d "$BASE_DEBUG_DIR/$RUN_TAG" ]] || \
       ls "$BASE_RESULT_DIR"/dataset_results_${CKPT}_${RUN_TAG}_*.txt &>/dev/null; then
        START_RUN=$((j + 1))
    else
        break
    fi
done

echo "=========================================="
echo "Repeated dataset evaluation"
echo "  CKPT:        $CKPT"
echo "  Repeats:     $NUM_REPEATS"
echo "  Samples:     $NUM_SAMPLES"
echo "  DEBUG_EXE:   ${DEBUG_EXE:-0}"
if [[ "$START_RUN" -gt 0 ]]; then
    echo "  Resuming:    from run_$(printf '%02d' "$START_RUN") (run_00..run_$(printf '%02d' $((START_RUN-1))) already done)"
fi
echo "=========================================="
echo ""

if [[ "$START_RUN" -ge "$NUM_REPEATS" ]]; then
    echo ">>> All $NUM_REPEATS runs already completed. Nothing to do."
    exit 0
fi

# --- Step 1-5: Build and program once (skip if resuming) ---
if [[ "$START_RUN" -eq 0 ]]; then
    echo ">>> Running steps 1-5 (build + transfer + program)..."
    CONSOLE_LOG_LEVEL="${CONSOLE_LOG_LEVEL:-INFO}" \
        ./run_dataset_eval.sh -s 6,7 "$NUM_SAMPLES"
    echo ""
    echo ">>> Steps 1-5 done."
else
    echo ">>> Resuming from run_$(printf '%02d' "$START_RUN"), skipping steps 1-5."
fi
echo ""

# --- Repeat step 6+6.5+7 ---
for ((i=START_RUN; i<NUM_REPEATS; i++)); do
    RUN_TAG=$(printf "run_%02d" "$i")
    echo "=========================================="
    echo "  Run $((i+1))/$NUM_REPEATS  ($RUN_TAG)"
    echo "=========================================="

    # Run eval (skip steps 1-5), with a unique output dir tag
    CONSOLE_LOG_LEVEL="${CONSOLE_LOG_LEVEL:-INFO}" \
    CKPT="${CKPT}" \
    DEBUG_EXE="${DEBUG_EXE:-0}" \
        ./run_dataset_eval.sh -s 1,2,3,4,5 "$NUM_SAMPLES"

    # Move debug dump to per-run subdir
    if [[ "${DEBUG_EXE:-0}" == "1" ]] && [[ -d "$BASE_DEBUG_DIR" ]]; then
        RUN_DEBUG_DIR="${BASE_DEBUG_DIR}/${RUN_TAG}"
        if [[ -d "$RUN_DEBUG_DIR" ]]; then
            rm -rf "$RUN_DEBUG_DIR"
        fi
        mkdir -p "$RUN_DEBUG_DIR"
        # Move sample_* dirs into run subdir
        for item in "$BASE_DEBUG_DIR"/sample_*; do
            [[ -e "$item" ]] && mv "$item" "$RUN_DEBUG_DIR/"
        done
        echo "  Debug dumps moved to $RUN_DEBUG_DIR/"
    fi

    # Rename the most recent result file to include run tag
    LATEST_RESULT=$(ls -t "$BASE_RESULT_DIR"/dataset_results_*.txt 2>/dev/null | head -1)
    if [[ -n "$LATEST_RESULT" ]]; then
        # Insert run tag: dataset_results_20260528_120000.txt -> dataset_results_run00_20260528_120000.txt
        DIRNAME=$(dirname "$LATEST_RESULT")
        BASENAME=$(basename "$LATEST_RESULT")
        NEW_NAME="${BASENAME/dataset_results_/dataset_results_${CKPT}_${RUN_TAG}_}"
        mv "$LATEST_RESULT" "$DIRNAME/$NEW_NAME"
        echo "  Result: $DIRNAME/$NEW_NAME"
    fi

    echo ""
done

echo "=========================================="
echo "All $NUM_REPEATS runs completed."
echo "  Debug dumps: $BASE_DEBUG_DIR/run_00/ ... run_$(printf '%02d' $((NUM_REPEATS-1)))/"
echo "  Results:     $BASE_RESULT_DIR/dataset_results_${CKPT}_run*_*.txt"
echo "=========================================="
