#!/bin/bash

# Shared scan programming steps for run_chiptest.sh and run_dataset_eval.sh
#
# Requires these variables to be set before sourcing:
#   REMOTE_HOST, REMOTE_PORT, REMOTE_USER, REMOTE_PASSWORD, REMOTE_BASE_PATH
#
# Usage:
#   source ./scan_steps.sh
#   scan_transfer_reg_files "$STEP_NUM" "$SKIP_FLAG"
#   scan_transfer_executable "$STEP_NUM" "$SKIP_FLAG"
#   scan_program_registers "$STEP_NUM" "$SKIP_FLAG"

NPZ_FILE_PATH="${NPZ_FILE_PATH:-scan_reg_files}"

scan_transfer_reg_files() {
    local STEP_NUM="$1"
    local SKIP="$2"
    if [[ "$SKIP" == true ]]; then
        echo "Step $STEP_NUM: Skipped."
        echo ""
        return 0
    fi
    echo "Step $STEP_NUM: Transferring scan_reg_files to remote server..."
    echo "y" | ./transfer_evl.sh --host "$REMOTE_HOST" --path "scan_gen/$NPZ_FILE_PATH"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to transfer $NPZ_FILE_PATH"
        exit 1
    fi
    echo ""
}

scan_transfer_executable() {
    local STEP_NUM="$1"
    local SKIP="$2"
    if [[ "$SKIP" == true ]]; then
        echo "Step $STEP_NUM: Skipped."
        echo ""
        return 0
    fi
    echo "Step $STEP_NUM: Transferring program_scan_reg to remote server..."
    echo "y" | ./transfer_evl.sh --host "$REMOTE_HOST" --path "scan_gen/scan_executable_make"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to transfer scan_executable_make"
        exit 1
    fi
    echo ""
}

scan_program_registers() {
    local STEP_NUM="$1"
    local SKIP="$2"
    if [[ "$SKIP" == true ]]; then
        echo "Step $STEP_NUM: Skipped."
        echo ""
        return 0
    fi
    echo "Step $STEP_NUM: Executing scan program on remote chip (timeout: 0.5s)..."
    echo ""
    sshpass -p "$REMOTE_PASSWORD" ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST \
               "source ~/.bashrc && source /home/root/.venv/bin/activate && \
                cd $REMOTE_BASE_PATH/scan_gen/scan_executable_make/build && timeout -s INT 0.5s ./program_scan_reg \
                $REMOTE_BASE_PATH/scan_gen/$NPZ_FILE_PATH; \
                cd /home/root/imcflow/xilinx/petalinux-csrc && make clear_time && make warmup > /dev/null 2>&1 && \
                tvm_status=\$?; exit \$tvm_status"

    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "SCAN programming completed successfully!"
        echo "=========================================="
    else
        echo ""
        echo "=========================================="
        echo "SCAN programming failed!"
        echo "=========================================="
        exit 1
    fi
    echo ""
}
