#!/bin/bash

# Shared functions for run_chiptest.sh and run_dataset_eval.sh
#
# Provides:
#   load_env          - Load remote config from .env file
#   scan_transfer_reg_files   - Transfer scan_reg_files to remote
#   scan_transfer_executable  - Transfer scan_executable_make to remote
#   scan_program_registers    - Program scan registers on remote chip
#
# Usage:
#   source ./scan_steps.sh
#   load_env
#   scan_transfer_reg_files "$STEP_NUM" "$SKIP_FLAG"

NPZ_FILE_PATH="${NPZ_FILE_PATH:-scan_reg_files}"

# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

load_env() {
    local SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local ENV_FILE="$SCRIPT_DIR/.env"

    if [[ ! -f "$ENV_FILE" ]]; then
        echo "Error: .env file not found at $ENV_FILE"
        echo ""
        echo "Create it with:"
        echo "  cat > $ENV_FILE << 'EOF'"
        echo "REMOTE_HOST=147.46.117.99"
        echo "REMOTE_PORT=1326"
        echo "REMOTE_USER=root"
        echo "REMOTE_PASSWORD=root"
        echo "REMOTE_BASE_PATH=/home/root/tvm/tvm_practice/test_imcflow/codegen"
        echo "EOF"
        exit 1
    fi

    # Read KEY=VALUE lines, skip comments and blanks
    while IFS='=' read -r key value; do
        key="${key// /}"
        value="${value// /}"
        [[ -z "$key" || "$key" == \#* ]] && continue
        export "$key=$value"
    done < "$ENV_FILE"

    # Validate required variables
    local REQUIRED=(REMOTE_HOST REMOTE_PORT REMOTE_USER REMOTE_PASSWORD REMOTE_BASE_PATH)
    for var in "${REQUIRED[@]}"; do
        if [[ -z "${!var}" ]]; then
            echo "Error: $var is not set in $ENV_FILE"
            exit 1
        fi
    done
}

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
