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
SCAN_SSH_RETRIES="${SCAN_SSH_RETRIES:-${REMOTE_SSH_RETRIES:-${FPGA_SSH_RETRIES:-3}}}"
SCAN_SSH_RETRY_DELAY_SECONDS="${SCAN_SSH_RETRY_DELAY_SECONDS:-${REMOTE_SSH_RETRY_DELAY_SECONDS:-${FPGA_SSH_RETRY_DELAY_SECONDS:-30}}}"
[[ "$SCAN_SSH_RETRIES" =~ ^[0-9]+$ ]] || SCAN_SSH_RETRIES=3
[[ "$SCAN_SSH_RETRY_DELAY_SECONDS" =~ ^[0-9]+$ ]] || SCAN_SSH_RETRY_DELAY_SECONDS=30

# ---------------------------------------------------------------------------
# chip lock (remote FPGA lock management)
# ---------------------------------------------------------------------------

SCRIPT_DIR_SCAN_STEPS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR_SCAN_STEPS/chip_lock.sh"

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
        echo "REMOTE_AUTH_METHOD=key"
        echo "REMOTE_BASE_PATH=/home/root/tvm/tvm_practice/test_imcflow/codegen"
        echo "EOF"
        exit 1
    fi

    # Read KEY=VALUE lines, skip comments and blanks
    while IFS='=' read -r key value; do
        key="${key// /}"
        value="${value// /}"
        [[ -z "$key" || "$key" == \#* ]] && continue
        if [[ -z "${!key:-}" ]]; then
            export "$key=$value"
        fi
    done < "$ENV_FILE"

    export REMOTE_AUTH_METHOD="${REMOTE_AUTH_METHOD:-key}"
    if [[ "$REMOTE_AUTH_METHOD" != "key" && "$REMOTE_AUTH_METHOD" != "password" ]]; then
        echo "Error: REMOTE_AUTH_METHOD must be key or password"
        exit 1
    fi

    # Validate required variables
    local REQUIRED=(REMOTE_HOST REMOTE_PORT REMOTE_USER REMOTE_BASE_PATH)
    for var in "${REQUIRED[@]}"; do
        if [[ -z "${!var}" ]]; then
            echo "Error: $var is not set in $ENV_FILE"
            exit 1
        fi
    done
    if [[ "$REMOTE_AUTH_METHOD" == "password" && -z "${REMOTE_PASSWORD:-}" ]]; then
        echo "Error: REMOTE_PASSWORD is required when REMOTE_AUTH_METHOD=password"
        exit 1
    fi
}

scan_ssh_display() {
    if [[ "${REMOTE_AUTH_METHOD:-key}" == "password" ]]; then
        echo "sshpass -p <redacted> ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST"
    else
        echo "ssh -o BatchMode=yes -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST"
    fi
}

scan_scp_display() {
    if [[ "${REMOTE_AUTH_METHOD:-key}" == "password" ]]; then
        echo "sshpass -p <redacted> scp -P $REMOTE_PORT"
    else
        echo "scp -o BatchMode=yes -P $REMOTE_PORT"
    fi
}

scan_retry() {
    local label="$1"
    shift
    local max_attempts=$((SCAN_SSH_RETRIES + 1))
    local attempt
    local status
    for ((attempt = 1; attempt <= max_attempts; attempt++)); do
        "$@"
        status=$?
        local retryable=false
        if [[ $status -eq 255 || ( "$label" == "scp" && $status -eq 1 ) ]]; then
            retryable=true
        fi
        if [[ $status -eq 0 || "$retryable" != true || $attempt -ge $max_attempts ]]; then
            return $status
        fi
        echo "  - Retry $label after exit $status (attempt ${attempt}/${max_attempts}) in ${SCAN_SSH_RETRY_DELAY_SECONDS}s" >&2
        sleep "$SCAN_SSH_RETRY_DELAY_SECONDS"
    done
    return $status
}

scan_ssh_once() {
    if [[ "${REMOTE_AUTH_METHOD:-key}" == "password" ]]; then
        sshpass -p "$REMOTE_PASSWORD" ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" "$@"
    else
        ssh -o BatchMode=yes -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" "$@"
    fi
}

scan_scp_once() {
    if [[ "${REMOTE_AUTH_METHOD:-key}" == "password" ]]; then
        sshpass -p "$REMOTE_PASSWORD" scp -P "$REMOTE_PORT" "$@"
    else
        scp -o BatchMode=yes -P "$REMOTE_PORT" "$@"
    fi
}

scan_ssh() {
    scan_retry ssh scan_ssh_once "$@"
}

scan_scp() {
    scan_retry scp scan_scp_once "$@"
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
    scan_ssh "cd $REMOTE_BASE_PATH/scan_gen/scan_executable_make/build && timeout -s INT 0.5s ./program_scan_reg \
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
