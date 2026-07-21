#!/bin/bash

# Script to transfer dataset directory to remote server
# Uses the same configuration as transfer_evl.sh

# Configuration (from transfer_evl.sh)
REMOTE_HOST="${1:-${REMOTE_HOST:-147.46.117.99}}"
REMOTE_PORT="${REMOTE_PORT:-1326}"
REMOTE_PATH="${REMOTE_BASE_PATH:-/home/root/tvm/tvm_practice/test_imcflow/codegen}"
REMOTE_AUTH_METHOD="${REMOTE_AUTH_METHOD:-key}"
REMOTE_PASSWORD="${REMOTE_PASSWORD:-}"
USERNAME="${REMOTE_USER:-root}"
REMOTE_SSH_RETRIES="${REMOTE_SSH_RETRIES:-${FPGA_SSH_RETRIES:-3}}"
REMOTE_SSH_RETRY_DELAY_SECONDS="${REMOTE_SSH_RETRY_DELAY_SECONDS:-${FPGA_SSH_RETRY_DELAY_SECONDS:-30}}"
[[ "$REMOTE_SSH_RETRIES" =~ ^[0-9]+$ ]] || REMOTE_SSH_RETRIES=3
[[ "$REMOTE_SSH_RETRY_DELAY_SECONDS" =~ ^[0-9]+$ ]] || REMOTE_SSH_RETRY_DELAY_SECONDS=30

remote_retry() {
    local label="$1"
    shift
    local max_attempts=$((REMOTE_SSH_RETRIES + 1))
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
        echo "  - Retry $label after exit $status (attempt ${attempt}/${max_attempts}) in ${REMOTE_SSH_RETRY_DELAY_SECONDS}s" >&2
        sleep "$REMOTE_SSH_RETRY_DELAY_SECONDS"
    done
    return $status
}

remote_ssh_once() {
    if [[ "$REMOTE_AUTH_METHOD" == "password" ]]; then
        sshpass -p "$REMOTE_PASSWORD" ssh -p "$REMOTE_PORT" "$USERNAME@$REMOTE_HOST" "$@"
    else
        ssh -o BatchMode=yes -p "$REMOTE_PORT" "$USERNAME@$REMOTE_HOST" "$@"
    fi
}

remote_scp_once() {
    if [[ "$REMOTE_AUTH_METHOD" == "password" ]]; then
        sshpass -p "$REMOTE_PASSWORD" scp -P "$REMOTE_PORT" "$@"
    else
        scp -o BatchMode=yes -P "$REMOTE_PORT" "$@"
    fi
}

remote_ssh() {
    remote_retry ssh remote_ssh_once "$@"
}

remote_scp() {
    remote_retry scp remote_scp_once "$@"
}

if [[ "$REMOTE_AUTH_METHOD" != "key" && "$REMOTE_AUTH_METHOD" != "password" ]]; then
    echo "Error: REMOTE_AUTH_METHOD must be key or password"
    exit 1
fi
if [[ "$REMOTE_AUTH_METHOD" == "password" && -z "$REMOTE_PASSWORD" ]]; then
    echo "Error: REMOTE_PASSWORD is required when REMOTE_AUTH_METHOD=password"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR_NAME="$(basename "$SCRIPT_DIR")"

# Optional 2nd arg: a subpath UNDER dataset/ (e.g. "vww/_staged") to transfer
# INSTEAD of the whole dataset/ dir. run_dataset_eval.sh passes the staged subset
# so a deep model's full 1.2GB array is never shipped. Without it, the whole
# dataset/ dir is sent (old behavior, kept for callers that want the full set).
SUBPATH="${2:-}"

if [[ -n "$SUBPATH" ]]; then
    LOCAL_SRC="$SCRIPT_DIR/$SUBPATH"
    if [[ ! -e "$LOCAL_SRC" ]]; then
        echo "  ✗ Subpath not found locally: $LOCAL_SRC"
        exit 1
    fi
    REMOTE_DEST_DIR="$REMOTE_PATH/$DIR_NAME/$(dirname "$SUBPATH")"
    echo "Transferring $DIR_NAME/$SUBPATH to $USERNAME@$REMOTE_HOST:$REMOTE_DEST_DIR..."
    # Only clear the staged subdir on the remote, not the whole dataset/ (other
    # datasets/subsets stay put). Then recreate its parent and copy the subset in.
    echo "  - Cleaning old $DIR_NAME/$SUBPATH on remote..."
    remote_ssh "rm -rf $REMOTE_PATH/$DIR_NAME/$SUBPATH && mkdir -p $REMOTE_DEST_DIR"
    if remote_scp -r "$LOCAL_SRC" "$USERNAME@$REMOTE_HOST:$REMOTE_DEST_DIR"; then
        echo "  ✓ Successfully transferred $DIR_NAME/$SUBPATH"
    else
        echo "  ✗ Failed to transfer $DIR_NAME/$SUBPATH"
        exit 1
    fi
    exit 0
fi

echo "Transferring $DIR_NAME to $USERNAME@$REMOTE_HOST:$REMOTE_PATH..."

# Clean old directory on remote
echo "  - Cleaning old directory on remote..."
remote_ssh "rm -rf $REMOTE_PATH/$DIR_NAME"

# Transfer
if remote_scp -r "$SCRIPT_DIR" "$USERNAME@$REMOTE_HOST:$REMOTE_PATH"; then
    echo "  ✓ Successfully transferred $DIR_NAME"
else
    echo "  ✗ Failed to transfer $DIR_NAME"
    exit 1
fi
