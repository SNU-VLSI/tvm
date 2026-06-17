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

remote_ssh() {
    if [[ "$REMOTE_AUTH_METHOD" == "password" ]]; then
        sshpass -p "$REMOTE_PASSWORD" ssh -p "$REMOTE_PORT" "$USERNAME@$REMOTE_HOST" "$@"
    else
        ssh -o BatchMode=yes -p "$REMOTE_PORT" "$USERNAME@$REMOTE_HOST" "$@"
    fi
}

remote_scp() {
    if [[ "$REMOTE_AUTH_METHOD" == "password" ]]; then
        sshpass -p "$REMOTE_PASSWORD" scp -P "$REMOTE_PORT" "$@"
    else
        scp -o BatchMode=yes -P "$REMOTE_PORT" "$@"
    fi
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
