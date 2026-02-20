#!/bin/bash

# Script to transfer dataset directory to remote server
# Uses the same configuration as transfer_evl.sh

# Configuration (from transfer_evl.sh)
REMOTE_HOST="147.46.117.99"
REMOTE_PORT="1326"
REMOTE_PATH="/home/root/tvm/tvm_practice/test_imcflow/codegen"
REMOTE_PASSWORD="root"
USERNAME="root"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR_NAME="$(basename "$SCRIPT_DIR")"

echo "Transferring $DIR_NAME to $USERNAME@$REMOTE_HOST:$REMOTE_PATH..."

# Clean old directory on remote
echo "  - Cleaning old directory on remote..."
sshpass -p "$REMOTE_PASSWORD" ssh -p "$REMOTE_PORT" "$USERNAME@$REMOTE_HOST" "rm -rf $REMOTE_PATH/$DIR_NAME"

# Transfer
if sshpass -p "$REMOTE_PASSWORD" scp -P "$REMOTE_PORT" -r "$SCRIPT_DIR" "$USERNAME@$REMOTE_HOST:$REMOTE_PATH"; then
    echo "  ✓ Successfully transferred $DIR_NAME"
else
    echo "  ✗ Failed to transfer $DIR_NAME"
    exit 1
fi
