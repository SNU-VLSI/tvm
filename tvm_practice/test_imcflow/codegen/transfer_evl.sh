#!/bin/bash

# Script to transfer _evl directories or arbitrary paths to remote server
# Usage: ./transfer_evl.sh <pattern> [username]
#    or: ./transfer_evl.sh --path <directory> [username]

# Default configuration
REMOTE_HOST="147.46.117.99"
REMOTE_PORT="1326"
REMOTE_BASE_PATH="/home/root/tvm/tvm_practice/test_imcflow/codegen"
REMOTE_PATH="$REMOTE_BASE_PATH/eval_dir"
REMOTE_PASSWORD="root"
LOCAL_CODEGEN_DIR="/root/project/tvm/tvm_practice/test_imcflow/codegen"
TEST_OUTPUTS_DIR="$LOCAL_CODEGEN_DIR/eval_dir"

# Function to display help message
show_help() {
    echo "Usage: $0 <dir_name> [username]"
    echo "   or: $0 --path <directory> [username]"
    echo "   or: $0 -p <directory> [username]"
    echo ""
    echo "Transfer _evl directories or arbitrary directories to remote server"
    echo ""
    echo "Arguments:"
    echo "  dir_name    Exact directory name in eval_dir/ (e.g., 'one_relu_evl.linux')"
    echo "  --path, -p  Transfer a specific directory (absolute or relative path)"
    echo "  username    Optional username for remote connection (default: root)"
    echo ""
    echo "Examples:"
    echo "  $0 one_relu_evl.linux                    # Transfer one_relu_evl.linux"
    echo "  $0 resnet8_subset31_pretrained_orig_evl.linux"
    echo "  $0 --path /path/to/dir                   # Transfer specific directory"
    echo "  $0 -p ./host_binary_make                 # Transfer specific directory"
    echo ""
    echo "Configuration:"
    echo "  Remote host: $REMOTE_HOST"
    echo "  Remote port: $REMOTE_PORT"
    echo "  Remote path: $REMOTE_PATH"
    exit 0
}

# Check for help flag
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]] || [[ $# -eq 0 ]]; then
    show_help
fi

# Parse arguments
CUSTOM_PATH=""
if [[ "$1" == "--path" ]] || [[ "$1" == "-p" ]]; then
    CUSTOM_PATH="$2"
    USERNAME="${3:-root}"
else
    PATTERN="$1"
    USERNAME="${2:-root}"
fi

# Handle custom path mode
if [[ -n "$CUSTOM_PATH" ]]; then
    # Check if custom path exists
    if [[ ! -e "$CUSTOM_PATH" ]]; then
        echo "Error: Path '$CUSTOM_PATH' does not exist"
        exit 1
    fi

    # Check if it's a directory
    if [[ ! -d "$CUSTOM_PATH" ]]; then
        echo "Error: Path '$CUSTOM_PATH' is not a directory"
        exit 1
    fi

    # Get the relative path from codegen directory
    CUSTOM_PATH_ABS="$(cd "$CUSTOM_PATH" && pwd)"
    CUSTOM_PATH_REL="${CUSTOM_PATH_ABS#$LOCAL_CODEGEN_DIR/}"
    CUSTOM_PATH_PARENT="$(dirname "$CUSTOM_PATH_REL")"

    # Set remote path to maintain the same relative structure under codegen
    if [[ "$CUSTOM_PATH_PARENT" == "." ]]; then
        CUSTOM_REMOTE_PATH="$REMOTE_BASE_PATH"
    else
        CUSTOM_REMOTE_PATH="$REMOTE_BASE_PATH/$CUSTOM_PATH_PARENT"
    fi

    MATCHING_DIRS=("$CUSTOM_PATH")
    echo "Custom path mode: transferring '$CUSTOM_PATH_REL' to $CUSTOM_REMOTE_PATH"
else
    # Check if test_outputs directory exists
    if [[ ! -d "$TEST_OUTPUTS_DIR" ]]; then
        echo "Error: Directory '$TEST_OUTPUTS_DIR' does not exist"
        exit 1
    fi

    # Exact directory name matching (no pattern expansion)
    DIR_NAME="$PATTERN"
    TARGET_DIR="$TEST_OUTPUTS_DIR/$DIR_NAME"

    if [[ ! -d "$TARGET_DIR" ]]; then
        echo "Error: Directory '$DIR_NAME' not found in $TEST_OUTPUTS_DIR"
        echo ""
        echo "Available _evl directories:"
        ls -1 "$TEST_OUTPUTS_DIR" | grep "_evl" || echo "  (none)"
        exit 1
    fi

    MATCHING_DIRS=("$TARGET_DIR")
fi

# Display found directories
echo "Found ${#MATCHING_DIRS[@]} matching director(y/ies):"
for dir in "${MATCHING_DIRS[@]}"; do
    echo "  - $(basename "$dir")"
done
echo ""

# Confirm transfer
DISPLAY_REMOTE_PATH="${CUSTOM_REMOTE_PATH:-$REMOTE_PATH}"
read -p "Transfer these directories to $USERNAME@$REMOTE_HOST:$DISPLAY_REMOTE_PATH? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Transfer cancelled"
    exit 0
fi

# Transfer each directory
echo "Starting transfer..."
TRANSFER_COUNT=0
FAILED_COUNT=0

for dir in "${MATCHING_DIRS[@]}"; do
    dir_name=$(basename "$dir")
    echo "Transferring $dir_name..."

    # Check if this is a test folder (_evl or _evl.linux/_evl.baremetal) and not a custom path
    if [[ "$dir_name" == *_evl || "$dir_name" == *_evl.linux || "$dir_name" == *_evl.baremetal ]] && [[ -z "$CUSTOM_PATH" ]]; then
        # For test folders, only transfer specific subdirectories
        SUBDIRS_TO_TRANSFER=("host_binary_make" "test_inputs" "test_outputs" "test_references" "build")

        # Remove existing directory on remote to ensure clean overwrite
        echo "  - Cleaning old directory on remote..."
        sshpass -p "$REMOTE_PASSWORD" ssh -p "$REMOTE_PORT" "$USERNAME@$REMOTE_HOST" "rm -rf $REMOTE_PATH/$dir_name"

        # Create the remote directory
        sshpass -p "$REMOTE_PASSWORD" ssh -p "$REMOTE_PORT" "$USERNAME@$REMOTE_HOST" "mkdir -p $REMOTE_PATH/$dir_name"

        SUBDIR_SUCCESS=true
        for subdir in "${SUBDIRS_TO_TRANSFER[@]}"; do
            subdir_path="$dir/$subdir"
            if [[ -e "$subdir_path" ]]; then
                echo "  - Transferring $subdir..."
                if ! sshpass -p "$REMOTE_PASSWORD" scp -P "$REMOTE_PORT" -r "$subdir_path" "$USERNAME@$REMOTE_HOST:$REMOTE_PATH/$dir_name/"; then
                    echo "    ✗ Failed to transfer $subdir"
                    SUBDIR_SUCCESS=false
                fi
            else
                echo "  - Skipping $subdir (not found)"
            fi
        done

        if [[ "$SUBDIR_SUCCESS" == true ]]; then
            echo "  ✓ Successfully transferred $dir_name (selective subdirectories)"
            ((TRANSFER_COUNT++))
        else
            echo "  ✗ Failed to transfer some subdirectories in $dir_name"
            ((FAILED_COUNT++))
        fi
    else
        # For custom paths or non-evl directories, transfer everything with clean overwrite
        # Use CUSTOM_REMOTE_PATH if set, otherwise REMOTE_PATH
        TARGET_REMOTE_PATH="${CUSTOM_REMOTE_PATH:-$REMOTE_PATH}"
        echo "  - Cleaning old directory on remote ($TARGET_REMOTE_PATH/$dir_name)..."
        sshpass -p "$REMOTE_PASSWORD" ssh -p "$REMOTE_PORT" "$USERNAME@$REMOTE_HOST" "rm -rf $TARGET_REMOTE_PATH/$dir_name"

        if sshpass -p "$REMOTE_PASSWORD" scp -P "$REMOTE_PORT" -r "$dir" "$USERNAME@$REMOTE_HOST:$TARGET_REMOTE_PATH"; then
            echo "  ✓ Successfully transferred $dir_name to $TARGET_REMOTE_PATH"
            ((TRANSFER_COUNT++))
        else
            echo "  ✗ Failed to transfer $dir_name"
            ((FAILED_COUNT++))
        fi
    fi
done

# Summary
echo ""
echo "Transfer complete:"
echo "  Successfully transferred: $TRANSFER_COUNT"
if [[ $FAILED_COUNT -gt 0 ]]; then
    echo "  Failed: $FAILED_COUNT"
    exit 1
fi
