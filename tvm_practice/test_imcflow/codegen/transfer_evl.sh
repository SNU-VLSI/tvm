#!/bin/bash

# Script to transfer _evl directories or arbitrary paths to remote server
# Usage: ./transfer_evl.sh <pattern> [username]
#    or: ./transfer_evl.sh --path <directory> [username]

# Default configuration
REMOTE_HOST="147.46.117.99"
REMOTE_PORT="1326"
REMOTE_PATH="/home/root/tvm/tvm_practice/test_imcflow/codegen/."
REMOTE_PASSWORD="root"
TEST_OUTPUTS_DIR="/root/project/tvm/tvm_practice/test_imcflow/codegen/."

# Function to display help message
show_help() {
    echo "Usage: $0 <pattern> [username]"
    echo "   or: $0 --path <directory> [username]"
    echo "   or: $0 -p <directory> [username]"
    echo ""
    echo "Transfer _evl directories matching a pattern or arbitrary directories to remote server"
    echo ""
    echo "Arguments:"
    echo "  pattern     Pattern to match test directories (e.g., 'one_*', 'resnet8_*', 'one_relu')"
    echo "              Use '*' to match all _evl directories"
    echo "              Can be specified with or without '_evl' suffix (e.g., 'one_relu' or 'one_relu_evl')"
    echo "  --path, -p  Transfer a specific directory (absolute or relative path)"
    echo "  username    Optional username for remote connection (default: root)"
    echo ""
    echo "Examples:"
    echo "  $0 one_*                    # Transfer all one_*_evl directories"
    echo "  $0 one_relu                 # Transfer one_relu_evl directory"
    echo "  $0 one_relu_evl             # Transfer one_relu_evl directory (same as above)"
    echo "  $0 resnet8_* myuser         # Transfer all resnet8_*_evl with custom username"
    echo "  $0 '*'                      # Transfer all _evl directories"
    echo "  $0 --path /path/to/dir      # Transfer specific directory"
    echo "  $0 -p ./host_binary_make    # Transfer specific directory"
    echo ""
    echo "Available test patterns (from MODEL_REGISTRY):"
    echo "  - one_*: one_relu, one_conv_small, one_conv_big, one_mmquant, etc."
    echo "  - resnet8_*: resnet8_subset01-25 variants"
    echo "  - conv_*: conv_quant_conv, etc."
    echo "  - residual_*: residual_model, residual_rnd_model"
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

    MATCHING_DIRS=("$CUSTOM_PATH")
    echo "Custom path mode: transferring directory '$(basename "$CUSTOM_PATH")'"
else
    # Check if test_outputs directory exists
    if [[ ! -d "$TEST_OUTPUTS_DIR" ]]; then
        echo "Error: Directory '$TEST_OUTPUTS_DIR' does not exist"
        exit 1
    fi

    # Normalize pattern - if it already ends with _evl, use as-is, otherwise append _evl
    SEARCH_PATTERN="$PATTERN"
    if [[ "$PATTERN" != "*" ]] && [[ "$PATTERN" != *_evl ]] && [[ "$PATTERN" != *\*_evl ]]; then
        SEARCH_PATTERN="${PATTERN}_evl"
    fi

    # Find matching _evl directories
    echo "Searching for directories matching pattern '$SEARCH_PATTERN' in $TEST_OUTPUTS_DIR..."
    MATCHING_DIRS=()

    # Handle the pattern matching
    if [[ "$PATTERN" == "*" ]]; then
        # Match all _evl directories
        while IFS= read -r -d '' dir; do
            MATCHING_DIRS+=("$dir")
        done < <(find "$TEST_OUTPUTS_DIR" -maxdepth 1 -type d -name "*_evl" -print0)
    else
        # Match specific pattern
        while IFS= read -r -d '' dir; do
            MATCHING_DIRS+=("$dir")
        done < <(find "$TEST_OUTPUTS_DIR" -maxdepth 1 -type d -name "$SEARCH_PATTERN" -print0)
    fi

    # Check if any directories were found
    if [[ ${#MATCHING_DIRS[@]} -eq 0 ]]; then
        echo "Error: No directories found matching pattern '$SEARCH_PATTERN'"
        echo ""
        echo "Available _evl directories:"
        ls -1 "$TEST_OUTPUTS_DIR" | grep "_evl$" || echo "  (none)"
        exit 1
    fi
fi

# Display found directories
echo "Found ${#MATCHING_DIRS[@]} matching director(y/ies):"
for dir in "${MATCHING_DIRS[@]}"; do
    echo "  - $(basename "$dir")"
done
echo ""

# Confirm transfer
read -p "Transfer these directories to $USERNAME@$REMOTE_HOST:$REMOTE_PATH? (y/n) " -n 1 -r
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

    if sshpass -p "$REMOTE_PASSWORD" scp -P "$REMOTE_PORT" -r "$dir" "$USERNAME@$REMOTE_HOST:$REMOTE_PATH"; then
        echo "  ✓ Successfully transferred $dir_name"
        ((TRANSFER_COUNT++))
    else
        echo "  ✗ Failed to transfer $dir_name"
        ((FAILED_COUNT++))
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
