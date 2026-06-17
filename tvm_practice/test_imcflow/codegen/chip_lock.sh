#!/bin/bash
# chip_lock.sh - FPGA chip lock utilities for shell scripts
#
# Provides acquire/release/status for a remote lock file on the FPGA server.
# Lock file: /tmp/imcflow.lock (on remote)
# User ID:   read from ~/.imcflow.env (USER_ID=xxx), defaults to "unknown"
#
# Usage:
#   source chip_lock.sh
#   chip_lock_acquire "run_dataset_eval.sh"   # exits on failure
#   ...  chip work ...
#   chip_lock_release
#
# Requires: REMOTE_HOST, REMOTE_PORT, REMOTE_USER (from .env or caller env).
# REMOTE_AUTH_METHOD=key uses authorized_keys. REMOTE_AUTH_METHOD=password uses sshpass.

CHIP_LOCKFILE="/tmp/imcflow_user.lock"

# ANSI color codes
_RED='\033[1;31m'
_YELLOW='\033[1;33m'
_GREEN='\033[1;32m'
_NC='\033[0m' # No Color

_chip_lock_get_user_id() {
    local env_file="$HOME/.imcflow.env"
    if [[ -f "$env_file" ]]; then
        local uid
        uid=$(grep -E '^USER_ID=' "$env_file" | head -1 | cut -d'=' -f2 | tr -d '[:space:]')
        if [[ -n "$uid" ]]; then
            echo "$uid"
            return
        fi
    fi
    echo "unknown"
}

_chip_lock_ssh() {
    if [[ "${REMOTE_AUTH_METHOD:-key}" == "password" ]]; then
        sshpass -p "$REMOTE_PASSWORD" ssh -o ConnectTimeout=10 -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" "$1"
    else
        ssh -o BatchMode=yes -o ConnectTimeout=10 -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" "$1"
    fi
}

# Check if stale: lock exists but no chip-related process is running on remote
_chip_lock_is_stale() {
    local procs
    procs=$(_chip_lock_ssh "pgrep -a 'test_imcflow|execute_graph|program_scan_reg' 2>/dev/null")
    if [[ -z "$procs" ]]; then
        return 0  # stale
    fi
    return 1  # not stale, processes are running
}

chip_lock_acquire() {
    local script_name="${1:-unknown_script}"
    local user_id
    user_id=$(_chip_lock_get_user_id)
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # Check if lock exists on remote
    local existing_lock
    existing_lock=$(_chip_lock_ssh "cat $CHIP_LOCKFILE 2>/dev/null")
    local ssh_status=$?

    # Check for SSH connectivity failure (exit code 255 = connection error)
    if [[ $ssh_status -eq 255 ]]; then
        echo ""
        echo -e "${_RED}=========================================${_NC}"
        echo -e "${_RED}  CHIP UNREACHABLE - ABORTING${_NC}"
        echo -e "${_RED}=========================================${_NC}"
        echo -e "${_RED}  Cannot connect to ${REMOTE_HOST}:${REMOTE_PORT}${_NC}"
        echo -e "${_RED}  SSH connection failed (timeout or refused).${_NC}"
        echo -e "${_YELLOW}  Check that the FPGA board is powered on and${_NC}"
        echo -e "${_YELLOW}  network-reachable, then retry.${_NC}"
        echo -e "${_RED}=========================================${_NC}"
        echo ""
        exit 1
    fi

    if [[ -n "$existing_lock" ]]; then
        # Lock exists - always abort (never auto-remove)
        echo ""
        echo -e "${_RED}=========================================${_NC}"
        echo -e "${_RED}  CHIP LOCKED - ABORTING${_NC}"
        echo -e "${_RED}=========================================${_NC}"
        echo -e "${_RED}  Lock info:${_NC}"
        echo "$existing_lock" | while IFS= read -r line; do
            echo -e "${_RED}    $line${_NC}"
        done
        if _chip_lock_is_stale; then
            echo -e "${_YELLOW}  (no chip process detected - may be stale)${_NC}"
            echo -e "${_YELLOW}  To remove: ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST 'rm $CHIP_LOCKFILE'${_NC}"
        fi
        echo -e "${_RED}=========================================${_NC}"
        echo ""
        exit 1
    fi

    # Acquire lock
    local lock_content="user_id: ${user_id}
script: ${script_name}
started: ${timestamp}"

    _chip_lock_ssh "cat > $CHIP_LOCKFILE << 'LOCKEOF'
${lock_content}
LOCKEOF"

    if [[ $? -eq 255 ]]; then
        echo ""
        echo -e "${_RED}=========================================${_NC}"
        echo -e "${_RED}  CHIP UNREACHABLE - ABORTING${_NC}"
        echo -e "${_RED}=========================================${_NC}"
        echo -e "${_RED}  Connected for check but failed to write lock.${_NC}"
        echo -e "${_RED}  SSH connection to ${REMOTE_HOST}:${REMOTE_PORT} lost.${_NC}"
        echo -e "${_RED}=========================================${_NC}"
        echo ""
        exit 1
    fi

    echo -e "${_GREEN}[CHIP LOCK] Acquired (user=${user_id}, script=${script_name})${_NC}"
}

chip_lock_release() {
    _chip_lock_ssh "rm -f $CHIP_LOCKFILE" 2>/dev/null
    if [[ $? -eq 255 ]]; then
        echo -e "${_YELLOW}[CHIP LOCK] Warning: could not release lock (chip unreachable).${_NC}"
        echo -e "${_YELLOW}  Lock file ${CHIP_LOCKFILE} may remain on remote.${_NC}"
        echo -e "${_YELLOW}  Remove manually when chip is back: ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST 'rm $CHIP_LOCKFILE'${_NC}"
    else
        echo -e "${_GREEN}[CHIP LOCK] Released${_NC}"
    fi
}

chip_lock_status() {
    local existing_lock
    existing_lock=$(_chip_lock_ssh "cat $CHIP_LOCKFILE 2>/dev/null")
    local ssh_status=$?

    if [[ $ssh_status -eq 255 ]]; then
        echo -e "${_RED}[CHIP LOCK] Chip is UNREACHABLE (${REMOTE_HOST}:${REMOTE_PORT})${_NC}"
        echo -e "${_RED}  Cannot determine lock status - SSH connection failed.${_NC}"
    elif [[ -z "$existing_lock" ]]; then
        echo -e "${_GREEN}[CHIP LOCK] Chip is FREE${_NC}"
    else
        echo -e "${_YELLOW}[CHIP LOCK] Chip is BUSY:${_NC}"
        echo "$existing_lock" | while IFS= read -r line; do
            echo -e "${_YELLOW}  $line${_NC}"
        done

        if _chip_lock_is_stale; then
            echo -e "${_YELLOW}  (stale - no chip process detected)${_NC}"
        fi
    fi
}
