#!/bin/bash

# Shared tagged whole-run power measurement steps for chip runners.
# This file is sourced after scan_steps.sh, so scan_ssh/scan_scp and .env are
# already available.  It deliberately does nothing unless a power config is
# selected by --power-config or IMCFLOW_POWER_CONFIG.

POWER_ENABLED=false
POWER_SESSION_ID=""
POWER_LOCAL_REQUEST=""
POWER_REMOTE_REQUEST=""
POWER_LOCAL_RESULT_DIR=""
POWER_STATUS=0


power_sanitize_id() {
    local value="$1"
    value="${value//[^A-Za-z0-9_-]/_}"
    printf '%s' "${value:0:80}"
}


power_validate_endpoint() {
    if [[ -z "${POWER_MEASUREMENT_HOST:-}" ]]; then
        echo "Error: POWER_MEASUREMENT_HOST must be the measurement-server address reachable from the board" >&2
        return 1
    fi
    if [[ ! "$POWER_MEASUREMENT_HOST" =~ ^[A-Za-z0-9.:-]+$ ]]; then
        echo "Error: unsafe POWER_MEASUREMENT_HOST: $POWER_MEASUREMENT_HOST" >&2
        return 1
    fi
    POWER_MEASUREMENT_PORT="${POWER_MEASUREMENT_PORT:-9910}"
    if [[ ! "$POWER_MEASUREMENT_PORT" =~ ^[0-9]+$ ]] ||
       (( POWER_MEASUREMENT_PORT < 1 || POWER_MEASUREMENT_PORT > 65535 )); then
        echo "Error: invalid POWER_MEASUREMENT_PORT: $POWER_MEASUREMENT_PORT" >&2
        return 1
    fi
}


power_revision_preflight() {
    local tvm_root
    local board_tvm_root="${POWER_BOARD_TVM_ROOT:-/home/root/tvm}"
    local meas_repo="${POWER_MEASUREMENT_UTILS_REPO:-/home/jaeyongjang/project.local/measurement_utils}"
    local result_ssh="${POWER_RESULT_SSH_HOST:-meas-2}"
    local binary_info=""
    local expected_binary_info=""
    local master_status=""
    local board_status=""
    local server_status=""

    tvm_root="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)" || return 1
    POWER_MASTER_TVM_REV="$(git -C "$tvm_root" rev-parse HEAD)" || return 1
    POWER_MASTER_MEASUREMENT_REV="$(git -C "$tvm_root/3rdparty/measurement_utils" rev-parse HEAD)" || return 1
    POWER_BOARD_TVM_REV="$(scan_ssh "git -C $board_tvm_root rev-parse HEAD")" || return 1
    POWER_BOARD_MEASUREMENT_REV="$(scan_ssh "git -C $board_tvm_root/3rdparty/measurement_utils rev-parse HEAD")" || return 1
    POWER_SERVER_MEASUREMENT_REV="$(ssh "$result_ssh" "git -C $meas_repo rev-parse HEAD")" || return 1

    POWER_BOARD_TVM_REV="${POWER_BOARD_TVM_REV//$'\r'/}"
    POWER_BOARD_MEASUREMENT_REV="${POWER_BOARD_MEASUREMENT_REV//$'\r'/}"
    POWER_SERVER_MEASUREMENT_REV="${POWER_SERVER_MEASUREMENT_REV//$'\r'/}"
    master_status="$(git -C "$tvm_root" status --porcelain --untracked-files=no)" || return 1
    board_status="$(scan_ssh "git -C $board_tvm_root status --porcelain --untracked-files=no")" || return 1
    server_status="$(ssh "$result_ssh" "git -C $meas_repo status --porcelain --untracked-files=no")" || return 1
    board_status="${board_status//$'\r'/}"
    server_status="${server_status//$'\r'/}"
    if [[ -n "$master_status" || -n "$board_status" || -n "$server_status" ]]; then
        echo "Error: tracked repository changes must be committed before a power run" >&2
        [[ -n "$master_status" ]] && echo "  master: $master_status" >&2
        [[ -n "$board_status" ]] && echo "  board: $board_status" >&2
        [[ -n "$server_status" ]] && echo "  meas-2: $server_status" >&2
        return 1
    fi
    if [[ "$POWER_MASTER_TVM_REV" != "$POWER_BOARD_TVM_REV" ]]; then
        echo "Error: TVM revision mismatch before power run" >&2
        echo "  master=$POWER_MASTER_TVM_REV" >&2
        echo "  board =$POWER_BOARD_TVM_REV" >&2
        return 1
    fi
    if [[ "$POWER_MASTER_MEASUREMENT_REV" != "$POWER_BOARD_MEASUREMENT_REV" ||
          "$POWER_MASTER_MEASUREMENT_REV" != "$POWER_SERVER_MEASUREMENT_REV" ]]; then
        echo "Error: measurement_utils revision mismatch before power run" >&2
        echo "  master=$POWER_MASTER_MEASUREMENT_REV" >&2
        echo "  board =$POWER_BOARD_MEASUREMENT_REV" >&2
        echo "  meas-2=$POWER_SERVER_MEASUREMENT_REV" >&2
        return 1
    fi
    if [[ -n "${POWER_BUILD_METADATA:-}" ]]; then
        if [[ ! -f "$POWER_BUILD_METADATA" ]]; then
            echo "Error: codegen build metadata not found: $POWER_BUILD_METADATA" >&2
            return 1
        fi
        python "$SCRIPT_DIR/scripts/power_request.py" validate-build-identity \
            --metadata "$POWER_BUILD_METADATA" \
            --tvm-rev "$POWER_MASTER_TVM_REV" \
            --measurement-rev "$POWER_MASTER_MEASUREMENT_REV" || return 1
        POWER_CODEGEN_TVM_REV="$POWER_MASTER_TVM_REV"
        POWER_CODEGEN_MEASUREMENT_REV="$POWER_MASTER_MEASUREMENT_REV"
    fi
    if [[ -n "${POWER_REMOTE_BINARY:-}" ]]; then
        if [[ ! "$POWER_REMOTE_BINARY" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
            echo "Error: unsafe POWER_REMOTE_BINARY: $POWER_REMOTE_BINARY" >&2
            return 1
        fi
        binary_info="$(scan_ssh "$POWER_REMOTE_BINARY --power-build-info")" || {
            echo "Error: cannot read power build identity from $POWER_REMOTE_BINARY" >&2
            return 1
        }
        binary_info="${binary_info//$'\r'/}"
        expected_binary_info="IMCFLOW_POWER_BUILD_INFO tvm=$POWER_MASTER_TVM_REV measurement_utils=$POWER_MASTER_MEASUREMENT_REV dirty=0"
        if [[ "$binary_info" != "$expected_binary_info" ]]; then
            echo "Error: deployed binary revision mismatch before power run" >&2
            echo "  expected=$expected_binary_info" >&2
            echo "  binary  =$binary_info" >&2
            return 1
        fi
        POWER_BINARY_TVM_REV="$POWER_MASTER_TVM_REV"
        POWER_BINARY_MEASUREMENT_REV="$POWER_MASTER_MEASUREMENT_REV"
    fi
    echo "[POWER] revisions: tvm=$POWER_MASTER_TVM_REV measurement_utils=$POWER_MASTER_MEASUREMENT_REV"
}


power_probe_server() {
    local result_ssh="${POWER_RESULT_SSH_HOST:-meas-2}"
    local meas_python="${POWER_MEASUREMENT_PYTHON:-/home/jaeyongjang/anaconda3/envs/imcflow/bin/python}"
    ssh "$result_ssh" "$meas_python -c 'import socket; s=socket.create_connection((\"127.0.0.1\", int(\"$POWER_MEASUREMENT_PORT\")), 2); s.sendall(b\"HELLO 2\\n\"); print(s.makefile(\"rb\").readline().decode().strip()); s.close()'" 2>/dev/null
}


power_ensure_server() {
    local result_ssh="${POWER_RESULT_SSH_HOST:-meas-2}"
    local meas_repo="${POWER_MEASUREMENT_UTILS_REPO:-/home/jaeyongjang/project.local/measurement_utils}"
    local meas_python="${POWER_MEASUREMENT_PYTHON:-/home/jaeyongjang/anaconda3/envs/imcflow/bin/python}"
    local result_root="${POWER_RESULT_BASE_PATH:-/tmp/power_tagged_measurements}"
    local local_dmm_config="$SCRIPT_DIR/dmm_configs/dmm_gpib3.json"
    local remote_dmm_config="${POWER_DMM_CONFIG_REMOTE:-/tmp/imcflow_dmm_gpib3.json}"
    local probe=""
    local attempt

    probe="$(power_probe_server || true)"
    if [[ -z "$probe" ]]; then
        echo "[POWER] starting direct-PyVISA daemon on $result_ssh:$POWER_MEASUREMENT_PORT"
        scp "$local_dmm_config" "$result_ssh:$remote_dmm_config" || return 1
        ssh "$result_ssh" "cd $meas_repo && nohup $meas_python -m ps_ctrl.cli.power_tagged_measurement_server --host 0.0.0.0 --port $POWER_MEASUREMENT_PORT --config $remote_dmm_config --result-root $result_root --log-file /tmp/power_tagged_measurement.log </dev/null >/tmp/power_tagged_measurement.stdout 2>&1 &" || return 1
        for attempt in $(seq 1 20); do
            probe="$(power_probe_server || true)"
            [[ -n "$probe" ]] && break
            sleep 0.25
        done
    fi
    if [[ "$probe" != "HELLO_OK 2 $POWER_MASTER_MEASUREMENT_REV" ]]; then
        echo "Error: tagged measurement server revision/protocol mismatch: ${probe:-unreachable}" >&2
        return 1
    fi
    echo "[POWER] measurement server ready: $probe"
}


power_prepare() {
    local requested_config="$1"
    local model_name="$2"
    local local_result_parent="$3"
    shift 3
    local config="$requested_config"
    local safe_model
    local timestamp
    local short_rev
    local request_dir
    local prepare_args

    if [[ -z "$config" ]]; then
        config="${IMCFLOW_POWER_CONFIG:-}"
    fi
    if [[ -z "$config" ]]; then
        POWER_ENABLED=false
        return 0
    fi
    if [[ ! -f "$config" && -f "$SCRIPT_DIR/$config" ]]; then
        config="$SCRIPT_DIR/$config"
    fi
    if [[ ! -f "$config" ]]; then
        echo "Error: power config not found: $config" >&2
        return 1
    fi
    python -c 'import json, numpy, sys; assert sys.version_info >= (3, 8)' || {
        echo "Error: run 'activate' on the master before using power measurement" >&2
        return 1
    }
    python "$SCRIPT_DIR/scripts/power_request.py" config-status "$config"
    local config_status=$?
    if [[ $config_status -eq 10 ]]; then
        POWER_ENABLED=false
        return 0
    elif [[ $config_status -ne 0 ]]; then
        return $config_status
    fi
    power_validate_endpoint || return 1
    power_revision_preflight || return 1
    power_ensure_server || return 1

    safe_model="$(power_sanitize_id "$model_name")"
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    short_rev="${POWER_MASTER_TVM_REV:0:8}"
    POWER_SESSION_ID="${timestamp}_${safe_model}_${short_rev}_$$"
    POWER_SESSION_ID="${POWER_SESSION_ID:0:128}"
    request_dir="$local_result_parent/power_requests"
    mkdir -p "$request_dir"
    POWER_LOCAL_REQUEST="$request_dir/$POWER_SESSION_ID.json"
    POWER_REMOTE_REQUEST="/var/volatile/imcflow_power_$POWER_SESSION_ID.json"
    POWER_LOCAL_RESULT_DIR="$local_result_parent/power/$POWER_SESSION_ID"

    prepare_args=(
        python "$SCRIPT_DIR/scripts/power_request.py" prepare
        --config "$config"
        --output "$POWER_LOCAL_REQUEST"
        --session-id "$POWER_SESSION_ID"
        --metadata "tvm_git_rev=$POWER_MASTER_TVM_REV"
        --metadata "board_tvm_git_rev=$POWER_BOARD_TVM_REV"
        --metadata "measurement_utils_git_rev=$POWER_MASTER_MEASUREMENT_REV"
        --metadata "board_measurement_utils_git_rev=$POWER_BOARD_MEASUREMENT_REV"
        --metadata "measurement_server_utils_git_rev=$POWER_SERVER_MEASUREMENT_REV"
    )
    if [[ -n "${POWER_REMOTE_BINARY:-}" ]]; then
        prepare_args+=(
            --metadata "binary_tvm_git_rev=$POWER_BINARY_TVM_REV"
            --metadata "binary_measurement_utils_git_rev=$POWER_BINARY_MEASUREMENT_REV"
        )
    fi
    if [[ -n "${POWER_BUILD_METADATA:-}" ]]; then
        prepare_args+=(
            --metadata "codegen_tvm_git_rev=$POWER_CODEGEN_TVM_REV"
            --metadata "codegen_measurement_utils_git_rev=$POWER_CODEGEN_MEASUREMENT_REV"
        )
    fi
    while [[ $# -gt 0 ]]; do
        prepare_args+=(--metadata "$1")
        shift
    done
    "${prepare_args[@]}"
    local prepare_status=$?
    if [[ $prepare_status -eq 10 ]]; then
        POWER_ENABLED=false
        return 0
    elif [[ $prepare_status -ne 0 ]]; then
        return $prepare_status
    fi

    scan_scp "$POWER_LOCAL_REQUEST" "$REMOTE_USER@$REMOTE_HOST:$POWER_REMOTE_REQUEST" || return 1
    POWER_ENABLED=true
    echo "[POWER] session prepared: $POWER_SESSION_ID"
}


power_remote_environment() {
    if [[ "$POWER_ENABLED" != true ]]; then
        return 0
    fi
    printf 'IMCFLOW_POWER_REQUEST=%s POWER_MEASUREMENT_HOST=%s POWER_MEASUREMENT_PORT=%s ' \
        "$POWER_REMOTE_REQUEST" "$POWER_MEASUREMENT_HOST" "$POWER_MEASUREMENT_PORT"
}


power_fetch_result() {
    local result_ssh="${POWER_RESULT_SSH_HOST:-meas-2}"
    local result_root="${POWER_RESULT_BASE_PATH:-/tmp/power_tagged_measurements}"
    local local_parent
    local attempt
    if [[ "$POWER_ENABLED" != true ]]; then
        return 0
    fi
    for attempt in $(seq 1 20); do
        if ssh "$result_ssh" "test -f $result_root/$POWER_SESSION_ID/summary.json"; then
            break
        fi
        sleep 0.25
    done
    local_parent="$(dirname "$POWER_LOCAL_RESULT_DIR")"
    mkdir -p "$local_parent"
    if ! scp -r "$result_ssh:$result_root/$POWER_SESSION_ID" "$local_parent/"; then
        echo "Error: failed to fetch power session $POWER_SESSION_ID" >&2
        echo "Retry: scp -r $result_ssh:$result_root/$POWER_SESSION_ID $local_parent/" >&2
        return 1
    fi
    python "$SCRIPT_DIR/scripts/power_request.py" validate-result "$POWER_LOCAL_RESULT_DIR"
}


power_finalize_run() {
    local workload_status="$1"
    POWER_STATUS=0
    if [[ "$POWER_ENABLED" == true ]]; then
        power_fetch_result || POWER_STATUS=$?
    fi
    if [[ $workload_status -ne 0 ]]; then
        return "$workload_status"
    fi
    return "$POWER_STATUS"
}
