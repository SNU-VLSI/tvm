#!/usr/bin/env bash
# Copy a local DMM config to meas-2 and start a direct-PyVISA bridge there.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEGEN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MEAS_HOST="meas-2"
PORT=9911
CONFIG="${CODEGEN_DIR}/power_config/dmm_gpib124.json"
REMOTE_DIR="/tmp/imcflow_power_config"
LOG_PATH=""
EXPECTED_DMM_NAMES="${IMCFLOW_POWER_DMM_NAMES:-}"

usage() {
  cat <<'EOF'
Usage: start_power_bridge_meas2.sh [options]
  --config PATH     local JSON config
  --port PORT       meas-2 listen port (default: 9911)
  --host HOST       SSH host alias (default: meas-2)
  --log-file PATH   meas-2 bridge log path
  --expected-dmm-names NAMES
                     require this comma-separated logical-name order
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --host) MEAS_HOST="$2"; shift 2 ;;
    --log-file) LOG_PATH="$2"; shift 2 ;;
    --expected-dmm-names) EXPECTED_DMM_NAMES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
[[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT >= 1024 && PORT <= 65535 )) || {
  echo "Invalid port: $PORT" >&2; exit 2;
}

CONFIG_DMM_NAMES="$(python3 -c '
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    config = json.load(stream)
power = config.get("POWER")
if not isinstance(power, dict) or not power:
    raise SystemExit("config POWER must be a non-empty object")
print(",".join(power))
' "$CONFIG")"
if [[ -n "$EXPECTED_DMM_NAMES" && "$CONFIG_DMM_NAMES" != "$EXPECTED_DMM_NAMES" ]]; then
  echo "DMM logical-name mismatch: config=${CONFIG_DMM_NAMES}, expected=${EXPECTED_DMM_NAMES}" >&2
  exit 2
fi

REMOTE_CONFIG="${REMOTE_DIR}/$(basename "$CONFIG")"
LOG_PATH="${LOG_PATH:-/tmp/power_v2_bridge_${PORT}.log}"
ssh -o BatchMode=yes "$MEAS_HOST" "mkdir -p '${REMOTE_DIR}'"
scp -q "$CONFIG" "${MEAS_HOST}:${REMOTE_CONFIG}"

# Never replace an existing bridge; choose another port if it is occupied.
ssh -o BatchMode=yes "$MEAS_HOST" \
  "PORT='${PORT}' LOG_PATH='${LOG_PATH}' CONFIG='${REMOTE_CONFIG}' bash -s" <<'REMOTE'
set -euo pipefail
if ss -ltn "sport = :${PORT}" | grep -q LISTEN; then
  echo "A process is already listening on port ${PORT}; refusing to replace it." >&2
  exit 3
fi
source /home/jaeyongjang/anaconda3/etc/profile.d/conda.sh
conda activate imcflow
nohup measure-bridge-daemon --host 0.0.0.0 --port "${PORT}" \
  --config "${CONFIG}" --log-file "${LOG_PATH}" --log-level INFO \
  >/dev/null 2>&1 &
echo $! > "/tmp/imcflow_power_bridge_${PORT}.pid"
sleep 1
kill -0 "$(cat "/tmp/imcflow_power_bridge_${PORT}.pid")" 2>/dev/null || {
  echo "Bridge exited; inspect ${LOG_PATH}" >&2
  exit 4
}
REMOTE

echo "Bridge started: ${MEAS_HOST}:${PORT}"
echo "Config on meas-2: ${REMOTE_CONFIG}"
echo "Logical DMM names: ${CONFIG_DMM_NAMES}"
echo "Log on meas-2: ${LOG_PATH}"
echo "Set DMM_BRIDGE_HOST to meas-2's board-facing IP and DMM_BRIDGE_PORT=${PORT}."
