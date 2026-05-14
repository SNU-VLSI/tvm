#!/bin/bash
# Fetch deploy debug pkl from GPU server using checkpoint alias.
#
# Usage:
#   ./scripts/sync_artifacts.sh [ckpt_alias]
#
# Examples:
#   ./scripts/sync_artifacts.sh                        # uses default alias
#   ./scripts/sync_artifacts.sh mapaware_bugfix_ndis32
#   ./scripts/sync_artifacts.sh col_ndis32
#   BOARD=B2 VMODE=half ./scripts/sync_artifacts.sh concat_ndis16
#
# The checkpoint alias is resolved via CIM/checkpoints registry.
# The pkl is expected at <resolved_imcflow_path>/debug_model.with_noise.pkl
# on the GPU server.
#
# Environment:
#   BOARD     — board name (default: B2)
#   VMODE     — voltage mode (default: half)
#   GPU_USER  — override SSH user (default: jihoon.park)
#   GPU_HOST  — override SSH host (default: 147.46.91.206)
#   GPU_PORT  — override SSH port (default: 1326)
#   GPU_CIM   — override CIM project root on GPU server
#              (default: /home/${GPU_USER}/Project/CIM)

set -euo pipefail

CODEGEN_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CIM_DIR="${CIM_DIR:-/root/project/CIM}"

GPU_USER="${GPU_USER:-jihoon.park}"
GPU_HOST="${GPU_HOST:-147.46.91.206}"
GPU_PORT="${GPU_PORT:-1326}"
GPU_CIM="${GPU_CIM:-/home/${GPU_USER}/Project/CIM}"

BOARD="${BOARD:-B2}"
VMODE="${VMODE:-half}"
CKPT_ALIAS="${1:-}"

# Resolve alias to relative path (e.g. trained_models/.../imcflow/<ts>)
REL_PATH=$(python3 -c "
import json, os, sys
registry_dir = os.path.join('${CIM_DIR}', 'checkpoints')
name = '${BOARD}'.lower() + '_' + '${VMODE}'.lower()
with open(os.path.join(registry_dir, name + '.json')) as f:
    reg = json.load(f)
key = '${CKPT_ALIAS}' or reg['default']
if key not in reg['entries']:
    avail = list(reg['entries'].keys())
    print(f\"ERROR: unknown alias '{key}'. Available: {avail}\", file=sys.stderr)
    sys.exit(1)
print(os.path.join(reg['_base'], reg['entries'][key]))
")

SCP_CMD="scp -P ${GPU_PORT}"
REMOTE="${GPU_USER}@${GPU_HOST}"

# Resolve the actual alias key (for directory naming)
RESOLVED_ALIAS=$(python3 -c "
import json, os
name = '${BOARD}'.lower() + '_' + '${VMODE}'.lower()
with open(os.path.join('${CIM_DIR}', 'checkpoints', name + '.json')) as f:
    reg = json.load(f)
print('${CKPT_ALIAS}' or reg['default'])
")

src="${GPU_CIM}/${REL_PATH}/debug_model.with_noise.pkl"
dst_dir="${CODEGEN_DIR}/debugging/${RESOLVED_ALIAS}"
dst="${dst_dir}/debug_model.with_noise.pkl"

mkdir -p "$dst_dir"

echo "Alias: ${CKPT_ALIAS:-$(python3 -c "
import json, os
with open(os.path.join('${CIM_DIR}', 'checkpoints', '${BOARD}'.lower() + '_' + '${VMODE}'.lower() + '.json')) as f:
    print(json.load(f)['default'])
")} (default)"
echo "Fetching ${REMOTE}:${src}"
echo "     -> ${dst}"
${SCP_CMD} "${REMOTE}:${src}" "$dst"
echo "Done."
