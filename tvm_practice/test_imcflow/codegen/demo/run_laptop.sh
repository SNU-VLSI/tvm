#!/usr/bin/env bash
# 노트북(WSL) mock 데모 실행 스크립트.
# 칩 연결 불필요 — fixture 를 재생해 프론트를 육안 확인한다.
#
# 사용:
#   ./run_laptop.sh            # 가상환경 만들고 의존성 설치 후 서버 기동
#   ./run_laptop.sh --no-setup # 설치 건너뛰고 바로 기동 (이미 셋업된 경우)
#
# 기동되면 브라우저에서 http://127.0.0.1:8079 접속.
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
PORT="${PORT:-8079}"

if [[ "${1:-}" != "--no-setup" ]]; then
  echo "[setup] creating venv + installing requirements (최초 1회, torchvision 다운로드로 수 분 걸릴 수 있음)..."
  python3 -m venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  pip install --quiet --upgrade pip
  pip install --quiet -r requirements.txt
else
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

# 노트북 모드: 사진 소스 = git-lfs 로 박제된 full npy (다운로드 불필요)
export DEMO_IMAGE_SOURCE="${DEMO_IMAGE_SOURCE:-full_npy}"
echo "[run] image_source = ${DEMO_IMAGE_SOURCE}  (staged_npy 로 바꾸려면 DEMO_IMAGE_SOURCE=staged_npy)"
if [[ ! -s fixtures/cifar10_test_images.npy ]]; then
  echo "[warn] fixtures/cifar10_test_images.npy 가 비어있음 — git-lfs pull 이 필요할 수 있습니다:"
  echo "       git lfs install && git lfs pull"
fi
echo "[run] 서버 기동: http://127.0.0.1:${PORT}  (Ctrl-C 로 종료)"
cd backend
exec python3 -m uvicorn app:app --host 127.0.0.1 --port "$PORT"
