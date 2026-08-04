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

# 포트 선점 확인을 맨 앞에서 한다. 안 그러면 venv 설치를 다 마친 뒤에야 uvicorn 이
# "address already in use" 를 뱉는데, 그 메시지가 "Application startup complete" 뒤에
# 묻혀서 원인이 잘 안 보인다.
if command -v ss >/dev/null 2>&1 && ss -ltnH "sport = :$PORT" 2>/dev/null | grep -q .; then
  echo "[error] 포트 $PORT 를 이미 다른 프로세스가 쓰고 있습니다."
  ss -ltnpH "sport = :$PORT" 2>/dev/null | sed 's/^/        /'
  echo "        먼저 내리세요:  ./stop_demo.sh"
  echo "        다른 포트로 띄우려면:  PORT=8080 $0 ${*:-}"
  exit 1
fi

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

# 사진 소스는 config 의 image_source 가 정한다 (resnet8=full_npy, kws/vww=staged_npy).
# ⚠️ 예전엔 여기서 full_npy 를 강제 export 했는데, 그러면 kws/vww 의 staged_npy 까지
#    덮어써 MFCC/COCO 대신 CIFAR 사진이 뜬다(결과는 kws 인데 화면은 고양이 사진).
#    환경변수는 사용자가 명시했을 때만 오버라이드로 동작한다.
if [[ -n "${DEMO_IMAGE_SOURCE:-}" ]]; then
  echo "[run] image_source = ${DEMO_IMAGE_SOURCE}  (환경변수 오버라이드)"
else
  echo "[run] image_source = config 값 사용 (오버라이드하려면 DEMO_IMAGE_SOURCE=full_npy|staged_npy)"
fi
if [[ "${DEMO_WORKLOAD:-resnet8}" == "resnet8" && ! -s fixtures/cifar10_test_images.npy ]]; then
  echo "[warn] fixtures/cifar10_test_images.npy 가 비어있음 — git-lfs pull 이 필요할 수 있습니다:"
  echo "       git lfs install && git lfs pull"
fi
# 칩 stdout 원본 로그. 웹앱 로그(uvicorn)에는 HTTP 접근 기록만 남고 칩 출력은 파서가
# 소비해 버리므로, 원본이 필요하면 이걸 켠다. tail -f 로 실시간 확인 가능.
if [[ -n "${DEMO_CHIP_RAW_LOG:-}" ]]; then
  echo "[run] chip raw log = ${DEMO_CHIP_RAW_LOG}   (tail -f ${DEMO_CHIP_RAW_LOG})"
else
  echo "[run] chip raw log = off  (켜려면 DEMO_CHIP_RAW_LOG=/tmp/chip_raw.log $0 ...)"
fi
echo "[run] 서버 기동: http://127.0.0.1:${PORT}  (Ctrl-C 로 종료)"
cd backend
exec python3 -m uvicorn app:app --host 127.0.0.1 --port "$PORT"
