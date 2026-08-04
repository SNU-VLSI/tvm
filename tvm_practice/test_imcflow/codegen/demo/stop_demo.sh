#!/usr/bin/env bash
# 데모 웹앱 강제 종료.
#
# 왜 포트로 찾나 (이름으로 찾지 않는 이유):
#   pgrep -f "uvicorn app:app" 방식은 세 가지로 새는 게 확인됐다.
#   (1) run_laptop.sh 래퍼와 uvicorn 이 쌍으로 잡혀 `head -1` 로 죽이면 절반만 죽는다
#   (2) 검색 커맨드 자신의 커맨드라인이 매칭돼 자기를 죽인다
#   (3) 서버가 두 벌 떠 있으면 하나만 죽고 나머지가 계속 8079 를 잡는다
#   실제로 그렇게 살아남은 구버전 서버가 칩 추론을 중복 실행해 보드가 wedge 되고
#   전원 재인가까지 간 적이 있다(2026-07-30). 포트를 쥔 PID 가 유일한 진실이다.
#
# 사용:
#   ./stop_demo.sh            # 8079 를 쥔 프로세스 + run_laptop.sh 래퍼 종료
#   PORT=9000 ./stop_demo.sh  # 다른 포트
#   ./stop_demo.sh --ssh      # 칩에 남은 추론 ssh 세션까지 정리 (스윕 직전에 유용)
set -uo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8079}"
KILL_SSH=0
[[ "${1:-}" == "--ssh" ]] && KILL_SSH=1

# 포트를 LISTEN 중인 PID 들. ss -> lsof -> fuser 순으로 있는 걸 쓴다.
listeners() {
  if command -v ss >/dev/null 2>&1; then
    ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u
  elif command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | sort -u
  else
    fuser -n tcp "$PORT" 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$' | sort -u
  fi
}

kill_pids() {  # $1=signal, 나머지=pids
  local sig="$1"; shift
  for p in "$@"; do
    [[ "$p" == "$$" || "$p" == "$PPID" ]] && continue   # 자기 자신은 건드리지 않는다
    kill "-$sig" "$p" 2>/dev/null && echo "  $sig -> $p ($(ps -o comm= -p "$p" 2>/dev/null || echo '?'))"
  done
}

pids=$(listeners)
if [[ -z "$pids" ]]; then
  echo "[stop] :$PORT 을 쥔 프로세스 없음"
else
  echo "[stop] :$PORT 리슨 중: $(echo "$pids" | tr '\n' ' ')"
  kill_pids TERM $pids
  for _ in $(seq 10); do            # SIGTERM 으로 정리될 시간을 준다 (최대 5초)
    sleep 0.5
    [[ -z "$(listeners)" ]] && break
  done
  left=$(listeners)
  if [[ -n "$left" ]]; then
    echo "[stop] 안 죽어서 SIGKILL"
    kill_pids KILL $left
    sleep 1
  fi
fi

# uvicorn 을 띄운 run_laptop.sh 래퍼가 남으면 다음 기동 때 헷갈린다. 같이 정리.
wrappers=$(pgrep -f "run_laptop\.sh" 2>/dev/null | grep -v "^$$\$" || true)
if [[ -n "$wrappers" ]]; then
  echo "[stop] run_laptop.sh 래퍼 정리"
  kill_pids TERM $wrappers
fi

if [[ "$KILL_SSH" == "1" ]]; then
  # 칩에서 도는 추론까지 정리. 로컬 ssh 만 죽이면 원격 프로세스가 남을 수 있어 원격 pkill 도 한다.
  # ⚠️ 패턴의 첫 글자를 [] 로 감싸지 않으면 pkill 이 자기를 담은 래퍼 셸까지 죽여
  #    커맨드 꼬리의 clear_time+warmup 이 실행되지 않는다.
  sshpids=$(pgrep -f "ssh -o BatchMode=yes -p" 2>/dev/null || true)
  if [[ -n "$sshpids" ]]; then
    echo "[stop] 칩 ssh 세션 정리"
    kill_pids KILL $sshpids
  fi
  echo "[stop] 원격 추론 프로세스 정리"
  ssh -o BatchMode=yes -o ConnectTimeout=10 petalinux2 \
      'pkill -f "[e]xecute_graph_for_dataset"' 2>/dev/null
  echo "  원격 잔여: $(ssh -o BatchMode=yes -o ConnectTimeout=10 petalinux2 \
      'pgrep -fc "[e]xecute_graph_for_dataset" || echo 0' 2>/dev/null || echo '확인 실패')"
fi

if [[ -z "$(listeners)" ]]; then
  echo "[stop] ✅ :$PORT 해제됨"
else
  echo "[stop] ❌ :$PORT 아직 점유 중: $(listeners | tr '\n' ' ')"
  exit 1
fi
