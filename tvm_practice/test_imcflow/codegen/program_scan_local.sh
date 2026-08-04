#!/bin/bash
# program_scan_local.sh — 보드(petalinux) 안에서 직접 scan register 를 프로그램한다.
#
# 원래 이 동작은 다른 서버에서 run_dataset_eval.sh 의 Step 5 로 실행됐다. 그 스텝은
# scan_steps.sh:scan_program_registers() 가 ssh 로 이 보드에 들어와 커맨드를 던지는
# 구조라, 보드에 이미 앉아 있을 때는 쓸 수 없었다. 이 스크립트는 같은 커맨드를 ssh 없이
# 그대로 실행한다.
#
# 원본(scan_steps.sh)과 맞춘 부분:
#   - venv 활성화 후 실행 (program_scan_reg 가 python 런타임에 의존)
#   - `timeout -s INT 0.5s` : 원본과 동일. 이 프로그램은 레지스터를 다 쓴 뒤 스스로
#     끝나지 않으므로 0.5초 뒤 SIGINT 로 끊는 게 정상 동작이다. 타임아웃으로 인한
#     종료코드(124/130)를 실패로 보면 안 된다.
#   - 프로그램 직후 clear_time + warmup (IMC 아날로그 코어를 깨운다)
#
# 락: 다른 서버의 스크립트들이 쓰는 /tmp/imcflow_user.lock 규약을 그대로 따른다.
#     보드에서 직접 돌리는 동안 원격에서 들어오는 작업이 겹치지 않게 하려면 필요하다.
#     (chip_lock.sh 는 ssh 로 이 파일을 읽고 쓴다 — 형식이 같아야 서로 보인다.)
#
# 사용:
#   ./program_scan_local.sh              # 락 잡고 scan 프로그램 + warmup
#   ./program_scan_local.sh --no-lock    # 락 없이 (이미 상위 스크립트가 잡은 경우)
#   ./program_scan_local.sh --force      # 남의 락이 있어도 진행 (스테일 확신할 때만)
#   ./program_scan_local.sh --status     # 락 상태만 보고 종료
#   NPZ_DIR=scan_reg_files_alt ./program_scan_local.sh
set -uo pipefail

BASE="${BASE:-/home/root/tvm/tvm_practice/test_imcflow/codegen}"
NPZ_DIR="${NPZ_DIR:-scan_reg_files}"
VENV="${VENV:-/home/root/.venv}"
WARMUP_DIR="${WARMUP_DIR:-/home/root/imcflow/xilinx/petalinux-csrc}"
LOCKFILE="${CHIP_LOCKFILE:-/tmp/imcflow_user.lock}"
SCAN_TIMEOUT="${SCAN_TIMEOUT:-0.5s}"

USE_LOCK=1; FORCE=0
for a in "$@"; do
  case "$a" in
    --no-lock) USE_LOCK=0 ;;
    --force)   FORCE=1 ;;
    --status)  if [[ -s "$LOCKFILE" ]]; then echo "[lock] BUSY:"; sed 's/^/  /' "$LOCKFILE";
               else echo "[lock] FREE"; fi; exit 0 ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

SCAN_BUILD="$BASE/scan_gen/scan_executable_make/build"
NPZ_PATH="$BASE/scan_gen/$NPZ_DIR"

# ── 사전 점검 ── 없는 경로로 들어가면 program_scan_reg 가 조용히 아무것도 안 하고 끝난다
for p in "$SCAN_BUILD/program_scan_reg" "$NPZ_PATH" "$VENV/bin/activate" "$WARMUP_DIR"; do
  [[ -e "$p" ]] || { echo "[error] 없음: $p" >&2; exit 1; }
done
npz_count=$(ls "$NPZ_PATH"/*.npz 2>/dev/null | wc -l)
[[ "$npz_count" -gt 0 ]] || { echo "[error] $NPZ_PATH 에 npz 가 없습니다" >&2; exit 1; }
echo "[info] npz ${npz_count}개  ($NPZ_PATH)"

# ── 락 ──
release_lock() { [[ "$USE_LOCK" == 1 ]] && rm -f "$LOCKFILE"; }
if [[ "$USE_LOCK" == 1 ]]; then
  if [[ -s "$LOCKFILE" && "$FORCE" != 1 ]]; then
    echo "[lock] 다른 작업이 칩을 쓰고 있습니다 — 중단"
    sed 's/^/  /' "$LOCKFILE"
    # 락은 있는데 칩 관련 프로세스가 없으면 스테일일 수 있다(chip_lock.sh 와 같은 판정).
    if ! pgrep -f 'test_imcflow|execute_graph|program_scan_reg' >/dev/null 2>&1; then
      echo "  (칩 프로세스가 안 보입니다 — 스테일일 수 있음. 확실하면 --force 또는 rm $LOCKFILE)"
    fi
    exit 1
  fi
  cat > "$LOCKFILE" <<LOCKEOF
user_id: $(id -un)@$(hostname)
script: program_scan_local.sh
started: $(date '+%Y-%m-%d %H:%M:%S')
LOCKEOF
  trap release_lock EXIT INT TERM HUP   # 어떤 경로로 끝나도 락은 푼다
  echo "[lock] 획득"
fi

# ── scan programming ──
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$SCAN_BUILD" || exit 1
echo "[scan] program_scan_reg $NPZ_PATH  (timeout -s INT $SCAN_TIMEOUT)"
timeout -s INT "$SCAN_TIMEOUT" ./program_scan_reg "$NPZ_PATH"
rc=$?
# 124=timeout, 130=SIGINT — 원본이 의도한 정상 종료 경로다. 0 도 정상.
if [[ $rc -ne 0 && $rc -ne 124 && $rc -ne 130 ]]; then
  echo "[error] program_scan_reg 실패 (exit $rc)" >&2
  exit 1
fi
echo "[scan] 완료 (exit $rc)"

# ── warmup ── IMC 아날로그 코어를 깨운다. 이거 없으면 추론이 조용히 전부 한 클래스만 낸다.
cd "$WARMUP_DIR" || exit 1
echo "[warmup] clear_time && warmup"
make clear_time >/dev/null 2>&1
if ! make warmup >/dev/null 2>&1; then
  echo "[error] warmup 실패" >&2
  exit 1
fi
echo "[warmup] 완료"
echo "[done] scan programming + warmup 정상 종료"
