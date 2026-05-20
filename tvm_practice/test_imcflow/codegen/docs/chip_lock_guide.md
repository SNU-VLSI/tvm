# Chip Lock Guide

FPGA 칩은 단일 사용자 리소스이므로, 동시 접근 시 충돌을 방지하기 위해 락 메커니즘을 사용한다. Shell과 Python 두 가지 구현이 존재하며, 동일한 원격 락 파일을 공유한다.

## 공통 개요

- 락 파일: 원격 FPGA 서버의 `/tmp/imcflow_user.lock`
- 사용자 식별: `~/.imcflow.env`의 `USER_ID` 값
- stale 감지: 원격에서 `test_imcflow|execute_graph|program_scan_reg` 프로세스 존재 여부로 판단
- 기존 락이 존재하면 stale 여부와 관계없이 항상 abort (자동 삭제하지 않음)

| 구현 | 위치 | 사용처 |
|---|---|---|
| Shell | `codegen/chip_lock.sh` | `run_chiptest.sh`, `run_dataset_eval.sh` |
| Python | `imcflow/xilinx/measurement/common/chip_lock.py` | `run_planner.py`, `run_eval.py` |

---

## 1. Shell 패턴 (codegen)

### API

| 함수 | 역할 |
|---|---|
| `chip_lock_acquire "<script_name>"` | 락 획득. 이미 잠겨있거나 칩 미도달 시 `exit 1` |
| `chip_lock_release` | 락 해제. 칩 미도달 시 경고만 출력 |
| `chip_lock_status` | 현재 락 상태 조회 (FREE / BUSY / UNREACHABLE) |

### 필수 구조

```bash
#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scan_steps.sh"   # chip_lock.sh + load_env 포함
load_env                              # .env에서 REMOTE_* 변수 로드

# ── 로컬 작업 (빌드, 전처리 등) ──
# 칩이 필요 없는 작업은 락 획득 전에 수행한다.

# ── 칩 락 획득 ──
chip_lock_acquire "my_script.sh"
trap 'chip_lock_release' EXIT         # 어떤 경로로든 종료 시 반드시 해제

# ── 원격 작업 (전송, 실행 등) ──
# trap에 의해 자동 release
```

### 핵심 규칙

1. **`source scan_steps.sh` + `load_env`로 시작** -- `chip_lock.sh`가 `scan_steps.sh` 내부에서 source되므로 `scan_steps.sh`만 source하면 된다.
2. **로컬 작업 먼저, 락 획득은 최대한 늦게** -- 빌드/전처리를 먼저 완료하여 칩 점유 시간을 최소화한다.
3. **`trap 'chip_lock_release' EXIT` 필수** -- 정상 종료, 에러, Ctrl+C 모두 커버한다.
4. **`chip_lock_acquire`에 스크립트 이름 전달** -- 락 파일에 기록되어 점유자를 식별할 수 있다.

### 기존 예시

- `run_chiptest.sh`: Step 1(빌드) 후 락 획득 -> Step 2~6(전송+실행)
- `run_dataset_eval.sh`: Step 1(빌드) 후 락 획득 -> Step 2~7(전송+실행+결과 회수)

---

## 2. Python 패턴 (measurement)

### API

```python
from common.chip_lock import ChipLock

# 방법 1: 명시적 acquire/release + atexit
lock = ChipLock(ssh_client, script_name="my_script.py")
lock.acquire()
import atexit
atexit.register(lock.release)

# 방법 2: context manager
with ChipLock(ssh_client, script_name="my_script.py"):
    ...  # chip work
```

| 메서드 | 역할 |
|---|---|
| `lock.acquire(script_name=None)` | 락 획득. 실패 시 `SystemExit(1)` raise |
| `lock.release()` | 락 해제. 칩 미도달 시 경고 로그만 출력 |
| `lock.status()` | 현재 락 상태 조회, 락 내용 또는 `None` 반환 |

SSH 연결 실패 시 `ChipUnreachableError`를 내부적으로 사용하며, `acquire()`에서 이를 잡아 `SystemExit(1)`로 변환한다.

### 필수 구조

```python
from common import client
from common.chip_lock import ChipLock
import atexit

# 1. SSH 연결
ssh_client = client.open_ssh(host, user, port, keyfile=keyfile)

# 2. 로컬 작업 (인자 파싱, 디렉토리 생성, 설정 로드 등)
#    칩이 필요 없는 작업은 락 획득 전에 수행한다.

# 3. 칩 락 획득 (dryrun이 아닐 때만)
if not args.dryrun:
    chip_lock = ChipLock(ssh_client, script_name="my_script.py (details)")
    chip_lock.acquire()
    atexit.register(chip_lock.release)

# 4. 칩 작업 수행
```

### 핵심 규칙

1. **`atexit.register(lock.release)` 또는 `with` 문 사용** -- 예외, KeyboardInterrupt 등 어떤 경로로든 종료 시 락이 해제되어야 한다.
2. **dryrun 모드에서는 락을 획득하지 않는다** -- `run_planner.py`와 `run_eval.py` 모두 `--dryrun` 시 락 획득을 건너뛴다.
3. **script_name에 상세 정보 포함** -- `"run_planner.py (GAPlanner)"` 처럼 어떤 작업인지 식별 가능하게 한다.
4. **락용 SSH 클라이언트는 별도로 열어도 된다** -- `run_planner.py`는 작업용과 별개의 `lock_ssh`를 사용한다.

### 기존 예시

- `run_planner.py`: argparse 후 락 획득 -> `planner.run()` -> atexit으로 release
- `run_eval.py`: 설정/연결 후 락 획득 -> executor 실행 -> atexit으로 release

---

## 공통 주의사항

- **stale 락 수동 삭제**: 프로세스가 비정상 종료(kill -9, 네트워크 단절 등)하면 원격 락 파일이 남을 수 있다. `chip_lock_status` (shell) 또는 `lock.status()` (Python)로 확인 후, stale이면 `ssh <remote> 'rm /tmp/imcflow_user.lock'`으로 삭제한다.
- **중첩 락 금지**: 하나의 프로세스에서 `acquire`를 두 번 호출하면 두 번째에서 자기 자신의 락에 의해 abort된다.
- **Shell과 Python은 동일한 락 파일을 공유한다** -- codegen 스크립트가 칩을 점유 중이면 measurement 스크립트도 차단되고, 그 반대도 마찬가지다.

## Claude에게: 칩 사용 스크립트 작성 시

새로운 칩 사용 스크립트를 작성할 때 반드시 이 가이드의 패턴을 따를 것:

- **Shell**: `source scan_steps.sh` -> `load_env` -> 로컬 작업 -> `chip_lock_acquire` -> `trap EXIT` -> 원격 작업
- **Python**: SSH 연결 -> 로컬 작업 -> `ChipLock.acquire()` -> `atexit.register(release)` -> 칩 작업
- 락 획득 시점은 최대한 늦추고, 해제는 자동화(trap/atexit/with)할 것
- dryrun 옵션이 있다면 락 획득을 건너뛸 것
