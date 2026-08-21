# Power v2 legacy protocol 이식 계획

## 1. 목표

`power_v2`는 tagged measurement 구현을 재사용하지 않고 처음부터 다시 구성한다.

- TVM은 현재 `chip_acc_measure`를 기준으로 `power_v2` 브랜치를 만든다.
- measurement_utils는 현재 `main`을 기준으로 `power_v2` 브랜치를 만든다.
- board와 measurement server 사이는 기존 `power` 브랜치가 사용했던 단순 TCP
  protocol을 사용한다.
- TVM의 기존 `power` 브랜치를 직접 merge하지 않는다. 필요한 power 측정 코드만
  현재 `chip_acc_measure` 코드 구조에 맞춰 수동으로 이식한다.
- tag, session ID, clock synchronization, reading metadata 정렬, region loop 및
  tagged artifact 형식은 도입하지 않는다.
- power 측정을 끄면 현재 `chip_acc_measure`와 실행 동작이 같아야 한다.

## 2. 확인된 기준점

### TVM

- 시작 브랜치: `chip_acc_measure`
- 시작 커밋: `dc6e7b3e280591f56cddca5927c2bd40b025d709`
- 참고 브랜치: `power`
- 참고 HEAD: `d347280abe65ab6c5b5eb36f8f9574c68f2526d5`
- 두 브랜치의 merge base: `0ae5f6125030d834a3273918934f122e3944683f`

`chip_acc_measure..power`에는 power와 무관한 오래된 codegen, 모델, dataset,
debugging 변경도 많이 포함되어 있다. 따라서 일반 merge는 사용하지 않는다.

기존 `power`에서 참고할 주요 커밋은 다음과 같다.

| 커밋 | 참고할 내용 | 처리 |
|---|---|---|
| `40f50670f2` | Linux generated kernel의 DMM start/end, CMake 연결 | 현재 코드에 수동 이식 |
| `ffc73adfb5` | MODEL/REGION/TILE 측정 범위 | 현재 kernel 구조에 맞춰 수동 이식 |
| `c5e962e2e5` | measurement_utils submodule 및 C API 연결 | `power_v2` submodule pin으로 재구성 |
| `58788eb850` | power 실행 환경 전달 | 현재 runner script에 필요한 부분만 반영 |
| `d347280abe` | DMM 설정과 MMIO debug 변경 | DMM 설정만 참고하고 MMIO debug는 제외 |

### measurement_utils

- 시작 브랜치: `main`
- 시작 커밋: `a6ad97adf9e2474f28b3cf190c48a61ad96c3c63`
- legacy protocol 기준 커밋: `0f136b05fdadeb7800918ad9c87616130259d54f`
- `0f136b05`는 현재 `main`의 조상이다.

즉 현재 `main`에는 이미 다음 legacy 구성 요소가 들어 있다.

- `capi/dmm_measure.c`, `capi/dmm_measure.h`
- `ps_ctrl/cli/measure_bridge_daemon.py`
- `START -> STARTED -> GO -> RESULT -> CLOSE` TCP protocol
- `wait` 및 `now` mode
- DMM exclusive reservation과 관련 test

따라서 measurement_utils의 `power_v2`는 기존 protocol을 다시 cherry-pick하지
않는다. `main`에서 분기한 후 현재 환경에서 protocol을 검증하고, 꼭 필요한
수정만 추가한다.

## 3. 목표 구조

```text
master server
  TVM power_v2 compile/build/run
            |
            | SSH로 실행 및 binary 배포
            v
board (PetaLinux)
  generated C code
  dmm_start_current_now() 또는 dmm_start_current()
            |
            | TCP 9900
            v
measurement server (meas-2)
  measure_bridge_daemon.py
            |
            | 기존 measurement_utils RPC/DMM 경로
            v
  PyVISA / Keysight backend / GPIB3 DMM
```

legacy TCP 흐름은 다음과 같이 유지한다.

```text
board -> START <DMM 설정> --mode <now|wait>
server -> STARTED
server -> DMM 측정 시작(GET)

board에서 지정한 power 구간 실행

board -> GO
server -> DMM 측정 종료/버퍼 read
server -> RESULT <name> avg=<value> count=<count>
board -> CLOSE
```

측정 중에는 SCP, plotting, Git 검사 또는 clock sync를 수행하지 않는다.
결과 파일을 master로 복사하는 기능이 필요하면 전체 workload가 종료된 뒤 runner가
별도 단계에서 수행하도록 한다.

## 4. 브랜치 생성과 저장소 관계

### 4.1 measurement_utils

1. submodule 작업 트리가 깨끗한지 확인한다.
2. `main`을 checkout하고 `origin/main`과 fast-forward 상태인지 확인한다.
3. `main`에서 `power_v2`를 생성한다.
4. legacy daemon/C API test가 수정 없이 통과하는 기준 commit을 먼저 만든다.
5. 필요한 보완을 작은 commit으로 추가하고 `origin/power_v2`에 push한다.

예정 브랜치 관계:

```text
measurement_utils/main (a6ad97a...)
  `-- measurement_utils/power_v2
```

### 4.2 TVM

1. 현재 untracked `_run.sh` 등 사용자 파일을 보존한다.
2. `chip_acc_measure`가 `origin/chip_acc_measure`와 같은지 확인한다.
3. `chip_acc_measure`에서 `power_v2`를 생성한다.
4. TVM submodule을 measurement_utils `power_v2`의 검증된 commit으로 pin한다.
5. 필요하면 `.gitmodules`의 추적 branch도 `power_v2`로 명시하되, 실제 build와
   배포 재현성은 항상 gitlink의 정확한 commit으로 보장한다.

예정 브랜치 관계:

```text
TVM/chip_acc_measure (dc6e7b3...)
  `-- TVM/power_v2
        `-- 3rdparty/measurement_utils @ power_v2의 고정 commit
```

## 5. measurement_utils 구현 및 검증

### 5.1 유지할 API와 protocol

다음 legacy C API를 그대로 사용한다.

- `dmm_start_current()`
- `dmm_wait_result()`
- `dmm_start_current_now()`
- `dmm_get_result_now()`
- `dmm_close()`
- `dmm_last_error()`

다음 기능은 추가하거나 이식하지 않는다.

- `dmm_session_start_file()`
- `dmm_tag_set()`, `dmm_tag_clear()`, `dmm_tag_event()`
- `HELLO`, `SYNC`, `CLOCK_SYNC`, `START_JSON`, `STOP`, `ABORT`
- tagged session artifact와 tag별 sample 분류

### 5.2 최소 보완 후보

먼저 기존 코드를 그대로 검증하고 실패할 때만 다음을 수정한다.

- meas-2의 `imcflow` conda 환경에서 daemon entry point가 정상 설치되는지 확인
- GPIB3 장비 이름과 `GPIB1::3::INSTR` mapping 확인
- 연결 실패, START timeout, GO timeout, client disconnect 시 DMM reservation 해제
- raw current sample을 measurement server의 명시적 경로에 저장
- result 파일 이름 충돌 방지를 위한 runner-provided prefix 지원

새 기능을 넣더라도 legacy command 순서와 응답 형식은 바꾸지 않는다.

## 6. TVM 선택적 이식

### 6.1 활성화 설정

power 기능은 환경변수로 명시적으로 켜며 기본값은 OFF로 한다.

```text
IMCFLOW_MEASURE_POWER=0                 # 기본값
IMCFLOW_POWER_SCOPE=REGION              # MODEL | REGION | TILE
IMCFLOW_POWER_MODE=now                  # now | wait
DMM_BRIDGE_HOST=<meas-2 IP 또는 hostname>
DMM_BRIDGE_PORT=9900
```

`IMCFLOW_MEASURE_POWER`가 false일 때는 다음을 보장한다.

- generated C에 DMM start/end 호출을 넣지 않는다.
- power 측정용 runtime work를 실행하지 않는다.
- 기존 MMIO barrier, warmup, retry 및 interrupt 순서를 변경하지 않는다.
- measurement server 연결이 없어도 compile/run이 정상 동작한다.

### 6.2 `ext_codegen.py`

기존 `power` 브랜치의 파일 전체를 복사하지 않고 다음 부분만 현재
`chip_acc_measure`의 `KernelCodeGenerator`에 이식한다.

1. power enable, scope, mode 환경변수 parsing과 validation
2. Linux generated code의 `dmm_measure.h` include
3. DMM configuration을 만드는 start code generator
4. 결과를 읽고 `dmm_close()`를 호출하는 end code generator
5. MODEL/REGION/TILE별 begin/end 삽입 위치

초기 동작은 기존 `power`와 호환되도록 한다.

| Scope | Begin | End | 초기 LOOP |
|---|---|---|---|
| MODEL | 첫 IMCFlow region 실행 전 | 마지막 IMCFlow region 종료 후 | 없음 |
| REGION | 각 region의 warmup 이후 | 해당 region의 output read 이후 | 없음 |
| TILE | tile input transfer 이후, invoke 직전 | invoke 완료 직후, output read 전 | 없음 |

초기 기본 scope는 기존 `power`와 같은 `REGION`으로 둔다. 단, scope는 기존처럼
Python enum에 하드코딩하지 않고 환경변수로 선택 가능하게 한다.

다음 코드는 이식하지 않는다.

- tagged measurement 관련 모든 emit 함수
- power region loop macro와 minimum sample loop
- clock synchronization과 timestamp 변환
- power 측정을 위해 추가했던 실험용 MMIO barrier
- `DEBUG_PRINT_INSTRUMENT` 및 MMIO trace 변경
- 기존 `power` 브랜치의 오래된 retry, interrupt, model registry 코드

### 6.3 DMM configuration

초기 hardware gate는 현재 사용 가능한 GPIB primary address 3 장비 하나만
사용한다. 기존 `power`의 VDD/DDA/DDC 3-DMM hardcoding을 그대로 복사하지 않는다.

- measurement server의 inventory/config가 GPIB3을 가리키게 한다.
- generated C에는 logical DMM name, NPLC, interval, sample count, current range,
  reset, server output path만 전달한다.
- 전압과 power 환산은 acquisition protocol과 분리한다. 초기 결과의 기준값은
  raw current와 평균 current이다.
- DMM parameter는 가능한 한 runner 환경 또는 설정 파일에서 결정하고
  `ext_codegen.py` 상수 수정 없이 바꿀 수 있게 한다.

### 6.4 CMake와 linking

현재 host binary template과 dataset binary 양쪽에 다음을 적용한다.

- `3rdparty/measurement_utils/capi/dmm_measure.c`를 한 번만 compile한다.
- generated model과 executable이 동일한 C API instance를 사용하게 한다.
- power OFF build에서도 link는 가능하지만 runtime TCP 연결은 발생하지 않게 한다.
- ARM/PetaLinux cross compile에서 추가 Python 또는 PyVISA dependency가 생기지
  않는지 확인한다. board에는 POSIX socket C code만 필요하다.

### 6.5 실행 script

현재 `run_chiptest.sh`와 `run_dataset_eval.sh` 구조를 유지하면서 필요한 환경변수만
board에 전달한다.

- `IMCFLOW_MEASURE_POWER`
- `IMCFLOW_POWER_SCOPE`
- `IMCFLOW_POWER_MODE`
- `DMM_BRIDGE_HOST`, `DMM_BRIDGE_PORT`
- DMM acquisition parameter와 server output prefix

기존 `power`의 script 전체를 복사하지 않는다. 특히 현재 dataset selection,
checkpoint, BUGFIX mode, warmup 및 timeout 처리를 보존한다.

## 7. 단계별 검증

### Gate 1: measurement_utils unit test

master의 measurement_utils `power_v2`에서 legacy C API와 bridge daemon test를
실행한다.

- START parsing
- wait/now mode
- GO 후 RESULT 순서
- CLOSE/disconnect cleanup
- 복수 client의 DMM exclusive reservation
- timeout과 error response

### Gate 2: meas-2에서 daemon 단독 검증

```bash
ssh meas-2
conda activate imcflow
```

다음을 확인한다.

1. measurement_utils `power_v2` checkout 및 revision 확인
2. GPIB3 configuration 확인
3. legacy RPC/DMM backend 준비
4. `measure_bridge_daemon.py`를 TCP 9900에서 실행
5. mock 또는 query 명령으로 장비 연결 확인

daemon은 tagged server와 동시에 같은 DMM을 점유하지 않게 한다.

### Gate 3: board의 standalone C test

TVM을 연결하기 전에 board에서 작은 C program으로 protocol 전체를 검증한다.

```bash
ssh petalinux
```

검증 순서:

1. `dmm_start_current_now()` 호출
2. 짧은 sleep 또는 간단한 workload 실행
3. `dmm_get_result_now()` 호출
4. `RESULT`의 name, average, count 확인
5. `dmm_close()` 호출
6. meas-2의 raw sample 파일 확인

이 단계가 통과하기 전에는 TVM generated code에 측정 호출을 넣지 않는다.

### Gate 4: TVM power OFF regression

master에서는 항상 기존 Python venv를 먼저 활성화한다.

```bash
activate
```

`power_v2`에서 `IMCFLOW_MEASURE_POWER=0`으로 ResNet을 compile/build/run하고 현재
`chip_acc_measure`와 비교한다.

- generated accelerator instruction이 불필요하게 바뀌지 않는지
- host binary가 measurement server 없이 실행되는지
- accuracy와 timeout/deadlock 여부가 기준과 같은지

### Gate 5: 최소 power ON smoke test

먼저 ResNet sample 하나와 `REGION + now` mode로 실행한다.

- 각 region에서 STARTED/RESULT가 정확히 한 번씩 대응되는지
- region 사이에서 socket 또는 DMM reservation이 누수되지 않는지
- 반환 sample count가 0이 아닌지
- power OFF에는 없던 accelerator deadlock이 생기지 않는지
- meas-2에 raw current 파일이 생성되는지

실패 시 MODEL/TILE을 동시에 디버깅하지 않고 REGION 한 범위만 고친다.

### Gate 6: scope 확장

REGION이 안정된 후 순서대로 확인한다.

1. TILE + now
2. MODEL + now
3. REGION + wait
4. 필요한 경우 MODEL/TILE + wait

각 test에서 begin/end 횟수, result count, DMM reservation release 및 board 실행
완료 여부를 기록한다.

### Gate 7: 결과 수집

측정 중에는 파일 전송을 하지 않는다. 전체 evaluation이 끝난 뒤 master runner가
meas-2의 결과 디렉터리를 한 번만 가져오는 별도 finalize 단계를 추가한다.

- board는 SCP를 실행하지 않는다.
- `POWER_REGION_END` 또는 DMM result 함수 안에서 SCP를 실행하지 않는다.
- 전송 실패는 측정 결과 보존 여부를 알리는 명확한 error로 처리한다.
- plot은 master로 가져온 raw current 파일을 입력으로 별도 생성한다.
- 기본 local artifact 경로는 `eval_dir/<model>/power/<run_id>/`로 한다.

## 8. commit 및 sync 순서

작은 단위로 다음 순서를 사용한다.

### measurement_utils

1. `test(power): validate legacy bridge protocol on power_v2`
2. 필요한 경우 `fix(power): harden legacy GPIB3 measurement flow`
3. `origin/power_v2` push
4. meas-2에서 `power_v2` pull 및 test

### TVM

1. `chore(power): pin measurement_utils power_v2 baseline`
2. `feat(imcflow): port legacy DMM measurement controls`
3. `build(imcflow): link legacy DMM C runtime into host binaries`
4. `feat(codegen): forward legacy power measurement configuration`
5. `test(power): add power-off and protocol regression coverage`
6. `docs(power): document power_v2 measurement workflow`
7. `origin/power_v2` push
8. board에서 TVM `power_v2`와 정확한 submodule commit을 checkout하여 sync 확인

각 hardware test 전에 다음 revision을 함께 기록한다.

- master TVM commit
- board TVM commit
- TVM이 pin한 measurement_utils commit
- meas-2 measurement_utils commit

네 값이 의도한 조합과 다르면 측정을 시작하지 않는다. 단, TVM 전체 작업 트리의
무관한 문서나 local script까지 clean일 것을 요구하지 않고, board에 배포되는 코드와
measurement_utils revision만 검증한다.

## 9. 완료 조건

- TVM `power_v2`가 `chip_acc_measure`에서 분기되어 있다.
- measurement_utils `power_v2`가 `main`에서 분기되어 있다.
- TVM은 measurement_utils `power_v2`의 정확한 commit을 pin한다.
- tagged protocol 또는 tag API 없이 legacy START/GO/RESULT protocol로 측정한다.
- power OFF ResNet 결과가 `chip_acc_measure` 기준과 동일하다.
- board standalone test와 TVM REGION power test가 GPIB3에서 통과한다.
- raw current sample과 평균/count 결과가 meas-2에 남는다.
- runtime 측정 구간에는 SCP와 plotting이 포함되지 않는다.
- 변경 사항과 실행 명령이 별도 quickstart 문서에 정리되어 있다.

## 10. 복구 기준

- TVM 문제 발생 시 `power_v2`만 폐기하면 `chip_acc_measure`는 영향을 받지 않는다.
- measurement_utils 문제 발생 시 `power_v2`만 폐기하면 `main`은 영향을 받지 않는다.
- 기존 `power`, `feat/power_tagged_measurement` 브랜치는 참고용으로 유지하며 reset,
  force-push 또는 history rewrite를 하지 않는다.
- hardware failure가 발생하면 board 재부팅 전에 마지막 START/GO/RESULT 상태와
  accelerator 진행 위치를 보존한다.

## 11. 수행 결과 (2026-08-21)

### 11.1 revision과 배포 상태

| 위치 | branch | revision | 상태 |
|---|---|---|---|
| master TVM | `power_v2` | `04c661caed` | `origin/power_v2` push 완료 |
| master measurement_utils submodule | `power_v2` | `5c79c1e9ed` | TVM gitlink로 고정 |
| meas-2 measurement_utils | `power_v2` | `5c79c1e9ed` | pull 및 실행 확인 |
| board | standalone C client 및 TVM ARM binary | 위 revision에서 build | 실행 확인 |

TVM은 `chip_acc_measure`의 `dc6e7b3e28`에서, measurement_utils는 `main`의
`a6ad97adf9`에서 각각 분기했다. 사용자 소유 untracked 파일인 `_run.sh`는 수정하거나
commit하지 않았다.

### 11.2 구현 결과

- measurement_utils에 `DMM_GPIB3 -> GPIB1::3::INSTR` 설정을 추가했다.
- legacy C client가 숫자 IP뿐 아니라 hostname도 사용할 수 있도록
  `getaddrinfo()` 기반 연결로 보완했다.
- standalone `now` mode C smoke client를 추가했다.
- TVM generated C에는 power가 활성화된 경우에만 legacy
  `START -> STARTED -> GO -> RESULT -> CLOSE` 호출을 삽입한다.
- `MODEL`, `REGION`, `TILE` scope와 `now`, `wait` mode를 선택할 수 있게 했으며,
  기본값은 power OFF, `REGION`, `now`이다.
- dataset/chiptest runner는 bridge 주소를 board에 전달한다.
- raw sample SCP는 region 종료 시점이 아니라 전체 evaluation이 끝난 뒤 한 번만
  수행한다.
- tag, clock sync, metadata 정렬, region loop 구현은 포함하지 않았다.

### 11.3 검증 결과

- measurement_utils bridge daemon test: `47 passed`
- C API syntax 및 ARM cross build: 통과
- board standalone GPIB3 actual measurement:
  `avg=0.0247196827`, `count=21381`
- ResNet `REGION + now`, sample 0:
  - evaluation 완료, failed sample 0, accelerator deadlock 없음
  - 4개 region 모두 START/GO/RESULT/CLOSE가 일대일로 완료
  - sample count: `5932`, `8357`, `15534`, `15546`
  - raw sample은 meas-2와 master 양쪽에 보존
- ResNet power OFF, 같은 sample 0:
  - evaluation 완료, failed sample 0, accelerator deadlock 없음
  - generated C에서 DMM include/call이 생성되지 않음을 확인
  - power ON/OFF 모두 이 단일 sample의 accuracy는 `0/1`로 동일

master에 수집된 power ON raw data는 다음 디렉터리에 있다.

```text
tvm_practice/test_imcflow/codegen/eval_results/power/20260821_175104/
```

meas-2 원본은 `/tmp/power_v2_resnet_*.txt`이며, standalone 원본은
`/tmp/power_v2_standalone.txt`이다. 검증용 daemon은 다른 사용자의 9900/1329
process를 건드리지 않기 위해 bridge `9910`, RPC `1330`을 사용했다.

### 11.4 후속 검증 범위

최소 완료 조건인 standalone actual DMM, power-OFF regression, TVM
`REGION + now` actual DMM gate까지 수행했다. `TILE + now`, `MODEL + now`,
`REGION + wait`의 실제 장시간 hardware matrix는 최소 legacy baseline과 분리해
후속 실험에서 순차 수행한다. 해당 scope/mode의 codegen 경로와 환경변수 parsing은
이번 구현에 포함되어 있다.
