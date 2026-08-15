# Tag 기반 power 측정 재설계 계획

## 목표

기존 TVM `power` 브랜치의 `MODEL`/`REGION`/`TILE` 측정 mode를 가져오지 않고,
현재 `chip_acc_measure` 코드 위에 새로운 power 측정 구조를 구현한다.

핵심 원칙은 다음과 같다.

1. power 측정 session은 실행 전체를 한 번만 감싼다.
2. `REGION`, `TILE` 같은 고정 측정 mode를 없앤다.
3. 실행 중 원하는 지점에서 임의의 tag를 추가한다.
4. 모든 current sample에 그 시점의 active tag가 연결되도록 한다.
5. 기본 acquisition mode는 `now`다.
6. 측정할 rail과 DMM acquisition 설정은 runner가 관리한다.
7. board는 measurement server에 직접 TCP로 요청하고, measurement server는
   로컬 PyVISA로 DMM을 제어한다.
8. 측정 결과는 session 단위 artifact로 만들고 runner가 자동으로 회수한다.
9. 기존 TCP bridge/RPyC 코드는 삭제하거나 재작성하지 않고, 새 경로에서는
   사용하지 않는다.
10. TVM/CMake/codegen을 수정하기 전에 board의 standalone C test로
    `START → TAG → STOP → artifact` 전체 경로를 먼저 통과시킨다.

현재 변경 대상 repository는 다음 두 개다.

| Repository | 담당 기능 |
|---|---|
| `SNU-VLSI/measurement_utils` | direct PyVISA server, protocol v2, C API, standalone smoke test, artifact writer/loader |
| `SNU-VLSI/tvm` | measurement_utils submodule pin, DMM/runner config, CMake, host wrapper, codegen tag, runner, 문서 |

IMCFlow repository의 RTL이나 runner source를 변경하는 것은 현재 범위에 없다.
추후 IMCFlow 변경이 필요해지면 이 두 branch에 섞지 않고 별도 branch와 commit
계획을 추가한다.

## 수행 결과 (2026-08-15)

이 문서의 Phase 0~7 구현과 standalone/one-conv hardware gate를 완료했다.

- `measurement_utils`에는 RPyC를 거치지 않는 direct-PyVISA v2 server, POSIX C
  client, 8-round clock sync, process-wide VISA lock/reservation, disconnect partial
  finalize, canonical JSON/NPZ artifact와 tag-state summary가 구현됐다.
- TVM에는 single/dataset의 normal/debug 네 host wrapper, 정확히 한 개의 shared
  C measurement runtime, Linux-only generated kernel tag, runner request/daemon/SCP
  자동화, summary/filter/plot utility가 구현됐다.
- master/board/meas-2의 기존 checkout만 재사용했다. 기존 원격 변경은 각각
  `wip/meas2_pre_power_tagged_20260815`와
  `wip/petalinux_pre_power_tagged_20260815`에 commit/push하여 보존했다.
- measurement_utils 구현 및 hardware 수정 revision은
  `76da38695d9ca986da6e9b1be99c04f72a43c064`이다.
- `ssh petalinux`, `ssh meas-2`, master `activate`, meas-2 `imcflow` conda,
  Keysight backend의 `GPIB1::3::INSTR` IDN과 tracked GPIB3 mapping을 확인했다.
- board standalone C smoke를 실제 GPIB3에서 반복 실행했다. trigger-aligned run
  `20260815T110050Z_board_c_trigger_aligned_ac8c9ffe_3252672`는 287 samples,
  0.006 s interval, 8 clock-sync samples, 약 0.214 ms uncertainty, ordered
  idle/busy/event/clear와 두 active phase state를 기록했다.
- 2초 강제 종료 session
  `20260815T110302Z_board_c_forced_disconnect_ac8c9ffe_3267909`는 `partial` artifact를
  남겼고, 직후 정상 session이 성공하여 DMM/VISA reservation 해제를 확인했다.
- `scan_gen/scan_reg_files`가 `const_scan_reg_files/0x00`을 가리키고 16개 NPZ의
  nonzero count가 모두 0임을 확인한 뒤 board에서 zero scan programming을 했다.
- 실제 `one_conv_small` Linux/ARM 실행
  `20260815T111838Z_one_conv_small_final_0abc04bd_3377103`이 workload와 측정 모두
  exit 0으로 끝났고 22개 phase/kernel/stage/tile tag가 순서대로 저장됐다.
  이 run의 default config `voltage_V=1.0`과 `measured_power` 값은 placeholder이므로
  current path/tag 검증용이며, 물리 rail power 결과로 인용하면 안 된다.
- power-disabled one-conv도 정상 종료했다. enabled/disabled raw output은 exact
  match하지 않았지만 disabled 반복끼리도 906/1024 element가 달라졌고 평균 절대
  차이가 각각 약 60.93, 60.82로 같아, 이번 한 샘플에서는 power tag 유무보다
  기존 analog repeat variation이 지배적이었다. 정확도 회귀 결론은 dataset 반복
  통계로 내려야 한다.
- 코드 검증은 tagged server 10 tests, legacy bridge/RPC 58 tests, TVM workflow
  9 tests, 실제 Linux-generated one-conv AArch64 link, single/dataset normal/debug
  AArch64 link를 통과했다.

최종 runner는 hardware 접근 전에 다음을 모두 fail-fast로 검사한다.

1. master/board/meas-2 tracked working tree가 clean인지
2. master와 board TVM revision이 같은지
3. master gitlink/board submodule/meas-2 measurement_utils revision이 같은지
4. `build_metadata.json`의 clean codegen revision이 현재 revision과 같은지
5. 배포 binary의 `--power-build-info` clean link revision이 현재 revision과 같은지
6. daemon HELLO protocol/revision이 같은지

ResNet single-input과 KWS/VWW dataset 장시간 acceptance는 코드나 연결 실패 때문이
아니라, 사용자가 실제 DMM probe rail 이름과 수동 설정한 voltage를 확정한 별도
power config로 수행해야 하므로 이 implementation run에서는 실행하지 않았다.
placeholder default로 장시간 power 값을 생성하지 않는 것이 원칙이다.

두 repository의 작업 branch 이름은 동일하게 맞춘다.

```text
TVM:               feat/power_tagged_measurement
measurement_utils: feat/power_tagged_measurement
```

TVM branch는 master에 이미 생성되어 있다. `measurement_utils`는 TVM의 git
submodule이므로 구현 시작 시 nested repository에서도 `main` 기반으로 위 feature
branch를 만들고 `origin`에 push한다. TVM에는 검증되어 push된 measurement_utils
commit만 submodule gitlink로 반영한다.

## Repository branch와 server sync 정책

### Repository 배치

| Server | TVM | measurement_utils | 역할 |
|---|---|---|---|
| master | `/root/project/tvm` | `/root/project/tvm/3rdparty/measurement_utils` | 유일한 기본 개발/commit/push 위치 |
| measurement server (`meas-2`) | 필요 없음 | `/home/jaeyongjang/project.local/measurement_utils` | server 실행 및 DMM hardware 검증 |
| board (`petalinux`) | `/home/root/tvm` | `/home/root/tvm/3rdparty/measurement_utils` submodule | C smoke build/run, TVM binary run |

두 repository의 `origin`은 각각 다음 remote다.

```text
git@github.com:SNU-VLSI/tvm.git
git@github.com:SNU-VLSI/measurement_utils.git
```

### 기존 변경 보존 후 같은 checkout 재사용

현재 원격 checkout에는 이번 작업과 무관한 변경이 있다.

- `meas-2:/home/jaeyongjang/project.local/measurement_utils`는 `main`이며
  `example/configs/ps_B1_config.json`이 수정되어 있다.
- `petalinux:/home/root/tvm`은 `imcflow` branch이고 tracked/untracked 변경이 있으며
  `3rdparty/measurement_utils` submodule도 초기화되지 않았다.

폴더를 추가로 만들지 않는다. 대신 현재 변경을 먼저 별도 WIP branch의 명시적인
commit으로 보존하고 origin에 push한 뒤, **같은 working directory**에서
`feat/power_tagged_measurement`로 switch한다.

```text
meas-2 preservation branch:    wip/meas2_pre_power_tagged_20260815
petalinux preservation branch: wip/petalinux_pre_power_tagged_20260815
```

one-time 전환 절차는 다음과 같다.

1. `meas-2`의 기존 measurement_utils checkout에서 현재 변경 diff를 확인한다.
2. 현재 `main`에서 `wip/meas2_pre_power_tagged_20260815` branch를 만들고
   `ps_B1_config.json` 변경을 설명하는 commit으로 저장한 뒤 push한다.
3. board의 기존 TVM checkout에서 tracked/untracked 변경과 파일 크기를 확인한다.
4. 현재 `imcflow`에서 `wip/petalinux_pre_power_tagged_20260815` branch를 만들고
   dataset runner/source/generated template 변경을 설명하는 보존 commit으로
   저장한 뒤 push한다.
5. 두 working tree의 `git status --porcelain`이 비었는지 확인한다. 누락된 변경이
   있으면 삭제하거나 stash하지 않고 같은 WIP branch에 추가 commit한다.
6. master에서 두 feature branch가 push된 것을 확인한 뒤, `meas-2`는 같은
   measurement_utils 폴더에서, board는 같은 TVM 폴더에서 feature branch로
   switch한다.
7. board TVM feature branch에서 `submodule sync --recursive`와
   `submodule update --init --recursive`를 수행해 기존에 초기화되지 않았던
   measurement_utils 경로를 준비한다.

보존 commit은 기능 branch에 merge하지 않는다. 목적은 기존 변경을 안전하게
남기고 working tree를 비워 branch switch가 가능하게 만드는 것이다. 이후 feature
branch에서 새 dirty 변경이 발견되면 자동 commit/reset/clean하지 않고 sync/test를
중단한다.

### Commit publish와 pull 순서

one-time WIP 보존이 끝난 뒤에는 master를 기본 단일 writer로 사용하며 한
iteration은 항상 다음 순서를 지킨다.

1. master의 nested `measurement_utils` feature branch에서 변경하고 local unit
   test를 수행한다.
2. measurement_utils 변경을 commit하고
   `origin/feat/power_tagged_measurement`에 push한다.
3. 해당 commit을 master TVM의 `3rdparty/measurement_utils` gitlink로 기록한다.
4. TVM 변경 유무와 관계없이 갱신된 submodule pointer를 commit하고
   `origin/feat/power_tagged_measurement`에 push한다.
5. `meas-2`의 기존 measurement_utils 폴더에서 `fetch` 후
   `pull --ff-only origin feat/power_tagged_measurement`를 수행한다.
6. board의 기존 TVM 폴더에서 `fetch`, `pull --ff-only`,
   `submodule sync --recursive`, `submodule update --init --recursive` 순으로
   동기화한다.
7. `meas-2` server와 board TVM submodule이 동일 measurement_utils commit에서
   동작하는지 확인한다.
8. 아래 commit identity gate를 통과한 뒤에만 board smoke 또는 TVM test를
   실행한다.

원격 server에서 즉석으로 source를 수정하지 않는다. 불가피하게 원격에서 진단용
수정을 만들었다면 hardware test를 계속하지 말고 별도 commit으로 push한 뒤
master에서 `pull --ff-only`하여 다시 단일 history로 맞춘다. force-push와 merge
commit 기반 sync는 사용하지 않는다.

### Commit identity gate

각 hardware run 전에 다음 revision을 기록하고 비교한다.

- master TVM `HEAD`
- master TVM gitlink가 가리키는 measurement_utils commit
- master nested measurement_utils `HEAD`
- `meas-2` 기존 checkout을 feature branch로 switch한 measurement_utils `HEAD`
- board TVM submodule의 measurement_utils `HEAD`
- board TVM `HEAD`와 실제 binary build manifest의 TVM revision

standalone 단계에서는 세 장비의 measurement_utils SHA가 모두 같아야 한다. TVM
통합 이후에는 master/board TVM SHA도 같고, master TVM gitlink, board submodule,
`meas-2` measurement_utils SHA도 모두 같아야 한다. 하나라도 다르면 DMM을
시작하지 않는다.

revision 숫자만 같고 generated MLF나 배포 binary가 오래된 경우도 막는다.
codegen은 `build_metadata.json`에 TVM/measurement_utils revision과 tracked-tree
dirty 여부를 기록한다. CMake는 host executable에 link 시점의 두 revision과
dirty 여부를 내장하며 binary는 `--power-build-info`로 이를 출력한다. runner는
master/board/meas-2 tracked tree clean, codegen metadata, deployed binary를 모두
검사한 뒤에만 daemon/DMM 단계로 진행한다.

revision은 runner log와 `session.json`에 최소 다음 필드로 남긴다.

```json
{
  "tvm_git_rev": "<40-hex>",
  "measurement_utils_git_rev": "<40-hex>",
  "board_tvm_git_rev": "<40-hex>",
  "measurement_server_git_rev": "<40-hex>"
}
```

`power_tagged_measurement_server.py --version` 또는 server HELLO 응답으로 server의
measurement_utils revision을 제공하고, board binary에는 build 시점의 두 revision을
manifest/string으로 포함한다. master runner가 run 시작 전에 이 값을 비교한다.

## 기존 `power` 구현에서 가져올 것과 버릴 것

### 가져올 개념

- Linux host에서 `measurement_utils` C API를 호출하는 구조
- board C API → TCP measurement server → local `DmmManager`/PyVISA 흐름
- `DMM_GPIB3` 같은 논리적 measurement target으로 DMM을 선택하는 구조
- DMM group trigger와 `now` 방식의 동시 중단/결과 회수
- DMM reservation, timeout, reconnect 처리

### 가져오지 않을 것

- `PowerMeasurePhase.MODEL/REGION/TILE`
- `ext_codegen.py` 안에 고정된 DMM 이름과 acquisition parameter
- region별로 DMM session을 반복해서 열고 닫는 방식
- compile-time `self.func_name` 기반 output filename
- `run_chiptest.sh`의 hardcoded remote 주소와 구형 scan 실행 코드
- `dmm_measure.c`를 여러 target에 중복 compile하는 CMake 구조
- power 측정을 위해 전역으로 켜 둔 MMIO trace

### 기존 bridge/RPyC 코드 처리 원칙

기존 `measurement_utils`의 다음 코드는 삭제하지 않고 그대로 보존한다.

- `ps_ctrl/cli/measure_bridge_daemon.py`
- `ps_ctrl/rpc.py`
- `RemoteDmmManager`와 기존 line-oriented v1 protocol 사용자

새 tagged power 경로에서는 이 코드들을 import하거나 실행하지 않는다. 새
`ps_ctrl/cli/power_tagged_measurement_server.py` process가 `DmmManager`를 로컬 객체로
만들고 PyVISA를 직접 호출한다. 따라서 tagged measurement의 runtime data
path에는 RPyC server, RPyC client, netref 변환, 별도 bridge process가 없다.

기존 코드는 과거 실험 재현과 참고를 위해 남겨 두며, 새 기능 때문에 동작을
바꾸거나 삭제하지 않는다. 향후 PyVISA가 실행되는 장비와 measurement server를
물리적으로 분리해야 할 때만 RPyC backend 재사용을 별도 과제로 검토한다.

## 전체 architecture

```text
Master server (현재 server, Docker)
  ├─ power request JSON 생성/검증
  ├─ session_id 생성
  ├─ SSH/SCP 대상: petalinux
  ├─ request JSON과 executable을 board에 전송
  └─ board executable 실행 명령
             │ SSH/SCP
             ▼
FPGA board (PetaLinux ARM)
  ├─ dmm_session_start_file(request.json)
  ├─ 실행 전체 동안 측정 session 유지
  ├─ dmm_tag_set()/dmm_tag_clear()/dmm_tag_event()
  └─ dmm_session_stop()
             │ direct TCP (START/TAG/STOP)
             ▼
Measurement server (SSH alias: meas-2)
  ├─ runner request 검증
  ├─ measurement target을 server inventory의 실제 DMM으로 resolve
  ├─ local DmmManager/PyVISA로 group trigger와 sampling 시작
  ├─ tag event와 clock metadata 기록
  ├─ STOP에서 모든 DMM을 중단하고 sample 회수
  ├─ sample timestamp와 active tag를 결합
  └─ session artifact 저장
             │ local USB
             ▼
DMM 장비

Master server
  └─ SSH/SCP 대상 meas-2에서 result artifact 회수
             │
             ▼
Master runner output
  └─ eval_dir/.../power/<session_id>/
```

명령 경로와 측정 경로를 구분한다. master는 SSH로 board의 executable을
실행하지만 DMM 명령을 중계하지 않는다. workload를 실행하는 board가
measurement server에 TCP connection을 직접 열고, measurement server가 같은
process 안에서 로컬 PyVISA를 호출한다.

```text
control:     master ──SSH──▶ board
measurement: board  ──TCP──▶ measurement server ──PyVISA/USB──▶ DMM
result:      master ◀─SCP─── measurement server
```

### SSH 접근 정보와 검증 시 사용 원칙

master server의 SSH 설정에는 다음 alias가 준비되어 있다.

```bash
ssh petalinux  # FPGA board
ssh meas-2     # measurement server
```

구현 및 hardware acceptance 검증에서 이 두 경로를 사용한다.

- `ssh petalinux`: board binary 실행, board log 확인, power config staging 확인,
  measurement TCP endpoint 도달 여부 확인
- `ssh meas-2`: measurement daemon 실행/상태 확인, PyVISA resource와 DMM 연결
  확인, server log와 session artifact 확인
- master server: runner 실행과 양쪽 artifact 회수/비교

Python 환경은 각 server에서 다음과 같이 고정한다.

```text
master server: activate 명령으로 project Python venv 활성화
meas-2:       conda activate imcflow
board:        Python 환경 불필요; PetaLinux C client만 실행
```

master에서 JSON preflight, runner, codegen 관련 Python command를 실행하기 전에
반드시 `activate`를 실행한다. `meas-2`에서 measurement daemon, PyVISA discovery,
DMM hardware test를 실행할 때는 반드시 `imcflow` conda 환경을 활성화한다. system
Python이나 다른 conda environment를 hardware 검증 결과로 사용하지 않는다.

`petalinux`와 `meas-2`는 **master의 SSH alias**다. board가 TCP로 접속할
measurement endpoint는 board에서 실제로 resolve/reach 가능한 IP 또는 hostname을
`POWER_MEASUREMENT_HOST`로 별도 전달한다. `meas-2`라는 SSH alias가 board에서도
그대로 resolve된다고 가정하지 않는다. SSH alias, IP, username, key는 tracked
source/config에 hardcode하지 않고 master의 SSH config와 `.env`에서 관리한다.

## 책임 분리

### Runner가 관리할 것

- power 측정 활성화 여부
- session ID와 실험 metadata
- 측정할 rail 목록
- rail별 voltage metadata
- NPLC, sample interval, sample count, current range, reset 여부
- `now`/`wait` mode 선택 (`now`가 기본)
- 전체 실행을 덮을 duration budget
- board와 measurement server 접속 정보
- config 전송, 실행 후 결과 회수

### Measurement server가 관리할 것

- board의 tagged protocol을 받는 TCP listener
- VISA address와 실제 DMM 장비 연결 정보
- 논리적 measurement target → 실제 DMM mapping
- 동시에 같은 DMM을 쓰지 못하게 하는 reservation
- runner request schema와 값 검증
- 장비가 지원하는 interval/range/count로의 resolve
- group trigger, 중단, sample 수집
- tag와 sample 정렬
- canonical result artifact 생성

measurement server는 새 Python daemon 하나로 구성한다. 이 daemon이 TCP session
처리, request 검증, reservation, local `DmmManager`, PyVISA 호출, artifact 저장을
모두 소유한다. PyVISA 호출은 process-wide lock으로 직렬화하고, 장비 연결 실패
시 같은 process 안에서 resource를 닫고 다시 연다.

물리 장비 inventory까지 runner가 넘기게 만들지는 않는다. runner는
`DMM_GPIB3` 같은 논리적 measurement target과 측정 parameter만 보내고, 실제
`GPIB...` 주소는 server의 `DMM_CONFIG`에 남긴다. 이렇게 해야 board별 runner
설정과 측정 장비 배선 정보가 섞이지 않는다.

### 현재 사용할 DMM configuration

현재 tagged power 측정에서 사용할 수 있는 DMM은 GPIB primary address 3의 장비
한 대다. 기존 `measurement_utils/example/configs/dmm_config.json`과 같은
`DEVICE`/`POWER`/`PRESET` schema로 다음 tracked config를 사용한다.

```text
tvm_practice/test_imcflow/codegen/dmm_configs/dmm_gpib3.json
```

핵심 mapping은 다음과 같다.

```text
DMM_GPIB3 → dmm_gpib3 → GPIB1::3::INSTR
```

- `GPIB1::3::INSTR`의 `1`은 measurement server의 GPIB interface 번호이고,
  마지막 `3`이 사용할 DMM primary address다.
- PyVISA의 default backend가 아니라 `measurement_utils`의 `VISA_BACKEND`, 즉
  `/opt/keysight/iolibs/libktvisa32.so`를 사용한다. `meas-2`의 `imcflow` 환경에서
  이 backend로 `GPIB1::3::INSTR`가 enumerate되는 것을 확인했다.
- tagged measurement server는 시작할 때 `DMM_CONFIG`가 배포된 이 JSON을
  가리키도록 한다.
- 이 JSON은 master에서 관리하고 daemon 배포/시작 전에 `meas-2`로 SCP한다.
- board에는 이 physical inventory JSON을 복사하지 않는다. board가 전달하는
  runner request에는 논리 이름 `DMM_GPIB3`만 들어간다.
- 현재 config에는 DMM이 한 대만 있으므로 한 session에서 하나의 measurement
  target만 허용한다. 실제로 연결한 power rail과 voltage는 runner request의
  metadata와 `voltage_V`에 기록한다.
- 새 daemon을 실행하거나 config를 hardware로 검증할 때는 `ssh meas-2` 후
  `conda activate imcflow` 환경을 사용한다.

## Runner power request schema

runner가 관리할 config는 별도의 versioned JSON으로 만든다. 예시는 다음과 같다.

```json
{
  "schema_version": 1,
  "enabled": true,
  "session_id": "20260815T103015_resnet8_chip3_a1b2c3d4",
  "mode": "now",
  "duration_budget_s": 300,
  "defaults": {
    "nplc": 0.001,
    "sample_interval_s": "auto",
    "sample_count": 50000,
    "current_range_A": 0.1,
    "reset": true,
    "autozero": false
  },
  "rails": [
    {"name": "DMM_GPIB3", "voltage_V": 1.00}
  ],
  "metadata": {
    "model": "resnet8_subset31_pretrained_orig",
    "board": "B2",
    "chip": "chip3",
    "checkpoint": "n32_signed_sample",
    "measured_power": "VDD"
  }
}
```

설계 규칙은 다음과 같다.

- `mode`를 생략하면 `now`다.
- `rails`에는 measurement server inventory의 논리 이름을 넣는다. 현재
  deployment에서는 `DMM_GPIB3` 하나만 유효하다.
- rail entry에 parameter를 추가하면 `defaults`를 override한다.
- `voltage_V`는 power 계산 metadata이며 DMM 전류 측정 설정이나 power supply
  voltage를 변경하지 않는다.
- `metadata.measured_power`에는 DMM probe가 실제로 연결된 rail 이름을 기록한다.
  이 값은 장비 선택에 사용하지 않고 결과 해석과 추적에만 사용한다.
- `session_id`에는 영문, 숫자, `_`, `-`만 허용한다. server가 임의 경로를
  request로 받지 않도록 한다.
- server는 measurement target이 inventory에 없거나 parameter가 장비 범위를 벗어나면
  sampling 전에 요청을 거부한다.
- power config가 주어졌는데 session 시작에 실패하면 기본적으로 workload도
  실행하지 않는다. 측정되지 않은 실행을 정상 실험으로 오인하지 않기 위함이다.

tracked example config를 다음 위치에 추가한다.

```text
tvm_practice/test_imcflow/codegen/power_configs/default.json
tvm_practice/test_imcflow/codegen/power_configs/short_run.json
```

`default.json`은 최대 300초 실행 전체 coverage를 우선하여 50,000 samples에
대해 약 6 ms interval을 선택한다. `short_run.json`은 최대 5초 실행에서 높은
시간 해상도를 얻기 위한 one-conv/smoke profile이다. 두 파일 모두
`voltage_V=1.0`과 `measured_power`가 placeholder이므로 실제 rail/voltage를
확인한 실험 config를 복사해 값만 명시적으로 바꾼다.

board/chip별 실제 voltage나 선택 rail이 다른 config는 별도 파일로 두고,
credential이나 server private path는 `.env`에서 관리한다.

## “항상 전체 시간 측정”의 정확한 범위

측정 session의 lifetime은 region이나 tile이 아니라 host executable이 소유한다.

### Single-input runner

```text
power request load / measurement server preflight
DMM group trigger
TAG phase=process_setup
TVM platform/graph/parameter/input setup
TAG phase=graph_execute
TVMGraphExecutor_Run
TAG phase=output
output read/write
TAG phase=cleanup
TVM cleanup
STOP and result finalize
```

### Dataset runner

```text
DMM group trigger
dataset/graph/parameter setup
for each sample:
    sample=<dataset index>
    input setup
    graph execution
    output/accuracy calculation
cleanup
STOP and result finalize
```

따라서 “전체 시간”은 **host executable이 power session을 시작한 뒤 TVM setup,
모든 graph 실행, output 처리와 cleanup을 끝낼 때까지**다. SSH 접속, artifact
전송, scan programming처럼 executable 실행 이전의 runner 작업은 포함하지 않는다.

generated IMCFlow kernel 안에서 실행되는 per-kernel warmup, MMIO transfer,
retry는 이 session 안에 있으므로 모두 측정된다. 필요한 부분만 분석할 때는
측정 session을 잘라 다시 실행하지 않고 tag로 구분한다.

## Tag model

단순 문자열 begin/end pair보다 일반적인 **active tag map**을 사용한다. tag는
고정 enum이 아니며 임의의 key/value를 가진다.

```c
int dmm_tag_set(const char* key, const char* value);
int dmm_tag_clear(const char* key);
int dmm_tag_event(const char* name);
```

예시는 다음과 같다.

```c
dmm_tag_set("phase", "setup");

dmm_tag_set("sample", "42");
dmm_tag_set("phase", "graph_execute");

dmm_tag_set("kernel", "imcflow_region3");
dmm_tag_set("tile", "1");
/* work */
dmm_tag_clear("tile");
dmm_tag_clear("kernel");

dmm_tag_set("phase", "postprocess");
dmm_tag_clear("sample");
```

각 sample은 해당 timestamp에 활성화된 전체 map을 가진다.

```json
{
  "phase": "graph_execute",
  "sample": "42",
  "kernel": "imcflow_region3",
  "tile": "1"
}
```

이 구조에서는 `region`과 `tile`도 특별한 mode가 아니다. 필요한 곳에서 쓰는
일반 tag key일 뿐이며, 이후 `retry`, `dma`, `cpu_op`, `warmup` 같은 tag도 같은
protocol로 추가할 수 있다.

`dmm_tag_event()`는 active state를 바꾸지 않는 순간 marker다. retry 발생이나
timeout 같은 사건 기록에 사용한다.

## Tag 전송과 측정 교란 최소화

tag마다 DMM에 `DATA:POIN?`를 query하면 tag 호출 자체가 느려지고 sampling을
교란할 수 있다. 따라서 tag 경로에서는 DMM query를 하지 않는다.

1. C util은 tag 호출 시 client monotonic timestamp와 sequence number를 붙인다.
2. tag message는 board에서 measurement server로 연결된 같은 TCP stream의
   ordering을 이용하여 fire-and-forget으로 전송한다.
3. measurement server는 receive timestamp도 함께 저장한다.
4. STOP은 이전 tag message 뒤에 같은 TCP stream으로 오므로 모든 tag가 먼저
   처리됐음이 보장된다.

tag API는 정상 경로에서 server ACK를 기다리지 않는다. socket write 실패만
즉시 반환하여 kernel 실행에 network RTT가 추가되지 않도록 한다.

## Board/server clock 정렬

DMM sample과 board tag는 서로 다른 clock domain에 있다. session 시작 전에
TCP clock sync를 수행한다.

```text
board C util                    measurement server
    SYNC(client_send_ns)  ───▶  record recv/send monotonic ns
                         ◀───  SYNCED(server_recv_ns, server_send_ns)
```

- 여러 번 왕복한 뒤 RTT가 가장 작은 sample로 clock offset을 추정한다.
- 이 sync는 DMM sampling 시작 전에 수행하므로 측정 구간을 오염시키지 않는다.
- 각 tag에는 client timestamp, server clock으로 변환한 timestamp, sequence가
  들어간다.
- server receive timestamp와 추정 uncertainty도 보존한다.

software clock sync와 TCP를 사용하므로 tag boundary는 cycle-accurate하지 않다.
정확도 한계는 DMM sample interval, NPLC integration aperture, clock offset 오차의
합으로 manifest에 기록한다. 수십 microsecond 이하의 정확한 경계가 필요하면
향후 GPIO/trigger line 같은 hardware marker가 필요하다.

## DMM sample timestamp 생성

measurement server는 group trigger 전후의 `monotonic_ns`를 기록하고 midpoint를
session sample origin으로 사용한다. `SAMP:TIM MIN`처럼 문자열 interval을 받은
경우 DMM에서 resolve된 실제 `SAMP:TIM?` 값을 rail별로 저장한다.

rail별 sample timestamp는 다음 convention으로 생성한다.

```text
sample_time[i] = trigger_origin + i × actual_sample_interval
```

필요하면 NPLC integration aperture의 중앙으로 보정할 수 있도록 원본 NPLC와
line frequency metadata를 함께 남긴다.

server는 tag event를 시간순으로 replay하여 각 sample에 다음을 materialize한다.

- `tag_state_id`: 해당 시점의 active tag map ID
- `current_A`
- `time_from_trigger_s`
- 선택한 rail의 `voltage_V`
- `power_W = current_A × voltage_V`

원본 tag event도 별도로 남겨 재분석이 가능하게 한다.

## 장시간 측정과 50,000 sample 제한

현재 DMM code는 한 burst의 sample count를 최대 50,000으로 제한한다. MIN
interval로 50,000개를 설정하면 긴 dataset 실행 전체를 덮지 못할 수 있다.

기본 정책은 `duration_budget_s`를 기반으로 interval을 자동 선택하는 것이다.

```text
required_interval >= duration_budget_s / sample_count
resolved_interval = max(requested_min_interval, required_interval)
```

- runner는 single run과 dataset run의 timeout을 duration budget으로 전달한다.
- server는 실제 DMM이 적용한 interval과 최대 coverage 시간을 START 응답으로
  반환한다.
- coverage가 duration budget보다 작으면 workload를 시작하지 않고 실패한다.
- 실제 실행이 budget을 초과해 buffer가 먼저 끝나면 result를 `truncated`로
  표시하고 runner의 power 단계는 실패 처리한다.

이 방식은 전체 시간을 보장하는 대신 긴 실행에서 시간 해상도가 낮아진다.
50,000개를 넘는 고해상도 장시간 측정이 필요하면 후속 단계로 server-side
continuous/chunked acquisition을 추가한다. chunk 전환 사이의 gap을 검증하지
않은 상태에서 자동 chunking을 먼저 도입하지 않는다.

## 새 C API

기존 함수는 compatibility를 위해 유지하되, TVM에서는 다음 session API만
사용한다.

```c
int dmm_session_start_file(const char* request_json_path);
int dmm_tag_set(const char* key, const char* value);
int dmm_tag_clear(const char* key);
int dmm_tag_event(const char* name);
int dmm_session_stop(void);
int dmm_session_abort(const char* reason);
int dmm_session_is_active(void);
const char* dmm_session_id(void);
const char* dmm_last_error(void);
```

설계 세부사항:

- C util은 JSON 내용을 해석하지 않고 크기 제한만 확인한 뒤 server로 전달한다.
- schema validation과 default 적용은 server 한 곳에서 수행한다.
- config와 tag 문자열은 length-prefixed frame으로 보내 공백, 쉼표, UTF-8 label을
  안전하게 지원한다.
- session이 비활성일 때 tag 함수는 성공하는 no-op이다.
- active session은 process당 하나만 허용한다.
- STOP은 `now`에서 모든 선택 DMM에 먼저 `ABORt`를 보내고 그 다음 결과를
  순서와 무관하게 회수한다.
- session 중 socket이 끊기면 server는 DMM을 중단하고 가능한 result를 `partial`
  상태로 저장한 뒤 reservation을 해제한다. 명시적 `ABORT`는 `aborted`다.

## Protocol v2

기존 command-line-like `START --names ...` v1 protocol과 그 bridge/RPyC 코드는
그대로 남기되 새 tagged measurement server에서는 사용하지 않는다. 새 C API와
새 server는 versioned v2 frame protocol만 사용한다.

```text
HELLO 2
SYNC ...
CLOCK_SYNC <offset_ns> <best_rtt_ns> <uncertainty_ns> <sample_count>
START_JSON <payload_length>\n<payload>
TAG_SET <seq> <client_ns> <key_len> <value_len>\n<key><value>
TAG_CLEAR <seq> <client_ns> <key_len>\n<key>
TAG_EVENT <seq> <client_ns> <name_len>\n<name>
STOP
ABORT <reason_length>\n<reason>
```

대표 응답:

```text
HELLO_OK 2
CLOCK_SYNCED
STARTED <session_id> <resolved_config_length>\n<resolved_config_json>
STOPPED <session_id> <summary_length>\n<summary_json>
ERROR <code> <message>
```

기존 v1 client/server test는 기존 코드에 대한 regression test로 유지한다. 새
v2 server가 v1을 동시에 지원하게 만들지는 않는다. 두 경로를 분리하여 새
server에 불필요한 compatibility layer와 RPyC dependency가 들어오지 않게 한다.

## Result artifact

measurement server의 canonical directory는 server가 관리하는 root 아래로
고정한다.

```text
<POWER_RESULT_ROOT>/<session_id>/
├── request.json
├── resolved_config.json
├── session.json
├── tags.jsonl
├── summary.json
└── rails/
    ├── VDD.npz
    ├── DDA.npz
    └── DDC.npz
```

rail NPZ에는 최소한 다음 array가 들어간다.

```text
current_A[]
time_from_trigger_s[]
power_W[]
tag_state_id[]
```

`session.json`에는 다음 정보를 저장한다.

- schema version과 session status: `complete`, `partial`, `aborted`, `truncated`
- 시작/종료 시간과 trigger timing
- clock sync offset/RTT/uncertainty
- requested/resolved DMM configuration
- rail별 실제 sample interval/count/coverage
- configured buffer coverage와 실제 회수된 `collected_coverage_s`; 완료 session은
  두 coverage 모두에 대해 truncation 검사를 통과해야 함
- TVM model, board, chip, checkpoint, git revision 등 runner metadata
- tag drop/send error 여부

`tags.jsonl`은 원본 tag event와 active state snapshot ID mapping을 보존한다.
`summary.json`은 rail별/active-tag별 평균 current, 평균 power, energy를 제공하되
NPZ 원본으로 언제든 다시 계산할 수 있게 한다.

## Runner interface

두 chip runner에 같은 option을 추가한다.

```bash
./run_chiptest.sh --power-config power_configs/default.json \
  resnet8_subset31_pretrained_orig_evl.linux random

./run_dataset_eval.sh --power-config power_configs/vdd_only.json 100
```

환경변수만으로도 override할 수 있게 하되 CLI를 우선한다.

```text
IMCFLOW_POWER_CONFIG
POWER_BOARD_SSH_HOST
POWER_MEASUREMENT_HOST
POWER_MEASUREMENT_PORT
POWER_RESULT_SSH_HOST
POWER_RESULT_BASE_PATH
```

기본 deployment에서는 `POWER_BOARD_SSH_HOST=petalinux`,
`POWER_RESULT_SSH_HOST=meas-2`를 사용한다. `POWER_MEASUREMENT_HOST`는 board에서
measurement server로 접속할 때 사용하는 주소이므로 별도 설정한다. SSH
credential은 tracked JSON에 넣지 않고 master의 SSH config 또는 `.env`로
관리한다.

공용 `power_steps.sh`를 추가하여 `run_chiptest.sh`와
`run_dataset_eval.sh`가 다음 기능을 공유하게 한다.

1. config schema preflight
2. unique session ID 생성
3. runtime metadata 병합
4. normalized request JSON 생성
5. request를 board tmpfs에 전송
6. remote command에 config/measurement server 환경변수 전달
7. 실행 성공/실패와 관계없이 session finalize 확인
8. `meas-2`에서 result directory SCP
9. local artifact completeness 검증

local 저장 위치는 다음과 같이 통일한다.

```text
eval_dir/<model>_evl.linux.../power/<session_id>/
```

dataset 결과도 model eval directory 아래 같은 구조를 사용하고, dataset 이름과
sample selection은 session metadata에 기록한다.

현재 deployment에서는 measurement server가 master와 다른 `meas-2`이므로 SSH/SCP
경로를 사용한다. 향후 measurement server와 runner가 같은 머신인 구성을
지원한다면 local copy를 선택할 수 있지만 최종 directory layout은 동일하게
유지한다.

## Host executable 변경

다음 네 executable source에 공통 power session wrapper를 적용한다.

- `host_binary_make.template/src/execute_graph.c`
- `host_binary_make.template/src/debug_execute_graph.c`
- `host_binary_make.dataset/src/execute_graph_for_dataset.c`
- `host_binary_make.dataset/src/debug_execute_graph_for_dataset.c`

중복을 줄이기 위해 `power_measure_runtime.h/.c` helper를 두 template에서 공유한다.

주요 tag는 다음과 같다.

| 위치 | tag update |
|---|---|
| session 시작 직후 | `phase=process_setup` |
| graph/params 준비 완료 | `phase=input_setup` |
| graph 실행 직전 | `phase=graph_execute` |
| output 처리 | `phase=output` |
| cleanup | `phase=cleanup` |
| dataset iteration 시작 | `sample=<absolute dataset index>` |
| failed sample | `event=sample_timeout` |

모든 early return을 `goto cleanup` 계열 단일 종료 경로로 정리하여 가능한 경우
항상 `dmm_session_stop()`을 호출한다. `SIGTERM`/timeout과 process crash는 server의
disconnect finalize가 partial artifact를 남기도록 한다.

## `ext_codegen.py` 변경

`ext_codegen.py`는 session을 시작하거나 DMM config를 알지 않는다. Linux
generated kernel에 tag call만 삽입한다.

```text
kernel=<full generated function name>
kernel_stage=device_setup
kernel_stage=reset
kernel_stage=warmup
kernel_stage=compiled_transfer
kernel_stage=const_transfer
kernel_stage=policy_update
kernel_stage=invoke
tile=<index>
kernel_stage=output_transfer
```

기존 retry, interrupt fallback, stage heartbeat, MMIO barrier 흐름을 유지하면서
각 기존 statement 사이에 tag call만 추가한다. retry가 발생하면
`dmm_tag_event("retry")`와 attempt metadata를 남긴다.

tagging은 codegen helper 한 곳을 통한다.

```python
def emit_power_tag_set(self, key, value): ...
def emit_power_tag_clear(self, key): ...
def emit_power_tag_event(self, name): ...
```

문자열 escaping을 helper에서 수행하며 baremetal에서는 빈 문자열을 반환한다.
`PowerMeasurePhase`와 region 수 추론은 만들지 않는다.

## CMake 변경

현재 branch의 target naming과 retry compile definition을 보존한다.

- `measurement_utils/capi/dmm_measure.c`를 별도 static library로 정확히 한 번만
  compile한다.
- host executable과 generated `tvm_model`은 같은 library symbol을 참조하여
  process-global session state가 하나만 존재하게 한다.
- Linux ARM host build에서 power support를 포함한다.
- runtime config가 없으면 session/tag API는 no-op이므로 DMM 설정을 바꿀 때마다
  model을 rebuild할 필요가 없다.
- baremetal과 x86 simulation은 no-op stub을 사용하고 network dependency를
  강제하지 않는다.

power rail, NPLC, interval, mode는 runtime JSON이므로 build manifest의 rebuild
판단 요소에 넣지 않는다. tag를 삽입하는 codegen source revision은 기존 source
fingerprint를 통해 자연스럽게 rebuild 대상이 된다.

## 실패 처리

| 상황 | 처리 |
|---|---|
| invalid config/unknown measurement target | DMM 시작 전 실패, workload 실행 안 함 |
| measurement server 연결 실패 | power-enabled run 실패 |
| 일부 DMM start 실패 | 시작한 DMM 모두 중단, reservation 해제, workload 실행 안 함 |
| tag send 실패 | workload는 계속, session을 degraded로 표시 |
| workload non-zero exit | partial sample finalize 후 원래 exit status 보존 |
| board SSH timeout/process kill | measurement server가 disconnect 감지 후 partial finalize |
| runner가 결과를 못 받음 | measurement 결과는 server에 보존, session ID로 재-fetch 가능 |
| sample buffer 조기 종료 | `truncated`, power 단계 실패 |

runner는 executable status와 power status를 별도로 보고한다. accuracy 실행은
성공했지만 power artifact가 불완전한 경우 전체를 단순 성공으로 표시하지 않는다.

## 구현 단계

### Phase 0: 최소 환경 preflight

1. master에서 `ssh petalinux`와 `ssh meas-2` 접속을 각각 확인한다.
2. master에서 `activate` 후 runner/preflight Python 환경을 확인한다.
3. `meas-2`에서 `conda activate imcflow` 후 PyVISA와 measurement utility import를
   확인한다.
4. board에서 measurement server의 IP/hostname이 resolve되고 route가 있는지
   확인한다. TCP port 확인은 Phase 1의 실제 server를 띄운 뒤 수행한다.
5. `meas-2`의 `imcflow` 환경에서 Keysight VISA backend로
   `GPIB1::3::INSTR`를 enumerate하고 `*IDN?`까지만 확인한다.
6. `dmm_gpib3.json`을 `DmmManager`로 load하고 `DMM_GPIB3`가 올바른 VISA
   address로 resolve되는지 확인한다.
7. board에 C compiler가 있는지 확인하고, 없으면 사용할 ARM cross compiler를
   master에서 확인한다.

Phase 0에서는 DMM acquisition parameter를 최적화하거나 TVM binary를 실행하지
않는다. 환경과 장비 식별이 맞는지만 확인하고 바로 standalone smoke로 넘어간다.

### Phase 1: standalone board-to-DMM vertical slice

이 단계는 **TVM 통합 진입 gate**다. TVM source, CMake, generated kernel은 아직
수정하지 않는다. 가장 작은 실제 경로만 먼저 만든다.

1. one-time WIP 보존 commit/push를 끝내고 `meas-2`와 board의 기존 checkout을
   각각 `feat/power_tagged_measurement`로 switch한다.
2. 기존 bridge/RPyC 파일과 분리된
   `ps_ctrl/cli/power_tagged_measurement_server.py`의 최소 버전을 만든다.
3. 최종 v2 framing 중 `HELLO`, `START_JSON`, `TAG_SET`, `TAG_CLEAR`,
   `TAG_EVENT`, `STOP`만 먼저 구현한다.
4. server는 local `DmmManager`와 Keysight VISA backend를 사용해
   `DMM_GPIB3` 측정을 `now` mode로 시작/종료한다.
   최소 버전이라도 client disconnect/timeout에서 DMM을 중단하고 VISA resource를
   `finally`에서 닫는 cleanup은 반드시 포함한다.
5. C API의 최소 `dmm_session_start_file()`, tag 3종, `dmm_session_stop()`을
   구현한다.
6. `measurement_utils/tests/capi/power_tagged_smoke.c`를 추가한다. 이 binary는
   TVM runtime이나 generated model에 의존하지 않는 POSIX C program이다.
7. 변경을 measurement_utils에 commit/push하고 TVM gitlink도 commit/push한 뒤,
   `meas-2`는 같은 checkout에서 pull하고 board는 TVM pull/submodule update를
   수행하여 세 measurement_utils SHA가 같은지 확인한다.
8. board에 C compiler가 있으면 `ssh petalinux`에서 build하고, 없으면 master의
   기존 ARM cross-compile 환경으로 build한 뒤 board에 SCP한다.
9. `meas-2`에서는 `conda activate imcflow` 후 최소 server를 실행하고
   `DMM_CONFIG`는 배포한 `dmm_gpib3.json`을 가리키게 한다.
10. master에서 `ssh petalinux`로 smoke binary를 실행하고, board가 measurement
   server에 직접 TCP connection을 여는지 확인한다.
11. `meas-2`의 raw sample/tag artifact를 master로 수동 SCP해 내용을 확인한다.

standalone C test의 고정 sequence는 다음과 같이 한다.

```text
START session=board_c_smoke target=DMM_GPIB3 mode=now
TAG_SET phase=idle
sleep for a bounded interval
TAG_SET phase=busy
run a bounded CPU busy loop
TAG_EVENT smoke_checkpoint
TAG_CLEAR phase
STOP
```

`sleep`과 busy loop는 전력 차이를 acceptance 조건으로 삼기 위한 것이 아니라,
서로 구분되는 시간 구간에 tag를 만들기 위한 것이다. 실제 current가 유의미하게
달라지지 않더라도 sample과 tag time/state가 올바르게 결합되면 된다.

다음 조건을 모두 만족해야 Phase 2 이후와 TVM 통합으로 진행한다.

- board C process가 exit code 0으로 종료되고 `STARTED`/`STOPPED`를 받는다.
- `meas-2` log에서 client가 board 주소이며 session ID가 일치한다.
- GPIB3 DMM에서 current sample이 1개 이상 회수된다.
- `idle`, `busy`, `smoke_checkpoint` tag가 전송 순서대로 저장된다.
- sample에 최소 `idle`/`busy` 두 tag state가 materialize된다.
- STOP 후 VISA resource와 DMM reservation이 해제된다.
- 결과를 master로 가져와 JSON/NPZ를 열 수 있다.

하나라도 실패하면 TVM 쪽 변경을 시작하지 않고 board TCP, protocol framing,
PyVISA 또는 DMM 문제를 이 작은 프로그램에서 먼저 해결한다.

### Phase 2: `measurement_utils` protocol/session hardening

standalone vertical slice에서 검증한 경로를 유지하면서 production 기능을 채운다.

1. request schema parser와 parameter validator 완성
2. protocol frame partial read/write와 size limit 처리
3. clock sync와 tag event timestamp 변환 추가
4. process-wide VISA lock, reservation, timeout, reconnect 추가
5. selected measurement target session manager 일반화
6. STOP/ABORT/disconnect partial finalize 구현
7. duration budget과 50,000 sample coverage preflight 추가
8. canonical session artifact writer 완성
9. sample timestamp/tag-state materialization과 summary 구현
10. GPIB3 DMM에서 `SAMP:TIM?`, `now` ABORt 후 record 보존, 50,000 sample
    coverage와 최대 interval을 검증
11. tag clock sync 오차와 TCP send overhead를 측정하고 manifest timing
    uncertainty 계산법 확정

### Phase 3: C API hardening

1. standalone test에서 사용한 `dmm_session_*` API를 production 수준으로 확장
2. config file size/path/error response 처리
3. non-blocking ordered tag send 추가
4. clock sync와 timestamp 변환 추가
5. inactive-session no-op과 backward-compatible API 유지
6. fake measurement server 기반 C test 추가
7. standalone smoke test가 같은 C API로 계속 통과하는지 regression 확인

### Phase 4: TVM build와 host executable 통합

1. 검증되어 origin에 push된 measurement_utils commit으로 TVM submodule gitlink를
   고정
2. CMake에 single measurement runtime library 추가
3. normal/debug single runner에 전체 session wrapper 추가
4. normal/debug dataset runner에 전체 session wrapper 추가
5. dataset sample tag와 공통 phase tag 추가
6. early return/cleanup 경로 정리
7. TVM commit을 push하고 board의 기존 `/home/root/tvm` checkout에서
   `pull --ff-only`와 recursive submodule update 수행
8. master/board TVM SHA와 master gitlink/board submodule/`meas-2`
   measurement_utils SHA identity 확인

### Phase 5: generated kernel tag 추가

1. generic tag codegen helper 추가
2. kernel/stage/tile tag 삽입
3. retry event tag 추가
4. 기존 generated code ordering이 tag 외에는 바뀌지 않았는지 golden diff 확인

### Phase 6: runner config/전송/회수 자동화

1. example power config와 `dmm_configs/dmm_gpib3.json` 추가
2. GPIB3 config를 `meas-2`에 배포하고 daemon의 `DMM_CONFIG`로 연결
3. `power_steps.sh` 추가
4. `run_chiptest.sh --power-config` 연결
5. `run_dataset_eval.sh --power-config` 연결
6. `meas-2` result SCP와 local manifest 병합
7. 실패 후 session ID 기반 재-fetch command 추가
8. run 전 commit identity preflight와 mismatch fail-fast 추가
9. 네 revision을 runner log와 `session.json`에 기록

### Phase 7: 분석 utility

1. result loader 추가
2. tag key/value filter와 구간별 current/power/energy summary 추가
3. tag timeline plot 추가
4. 기존 txt notebook 대신 versioned artifact를 읽는 notebook/script 추가

## Test 계획

### Standalone board-to-DMM gate

이 검증은 Phase 0 preflight 다음의 첫 functional test다. 최소 parser/C API unit
check만 수행한 뒤 바로 실행하며, production hardening과 TVM 구현 **이전**에
통과해야 한다.

1. `ssh meas-2` 접속 후 `imcflow` conda 환경에서 server를 시작한다.
2. master, `meas-2`, board의 measurement_utils branch와 SHA가 같은지 확인한다.
3. server가 `dmm_gpib3.json`을 load하고 `DMM_GPIB3` reservation을 얻는지
   확인한다.
4. `ssh petalinux`에서 standalone C smoke binary를 실행한다.
5. packet capture 없이도 양쪽 log의 session ID, sequence, client/server
   timestamp로 동일 session임을 대조한다.
6. 정상 START/TAG/STOP 한 번을 수행한다.
7. 두 번째 실행으로 이전 session의 VISA resource와 reservation이 남지 않았는지
   확인한다.
8. artifact에서 sample count, ordered tag event, tag state ID를 확인한다.
9. 결과를 master로 SCP하고 `activate` 환경에서 loader로 읽는다.

이 gate에서는 TVM model, graph executor, IMCFlow kernel을 사용하지 않는다.
문제가 생기면 세 장비 간 최소 경로만 디버깅할 수 있어야 한다.

### `measurement_utils` unit test

- tagged server가 local fake DMM backend를 직접 호출하고 RPyC를 import하지 않음
- `dmm_gpib3.json`에서 `DMM_GPIB3`가 `GPIB1::3::INSTR`로 resolve됨
- mode 생략 시 `now` 선택
- 1개/복수 rail 선택과 unknown rail reject
- defaults와 rail override merge
- session ID/path traversal reject
- protocol frame partial read/write
- tag sequence/order/state replay
- disconnect 시 partial finalize와 reservation release
- fake DMM record를 tag state별로 정확히 분류
- duration budget coverage 부족 시 preflight reject
- truncated session 판정
- 기존 bridge/RPyC v1 START/GO test regression

### C API test

- fake measurement TCP server 대상 START/TAG/STOP
- standalone smoke source가 TVM header/library 없이 ARM용으로 build됨
- config size/invalid path/error response
- tag fire-and-forget ordering
- broken pipe와 timeout
- inactive session tag no-op
- clock sync offset 선택
- old `dmm_start_current_now()` compatibility

### TVM/codegen test

- standalone board-to-DMM gate가 통과하지 않으면 이 test 단계에 진입하지 않음
- Linux generated C에 expected tag call 존재
- baremetal generated C에 network call 없음
- retry/warmup/MMIO barrier ordering 보존
- tile factor가 달라도 tag set/clear balance 유지
- power support 포함 ARM build link 성공
- normal/debug, single/dataset binary 모두 duplicate symbol 없이 build
- power config 없이 기존 실행 결과와 behavior 동일

### Runner test

- master Python command가 `activate`된 venv에서 실행됨
- 초기 remote 변경은 WIP preservation branch에 commit/push한 후 feature branch로
  switch됨
- feature branch가 dirty하면 pull하지 않고 sync 단계가 중단됨
- feature branch pull이 fast-forward가 아니면 실패
- master/board TVM과 세 measurement_utils revision mismatch 시 DMM 시작 전 실패
- session metadata에 실제 네 revision이 저장됨
- `--power-config` parsing과 `.env` override
- config staging 경로와 remote environment 생성
- board execution 실패 후에도 result fetch 시도
- `petalinux` 실행과 `meas-2` result fetch 경로
- board-routable measurement host와 master SSH alias가 서로 달라도 정상 동작
- SCP 실패 시 session ID를 포함한 재시도 안내
- 결과 manifest에 model/board/chip/checkpoint/build metadata 포함

### Hardware acceptance test

standalone gate가 끝난 뒤 TVM 통합 결과만 여기서 검증한다.

1. `DMM_GPIB3`를 선택한 one-conv 측정
2. tag가 없는 구간도 implicit session tag로 분류되는지 확인
3. region/tile tag가 trace에서 순서대로 보이는지 확인
4. ResNet single input 전체 실행 coverage 확인
5. KWS/VWW dataset 실행에서 sample index tag 확인
6. timeout/강제 종료 후 partial artifact와 DMM reservation 해제 확인
7. `meas-2` artifact가 master output으로 자동 회수되는지 확인
8. power disabled baseline과 accuracy/runtime regression 비교

## 완료 기준

- `MODEL/REGION/TILE` 측정 mode가 코드에 없다.
- TVM 통합 전에 standalone board C test의 START/TAG/STOP와 실제 GPIB3 sample
  artifact가 검증된다.
- TVM과 measurement_utils 모두 `feat/power_tagged_measurement` branch로 push되고,
  remote server는 `pull --ff-only`로 동기화된다.
- hardware run 전 master/board TVM 및 master/board/`meas-2` measurement_utils
  commit identity가 검증된다.
- power-enabled executable당 DMM session이 정확히 한 번 시작되고 한 번 종료된다.
- mode 미지정 시 `now`가 사용된다.
- runner JSON만 바꿔 측정 rail과 acquisition parameter를 변경할 수 있다.
- `DMM_GPIB3`가 tracked config에서 `GPIB1::3::INSTR`로 유일하게 resolve된다.
- tag key/value를 C 코드 어디서든 추가할 수 있다.
- 모든 rail sample에 `tag_state_id`가 존재한다.
- result artifact가 session ID 아래 생성되고 runner로 자동 회수된다.
- 새 tagged 실행 경로에서 board가 measurement server에 직접 TCP로 접속하고,
  measurement server가 RPyC 없이 로컬 PyVISA를 호출한다.
- 기존 bridge/RPyC 코드와 v1 regression test는 그대로 유지된다.
- 실패/timeout에서도 partial artifact와 reservation cleanup이 보장된다.
- power config가 없을 때 기존 chip execution과 accuracy에 변화가 없다.
- current retry, warmup, MMIO barrier, chip lock, remote `.env` 구조가 유지된다.

## 권장 commit 순서

### `measurement_utils` repository

1. `feat(power): add versioned measurement request schema`
2. `feat(power): add direct PyVISA tagged measurement server`
3. `feat(capi): add tagged session client and standalone board smoke test`
4. `test(power): validate standalone board-to-DMM vertical slice`
5. `feat(power): harden tagged protocol and clock metadata`
6. `feat(power): materialize tagged rail artifacts`
7. `test(power): cover direct server, tagging, abort, and legacy regression`

### TVM repository

1. `build(imcflow): link a single measurement runtime for Linux hosts`
2. `feat(imcflow): wrap host executables in whole-run power sessions`
3. `feat(imcflow): emit generic kernel power tags`
4. `feat(codegen): manage power requests and fetch session artifacts`
5. `test(imcflow): verify tagged power build and runner workflows`
6. `docs(imcflow): document tagged whole-run power measurement`

이 순서에서는 `measurement_utils`의 API와 artifact contract를 먼저 고정하고,
그 commit을 TVM submodule pointer로 반영한 뒤 TVM 통합을 진행한다. 기존
`power` branch는 behavior reference로만 사용하고 commit merge/cherry-pick은
하지 않는다. 기존 bridge/RPyC 파일도 보존하되 새 tagged measurement 실행
경로에는 연결하지 않는다.
