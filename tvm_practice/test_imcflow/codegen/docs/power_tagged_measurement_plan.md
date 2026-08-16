# Tag 기반 power 측정 및 DMM metadata 시간 정렬 계획

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
    `START → TAG → STOP_BEGIN → FINALIZE → artifact` 전체 경로를 먼저 통과시킨다.
11. sample의 주 시간축은 measurement server의 GET 호출 midpoint가 아니라 DMM이
    저장한 reading metadata의 첫 sample 시각과 실제 sample interval로 만든다.
    Board tag와 DMM sample은 measurement server 시간축으로 변환하고 변환
    불확실성도 결과에 함께 저장한다.

### Region scope 확장 (2026-08-16)

후속 실험에서는 위 1~2번의 whole-run-only 원칙을 기본 호환 mode로 유지하면서,
기존 `power` branch의 region acquisition을 일반화한 두 번째 scope를 추가한다.

| 축 | 값 | 의미 |
|---|---|---|
| acquisition `scope` | `continuous` | host process 전체에서 DMM session 한 번 |
| acquisition `scope` | `region` | generated IMCFLOW kernel마다 독립 DMM session |
| stop `mode` | `now` / `wait` | scope와 독립적인 DMM 종료 정책, 기본 `now` |
| annotation | async tag set/clear/event | 활성 session 안에서 DMM 재시작 없이 상태 기록 |

`scope=region`의 실행 순서는 다음과 같다.

```text
kernel device setup/reset/warmup
  → power_region_begin(kernel): configure/INIT/GET/STARTED (blocking)
  → compiled/const transfer, policy update, invoke, tile 실행과 async tag
  → power_region_end(): STOP_BEGIN/ABOR/finalize (blocking)
  → device cleanup
```

Region은 중첩하지 않는다. 각 region은 parent run 아래
`<parent_session_id>/regions/rNNNN_<kernel>/`에 독립 raw CSV/NPZ/summary를 만든다.
기존 `continuous` config와 artifact layout은 그대로 유지한다. Protocol v4의
`START_REGION`만 server가 parent request의 session ID를 안전한 child ID로 바꾸며,
board C client는 JSON을 해석하지 않는다.

기존 `power` branch의 `MODEL`/`REGION`/`TILE` 전역 enum과 달리, 새 구조에서
model/region/tile은 begin/end를 어디에 삽입할지 결정하는 instrumentation policy다.
region 내부의 transfer/invoke/tile 구분은 모두 비동기 tag로 처리하므로 tile마다
DMM을 다시 시작할 필요가 없다.

현재 변경 대상 repository는 다음 두 개다.

| Repository | 담당 기능 |
|---|---|
| `SNU-VLSI/measurement_utils` | direct PyVISA server, protocol v4, C API, standalone smoke test, artifact writer/loader |
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
- clean-tree gate를 포함한 TVM revision
  `2ebe92f7baa557b12b29102c7cdb58df8fc57e88`에서 codegen과 ARM binary를 다시
  만들고 `short_run.json`으로 최종 hardware run
  `20260815T113000Z_one_conv_short_2ebe92f_manual`을 수행했다. workload와 측정은
  모두 성공했고, 0.0001 s 실제 interval에서 3,079 samples와 22 tag events를
  수집했다. 13개 sampled tag state에 `process_setup`, `input_setup`,
  `graph_execute`, kernel stage, tile 0, `output`, `cleanup`이 모두 나타났다.
  이 run 역시 1.0 V와 rail 이름이 placeholder이므로 전류 및 tag 시간 정렬
  검증 결과로만 사용한다.
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

## 추가 구현 및 수행 결과: DMM reading metadata 기반 3-clock 정렬 (2026-08-16)

### 완료된 구현과 revision

추가 계획의 Phase M0~M5를 실제 GPIB3에서 완료했다. 현재 implementation 기준
revision은 다음과 같다.

| Repository | Revision | 내용 |
|---|---|---|
| `measurement_utils` | `3a81f83dc2d15e88d3ea496aff548aac577a4e09` | protocol v3, DMM metadata/clock primitive, raw CSV, schema-v2 정렬, freeze coverage 수정 |
| `tvm` | `aced9789a0dea9ced2f2a83163b82e52223d8adc` | 위 submodule pin, protocol-v3 daemon probe, schema-v2 loader/validator/plot |

구현 결과는 다음과 같다.

- Board C client와 measurement server가 `HELLO 3`, start/end 8-round clock sync,
  Board realtime/monotonic anchor, `STOP_BEGIN → STOP_READY → end sync → FINALIZE`를
  사용한다. public `dmm_session_stop()` API는 바꾸지 않았다.
- DMM sample의 canonical origin은 GET midpoint가 아니라 raw CSV의 `Start time`과
  `Sample interval`이다. GET bracket은 `trigger_get_diagnostic`으로만 보존한다.
- `time_alignment.json`, byte-preserved `raw/DMM_GPIB3.csv`, SHA-256 manifest와
  schema-v2 NPZ를 생성한다. NPZ에는 reading number/current, DMM-first/trigger 기준
  시간, server wall/monotonic 시간, power, tag state와 boundary ambiguity가 있다.
- 실제 hardware path에서 metadata가 없거나 raw/buffer 값이 다르면 GET 시간으로
  fallback하지 않고 run을 실패시킨다. schema-v1은 fake/legacy artifact reader에만
  남겼다.
- TVM utility는 schema-v2 completeness와 raw path/checksum을 검증하고, plot의 기본
  축을 DMM first-reading time으로 사용한다. summary는 ambiguous sample 제외 옵션을
  제공한다.

### M0 GPIB3 characterization 결과

대상 장비는 `Keysight Technologies,34465A,MY60036079`이며 firmware 응답은
`A.03.03-03.15-03.03-00.52-05-02`였다.

- `SYST:DATE?;TIME?` 한 번의 GPIB query bracket은 대략 92~94 ms였다. query 응답은
  millisecond를 포함하지만 이 bracket 때문에 한 번의 절대 clock 관측 uncertainty는
  약 42~49 ms다.
- 10 ms, 1 ms, 100 us에서 각각 20 readings를 저장했다. CSV `Sample interval`은
  각각 `0.010000`, `0.001000`, `0.000100`이었고 CSV count, DMM buffer count와 값이
  모두 정확히 일치했다.
- CSV first-reading timestamp는 millisecond 세 자리이므로 metadata 자체의
  resolution uncertainty 하한을 1 ms로 둔다.
- `SYST:TIME`은 fractional-second 문법을 오류 없이 받지만 fractional 값을
  안정적으로 적용하지 않는다. 현재 초에 fractional lead를 더해 즉시 쓰면 clock
  phase에 따라 약 1초 오차가 남을 수 있었다. 최종 구현은 미래의 정수 초 경계를
  기다린 뒤 `.000`을 쓰며 scheduled target과 write bracket을 artifact에 기록한다.
  실제 경계 실험에서 0 ms lead의 offset은 약 2.37 ms였고, 전체 관측 uncertainty는
  약 49 ms였다.
- 모든 시험에서 `MMEM:FORM:READ:INF`는 기존 `OFF`로 복구했고 정확히 생성한
  `INT:\\ptm_*.csv`만 삭제했다.

### M5 standalone hardware gate 결과

`meas-2`의 `imcflow` conda Python과 Keysight VISA backend, board의 PetaLinux C
client를 `147.46.117.49:9910`으로 직접 연결했다. master/board TVM과
master/board/meas-2 measurement_utils revision을 매 run 전에 일치시켰다.

| Session | Interval | Samples | Status | Sample uncertainty | Ambiguous | Raw/NPZ |
|---|---:|---:|---|---:|---:|---|
| `m5_metadata_10ms_retry2_3a81f83` | 10 ms | 173 | complete | 49,007,284 ns | 30 | exact match |
| `m5_metadata_1ms_3a81f83` | 1 ms | 1,797 | complete | 69,939,146 ns | 423 | exact match |
| `m5_metadata_100us_3a81f83` | 100 us | 18,723 | complete | 36,866,756 ns | 2,225 | exact match |
| `m5_disconnect_partial_3a81f83` | 10 ms | 46 | partial | 38,050,835 ns | 7 | exact match |
| `m5_post_disconnect_recovery_3a81f83` | 10 ms | 176 | complete | 54,386,691 ns | 33 | exact match |

세 정상 interval run 모두 idle/busy/event/clear tag 순서를 보존했고 runner의
schema-v2 validator를 통과했다. 10 ms run의 clock setter는 DMM/server offset을
약 344 ms에서 41 ms로 줄였고, 1 ms run에서는 약 197 ms에서 4.5 ms로 줄였다.
100 us run은 시작 offset 약 77 ms가 policy limit 안이라 persistent clock을 쓰지
않았다.

100 us run의 첫 reading nominal time은 GET midpoint보다 약 20 ms 앞이었지만 전체
sample uncertainty 36.9 ms 안에 있다. 따라서 이 결과는 sample 간 100 us 간격은
정확히 보존하지만 tag와 sample의 sub-millisecond absolute 경계를 확정한다고
주장하지 않는다. 세 경계 주변 2,225 readings를 ambiguous로 표시한 것이 의도한
보수적 결과다.

강제 disconnect는 client exit 124 뒤 `reason=client disconnected`인 partial
artifact를 남겼다. raw checksum/value, metadata 복원과 internal-file 삭제가 모두
정상이며 daemon을 재시작하지 않은 다음 session도 complete로 끝났다. 최종 장비
상태는 metadata `OFF`, `SYST:ERR? = +0,"No error"`였다.

hardware에서 다음 세 통합 오류를 발견해 함께 수정했다.

1. TVM daemon probe가 protocol v2를 기대하던 문제를 v3로 맞췄다.
2. fractional clock write의 phase-dependent 약 1초 오차를 whole-second scheduling으로
   바꿨다.
3. `now` mode에서 post-ABOR stop time을 coverage 기준으로 써 정상 run을
   `truncated`로 분류하던 문제를 DMM freeze command 시작 시각 기준으로 바꿨다.

검증은 measurement_utils 관련 79 tests, metadata/protocol 17 focused tests,
TVM workflow 11 tests, C client `-Wall -Wextra -Werror` build를 통과했다.

### M6 TVM one-conv hardware acceptance 결과

BUGFIX default-off인 TVM `02beaed1f6dc9710afa63c2dfb91cf1f0ec69a02`와
measurement_utils `3a81f83dc2d15e88d3ea496aff548aac577a4e09`에서
`one_conv_small`을 Linux/AArch64로 다시 codegen/link했다. 이 모델은 tracked
handcraft directory가 없어서 standard codegen(`with_patch=false`)을 사용했다.

실행 전에 다음 gate가 모두 통과했다.

- master/board TVM SHA와 master/board/meas-2 measurement_utils SHA 일치
- 세 tracked tree clean
- `build_metadata.json`의 TVM/measurement_utils revision과 `dirty=0`
- board `execute_graph --power-build-info`의 동일 revision과 `dirty=0`
- daemon `HELLO_OK 3`와 measurement_utils revision 일치
- `scan_gen/scan_reg_files → const_scan_reg_files/0x00`, 16 NPZ 전체 nonzero 0,
  board scan programming exit 0

session `m6_one_conv_bugfixoff_02beaed_20260816a`에서 ARM graph executor는 input,
graph execution, output 저장과 cleanup을 모두 성공했고 power session도 정상
finalize됐다.

| 항목 | 결과 |
|---|---|
| status / tags | `complete` / 22 ordered events |
| interval / samples | 100 us / 3,026 |
| timestamp source | `dmm_reading_metadata` |
| raw CSV / NPZ current | 3,026 values exact match |
| raw checksum / internal file | SHA-256 match / exact file deleted |
| metadata restore | 기존 `OFF`로 복구 |
| sample uncertainty | 75,006,562 ns |
| ambiguous samples | 1,598 |
| DMM clock set | 약 986 ms → 51 ms, session end 약 99 ms |

`process_setup`, `input_setup`, `graph_execute`, kernel 이름, `device_setup`,
`compiled_transfer`, `const_transfer`, `policy_update`, tile 0 input/output transfer,
`output`, `cleanup` tag가 순서대로 저장됐다. workload의 kernel 구간은 수 ms보다
짧은 반면 DMM clock query uncertainty가 수십 ms이므로 ambiguity를 제외하면 여러
빠른 state의 확정 sample 수가 0인 것이 정상이다. 이 run으로 tag 호출 순서와 raw
sample 보존은 검증하지만 per-kernel 절대 power를 정밀하게 분리했다고 해석하지
않는다.

artifact를 TVM eval result의 `power/<session_id>` 경로로 회수해 schema-v2
validator, ambiguity-excluding summary와 DMM-first-reading 기본 축 plot을 모두
실행했다. `power_timeline.png`도 같은 artifact에서 생성됐다. execution 환경의 명령
정책 때문에 destructive cleanup을 포함한 전체 wrapper는 한 번에 호출하지 않고,
삭제 없는 rsync와 runner가 사용하는 동일 preflight/request/board execution/SCP/
validation 단계로 나누어 수행했다.

M6 중 두 runner 문제도 수정했다.

1. `_evl.linux`에서 `main.py` registry key를 만들 때 잘못 `.linux`를 다시 붙이던
   문제를 수정했다.
2. BUGFIX default-off의 `_evl.linux.bugfixoff` folder를 허용하고, Linux/ARM target을
   명시적으로 source하며, handcraft가 없는 모델용 `--no-patch` option을 추가했다.

ResNet/KWS/VWW 장시간 측정은 실제 DMM probe가 물린 rail 이름과 사용자가 설정한
전압을 placeholder가 아닌 config에 확정한 뒤 수행한다. 현재 one-conv의
`voltage_V=1.0`과 `measured_power`는 통합 검증용이므로 물리 power 결과로 인용하지
않는다.

### 현재 baseline과 변경 이유

현재 `measurement_utils` `ca3a9b923c126e4eba0f2c9c8a3e23c41f8bf096`은
GPIB interface open/close 시간을 제외하고 실제 GET 호출만
`trigger_before_monotonic_ns`와 `trigger_after_monotonic_ns`로 감싼다. TVM
`491eb94e38241aa82d1910a8734d88056e45d8aa`가 이 revision을 pin한다. 기존 sample
origin은 이 GET bracket의 midpoint이고, Board tag는 8-round TCP clock sync로
measurement server monotonic clock에 변환된다.

GPIB3의 Keysight 34465A에서 PyVISA로 다음 command를 실제 검증했다.

```text
MMEM:FORM:READ:INF ON
MMEM:STOR:DATA RDG_STORE,"INT:\imcflow_timestamp_test_20260816.csv"
MMEM:UPL? "INT:\imcflow_timestamp_test_20260816.csv"
```

20개 reading을 0.01 s 간격으로 저장한 파일에는 다음 정보가 들어갔다.

```csv
Start date:,08/15/2026,Start time:,14:09:17.620
Sample interval:,0.010000
Reading #,Reading
1,+2.68086926E-02
```

`MMEM:FORM:READ:INF?`는 테스트 전 `0`, 테스트 중 `1`, 복구 후 `0`이었고 DMM은
`No error`를 반환했다. 이 결과에 따라 DMM이 기록한 첫 reading 시각을 sample
origin으로 사용한다. GET bracket은 제거하지 않고 DMM metadata가 비정상이거나
장비 동작을 분석할 때 쓰는 진단값으로 남긴다.

관찰된 CSV 시작 시각은 millisecond 세 자리까지만 표현됐다. 실제 100 us sampling
에서는 시작 위치에 여러 sample의 모호성이 생길 수 있으므로, 구현은 보이지 않는
정밀도를 가정하지 않는다. hardware characterization에서 CSV/지원 format의 실제
정밀도와 timestamp 의미를 확인하고, 확인된 해상도를 정렬 uncertainty의 하한으로
반드시 기록한다.

### 시간 동기화의 목적과 기준 clock

목표는 세 장비 화면의 시계를 같은 숫자로 만드는 것이 아니라 다음 두 사건을 같은
시간축에서 비교하는 것이다.

1. Board C 코드가 tag 함수에서 timestamp를 얻은 순간
2. DMM이 각 current reading을 얻은 순간

measurement server를 canonical clock으로 사용한다. 정렬에는 wall clock 하나만
쓰지 않고 다음 clock domain과 변환을 명시한다.

```text
Bmono: Board CLOCK_MONOTONIC
Mmono: measurement server CLOCK_MONOTONIC
Mwall: measurement server CLOCK_REALTIME, UTC
Dwall: DMM reading metadata calendar clock, UTC로 해석

Bmono ──TCP minimum-RTT sync────────────▶ Mmono
Dwall ──bracketed DMM clock calibration─▶ Mwall
Mwall ──paired wall/monotonic anchor────▶ Mmono
```

sample과 tag를 비교하는 최종 축은 `Mmono`다. monotonic clock을 사용하면 측정 중
NTP 보정이나 사용자의 wall-clock 변경으로 sample/tag 순서가 뒤집히는 것을 막을 수
있다. `Mwall`은 DMM의 calendar timestamp를 `Mmono`에 연결하고 사람이 읽을 수 있는
UTC 결과를 만드는 데 사용한다.

Board wall clock은 정렬의 필수 입력이 아니다. 기존처럼 tag마다 `Bmono`를 보내고
TCP sync로 바로 `Mmono`에 변환하는 경로가 더 안정적이다. 다만 세 장비 상태를
감사할 수 있도록 session 시작 시 Board의 tightly-paired realtime/monotonic anchor와
가능하면 NTP 상태를 진단 metadata로 저장한다.

Master server clock은 workload 명령과 artifact 회수 시각을 기록할 뿐 sample/tag
정렬 공식에는 들어가지 않는다. 따라서 master와 다른 두 server의 wall clock 차이가
sample boundary를 바꾸면 안 된다. measurement server의 NTP 상태는 absolute UTC의
신뢰도를 설명하는 metadata이며, Board tag 정렬은 NTP 대신 session별 monotonic TCP
sync 결과를 사용한다.

### 반드시 파악하고 저장할 관계

| 관계 | 주 용도 | 측정 방법 | 주요 불확실성 |
|---|---|---|---|
| `Bmono → Mmono` | tag 호출 시각 변환 | 기존 8-round TCP sync에서 minimum RTT round 선택 | network asymmetry, RTT/2, C timestamp 후 socket write까지의 비용 |
| `Dwall → Mwall` | DMM 첫 reading 시각 변환 | DMM 날짜/시간 query 전후의 server wall/monotonic bracket; session 전후 반복 | VISA/GPIB 왕복, DMM clock query 해상도, clock drift |
| `Mwall → Mmono` | DMM wall time을 tag 비교 축으로 변환 | server에서 `monotonic-before → realtime → monotonic-after` anchor 기록 | local clock read bracket, 측정 중 wall-clock step |
| `Bmono → Dwall` | 직접 측정하지 않음 | 위 변환을 합성 | 각 변환 uncertainty의 합 |

DMM clock offset이 0이어야 할 필요는 없다. offset과 drift를 충분히 정확하게 알면
변환할 수 있다. 기본 정책은 먼저 read-only calibration으로 관계를 측정하는 것이다.
DMM 시간이 설정한 허용 범위를 벗어난 경우에만 server inventory의 명시적 policy에
따라 PyVISA로 시간을 맞춘 뒤 다시 calibration한다. 시간 설정 전후 값과 실행한 SCPI
command는 모두 기록하며, session마다 무조건 DMM persistent clock을 덮어쓰지 않는다.

### Tag 적용 규칙

첫 DMM reading 시각을 `D0`, metadata의 sample interval을 `P`, 첫 reading 번호를
1이라고 하면 다음처럼 계산한다.

```text
Dsample[i] = D0 + (i - 1) * P
Msample[i] = DMM-to-server-wall(Dsample[i])
Msample_mono[i] = wall-to-monotonic(Msample[i])
Mtag = board-to-server-monotonic(Btag)
```

tag state 변경은 `Msample_mono[i] >= Mtag`인 첫 sample부터 적용한다. DMM timestamp가
reading integration의 시작/중앙/끝 중 무엇을 의미하는지는 장비 문서와 controlled
test로 확정하여 `timestamp_semantics`로 저장한다. tag와 sample의 uncertainty interval이
겹치면 임의로 한쪽에 넣었다고 숨기지 않고 해당 경계를 `ambiguous`로 표시한다.

```text
tag interval:    [Mtag - Utag, Mtag + Utag]
sample interval: [Msample - Usample, Msample + Usample]
```

정렬 uncertainty에는 최소한 Board/server sync, DMM/server calibration, DMM metadata
해상도, session 동안의 DMM drift bound, server wall/monotonic anchor bracket과 DMM
integration aperture의 의미가 포함된다. 기존 GET uncertainty는 metadata origin의
uncertainty에 더하지 않고 독립 진단값으로 보존한다.

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
             │ direct TCP v3 (START/TAG/two-phase STOP)
             ▼
Measurement server (SSH alias: meas-2)
  ├─ runner request 검증
  ├─ measurement target을 server inventory의 실제 DMM으로 resolve
  ├─ Board/server와 DMM/server clock 관계 calibration
  ├─ local DmmManager/PyVISA로 group trigger와 sampling 시작
  ├─ tag event와 clock metadata 기록
  ├─ STOP_BEGIN에서 DMM buffer freeze
  ├─ reading metadata 포함 raw file 저장/upload/검증
  ├─ DMM metadata sample timestamp와 active tag를 결합
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
STOP_BEGIN: DMM buffer freeze
Board/server end clock sync
FINALIZE: metadata/raw upload and result materialization
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
STOP_BEGIN: DMM buffer freeze
Board/server end clock sync
FINALIZE: metadata/raw upload and result materialization
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
4. `STOP_BEGIN`은 이전 tag message 뒤에 같은 TCP stream으로 오므로 모든 tag가
   먼저 처리됐음이 보장된다.

tag API는 정상 경로에서 server ACK를 기다리지 않는다. socket write 실패만
즉시 반환하여 kernel 실행에 network RTT가 추가되지 않도록 한다.

## 3-clock 정렬 상세 구현

### 1. Board와 measurement server monotonic sync

현재 8-round TCP sync와 minimum-RTT 선택을 유지한다.

```text
board C util                    measurement server
    SYNC(client_send_ns)  ───▶  record recv/send monotonic ns
                         ◀───  SYNCED(server_recv_ns, server_send_ns)
```

- tag 함수는 state를 변경하거나 socket write를 시작하기 직전에 `CLOCK_MONOTONIC`
  timestamp를 얻는다.
- 여러 round 중 RTT가 가장 작은 round로 `Bmono → Mmono` offset을 계산한다.
- 각 tag에는 raw Board monotonic, 변환된 server monotonic, server receive monotonic,
  sequence, offset과 uncertainty를 보존한다.
- session 시작과 종료에 sync를 한 번씩 수행하여 offset 변화량을 측정한다. 변화가
  허용치를 넘으면 중간 tag 시간에 선형 drift correction을 적용하기 전에 clock
  source의 동작을 검증하고, 검증 전에는 변화량 전체를 uncertainty로 추가한다.
- Board의 realtime/monotonic anchor는 진단용으로 한 번 전송한다. tag 정렬 자체는
  realtime이나 Board NTP 상태에 의존하지 않는다.

### 2. Measurement server wall/monotonic anchor

measurement server에서 DMM clock query와 GET의 전후에 다음 순서로 anchor를 만든다.

```text
Mmono_before = CLOCK_MONOTONIC
Mwall        = CLOCK_REALTIME
Mmono_after  = CLOCK_MONOTONIC
```

대표 monotonic 시각은 midpoint로 두고 bracket의 절반을 local-anchor uncertainty로
저장한다. session 시작과 종료 anchor를 모두 남겨 측정 중 `CLOCK_REALTIME` step이나
비정상적인 frequency correction이 있었는지 검사한다. step이 발견되면 DMM metadata
기반 UTC array 생성은 실패 처리하되, raw DMM CSV와 monotonic tag trace는 보존한다.

모든 calendar timestamp는 내부적으로 UTC epoch nanosecond로 변환한다. DMM에는
timezone field가 없으므로 server inventory에 `CLOCK.TIMEZONE=UTC`를 명시하고 raw
date/time string도 함께 저장한다.

### 3. DMM clock calibration

`DmmManager`에 다음 책임을 가진 API를 추가한다. 실제 command spelling과 fractional
second 지원 범위는 Phase 0 hardware characterization에서 34465A로 확정한다.

```text
query_clock_bracketed(dmm_name, rounds=N)
set_clock_from_server(dmm_name, utc_time)       # policy가 허용할 때만
verify_reading_metadata_support(dmm_name)
```

한 calibration round는 다음 순서다.

1. server wall/monotonic before anchor를 기록한다.
2. PyVISA로 DMM date/time을 query한다.
3. server wall/monotonic after anchor를 기록한다.
4. midpoint와 DMM 응답으로 `Dwall → Mwall` offset을 계산한다.
5. 여러 round 중 유효 응답이면서 bracket이 가장 작은 결과를 대표값으로 선택한다.

session 시작 전과 reading 회수 후에 calibration을 수행한다. 두 offset 차이를
`observed_dmm_clock_drift_ns`로 저장한다. DMM RTC와 sampling interval이 같은 oscillator를
쓴다는 사실이 확인되기 전에는 drift를 sample interval에 자동 보정하지 않고 정렬
uncertainty에 보수적으로 포함한다.

server DMM inventory의 기존 `POWER.DMM_GPIB3` entry에 다음 policy를 추가한다.
`_inventory_targets()`는 현재 logical name에서 VISA address만 반환하므로, 구현 시
address와 clock/metadata policy를 가진 structured target을 반환하도록 확장한다.

```json
{
  "POWER": {
    "DMM_GPIB3": {
      "DEVICE": "dmm_gpib3",
      "READING_METADATA": true,
      "CLOCK": {
        "TIMEZONE": "UTC",
        "POLICY": "verify_and_set_if_needed",
        "MAX_OFFSET_MS": 100,
        "CALIBRATION_ROUNDS": 8
      }
    }
  }
}
```

- `verify_only`: offset만 측정하고 DMM clock을 변경하지 않는다.
- `verify_and_set_if_needed`: offset이 threshold를 넘으면 PyVISA로 맞춘 후 다시
  calibration한다.
- `disabled`는 unit-test fake backend와 명시적인 legacy 분석에만 허용한다.
- 실제 tagged power 기본값은 `POLICY=verify_and_set_if_needed`와
  `READING_METADATA=true`다.

DMM clock을 변경한 경우 `before`, `set_request`, `after`, SCPI error queue와
setting precision을 session에 기록한다. 장비 시간이 parse되지 않거나 재설정 후에도
offset/uncertainty threshold를 만족하지 못하면 workload 시작 전에 실패한다.

### 4. DMM metadata와 raw reading 회수

각 DMM은 session 시작 전에 현재 metadata setting을 저장하고 다음 순서로 준비한다.

```text
old_metadata_mode = MMEM:FORM:READ:INF?
MMEM:FORM:READ:INF ON
configure current acquisition
GET
```

`STOP`의 `now` mode에서는 다음 순서를 사용한다.

1. 현재처럼 모든 DMM에 먼저 `ABORt`를 보내 buffer를 freeze한다.
2. rail별로 collision이 없는 session 전용 internal filename을 만든다.
3. `MMEM:STOR:DATA RDG_STORE,<filename>`으로 같은 reading buffer를 저장한다.
4. `MMEM:UPL? <filename>`의 IEEE block payload를 byte 단위 그대로 회수한다.
5. raw CSV를 artifact에 먼저 durable write하고 SHA-256을 계산한다.
6. metadata와 current values를 strict parser로 읽는다.
7. 기존 `DATA:REM?` 결과와 count/value를 비교한 뒤 buffer를 비운다. NPZ의
   `current_A[]`는 검증을 통과한 raw CSV reading을 source of truth로 사용한다.
8. 업로드와 checksum 확인이 끝난 정확한 session 파일만 DMM 내부 저장소에서
   삭제한다. wildcard나 directory 단위 삭제는 사용하지 않는다.
9. 성공/실패와 관계없이 `MMEM:FORM:READ:INF`를 session 전 값으로 복구하고
   복구 query 결과를 기록한다.

파일명에는 전체 user-provided string을 직접 넣지 않고 sanitized session hash와 rail
index만 사용한다. metadata restore나 DMM file cleanup이 실패하면 current data가
있더라도 session에 cleanup error를 남기고 reservation을 해제한다. 다음 session은
preflight에서 DMM state를 다시 검증한다.

raw CSV parser는 다음을 모두 검사한다.

- `Start date`, `Start time`, `Sample interval`이 하나씩 존재하는지
- reading number가 1부터 연속인지
- CSV sample count와 `DATA:POIN?`/`DATA:REM?` count가 같은지
- metadata interval과 사전에 query한 `SAMP:TIM?` 값이 tolerance 안에서 같은지
- current 값이 finite number인지
- date/time precision과 fractional digit 수가 예상 format인지

### 5. DMM metadata 기반 sample timestamp 생성

rail별 primary sample origin은 GET midpoint가 아니라 CSV의 첫 reading timestamp다.

```text
dmm_sample_wall[i] = dmm_first_reading_wall + i * metadata_sample_interval
server_sample_wall[i] = dmm_sample_wall[i] + dmm_to_server_wall_offset
server_sample_mono[i] = wall_to_monotonic(server_sample_wall[i])
```

여기서 array index `i=0`은 CSV `Reading #=1`이다. `SAMP:TIM?` 값은 설정 검증용,
CSV `Sample interval`은 해당 raw file의 최종 시간축 source로 사용한다. 두 값이
tolerance를 벗어나면 자동으로 하나를 선택하지 않고 artifact를 `aborted`로 만든다.

기존 값은 다음 용도로 유지한다.

- `trigger_before_monotonic_ns`, `trigger_after_monotonic_ns`: 실제 GET 전달 구간
- `trigger_origin_monotonic_ns`: GET 진단용 midpoint
- `time_from_trigger_s[]`: compatibility array; 새 분석에서는 사용 중단 예정

새 NPZ에는 다음 array와 scalar를 추가한다.

```text
reading_number[]
current_A[]
power_W[]
time_from_first_reading_s[]
server_wall_time_ns[]
server_monotonic_time_ns[]
tag_state_id[]
tag_boundary_ambiguous[]
sample_time_uncertainty_ns
voltage_V
```

`server_monotonic_time_ns[]`는 한 boot 안에서 비교하기 위한 값이고, 장기 보존과
cross-server 분석에는 UTC `server_wall_time_ns[]`와 time-alignment metadata를
사용한다. integer nanosecond array를 먼저 만들고 초 단위 float는 표시/plot 단계에서
변환하여 긴 session의 float precision 손실을 피한다.

### 6. Tag state materialization과 uncertainty

현재 `_materialize_states()`가 사용하는 sample time 입력을 GET midpoint 기반 array에서
metadata 기반 `server_monotonic_time_ns[]`로 바꾼다. 각 tag boundary에 대해 확실하게
이전/이후인 sample은 기존 규칙으로 분류하고 uncertainty interval이 겹치는 sample은
`tag_boundary_ambiguous=true`로 표시한다.

```text
tag_uncertainty = board_server_sync_uncertainty
sample_uncertainty = dmm_server_clock_uncertainty
                   + server_wall_mono_anchor_uncertainty
                   + dmm_metadata_resolution_uncertainty
                   + dmm_clock_drift_bound
                   + timestamp_semantics_uncertainty
```

`sample_alignment_uncertainty_ns`는 tag별로
`tag_uncertainty + sample_uncertainty`를 기록한다. energy/average summary는 기본적으로
ambiguous sample을 포함하되 개수를 명시하고, loader/plot option으로 제외하여 민감도
분석을 할 수 있게 한다.

software clock sync와 TCP만 사용하므로 이 경로는 cycle-accurate하지 않다. 계산된
uncertainty가 요구 sample boundary보다 크면 결과를 더 정밀한 것처럼 표시하지 않는다.
수십 microsecond 이하의 확정 경계가 필요하면 GPIO/trigger line 같은 hardware marker가
별도로 필요하다.

### 7. Failure와 fallback 정책

실제 hardware run의 기본 `timestamp_source`는 `dmm_reading_metadata`다. 다음 경우에는
GET midpoint로 조용히 fallback하지 않는다.

- metadata enable/query 실패
- raw file 저장/업로드 실패
- timestamp parse 실패
- CSV와 DMM buffer의 sample count/value 불일치
- DMM/server clock calibration 실패 또는 uncertainty threshold 초과
- session 중 measurement server wall clock step 발견

이 경우 가능한 raw file, tag trace, GET bracket과 error queue를 남기고 session을
`aborted` 또는 `partial`로 끝낸다. 기존 GET-midpoint path는 fake backend, 과거 artifact
loader와 명시적인 `timestamp_source=trigger_bracket_legacy` 분석에만 남긴다.

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

## Tagged protocol v3 전환 계획

기존 command-line-like `START --names ...` v1 protocol과 그 bridge/RPyC 코드는
그대로 남기되 새 tagged measurement server에서는 사용하지 않는다. 새 C API와
server의 현재 protocol은 v2다. DMM metadata time model은 artifact 의미와 stop
sequence를 바꾸므로 tagged client/server를 v3로 함께 올리고, HELLO에서 구 version
혼용을 fail-fast한다. v1 bridge/RPyC protocol은 변경하지 않는다.

```text
HELLO 3
SYNC <phase> <client_send_monotonic_ns>
CLOCK_SYNC <phase> <offset_ns> <best_rtt_ns> <uncertainty_ns> <sample_count>
BOARD_CLOCK <phase> <mono_before_ns> <realtime_ns> <mono_after_ns>
START_JSON <payload_length>\n<payload>
TAG_SET <seq> <client_ns> <key_len> <value_len>\n<key><value>
TAG_CLEAR <seq> <client_ns> <key_len>\n<key>
TAG_EVENT <seq> <client_ns> <name_len>\n<name>
STOP_BEGIN
SYNC end ...
CLOCK_SYNC end ...
BOARD_CLOCK end ...
FINALIZE
ABORT <reason_length>\n<reason>
```

대표 응답:

```text
HELLO_OK 3
CLOCK_SYNCED <phase>
BOARD_CLOCKED <phase>
STARTED <session_id> <resolved_config_length>\n<resolved_config_json>
STOP_READY <session_id>
STOPPED <session_id> <summary_length>\n<summary_json>
ERROR <code> <message>
```

`phase`는 `start` 또는 `end`다. `STOP_BEGIN`을 받으면 server는 먼저 모든 DMM을
ABORt하여 measurement buffer를 freeze하고 `STOP_READY`를 반환한다. 그 뒤 Board가
end sync를 수행하므로 clock drift 측정을 위한 network/CPU activity가 current
measurement에 섞이지 않는다. `FINALIZE`에서 metadata 저장/업로드, raw-data 검증,
tag materialization과 artifact 작성을 수행한다. `STOP_BEGIN` 이후 disconnect되면
server가 독립적으로 partial finalize한다.

기존 v1 client/server test와 tagged v2 artifact loader test는 regression으로
유지한다. 새 v3 server가 v1/v2 frame을 동시에 받아들이게 만들지는 않는다. 두
실행 경로를 분리하여 새 server에 불필요한 compatibility layer와 RPyC dependency가
들어오지 않게 한다.

## Result artifact

measurement server의 canonical directory는 server가 관리하는 root 아래로
고정한다.

```text
<POWER_RESULT_ROOT>/<session_id>/
├── request.json
├── resolved_config.json
├── session.json
├── time_alignment.json
├── tags.jsonl
├── summary.json
├── raw/
│   ├── DMM_GPIB3.csv
│   └── checksums.json
└── rails/
    └── DMM_GPIB3.npz
```

rail NPZ에는 최소한 다음 array가 들어간다.

```text
reading_number[]
current_A[]
time_from_trigger_s[]
time_from_first_reading_s[]
server_wall_time_ns[]
server_monotonic_time_ns[]
power_W[]
tag_state_id[]
tag_boundary_ambiguous[]
sample_time_uncertainty_ns
```

`raw/DMM_GPIB3.csv`는 DMM에서 upload한 byte를 newline이나 숫자 format 변경 없이
그대로 저장한다. plot, NPZ와 summary가 사용한 `current_A[]`는 이 파일에서 parse한
값이어야 한다. `checksums.json`에는 raw byte SHA-256, size, parser version, CSV와
`DATA:REM?` 비교 결과를 기록한다.

`session.json`에는 다음 정보를 저장한다.

- artifact schema version 2와 session status: `complete`, `partial`, `aborted`,
  `truncated`
- 시작/종료 UTC와 monotonic 시각, trigger timing
- Board/server start/end clock sync offset/RTT/uncertainty
- DMM/server start/end clock calibration, offset, drift bound와 SCPI bracket
- server wall/monotonic anchor와 wall-clock step 검사 결과
- DMM raw start date/time string, UTC 해석, fractional precision,
  `timestamp_semantics`와 metadata restore 결과
- requested/resolved DMM configuration
- rail별 실제 sample interval/count/coverage
- configured buffer coverage와 실제 회수된 `collected_coverage_s`; 완료 session은
  두 coverage 모두에 대해 truncation 검사를 통과해야 함
- TVM model, board, chip, checkpoint, git revision 등 runner metadata
- tag drop/send error 여부

`time_alignment.json`에는 위 clock 변환의 원시 관측값과 선택된 calibration round,
공식, uncertainty component를 machine-readable 형태로 저장한다. 분석 utility는 이
파일과 raw CSV만으로 sample timestamp와 tag state를 다시 생성할 수 있어야 한다.

`tags.jsonl`은 원본 tag event와 active state snapshot ID mapping을 보존한다.
`summary.json`은 rail별/active-tag별 평균 current, 평균 power, energy를 제공하되
ambiguous sample 수와 비율도 함께 제공하고 NPZ/raw 원본으로 언제든 다시 계산할 수
있게 한다.

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
| DMM clock/metadata capability 또는 pre-start calibration 실패 | workload 시작 전 실패, GET midpoint fallback 금지 |
| 일부 DMM start 실패 | 시작한 DMM 모두 중단, reservation 해제, workload 실행 안 함 |
| tag send 실패 | workload는 계속, session을 degraded로 표시 |
| workload non-zero exit | partial sample finalize 후 원래 exit status 보존 |
| board SSH timeout/process kill | measurement server가 disconnect 감지 후 partial finalize |
| raw metadata upload/parse/count 검증 실패 | 보존 가능한 raw와 SCPI error를 남기고 `aborted`/`partial`, synthetic timestamp 생성 금지 |
| metadata mode restore/internal-file cleanup 실패 | cleanup error 기록, reservation 해제, 다음 session preflight 강제 |
| runner가 결과를 못 받음 | measurement 결과는 server에 보존, session ID로 재-fetch 가능 |
| sample buffer 조기 종료 | `truncated`, power 단계 실패 |

runner는 executable status와 power status를 별도로 보고한다. accuracy 실행은
성공했지만 power artifact가 불완전한 경우 전체를 단순 성공으로 표시하지 않는다.

## 기존 구현 단계 (완료된 baseline)

아래 Phase 0~7은 2026-08-15에 완료된 direct-PyVISA protocol v2와 GET-midpoint
baseline을 설명한다. 이 절의 v2/STOP/trigger-origin 표현은 당시 구현 기록이며,
앞 절의 metadata protocol v3 설계로 교체할 대상이다. 기존 vertical-slice 순서와
회귀 범위를 보존하기 위해 삭제하지 않는다.

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

## Metadata time alignment 구현 단계

### Phase M0: DMM timestamp capability 확정

Test 계획의 GPIB3 hardware characterization을 먼저 수행한다. 이 단계에서 DMM clock
query/set command, reading metadata raw format, first-reading timestamp precision과
semantics를 확정한다. 관찰 결과가 계획의 가정과 다르면 parser나 uncertainty model을
먼저 고치며, GET midpoint로 되돌아가서 통과시키지 않는다.

### Phase M1: DMM raw metadata pipeline

1. `DmmManager`에 metadata mode query/enable/restore와 exact internal file
   store/upload/delete API를 추가한다.
2. `now` 중단을 `abort/freeze`, `store/upload`, `drain/compare` 단계로 분리한다.
3. raw byte writer, SHA-256, strict reading metadata parser를 추가한다.
4. exception injection test로 각 단계 실패 시 metadata restore, exact-file cleanup과
   VISA reservation 해제를 검증한다.
5. 기존 manager/RPyC 경로의 public behavior는 바꾸지 않고 tagged backend만 새 API를
   사용한다.

### Phase M2: Clock calibration과 time model

1. measurement server wall/monotonic paired anchor utility를 추가한다.
2. DMM clock multi-round bracket, minimum-bracket selection, UTC conversion과
   start/end drift 관측을 구현한다.
3. `TimeAlignment` data model에 raw observations, selected mapping, uncertainty
   components와 validation result를 모은다.
4. DMM first-reading timestamp와 interval로 integer-nanosecond sample axis를 만든다.
5. GET timing은 별도 diagnostic field로 유지하고 metadata path와 섞이지 않는지
   fake-clock unit test로 확인한다.

### Phase M3: Protocol v3와 Board end sync

1. C client와 tagged server를 함께 HELLO v3로 올린다.
2. Board realtime/monotonic start/end anchor와 phase가 있는 clock sync frame을
   추가한다.
3. `STOP_BEGIN`에서 DMM을 먼저 freeze하고 `STOP_READY` 뒤 end clock sync,
   `FINALIZE`에서 raw 회수와 artifact 생성을 수행한다.
4. `STOP_BEGIN` 전후 disconnect, timeout과 duplicate frame의 idempotency/partial
   finalize를 검증한다.
5. `dmm_session_stop()` public C API는 그대로 유지하여 TVM host code 변경을
   최소화한다.

### Phase M4: Schema v2 artifact와 분석 utility

1. `time_alignment.json`, raw file/checksum, metadata-aligned NPZ를 생성한다.
2. `_materialize_states()`를 metadata sample monotonic axis로 전환한다.
3. ambiguous boundary flag와 uncertainty-aware summary/filter를 추가한다.
4. schema-v1 loader는 read-only compatibility로 유지하고 새 run을 schema v1로
   쓰는 기능은 제공하지 않는다.
5. raw file에서 NPZ/tag state를 재생성하는 deterministic rebuild test를 추가한다.

### Phase M5: Standalone hardware gate

1. `meas-2`에 같은 measurement_utils revision을 배포하고 `imcflow` conda로 daemon을
   실행한다.
2. `petalinux`에서 같은 revision으로 link한 standalone C smoke를 실행한다.
3. 10 ms controlled tag boundary로 raw/clock/artifact contract를 확인한다.
4. 1 ms와 100 us로 반복하고 timestamp resolution보다 작은 경계가 ambiguous로
   표시되는지 확인한다.
5. 연속 run, 강제 disconnect와 partial finalize를 거쳐 DMM state 누출이 없는지
   확인한다.

### Phase M6: TVM/runner 통합과 acceptance

1. 검증된 measurement_utils commit을 TVM gitlink로 pin하고 Board/`meas-2`를
   `pull --ff-only`로 맞춘다.
2. master runner가 raw/time-alignment/schema-v2 completeness와 checksum을 검사하고
   모두 SCP하도록 갱신한다.
3. loader/plot default x-axis를 DMM metadata time으로 전환한다.
4. one-conv에서 gate를 통과한 뒤 ResNet, KWS, VWW 순으로 acceptance를 수행한다.
5. 각 run에서 실제 rail/voltage, revision identity, uncertainty와 ambiguous sample
   비율을 함께 보고한다.

## Test 계획

### Phase M0: GPIB3 timestamp hardware characterization

TVM이나 Board C code를 변경하기 전에 `ssh meas-2`에서 `imcflow` conda 환경을
사용하여 Keysight 34465A 한 대만 대상으로 다음을 검증한다. 이 단계는 장비의
읽기/설정 capability를 확정하는 read-mostly test다. metadata mode처럼 원래 상태로
돌릴 수 있는 persistent 값은 기존 값을 먼저 저장하고 `finally`에서 복구한다.
DMM clock은 테스트 목적으로 과거 값으로 복구하지 않으며, 설정 capability 검증이
필요하면 inventory policy를 명시적으로 켠 뒤 UTC로 맞추고 재검증한다.

1. IDN/options/firmware와 `MMEM:FORM:READ:INF?` support를 기록한다.
2. DMM date/time query command, 응답 format, timezone 부재, fractional-second
   precision과 query RTT 분포를 확인한다.
3. server time을 DMM에 설정할 때 지원되는 최소 단위와 setting latency를 확인한다.
   clock을 실제 변경했다면 UTC로 맞은 상태를 재확인하고 before/set/after를 남긴다.
4. 10 ms, 1 ms, 실제 target 100 us interval로 각각 짧게 측정한다.
5. CSV와 장비가 지원하는 다른 reading file format의 start timestamp precision을
   비교한다. 더 정밀한 format이 있으면 raw canonical format으로 선택하되 사람이
   볼 수 있는 CSV도 함께 보존한다.
6. metadata의 `Start time`이 첫 reading의 integration 시작/중앙/끝 중 어느 쪽인지
   manual과 controlled delayed-trigger test로 확인한다.
7. metadata sample interval, `SAMP:TIM?`, reading count와 `DATA:REM?` 값을 비교한다.
8. session 전후 DMM/server offset으로 짧은 시간 동안의 drift와 반복성을 측정한다.
9. exact internal test file만 정리되었고 metadata setting이 이전 값으로 복구됐는지
   query한다.

결과는 실제 command/응답, 최소·중앙·최대 bracket, timestamp resolution과 선택한
`timestamp_semantics`를 이 문서의 수행 결과에 추가한다. 이 결과 없이 100 us
sample에 sub-millisecond absolute alignment를 주장하지 않는다.

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
6. 정상 `START → TAG → STOP_BEGIN → end sync → FINALIZE`를 한 번 수행한다.
7. 두 번째 실행으로 이전 session의 VISA resource와 reservation이 남지 않았는지
   확인한다.
8. artifact에서 raw CSV checksum, metadata first-reading time, sample count, ordered
   tag event, tag state ID와 ambiguous boundary를 확인한다.
9. 결과를 master로 SCP하고 `activate` 환경에서 loader로 읽는다.
10. 10 ms interval과 의도적인 50 ms sleep/tag를 먼저 사용하여 예상 sample boundary와
    계산 결과를 사람이 확인한 뒤 실제 100 us 설정으로 반복한다.
11. Board/server 및 DMM/server start/end offset, drift, 모든 uncertainty component가
    artifact에 존재하고 계산 결과가 component 합과 같은지 확인한다.
12. metadata mode가 이전 값으로 복구되고 DMM internal test file과 reservation이
    남지 않았는지 확인한다.

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
- DMM date/time parser와 UTC epoch 변환
- DMM clock calibration에서 minimum-bracket round 선택과 offset 계산
- server wall/monotonic paired anchor와 wall-clock step 검출
- metadata CSV header/reading strict parse와 raw byte checksum
- metadata interval/count/value mismatch reject
- metadata enable/restore와 예외 경로의 exact-file cleanup
- DMM/server start/end drift bound 계산
- metadata sample time을 server monotonic으로 변환하고 tag state를 materialize
- uncertainty interval이 겹치는 tag boundary의 ambiguous 표시
- 실제 hardware request에서 metadata 실패 시 GET midpoint로 fallback하지 않음
- disconnect 시 partial finalize와 reservation release
- fake DMM record를 tag state별로 정확히 분류
- duration budget coverage 부족 시 preflight reject
- truncated session 판정
- 기존 bridge/RPyC v1 START/GO와 tagged artifact schema v1 loader regression

### C API test

- fake measurement TCP server 대상 v3 START/TAG/two-phase STOP
- standalone smoke source가 TVM header/library 없이 ARM용으로 build됨
- config size/invalid path/error response
- tag fire-and-forget ordering
- broken pipe와 timeout
- inactive session tag no-op
- start/end clock sync offset 선택과 Board wall/monotonic anchor
- `STOP_BEGIN` 뒤 DMM freeze ACK를 받은 후 end sync와 `FINALIZE` 순서
- `STOP_BEGIN` 이후 disconnect 시 server-side partial finalize
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
- raw CSV, checksum, `time_alignment.json`, schema-v2 NPZ completeness 검사
- plot/summary가 `time_from_trigger_s`가 아니라 metadata-aligned time을 기본 사용
- ambiguous sample 포함/제외 option이 동일 raw artifact에서 재현 가능한지 검사

### Hardware acceptance test

standalone gate가 끝난 뒤 TVM 통합 결과만 여기서 검증한다.

1. `DMM_GPIB3`를 선택한 one-conv에서 raw CSV와 metadata-aligned NPZ를 생성한다.
2. GET midpoint와 metadata first-reading time 차이는 진단값으로 기록하되 tag
   materialization이 metadata time을 사용했는지 확인한다.
3. tag가 없는 구간도 implicit session tag로 분류되는지 확인한다.
4. region/tile tag가 trace에서 순서대로 보이고 경계 sample의 ambiguity가 기대한
   위치에만 나타나는지 확인한다.
5. ResNet single input 전체 실행 coverage와 start/end clock drift를 확인한다.
6. KWS/VWW dataset 실행에서 sample index tag와 장시간 drift bound를 확인한다.
7. timeout/강제 종료 후 raw/partial artifact, DMM metadata restore와 reservation
   해제를 확인한다.
8. `meas-2` artifact가 master output으로 자동 회수되고 raw checksum이 유지되는지
   확인한다.
9. power disabled baseline과 accuracy/runtime regression을 비교한다.
10. daemon을 재시작하지 않고 두 번 연속 실행하여 DMM clock/metadata/internal-file
    state가 다음 session에 누출되지 않는지 확인한다.

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
- 실제 hardware run의 sample origin은 DMM reading metadata의 first-reading timestamp다.
  GET midpoint는 진단값이며 silent fallback으로 사용되지 않는다.
- Board tag, DMM sample과 measurement server clock의 모든 변환, start/end offset,
  drift와 uncertainty component가 `time_alignment.json`에 저장된다.
- DMM에서 upload한 raw reading file이 byte 그대로 보존되고, NPZ/plot에 사용한
  `current_A[]`와 checksum 및 sample count가 일치한다.
- tag가 어느 sample boundary에 속하는지 uncertainty로 확정할 수 없는 경우
  `tag_boundary_ambiguous`가 표시된다.
- metadata mode와 DMM clock persistent setting은 policy대로 관리되고 성공/실패
  경로 모두에서 이전 metadata mode가 복구된다.
- 100 us sampling 결과는 실제 확인한 DMM timestamp resolution보다 높은 absolute
  time precision을 주장하지 않는다.
- result artifact가 session ID 아래 생성되고 runner로 자동 회수된다.
- 새 tagged 실행 경로에서 board가 measurement server에 직접 TCP로 접속하고,
  measurement server가 RPyC 없이 로컬 PyVISA를 호출한다.
- 기존 bridge/RPyC 코드와 v1 regression test는 그대로 유지된다.
- 실패/timeout에서도 partial artifact와 reservation cleanup이 보장된다.
- power config가 없을 때 기존 chip execution과 accuracy에 변화가 없다.
- current retry, warmup, MMIO barrier, chip lock, remote `.env` 구조가 유지된다.

## 권장 추가 commit 순서

아래는 이미 완료된 direct server/tagged session 구현 위에 metadata time model을
추가하는 순서다. 현재 두 repository의 `feat/power_tagged_measurement` branch를
그대로 사용하며 새 worktree나 추가 branch를 만들지 않는다. 각 단계는 local test와
해당 server smoke를 통과한 뒤 push하고, 다른 server에서는 dirty tree가 없는 것을
확인한 후 `pull --ff-only`로 동기화한다.

### `measurement_utils` repository

1. `feat(dmm): capture reading metadata and preserve raw records`
   - metadata support query, enable/restore, unique internal filename, upload/parser,
     SHA-256와 exact-file cleanup을 한 commit에 묶는다.
2. `feat(power): calibrate DMM and server clock domains`
   - DMM date/time bracket, UTC policy, wall/monotonic anchor, offset/drift/uncertainty
     model을 추가한다.
3. `feat(capi): add end clock sync and two-phase measurement stop`
   - tagged protocol v3, Board clock anchor, `STOP_BEGIN/FINALIZE`, disconnect partial
     finalize를 구현한다.
4. `feat(power): align samples from DMM reading metadata`
   - GET midpoint 대신 metadata first-reading time을 사용하고 schema-v2 NPZ와
     `time_alignment.json`을 만든다.
5. `feat(power): report uncertain tag boundaries`
   - uncertainty component, ambiguous sample flag, summary 포함/제외 계산을 추가한다.
6. `test(power): verify metadata timing, raw integrity, and clock failure paths`
   - fake unit/integration, v1 regression, schema-v1 loader, GPIB3 standalone 결과를
     포함한다.

### TVM repository

1. `build(imcflow): update measurement utils for metadata time alignment`
   - 검증되어 push된 measurement_utils revision으로 gitlink를 갱신한다.
2. `feat(codegen): validate and fetch metadata-aligned power artifacts`
   - runner completeness/checksum/schema 검사와 raw artifact SCP를 추가한다.
3. `feat(imcflow): analyze power samples on DMM metadata timeline`
   - loader, filter, summary와 plot의 기본 x-axis 및 ambiguous option을 갱신한다.
4. `test(imcflow): cover metadata power artifact workflows`
   - schema v1 호환, schema v2 필수 파일, checksum failure, ambiguity option과
     power-disabled regression을 검증한다.
5. `docs(imcflow): document three-clock power sample alignment`
   - Phase 0/standalone/one-conv 실제 측정값과 운영 절차를 이 문서에 반영한다.

이 순서에서는 `measurement_utils`의 API와 artifact contract를 먼저 고정하고,
그 commit을 TVM submodule pointer로 반영한 뒤 TVM 통합을 진행한다. 기존
`power` branch는 behavior reference로만 사용하고 commit merge/cherry-pick은
하지 않는다. 기존 bridge/RPyC 파일도 보존하되 새 tagged measurement 실행
경로에는 연결하지 않는다.
