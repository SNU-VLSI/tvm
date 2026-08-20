# Power 측정 quick start

이 문서는 FPGA board에서 모델을 실행하면서 measurement server의 DMM으로
전류와 power를 측정하는 방법, 그리고 구현의 핵심 흐름만 요약한다. 기본 권장
설정은 **kernel region마다 측정하는 `scope=REGION`, 실행 종료 시 즉시 측정을
멈추는 `mode=now`**이다.

## 1. 사용법

### 구성과 사전 조건

- master server: TVM 실행 및 결과 수집. Python 환경은 `activate`로 활성화한다.
- board: `ssh petalinux`로 접근하는 PetaLinux FPGA board이다.
- measurement server: `ssh meas-2`로 접근하며 `imcflow` conda 환경과 PyVISA를
  사용한다.
- DMM: [dmm_gpib3.json](../dmm_configs/dmm_gpib3.json)에 정의된
  `DMM_GPIB3` (`GPIB1::3::INSTR`)을 사용한다.
- master와 board의 TVM commit, master·board·measurement server의
  `measurement_utils` commit이 서로 같고 tracked working tree가 모두 clean해야
  한다. 실행 binary에 기록된 build commit도 같아야 한다. runner가 실행 전에
  이 조건을 자동 검사한다.
- board에서 measurement server의 TCP 주소에 접근할 수 있어야 한다. 기본
  port는 `9910`이다.

master에서 다음 환경을 준비한다.

```bash
cd /root/project/tvm/tvm_practice/test_imcflow/codegen
activate

# board에서 접근 가능한 measurement server의 IP 또는 hostname
export POWER_MEASUREMENT_HOST=<measurement-server-address>
export POWER_MEASUREMENT_PORT=9910
```

runner는 measurement server가 꺼져 있으면 `meas-2`에 SSH로 접속하여
`power_tagged_measurement_server`를 자동 실행한다. 이때 로컬 DMM configuration을
measurement server로 복사하므로 board에는 PyVISA가 필요하지 않다.

### 단일 chip test

```bash
./run_chiptest.sh \
  --power-config power_configs/region.json \
  resnet8_subset31_pretrained_orig_evl.linux random
```

`--power-config` 대신 환경변수로도 지정할 수 있다.

```bash
export IMCFLOW_POWER_CONFIG=power_configs/region.json
./run_chiptest.sh resnet8_subset31_pretrained_orig_evl.linux random
```

### Dataset 평가

다음 예시는 ResNet의 CIFAR-10 sample 0, 1을 실행한다.

```bash
./run_dataset_eval.sh \
  --power-config power_configs/region.json \
  --model resnet8_subset31_pretrained_orig_evl.linux \
  --dataset cifar10 \
  --indices 0,1
```

주요 runner option은 다음과 같다.

| Option | 의미 |
|---|---|
| `--power-config FILE` | power 측정을 활성화하고 측정 JSON을 선택한다. 생략하면 측정하지 않는다. |
| `--skip LIST` | 지정한 runner step을 생략한다. 기존 binary를 재사용할 때도 revision 검사는 수행된다. |
| `--model DIR` | dataset runner에서 사용할 generated model directory이다. |
| `--dataset NAME` | `dataset/` 아래의 dataset 이름이다. |
| `--indices LIST` | 측정할 sample index를 직접 지정한다. |
| `--output DIR` | dataset 평가 결과를 저장할 로컬 경로이다. |

### Power configuration

제공되는 설정은 다음과 같다.

| 파일 | 용도 |
|---|---|
| [region.json](../power_configs/region.json) | 각 generated kernel region을 별도 DMM trace로 측정한다. |
| [tile.json](../power_configs/tile.json) | 각 tile의 `RUN` write부터 interrupt ACK 및 `INTR_DONE` write까지를 별도 DMM trace로 측정한다. 이 구간에는 안정성을 위한 invoke MMIO barrier가 포함되며 TILE loop는 항상 비활성화된다. |
| [default.json](../power_configs/default.json) | 기본 `REGION` 측정이며 최대 예상 시간은 300초이다. |
| [short_run.json](../power_configs/short_run.json) | 짧은 `MODEL` 측정용이며 예상 시간은 5초이다. |

JSON의 주요 항목은 다음과 같다.

| 항목 | 값과 의미 |
|---|---|
| `enabled` | `false`이면 runner는 power 측정을 건너뛴다. |
| `scope` | TVM-only 배치 정책. `MODEL`은 model 실행 전체, `REGION`은 generated region kernel, `TILE`은 tile invoke마다 별도 trace를 만든다. 기본은 `REGION`이다. |
| `mode` | `now`만 지원한다. region END에서 DMM을 즉시 멈추고 현재 sample을 회수한다. |
| `region_loop.loop_enable` | `MODEL` scope에서만 지원한다. `true`이면 model body 전체를 최소 조건이 충족될 때까지 반복한다. `REGION`과 `TILE` scope에서는 loop를 사용할 수 없다. |
| `region_loop.min_samples` | 모든 rail에서 확보해야 할 최소 acquired sample 수이다. |
| `region_loop.min_seconds` | 실제 DMM GET 이후 확보해야 할 최소 시간이다. sample/time 조건을 함께 지정하면 둘 다 만족해야 한다. |
| `duration_budget_s` | 예상 측정 시간이다. `sample_interval_s=auto` 계산과 buffer coverage 검증에 사용된다. |
| `sample_count` | DMM buffer에 설정할 sample 수이며 최대 50,000이다. |
| `sample_interval_s` | 숫자(초), `auto`, 또는 `MIN`. `auto`는 `max(20 us, duration_budget_s / sample_count)`로 계산된다. |
| `nplc` | 적분 시간 설정이다. 작을수록 빠르지만 noise가 증가할 수 있다. |
| `current_range_A` | DMM current range이다. `null` 또는 0 이하는 autorange로 해석한다. |
| `autozero`, `reset` | 측정 시작 시 DMM autozero와 reset 사용 여부이다. |
| `rails[].name` | DMM configuration에 등록된 측정 target 이름이다. 현재 기본값은 `DMM_GPIB3`이다. |
| `rails[].voltage_V` | 측정 전류를 `power_W = current_A * voltage_V`로 환산할 때 사용하는 실제 rail 전압이다. 전압을 설정하거나 변경하는 option은 아니다. |
| `metadata` | 결과의 `session.json`에 함께 기록할 사용자 metadata이다. |

여러 rail을 측정하려면 measurement server의 DMM configuration에 각 장비를
등록한 후 `rails`에 추가한다. 현재 제공되는 configuration에는 GPIB address 3의
DMM 한 대만 등록되어 있다.

### 결과 확인

결과는 measurement server에서 자동으로 master에 복사되고 무결성 검사를 거친다.

- chip test: `eval_dir/<model>_evl.linux/power/<session-id>/`
- dataset 평가: `eval_dir/<model>_evl.linux/power/<session-id>/`
- 모든 scope: 위 session directory 아래 `regions/<region-id>/`마다 결과가 있다.

각 측정 결과의 주요 파일은 다음과 같다.

| 파일 | 내용 |
|---|---|
| `summary.json` | 평균 전류, 평균 power, energy, sample 수 및 측정 상태 |
| `rails/DMM_GPIB3.npz` | plot에 사용하는 sample별 전류, power, timestamp, tag state와 ambiguity 여부 |
| `raw/DMM_GPIB3.csv` | DMM에서 받은 reading metadata 포함 원본 CSV |
| `raw/checksums.json` | raw file의 SHA-256과 parsing 검증 정보 |
| `power_trace.png` | runner가 NPZ에 저장된 전체 sample로 자동 생성한 전류/power/tag-state plot |
| `tags.jsonl` | tag/event 수신 timestamp와 tag state 정의 |
| `time_alignment.json` | board/server/DMM clock 정렬 및 sample timestamp uncertainty |
| `request.json`, `resolved_config.json`, `session.json` | 요청, 실제 적용 설정, commit 및 session metadata |
| `tvm_power_manifest.json` | TVM scope, loop policy와 해당 run의 region ID 목록 |

개별 `regions/<region-id>` directory에 대해 다음
명령을 사용할 수 있다.

```bash
python scripts/power_request.py validate-result <result-dir>
python scripts/power_request.py summarize <result-dir> --exclude-ambiguous
python scripts/power_request.py summarize <result-dir> --tag kernel_stage=invoke
python scripts/power_request.py plot <result-dir> \
  --rail DMM_GPIB3 --output power_timeline.png
```

tag 경계와 timestamp uncertainty가 겹치는 sample은
`tag_boundary_ambiguous=true`로 저장된다. 정확한 구간 통계가 필요하면
`--exclude-ambiguous`를 사용한다.

workload가 DMM reading-memory coverage보다 길면 결과 상태는 `truncated`가 된다.
runner는 이를 warning으로 표시하되 실행을 실패시키지 않는다. raw checksum과 NPZ
array 정합성 검증을 마친 뒤, 확보된 sample 전체(현재 장비에서는 최대 50,000개)를
사용해 `power_trace.png`를 생성한다. `partial`이나 artifact 무결성 오류는 계속
실패로 처리한다.

## 2. 핵심 구현 방식

전체 경로는 다음과 같다.

```text
master runner
  -> request JSON을 board로 전송
  -> board binary가 TCP로 measurement server에 명령
  -> measurement server가 PyVISA로 DMM 제어
  -> measurement server가 결과 생성
  -> master runner가 SCP로 결과 회수 및 검증
```

### Master runner

[power_steps.sh](../power_steps.sh)는 `run_chiptest.sh`와
`run_dataset_eval.sh`가 공통으로 사용하는 orchestration code이다.

1. config와 TCP endpoint를 검사한다.
2. 세 시스템의 repository revision과 generated binary의 build identity를
   검사한다.
3. measurement server protocol/revision을 확인하고 필요하면 server를 자동
   실행한다.
4. [power_request.py](../scripts/power_request.py)로 session request를 만들고
   board에 복사한다.
5. board 실행 환경에 `IMCFLOW_POWER_REQUEST`, `POWER_MEASUREMENT_HOST`,
   `POWER_MEASUREMENT_PORT`, `IMCFLOW_POWER_SCOPE`와 minimum loop policy를 전달한다.
6. 실행 후 measurement server의 결과 directory를 SCP로 가져오고 모든 필수
   artifact와 raw checksum을 검사한다.

### Board runtime과 generated code

[power_measure_runtime.c](../power_runtime/power_measure_runtime.c)는 host binary의
power 측정 lifecycle을 관리한다. host 실행 code는 시작과 종료에 각각
`power_measure_runtime_start()`와 `power_measure_runtime_finish()`를 호출하며,
dataset loop에서는 `phase`, `sample`, timeout event tag도 기록한다.

TVM은 `MODEL`, `REGION`, `TILE` 중 선택된 위치에만 macro pair를 활성화한다.
measurement_utils에는 scope를 전달하지 않으며, 모든 활성 macro pair는 동일한
non-nested `power_region_begin/end`로 보인다. loop가 꺼져 있으면 body를 한 번,
켜져 있으면 minimum sample/time 조건을 만족할 때까지 같은 DMM trace 안에서
반복한다.

[ext_codegen.py](../../../../python/tvm/relay/backend/contrib/imcflow/ext_codegen.py)는
Linux kernel code에 region과 비동기 tag를 삽입한다. region은 per-kernel warmup
뒤에 시작하고 transfer, policy update, invoke, tile 실행을 포함한 뒤 cleanup 전에
종료된다. region 내부에는 다음 tag가 들어간다.

- `kernel`: generated kernel 이름
- `kernel_stage`: `compiled_transfer`, `const_transfer`, `policy_update`, `invoke`,
  `input_transfer`, `output_transfer`, retry cleanup 등
- `tile`: tile index
- `retry_attempt`: retry 번호
- `retry`, `region_end`: 순간 event

[power_region.c](../../../../3rdparty/measurement_utils/capi/power_region.c)는 일반 C
application에서도 사용할 수 있는 `POWER_REGION_BEGIN/END`와 `power_tag_*` API,
non-nesting 검사 및 minimum loop state machine을 제공한다.
[dmm_measure.c](../../../../3rdparty/measurement_utils/capi/dmm_measure.c)는 board에서
동작하는 C TCP client이다. board는 이 code를 통해 measurement server에 protocol
handshake, session/region 시작, tag 변경, 종료 명령을 보낸다. DMM 제어 library나
PyVISA는 board에 필요하지 않다.

### Measurement server와 timestamp 정렬

[power_tagged_measurement_server.py](../../../../3rdparty/measurement_utils/ps_ctrl/cli/power_tagged_measurement_server.py)는
measurement server에서 동작한다. TCP 요청을 받아 DMM을 독점 예약하고 PyVISA로
설정한 뒤 각 power region마다 GPIB GET으로 측정을 시작한다. progress 요청에는
`DATA:POIN?`의 non-destructive live count와 GET 이후 server monotonic elapsed를
반환하며, region END에서는 DMM acquisition을 즉시 freeze한다.

server는 DMM reading metadata의 첫 sample 시각과 sample interval, DMM clock
calibration, board/server clock anchor를 함께 사용해 각 sample을 server monotonic
time에 정렬한다. 비동기 tag는 server가 packet을 받은 monotonic timestamp로
기록되며, 각 sample의 timestamp/uncertainty와 비교해 tag state를 할당한다. 경계가
불확실한 sample은 제거하지 않고 ambiguity flag와 함께 raw/NPZ 결과에 보존한다.
