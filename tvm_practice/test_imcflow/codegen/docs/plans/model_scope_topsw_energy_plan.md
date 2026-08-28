# MODEL scope 기반 tile energy 및 TOPS/W 계산 계획

## 0. Context handoff와 현재 상태

이 절은 대화 context 없이도 작업을 재개하기 위한 기준점이다. 아래 commit hash는
계획 작성 시점의 snapshot이며 이후 정상적인 commit으로 바뀔 수 있다. 다만 기존
working tree 변경은 모두 관련 작업 또는 사용자 변경이므로 임의로 reset/checkout하지
않는다.

### 0.1 Repository snapshot

| Repository | Path | Branch | HEAD | 현재 상태 |
|---|---|---|---|---|
| TVM | `/root/project/tvm` | `chip_acc_measure` | `84061f373beb61375d85812e3e043b858d323cee` | measurement_utils submodule 및 power 문서/config 변경, 이 계획 파일이 untracked |
| measurement_utils | `/root/project/tvm/3rdparty/measurement_utils` | `main` | `3abb14b7e55ac441aad8d4ef7a17238b9ee0c0c2` | direct bridge/unbuffered raw 처리와 관련된 C/Python/test 변경이 미commit 상태 |
| IMCFLOW RTL | `/root/project/imcflow` | `param_update` | `0800bd96a1ae81fb3f8d06ad3944bdfb4c5ff080` | gem5 submodule 변경이 존재함 |

measurement_utils의 `capi/dmm_measure.c`, `ps_ctrl/cli/measure_bridge_daemon.py`,
`ps_ctrl/manager.py` 및 관련 test 변경은 보존한다. 이 계획을 구현하기 전에 세 repo의
`git status --short`, branch, HEAD를 다시 기록하고 결과 metadata에도 실제 revision을
남긴다.

### 0.2 현재 완료된 것과 아직 하지 않은 것

현재 완료된 변경은 다음 두 가지뿐이다.

- 이 계획서 작성
- `power_config/dmm_gpib124.json`의 logical rail 이름을 GPIB1/2/4 순서로
  `VDD`/`DDA`/`DDC`로 변경하고 JSON 문법을 검증함

두 변경 모두 현재 TVM working tree의 local 변경이며 아직 commit/push하지 않았다.

아래 항목은 **계획 작성 시점에는 구현 또는 실행되지 않았다.** 2026-08-27의 실제
수행 결과는 바로 다음 0.2.1 절을 기준으로 한다.

- `rtl_region_cycles.py`의 `fsdb_cli` 전환 및 tile별 두 anchor 추출
- RTL timing JSON과 codegen tile manifest 생성
- `analyze_model_tile_energy.py`, peak detector, fractional integration, plot
- voltage/clock/operation-count 분석 config 생성
- 변경된 VDD/DDA/DDC config를 meas-2에 복사하고 bridge를 재시작하는 작업
- VDD/DDA/DDC 이름으로 새 MODEL power trace를 측정하는 작업
- 새 RTL simulation/FSDB 생성 및 end-to-end TOPS/W 검증

### 0.2.1 2026-08-27 수행 결과

다음 구현과 검증을 완료했다.

- `rtl_region_cycles.py`를 `fsdb_cli` direct path로 전환하고 기존 region/poll 출력과
  tile timing schema를 구현함
- compile metadata에 tile manifest, codegen fingerprint, dataset/sample/BUGFIX와 세 repo
  revision/dirty 상태를 기록하도록 구현함
- MODEL raw 검증, rail별 독립 peak detector, manual override, fractional integration,
  gross/dynamic charge·energy, tile/region/model 합계, sensitivity, JSON/CSV/raw/plot 생성을
  `scripts/analyze_model_tile_energy.py`에 구현함
- `power_config/power_energy_b2.json`과 사용 문서, bridge logical-name 사전 검증을 추가함
- 새 unit test 9개와 관련 power test를 합쳐 15개 test가 통과함
- 기존 MODEL raw의 detector-only 검증에서 VDD는 6개 tile을 고를 충분한 direct
  candidate가 있었고 DDA/DDC는 각각 5개만 검출되어 예상대로 region 4 manual override가
  필요함을 확인함

기존 133 MB B1 FSDB에 대한 `fsdb_cli` integration에서는 기존 region aggregate와 동일한
`60849/47437/17970/4878` cycle을 재현했다. 그러나 tile timing 생성은 fail-closed 됐다.
Region 1의 top RUN pulse 5개는 준비용 2개와 tile 3개 구조인데, 16개
`u_imce_ctrl.compute_start`의 pulse가 준비 구간에 있고 실제 tile RUN 구간에는 없었다.
따라서 이 계획의 `run_state_start <= any_imcu_compute_start < run_state_end` invariant가
성립하지 않는다. B2 동일 binary의 새 FSDB에서도 먼저 이 신호 의미를 확인해야 하며,
동일하다면 DDA start anchor를 실제 tile마다 pulse하는 RTL 신호로 변경하거나 testbench에
그 anchor를 추가한 뒤 계획/schema를 갱신해야 한다.

현재 `.env`가 B2를 가리키는 상태에서 `ssh petalinux`의
`/sys/kernel/debug/clk/clk_summary`를 읽어 `pl0_ref=99,990,000 Hz`를 확인했고 분석
config에 기록했다. VDD/DDA/DDC voltage, operation count/MAC 규칙,
`all_chip_power` 여부는 아직 확인되지 않아 해당 config 값은 `null`로 유지했다. 따라서 새
bridge/capture와 Joule/TOPS/W 계산은 실행하지 않았다. 기존 FSDB는 B1 metadata이고 기존
MODEL power run은 B2이며 fingerprint도 없으므로 둘을 energy 계산에 결합하지 않았다.

`_run.sh`의 B2 예시는 `IMCFLOW_POWER_DMM_NAMES=VDD,DDA,DDC`로 갱신했다. 이 목록은
generated C에 들어가는 compile-time 설정이므로 이름이나 순서를 바꾸면 TVM compile과
ARM host binary build를 다시 해야 한다.

### 0.2.2 2026-08-28 RTL anchor 및 사용자 입력 반영

0.2.1의 `compute_start` blocker를 RTL과 기존 FSDB로 재분석했다. IMCU 입력은
`u_imce_datapath/bshr` ready-valid interface를 거쳐 `u_imcu_core.core_rx`로 전달되므로
16개 IMCE의 첫 `bshr.valid && bshr.ready` handshake를 DDA anchor로 변경했다. 기존
FSDB에서 앞의 tile 5개에는 handshake가 있고 region 4에는 없음을 확인했다. Region의
마지막 manifest tile-count개 RUN을 실제 tile로 분류하므로 6개 tile
timing을 모두 생성하며 region 4 DDA anchor는 정상적으로 `null`이다.

- FSDB board는 검증/fingerprint에서 제외하여 동일 codegen의 B1/B2 공용 사용을 허용함
- `dmm_gpib124.json`에 VDD/DDA/DDC 전압 1.0/1.1/1.2 V를 기록함
- 원본 Relay의 conv/depthwise-conv MAC을 자동 계산하고 dense를 제외함
- `IMCFLOW_MAC_COUNTING=1|2`, 기본 2로 OP 환산함
- 세 rail 합계를 total-chip energy로 명명함
- rail peak 또는 DDA RTL handshake가 없으면 `missing_peak_zero`로 0 J 처리함
- 실제 ResNet8 FSDB에서 6개 tile, 5개 IMCU handshake, region 4 null을 검증함

### 0.3 세 장비와 실행 환경

| 역할 | 접근 방법 | 필수 환경/역할 |
|---|---|---|
| master | 현재 서버, `/root/project/tvm` | Python 작업 전 `activate`; 명시적 interpreter는 `/root/project/tvm/tvm_practice/tvm_env/bin/python3` |
| FPGA board | `ssh petalinux` | B2 Petalinux/ARM에서 generated host binary와 measurement C client 실행 |
| measurement server | `ssh meas-2` | `conda activate imcflow`; USB/GPIB DMM을 PyVISA direct bridge로 제어 |

master와 board에는 PyVISA가 필요 없다. Board의 C client가 meas-2의 direct bridge에
TCP로 접속하고, meas-2만 PyVISA로 DMM을 제어한다. 결과 수집은 각
`POWER_REGION_END()` 내부가 아니라 전체 dataset evaluation이 끝난 뒤 runner가
meas-2에서 master로 SCP한다. 따라서 SCP 시간은 측정 region에 포함되지 않는다.

Board 재부팅 뒤에는 `.env`가 B2를 가리키는지 확인하고, 필요하면 다음 warmup을 먼저
수행한다.

```bash
ssh petalinux 'cd /home/root/imcflow/xilinx/petalinux-csrc && make warmup'
```

Bridge는 master에서 다음과 같이 시작한다. 이 스크립트는 JSON을
`meas-2:/tmp/imcflow_power_config/`로 SCP하고 meas-2의 `imcflow` conda 환경에서
daemon을 시작한다. 기본 9911 port가 이미 사용 중이면 기존 process를 죽이지 않고
실패하므로 기존 daemon의 config를 확인하거나 별도 port를 사용한다.

```bash
cd /root/project/tvm/tvm_practice/test_imcflow/codegen
./scripts/start_power_bridge_meas2.sh \
  --config power_config/dmm_gpib124.json --port 9911 \
  --expected-dmm-names VDD,DDA,DDC

export DMM_BRIDGE_HOST=<meas-2의 board-facing IP>
export DMM_BRIDGE_PORT=9911
export DMM_MEASUREMENT_SSH_HOST=meas-2
export IMCFLOW_POWER_DMM_NAMES=VDD,DDA,DDC
```

기본 meas-2 log는 `/tmp/power_v2_bridge_9911.log`이고 PID file은
`/tmp/imcflow_power_bridge_9911.pid`이다. 새 logical name은 daemon을 위 config로
재시작한 뒤에만 유효하다. 기존 daemon은 여전히 `DMM_GPIB1/2/4` 이름을 가지고 있을
수 있으므로 run 전에 bridge log/config와 TCP port를 확인한다.

이 시작 script가 meas-2로 배포하는 것은 JSON뿐이다. measurement_utils Python/C
source나 conda package를 갱신하지 않는다. 이후 bridge protocol/code를 수정한다면
measurement_utils `main`의 정확한 commit을 meas-2 `imcflow` 환경에도 별도로 반영하고,
board에 링크되는 C client와 daemon의 protocol 호환성을 먼저 확인해야 한다.

### 0.4 기준 모델, binary 및 기존 power artifact

첫 end-to-end 대상은 다음과 같이 고정한다.

- model: `resnet8_subset31_pretrained_orig`
- board: B2
- checkpoint alias: `chip3_run4_ft_e80_iter003`
- dataset/sample: CIFAR-10, sample 0, random seed 42
- `IMCFLOW_BUGFIX=off`, `ACC_MASK=1`, `VMode.HALF`
- eval directory:
  `tvm_practice/test_imcflow/codegen/eval_dir/resnet8_subset31_pretrained_orig_evl.linux.bugfixoff`

재현 기준으로 사용할 기존 power run은 다음 세 개다.

```text
power/20260827T070729Z_resnet8_subset31_pretrained_orig_evl.linux.bugfixoff_84061f37_619508  # TILE
power/20260827T072453Z_resnet8_subset31_pretrained_orig_evl.linux.bugfixoff_84061f37_692723  # REGION
power/20260827T074310Z_resnet8_subset31_pretrained_orig_evl.linux.bugfixoff_84061f37_758944  # MODEL
```

MODEL run은 old logical name `DMM_GPIB1/2/4`, `now` mode, NPLC 0.001,
requested interval `MIN`, requested count 50000, current range 0.1 A를 사용했다. Sidecar에
기록된 실제 interval은 세 DMM 모두 20 us이고 실제 sample 수는 각각
47437/50000/50000이다. 이 interval은 해당 artifact에만 유효하며 새 run에서는 항상
sidecar 또는 `measurement_bridge.log`의 실제 값을 다시 읽는다.

MODEL sidecar의 software-tag trigger uncertainty는 127,958,209 ns, 즉 20 us 기준 약
6398 samples이다. 그러므로 기존 tag sample을 RUN 적분의 시작/끝으로 사용하지 않는다.
기존 결과 파일은 alias로 읽되 원본 파일명이나 raw 값은 수정하지 않는다.

현재 관찰된 구조는 다음과 같으며 detector의 회귀 기준이지 hard-coded 정답은 아니다.

- 전체 tile 수는 region별 `3 + 1 + 1 + 1 = 6`으로 예상한다.
- Region 1은 각 rail에서 짧은 두 pulse와 긴 plateau가 관찰되어 tile 3개와 대응한다.
- Region 2와 3은 각각 주요 상승 1개가 관찰된다.
- Region 4는 VDD에 작은 transient가 있으나 DDA/DDC peak가 약하거나 보이지 않는다.
- MODEL trace에는 region 1~3에 대응하는 강한 cluster 세 개가 보인다.
- VDD의 region 사이 staircase/ramp는 CPU↔IMCFLOW transfer/setup/reset/warmup 영향으로
  보이며 tile RUN energy에 포함하지 않는다.
- 기존 MODEL/REGION의 active-current median은 region 1~3에서 약 1.4% 이내로
  일치했지만, REGION/TILE `now` trace는 high 상태에서 끝나는 경우가 있어 trace 끝을
  RUN 종료로 간주할 수 없다.

각 DMM은 별도로 trigger되어 공통 sample-zero나 공통 sample index를 갖지 않는다.
따라서 rail마다 peak 시작을 독립적으로 검출해야 하며 한 rail에서 얻은 index/offset을
다른 rail에 자동 전파하지 않는다.

### 0.5 현재 code boundary와 RTL signal 출처

Power scope 생성 코드는
`python/tvm/relay/backend/contrib/imcflow/ext_codegen.py`에 있다. 현재 generated C에서
REGION scope는 compiled block/const transfer/policy update 이전에 시작하고 마지막
output read 뒤에 끝난다. MODEL scope는 첫 IMCFLOW region부터 마지막 region까지
연속 측정한다. 그러므로 CPU↔IMCFLOW 구간을 제거하려면 C의 region 경계를 바꾸는 것이
아니라 이 계획대로 MODEL raw에서 RTL RUN window만 선택해야 한다.

현재 invoke sequence는 PC/interrupt 설정, `STATE_REG_IDX=SET_RUN_CODE`, interrupt wait,
interrupt ACK, `INTR_DONE=1` 순서다. 다음 kernel 시작에는 reset pointer 설정과,
`IMCFLOW_NO_PERKERNEL_WARMUP=0`인 현재 기준에서 `make clear_time`/`make warmup`이 있다.
`POWER_REGION_END()`의 `now` 처리(`GO`, DMM `ABORt`, buffer read)는 DMM acquisition을
끝낼 뿐 IMCFLOW를 IDLE로 만드는 명령이 아니다.

RTL/FSDB의 기준 신호와 소스는 다음과 같다.

- 기존 분석 도구:
  `tvm_practice/test_imcflow/codegen/tools/rtl_region_cycles.py`
  (현재는 아직 `fsdb2vcd`를 사용함)
- 사용할 FSDB API: `/root/project/imcflow/tools/fsdb_cli`
- top busy signal: testbench의 `testbench_imcflow_gem5.imcflow_state_o`
- busy 생성: `pmap/modules/top/source/imcflow_impl.sv`의
  `imcflow_state_o = ~(top_ctrl_imcflow_state == IMCFLOW_S_IDLE)`
- IMCU compute 시작: `pmap/modules/imce/source/imce_fsm.sv`의
  `compute_start = (recv_if.data.cmd == CMD_COMPUTE) && recv_hs`

16개 IMCU의 exact FSDB hierarchy는 아직 확정하지 않았다. 구현 시
`fsdb_cli.find_signals()`로 발견하고 정확히 16개인지 검증한 뒤 resolved path를 timing
JSON에 저장한다. Verdi의 `fsdbdebug/fsdbextract/fsdbreport`가 PATH에 있어야 한다.

### 0.6 아직 필요한 외부 입력과 명시적 가정

다음 값은 현재 확인되지 않았으므로 추정하거나 100 MHz/임의 전압으로 채우지 않는다.

| 입력 | 필요한 이유 | 누락 시 동작 |
|---|---|---|
| B2 실제 IMCFLOW clock Hz | RTL cycle을 chip 실행 시간으로 변환 | `pl0_ref=99,990,000 Hz` readback 확인 |
| VDD/DDA/DDC 실제 voltage V | current를 energy로 변환 | 1.0/1.1/1.2 V를 DMM config에 기록 |
| ResNet operation count | TOPS/W 분자 | conv/depthwise MAC 12,500,992 자동 계산 |
| MAC을 1 또는 2 ops로 세는 규칙 | 결과 해석 일관성 | `IMCFLOW_MAC_COUNTING`, 기본 2 |
| 세 rail이 chip power 전체인지 여부 | total chip energy 명칭 결정 | 사용자 확인: 세 rail 합계를 total chip으로 사용 |

RTL timing은 동일 model, sample 0, checkpoint/codegen fingerprint, BUGFIX mode와
일치하는 simulation에서 생성한다. Cycle 수가 input-independent하다는 별도 증거가
없으면 다른 sample이나 binary의 timing JSON을 재사용하지 않는다.

Raw legacy 파일에는 DMM 자체 reading timestamp가 없으므로 각 trace 내부에서 sample이
`sample_interval_ns` 간격으로 균등하다고 가정한다. 이 가정과 rail별 독립 time origin을
결과 metadata에 기록한다. Peak detector의 smoothing window, MAD multiplier,
hysteresis threshold, minimum width, manual override도 숨은 상수로 두지 않고 CLI/config
값과 최종 사용값을 결과에 저장한다.

Peak 시작을 RTL anchor로 보는 것은 software-only 정렬의 v1 근사다. 실제 경계에는
DMM aperture/analog response, threshold 선택, 20 us sample 양자화, chip과 RTL의 실행
차이가 포함된다. 따라서 peak 위치 ±1 sample만이 아니라 threshold 변화와 검출 가능한
edge 폭에 따른 시간/energy 민감도를 함께 보고한다. 이 방식은 rail 사이의 wall-clock을
동기화하지 않으며, 서로 독립적인 trace에서 동일한 논리 tile의 energy를 각각 구해
합산한다.

### 0.7 재개 시 권장 순서

1. 세 repo의 branch/HEAD/dirty 상태를 기록하고 기존 변경을 보존한다.
2. B2 clock, 세 rail voltage, operation-count 규칙을 사용자에게 확인한다.
3. `fsdb_cli` adapter와 RTL tile timing JSON을 먼저 구현하고 synthetic test를 통과시킨다.
4. 같은 binary/config의 RTL FSDB에서 6개 tile과 두 anchor를 검증한다.
5. peak detector와 fractional integrator를 구현해 기존 MODEL artifact로 offline 검증한다.
6. meas-2 bridge를 새 VDD/DDA/DDC config로 시작한 뒤 MODEL `now` trace를 새로 얻는다.
7. 새 raw, RTL timing, 분석 config의 fingerprint를 맞춘 뒤 energy와 TOPS/W를 계산한다.

새 capture는 `eval_dir/<model>/power/<run_id>/`에 저장하며 기존 세 run directory를
삭제하거나 덮어쓰지 않는다. Implementation commit, bridge deployment, RTL run, chip
run은 각각 별도 상태로 기록해 어느 단계까지 완료됐는지 분명히 한다.

### 0.8 v1에서 하지 않는 것

- software tag나 board↔meas-2 clock sync를 적분 경계로 사용하지 않음
- DMM 하나의 sample index를 다른 DMM에 복사하거나 rail 간 공통 offset을 추정하지 않음
- DMM trace의 falling edge 또는 REGION/TILE trace 끝으로 RUN duration을 정하지 않음
- power region C boundary나 legacy direct-bridge protocol을 이번 분석을 위해 변경하지 않음
- FSDB를 VCD로 변환하거나 `fsdb2vcd` fallback을 제공하지 않음
- DDA 종료점을 별도 IMCU compute-end로 세분화하지 않음; v1은 RUN end를 공통 사용
- Region 4의 약한 peak를 다른 rail 정보만으로 자동 보간하지 않음

## 1. 목표

ResNet의 RTL simulation에서 얻은 tile별 IMCFLOW RUN cycle과 실제 chip의
MODEL-scope DMM trace를 결합하여 다음 값을 계산한다.

- tile별 RUN cycle 및 RUN time
- MODEL raw trace에서 tile별 RUN sample 구간
- DMM/rail별 tile energy
- 모든 측정 rail을 합한 tile/model total energy
- 이후 model operation count와 결합할 수 있는 TOPS/W 입력값

이 버전에서는 **각 rail의 MODEL trace에서 low-to-high peak 시작점을 독립적으로
검출**한다. DMM 간 trigger offset이 있으므로 한 DMM의 start sample을 다른 DMM에
복사하지 않는다. Software tag는 peak 식별을 보조할 수 있지만, 현재 약 128 ms의
trigger uncertainty가 있으므로 v1의 적분 경계로 사용하지 않는다.

Rail과 RTL 시작 anchor는 다음과 같이 고정한다.

| Logical rail | Physical DMM | RTL start anchor |
|---|---|---|
| `VDD` | DMM1, `GPIB1::1::INSTR` | `imcflow_state_o`가 RUN으로 바뀌는 시점 |
| `DDA` | DMM2, `GPIB1::2::INSTR` | 16개 IMCU 중 최초 input ready-valid handshake 시점 |
| `DDC` | DMM4, `GPIB1::4::INSTR` | `imcflow_state_o`가 RUN으로 바뀌는 시점 |

## 2. 측정 정의

### 2.1 RUN 시간

Tile마다 다음 두 시작점을 RTL에서 모두 저장한다.

1. `run_state_start`: top-level `imcflow_state_o`가 0에서 1로 바뀌는 cycle
2. `any_imcu_input_handshake`: 16개 IMCU 중 최초 `bshr.valid && bshr.ready` cycle

`run_state_end`는 `imcflow_state_o`가 다시 0으로 바뀌는 cycle로 정의한다. v1의
rail별 RTL interval은 다음과 같다.

- VDD/DDC: `[run_state_start, run_state_end)`
- DDA: `[any_imcu_input_handshake, run_state_end)`; handshake가 없으면 0 J

즉 DDA는 IMCU compute가 처음 시작되기 전의 RUN 준비 구간을 제외한다. 이후 IMCU의
마지막 compute 종료 신호까지 별도 추적할 필요가 생기면 DDA end anchor도 세분화한다.
CPU의 MMIO transfer, interrupt wait 자체, DMM/TCP 처리 시간은 포함하지 않는다.

cycle 수를 재사용 가능한 기준값으로 저장한다. 각 interval은 RTL 기준 시간과 실제
chip 기준 시간을 구분한다.

- `rtl_run_state_time_s = run_state_cycles / rtl_clock_hz`
- `rtl_imcu_to_end_time_s = imcu_to_run_end_cycles / rtl_clock_hz`
- `chip_run_state_time_s = run_state_cycles / chip_clock_hz`
- `chip_imcu_to_end_time_s = imcu_to_run_end_cycles / chip_clock_hz`

실제 energy 적분에는 rail anchor에 맞는 chip 기준 시간을 사용한다. RTL과 chip이
모두 100 MHz라는 사실이 확인된 경우에만 RTL 시간과 chip 시간이 동일하다.

### 2.2 Energy

DMM `d`의 tile `t`에 대한 gross energy는 다음과 같이 계산한다.

```text
E[d,t] = V[d] * integral(I[d,t], dt)
```

Sample 경계에 걸치는 RUN 시작/종료는 overlap 비율로 가중한다.

```text
E[d,t] = V[d] * sum(I[d,n] * overlap(sample_n, run_window_t))
```

모든 측정 rail의 total energy는 다음과 같다.

```text
E_tile[t]  = sum(E[d,t] for included DMM d)
E_model    = sum(E_tile[t])
```

기본 결과는 idle을 빼지 않은 gross energy로 한다. 분석용으로 각 peak 직전
baseline을 뺀 `dynamic_energy_j`도 함께 계산하지만 TOPS/W 기본값에는 사용하지
않는다.

`power_config/dmm_gpib124.json`의 rail entry에 분석용 `VOLTAGE_V`도 기록하고,
daemon은 알 수 없는 추가 field를 무시한다. 분석 설정은 이 DMM config를 참조한다.

```json
{
  "schema_version": 1,
  "chip_clock_hz": 99990000,
  "dmm_config": "dmm_gpib124.json",
  "missing_peak_policy": "zero_energy",
  "rails": {
    "VDD": {"voltage_v": null, "include_in_total": true, "rtl_anchor": "run_state"},
    "DDA": {"voltage_v": null, "include_in_total": true, "rtl_anchor": "any_imcu_input_handshake"},
    "DDC": {"voltage_v": null, "include_in_total": true, "rtl_anchor": "run_state"}
  }
}
```

DMM raw current는 이미 A 단위이므로 값을 그대로 사용한다.
`chip_clock_hz`, `voltage_v`가 비어 있거나 측정하지 않은 rail이 있으면 `total chip
energy`라고 표시하지 않고 조건에 따라 `measured-rail energy` 또는 `charge integral`만
출력한다. 확인되지 않은 값을 default로 추론하지 않는다.

새 측정의 logical DMM 이름은 `VDD,DDA,DDC`를 사용한다. 기존 MODEL raw의
`DMM_GPIB1,DMM_GPIB2,DMM_GPIB4`는 각각 `VDD,DDA,DDC` alias로 읽어 기존 결과도
분석할 수 있게 한다.

Measurement daemon은 수정된 `power_config/dmm_gpib124.json`으로 재시작하고, 새
power run에는 `IMCFLOW_POWER_DMM_NAMES=VDD,DDA,DDC`를 사용한다. Daemon config와
TVM 환경변수의 logical name이 다르면 run 전에 실패시킨다.

## 3. Phase 1: RTL tile RUN timing 생성

### 3.1 기존 도구 확장

`tools/rtl_region_cycles.py`는 이미 FSDB의 `imcflow_state_o`를 이용해 region별
busy cycle과 pulse 수를 계산한다. 이 코드를 공통 parser로 사용하고 다음 기능을
추가한다.

- `--granularity tile`
- `--output <rtl_tile_timing.json>`
- 각 tile의 `run_state_start`/`run_state_end` cycle 저장
- 각 tile의 optional `any_imcu_input_handshake` cycle 저장
- 두 start anchor 사이의 `imcu_input_delay_cycles` 저장
- region별 pulse를 tile index와 매핑
- 기존 region 합계 출력과 하위 호환 유지

FSDB 접근에는 `/root/project/imcflow/tools/fsdb_cli`의 Python API를 직접 사용한다.
기존 `rtl_region_cycles.py`의 `fsdb2vcd` 변환 경로는 제거한다. 분석 스크립트는
`IMCFLOW_DIR`이 설정되어 있으면 `$IMCFLOW_DIR/tools`를, 아니면
`/root/project/imcflow/tools`를 Python import path로 사용한다.

권장 호출 흐름은 다음과 같다.

```python
import fsdb_cli as fsdb

matches = fsdb.find_signals(fsdb_path, signal_pattern)
report = fsdb.report(fsdb_path, resolved_signal_paths)
events = report.events()
```

`find_signals()`로 실제 hierarchy의 signal path를 찾은 뒤 exact path를 확정하고,
`report().events()`의 carried-forward value-change event에서 rise/fall cycle을
계산한다. 전체 FSDB를 VCD로 변환하거나 임시 VCD 파일을 만들지 않는다. 동일 FSDB를
반복 분석할 때는 `fsdbreport` CSV 결과를 analysis directory에 cache할 수 있지만,
최종 JSON에는 원본 FSDB 경로/hash와 resolved signal path를 기록한다.

`fsdb_cli` 방식만 canonical timing으로 인정한다. `--method poll` 결과는 debugging
및 FSDB 결과와의 오차 비교용으로만 저장하며, `fsdb2vcd` fallback은 제공하지 않는다.

Top-level RUN anchor는 기존 `testbench_imcflow_gem5.imcflow_state_o`를 사용한다.
IMCU anchor는 각 IMCE의 `u_imce_datapath/bshr` ready-valid interface를 사용한다.
이 interface는 linebuffer 출력이 `u_imcu_core.core_rx`로 실제 수락되는 경계다.
`fsdb_cli.report()`에서 16쌍의 `valid && ready`가 최초로 참인 cycle을
`any_imcu_input_handshake`로 기록한다. 함께 시작한 IMCU가 여러 개면 최초 cycle과
해당 좌표 목록도 저장한다.

Signal discovery 결과는 다음 조건을 만족해야 한다.

- RUN signal은 exact match 1개
- IMCU input valid/ready signal은 각각 exact match 16개
- 모든 signal의 time unit이 동일함
- unknown/X/Z transition은 start event로 취급하지 않음

개수가 맞지 않거나 hierarchy가 모호하면 분석을 중단하고 발견한 후보 path를 출력한다.

### 3.2 Tile 식별

busy pulse를 무조건 tile로 간주하지 않는다. Policy update 및 tile 이전의 준비용
RUN이 별도 pulse를 만들 수 있기 때문이다.

다음 정보를 함께 사용해 실제 tile RUN pulse를 분류한다.

1. codegen의 region function 순서와 `tiling_factor`
2. `gem5_output.log`의 region/tile 실행 메시지
3. `vcs_sim.log`의 `SET_RUN_CODE` MMIO transaction
4. FSDB의 `imcflow_state_o` rise/fall interval
5. 각 region에서 setup/policy 뒤에 위치하는 manifest 개수만큼의 마지막 RUN interval

필요하면 compile 시 다음과 같은 작은 tile manifest를 생성한다.

```json
{
  "regions": [
    {"region_index": 1, "function": "...region1...", "tile_count": 3},
    {"region_index": 2, "function": "...region2...", "tile_count": 1}
  ]
}
```

region RUN 수가 manifest의 `tile_count`보다 작으면 실패한다. 마지막 N개 RUN을 tile로,
그 앞의 RUN을 setup/policy로 기록한다.

### 3.3 RTL timing JSON

기본 저장 위치는 RTL eval directory의 다음 파일로 한다.

```text
eval_dir/<model>.baremetal/analysis/rtl_tile_timing.json
```

Schema 예시는 다음과 같다.

```json
{
  "schema_version": 2,
  "model": "resnet8_subset31_pretrained_orig",
  "rtl_method": "fsdb_cli",
  "fsdb_cli_root": "/root/project/imcflow/tools/fsdb_cli",
  "fsdb_cli_revision": "<imcflow-tools-commit>",
  "busy_signal": "testbench_imcflow_gem5.imcflow_state_o",
  "imcu_input_valid_signals": ["<16 resolved hierarchy paths>"],
  "imcu_input_ready_signals": ["<16 resolved hierarchy paths>"],
  "rtl_clock_hz": 100000000,
  "rtl_revision": "<commit-or-build-manifest-hash>",
  "tvm_revision": "<tvm-commit>",
  "measurement_utils_revision": "<measurement-utils-commit>",
  "board": "B2",
  "checkpoint_alias": "chip3_run4_ft_e80_iter003",
  "dataset": "cifar10",
  "sample_index": 0,
  "random_seed": 42,
  "imcflow_bugfix": false,
  "codegen_fingerprint": "<model/codegen hash>",
  "regions": [
    {
      "region_index": 1,
      "function": "...region1...",
      "tiles": [
        {
          "tile_index": 0,
          "run_state_start_cycle": 0,
          "any_imcu_input_handshake_cycle": 0,
          "first_imcu_coordinates": [[0, 0]],
          "run_state_end_cycle": 0,
          "imcu_input_delay_cycles": 0,
          "run_state_cycles": 0,
          "imcu_to_run_end_cycles": 0,
          "run_state_time_s": 0.0,
          "imcu_to_run_end_time_s": 0.0
        }
      ]
    }
  ],
  "total_run_state_cycles": 0,
  "total_run_state_time_s": 0.0
}
```

위 `rtl_clock_hz: 100000000`은 현재 testbench clock에서 실제로 확인했을 때의 예시이며
B2 chip clock 값이 아니다. Parser는 FSDB time unit과 testbench clock period로 이 값을
검증해 저장한다. Chip energy 분석 config의 `chip_clock_hz`는 별도로 확인해야 한다.

RTL revision, BUGFIX mode, codegen fingerprint가 현재 chip binary와 맞지 않으면 timing
JSON 재사용을 거부한다.

각 tile에 대해 다음 invariant도 검사한다.

```text
run_state_start <= any_imcu_input_handshake < run_state_end
```

Handshake가 없는 depthwise tile은 anchor를 `null`로 저장하고 DDA energy를 0 J로 한다.

## 4. Phase 2: MODEL raw에서 tile peak 시작점 검출

새 분석 도구를 추가한다.

```text
scripts/analyze_model_tile_energy.py
```

입력은 다음과 같다.

```bash
python scripts/analyze_model_tile_energy.py \
  <MODEL_POWER_RUN_DIR> \
  --rtl-timing <RTL_EVAL_DIR>/analysis/rtl_tile_timing.json \
  --analysis-config power_config/power_energy_b2.json
```

### 4.1 Raw 및 metadata 검증

- `build_metadata.json`의 scope가 `MODEL`인지 확인
- DMM logical name과 raw 파일을 `power_metadata.json`으로 매핑
- raw capture가 DMM별 정확히 하나인지 확인
- tag sidecar의 `sample_interval_ns`와 실제 sample 수 확인
- NaN/Inf, 빈 trace, overwrite 실패, 중복 capture 검사
- RTL timing의 model/codegen fingerprint 확인

각 DMM은 trigger 시작 offset과 actual sample 수가 다를 수 있으므로 **동일 sample
index를 DMM 사이의 동일 시각이라고 가정하지 않는다.** VDD, DDA, DDC trace에서
각 tile의 peak 시작점을 완전히 독립적으로 검출한다.

### 4.2 Peak 후보 검출

각 raw trace에 다음 절차를 적용한다.

1. 이동 median으로 짧은 DMM noise 제거
2. peak 이전 구간의 median/MAD로 local idle baseline 계산
3. high/low 두 threshold를 사용하는 hysteresis rising-edge 검출
4. 너무 짧은 noise spike는 별도 candidate로 보존하되 우선순위를 낮춤
5. RTL JSON의 region/tile 순서 및 예상 tile 수만큼 candidate를 순서대로 매핑
6. rail별 candidate를 해당 RTL anchor와 연결

Rail별 연결 규칙은 다음과 같다.

```text
VDD peak start -> RTL run_state_start
DDA peak start -> RTL any_imcu_input_handshake
DDC peak start -> RTL run_state_start
```

ResNet의 현재 관찰 결과를 초기 회귀 기준으로 사용한다.

- Region 1: 3 tiles, 각 DMM에서 2개의 짧은 pulse와 1개의 plateau
- Region 2: 1 tile, 주요 상승 1개
- Region 3: 1 tile, 주요 상승 1개
- Region 4: 1 tile이나 GPIB2/GPIB4의 상승이 매우 작거나 보이지 않음

완전한 peak sequence를 가진 rail을 tile 순서 기준으로 사용하고 rail-local time offset을
맞춘다. 대응 peak 자체가 없는 rail/tile에는 다른 rail의 current sample을 대신 사용하지
않고 charge/energy를 0으로 기록한다. `--peak-overrides`는 선택적으로 유지한다.

다음 경우에는 분석을 실패시킨다.

- tile 기준으로 쓸 완전한 rail이 하나도 없음
- 두 tile window가 겹침
- 적분 window가 raw 범위를 벗어남
- DMM 간 peak 순서 불일치
- rail별 peak의 시간 순서가 RTL tile 순서와 맞지 않음

수동 검증을 위해 `--peak-overrides <json>`도 지원한다. Override에는 rail, region,
tile별 start sample을 명시하며 결과 metadata에 수동 입력임을 기록한다.

## 5. Phase 3: RTL duration을 이용한 sample 선택 및 적분

각 rail/tile에서 해당 rail의 peak와 RTL anchor를 사용해 다음 값을 계산한다.

```text
start_time = detected_start_sample * sample_interval
VDD/DDC duration = (run_state_end_cycle - run_state_start_cycle) / chip_clock_hz
DDA duration     = (run_state_end_cycle - any_imcu_input_handshake_cycle) / chip_clock_hz
end_time   = start_time + duration
```

따라서 세 rail의 start sample은 서로 달라도 되고, DDA의 적분 길이도 VDD/DDC보다
짧을 수 있다. 모든 rail의 RTL 종료 기준은 v1에서 `run_state_end`로 통일한다.

`duration / sample_interval`이 정수가 아닐 수 있으므로 단순히 sample 수를 반올림하지
않고 첫/마지막 sample의 시간 overlap을 이용해 fractional integration한다. RUN 시간이
20 us sample interval보다 짧은 경우에도 한 sample 전체를 사용하지 않고 실제 overlap
비율만 적용한다.

Tile별로 다음 artifact를 MODEL power run directory에 생성한다.

```text
tile_energy.json
tile_energy.csv
run_only_power_trace.png
run_only_samples/
  region01_tile00_VDD.json
  region01_tile00_DDA.json
  region01_tile00_DDC.json
  ...
```

`run_only_samples`에는 원본 sample index, current, sample overlap time을 저장하여 적분을
재현할 수 있게 한다. 원본 raw 파일은 변경하지 않는다.

`tile_energy.json`에는 다음 정보를 포함한다.

- 두 RTL start anchor, RUN end, 적용한 chip clock
- rail별 선택한 RTL anchor
- rail별 독립 start/end sample 및 검출 방식(direct/manual)
- 사용한 sample interval과 fractional boundary weight
- 원본 raw/FSDB hash와 TVM, measurement_utils, IMCFLOW/fsdb_cli revision
- model, checkpoint, dataset/sample, random seed, BUGFIX 및 board-independent codegen fingerprint
- peak detector의 모든 threshold/window와 raw time-axis 가정
- baseline current
- charge integral(C)
- gross/dynamic energy(J)
- rail 합계 및 model 합계
- peak-start ±1 sample 및 threshold variation에 따른 energy 민감도
- warning/incomplete 상태

Plot에는 전체 MODEL raw 위에 다음을 표시한다.

- rail별로 독립 검출한 tile 시작 rising edge
- rail에 대응하는 RTL interval로 결정된 tile 종료 지점
- 실제 적분에 포함된 구간의 음영
- region/tile label
- 검출 실패 또는 offset 추정 구간의 다른 색상

## 6. TOPS/W 연결

컴파일 때 원본 Relay의 conv/depthwise-conv MAC 수를 metadata에 저장하며 dense는
제외한다. `IMCFLOW_MAC_COUNTING=1|2`로 MAC당 OP 수를 정하고 기본값은 2이다.

```text
TOPS/W = total_operations / total_energy_j / 1e12
```

이는 동일 inference에 대해 `TOPS / average_power`를 계산한 값과 같다. 전체 rail을
측정하지 않았으면 결과 이름을 `measured_rails_tops_per_w`로 제한한다.

## 7. 테스트 계획

### 7.1 Unit test

- synthetic `fsdb_cli.Event` sequence의 known RUN pulses에서 tile별 cycle 추출
- 16개 IMCU input ready-valid pair 중 최초 handshake와 동시 좌표 추출
- optional `run_state_start <= any_imcu_input_handshake < run_state_end` 검증
- `find_signals()`가 RUN 1개/IMCU 16개를 찾지 못할 때 fail-closed 동작
- 작은 known FSDB fixture에 대한 `fsdb_cli.report().events()` integration test
- policy/setup pulse와 tile pulse 분류
- RTL tile count mismatch 검출
- noise, spike, plateau가 있는 synthetic current trace의 rising-edge 검출
- VDD/DDA/DDC별 독립 rising-edge 검출
- missing peak=0 J와 manual override 처리
- constant current/voltage에 대한 fractional integration 정답 비교
- VDD/DDC RUN interval과 DDA IMCU-to-RUN-end interval 선택 검증
- idle-subtracted energy와 gross energy 계산
- voltage/clock 누락 시 fail-closed 동작

### 7.2 ResNet end-to-end 검증

1. RTL runner로 ResNet sample 0을 실행한다.
2. `fsdb_cli`로 FSDB를 직접 읽어 tile timing JSON을 생성한다.
3. 기존 MODEL B2/GPIB1·2·4 raw에 분석기를 실행한다.
4. RTL tile 수와 검출 peak 수를 비교한다.
5. `run_only_power_trace.png`를 기존 MODEL/REGION/TILE plot과 시각 비교한다.
6. tile energy 합과 model energy 합이 동일한지 확인한다.
7. threshold와 peak start를 ±1 sample 변화시켜 energy 민감도를 기록한다.

Region 4처럼 일부 rail에서 peak가 보이지 않으면 자동 성공으로 처리하지 않는다.
수동 start sample을 적용한 결과임을 명확하게 표시한다.

## 8. 구현 순서

1. `rtl_region_cycles.py`의 `fsdb2vcd` 경로를 `fsdb_cli` adapter로 교체
2. 두 RTL anchor의 tile JSON 출력과 signal discovery 검증 추가
3. codegen tile manifest 및 fingerprint 추가
4. RTL synthetic/unit test 추가
5. power analysis config와 voltage/clock validation 추가
6. VDD/DDA/DDC 독립 MODEL peak detector 및 overlay plot 구현
7. fractional sample extractor와 energy integrator 구현
8. JSON/CSV/run-only raw artifact 생성
9. 기존 ResNet RTL 및 MODEL raw로 end-to-end 검증
10. operation count metadata를 연결하여 TOPS/W 출력 추가

## 9. 완료 기준

- 모든 RTL tile의 RUN/최초 IMCU compute start와 RUN end가 JSON으로 저장됨
- VDD/DDA/DDC 각각의 검출 event 수가 RTL tile 수와 일치함
- 각 energy 값이 어떤 raw sample과 boundary weight로 계산됐는지 재현 가능함
- CPU↔IMCFLOW transfer 구간이 적분 window에서 제외됨
- voltage, chip clock, RTL/codegen revision이 검증되지 않으면 TOPS/W를 출력하지 않음
- ResNet 결과에 tile별/region별/model total energy와 uncertainty가 함께 저장됨
