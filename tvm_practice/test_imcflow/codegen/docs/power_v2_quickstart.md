# Power v2 legacy measurement quickstart

`power_v2`는 tag나 clock synchronization 없이 measurement_utils의 기존
`START -> STARTED -> GO -> RESULT -> CLOSE` TCP protocol을 사용한다. power 측정은
기본적으로 꺼져 있으며, board에는 socket C client만 들어간다. PyVISA와 DMM
제어는 meas-2의 direct bridge process 하나에서 실행된다. 별도 RPyC server는
사용하지 않는다.

## 1. Revision 준비

master의 TVM과 meas-2의 measurement_utils가 모두 `power_v2`인지 확인한다.

```bash
# master
cd /root/project/tvm
git switch power_v2
git submodule update --init 3rdparty/measurement_utils

# measurement server
ssh meas-2
cd /home/jaeyongjang/project.local/measurement_utils
git switch power_v2
git pull --ff-only
conda activate imcflow
```

TVM은 submodule gitlink로 정확한 measurement_utils commit을 고정한다.

## 2. meas-2 server 실행

사용하는 설정은 다음 파일이다.

```text
example/configs/dmm_gpib3_config.json
```

이 설정의 logical name `DMM_GPIB3`은 `GPIB1::3::INSTR`에 대응한다. direct bridge
daemon 하나가 TCP 요청 수신과 PyVISA DMM 제어를 모두 담당한다. 아래 9910은 다른
사용자의 9900 server와 충돌하지 않도록 선택한 예시 port다.

```bash
# meas-2
conda activate imcflow
cd /home/jaeyongjang/project.local/measurement_utils
measure-bridge-daemon --host 0.0.0.0 --port 9910 \
  --config "$PWD/example/configs/dmm_gpib3_config.json" \
  --log-file /tmp/power_v2_bridge.log --log-level INFO
```

board가 접근할 `DMM_BRIDGE_HOST`에는 meas-2의 board-facing IP 또는 board에서
resolve할 수 있는 host 이름을 사용한다.

## 3. standalone protocol 확인

TVM을 실행하기 전에 board에서 작은 C test로 DMM path를 확인할 수 있다.

```bash
# master에서 cross compile
cd /root/project/tvm
aarch64-linux-gnu-gcc -std=c11 -static \
  -I 3rdparty/measurement_utils/capi \
  -o /tmp/test_dmm_now_single.aarch64 \
  3rdparty/measurement_utils/tests/capi/test_dmm_now_single.c \
  3rdparty/measurement_utils/capi/dmm_measure.c

scp /tmp/test_dmm_now_single.aarch64 petalinux:/tmp/test_dmm_now_single
ssh petalinux
DMM_BRIDGE_HOST=<meas-2-ip> DMM_BRIDGE_PORT=9910 \
  /tmp/test_dmm_now_single
```

성공하면 board에는 `RESULT DMM_GPIB3 ... count=<nonzero>`가 출력되고 meas-2에는
`/tmp/power_v2_standalone.txt` raw current trace가 남는다.

## 4. ResNet compile과 실행

master에서는 `activate`로 TVM Python 환경을 먼저 활성화한다.

```bash
activate
cd /root/project/tvm/tvm_practice/test_imcflow/codegen
source imcflow-linux.sh

export IMCFLOW_MEASURE_POWER=1
export IMCFLOW_POWER_SCOPE=REGION
export IMCFLOW_POWER_MODE=now
export IMCFLOW_POWER_DMM_NAME=DMM_GPIB3
export IMCFLOW_POWER_SERVER_OUTPUT_PREFIX=/tmp/power_v2_resnet
export DMM_BRIDGE_HOST=<meas-2-ip>
export DMM_BRIDGE_PORT=9910

CKPT=chip3_run4_ft_e80_iter003 \
MODEL_PROFILE=resnet8 DATASET_NAME=cifar10 \
IMCFLOW_BUGFIX=off ACC_MASK=1 \
IMCFLOW_NO_PERKERNEL_WARMUP=0 IMCFLOW_MMIO_BARRIER=100 \
python3 -u main.py \
  --model resnet8_subset31_pretrained_orig \
  --acc-mask 1 --ref-models transformed --random-seed 42 \
  --dataset cifar10 --sample 0 --stop-at compile

(
  export DEBUG_EXE=0
  cd host_binary_make.dataset
  ./build.sh \
    ../eval_dir/resnet8_subset31_pretrained_orig_evl.linux.bugfixoff \
    arm 1
)

CKPT=chip3_run4_ft_e80_iter003 DATASET_NAME=cifar10 \
DEBUG_EXE=0 CONSOLE_LOG_LEVEL=INFO IMCFLOW_BUGFIX=off \
./run_dataset_eval.sh -s 1 -i 0 \
  -m resnet8_subset31_pretrained_orig_evl.linux.bugfixoff
```

`run_dataset_eval.sh`은 board process에 `DMM_BRIDGE_HOST`와
`DMM_BRIDGE_PORT`를 전달한다. 전체 evaluation과 board result 수집이 끝난 다음에만
meas-2 raw file을 master로 SCP한다.

### GPIB 1, 2, 4 동시 측정

meas-2에 TVM checkout을 만들 필요는 없다. master에서 아래 스크립트를 실행하면
TVM 소유 JSON을 meas-2의 `/tmp`로 복사한 뒤, `imcflow` conda 환경의 direct bridge를
9911 포트에 시작한다. 기존 9910 GPIB3 bridge는 건드리지 않는다.

```bash
./scripts/start_power_bridge_meas2.sh \
  --config power_config/dmm_gpib124.json --port 9911

export DMM_BRIDGE_HOST=<meas-2-board-facing-ip>
export DMM_BRIDGE_PORT=9911
export IMCFLOW_POWER_DMM_NAMES=DMM_GPIB1,DMM_GPIB2,DMM_GPIB4
```

이 목록은 compile-time 설정이다. 변경하면 TVM compile과 host binary build를 다시
실행해야 한다. run 결과에는 DMM별 raw `.txt`, tag sidecar, `plots/` PNG와
`power_metadata.json`이 저장된다.

## 5. 설정

| 환경변수 | 기본값 | 의미 |
|---|---:|---|
| `IMCFLOW_MEASURE_POWER` | `0` | power code 생성 활성화 |
| `IMCFLOW_POWER_SCOPE` | `REGION` | `MODEL`, `REGION`, `TILE` |
| `IMCFLOW_POWER_MODE` | `now` | `now` 또는 `wait` |
| `IMCFLOW_POWER_DMM_NAME` | `DMM_GPIB3` | meas-2 config의 logical DMM name |
| `IMCFLOW_POWER_DMM_NAMES` | unset | comma-separated ordered logical DMM list; 설정 시 singular를 대체 |
| `IMCFLOW_POWER_NPLC` | `0.001` | DMM integration time |
| `IMCFLOW_POWER_INTERVAL_S` | `-1` | sample interval; 음수는 `MIN` |
| `IMCFLOW_POWER_SAMPLE_COUNT` | `50000` | DMM acquisition buffer 목표 count |
| `IMCFLOW_POWER_CURRENT_RANGE` | `0.1` | current range(A); 0 이하는 auto |
| `IMCFLOW_POWER_RESET` | `1` | 각 START 전 DMM reset |
| `IMCFLOW_POWER_START_TIMEOUT_S` | `30` | `STARTED` timeout |
| `IMCFLOW_POWER_RESULT_TIMEOUT_S` | `300` | `RESULT` timeout |
| `IMCFLOW_POWER_SERVER_OUTPUT_PREFIX` | `/tmp/imcflow_power` | meas-2 raw file prefix |
| `DMM_BRIDGE_HOST` | `127.0.0.1` | board가 접속할 bridge 주소 |
| `DMM_BRIDGE_PORT` | `9900` | bridge TCP port |
| `DMM_MEASUREMENT_SSH_HOST` | `meas-2` | run 종료 뒤 raw file을 가져올 SSH alias |
| `DMM_BRIDGE_LOG_PATH` | `/tmp/power_v2_bridge.log` | run 구간만 복사할 direct bridge/DMM log |
| `IMCFLOW_POWER_RUN_ID` | 자동 생성 | 결과 디렉터리의 run ID |
| `IMCFLOW_POWER_LOCAL_RESULT_DIR` | `eval_dir/<model>/power/<run_id>` | 전체 local 결과 디렉터리 override |

`now` mode는 scope 끝에서 즉시 acquisition을 중단하고 현재까지의 sample을 읽는다.
`wait` mode는 scope 끝에서 설정된 `IMCFLOW_POWER_SAMPLE_COUNT`가 채워질 때까지
blocking한다.

Scope 경계는 다음과 같다.

| Scope | Begin | End |
|---|---|---|
| `MODEL` | 첫 IMCFlow region의 warmup 뒤 | 마지막 region output read 뒤 |
| `REGION` | 각 region warmup 뒤 | 해당 region output read 뒤 |
| `TILE` | tile input transfer 뒤, invoke 직전 | invoke 완료 뒤, output read 전 |

## 6. 결과 위치

- meas-2 raw current: `<IMCFLOW_POWER_SERVER_OUTPUT_PREFIX>_<scope-name>.txt`
- master 복사본: `eval_dir/<model>/power/<run_id>/` 또는
  `IMCFLOW_POWER_LOCAL_RESULT_DIR`
- board의 평균/count: dataset evaluation console log의 `[POWER]` line
- board accuracy 결과: 기존 `eval_results/dataset_results_*.txt`
- `measurement_bridge.log`: START/GO protocol, raw output filename, 실제 interval,
  GET/ABORt와 read count
- `build_metadata.json`: generated C에 내장된 scope/mode/DMM 요청 설정

raw file에는 DMM에서 읽은 current sample list가 저장된다. 전체 evaluation이 끝나고
SCP가 성공하면 runner가 다음 plot도 자동 생성한다.

- `power_trace.png`: run의 모든 legacy raw trace를 subplot으로 구성한 통합 plot
- `plots/<raw-file-stem>.png`: region/model/tile raw file별 개별 plot

legacy raw 형식에는 DMM reading timestamp가 없으므로 x축은 시간이 아니라 sample
index이다. 같은 raw file에 여러 측정 결과가 append되어 있으면 plot의 회색 세로선이
각 capture의 경계를 나타낸다. `eval_dir`을 같은 model로 다시 compile해도 기존
`power/` run 디렉터리는 보존된다. tag, event, clock offset은 생성하지 않는다.

모든 scope는 compile 시 정한 하나의 `IMCFLOW_POWER_INTERVAL_S`를 사용한다. 음수로
`MIN`을 요청하면 DMM이 결정한 실제 값은 `measurement_bridge.log`의
`current burst 시작 (interval=...s)`에서 확인한다. START log에는 raw output file과
요청 설정도 함께 기록되므로 바로 다음 actual-interval line과 대응할 수 있다.

`REGION`과 `TILE` scope는 raw filename으로 측정 대상을 구분할 수 있다. 반면 `MODEL`
scope의 `..._model.txt`는 첫 region부터 마지막 region까지 하나의 연속 acquisition이다.
legacy protocol에는 trace 도중 경계 marker가 없으므로 model trace 안에서 각 region의
정확한 sample 범위를 복원할 수 없다. region별 분석이 필요하면 `REGION` scope로 다시
측정한다. 하나의 연속 model trace에 region 경계까지 표시하려면 별도의 event/timestamp
protocol을 추가해야 하며 이는 legacy protocol의 범위 밖이다.

power를 완전히 끄려면 compile과 run 모두에서 다음을 사용한다.

```bash
export IMCFLOW_MEASURE_POWER=0
```

이 경우 generated C에는 DMM include/call이 들어가지 않고 measurement server가 없어도
기존 `chip_acc_measure` 경로로 실행된다.
