# MODEL scope 에너지 및 TOPS/W 실행 가이드

이 문서는 ResNet8 sample 0을 기준으로 다음 순서를 처음부터 실행하는 방법을 설명한다.

1. Linux/ARM chip binary를 MODEL power 설정으로 컴파일한다.
2. B1/B2 공용 FSDB에서 tile별 RTL 시간을 추출한다.
3. B2에서 VDD/DDA/DDC MODEL current trace를 측정한다.
4. 각 tile과 전체 inference의 energy 및 TOPS/W를 계산한다.

분석기는 power capture와 RTL timing의 model, checkpoint, dataset/sample, seed, BUGFIX,
codegen fingerprint가 다르면 결과를 만들지 않는다. FSDB의 board metadata만은 비교에서
제외하므로 동일 codegen의 B1/B2 FSDB를 공용으로 사용할 수 있다.

## 1. 작업 디렉터리와 공통 변수

master 서버에서 실행한다.

```bash
cd /root/project/tvm/tvm_practice/test_imcflow/codegen
source /root/project/tvm/tvm_practice/tvm_env/bin/activate

CODEGEN_DIR=/root/project/tvm/tvm_practice/test_imcflow/codegen
MODEL=resnet8_subset31_pretrained_orig
LINUX_EVAL="$CODEGEN_DIR/eval_dir/${MODEL}_evl.linux.bugfixoff"
RTL_EVAL="$CODEGEN_DIR/eval_dir/${MODEL}_evl.baremetal.bugfixoff"
ENERGY_CONFIG="$CODEGEN_DIR/power_config/power_energy_b2.json"
```

`.env`가 B2 board 주소를 가리키는지 확인하고, board가 재부팅됐다면 기존 절차대로
warmup을 먼저 수행한다.

```bash
ssh petalinux 'cd /home/root/imcflow/xilinx/petalinux-csrc && make warmup'
```

## 2. 전압과 분석 설정 확인

전압은 DMM 설정 파일의 다음 항목에서 바꿀 수 있다.

```json
"POWER": {
  "VDD": { "DEVICE": "dmm_gpib1", "VOLTAGE_V": 1.0 },
  "DDA": { "DEVICE": "dmm_gpib2", "VOLTAGE_V": 1.1 },
  "DDC": { "DEVICE": "dmm_gpib4", "VOLTAGE_V": 1.2 }
}
```

파일은 `power_config/dmm_gpib124.json`이다. `power_energy_b2.json`은 이 파일을
`dmm_config`로 참조하며 현재 주요 설정은 다음과 같다.

- chip clock: `99,990,000 Hz`
- missing peak: 해당 rail/tile energy를 `0 J`로 처리
- VDD/DDA/DDC 합계: 전체 chip energy
- DDA RTL anchor: 최초 IMCU input `valid && ready` handshake

## 3. MODEL power가 포함된 Linux binary 컴파일

다음 power 환경변수는 generated C에 들어가므로 compile 때 설정해야 한다. DMM 이름과
순서를 바꾼 뒤에는 반드시 다시 컴파일하고 ARM host binary도 다시 빌드한다.

```bash
source imcflow-linux.sh

IMCFLOW_MEASURE_POWER=1 \
IMCFLOW_POWER_SCOPE=MODEL \
IMCFLOW_POWER_MODE=wait \
IMCFLOW_POWER_DMM_NAMES=VDD,DDA,DDC \
IMCFLOW_POWER_SAMPLE_COUNT=50000 \
IMCFLOW_POWER_SERVER_OUTPUT_PREFIX=/tmp/power_b2_gpib124_resnet \
DMM_BRIDGE_HOST=147.46.117.49 DMM_BRIDGE_PORT=9911 \
CKPT=chip3_run4_ft_e80_iter003 BOARD=B2 \
MODEL_PROFILE=resnet8 DATASET_NAME=cifar10 \
IMCFLOW_BUGFIX=off ACC_MASK=1 \
IMCFLOW_NO_PERKERNEL_WARMUP=0 \
IMCFLOW_MMIO_BARRIER=100 \
python3 -u main.py \
  --model resnet8_subset31_pretrained_orig \
  --acc-mask 1 \
  --ref-models transformed \
  --random-seed 42 \
  --dataset cifar10 \
  --sample 0 \
  --stop-at compile
```

컴파일이 끝나면 다음 두 파일이 있어야 한다.

```bash
ls -lh "$LINUX_EVAL/build_metadata.json" "$LINUX_EVAL/tile_manifest.json"
```

이어서 ARM host binary를 빌드한다.

```bash
(
  export DEBUG_EXE=0
  cd host_binary_make.dataset
  ./build.sh ../eval_dir/resnet8_subset31_pretrained_orig_evl.linux.bugfixoff arm 1
)
```

## 4. RTL tile timing 생성

### 4.1 새 FSDB를 생성하는 경우

Linux compile과 model/checkpoint/sample/seed/BUGFIX/acc-mask를 같게 유지한다. RTL license
환경도 필요하다.

```bash
source imcflow-baremetal.sh

IMCFLOW_RUNNER=rtl \
IMCFLOW_DIR=/root/project/imcflow \
SNPSLMD_LICENSE_FILE=1727@147.46.168.128 \
CKPT=chip3_run4_ft_e80_iter003 BOARD=B2 \
MODEL_PROFILE=resnet8 DATASET_NAME=cifar10 \
IMCFLOW_BUGFIX=off ACC_MASK=1 \
IMCFLOW_NO_PERKERNEL_WARMUP=0 \
IMCFLOW_MMIO_BARRIER=100 \
python3 -u main.py \
  --model resnet8_subset31_pretrained_orig \
  --acc-mask 1 \
  --ref-models transformed \
  --random-seed 42 \
  --dataset cifar10 \
  --sample 0 \
  --stop-at simulate
```

### 4.2 기존 FSDB를 재사용하는 경우

기존 FSDB는 board와 무관하게 사용할 수 있다. 다만 함께 전달할 `tile_manifest.json`은
현재 chip binary와 codegen이 같아야 한다. 기존 RTL eval에 current manifest가 없다면
같은 옵션으로 baremetal compile을 실행해 manifest를 만든 뒤 기존 FSDB와 조합한다.
tile 수를 눈으로 보고 임의 manifest를 작성하지 않는다.

FSDB와 manifest를 확인한다.

```bash
ls -lh "$RTL_EVAL"/logs/rtl_runner/*.fsdb
ls -lh "$RTL_EVAL/tile_manifest.json"
```

### 4.3 timing JSON 추출

```bash
RTL_TIMING="$RTL_EVAL/analysis/rtl_tile_timing.json"

python3 tools/rtl_region_cycles.py "$RTL_EVAL" \
  --granularity tile \
  --manifest "$RTL_EVAL/tile_manifest.json" \
  --output "$RTL_TIMING"
```

추출기는 각 region의 마지막 manifest tile-count개 RUN을 tile로 사용한다. DDA 시작은
16개 IMCE 중 최초 `u_imce_datapath/bshr/valid && ready` cycle이다. IMCU input
handshake가 없는 tile은 anchor가 `null`이며 DDA energy는 이후 0 J가 된다.

결과의 tile 수와 anchor를 확인한다.

```bash
python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
tiles = [t for r in d["regions"] for t in r["tiles"]]
print("schema:", d["schema_version"])
print("FSDB board metadata:", d.get("board"))
print("tile count:", len(tiles))
print("IMCU input anchors:", [t["any_imcu_input_handshake_cycle"] for t in tiles])
' "$RTL_TIMING"
```

현재 회귀 FSDB에서는 tile 6개, handshake 5개, region 4 anchor `None`이 정상 결과다.

## 5. DMM bridge 시작

9911 port에 동일 config의 bridge가 이미 실행 중이면 이 단계는 건너뛴다. 시작 스크립트는
사용 중인 port의 process를 교체하지 않고 실패한다.

```bash
./scripts/start_power_bridge_meas2.sh \
  --config power_config/dmm_gpib124.json \
  --port 9911 \
  --expected-dmm-names VDD,DDA,DDC
```

다른 port를 선택했다면 compile과 evaluation의 `DMM_BRIDGE_PORT`도 같은 값으로 바꾼다.

## 6. B2 MODEL current capture

다음 명령은 sample 0 inference 한 번을 실행하고 VDD/DDA/DDC trace를 Linux eval의
`power/<run_id>/`에 수집한다.

```bash
source imcflow-linux.sh

IMCFLOW_MEASURE_POWER=1 \
DMM_BRIDGE_HOST=147.46.117.49 DMM_BRIDGE_PORT=9911 \
DMM_MEASUREMENT_SSH_HOST=meas-2 \
IMCFLOW_POWER_SERVER_OUTPUT_PREFIX=/tmp/power_b2_gpib124_resnet \
CKPT=chip3_run4_ft_e80_iter003 DATASET_NAME=cifar10 \
DEBUG_EXE=0 CONSOLE_LOG_LEVEL=INFO IMCFLOW_BUGFIX=off \
./run_dataset_eval.sh \
  -s 1 \
  -i 0 \
  -m resnet8_subset31_pretrained_orig_evl.linux.bugfixoff
```

생성된 directory를 확인하고 가장 최근 경로를 직접 `POWER_RUN`에 넣는다.

```bash
ls -dt "$LINUX_EVAL"/power/*
POWER_RUN="$LINUX_EVAL/power/<방금 생성된 run_id>"
```

다음 파일들이 있어야 한다.

```bash
ls -lh "$POWER_RUN/build_metadata.json" "$POWER_RUN/power_metadata.json"
ls -lh "$POWER_RUN"/*.txt "$POWER_RUN"/*.tags.json
```

과거 MODEL capture에 새 `codegen_fingerprint`가 없으면 strict 분석에는 사용할 수 없다.
그 경우 현재 코드로 compile과 capture를 다시 실행해야 한다. 이는 FSDB board 문제와는
별개이며, 측정 current가 정확히 어느 binary에서 나온 것인지 보장하기 위한 조건이다.

## 7. Peak detector만 먼저 점검

timing JSON에서 tile 수를 확인한 뒤 같은 숫자를 `--expected-tiles`에 넣는다. ResNet8
회귀 설정의 예시는 6이다.

```bash
python3 scripts/analyze_model_tile_energy.py "$POWER_RUN" \
  --analysis-config "$ENERGY_CONFIG" \
  --detect-only \
  --expected-tiles 6
```

`missing_peak_policy=zero_energy`에서는 VDD처럼 완전한 기준 rail 하나에 tile 수만큼 peak가
있으면 성공한다. DDA/DDC에 region 4 peak가 없어도 오류가 아니다.

## 8. Energy 및 TOPS/W 계산

MAC 1회를 1 OP로 계산하려면 다음과 같이 실행한다.

```bash
IMCFLOW_MAC_COUNTING=1 \
python3 scripts/analyze_model_tile_energy.py "$POWER_RUN" \
  --rtl-timing "$RTL_TIMING" \
  --analysis-config "$ENERGY_CONFIG"
```

MAC 1회를 multiply와 accumulate의 2 OP로 보고 싶으면 값만 2로 바꾼다.

```bash
IMCFLOW_MAC_COUNTING=2 \
python3 scripts/analyze_model_tile_energy.py "$POWER_RUN" \
  --rtl-timing "$RTL_TIMING" \
  --analysis-config "$ENERGY_CONFIG"
```

성공하면 `POWER_RUN` 아래에 다음 파일이 생성된다.

- `tile_energy.json`: 전체 metadata, tile/region/model 합계, sensitivity
- `tile_energy.csv`: rail/tile별 표 형식 결과
- `run_only_power_trace.png`: 검출 peak와 실제 적분 window
- `run_only_samples/*.json`: 적분에 사용한 원본 sample과 boundary weight

핵심 결과는 다음 명령으로 확인한다.

```bash
python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
print("status:", d["status"])
print("conv MACs:", d["totals"]["conv_mac_count"])
print("ops/MAC:", d["totals"]["mac_counting"])
print("total operations:", d["totals"]["operation_count"])
print("total chip energy (J):", d["totals"]["total_chip_energy_j"])
print("TOPS/W:", d["totals"]["tops_per_w"])
' "$POWER_RUN/tile_energy.json"
```

## 9. 계산 정의

각 rail의 current sample을 해당 RTL duration과 겹치는 시간만큼 fractional integration한다.

```text
rail charge  Q = Σ(current_sample × overlapped_sample_time)
rail energy  E = Q × rail_voltage
chip energy    = E_VDD + E_DDA + E_DDC
operations     = conv/depthwise_conv_MAC_count × IMCFLOW_MAC_COUNTING
TOPS/W         = operations / chip_energy_j / 1e12
```

기본 TOPS/W는 idle baseline을 빼지 않은 `gross energy`를 사용한다. 분석용으로 baseline을
뺀 `dynamic_energy_j`도 함께 출력하지만 TOPS/W 기본 분모로 사용하지 않는다. 모델 MAC
수는 원본 Relay의 conv와 depthwise conv에서 자동 계산하며 dense는 제외한다.

Peak가 검출되지 않거나 DDA RTL handshake가 없으면 해당 rail/tile은 다음처럼 기록된다.

```json
{
  "detection_method": "missing_peak_zero",
  "gross_charge_c": 0.0,
  "gross_energy_j": 0.0
}
```

## 10. 필요할 때만 manual peak override

기본 정책은 미검출 peak를 0 J로 처리한다. 실제 plot을 확인한 뒤 detector가 놓친 명확한
peak를 강제로 사용하고 싶을 때만 rail-local start sample을 지정한다.

```json
{
  "DDA": {"region04_tile00": 41234},
  "DDC": {"region04_tile00": 41190}
}
```

예를 들어 위 내용을 `overrides.json`으로 저장했다면 다음처럼 실행한다.

```bash
IMCFLOW_MAC_COUNTING=1 \
python3 scripts/analyze_model_tile_energy.py "$POWER_RUN" \
  --rtl-timing "$RTL_TIMING" \
  --analysis-config "$ENERGY_CONFIG" \
  --peak-overrides overrides.json
```

## 11. 자주 발생하는 오류

### `power/RTL identity mismatch`

Power capture와 timing의 checkpoint, sample, seed, BUGFIX 또는 fingerprint가 다르다.
동일 옵션으로 Linux compile/capture와 RTL manifest/timing을 다시 만든다. B1/B2 board
차이만으로는 이 오류가 발생하지 않는다.

### `at least one rail must detect every tile`

Tile 순서를 잡을 완전한 기준 rail이 없다. `--detect-only` 출력과 raw plot을 확인하고
detector threshold를 조정하거나 측정을 다시 한다.

### `integration window exceeds raw trace`

DMM capture가 inference보다 먼저 끝났다. `IMCFLOW_POWER_SAMPLE_COUNT`를 늘리고 compile,
ARM binary build, capture를 다시 실행한다.

### DMM logical-name mismatch

Compile의 `IMCFLOW_POWER_DMM_NAMES`와 bridge config의 `POWER` key 순서가 다르다. 둘 다
`VDD,DDA,DDC`로 맞춘 뒤 다시 컴파일한다.

### TOPS/W가 `null`

`build_metadata.json`에 `conv_mac_count`가 없거나 rail voltage/energy가 완전하지 않은
legacy run일 가능성이 크다. 현재 코드로 compile/capture했는지와 DMM config의
`VOLTAGE_V`를 확인한다.
