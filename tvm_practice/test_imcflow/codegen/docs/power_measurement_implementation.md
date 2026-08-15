# IMCFlow/TVM power 측정 구현 요약

이 문서는 TVM `power` 브랜치와 같은 시기에 IMCFlow 저장소에 추가된
power-analysis 코드를 기준으로, 실제 chip의 전류를 측정하고 전력으로
환산하는 전체 흐름을 정리한다.

## 분석 범위

### TVM

- 브랜치: `power`
- 분석한 HEAD: `d347280abe` (`Add MMIO trace debugging and update power measurement config`)
- power 작업 시작점: `38be6e55fe`
- 주요 first-parent commit:
  - `40f50670f2`: Linux generated kernel에 power 측정 코드 삽입
  - `ffc73adfb5`: MODEL/REGION/TILE 측정 phase 추가
  - `429e7a5bb7`: kernel launch 진단 출력 보강
  - `c5e962e2e5`: `measurement_utils`를 TVM submodule로 사용
  - `58788eb850`: power 측정 실행 script 수정
  - `9ce0665acd`: 당시 `origin/imcflow` 변경 merge 및 script 갱신
  - `d347280abe`: MMIO trace와 최종 DMM 설정 반영

### IMCFlow

현재 로컬과 `origin`에는 이름이 정확히 `power`인 branch ref가 없다. 따라서
TVM `power` 작업 직후인 2026-02-26에 만들어졌고 현재 여러 IMCFlow branch에
포함된 다음 commit을 대응 구현으로 분석했다.

- commit: `f83a087f` (`Add power measure folder`)
- 추가 위치: `xilinx/measurement/power_analysis/`
- 후속 commit `a10888b1`은 notebook을 다시 저장했지만, 최초 power 측정
  분석 구조 자체는 `f83a087f`에서 만들어졌다.

즉, 이 문서에서 말하는 “IMCFlow power 코드”는 별도 branch 이름이 아니라
위 commit의 분석 notebook과 측정 결과 sample을 뜻한다.

현재 checkout을 바꾸지 않고 당시 코드를 확인하려면 다음 명령을 사용할 수
있다.

```bash
git -C /root/project/tvm diff 38be6e55fe..power
git -C /root/project/tvm show power:python/tvm/relay/backend/contrib/imcflow/ext_codegen.py
git -C /root/project/imcflow show f83a087f:xilinx/measurement/power_analysis/imcflow_power_analysis.ipynb
```

## 핵심 구조

```text
TVM codegen
  └─ generated IMCFlow kernel C에 DMM start/end 호출 삽입
             │
             ▼
ARM Linux board의 execute_graph
  └─ measurement_utils C API
             │ TCP (기본 127.0.0.1:9900)
             ▼
measurement PC의 measure-bridge-daemon
  └─ VDD / DDA / DDC DMM에 측정 요청
             │
             ├─ bridge host 측 <kernel>_<rail>.txt
             └─ DMM RPC server 측 <kernel>_<rail>_server.txt
                              │
                              ▼
IMCFlow power_analysis notebook
  └─ current trace plot + 평균 전류/전력 계산
```

중요한 점은 TVM이 직접 power 값을 측정하는 것이 아니라 **세 rail의 current
sample을 수집**한다는 것이다. 실제 power는 IMCFlow notebook에서 rail별
전압을 곱해 계산한다.

## TVM `power` 브랜치가 만든 코드

### 1. power 측정 활성화 조건

`power:python/tvm/relay/backend/contrib/imcflow/ext_codegen.py`는
host OS와 환경변수로 power code 생성 여부를 결정한다.

- `IMCFLOW_HOST_OS=baremetal`
  - `MEASURE_POWER=False`로 고정된다.
  - DMM header와 호출이 생성되지 않는다.
- `IMCFLOW_HOST_OS=linux`
  - `IMCFLOW_MEASURE_POWER`가 `1`, `true`, `yes` 중 하나이면 측정 코드를
    생성한다.
  - 값을 생략하면 기본값은 `0`, 즉 비활성이다.

power branch의 `tvm_practice/test_imcflow/codegen/.envrc`는 다음 설정을
기본으로 선택했다.

```bash
export IMCFLOW_HOST_OS=linux
export IMCFLOW_HOST_ISA=arm
export IMCFLOW_MEASURE_POWER=1
```

단, CMake 조건은 `IMCFLOW_MEASURE_POWER`가 문자열로 정확히 `1`일 때만
`dmm_measure.c`를 link한다. Codegen은 `true`와 `yes`도 받아들이지만 이 값을
사용하면 generated C에는 DMM 호출이 있고 binary에는 구현이 link되지 않는
불일치가 생길 수 있다. 따라서 이 branch에서는 `1`을 사용해야 안전하다.

### 2. 측정 phase

다음 세 가지 측정 범위를 enum으로 구현했다.

| Phase | DMM start 위치 | DMM end 위치 | 포함되는 범위 |
|---|---|---|---|
| `MODEL` | 첫 번째 region kernel의 reset/warmup 다음 | 마지막 region kernel의 모든 tile/output 처리 다음 | 여러 region에 걸친 전체 model 실행 구간 |
| `REGION` | 각 region kernel의 reset/warmup 다음 | 해당 region의 모든 tile/output 처리 다음 | region별 instruction/constant transfer, policy update, 실행, output read |
| `TILE` | tile input transfer 다음, NPU invoke 직전 | NPU invoke 직후, output read 전 | tile의 NPU 실행 중심 구간 |

`power` HEAD의 실제 기본값은 환경변수가 아니라 코드에 고정된
`PowerMeasurePhase.REGION`이다. 다른 phase를 쓰려면 `POWER_MEASURE_PHASE`를
수정하고 codegen과 host binary build를 다시 해야 한다.

MODEL boundary는 함수 이름에 `region1`이 포함되는지, 그리고
`region<전체 region 수>`가 포함되는지로 판정한다. 따라서 generated function
이 이 naming convention을 따르지 않으면 MODEL start/end가 만들어지지 않는다.

### 3. warmup과 측정 시작 위치

모든 Linux region kernel은 power 측정 시작 전에 다음 명령을 실행하도록
생성된다.

```text
make .../petalinux-csrc clear_time
make .../petalinux-csrc warmup
```

그 다음 active phase에 맞춰 DMM을 시작한다. 따라서 REGION/TILE 측정에서는
해당 kernel 시작 시의 warmup 자체는 측정 구간에서 제외된다. 반면 MODEL
측정은 region 1에서 시작한 후 마지막 region에서 끝나므로, 중간 region
kernel이 자체적으로 수행하는 reset/warmup과 region 사이 host overhead도
전체 model window에 들어갈 수 있다.

### 4. DMM 측정 설정

세 DMM을 동시에 사용한다.

| 항목 | VDD | DDA | DDC |
|---|---:|---:|---:|
| DMM name | `VDD` | `DDA` | `DDC` |
| NPLC | 0.001 | 0.001 | 0.001 |
| sample interval | MIN (`-1`) | MIN (`-1`) | MIN (`-1`) |
| 최대 sample count | 50,000 | 50,000 | 50,000 |
| current range | 0.1 A | 0.1 A | 0.1 A |
| 측정 전 reset | yes | yes | yes |

측정은 `now` mode를 사용한다.

1. `dmm_start_current_now(3, configs)`로 세 DMM sampling을 non-blocking으로
   시작한다.
2. 측정 대상 kernel 구간을 실행한다.
3. `dmm_get_result_now()`를 세 번 호출하여 그 시점까지 수집된 각 DMM의
   평균 전류와 sample 수를 받는다.
4. 결과를 stderr에 출력하고 `dmm_close()`로 연결을 정리한다.

따라서 실행 구간이 50,000 sample보다 먼저 끝나면 실제 파일의 sample 수는
50,000보다 작을 수 있다.

### 5. 측정 파일 이름

REGION과 MODEL phase에서 bridge-host 파일은 다음 형식이다.

```text
<generated_func_name>_vdd.txt
<generated_func_name>_dda.txt
<generated_func_name>_ddc.txt
```

DMM RPC server 측에는 `_server`가 추가된다.

```text
<generated_func_name>_vdd_server.txt
```

TILE phase에서는 함수 이름 뒤에 tile 번호가 들어간다.

```text
<generated_func_name>_tile<t_idx>_vdd.txt
```

MODEL 측정도 start를 생성한 첫 region의 `self.func_name`으로 파일명을
만들기 때문에, 전체 model 측정 결과인데도 이름에는 `region1`이 들어간다.
IMCFlow의 `modelwise/` sample이 region 1 이름을 가진 이유가 이것이다.

### 6. `measurement_utils` 연결

TVM은 `3rdparty/measurement_utils` submodule을 추가했고, `power` HEAD에서는
commit `0f136b05fdadeb7800918ad9c87616130259d54f`를 가리킨다.

Linux power build일 때 다음 CMake 파일들이
`$TVM_HOME/3rdparty/measurement_utils/capi/dmm_measure.c`를 model library와
host runner에 포함한다.

- `tvm_practice/test_imcflow/codegen/host_binary_make.template/CMakeLists.txt`
- `tvm_practice/test_imcflow/codegen/host_binary_make.dataset/CMakeLists.txt`

생성된 kernel은 `dmm_measure.h`를 include한다. C API는 board에서 TCP bridge로
측정 요청을 보낸다.

- `DMM_BRIDGE_HOST`: bridge daemon 주소, 기본 `127.0.0.1`
- `DMM_BRIDGE_PORT`: bridge daemon port, 기본 `9900`
- 외부 측정 PC에서 `measure-bridge-daemon --host 0.0.0.0 --port 9900` 실행 필요
- bridge daemon은 별도의 DMM RPC server와 config를 사용한다.

즉 board와 DMM을 제어하는 PC가 다르면 `DMM_BRIDGE_HOST`를 실제 측정 PC의
주소로 지정해야 한다. 기본값 그대로면 board 자신의 localhost에 daemon이
있다고 가정한다.

### 7. build와 실행 script

`power:tvm_practice/test_imcflow/codegen/run_chiptest.sh`는 다음 절차를
자동화하도록 수정됐다.

1. Linux/ARM용 model artifact 생성
2. eval directory를 원격 board로 전송
3. scan register NPZ 전송
4. scan programming executable 전송
5. 원격 board에서 scan programming 후 `clear_time`/`warmup`
6. `execute_graph`용 `run.sh`와 10회 반복용 `run_loop.sh` 생성 및 전송

세 번째 positional argument로 board host IP를 바꿀 수 있으며,
`transfer_evl.sh`에도 `--host` 옵션이 추가됐다. dataset transfer script도
host를 argument로 받을 수 있고, dataset 원격 실행은 Python virtual
environment를 활성화한 뒤 시작하도록 바뀌었다.

주의할 점은 `power` HEAD의 `run_chiptest.sh`에서 step 6의 실제 원격 SSH 실행
부분이 주석 처리되어 있다는 것이다. 이 step은 `run.sh`와 `run_loop.sh`를
board에 복사하지만 곧바로 model을 실행하지 않는다. 따라서 board에서
다음 중 하나를 별도로 실행해야 실제 측정이 시작된다.

```bash
bash run.sh
bash run_loop.sh
```

또한 script는 측정 결과 txt를 bridge host/DMM RPC server에서 로컬 IMCFlow
`power_analysis/`로 회수하는 단계는 제공하지 않는다. 결과 수집과
`modelwise/`, `regionwise/` 분류는 별도 작업이다.

### 8. MMIO trace

`DEBUG_MMIO_TRACE=True`가 기본으로 설정되어, generated kernel이 주요 MMIO
작업의 시작과 완료를 stderr에 출력한다.

- reset register write
- policy PC 설정과 NPU invoke
- instruction/object 및 C variable의 NPU memory write
- output memory read

이 trace는 측정 중 멈춘 위치를 찾는 데 유용하다. 다만 REGION과 MODEL
측정 window 안에 포함된 stderr 출력과 host-side transfer가 측정 시간과 평균에
영향을 줄 수 있으므로, 순수 NPU core power와 동일한 값으로 해석하면 안 된다.

## IMCFlow 쪽에 추가된 분석 코드

### 1. 저장된 측정 결과

`f83a087f`는
`xilinx/measurement/power_analysis/` 아래에 다음 파일을 추가했다.

- `modelwise/`: 전체 model 측정 1개 × VDD/DDA/DDC = 3개 current trace
- `regionwise/`: region 1–4 × VDD/DDA/DDC = 12개 current trace
- `imcflow_power_analysis.ipynb`: trace 시각화와 평균 power 계산 notebook

각 `.txt`는 current sample 배열을 담고 있다. TVM codegen의 측정 파일 이름을
그대로 사용하기 때문에 긴 `tvmgen_default_...region...` 이름을 가진다.
commit에 저장된 15개 결과 파일은 각각 10줄이며, 한 줄이 한 번의 실행에서
append된 하나의 burst다. 이는 `run_loop.sh`가 `run.sh`를 10회 실행하도록
만든 구조와 대응한다.

### 2. notebook 처리

[`imcflow_power_analysis.ipynb`](../../../../../imcflow/xilinx/measurement/power_analysis/imcflow_power_analysis.ipynb)는
`ps_ctrl.analysis`의 다음 helper를 사용한다.

- `load_current_records()`: rail별 current sample 로드
- `plot_current_records()`: model/region별 VDD, DDA, DDC trace plot
- `compute_active_power(..., mode="flatten")`: sample을 평탄화하여 평균 전류와
  평균 power 계산

notebook에 고정된 rail 전압은 다음과 같다.

| Rail | 분석 전압 |
|---|---:|
| VDD | 1.00 V |
| DDA | 1.13 V |
| DDC | 1.17 V |

출력은 rail별 평균 current와 power다.

```text
avg_power = avg_current × configured_voltage
```

notebook은 세 rail의 power를 합산한 total chip power를 자동으로 계산하지
않으며, idle/base power를 빼는 처리도 보이지 않는다. 또한 전압이 notebook에
상수로 들어 있으므로 실제 측정 때 설정한 전압이 다르면 값을 수정해야 한다.

다만 재현성 문제가 있다. TVM `power` HEAD가 pin한 `measurement_utils`
`0f136b05`의 `ps_ctrl.analysis`는 다음 symbol만 export한다.

```text
load_current_records, parse_line, CurrentHistogram, HistogramResult
```

notebook이 import하는 `compute_active_power`와 `plot_current_records`는 이
revision에 없다. 따라서 clean checkout에서 이 notebook은 import 단계에서
실패하며, 작성 당시 사용한 별도의/다른 revision의 `ps_ctrl` 분석 utility가
추가로 필요하다. 위 power 계산은 notebook 호출이 의도하는 계산을 요약한
것이며, 누락된 두 helper의 구현은 해당 commit들 안에서 확인할 수 없다.

## 측정값 해석 시 주의사항

1. **기본 phase는 REGION이다.** 환경변수만으로 MODEL/TILE을 선택할 수 없다.
2. **REGION 측정은 compute만 재는 것이 아니다.** instruction/constant/input
   transfer, policy update, invoke, output read와 일부 host overhead가 포함된다.
3. **MODEL 측정에는 region 사이 작업이 포함될 수 있다.** 첫 region에서 연
   DMM session을 마지막 region에서 닫는 구조다.
4. **`IMCFLOW_MEASURE_POWER=1`을 사용해야 한다.** Codegen과 CMake의 허용값이
   서로 다르다.
5. **bridge daemon은 별도 준비가 필요하다.** board가 측정 PC와 통신할 수
   있어야 한다.
6. **전압은 측정 파일에 함께 기록되지 않는다.** notebook의 1.00/1.13/1.17 V
   값이 실제 실험 조건과 일치해야 한다.
7. **저장된 값은 current trace다.** total/active/dynamic power를 구분하려면
   rail 합산과 idle subtraction 정책을 추가로 정의해야 한다.
8. **실행과 결과 회수는 완전 자동화되지 않았다.** `run_chiptest.sh`는 실행
   script를 board에 보내지만 실제 원격 실행과 결과 수집은 수동 단계다.
9. **분석 notebook dependency가 불완전하다.** pin된 `measurement_utils`에는
   notebook이 요구하는 helper 두 개가 없어 추가 구현이나 맞는 과거 revision이
   필요하다.

## 코드가 의도한 재현 순서

```text
1. TVM power branch와 measurement_utils submodule 준비
2. IMCFLOW_HOST_OS=linux, IMCFLOW_HOST_ISA=arm,
   IMCFLOW_MEASURE_POWER=1 설정
3. 필요한 경우 ext_codegen.py의 MODEL/REGION/TILE phase 선택
4. model을 다시 codegen하고 host binary를 다시 build
5. 측정 PC에서 DMM RPC/bridge daemon 준비
6. board에 artifact와 scan 값을 전송하고 scan programming
7. board에서 run.sh 또는 run_loop.sh 실행
8. bridge host/DMM RPC server의 rail별 txt 결과 회수
9. 누락된 ps_ctrl 분석 helper를 제공하는 환경 준비
10. 실제 rail 전압을 notebook에 반영한 뒤 trace와 평균 power 분석
```

요약하면 TVM `power` 브랜치는 **측정 시작/종료 시점을 generated kernel에
삽입하고 DMM bridge를 build/run 경로에 연결**했으며, IMCFlow의 대응 commit은
**실제 VDD/DDA/DDC current trace를 보관하고 model/region별 power를 분석하는
notebook**을 추가했다.
