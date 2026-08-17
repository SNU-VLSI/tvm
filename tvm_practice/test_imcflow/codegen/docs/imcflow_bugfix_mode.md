# IMCFLOW_BUGFIX 동작과 컴파일 파이프라인

이 문서는 현재 TVM `chip_acc_measure` 계열 코드에서
`IMCFLOW_BUGFIX=off`와 `IMCFLOW_BUGFIX=on`이 각각 무엇을 바꾸는지 설명한다.
특히 다음 두 종류의 "stage"를 구분한다.

1. `main.py`가 실행하는 TVM 테스트 파이프라인 stage
2. BUGFIX-off RTL deadlock을 피하기 위해 도입된 P0–P4 동기화 stage

## 핵심 요약

`IMCFLOW_BUGFIX`는 단순히 하나의 최적화를 켜고 끄는 옵션이 아니다. 같은
값이 TVM codegen과 RTL compile 양쪽을 함께 선택한다.

| 설정 | TVM codegen | RTL compile |
|---|---|---|
| `IMCFLOW_BUGFIX=off` (기본값) | 934/P0–P4 동기화, 검증 및 일부 SW workaround 사용 | `BUGFIX_*` RTL define을 넣지 않음 |
| `IMCFLOW_BUGFIX=on` | 과거 a8af 동기화/codegen 경로 사용 | `BUGFIX_*` RTL define 6개를 넣음 |

따라서 `off`라는 이름은 **TVM workaround를 끈다**는 뜻이 아니다.
`BUGFIX_*`로 보호된 **RTL의 hardware fix를 끈다**는 뜻이며, 이 RTL에서
deadlock과 overflow 문제를 피하기 위한 TVM 측 동기화/SW 보정은 오히려
활성화된다.

환경변수를 생략하면 현재 기본값은 `off`다. 대소문자는 정규화하지만
`on`과 `off` 이외의 값은 오류로 처리한다.

```python
mode = os.environ.get("IMCFLOW_BUGFIX", "off").strip().lower()
```

구현: [`tvm.contrib.imcflow.get_imcflow_bugfix_mode()`](../../../../python/tvm/contrib/imcflow.py)

## 전체 실행 흐름

일반적인 RTL 실행은 다음 순서로 진행된다.

```text
환경변수/모델/체크포인트 선택
        │
        ▼
1. TRANSFORM
        │
        ▼
2. CODEGEN ── BUGFIX off/on 차이가 가장 많이 발생하는 단계
        │
        ▼
3. GRAPH_EXECUTOR(COMPILE)
        │
        ▼
4. CPU_VALIDATION
        │
        ▼
5. SIMULATION ── RTL manifest 확인 및 필요 시 VCS 재컴파일
        │
        ▼
6. COMPARISON
```

`python main.py --stop-at simulate`는 1번부터 5번까지 실행하고 출력 비교는
생략한다. `--start-at simulate`는 기존 transformed model과 build artifact를
재사용하므로, BUGFIX mode나 checkpoint를 바꾼 뒤에는 사용하면 안 된다.

Pipeline stage 정의는
[`runners/pipeline_options.py`](../runners/pipeline_options.py)에 있다.

## 0. 실행 준비와 mode 선택

모델을 만들고 checkpoint를 로드한 뒤 eval directory를 준비한다.

- `off`: `eval_dir/<model>_evl.<host>.bugfixoff`
- `on`: `eval_dir/<model>_evl.<host>`

두 mode의 generated C++, host binary, 로그와 출력이 같은 디렉터리를
덮어쓰지 않도록 `off`에 `.bugfixoff` suffix를 붙인다.

이 시점의 mode는 이후 codegen과 RTL runner에 동일하게 전달된다.

## 1. TRANSFORM stage

Relay 모델을 IMCFlow에서 실행 가능한 형태로 변환한다. 대표적으로 IMCFlow
연산 partition, layout/tiling, memory 및 node mapping에 필요한 정보를 만든다.

`IMCFLOW_BUGFIX`의 주된 차이는 이 frontend 변환 자체보다 다음 CODEGEN
stage에 있다. 동일한 transformed graph라도 codegen이 생성하는 SEND,
RECV, LOAD_LB, SETFLAG, STANDBY 및 STEP 순서가 mode에 따라 달라질 수 있다.

## 2. CODEGEN stage

각 IMCFlow subgraph에 대해 다음 작업을 수행한다.

1. inter-node tensor edge를 수집한다.
2. edge별 router policy와 receive FIFO를 정한다.
3. SEND/RECV pair와 UUID를 만든다.
4. inode/IMCE code block을 생성한다.
5. send/receive 및 flag-handshake 수를 검증한다.
6. inode/IMCE C++와 device program을 생성한다.

### BUGFIX=off의 기본 동기화 경로

`off`에서는 `filter_contention=False`로 `SendRecvPairManager`를 생성하여
constant를 제외한 inter-node pair를 유지한다. 이 pair 정보는 producer와
consumer 사이의 rendezvous를 생성하는 데 사용된다.

주요 동작은 다음과 같다.

- inode data SEND 전에 consumer가 받을 준비가 되었는지 확인하는 pre-send
  rendezvous를 넣는다.
- IMCE의 `LOAD_LB`/`RECV`를 flag window로 감싸 producer와 consumer의 진행
  순서를 맞춘다.
- multicast 및 diamond 구조에서는 모든 관련 receiver가 준비된 뒤 producer가
  전송하도록 barrier를 구성한다.
- NoC operand가 있는 boundary col-group에서는 stall 중 STEP 상태가 꼬이지
  않도록 LOAD_LB/RECV/STEP 순서를 조정한다.
- padding row도 NoC operand를 drain하여 upstream FIFO가 가득 차는 것을 막는다.
- producer와 consumer의 burst 폭이 다르면 dummy drain/SEND를 추가하여 cadence를
  맞춘다.

이 동작들은 일반적인 성능 최적화가 아니라 BUGFIX-off RTL의 FIFO full/empty,
lost wakeup, stale STEP 상태에 의한 deadlock을 방지하기 위한 codegen 규칙이다.

### BUGFIX=on의 a8af 경로

`on`에서는 과거 a8af 동작을 재현한다.

- contention filtering 경로가 최종 pair set을 비우므로 934 방식의 전체
  inter-node rendezvous를 생성하지 않는다.
- inode SEND는 a8af post-send 경로를 사용한다.
- `LoadLBBlock`, `ConvBlock`, `RecvSendWrapper`는 각각 `_build_a8af()` 또는
  `_render_a8af()` 경로를 사용한다.
- BUGFIX-off 전용 Marker, burst padding과 P3/P4 DWCONV barrier를 사용하지 않는다.

이 경로가 가능한 이유는 RTL compile 단계에서 hardware BUGFIX define들이
활성화되기 때문이다.

## BUGFIX-off 동기화 P0–P4

P0–P4는 `main.py`의 pipeline stage가 아니다. CODEGEN 내부에서 producer와
consumer의 transfer/handshake granularity를 일치시키기 위해 순차적으로
도입된 설계 단계다.

### P0: edge별 sync contract 정의

`TensorEdgeInfo`에 다음 contract 정보를 추가한다.

- `channels_per_issue`
- `fill_order`
- `producer_send_per_sync`
- `consumer_recv_per_sync`
- `needs_flag_rendezvous`

P0 자체는 metadata만 정의하며 직접 instruction을 추가하지 않는다. `on`에서는
이 contract를 채우지 않아 a8af 동작을 유지한다.

### P1: producer가 sync contract 사용

inode `SendBlock`이 `producer_send_per_sync`를 읽고 몇 개의 packet마다 한 번
rendezvous할지 결정한다. 현재 기본 contract의 값 `1`은 packet마다 handshake
한다는 의미다.

`on`에서는 이 contract 기반 pre-send 경로 대신 a8af의 post-send 경로를
사용한다.

### P2: FIFO send/recv count 검증

edge마다 다음 invariant를 compile time에 확인한다.

```text
Σ SEND == Σ RECV
```

불일치하면 RTL에서 producer가 full FIFO에 막히거나 consumer가 empty FIFO를
기다리면서 20000-poll hang이 발생할 수 있다. `off`에서는 상세 edge 정보와
함께 `AssertionError`를 발생시킨다.

진단 목적으로만 다음 override가 있다.

```bash
IMCFLOW_SKIP_SYNC_ASSERT=1
```

`on`에서는 a8af 호환을 위해 이 assert를 실행하지 않는다.

### P3: conv/DWCONV input-side granularity 일치

policy table 작성 시 qconv와 qdwconv data edge에 sync contract를 기록한다.

| consumer | `channels_per_issue` | fill order | consumer cadence |
|---|---:|---|---:|
| qconv | 64 | channel pass → H → W → bitplane | packet당 1 |
| qdwconv | 16 | channel pass → H → W → bitplane | window당 4 |

특히 DWCONV의 bitplane burst에서 inode producer와 IMCE `LOAD_LB` consumer가
서로 다른 횟수로 flag를 올리는 문제를 막는다.

`on`에서는 contract를 기록하지 않고 기존 a8af DWCONV input 경로를 사용한다.

### P4: DWCONV middle-stage multicast barrier

DWCONV 앞의 minmaxquant 결과가 여러 IMCE로 multicast될 때 producer가 matching
flag를 올리지 않아 receiver의 `STANDBY`가 영원히 끝나지 않는 문제를 다룬다.

`off`에서는 다음을 수행한다.

- minmaxquant producer가 output SEND 전에 flag-2 pre-send barrier를 수행한다.
- 각 DWCONV consumer는 matching flag-2 LOAD_LB window를 사용한다.
- producer handshake 수와 각 consumer window 수가 같은지 검증한다.
- 불일치하면 simulation 전에 compile-time assert로 실패한다.

`on`에서는 minmaxquant의 bare recv/compute/send a8af 경로를 사용하고 P4
handshake validation도 생략한다.

### VWW용 BUGFIX-off 확장

VWW 첫 DWCONV는 DS-CNN과 달리 inode에서 직접 입력을 받고 `repeat=4`인
경로를 사용한다. `off`에서는 추가로 다음을 처리한다.

- 하나의 window로 4개 LOAD_LB를 감싸지 않고 LOAD_LB마다 flag-1 window를
  생성하여 inode의 packet당 handshake와 1:1로 맞춘다.
- channel group이 16개보다 작으면 DWCONV `src_mask`를 15까지 넓혀 상위
  lane도 정의된 값으로 기록한다. BUGFIX-off RTL에서 미기록 lane의 X가 NoC로
  전파되어 `$fatal`이 발생하는 것을 막는다.

두 동작 모두 `on`에서는 비활성화된다.

### Overflow software compensation

BUGFIX-off RTL에는 `BUG_FIX_OVERFLOW`가 없으므로 MULTL overflow software fix가
기본 활성화된다.

- `off`: `IMCFLOW_BUGFIX_OVERFLOW_SW` 기본값 ON
- `on`: `IMCFLOW_BUGFIX_OVERFLOW_SW` 기본값 OFF

명시적인 `IMCFLOW_BUGFIX_OVERFLOW_SW=0` 또는 `1`이 master knob보다 우선한다.

## 3. GRAPH_EXECUTOR / device compile stage

CODEGEN에서 생성한 inode/IMCE C++를 device instruction binary로 만들고 TVM
graph/MLF artifact를 생성한다.

inode와 IMCE program의 clang option은 두 mode에서 동일하다.

```text
inode: -O1 --target=INODE -mllvm=-force-nested-hardware-loop
imce : -O1 --target=IMCE  -mllvm=-force-hardware-loops
                         -mllvm=-force-nested-hardware-loop
```

즉 `IMCFLOW_BUGFIX`는 clang optimization level을 변경하지 않는다. 두 mode의
binary가 다른 이유는 `-O1` 차이가 아니라 이전 CODEGEN stage에서 생성한
instruction sequence가 다르기 때문이다.

구현: [`device_codegen.py`](../../../../python/tvm/relay/backend/contrib/imcflow/device_codegen.py)

## 4. CPU_VALIDATION stage

원본 또는 transformed Relay 모델을 CPU에서 실행해 reference output을 만든다.
이 stage는 RTL mode를 compile하지 않는다.

다만 `off`의 software overflow compensation과 같은 transformation/codegen
의미가 transformed reference와 RTL 결과 비교에 영향을 줄 수 있다. 또한
BUGFIX-off RTL에서 deadlock이 사라졌다는 사실만으로 numerical output이
BUGFIX-on과 같다는 의미는 아니다. hardware fix가 빠진 STEP/overflow/DWCONV
경로에서는 값 차이가 별도로 존재할 수 있다.

## 5. SIMULATION stage와 RTL compile

RTL runner setup은 다음 명령을 호출한다.

```bash
make ensure-compiled IMCFLOW_BUGFIX=<off|on>
```

### BUGFIX=off RTL define

다음 base define만 사용한다.

```text
SUPPORT_K2
SUPPORT_K3
SUPPORT_K5
SUPPORT_K7
FSIM
MEM_MODEL
DEBUG
```

다음 hardware bugfix define은 **포함하지 않는다**.

```text
BUGFIX_READ_FROM_GPR
BUGFIX_LOAD_USE_HAZARD
ROUTER_BUG_FIX
BUG_FIX_OVERFLOW
BUGFIX_STEP
BUGFIX_DWCONV
```

### BUGFIX=on RTL define

base define에 위의 `BUGFIX_*` 6개를 모두 추가한다. TVM codegen은 이에 맞춰
a8af 경로를 사용한다.

### build manifest에 의한 재컴파일

두 mode는 같은 RTL runner/build directory를 사용한다. 다음 값들이
`build/build_manifest.json`의 fingerprint에 포함된다.

- mode와 최종 `DEFINE`
- VCS/DPI option
- RTL, testbench, DPI 및 technology-model source 내용
- include tree
- compile-time path
- VCS executable과 version identity

기존 `simv`가 있어도 `on`에서 `off`로, 또는 `off`에서 `on`으로 바꾸면
manifest가 달라져 VCS가 자동 재컴파일된다. checkpoint와 runtime input은 RTL
구조를 바꾸지 않으므로 RTL build fingerprint에는 포함되지 않는다.

TVM 연결부: [`imcflow_runner.py`](../runners/imcflow_runner.py)

RTL Makefile:
`/root/project/imcflow/pmap/ISA_sim/gem5/tests/imcflow/rtl_runner/Makefile`

## 6. COMPARISON stage

IMCFlow simulation output과 CPU reference output을 비교한다.
`--stop-at simulate`에서는 이 stage를 실행하지 않는다.

BUGFIX-off의 sync stage는 주로 실행 완료와 deadlock 방지를 보장한다. 값의
동일성은 별도 비교가 필요하며, workload가 disabled hardware fix에 의존하면
BUGFIX-on과 numerical result가 다를 수 있다.

## Mode 비교표

| 항목 | `BUGFIX=off` | `BUGFIX=on` |
|---|---|---|
| 기본 여부 | 기본값 | 명시적으로 설정 |
| eval directory | `.bugfixoff` suffix | 기존 이름 |
| inter-node pair | 전체 pair 유지 | a8af filtering 결과 사용 |
| inode data SEND | pre-send rendezvous | a8af post-send 경로 |
| IMCE LOAD_LB/RECV | 934 window sync | a8af 경로 |
| P0/P1 contract | 사용 | 사용하지 않음 |
| P2 FIFO count assert | 실행 | 생략 |
| P3 DWCONV input contract | 실행 | 생략 |
| P4 multicast barrier/assert | 실행 | 생략 |
| VWW per-LOAD_LB fix | 실행 | 생략 |
| partial DWCONV X 방지 | 실행 | 생략 |
| overflow SW fix 기본값 | ON | OFF |
| RTL `BUGFIX_*` define | 없음 | 6개 모두 포함 |
| device clang optimization | `-O1` | `-O1` |

## 다른 실험 knob과의 관계

다음 옵션은 `IMCFLOW_BUGFIX`와 독립적인 실험/실리콘 workaround다. master
knob가 `off`라고 자동 활성화되지 않는다.

- `IMCFLOW_SERIALIZE_IMCU`
- `IMCFLOW_IMCU_INTRA_DRAIN`
- `IMCFLOW_MMIO_BARRIER`
- `IMCFLOW_MULTIBLOCK_FUSEDADD_BARE`
- `IMCFLOW_MULTIBLOCK_FUSEDADD_SAFE`
- `IMCFLOW_FEED_SPREAD`
- `IMCFLOW_FEED_PREFETCH`
- `IMCFLOW_DROP_PSUM`

BUGFIX mode 비교 실험에서는 위 값을 명시적으로 통제하지 않으면 서로 다른
원인의 변화가 섞일 수 있다.

## 실행 예시

BUGFIX-off RTL:

```bash
IMCFLOW_RUNNER=rtl \
IMCFLOW_DIR=/root/project/imcflow \
IMCFLOW_BUGFIX=off \
python -u main.py \
  --model resnet8_subset31_pretrained_orig \
  --stop-at simulate
```

BUGFIX-on RTL:

```bash
IMCFLOW_RUNNER=rtl \
IMCFLOW_DIR=/root/project/imcflow \
IMCFLOW_BUGFIX=on \
python -u main.py \
  --model resnet8_subset31_pretrained_orig \
  --stop-at simulate
```

실제 사용 mode와 checkpoint는 eval directory의 `build_metadata.json`, RTL
build mode와 source fingerprint는 RTL runner의 `build/build_manifest.json`,
send/recv 검증 결과는 eval directory의 `recv_send_consistency.txt`에서 확인할
수 있다.

## 주요 구현 위치

- mode parser 및 공통 knob: [`python/tvm/contrib/imcflow.py`](../../../../python/tvm/contrib/imcflow.py)
- pair 생성/filter: [`send_recv_sync.py`](../../../../python/tvm/relay/backend/contrib/imcflow/send_recv_sync.py)
- sync contract: [`policy_table_builder.py`](../../../../python/tvm/relay/backend/contrib/imcflow/policy_table_builder.py)
- inode SEND/RECV: [`inode_codeblock.py`](../../../../python/tvm/relay/backend/contrib/imcflow/inode_codeblock.py)
- IMCE LOAD_LB/RECV/SEND/STEP: [`imce_codeblock.py`](../../../../python/tvm/relay/backend/contrib/imcflow/imce_codeblock.py)
- compile-time invariant 검증: [`codegen.py`](../../../../python/tvm/relay/backend/contrib/imcflow/codegen.py)
- test pipeline: [`test.py`](../test.py)
- RTL runner 연결: [`imcflow_runner.py`](../runners/imcflow_runner.py)
