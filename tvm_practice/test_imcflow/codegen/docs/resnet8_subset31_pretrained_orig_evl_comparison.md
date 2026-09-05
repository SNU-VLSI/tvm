# ResNet8 IMCFLOW 생성 코드 비교

## 1. 비교 대상

| 구분 | 경로 | 생성 시각 |
|---|---|---|
| A (backup) | `backup/resnet8_subset31_pretrained_orig_evl.linux` | 2026-08-04 20:50 |
| B (bugfixoff) | `resnet8_subset31_pretrained_orig_evl.linux.bugfixoff` | 2026-09-03 18:06 |

이 문서는 다음 두 종류의 코드를 별도로 비교한다.

1. IMCFLOW에서 실행되는 코드
   - `build/tvmgen_default_imcflow_main_*/inode.cpp`
   - `build/tvmgen_default_imcflow_main_*/imce.cpp`
   - 여기서 생성된 `*_imem.bin`, `*_policy.bin`
2. CPU에서 실행되는 코드
   - `lib_graph_system-lib.tar`의 `codegen/host/src/default_lib0.c`
   - `default_lib1.c`
   - 18개 `default_lib*.cc` IMCFLOW host wrapper

## 2. 가장 중요한 결론

| 항목 | A | B | 판단 |
|---|---|---|---|
| Relay/graph 구조 | ResNet8, IMCFLOW subgraph 18개 | 동일 | 연산 topology는 사실상 동일 |
| IMCFLOW `acc_mask` | 15 | 1 | NPU 계산 설정이 동일하지 않음 |
| checkpoint | `resnet8_chip_noise_loop_iter_006` | `cifar10.chip4.update.run1_bootstrap_20260903_175312_2862875` | weight와 양자화 상수가 동일하지 않음 |
| CPU 연산 kernel 구현 | `default_lib1.c` | byte-identical | CPU kernel 코드는 동일 |
| IMCFLOW input 동기화 | 기존 SEND/LOAD | INODE/IMCE flag handshake와 delay 추가 | device 실행 순서가 달라짐 |
| 정상 interrupt 성공 경로 | IRQ 대기 -> ACK | IRQ 대기 -> **IDLE polling** -> ACK | B가 ACK 전에 IDLE을 기다림 |
| IMCFLOW imem 총 크기 | 75,232 B | 85,376 B | B가 10,144 B, 13.48% 증가 |
| policy binary | 360개 | 360개 | 전부 byte-identical |

따라서 B는 A에 단순히 한 가지 “bugfix off” 옵션만 적용한 결과가 아니다. checkpoint와 `acc_mask`가 함께 달라졌고, device 동기화 코드 및 CPU host interrupt 처리 코드도 달라졌다. 두 실행의 정확도나 성능 차이를 특정 bugfix 하나의 효과로 해석할 수 없다.

## 3. 빌드 조건 차이

`build_metadata.json` 기준 차이는 다음과 같다.

| 설정 | A | B |
|---|---|---|
| model | `resnet8_subset31_pretrained_orig` | 동일 |
| board / vmode | `B1` / `HALF` | 동일 |
| fixed IMCE core | `0,1` | 동일 |
| disabled columns | 동일한 B1 N32 설정, 32개 | 동일 |
| random seed | 42 | 42 |
| driver | v2 | v2 |
| `--acc-mask` | 15 | 1 |
| checkpoint alias | `resnet8_chip_noise_loop_iter_006` | `cifar10.chip4.update.run1_bootstrap_20260903_175312_2862875` |
| `imcflow_bugfix` metadata | 필드 없음 | `false` |
| TVM revision | 기록 없음 | `9fbabb1da8230e4a97bab066a5b10bd67e223bfb` |

`default.params`의 SHA-256도 서로 다르다.

| A | B |
|---|---|
| `56e3c1cd...c95f8ee` | `aa787408...47aeb` |

Relay text의 graph 연결, tensor shape, CPU/NPU partition은 유지되지만 다음 값이 바뀐다.

- 18개 `nn.imcflow_qconv` 모두 `acc_mask=15`에서 `acc_mask=1`로 변경된다.
- 여러 `qnn.imcflow_min_max_quantize`의 min/max 값이 변경된다.
- checkpoint가 다르므로 weight, bias, scale 등 constant 값이 변경된다.
- graph JSON의 차이는 18개 function hash 값이며 node 연결 구조는 동일하다.

## 4. IMCFLOW에서 실행되는 코드 비교

### 4.1 배치와 subgraph 구조

두 빌드 모두 IMCFLOW subgraph가 다음 18개이다.

```text
0, 3, 6, 9, 12, 15, 18, 21, 24,
27, 30, 33, 36, 39, 42, 45, 48, 51
```

`func_map.txt`와 `func_to_imce.txt`는 각각 byte-identical이다. 즉 function-to-device 배치 자체는 같다. 고정 배치 설정에 따라 실제 변경된 instruction binary는 각 subgraph의 `inode_0_0_imem.bin`과 `imce_0_1_imem.bin`이다.

전체 360개 imem binary 중 결과는 다음과 같다.

| binary | 동일 | 변경 | 설명 |
|---|---:|---:|---|
| `*_imem.bin` 전체 | 324 | 36 | 18개 `inode_0_0` + 18개 `imce_0_1` 변경 |
| `*_policy.bin` 전체 | 360 | 0 | 모든 policy binary가 byte-identical |
| `*cnt_base_addr.bin` | 36 | 0 | 모두 byte-identical |

`policy_table.txt`는 function 출력 순서가 달라 text hash는 다르지만, 최종 생성된 per-function policy binary 내용은 동일하다.

### 4.2 INODE 코드 변화

B의 active INODE 경로에는 IMCE로 data를 SEND하기 전 flag handshake가 추가됐다.

```cpp
// B에 추가된 동기화
__builtin_INODE_STANDBY(1, 1);
__builtin_INODE_SET_FLAG(1);
__builtin_INODE_STANDBY(1, 0);
__builtin_INODE_SET_FLAG(0);
__builtin_INODE_SEND(...);
```

또한 config SEND와 data SEND 뒤에 다음 delay가 추가됐다.

```cpp
for (int i = 0; i < 10; i++) {
  __asm__ volatile("nop");
}
```

18개 `inode.cpp`를 합산한 정적 call-site 변화는 다음과 같다. loop 내부 call은 runtime에 여러 번 실행될 수 있으므로 아래 값은 동적 실행 횟수가 아니라 C++ source에 나타나는 호출 위치 개수다.

| builtin/call site | A | B | 변화 |
|---|---:|---:|---:|
| `INODE_STANDBY` | 1,080 | 1,116 | +36 |
| `INODE_SET_FLAG` | 792 | 828 | +36 |
| `INODE_SEND` | 36 | 36 | 동일 |
| 명시적 NOP 위치 | 36 | 72 | +36 |
| 10회 NOP loop | 0 | 36 | +36 |
| `INODE_INTRT` | 54 | 54 | 동일 |
| `INODE_HALT` | 216 | 216 | 동일 |

즉 전송량이나 interrupt/halt 구조를 바꾼 것이 아니라, producer인 INODE와 consumer인 IMCE 사이의 ordering을 강화하고 SEND 후 간격을 늘린 변경이다.

### 4.3 IMCE 코드 변화

B에서는 `LOAD_LB` 전에 다음 consumer-side handshake가 추가됐다.

```cpp
__builtin_IMCE_SETFLAG(1);
__builtin_IMCE_STANDBY(0, 1);
__builtin_IMCE_SETFLAG(0);
__builtin_IMCE_LOAD_LB(0);
```

18개 `imce.cpp` 합산 결과는 다음과 같다.

| builtin call site | A | B | 변화 |
|---|---:|---:|---:|
| `IMCE_LOAD_LB` | 62 | 62 | 동일 |
| `IMCE_STANDBY` | 0 | 65 | +65 |
| `IMCE_SETFLAG` | 0 | 130 | +130 |
| `IMCE_STEP` | 92 | 92 | 동일 |
| `IMCE_SEND` | 368 | 368 | 동일 |
| `IMCE_STOP` | 18 | 18 | 동일 |

`IMCE_STANDBY` 1개와 `SETFLAG` 2개가 추가되는 source 위치가 65개다. `LOAD_LB`의 정적 위치는 62개지만 일부 생성 loop 구조 때문에 단순히 파일상의 `LOAD_LB` 개수와 handshake 위치 개수가 1:1로 집계되지는 않는다.

### 4.4 instruction binary 크기

#### 전체 크기

| 구분 | A | B | 증가량 |
|---|---:|---:|---:|
| INODE imem | 33,408 B | 34,560 B | +1,152 B (+3.45%) |
| IMCE imem | 41,824 B | 50,816 B | +8,992 B (+21.50%) |
| 합계 | 75,232 B | 85,376 B | +10,144 B (+13.48%) |
| INODE policy | 13,824 B | 13,824 B | 0 |
| IMCE policy | 36,288 B | 36,288 B | 0 |

각 subgraph의 active INODE imem은 모두 `1,856 B -> 1,920 B`, 즉 64 B 증가한다.

Active IMCE imem은 subgraph 종류별로 다음처럼 증가한다.

| function ID | A | B | 증가량 |
|---|---:|---:|---:|
| 0, 3, 9, 12, 30, 33, 36, 39, 42, 45 | 2,720 B | 3,264 B | +544 B |
| 6, 18, 21, 24, 27 | 1,504 B | 1,792 B | +288 B |
| 15, 48, 51 | 2,368 B | 3,072 B | +704 B |

### 4.5 IMCFLOW 계산 의미에 미치는 영향

두 device program은 binary-equivalent가 아니다.

- 동기화 추가는 SEND/LOAD race 가능성을 낮추는 대신 instruction 수와 stall 가능성을 늘린다.
- 10-NOP delay는 전송 사이 timing을 바꾼다.
- `acc_mask=15 -> 1`은 IMCE 누산 설정을 바꾸므로, hardware에서 두 값의 정의에 따라 수치 결과와 성능이 달라질 수 있다.
- checkpoint가 달라 weight/config data도 다르다.

따라서 출력 차이가 있다면 “동기화 코드”, `acc_mask`, checkpoint를 각각 동일하게 맞춘 추가 A/B build 없이는 어느 하나가 원인이라고 결론 내릴 수 없다.

## 5. CPU에서 실행되는 코드 비교

### 5.1 순수 CPU 연산 kernel

`lib_graph_system-lib.tar/codegen/host/src/default_lib1.c`는 두 빌드에서 SHA-256이 완전히 동일하다.

```text
97785ee3b2f7d3fcd6495d28a1ed85b3582fddc1a5464cfeb221d6bbf7d43beb
```

따라서 CPU에서 실행되는 다음 계열의 generated kernel 구현은 변하지 않았다.

- add, multiply, clip, cast
- layout transform, concatenate, split, take
- fused batch normalization
- min/max quantize, bitpack
- CPU conv2d, dense, bias add, ReLU, pooling, flatten

다만 구현 코드가 같다는 뜻이지 입력과 결과까지 같다는 뜻은 아니다. checkpoint parameter와 quantization threshold가 달라 동일 CPU kernel에 전달되는 값이 달라진다.

### 5.2 graph와 function registry

| 파일 | 차이 | 실행 의미 |
|---|---|---|
| `default.graph` | 18개 function `hash` 값 변경 | graph topology와 호출 연결은 동일 |
| `default_lib0.c` | function 선언/registry 배열 순서 변경 | 등록된 function 집합은 동일; graph 실행 순서 변경을 뜻하지 않음 |
| `metadata.json` | export timestamp 변경 | 계산 의미 없음 |
| `default.params` | binary 내용 변경 | checkpoint/constant 값이 다름 |

`default_lib0.c`는 두 쪽 모두 22,844 B, 398 line이며, 관찰된 diff는 IMCFLOW function 등록 순서의 재배열이다.

### 5.3 IMCFLOW host wrapper

18개 `default_lib*.cc`는 CPU에서 실행되며 다음을 담당한다.

- `/dev/uio5`, `/dev/uio4`, reset generator mmap
- IMEM, policy, weight, config, input data MMIO 전송
- PROGRAM/RUN state write
- interrupt 및 IDLE 대기
- interrupt ACK와 `INTR_DONE` 처리
- output MMIO read
- timeout retry와 resource cleanup

전체 wrapper 크기는 다음과 같다.

| 항목 | A | B | 변화 |
|---|---:|---:|---:|
| wrapper 수 | 18 | 18 | 동일 |
| 총 source line | 50,960 | 51,212 | +252, 정확히 파일당 +14 line |
| 총 source bytes | 947,937 B | 969,859 B | +21,922 B |

파일 번호(`default_lib2.cc` 등)와 IMCFLOW function ID의 대응 순서는 두 archive에서 다르다. 따라서 같은 `default_libN.cc`끼리 비교하면 서로 다른 subgraph를 비교할 수 있다. 이 보고서는 파일 번호가 아니라 내부의 `tvmgen_default_imcflow_main_<ID>` symbol로 대응시켰다.

Wrapper의 큰 numeric diff는 다음 세 종류가 섞여 있다.

1. checkpoint 변경으로 인한 embedded constant 배열 값 변경
2. IMCE imem 크기 증가에 따른 뒤쪽 IMEM base address 변경
3. graph traversal ID와 wrapper 출력 순서 변경

### 5.4 interrupt/IDLE 처리의 핵심 차이

정상적으로 UIO interrupt가 전달된 경우 A의 `wait_imcflow_interrupt()`는 UIO fd를 읽고 바로 성공한다.

```cpp
// A
read(fd, &info, sizeof(info));
return 0;
```

호출부의 정상 순서는 다음과 같다.

```text
STATE=RUN write
-> UIO interrupt wait/read
-> interrupt ACK generator write
-> INTR_DONE=1
```

B는 UIO fd를 읽은 뒤에도 `wait_for_idle()`을 호출한다.

```cpp
// B
read(fd, &info, sizeof(info));
return wait_for_idle(npu_pointer);
```

따라서 B의 정상 순서는 실제로 다음과 같다.

```text
STATE=RUN write
-> UIO interrupt wait/read
-> STATE가 IDLE인지 polling
-> interrupt ACK generator write
-> INTR_DONE=1
```

즉 B는 `interrupt -> ACK -> IDLE polling` 순서가 아니라 **`interrupt -> IDLE polling -> ACK` 순서**다.

이 변경의 의도는 generated INODE code가 `INTRT`를 실행한 직후 아직 최종 `HALT`까지 완료하지 않았을 수 있으므로, output MMIO를 읽기 전에 모든 INODE가 idle인지 확인하는 것이다.

#### ACK 전에 IDLE polling하면 deadlock인가?

현재 확인한 top-level RTL만 기준으로 하면 deadlock 조건은 아니다.

- INODE가 기다리는 ACK는 `controller.sv`의 `interrupt_ack_o`다.
- `interrupt_ack_o[i] = ~pending_interrupt[i] & interrupt_req_i[i]`이므로 첫 node interrupt request는 즉시 handshake된다.
- host의 interrupt ACK generator는 별도의 `cpu_interrupt_ack_i` 경로다.
- IMCFLOW RUN -> IDLE 전이는 `all_inode_idle`만 검사하며 `cpu_interrupt_ack_i`를 검사하지 않는다.

따라서 node는 host ACK 전에 `INTRT`를 통과하여 `HALT`할 수 있고, controller는 IDLE로 갈 수 있다. 이 RTL 관계에서는 B 순서가 완료 race를 막는 역할을 한다.

다만 실제 bitstream/integration에서 interrupt ACK generator가 node 진행이나 `all_inode_idle`에 추가로 영향을 준다면 B 순서는 `IDLE waiting <-> ACK waiting` 교착을 만들 수 있다. 이 경우 필요한 순서는 다음과 같다.

```text
UIO interrupt wait/read
-> interrupt ACK generator write
-> INTR_DONE=1
-> wait_for_idle()
```

따라서 실제 chip waveform에서 다음 신호를 함께 확인해야 한다.

```text
top_ctrl_interrupt_req
top_ctrl_interrupt_ack
interrupt_o
interrupt_ack_i (cpu_interrupt_ack_i)
inode_state[*]
top_ctrl_imcflow_state
```

### 5.5 그 밖의 host wrapper 차이

B에는 다음 진단 및 타입 보강이 추가됐다.

- polling 1,000회마다 8개 control register를 출력하는 `[PROBE]` 로그
- `npu_pointer`와 `int_ack_gen_pointer`를 `volatile uint32_t*`로 선언
- `munmap()` 호출 시 `volatile` 제거를 위한 `(void*)` cast
- UIO interrupt 성공 후에도 `wait_for_idle()` 수행

PROGRAM/RUN write, retry 횟수, ACK write 및 `INTR_DONE=1`의 기본 구조는 유지된다.

## 6. 실행 주체별 최종 비교

| 실행 주체 | 동일한 부분 | 달라진 부분 | 예상 영향 |
|---|---|---|---|
| CPU compute kernel | `default_lib1.c` 전체 | code 차이 없음; parameter/threshold는 다름 | kernel 자체 성능은 유사하나 수치 결과는 달라질 수 있음 |
| CPU graph runtime | topology, function 집합 | function hash와 registry 출력 순서 | 계산 순서 자체의 의미 변화는 확인되지 않음 |
| CPU IMCFLOW wrapper | 전송/PROGRAM/RUN/retry 기본 골격 | interrupt 뒤 IDLE poll, volatile/cast, probe log, constants/address | output read race 감소 가능; ACK-before-IDLE 시스템이라면 교착 위험 |
| INODE | SEND/PU/INTRT/HALT 기본 흐름 | SEND 전 flag handshake, SEND 후 10 NOP | producer/consumer ordering 강화, 실행 시간 증가 가능 |
| IMCE | LOAD/STEP/SEND/STOP 기본 흐름 | LOAD 전 flag handshake, `acc_mask` 변경 | race 감소 가능; instruction 수 및 계산 설정 변화 |
| NPU policy | 최종 360개 binary | 없음 | routing/policy 동작은 동일 |

## 7. 권장 후속 비교

원인을 분리하려면 아래 조건으로 재생성한 최소 2x2 비교가 필요하다.

| 실험 | checkpoint | acc_mask | device sync 변경 | host interrupt 순서 |
|---|---|---:|---|---|
| 기준 | 동일 checkpoint | 15 | off | IRQ -> ACK -> IDLE |
| acc_mask only | 동일 checkpoint | 1 | off | 동일 |
| device sync only | 동일 checkpoint | 동일 | on | 동일 |
| host wait only | 동일 checkpoint | 동일 | 동일 | IRQ -> IDLE -> ACK |

현재 A와 B만으로 확정할 수 있는 사실은 다음과 같다.

1. CPU 연산 kernel source는 동일하다.
2. IMCFLOW instruction binary는 active INODE/IMCE마다 변경됐다.
3. NPU policy binary는 동일하다.
4. B는 ACK 전에 IDLE을 polling한다.
5. checkpoint와 `acc_mask`도 동시에 바뀌었으므로 결과 차이의 단일 원인을 특정할 수 없다.
