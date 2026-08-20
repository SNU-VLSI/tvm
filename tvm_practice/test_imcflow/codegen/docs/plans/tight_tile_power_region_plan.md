# Tight TILE power-region 구현 계획

## 1. 목표

TVM power scope가 `TILE`일 때 각 tile의 bulk CPU↔IMCFlow transfer를 측정에서
제외하고, 실제 accelerator 실행 구간만 최대한 좁게 측정한다.

목표로 하는 generated C 순서는 다음과 같다.

```text
POWER_REGION_BEGIN
SET_RUN_CODE write (START)
interrupt wait       (WAIT)
POWER_REGION_END
```

다음 작업은 모두 TILE power region 밖에 둔다.

- inode PC register 설정
- interrupt arm
- START 이전 ordering barrier
- per-tile compiled block transfer
- per-tile input transfer
- wait 결과 debug 출력
- completion barrier
- interrupt ACK
- `INTR_DONE` write
- retry 판정과 cleanup
- output transfer

`REGION`과 `MODEL` scope의 power-region 위치와 실행 순서는 변경하지 않는다.
TILE scope의 loop는 현재 의도대로 항상 비활성화한다.

## 2. 설계 원칙

### 2.1 기존 general power-region macro 재사용

TILE 전용 power-region macro나 measurement_utils API를 새로 만들지 않는다.
기존 `POWER_REGION_BEGIN/POWER_REGION_END`와 이를 scope 선택에 연결하는 TVM
adapter를 그대로 사용한다.

generated TVM code에서는 기존 호출 형태를 유지한다.

```c
TVM_POWER_REGION_BEGIN(
    IMCFLOW_POWER_SCOPE_TILE,
    "<kernel-name>_tile_<index>");

/* START and WAIT */

TVM_POWER_REGION_END();
```

`TVM_POWER_REGION_BEGIN/END`는 scope 선택만 담당하며, 선택된 경우 내부적으로
measurement_utils의 general `power_region_begin/next/end` API를 사용한다. 따라서
measurement_utils에는 `MODEL`, `REGION`, `TILE` 개념을 추가하지 않는다.

### 2.2 TILE loop policy 유지

TVM runtime은 `IMCFLOW_POWER_SCOPE_TILE`이 선택되면 다음 policy를 강제한다.

```text
loop_enable = 0
min_samples = 0
min_seconds = 0
```

따라서 general macro의 body는 정확히 한 번만 실행된다. 기존
`power_measure_runtime_start()`의 TILE policy normalization을 유지하고 regression
test로 고정한다.

### 2.3 모든 scope에서 동일한 invoke 사용

TILE 측정을 위해 별도의 fenced/unfenced invoke 구현이나 runtime 분기를 만들지
않는다. 모든 scope와 power-disabled 실행에서 동일한 START/WAIT 구현을 사용한다.

```text
START direct MMIO write
interrupt wait
```

START 이전의 PC write, interrupt arm과 ordering barrier는 유지하고, WAIT 이후의
completion barrier, ACK와 `INTR_DONE`도 유지한다. 제거 대상은 START write accessor의
post barrier와 기존 `RUN doorbell visible before completion wait` barrier이다.

scope에 따른 차이는 power-region 경계뿐이다.

- `TILE`: START/WAIT를 TILE power region으로 감싼다.
- `REGION`: 기존 outer REGION power region 안에서 같은 START/WAIT를 실행한다.
- `MODEL`: 기존 outer MODEL power region 안에서 같은 START/WAIT를 실행한다.
- power disabled: power region 없이 같은 START/WAIT를 실행한다.

## 3. 현재 구조와 변경 경계

현재 `generateInvokeCode()`는 다음 작업을 하나의 함수에서 연속 생성한다.

```text
PC write
interrupt arm
SET_RUN_CODE write
post-RUN barrier
interrupt wait
completion barrier
ACK
INTR_DONE
```

또한 tile code는 per-tile input transfer 이후 `generateInvokeCode()` 전체를 TILE
power region으로 감싼다. 그 결과 PC write, interrupt arm, ACK까지 측정에
포함된다.

이를 다음 세 논리 구간으로 분리한다.

```text
invoke_prepare
    PC write
    PC/interrupt ordering barrier
    interrupt arm
    START 이전 barrier

invoke_start_wait
    SET_RUN_CODE write
    interrupt wait

invoke_finalize
    completion barrier
    interrupt ACK
    INTR_DONE write
```

함수 이름은 구현 시 기존 codegen 구조에 맞춰 조정할 수 있지만, 세 구간의 순서와
책임 경계는 위와 같이 고정한다.

## 4. TVM codegen 변경 계획

### 4.1 invoke code 분리

`python/tvm/relay/backend/contrib/imcflow/ext_codegen.py`에서 invoke 생성을 다음과
같이 분리한다.

1. 공통 prepare code를 생성한다.
2. tile invoke에서는 general power-region macro 안에 START와 WAIT만 생성한다.
3. TILE이 선택되지 않은 경우 TVM adapter가 실제 region을 열지 않고 macro body를
   한 번 실행한다.
4. 공통 finalize code를 생성한다.
5. common initial invoke도 같은 barrierless START/WAIT helper를 사용하되 TILE
   power-region macro로 감싸지 않는다.

개념적인 generated code는 다음과 같다.

```c
/* invoke_prepare: always outside TILE power region */
for (int i = 0; i < INODE_NUM; ++i)
  imcflow_mmio_write32(npu_pointer, PC_REG_IDX + i, pc_value);
imcflow_mmio_barrier();
enable_imcflow_interrupt(npu_fd);
imcflow_mmio_barrier();

TVM_POWER_REGION_BEGIN(IMCFLOW_POWER_SCOPE_TILE, tile_region_name);
npu_pointer[STATE_REG_IDX] = SET_RUN_CODE;
_wait_rc = wait_imcflow_interrupt(npu_fd, npu_pointer);
TVM_POWER_REGION_END();

/* invoke_finalize: always outside TILE power region */
imcflow_mmio_barrier();
generate_ack(int_ack_gen_pointer);
imcflow_mmio_barrier();
imcflow_mmio_write32(npu_pointer, INTR_DONE_REG_IDX, 1);
```

실제 코드는 Linux와 bare-metal, interrupt와 polling compile path를 보존해야 한다.
Linux의 TILE wrapper는 scope가 선택되지 않으면 body를 한 번만 통과한다. bare-metal
같이 power runtime을 사용하지 않는 target도 동일한 START/WAIT 순서를 생성한다.

### 4.2 START write에서 post barrier 제거

현재 일반 `imcflow_mmio_write32()` accessor 자체가 write 전후에 barrier를 수행한다.
따라서 기존의 명시적인 `RUN doorbell visible before completion wait` barrier만
삭제해서는 목표 순서가 만들어지지 않는다.

모든 invoke의 `SET_RUN_CODE` write는 post barrier가 없는 전용 emitter를 사용한다.
전용 public macro는 만들지 않고 codegen 내부 helper로 한정한다.

권장 생성 결과는 direct volatile MMIO write이다.

```c
npu_pointer[STATE_REG_IDX] = SET_RUN_CODE;
```

START 이전의 ordering은 power region을 열기 전에 수행한 barrier가 보장한다.
START 이후에는 다음 항목을 넣지 않는다.

- `imcflow_mmio_barrier()`
- `usleep()`
- debug print
- tag set/event
- progress query

이 direct write는 기존 all-MMIO-barrier 실험의 의도적인 단일 예외이며 모든
scope에서 일관되게 사용한다. generated source에는 검색 가능한 주석을 남긴다.

```c
/* IMCFLOW-INVOKE: RUN doorbell intentionally has no post barrier. */
```

### 4.3 tag 처리

TILE power region의 이름에 kernel과 tile index를 포함한다.

```text
<kernel-name>_tile_0
<kernel-name>_tile_1
...
```

START 직전의 tight 구간에 `tile` 또는 `kernel_stage` tag 호출을 넣지 않는다.
region 이름과 request metadata로 tile을 식별한다. general power-region API가 begin
과정에서 생성하는 `region`, policy, `region_iteration=0` metadata는 유지한다.

기존 per-tile `compiled_transfer`, `input_transfer`, `output_transfer` tag는 TILE
session 밖에서 호출되므로 TILE artifact에는 포함되지 않는다. REGION/MODEL scope가
선택된 경우에는 기존 outer region에 그대로 기록된다.

### 4.4 timeout 및 retry

`wait_imcflow_interrupt()`가 성공, timeout 또는 error를 반환하더라도 먼저
`POWER_REGION_END()`를 실행한다. 이후에 `_wait_rc`를 검사하고 기존 retry 또는
kernel failure 경로로 이동한다.

따라서 다음 경로 모두 active DMM session을 남기지 않아야 한다.

- 정상 interrupt
- polling fallback 성공
- interrupt timeout
- retry 요청
- retry exhaustion

retry를 수행할 경우 새 retry iteration의 tile invoke는 새로운 power region을
연다. retry trace는 region 이름 또는 기존 retry metadata로 구분한다.

## 5. measurement_utils 변경 범위

이번 변경에서는 다음을 추가하지 않는다.

- TILE 전용 macro
- TILE 전용 begin/end 함수
- measurement_utils의 scope 개념
- TILE loop 지원
- nested power region
- protocol command 추가

기존 `POWER_REGION_BEGIN/END`, non-nesting 검사, raw sample 저장, timestamp alignment,
tag materialization을 그대로 사용한다.

현재 `power_region_end()`는 내부 tag clear 이후 DMM `STOP_BEGIN`을 전송한다. 이번
계획은 우선 generated C의 경계를 `WAIT` 직후 `POWER_REGION_END()` 호출로 좁히는
것까지를 범위로 한다. REGION 동작과 general API semantics를 유지하기 위해 END
내부 순서는 변경하지 않는다.

검증 결과 WAIT 반환부터 실제 DMM freeze까지의 지연이 무시할 수 없을 정도로 크면,
이는 별도의 general power-region 종료 최적화로 다룬다. 그 변경은 TILE 전용 API가
아니라 모든 사용자가 선택할 수 있는 general API 개선으로 설계한다.

## 6. 테스트 계획

### 6.1 Codegen unit test

generated C에서 다음 순서를 문자열 위치와 구조로 검사한다.

```text
per-tile input transfer
PC writes
interrupt arm
pre-START barrier
TILE POWER_REGION_BEGIN
SET_RUN_CODE direct write
wait_imcflow_interrupt
POWER_REGION_END
completion barrier
ACK
INTR_DONE
output transfer
```

추가 assertion은 다음과 같다.

- START와 WAIT 사이에 barrier, sleep, debug print, tag 호출이 없다.
- WAIT와 `POWER_REGION_END` 호출 사이에 retry check, ACK, debug print가 없다.
- 모든 invoke START는 generic barrier-protected write accessor를 사용하지 않는다.
- REGION/MODEL/power-disabled 경로에도 START 이후 barrier가 생성되지 않는다.
- common initial invoke에는 TILE power region이 생기지 않는다.

### 6.2 Runtime unit test

- scope `TILE`에서 policy가 항상 `(loop=0, min_samples=0,
  min_seconds=0)`인지 검사한다.
- TILE macro body가 정확히 한 번 실행되는지 검사한다.
- REGION/MODEL 선택 시 TILE wrapper가 실제 region을 열지 않고 body만 한 번
  통과하는지 검사한다.
- timeout에서도 END가 한 번 호출되고 active region이 남지 않는지 검사한다.

### 6.3 기존 regression test

- measurement_utils C API와 protocol test를 그대로 통과시킨다.
- power-disabled generated code에서도 START 직후 WAIT가 이어지는지 확인한다.
- REGION/MODEL scope의 기존 begin/end 위치가 유지되는지 확인한다.
- host ARM binary build를 수행한다.

### 6.4 Board/DMM 검증

1. master에서 항상 `activate`로 TVM venv를 활성화한다.
2. TVM과 measurement_utils revision을 master, board, meas-2 사이에 동기화한다.
3. meas-2에서는 `imcflow` conda 환경의 measurement daemon을 사용한다.
4. ResNet sample 0을 `scope=TILE`, loop disabled로 실행한다.
5. region1의 tile 0, 1, 2에 각각 별도 artifact가 생성되는지 확인한다.
6. 각 artifact에서 bulk transfer tag/sample이 없는지 확인한다.
7. generated debug marker로 PC/interrupt 준비가 BEGIN 전에 수행됐는지 확인한다.
8. START 이후 interrupt wait가 정상 완료되고 deadlock이 없는지 확인한다.
9. inference output과 baseline 정확도가 동일한지 확인한다.
10. DMM raw data, `summary.json`, `tags.jsonl`, `power_trace.png`를 확인한다.

## 7. 완료 조건

다음을 모두 만족하면 구현을 완료한 것으로 본다.

- TILE scope만 `BEGIN → START → WAIT → END` 경계를 사용한다.
- TILE에서 loop는 항상 disabled이다.
- 모든 scope에서 START와 WAIT 사이에 post-RUN barrier가 없다.
- PC write, interrupt arm, ACK, INTR_DONE, input/output transfer는 TILE power region
  밖에 있다.
- TILE 전용 power-region macro/API가 추가되지 않는다.
- REGION/MODEL의 power-region 경계와 전체 control flow가 보존된다.
- timeout/retry에서도 DMM session이 정상 종료된다.
- board ResNet 실행이 deadlock 없이 완료되고 결과 정확도가 유지된다.
