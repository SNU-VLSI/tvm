# Power event/tag simplification plan

## 1. 목표

Power trace의 tag를 다음 두 종류로 명확히 분리한다.

1. **Event**
   - 특정 시점에 label을 붙이는 일회성 marker이다.
   - 예: `get_issue`, `tile_start`, `tile_end`.
   - active tag state를 변경하지 않으며 `tag_state_id`에도 영향을 주지 않는다.
2. **State tag**
   - 사용자가 명시적으로 설정하는 일반적인 key/value 상태 정보이다.
   - 기존 `set`/`clear` 기능은 유지한다.
   - measurement runtime이 region 이름이나 loop policy를 자동으로 state tag에 넣지는 않는다.

또한 power 측정과 IMCFLOW 실행의 latency-sensitive path에서 디버깅 및 진행 상황 출력을 제거한다. 정상 측정 경로에서는 timestamp 획득, 필요한 상태 갱신 및 하드웨어 제어 외의 작업을 최소화한다.

## 2. 최종 API 의미

### 2.1 Event 예약 key

`event`를 특별한 예약 key로 정의한다. 다음 두 호출은 동일한 의미를 갖게 한다.

```c
power_tag_set("event", "tile_start");
power_tag_event("tile_start");
```

두 호출 모두 일반 `TAG_SET`이 아니라 `TAG_EVENT`로 처리한다.

- active tag map을 변경하지 않는다.
- `tag_state_id`를 새로 만들지 않는다.
- event label과 호출 시점의 monotonic timestamp를 저장한다.
- 최종 `tags.jsonl`에는 `kind: "event"`, `name: "tile_start"` 형태로 기록한다.

하위 API인 `dmm_tag_set("event", value)`도 같은 규칙을 적용한다. Measurement server에서 `TAG_SET`의 key로 `event`가 직접 들어오는 경우에는 protocol misuse로 검출하여 상태 tag로 저장되지 않게 한다.

`power_tag_event()`와 `dmm_tag_event()`는 명확성과 기존 코드 호환성을 위해 유지한다.

### 2.2 일반 state tag

사용자가 명시적으로 사용하는 일반 key/value API는 유지한다.

```c
power_tag_set("phase", "compute");
power_tag_clear("phase");
```

이 정보는 기존처럼 해당 시점 이후 sample의 state에 적용되며 `tag_state_id` 생성에 사용된다. 단, `event` key는 이 동작의 예외이다.

## 3. Event hot-path 최적화

현재 `dmm_tag_event()`는 board timestamp를 얻은 뒤 TCP frame을 즉시 전송한다. `tile_start` 직전에 이를 호출하면 TCP 전송 시간이 실제 IMCFLOW START보다 앞에 추가되므로 실행 경로를 불필요하게 지연시킬 수 있다.

이를 다음 구조로 변경한다.

```text
power_tag_event(label)
  -> 함수 진입 직후 board monotonic timestamp 획득
  -> 고정 크기 local event queue에 timestamp와 label 저장
  -> timed body가 끝난 뒤 measurement server로 전송
```

구현 원칙은 다음과 같다.

- hot path에서 heap allocation을 하지 않는다.
- hot path에서 `printf`, `fprintf`, `snprintf`를 호출하지 않는다.
- 고정 크기 event array에 bounded copy 방식으로 label을 저장한다.
- TCP 전송 시점이 아니라 최초 API 호출 시점의 timestamp를 protocol frame에 넣는다.
- queue overflow나 잘못된 label은 error status에 기록하고 hot path에서 출력하지 않는다.
- queued event는 `power_region_end()`가 session stop 요청을 보내기 전에 flush한다.
- 초기 구현은 기존 `TAG_EVENT` frame을 연속 전송한다. event 수가 많아질 경우 별도 `TAG_EVENT_BATCH` protocol은 후속 최적화로 검토한다.

TILE 실행 순서는 다음처럼 유지한다.

```text
POWER_REGION_BEGIN
event("tile_start")
IMCFLOW START
interrupt WAIT
interrupt ACK / INTR_DONE
event("tile_end")
POWER_REGION_END
  -> queued event 전송
  -> measurement session 종료
```

`tile_start`와 `tile_end` timestamp 획득은 각각 START 직전과 interrupt completion handshake(ACK/INTR_DONE) 직후에 배치한다. ACK와 INTR_DONE 사이에는 instrumentation을 넣지 않는다. MMIO barrier의 존재 여부와 관계없이 event API 자체는 추가 barrier를 만들지 않는다.

## 4. GET event 기록

GPIB GET은 measurement server가 PyVISA를 통해 실제로 발행한다. 따라서 board에서 GET 시점을 추정하거나 TCP 왕복 시간을 사용하지 않고 measurement server가 실제 VISA 호출 주변에서 직접 event를 기록한다.

기본 marker는 다음과 같다.

```text
get_issue
```

이 event의 timestamp는 VISA GET 호출 직전에 measurement server monotonic clock으로 획득한다. 여러 DMM 또는 GPIB bus로 인한 GET 발행 구간도 보존할 필요가 있으면 다음 bracket도 metadata에 유지한다.

```text
get_issue_begin
get_issue_end
```

GET event는 board clock 변환을 거치지 않는다. 기존 GET timing bracket 및 DMM metadata를 사용하여 sample alignment uncertainty를 계산한다.

### 4.1 Clock domain과 timestamp 책임

시간은 board, measurement server, DMM의 세 clock domain으로 나눈다. Master server는 실행을 지시하고 결과를 수집하지만 정밀한 power timeline 계산에는 참여하지 않는다.

| 주체 | 사용하는 시간 | 기록 대상 | 기준 clock | 공통 timeline 변환 |
|---|---|---|---|---|
| Master server | 실행/파일 생성 시각 | 실험 시작·종료와 결과 이력 | Master wall/monotonic clock | 변환하지 않음 |
| Board | `client_monotonic_ns` | `tile_start`, `tile_end`, timeout 등의 TVM event | PetaLinux `CLOCK_MONOTONIC` | board↔measurement server clock sync 결과 사용 |
| Measurement server | `server_monotonic_ns` | GET 발행, TCP 수신, session 시작·종료 | Measurement server `CLOCK_MONOTONIC` | 공통 timeline 기준으로 그대로 사용 |
| DMM | reading metadata timestamp | 각 current sample의 상대 측정 시각 | DMM 내부 sample clock | GET timing bracket과 DMM metadata로 정렬 |

Event와 sample별 timestamp 책임은 다음과 같다.

| Event/sample | Timestamp 기록 주체 | 기록 시점 | 용도 |
|---|---|---|---|
| `get_issue` | Measurement server | PyVISA GET 호출 직전 | DMM trigger anchor |
| GET return | Measurement server | PyVISA GET 호출 직후 | GET 발행 uncertainty bracket |
| `tile_start` | Board | IMCFLOW START MMIO write 직전 | TILE 실행 시작 경계 |
| `tile_end` | Board | interrupt ACK/INTR_DONE completion handshake 직후 | TILE 실행 종료 경계 |
| TCP event receive | Measurement server | event frame 수신 시점 | 검증 및 clock conversion fallback |
| Current sample | DMM | DMM reading 획득 시점 | 실제 current trace |
| Region/session end | Measurement server | DMM fetch/stop 처리 시점 | 측정 session 수명 관리; `tile_end`와 구분 |

최종 plot은 measurement server monotonic clock을 공통 축으로 사용한다.

| 원본 timestamp | 공통 timeline 계산 |
|---|---|
| Measurement server event | 그대로 사용 |
| Board event | board timestamp에 clock-sync offset 적용 |
| DMM sample | GET anchor에 DMM relative reading timestamp 적용 |
| Plot 상대 시간 | 공통 timeline에서 첫 sample 또는 GET 기준을 차감 |

불확실성은 원인별로 분리하여 결과 metadata에 남긴다.

| 불확실성 | 원인 | 적용 대상 |
|---|---|---|
| Board clock-sync uncertainty | TCP 왕복과 clock offset 추정 오차 | `tile_start`, `tile_end` |
| GET issue uncertainty | PyVISA GET 호출 전후 bracket | DMM 측정 시작 anchor |
| DMM sample timing uncertainty | DMM timestamp 해상도와 trigger 지연 | 각 current sample |
| Sample alignment uncertainty | event와 가장 가까운 sample 사이의 간격 | plot의 event sample index |
| TCP 전송 지연 | queued event flush와 network 지연 | 원본 board timestamp를 보존하므로 event 시각에는 포함하지 않음 |

정확한 정렬에 필요한 핵심 관계는 board↔measurement server clock offset과 measurement server GET↔DMM reading timestamp 관계이다. Master server 시간은 결과 이력에만 사용한다.

## 5. 자동 region/policy state tag 제거

`3rdparty/measurement_utils/capi/power_region.c`의 `power_region_begin()`, `begin_iteration()`, `power_region_end()`에서 다음 자동 state tag의 set/clear를 모두 제거한다.

```text
region
region_loop_enable
region_min_samples
region_min_seconds
region_iteration
```

이에 따라 다음 hot-path 작업도 제거한다.

- policy 값을 문자열로 만드는 `snprintf`
- region 시작 시 연속적인 `TAG_SET` TCP 전송
- iteration 시작마다 실행하던 `region_iteration` 전송
- region 종료 시 수행하던 자동 `TAG_CLEAR` 전송

다음 정보는 tag가 아닌 기존 configuration/session metadata에 둔다.

- region name
- loop enable 여부
- minimum sample 수
- minimum seconds
- scope
- DMM resolved configuration

실제 loop 제어에 필요한 `power_region_context_t::iteration_count`는 내부 변수로 유지한다. 실제 iteration 수를 결과에 남겨야 한다면 state tag를 다시 사용하지 않고 region 종료 metadata에 기록한다. Measurement server summary가 `region_iteration` tag를 통해 iteration 수를 계산하는 기존 의존성도 이 metadata를 사용하도록 변경한다.

## 6. 자동 event 정리

정책 정보나 반복 상태를 설명하기 위한 자동 state tag는 제거하지만 실제로 발생한 단일 시점은 event로 표현할 수 있다.

기본 성공 경로에서는 다음 marker만 사용한다.

```text
get_issue
tile_start
tile_end
```

다음과 같은 비정상 상황은 실제로 발생했을 때만 event로 남길 수 있다.

```text
region_loop_insufficient_capacity
region_loop_progress_stalled
retry
sample_timeout
```

`region_loop_disabled`처럼 configuration에서 이미 알 수 있고 매번 자명하게 발생하는 event는 제거한다. `region_loop_min_reached`도 기본 trace에 꼭 필요하지 않다면 제거하고, 필요할 경우 loop 종료 metadata에 reason으로 기록한다.

## 7. 디버그 및 진행 출력 제거

### 7.1 TVM IMCFLOW codegen

`python/tvm/relay/backend/contrib/imcflow/ext_codegen.py`에서 생성되는 latency-sensitive C 코드의 디버그 출력을 제거한다.

- `DEBUG_PRINT_INSTRUMENT` 환경변수 처리
- `IMCFLOW_DEBUG_PRINT` macro와 모든 호출
- MMIO barrier begin/end 출력
- interrupt enable/wait/ACK 전후 출력
- tensor transfer loop progress 출력
- kernel 호출 및 TILE 시작 출력
- polling 시작, 성공 및 주기적 상태 출력

MMIO barrier, interrupt wait, retry 및 timeout 같은 실제 제어 동작은 유지하고 출력만 제거한다.

### 7.2 Power runtime과 measurement_utils C API

다음 정상 경로에서는 stdout/stderr 출력을 하지 않는다.

- runtime initialization 성공
- power region begin/end 성공
- event/tag set/clear 성공
- progress query 성공
- GET 발행 성공
- event queue flush 성공

오류는 함수 반환값과 `*_last_error()`에 저장한다. 초기화 실패, 최종 timeout 또는 session 종료 실패처럼 사용자가 실행 결과를 판단하는 데 필요한 오류는 측정 hot path 밖의 최상위 caller에서 한 번만 출력한다.

Measurement server의 Python logging도 정상 GET마다 출력하지 않도록 debug level로 내리거나 제거한다. 오류와 최종 session summary log는 유지한다.

## 8. TVM codegen tag 정리

`ext_codegen.py`에서 다음 자동 state tag 생성을 제거한다.

```text
kernel
kernel_stage
tile
retry_attempt
```

관련 `TAG_CLEAR` 생성도 함께 제거한다. 일반 state tag API 자체는 measurement_utils의 public API로 계속 제공한다.

TILE scope에서는 다음 event만 정확한 실행 경계에 삽입한다.

```text
event("tile_start")
event("tile_end")
```

retry와 timeout은 정상 상태 tag가 아니라 실제 발생 시점의 event로만 기록한다.

MODEL 및 REGION scope에도 자동 metadata state tag를 만들지 않는다. 추후 필요한 marker는 scope별로 명시적인 event 이름을 정의하여 추가한다.

## 9. Plot 및 결과 형식

Event는 state plot의 y축에 포함하지 않는다.

- `tag_state_id`는 일반 state tag의 set/clear만 반영한다.
- 일반 state tag가 없다면 전체 sample의 `tag_state_id`는 `0`이다.
- event는 current/power plot 위의 수직선으로 표시한다.
- 수직선 가까이에 `get_issue`, `tile_start`, `tile_end` label을 표시한다.
- event label이 겹칠 경우 위아래 위치를 교차하거나 별도의 event row를 사용한다.
- raw event 정보와 timing uncertainty는 `tags.jsonl`에 계속 저장한다.

Event는 가장 가까운 sample index도 계산하여 저장하되, 원본 timestamp와 uncertainty를 보존한다. 가장 가까운 sample에 정렬된 결과를 정확한 event timestamp로 오해하지 않도록 plot과 JSON에서 두 값을 구분한다.

## 10. 구현 순서

### Phase 1: measurement_utils C API

1. `event` 예약 key를 정의한다.
2. `power_tag_set("event", value)`와 `dmm_tag_set("event", value)`를 event 경로로 연결한다.
3. 고정 크기 local event queue를 구현한다.
4. event 호출 시 timestamp를 즉시 저장하고 region 종료 전에 flush한다.
5. 자동 region/policy/iteration state tag set/clear를 제거한다.
6. loop 결과를 tag 대신 명시적인 종료 metadata로 전달한다.
7. power C API 정상 경로의 디버그 및 진행 출력을 제거한다.

### Phase 2: measurement server

1. `event`가 state map에 들어가지 않도록 protocol validation을 강화한다.
2. GET 발행 직전에 server-side `get_issue` event를 기록한다.
3. loop summary가 `region_iteration` tag에 의존하지 않도록 변경한다.
4. plot에 event 수직선과 짧은 label을 표시한다.
5. 정상 GET 및 tag 수신의 상세 logging을 제거하거나 debug level로 조정한다.

### Phase 3: TVM

1. `ext_codegen.py`의 자동 state tag emit을 제거한다.
2. TILE scope의 START 직전과 interrupt ACK/INTR_DONE completion handshake 직후에 event를 삽입한다.
3. generated debug print support와 호출을 제거한다.
4. power runtime의 phase/sample 자동 state tag가 계속 필요한지 확인하고, 자동 metadata라면 제거한다.
5. 정상 power 실행 경로의 출력과 문자열 formatting을 제거한다.

### Phase 4: 문서 및 호환성 정리

1. power measurement quickstart의 tag 설명을 event/state 모델로 변경한다.
2. 기존 `power_tag_event()` 사용자는 변경 없이 동작하게 한다.
3. 기존 결과의 `region_*` state tag는 과거 형식으로 계속 읽을 수 있게 하되 새 결과에서는 생성하지 않는다.

## 11. 테스트 계획

### 11.1 measurement_utils 단위 테스트

- `event`가 `TAG_EVENT`로 직렬화되는지 확인한다.
- `power_tag_event()`와 `power_tag_set("event", ...)` 결과가 동일한지 확인한다.
- event를 여러 개 넣어도 active state map과 `tag_state_id`가 변하지 않는지 확인한다.
- 일반 key/value set/clear가 기존대로 동작하는지 확인한다.
- event timestamp가 flush 시점이 아니라 API 호출 시점인지 확인한다.
- event queue overflow가 출력 없이 명확한 error status를 반환하는지 확인한다.
- region begin/end에서 자동 state tag frame이 하나도 발생하지 않는지 확인한다.
- nested power region 검사가 기존대로 동작하는지 확인한다.

### 11.2 Measurement server 테스트

- GET 호출 시 `get_issue`가 정확히 한 번 생성되는지 확인한다.
- GET event가 server clock 기준이며 적절한 uncertainty를 갖는지 확인한다.
- event가 `tags.jsonl`에는 기록되지만 state definitions에는 포함되지 않는지 확인한다.
- 일반 tag set/clear만 sample의 `tag_state_id`를 변경하는지 확인한다.
- region summary가 `region_iteration` tag 없이 생성되는지 확인한다.
- 이전 형식의 결과를 읽고 plot하는 호환성이 유지되는지 확인한다.

### 11.3 TVM 생성 코드 테스트

생성된 C 코드에 다음 문자열이 없는지 검사한다.

```text
DEBUG_PRINT_INSTRUMENT
IMCFLOW_DEBUG_PRINT
region_loop_enable
region_min_samples
region_min_seconds
region_iteration
kernel_stage
retry_attempt
```

TILE path에는 다음 순서가 존재하는지 검사한다.

```text
tile_start event
IMCFLOW START
interrupt WAIT
interrupt ACK / INTR_DONE
tile_end event
POWER_REGION_END
```

### 11.4 실제 board/DMM 검증

1. 기존 ResNet TILE power configuration으로 compile한다.
2. measurement server와 board에 최신 코드를 sync한다.
3. TILE power 측정을 실행한다.
4. 각 region에서 다음을 확인한다.
   - sample 수 및 raw current data가 정상이다.
   - 자동 state tag가 없으므로 기본 `tag_state_id`가 전체에서 `0`이다.
   - `get_issue`, `tile_start`, `tile_end`가 plot에 수직선으로 보인다.
   - `tile_start`와 `tile_end`가 IMCFLOW 실행 구간을 올바르게 감싼다.
   - event timestamp uncertainty가 결과 JSON에 남는다.
5. instrumentation 적용 전후의 실행 시간을 비교해 event 기록 overhead를 측정한다.

## 12. 완료 조건

- `event`는 일회성 timestamp marker로만 동작한다.
- event는 `tag_state_id`를 변경하지 않는다.
- `region`, loop policy, iteration 등의 자동 state tag가 생성되지 않는다.
- 사용자가 직접 호출하는 일반 key/value set/clear 기능은 유지된다.
- GET, tile 시작 및 tile 끝을 event label로 식별할 수 있다.
- power 및 IMCFLOW latency-sensitive 정상 경로에 디버그 `printf`가 없다.
- ResNet TILE power 측정과 raw data/plot 생성이 정상적으로 완료된다.
