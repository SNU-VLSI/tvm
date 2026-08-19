# Power region 반복 측정 구현 계획

## 1. 목표

`POWER_REGION_BEGIN`과 `POWER_REGION_END` 사이의 C body를 필요하면 반복하여,
짧은 accelerator operation도 DMM 분석에 충분한 sample 수와 측정 시간을 확보한다.

```c
POWER_REGION_BEGIN(
    "resnet_qconv_3",
    1,       /* LOOP_ENABLE */
    2000,    /* MIN_SAMPLES */
    0.100    /* MIN_SECONDS */
);

run_region_body();

POWER_REGION_END();
```

macro는 `measurement_utils`가 제공하고, TVM generated Linux C code도 같은 API를
사용한다. macro 한 쌍은 항상 하나의 실제 power region을 열고 닫으며, body
iteration 사이에는 DMM을 재시작하지 않는다. 각 iteration은 비동기 tag로
구분한다.

`measurement_utils`에는 `scope` 개념을 두지 않는다. 모든 요청은 동일한
non-nested power region이다. `MODEL`, `REGION`, `TILE`은 TVM이 이 API를 어느
위치에서 몇 번 호출할지를 결정하기 위한 TVM-only policy이다.

## 2. 동작 규칙

argument는 `NAME`을 포함하여 다음 순서로 정의한다.

```text
POWER_REGION_BEGIN(
    NAME,
    LOOP_ENABLE,
    MIN_SAMPLES,
    MIN_SECONDS)
```

각 식은 정확히 한 번만 평가한다.

| Argument | 의미 |
|---|---|
| `NAME` | region artifact와 tag에 사용할 이름 |
| `LOOP_ENABLE` | `0`이면 body를 정확히 한 번 실행하고, `1`이면 조건에 따라 반복 |
| `MIN_SAMPLES` | 모든 rail이 확보해야 하는 최소 acquired sample 수. `0`이면 sample 최소 조건 없음 |
| `MIN_SECONDS` | DMM GET 시점부터 확보할 최소 시간. `0`이면 시간 최소 조건 없음 |

반복 규칙은 다음으로 고정한다.

1. body는 항상 최소 한 번 실행한다.
2. `LOOP_ENABLE=0`이면 최소 조건과 관계없이 한 번 실행한 뒤 region을 끝낸다.
3. 최소 조건은 **AND**이다. 지정된 `MIN_SAMPLES`와 `MIN_SECONDS`를 모두 만족해야
   정상 종료한다.
4. 여러 rail에서는 region 시작 이후 증가한
   `min(per_rail_sample_count)`가 `MIN_SAMPLES`에 도달해야 한다. 따라서 모든
   rail이 최소 sample 수를 만족한다.
5. 시간은 power region의 DMM GET 시점부터 measurement server monotonic clock으로
   계산한다.
6. `MIN_SAMPLES=0`, `MIN_SECONDS=0`이면 첫 body 실행 후 즉시 종료한다.
7. 별도의 `MAX_*` macro argument나 정상 종료 조건은 두지 않는다. 대신 요청의
   DMM buffer 용량으로 최소 조건 달성 가능성을 시작 전에 검사하고, progress
   정체·buffer 소진·통신 timeout은 정상 최대 종료가 아니라 명시적인 error로
   처리한다.
8. 새 power-region API의 END는 항상 현재까지의 acquisition을 즉시 freeze하고
   회수한다. 즉 measurement_utils API 자체에는 `mode` 선택도 두지 않는다.
   TVM 설정에서는 backward compatibility를 위해 `mode`를 읽되 `now`만 허용한다.

`MIN_SAMPLES`는 finalize 후 ambiguity를 제거한 sample 수가 아니라, 실행 중 DMM
reading memory에 실제로 추가된 sample 수를 의미한다. tag 경계 ambiguity는
기존과 같이 finalize 단계에서 별도로 계산한다. 마지막 body 때문에 최소값을
초과한 sample/time은 `minimum_overshoot` metadata로 기록한다.

## 3. 책임 경계

### measurement_utils

`measurement_utils`가 아는 개념은 다음뿐이다.

- 한 번에 하나만 활성화할 수 있는 `power_region_begin/end`
- active power region에 적용하는 비동기 `tag_set`, `tag_clear`, `tag_event`
- active region의 live sample/time progress
- region 종료 시 raw sample, timestamp alignment, tag state와 summary 생성

`scope`, TVM model/kernel/tile/sample, `MODEL`/`REGION`/`TILE` 정책은 알지 못한다. request와
protocol에서도 `scope`를 제거한다.

### TVM

TVM의 `scope`는 API 배치 정책이다.

- `MODEL`: 첫 region의 reset/warmup 다음부터 마지막 region의 tile/output 처리가
  끝날 때까지 model 실행 전체를 power region 하나로 감싼다.
- `REGION`: 각 generated region kernel의 reset/warmup 다음부터 해당 region의 모든
  tile/output 처리가 끝날 때까지 power region 하나를 만든다.
- `TILE`: 각 tile의 input transfer 다음, NPU invoke 직전부터 invoke 직후이자 output
  read 전까지 power region 하나를 만든다.

세 정책 모두 measurement_utils에는 동일한 begin/tag/end 호출로 보인다. TVM은
한 실행에서 정확히 하나의 scope만 선택하여 nested region을 만들지 않는다.

## 4. Macro와 C API 설계

### 4.1 Public header

`measurement_utils/capi/power_region.h`를 새로 만들고 다음 public type/function을
둔다. low-level TCP 함수는 계속 `dmm_measure.[ch]`가 담당한다.

```c
typedef struct {
    int loop_enable;
    uint64_t min_samples;
    double min_seconds;
} power_region_policy_t;

typedef struct {
    uint32_t abi_version;
    power_region_policy_t policy;
    uint64_t iteration_count;
    /* lifecycle, progress, stop reason 등 나머지 내부 상태 */
} power_region_context_t;

int power_region_runtime_init(const char *request_json_path);
int power_region_runtime_init_from_env(void);
int power_region_runtime_shutdown(void);
int power_region_begin(power_region_context_t *ctx,
                       const char *name,
                       power_region_policy_t policy);
int power_region_next(power_region_context_t *ctx);
int power_region_end(power_region_context_t *ctx);
int power_region_last_status(void);
const char *power_region_last_error(void);
```

macro는 내부 context를 만들고 `power_region_next()`가 참인 동안 body를 실행한다.
첫 `next()`는 body를 시작하고, 이후 호출은 직전 iteration 이후의 progress를
조회한 다음 반복 여부를 결정한다. `POWER_REGION_END()`는 어떤 종료 경로에서도
session finalize를 수행한다.

개념적인 expansion은 다음과 같다.

```c
#define POWER_REGION_BEGIN(NAME, LOOP, MIN_N, MIN_S) \
    do { \
        power_region_context_t _power_region_ctx; \
        power_region_begin(&_power_region_ctx, (NAME), \
            (power_region_policy_t){(LOOP), (MIN_N), (MIN_S)}); \
        while (power_region_next(&_power_region_ctx)) {

#define POWER_REGION_END() \
        } \
        (void)power_region_end(&_power_region_ctx); \
    } while (0)
```

실제 macro에는 argument type/range 검사, disabled/error state, compiler warning 방지
및 고유한 내부 symbol 처리를 추가한다.

### 4.2 TVM과 독립적인 사용

macro 구현은 TVM header, symbol, 환경변수에 의존하지 않는다. 일반 C application은
`dmm_measure.c`와 `power_region.c`를 link하고 measurement endpoint를 설정한 뒤
명시적으로 runtime을 초기화할 수 있다.

```c
setenv("POWER_MEASUREMENT_HOST", "192.0.2.10", 1);
setenv("POWER_MEASUREMENT_PORT", "9910", 1);

if (power_region_runtime_init(
        "/tmp/power_request.json") != 0) {
    /* handle measurement setup error */
}

POWER_REGION_BEGIN("my_accelerator_call", 1, 2000, 0.1);
my_accelerator_call();
POWER_REGION_END();

if (power_region_runtime_shutdown() != 0) {
    /* handle finalize error */
}
```

`power_region_runtime_init_from_env()`는 standalone script가 request path를
환경변수로 넘길 수 있게 하는 convenience API이다. 이것도
`measurement_utils`에서 정의하며 `IMCFLOW_*` 이름을 필수로 요구하지 않는다.
권장 generic 환경변수는 다음과 같다.

```text
POWER_REGION_REQUEST=/path/to/request.json
POWER_MEASUREMENT_HOST=<server>
POWER_MEASUREMENT_PORT=9910
```

초기화 API는 request와 endpoint, protocol state만 준비한다. 실제 DMM acquisition은
항상 `POWER_REGION_BEGIN`에서 시작하고 `POWER_REGION_END`에서 즉시 freeze/finalize
한다. `shutdown()`은 active region이 없는지 검사하고 client resource를 정리한다.

TVM의 `power_measure_runtime`은 이 generic API의 adapter일 뿐이며 standalone
사용자가 TVM runtime을 link할 필요가 없다.

### 4.3 Control-flow 및 non-nesting 규칙

- instrumentation이 비활성화된 경우에도 body는 한 번 실행한다.
- runtime/region 시작에 실패한 경우 body를 실행하지 않고 오류 status를 남긴다.
  standalone caller는 `power_region_last_status()`를 확인하고, TVM generated code는
  이를 기존 kernel failure로 전환한다.
- body의 `continue`는 iteration 종료/progress 검사로 이동하도록 지원한다.
- body의 `break`는 반복을 끝내고 region을 finalize하며 종료 이유를 `user_break`로
  기록한다.
- `return`, region 밖으로 나가는 `goto`, `longjmp`는 END cleanup을 우회하므로
  macro body 안에서 금지한다. TVM codegen의 기존 early return은 cleanup label과
  status variable을 사용하는 형태로 바꾼다.
- process 안에는 active power region이 최대 하나만 존재한다. begin은 network
  요청 전에 process-global active flag를 검사하고, 이미 활성화되어 있으면
  `POWER_REGION_ERR_NESTED`를 반환한다. region stack은 만들지 않는다.
- nested BEGIN이 실패하면 그 macro/function의 body는 실행하지 않는다. 기존 outer
  region은 계속 active 상태로 유지하여 caller가 정상적으로 tag를 보내고 END할 수
  있게 한다. nested 오류가 outer region을 암묵적으로 종료해서는 안 된다.
- active flag와 lifecycle은 mutex로 보호하여 다른 thread의 동시 begin도 nested로
  거부한다. END는 자신이 연 region만 닫을 수 있다.
- measurement server도 connection별 active region과 DMM reservation을 검사하여
  잘못된 client나 race에 대한 2차 방어를 한다.
- END 없이 새 BEGIN, active region 없이 END, region 중 runtime shutdown은 모두
  명시적인 오류이다. shutdown은 active region을 best-effort abort한 뒤 실패를
  반환한다.
- signal/process crash는 기존 `atexit`만으로 완전히 처리할 수 없으므로 server
  connection close 시 abort artifact를 남기는 기존 방식을 유지한다.

### 4.4 직접 사용하는 비동기 tag와 iteration tag

public API는 TVM wrapper 없이 직접 호출할 수 있게 다음 이름으로 정리한다.

```c
int power_tag_set(const char *key, const char *value);
int power_tag_clear(const char *key);
int power_tag_event(const char *name);
```

tag는 active power region에만 적용된다. active region이 없으면 조용히 다른
session에 붙이지 않고 `POWER_REGION_ERR_NO_ACTIVE`를 반환한다. 기존 `dmm_tag_*`
함수는 compatibility wrapper로 유지한 뒤 migration 기간 후 정리한다.

각 body 진입 직전에 다음 tag를 자동으로 보낸다.

```text
region=<NAME>
region_iteration=0,1,2,...
```

종료 시 `region_iteration`을 clear하고 다음 event 중 하나를 기록한다.

```text
region_loop_min_reached
region_loop_disabled
region_loop_user_break
region_loop_insufficient_capacity
region_loop_progress_stalled
region_loop_error
```

이렇게 하면 한 개의 연속 DMM trace 안에서 iteration별 평균 전류/power도 기존
tag-state 분석으로 계산할 수 있다.

## 5. 실제 sample/time progress 측정

### 5.1 먼저 수행할 hardware feasibility test

GPIB3 DMM에서 acquisition 중 `DATA:POIN?`를 호출해도 acquisition이나 reading
metadata 저장을 방해하지 않는지 먼저 검증한다.

1. `ssh meas-2`, `conda activate imcflow` 환경에서 GPIB3를 설정하고 GET을 보낸다.
2. 서로 다른 interval로 `DATA:POIN?`를 주기적으로 조회한다.
3. count가 단조 증가하고 조회가 acquisition을 멈추거나 buffer를 비우지 않는지
   확인한다.
4. polling 유무에 따른 sample interval, 누락 sample, raw metadata, query latency를
   비교한다.
5. 마지막 `ABORt`와 raw CSV upload 결과가 buffer count와 정확히 일치하는지
   확인한다.

이 검증을 통과하면 `DATA:POIN?` 값을 authoritative live count로 사용한다. 장비가
acquisition 중 조회를 안정적으로 지원하지 않으면 조용히 추정값으로 바꾸지 않고
구현을 중단해 정책을 재검토한다. 시간 기반 반복만 허용하는 fallback은 별도
option으로 명시적으로 추가할 때만 사용한다.

### 5.2 Measurement server API

`DmmManager`에 reading memory를 변경하지 않는 다음 method를 추가한다.

```python
query_current_sample_counts(dmm_names) -> dict[str, int]
```

TCP protocol은 version 4에서 5로 올리고 `POWER_REGION_BEGIN`,
`POWER_REGION_PROGRESS`, `POWER_REGION_END` command를 추가한다. protocol에는
`scope`나 `mode` field를 두지 않는다. BEGIN은 항상 새 DMM acquisition을 configure한
뒤 GET을 보내고, END는 항상 즉시 acquisition을 freeze하고 data를 회수한다.

client와 measurement server에는 active region이 최대 하나만 존재한다.
`POWER_REGION_BEGIN` 수신 시 이미 active region이 있으면 새 GET을 보내지 않고
`POWER_REGION_ERR_NESTED`로 거부한다. connection 단위 검사와 별도로 DMM reservation
단위 검사도 하여 다른 connection이 같은 장비에 region을 중첩하지 못하게 한다.

```text
client -> POWER_REGION_BEGIN <sequence> <name_bytes> <request_bytes>\n
server -> POWER_REGION_BEGIN_RESULT <json_bytes>\n
client -> POWER_REGION_PROGRESS <sequence> <board_monotonic_ns>\n
server -> POWER_REGION_PROGRESS_RESULT <json_bytes>\n
          { ...json payload... }
client -> POWER_REGION_END <sequence> <reason_bytes>\n
server -> POWER_REGION_END_RESULT <json_bytes>\n
```

progress response에는 최소한 다음 값이 들어간다. 모든 count는 새 GET으로 시작한
현재 power region의 직접 count이므로 baseline이나 delta field가 필요 없다.

```json
{
  "region_id": "...",
  "server_monotonic_ns": 0,
  "elapsed_from_get_ns": 0,
  "per_rail_sample_count": {"DMM_GPIB3": 1234},
  "min_sample_count": 1234,
  "per_rail_capacity": {"DMM_GPIB3": 50000}
}
```

시간 기준은 board clock이 아니라 measurement server monotonic clock이다.
region의 시간 원점은 실제 GET을 보낸 직후 기록한 시각이다. 따라서 loop 판단을
위해 board/server clock synchronization이나 기존 session baseline 계산이 필요 없다.

`dmm_measure.[ch]`에는 `dmm_power_region_begin()`,
`dmm_power_region_get_progress()`, `dmm_power_region_end()`를 추가한다. framed payload
길이 상한, region ID, sequence, count와 elapsed의 단조성 및 count가 configured
capacity를 넘지 않는지 검사한다. 이는 public `power_region_*` API의 transport
layer이며 TVM 관련 enum이나 field를 포함하지 않는다.

tag command는 기존 비동기 queue를 사용하되 active `region_id`에만 귀속시킨다.
END 이후 늦게 도착한 tag는 다음 region으로 넘기지 않고 오류로 기록한다.

### 5.3 Polling, capacity와 stall 보호

- 기본적으로 body 한 번이 끝날 때 progress를 한 번 조회한다.
- server/DMM query latency도 region trace에 포함되므로 결과에
  `progress_query_count`와 query latency를 기록한다.
- `MIN_SAMPLES`가 각 rail의 configured capacity보다 크면 body를
  시작하지 않고 `insufficient_capacity`로 실패한다.
- `MIN_SECONDS` 동안 예상되는 sample 수를 계산해 DMM
  capacity 안에서 달성 가능한지 검사한다.
- progress count와 elapsed가 여러 poll 동안 함께 증가하지 않으면 정상 종료하지
  않고 `progress_stalled` error로 session을 abort한다. poll 횟수/시간은 내부
  watchdog이며 public `MAX_*` 조건이 아니다.
- body 하나가 남은 buffer 전체보다 길 수 있으므로 첫 iteration에서
  overflow/truncation이 발생하면 artifact를 `truncated`로 남기고 명확히 실패한다.

## 6. TVM 연동

### 6.1 TVM-only runtime configuration

TVM의 기존 power JSON에는 `scope`, `mode`, optional `region_loop` object를 둔다.
이 세 항목은 TVM이 measurement API를 어디에 배치하고 어떤 body를 반복할지
결정하는 설정이며, measurement_utils request schema에는 포함하지 않는다. 기본값은
기존 동작을 보존하도록 loop disabled이다.

```json
{
  "scope": "REGION",
  "mode": "now",
  "region_loop": {
    "loop_enable": true,
    "min_samples": 2000,
    "min_seconds": 0.1
  }
}
```

`power_request.py`가 TVM policy의 type/range와 DMM capacity를 검증한다. scope의
canonical value는 기존 TVM phase와 같은 대문자 `MODEL`, `REGION`, `TILE`로 한다.
이 세 scope에서 loop policy를 허용하지만 `mode=now`가 아니면 TVM에서 실행 전에
거부한다. 여기서 `now`는 END에서 즉시 freeze하는 새 power-region API를 선택한다는
뜻이며 measurement_utils에 mode 문자열을 전달한다는 뜻이 아니다.
scope를 생략했을 때의 TVM 기본값도 기존 phase와 같은 `REGION`으로 한다.

설정은 명시적으로 두 층으로 분리한다.

1. `tvm_power_policy`: `scope`, `mode`, `region_loop`를 포함하며 TVM runner/runtime과
   codegen adapter만 읽는다.
2. `measurement_request`: DMM, rail, sample interval/count, output과 run metadata만
   포함한다. `scope`, `mode`, TVM kernel 이름은 넣지 않는다.

runner는 loop policy의 숫자와 boolean을 board의 TVM runtime 설정으로 전달한다.
`power_measure_runtime_start()`는 measurement_utils의 generic
`power_region_runtime_init()`을 초기화하지만 scope를 넘기지 않는다. public macro는
명시적인 argument를 받으며 generated code는 TVM runtime getter가 돌려주는 policy를
argument로 넘긴다.

`region_loop`가 없으면 다음 값으로 동작한다.

```text
LOOP_ENABLE=0
MIN_SAMPLES=0
MIN_SECONDS=0
```

기존 `duration_budget_s`와 DMM `sample_count`는 제거하지 않는다. 이 값들은 반복의
정상 종료 조건이 아니라 요청이 최소 조건을 수용할 수 있는지 확인하는 buffer
capacity와 server watchdog 계약으로 사용한다.

### 6.2 Scope별 기존 경계와 반복 body

TVM scope의 측정 경계는 기존 `power` branch의 세 phase와 동일하게 유지한다.

| TVM scope | BEGIN 위치 | END 위치 | loop body |
|---|---|---|---|
| `MODEL` | 첫 region kernel의 reset/warmup 다음 | 마지막 region의 모든 tile/output 처리 다음 | 여러 region에 걸친 model 실행 전체 |
| `REGION` | 각 region kernel의 reset/warmup 다음 | 해당 region의 모든 tile/output 처리 다음 | instruction/constant transfer, policy update, 모든 tile 실행/output |
| `TILE` | 각 tile input transfer 다음, NPU invoke 직전 | invoke 직후, output read 전 | 해당 tile의 NPU invoke |

`MODEL`에서는 region/tile 경계에서 비동기 tag만 보내며 추가 region을 열지 않는다.
`REGION`에서는 tile tag만 보내며 tile region을 열지 않는다. `TILE`에서는 model과
region level BEGIN을 만들지 않는다. 따라서 TVM의 어느 경로에서도 outer/inner
power region이 동시에 열리지 않는다. measurement_utils가 보는 lifecycle은 세
scope 모두 동일한
`power_region_begin -> async tags -> power_region_end`이다.

동일 binary에서 runtime scope를 선택하고 lexical macro body를 반복할 수 있도록
model, region, tile invoke의 실제 operation을 각각 helper로 분리한다. 개념적인
호출 구조는 다음과 같다.

```c
static int tile_invoke_body(/* ... */) {
    return invoke_npu(/* ... */);
}

static int generated_region_body(/* ... */) {
    /* instruction/constant transfer and policy update */
    for (tile = 0; tile < tile_count; ++tile) {
        transfer_tile_input(tile);
        if (tvm_power_scope_is(TILE)) {
            POWER_REGION_BEGIN(tile_name, loop_enable, min_samples, min_seconds);
            status = tile_invoke_body(/* ... */);
            POWER_REGION_END();
        } else {
            status = tile_invoke_body(/* ... */);
        }
        read_tile_output(tile);
    }
}

static int generated_region_entry(/* ... */) {
    reset_and_warmup();
    if (tvm_power_scope_is(REGION)) {
        POWER_REGION_BEGIN(region_name, loop_enable, min_samples, min_seconds);
        status = generated_region_body(/* ... */);
        POWER_REGION_END();
    } else {
        status = generated_region_body(/* ... */);
    }
}
```

model execution adapter는 첫 region의 reset/warmup 이후에 MODEL BEGIN을 호출하고,
마지막 region의 output 처리가 끝난 직후 END를 호출한다. 이를 위해 기존처럼 함수
이름의 `region1`/마지막-region 문자열에만 의존하지 않고, compile된 model manifest의
region 순서와 명시적인 first/last marker를 사용한다.

```text
MODEL:  first reset/warmup -> BEGIN -> region 1 ... region N -> END
REGION: reset/warmup -> BEGIN -> transfer/policy/all tiles/output -> END
TILE:   input transfer -> BEGIN -> invoke -> END -> output read
```

반복되는 body는 같은 input을 사용하고 output buffer를 매번 overwrite하며 마지막
iteration의 output만 호출자에게 남는다. 이 의미가 올바르려면 scope별로 다음을
검증해야 한다.

- `MODEL`: 여러 region을 같은 순서와 동일 input으로 다시 호출할 수 있는가
- `REGION`: transfer/policy/tile/output 전체를 다시 실행해도 다음 region의 input과
  최종 model output이 동일한가
- `TILE`: input을 다시 전송하지 않고 같은 invoke를 반복해도 안전한가
- accelerator가 동일 body를 다시 실행할 수 있도록 iteration마다 필요한 re-arm과
  interrupt ack가 수행되는가
- retry loop와 measurement loop가 중첩될 때 retry가 동일 iteration을 재시도하고
  성공한 뒤에만 `region_iteration`이 증가하는가
- stateful operator나 random/input mutation 때문에 반복 결과가 바뀌지 않는가

기존 측정 경계를 보존하기 위해 reset/warmup은 MODEL의 첫 시작 및 각 REGION 시작
앞에, tile input transfer는 TILE 시작 앞에 둔다. 이 경계 밖 동작을 iteration마다
다시 수행해야만 재실행 가능한 target이라면 측정 body 안으로 조용히 옮기지 않는다.
그 scope의 loop를 명시적으로 unsupported로 실패시키거나 별도 re-arm 정책을 설계한
뒤 활성화한다.

### 6.3 Artifact grouping과 metadata

- 두 host CMake template에 `power_region.c`와 header include path를 추가한다.
- build metadata에 macro/API version과 loop policy 전달 방식을 기록한다.
- measurement_utils는 모든 BEGIN마다 unique `region_id`와 동일한 artifact layout을
  만든다. MODEL/REGION/TILE별 artifact 분기는 만들지 않는다.
- TVM은 자신의 run manifest에 scope와 region ID 목록을 기록하여 결과를 묶는다.
  `MODEL`은 model 실행별 ID 하나, `REGION`은 region kernel 호출별 ID, `TILE`은 tile
  invoke별 ID를 가진다. 이 grouping 정보는 measurement_utils request의 scope가
  아니다.
- 각 region의 `request.json`, `resolved_config.json`, `session.json`, `summary.json`에
  generic request, region name/ID와 loop policy를 보존한다.
- `summary.json`에는 iteration 수, stop reason, 시작 GET과 종료 시점의 sample/time,
  최소 조건 초과량, progress query latency를 추가한다.
- 기존 `rails/*.npz`, raw CSV, timestamp alignment와 ambiguity schema는 유지한다.

## 7. 구현 순서

### Phase 0: DMM capability spike

- GPIB3 acquisition 중 `DATA:POIN?` 안전성 및 latency 측정
- raw metadata/buffer exact-match 검증
- 결과를 짧은 Markdown 기록으로 남기고 live count 사용 가능 여부 결정

### Phase 1: measurement_utils low-level 기능

- `DmmManager.query_current_sample_counts()` 추가
- protocol v5의 단일 power-region begin/progress/end 추가
- request schema와 Python/C protocol type에서 scope/mode 제거
- generic power-region begin/progress/end C API 및 response validation 추가
- client process-global active guard와 server DMM reservation guard 구현
- fake DMM/server unit test 추가
- measurement server에 progress query와 termination metadata 저장 추가

### Phase 2: measurement_utils macro engine

- `capi/power_region.[ch]`와 public macro 추가
- policy validation과 minimum-only state machine 구현
- standalone init/shutdown와 env convenience API 구현
- non-nested lifecycle, sequential region 재사용과 nested 오류 복구 구현
- iteration tag/event, capacity/stall 보호, error/status API 구현
- C fake-server smoke test로 loop 횟수와 finalization 검증

### Phase 3: TVM runtime/runner/config

- `power_request.py`의 `region_loop` schema와 validation 추가
- TVM-only policy와 scope/mode가 제거된 measurement request를 분리
- `power_steps.sh`에서 policy를 안전한 board 환경변수로 전달
- `power_measure_runtime.[ch]`가 generic measurement_utils runtime을 adapter로
  사용하되 scope를 measurement_utils로 전달하지 않도록 변경
- 두 CMake template에 새 source를 link하고 build identity 갱신
- model execution adapter와 `ext_codegen.py`에 기존 `MODEL`/`REGION`/`TILE` 경계를
  구현하고 선택된 한 scope에서만 BEGIN/END가 생성되는지 검사
- quick-start 문서에 설정과 결과 해석 추가

### Phase 4: 실제 장비 통합 검증

1. board의 작은 C test body를 `usleep`/counter와 함께 macro로 감싸
   sample/time별 예상 loop 횟수를 확인한다.
2. `LOOP_ENABLE=0`이 기존 region 결과와 동일한지 확인한다.
3. ResNet 한 sample에서 loop 전/후 output을 비교해 bit-exact인지 확인한다.
4. `MIN_SAMPLES`, `MIN_SECONDS`, 두 조건 동시 및 capacity/stall error를 각각
   검증한다.
5. retry/timeout, measurement server disconnect, body failure에서도 region artifact가
   abort/finalize되고 다음 run에서 DMM reservation이 남지 않는지 확인한다.
6. raw CSV, NPZ, tag state, ambiguity, iteration summary를 검증하고 master로 SCP된
   결과가 server 원본과 checksum이 같은지 확인한다.
7. 직접 작성한 C test와 TVM 양쪽에서 nested BEGIN이 GET 전에 거부되고, outer
   region은 손상 없이 계속 측정된 뒤 END되는지 확인한다.

## 8. 테스트 계획

### Python/server unit test

- progress command는 active region에서만 허용
- malformed length/count/sequence와 protocol v4 client 거부
- request/protocol에 `scope` 또는 `mode`가 들어오면 schema 오류로 거부
- per-rail direct count aggregation과 단조성 검사
- sample count 정체, 서로 다른 rail 속도, capacity 도달
- active region 중 두 번째 BEGIN 거부, END 없는 connection close의 abort artifact,
  정상 END 이후 다음 sequential BEGIN 허용
- 비동기 tag가 active region에만 귀속되고 END 이후 늦은 tag가 거부되는지 확인
- summary의 stop reason/minimum 초과량/progress latency schema

### C unit/smoke test

- disabled instrumentation과 `LOOP_ENABLE=0`: body 1회
- sample minimum, time minimum, 두 minimum의 AND 조건
- standalone explicit init와 env init
- 최소 조건을 달성하기 전 capacity 소진/progress stall
- `continue`, `break`, start/progress/finalize error
- 직접 함수와 macro 모두 sequential region 허용
- 같은 thread와 다른 thread의 nested/concurrent region을 network 요청 전에 거부
- nested macro의 body는 실행되지 않고 outer region/tag/finalize는 정상 유지
- active region 없는 tag/END와 active region 중 shutdown의 명시적 오류
- macro argument가 한 번만 평가되는지 확인
- protocol payload partial read, timeout, disconnect 처리

### TVM regression test

- generated Linux code에 macro pair와 policy getter가 정확히 삽입됨
- baremetal code에는 power macro가 삽입되지 않음
- MODEL/REGION에서는 해당 시작 전 reset/warmup이 region 밖에 있고, TILE에서는
  input transfer와 output read가 region 밖에 있는 기존 경계를 확인
- REGION body에는 transfer/policy/all tile/output이, TILE body에는 invoke만 포함됨
- `MODEL`, `REGION`, `TILE` scope 모두 동일 generated binary로 동작함
- MODEL에서는 model-level, REGION에서는 region-level, TILE에서는 tile-level
  BEGIN/END만 발생하여 server에서 nested 오류가 한 건도 발생하지 않음
- TVM manifest의 scope와 region ID grouping이 measurement request와 분리됨
- `mode=wait` loop config는 실행 전에 거부됨
- retry/cleanup의 모든 control-flow가 END finalizer를 통과함
- 기본 loop-disabled config가 기존 generated behavior와 동일함
- scope 생략 시 `REGION`으로 동작하고 invalid scope는 board 실행 전에 거부됨
- shell syntax, Python compile, C build 및 기존 power workflow test 전체 통과

## 9. 완료 기준

- GPIB3에서 live sample count query가 acquisition과 raw metadata를 손상시키지 않는다.
- loop-disabled 사용은 기존 region 측정과 동작 및 artifact가 호환된다.
- loop-enabled region은 지정된 최소 sample/time을 만족하거나 capacity/stall/통신
  error로 명시적으로 실패하며 조용한 무한 loop가 없다.
- 동일 macro가 TVM symbol 없이 standalone C application에서 동작한다.
- measurement_utils의 public API, request, protocol, artifact schema에 scope가 없다.
- TVM에서는 `MODEL`, `REGION`, `TILE` scope를 지원하고 mode=now만 허용한다.
- nested/concurrent BEGIN은 DMM GET 전에 명시적으로 거부되고 outer region은 계속
  정상 동작하며, sequential region은 허용된다.
- 여러 rail에서 모든 rail의 최소 sample 조건을 만족한다.
- body 반복 횟수와 각 iteration tag가 raw timestamp에 정렬된다.
- ResNet 반복 실행의 최종 output/accuracy가 loop-disabled 실행과 동일하다.
- 실패 및 timeout 경로에서도 DMM session/lock이 남지 않는다.
- master, board, measurement server의 revision/build identity 검사가 계속 적용된다.

## 10. 예상 위험과 대응

| 위험 | 대응 |
|---|---|
| acquisition 중 `DATA:POIN?`가 DMM timing을 교란 | Phase 0을 구현 선행 gate로 사용하고 실패 시 sample 기반 loop를 활성화하지 않음 |
| 짧은 body에서 progress TCP/GPIB overhead 비율이 큼 | overhead를 trace/summary에 기록하고, 필요 시 여러 body 실행 후 조회하는 batch option을 후속 추가 |
| MIN 조건이 DMM buffer capacity보다 큼 | macro 시작 전 configured capacity로 달성 가능성을 검사하고 즉시 실패 |
| DMM count가 증가하지 않아 무한 반복 | progress stall watchdog로 abort하고 정상 최소 달성과 구분 |
| 한 body가 남은 DMM buffer를 크게 초과 | capacity preflight와 truncation failure를 적용하고 결과에 실제 초과량 기록 |
| 반복 실행이 accelerator state/output을 바꿈 | 첫 버전은 전체 region body 반복, ResNet bit-exact 검증, stateful kernel은 opt-out |
| retry loop와 measurement loop의 cleanup 충돌 | early return 제거, 단일 cleanup label, retry와 iteration state machine을 분리해 테스트 |
| macro의 `return/goto`가 finalize를 우회 | 금지 규칙 문서화, generated code lint/test로 검사, 필요 시 후속 GCC cleanup guard 검토 |
| 둘 이상의 TVM scope adapter가 동시에 BEGIN | TVM runtime에서 `MODEL`/`REGION`/`TILE` 중 하나만 선택하고 generated-call trace 및 server nested guard로 이중 검증 |
| 여러 thread가 동시에 power region을 시작 | client mutex와 DMM reservation으로 두 번째 BEGIN을 GET 전에 거부하고 오류를 caller에 반환 |
| protocol/repository version 불일치 | protocol v5 handshake와 기존 three-system revision preflight 유지 |

## 11. 수행 결과

2026-08-18 기준으로 다음 구현과 검증을 완료했다.

- measurement_utils protocol v5의 단일 non-nested power-region
  BEGIN/PROGRESS/END
- scope/mode가 제거된 measurement request와 직접 `power_tag_*` API
- `DATA:POIN?` 기반 per-rail live count, GET 기준 elapsed와 capacity preflight
- standalone `POWER_REGION_BEGIN/END` minimum-only loop와 nested/concurrent guard
- TVM-only `MODEL`/`REGION`/`TILE` scope adapter와 기본 scope `REGION`
- MODEL acquisition을 첫 generated region의 reset/warmup 뒤까지 지연하는 gate
- REGION 전체 body 및 TILE invoke body 반복, retry cleanup과 macro nesting 처리
- TVM manifest의 scope/policy/region-ID grouping
- raw CSV, NPZ, tag, progress/iteration summary 유지

검증 결과는 다음과 같다.

- measurement_utils 전체 unit test: 114개 실행(113개 통과, 1개 skip)
- 추가 protocol/C macro test를 포함한 tagged server test: 18개 통과
- TVM power workflow/config/codegen/C syntax test: 15개 통과
- Python compile, Bash syntax, C11 `-Wall -Wextra -Werror` compile 통과
- GPIB3 live count가 단조 증가하고 polling 후 raw CSV/buffer 5,000 sample이
  정확히 일치함
- PetaLinux standalone C smoke에서 nested BEGIN `-3`, outer/sequential region
  finalize 및 전체 artifact 생성을 확인함

실장비 세부 결과는
[power_region_hardware_validation.md](../power_region_hardware_validation.md)에
기록했다. ResNet의 세 scope별 bit-exact/accuracy 비교는 master/board/meas-2의 새
revision을 정식으로 commit·동기화하고 model binary를 다시 생성한 뒤 수행하는 최종
배포 acceptance로 남긴다. 이번 검증에서는 운영 중인 protocol-v4 daemon과 원격 repo
checkout을 변경하지 않고 별도 임시 port 9911을 사용했다.
