# DAE 기존 driver multicast 수정 계획

상태: 2026-09-08 에이전트 측 수정·검증(1–6단계) 완료. RTL simulation은 미실행.
실행 결과와 주의사항: [DAE_RTL_DRIVER_FIX_RESULTS.md](DAE_RTL_DRIVER_FIX_RESULTS.md).
대상: `/root/project/tvm_latency`, `chip_acc_measure` 브랜치.
범위: **기존 driver + `IMCFLOW_BUGFIX=on`**. `driver-v2`, `single-qconv`, `BUGFIX=off` 검증은 제외한다.
RTL simulation은 사용자가 직접 실행한다.

## 문제

DAE 모델 `dae_toycar_full_pretrained`를 기존 driver로 컴파일하면
`AssertionError: Only one output edge is expected`가 발생한다.

DAE의 일반 qconv는 출력 128채널을 64채널 qconv 두 개로 분할한다.
두 qconv 모두 동일한 입력 전체를 필요로 하므로, 앞 min/max quantize 출력은
두 소비자로 multicast된다. 실패 당시 첫 quantize의 `custom_id=45`에서 확인한 edge는 다음과 같다.

```text
quantize(45) ──> qconv((47, 42))
             └─> qconv(46)
```

다음 두 함수는 이런 정상적인 분기에도 출력 edge가 하나라고 강제한다.

- `python/tvm/relay/backend/contrib/imcflow/imce_operation_handlers.py`의
  `MinMaxQuantizeHandler.consumer_is_non_multicast_split()`
- `python/tvm/relay/backend/contrib/imcflow/imce_codeblock.py`의
  `MinmaxQuantBlock.consumer_is_non_multicast_split()`

두 함수 모두 수정해야 한다. handler만 수정하면 block 렌더링에서 같은 오류가 발생한다.
이 가정은 `e6cf2a3c2df`에서 depthwise 전용 처리를 추가하며 도입됐으며,
이번 병합 이전에도 두 브랜치에 동일하게 존재했다.

해당 특수 처리가 필요한 이유는 depthwise의 분할 방식이 다르기 때문이다.
Depthwise에서는 채널 그룹마다 서로 다른 입력 구간을 서로 다른 소비자로 보내야 한다.
`build_mmquant_for_dwconv()`는 32채널 그룹별 RECV → MM_QUANT → GET_QREG → SEND를
구성하고 해당 그룹의 policy/FIFO를 선택한다. 이를 일반 multicast로 바꾸면 안 된다.

저장된 실패 상태에서는 multicast quantize 6곳 모두 동일 source tensor에서
일반 qconv 두 개로 연결됐고, 각 쌍의 FIFO와 policy 주소도 같았다.
이는 일반 송수신 wrapper의 multicast 조건과 일치하지만, RTL 정상 동작을 증명하지는 않는다.

`driver-v2`는 atomic qconv 단위의 다른 partition을 사용하므로,
그 경로의 성공은 기존 driver의 latency 검증을 대신하지 못한다.
기존 `DAE_RTL_LATENCY.md`의 v2 명령은 이 계획의 검증에 사용하지 않는다.

## 해결책

두 판별 함수에 동일한 좁은 분기 규칙을 적용한다. 가능하면 공통 helper를 사용해
handler와 render 단계의 판별이 어긋나지 않도록 한다.

| 출력 형태 | 처리 |
| --- | --- |
| 출력 없음 | 명시적인 오류 유지 |
| 출력 하나, 일반 소비자 | 기존 일반 경로 유지 |
| 출력 하나, multicast split | 기존 일반 split 경로 유지 |
| 출력 하나, non-multicast split | 기존 depthwise 전용 경로 유지 |
| 동일 source의 여러 일반 소비자 | 일반 multicast 경로 허용 |
| 서로 다른 source 또는 split이 섞인 여러 소비자 | 지원하지 않는 형태로 명시적으로 거부 |

여러 소비자는 source identity와 소비자 형태를 확인해야 한다.
단순히 `len(out_edges) != 1`이면 `False`를 반환하거나 첫 edge만 검사하는 방식은 사용하지 않는다.
FIFO/policy 일치 검증은 기존 wrapper의 검사를 유지한다.
split 메타데이터 누락을 일반 경로로 조용히 처리하지 않는다.

다음 코드는 이번 수정에서 동작을 바꾸지 않는다.

- `get_split_consumer_edge_info()`의 단일 출력·split 소비자 검사
- `build_mmquant_for_dwconv()`의 채널 그룹별 패킹, QREG 읽기, 목적지 선택
- 일반 wrapper의 multicast SEND 및 송수신 횟수 계산
- 배치·배선, partition, 하드웨어 BUGFIX 정의와 동기화 규칙

컴파일이 통과한 뒤 추가 문제가 발견되면 별도의 원인으로 기록한다.
assert 제거만으로 송수신·동기화의 정확성이 확보됐다고 판단하지 않는다.

## 검증 계획

### 1. 환경과 실행 규칙

호스트는 현재 `/root/project/tvm_latency`가 있는 Linux 환경을 사용한다.
TVM Python 소스와 native library는 이 worktree에서, Python 패키지는 기존
`/root/project/tvm/tvm_practice/tvm_env`에서 로드한다.
imcflow/gem5/VCS는 `/root/project/imcflow`의 현재 checkout을 사용한다.

아래 명령 블록은 **동일한 Bash 세션**에서 순서대로 실행한다.
실행 시마다 새 증거 디렉터리를 만들며 기존 build, 로그, simulation 결과를 삭제하지 않는다.
공유 RTL runner를 다른 작업이 사용 중이면 RTL 빌드/실행 시점을 조율한다.

```bash
bash
set -euo pipefail
cd /root/project/tvm_latency/tvm_practice/test_imcflow/codegen

export DAE_CHECKPOINT=/root/project/CIM/runs/dae_integration/initial/checkpoint.pth.tar
export DAE_PLAN_LOG_DIR="$(mktemp -d /tmp/dae-driver-bugfix-on.XXXXXX)"
printf 'Evidence directory: %s\n' "$DAE_PLAN_LOG_DIR"
test -f "$DAE_CHECKPOINT"

run_on() {
  direnv exec . env \
    IMCFLOW_RUNNER=rtl \
    IMCFLOW_DIR=/root/project/imcflow \
    IMCFLOW_TVM_CODEGEN_DIR=/root/project/tvm_latency/tvm_practice/test_imcflow/codegen \
    SNPSLMD_LICENSE_FILE=1727@147.46.168.128 \
    CKPT_PATH="$DAE_CHECKPOINT" \
    IMCFLOW_BUGFIX=on \
    IMCFLOW_HOST_OS=baremetal IMCFLOW_HOST_ISA=x86 \
    IMCFLOW_DEBUG=0 IMCFLOW_DEBUG_COMPUTE=1 IMCFLOW_DEBUG_NOC=0 \
    IMCFLOW_BIG_IMEM=0 IMCFLOW_MEASURE_POWER=0 \
    DEBUG_EXE=0 SRAM_BACKDOOR=1 PYTHONHASHSEED=0 \
    "$@"
}

git status --short | tee "$DAE_PLAN_LOG_DIR/git-status-before.txt"
git rev-parse HEAD | tee "$DAE_PLAN_LOG_DIR/tvm-revision.txt"
git -C /root/project/imcflow rev-parse HEAD | tee "$DAE_PLAN_LOG_DIR/imcflow-revision.txt"
git -C /root/project/imcflow/pmap/ISA_sim/gem5 rev-parse HEAD \
  | tee "$DAE_PLAN_LOG_DIR/gem5-revision.txt"
sha256sum "$DAE_CHECKPOINT" | tee "$DAE_PLAN_LOG_DIR/checkpoint.sha256"

run_on python -c 'import os, sys, tvm; print(sys.executable); print(tvm.__file__); print(tvm._ffi.base._LIB._name); print("BUGFIX=" + os.environ["IMCFLOW_BUGFIX"])'
```

기대 결과: TVM 소스·라이브러리는 각각 `tvm_latency/python/tvm`,
`tvm_latency/build/libtvm.so`를 가리키고 BUGFIX는 `on`이어야 한다.
체크포인트를 바꾸려면 baseline 생성 전에 `DAE_CHECKPOINT`를 변경하고 이후 동일하게 유지한다.
`CKPT_PATH`는 파일 경로이며 `CKPT`는 registry 별칭이므로 혼동하지 않는다.
VMODE/ACC_MASK는 아래 모든 모델 실행에서 `HALF`/`0`으로 고정한다.
이는 앞서 오류를 재현한 기본값이며, 측정 조건을 바꾸면 관련 비교도 다시 수행한다.

### 2. 수정 전 depthwise baseline 보존

소스를 수정하기 전에 실행한다. `one_dwconv_v2`는 32채널,
`one_dwconv_v3`는 64채널의 채널 분할, `one_dwconv_v4`는 64채널·32×32 공간 크기다.
세 모델 모두 min/max quantize → depthwise 구조를 포함한다.

```bash
for model in one_dwconv_v2 one_dwconv_v3 one_dwconv_v4; do
  if [[ -e "eval_dir/${model}_evl.baremetal" ]]; then
    mv "eval_dir/${model}_evl.baremetal" "$DAE_PLAN_LOG_DIR/${model}.preexisting"
  fi
  run_on python -u main.py --model "$model" --pattern random \
    --vmode HALF --acc-mask 0 --stop-at compile \
    > "$DAE_PLAN_LOG_DIR/${model}.before.log" 2>&1
  cp -a "eval_dir/${model}_evl.baremetal" "$DAE_PLAN_LOG_DIR/${model}.before"
done
```

각 모델의 종료 코드가 0인지 확인한다. baseline 자체가 실패하면 기존 문제로 기록하고,
수정 전후 동일성 검증에 통과했다고 처리하지 않는다. 64채널 모델에서는 생성된 `imce.cpp`에
`dwconv minmaxquant` 주석과 채널 그룹별 전송이 실제로 존재하는지도 확인한다.
기존 eval 디렉터리가 있으면 먼저 증거 디렉터리로 이동해 보존한 뒤 실행한다.

### 3. 판별 함수 수정 및 단위 테스트

수정과 함께 `unittests/test_minmax_quantize_multicast.py`를 새로 작성한다.
해당 테스트 파일은 구현 단계에서 추가했다.
두 판별 함수에 동일한 케이스를 적용한다.

- 단일 일반 소비자, 동일 source의 두 일반 qconv 소비자
- 단일 multicast split, 단일 non-multicast depthwise split
- 출력 없음, 다른 source 혼합, 일반 소비자와 split 혼합, 여러 split 소비자
- split 메타데이터 누락 시 오류 유지
- 두 함수가 같은 판별 결과를 내는지

```bash
run_on python -m pytest -q \
  unittests/test_minmax_quantize_multicast.py \
  unittests/test_deep_autoencoder_model.py \
  unittests/test_rtl_sample_routing.py \
  'unittests/test_rtl_build_mode.py::test_bugfix_mode_normalizes_valid_values[ON-on]' \
  unittests/test_build_metadata_integrity.py \
  unittests/test_split_conv_to_atomic_order.py \
  > "$DAE_PLAN_LOG_DIR/unit-tests.log" 2>&1
git diff --check
```

`test_rtl_build_mode.py` 전체 실행에는 off 테스트가 포함되므로 위처럼 on 케이스만 지정한다.
새 테스트도 `IMCFLOW_BUGFIX=on`으로 고정한다. off의 컴파일·simulation은 수행하지 않는다.

### 4. Depthwise 수정 후 생성 코드 비교

수정 전과 동일한 환경·모델 옵션으로 다시 컴파일한다.
기존 eval 디렉터리를 이동해서 보존하므로 비-DAE 모델의 재실행에서도 기존 결과를 지우지 않는다.

```bash
for model in one_dwconv_v2 one_dwconv_v3 one_dwconv_v4; do
  mv "eval_dir/${model}_evl.baremetal" "$DAE_PLAN_LOG_DIR/${model}.before-working"
  run_on python -u main.py --model "$model" --pattern random \
    --vmode HALF --acc-mask 0 --stop-at compile \
    > "$DAE_PLAN_LOG_DIR/${model}.after.log" 2>&1
  cp -a "eval_dir/${model}_evl.baremetal" "$DAE_PLAN_LOG_DIR/${model}.after"
done

python3 - <<'PY'
import os
from pathlib import Path
root = Path(os.environ['DAE_PLAN_LOG_DIR'])
for model in ('one_dwconv_v2', 'one_dwconv_v3', 'one_dwconv_v4'):
    before, after = root / (model + '.before'), root / (model + '.after')
    def sources(directory):
        return {p.relative_to(directory): p for name in ('imce.cpp', 'inode.cpp')
                for p in directory.rglob(name)}
    a, b = sources(before), sources(after)
    assert a and a.keys() == b.keys(), (model, 'generated source set differs')
    changed = [str(p) for p in sorted(a) if a[p].read_bytes() != b[p].read_bytes()]
    assert not changed, (model, changed)
    print(model, 'identical generated sources:', len(a))
PY
```

합격 기준은 생성된 IMCE/INODE 코드의 byte 단위 동일성이다.
차이가 있으면 diff와 HW 배치·정책·입력 파일을 비교해 원인을 조사한다.
시간 제한이 있는 ILP 배치 결과가 달라진 경우도 자동으로 무시하거나 성공으로 처리하지 않는다.
동일성 비교를 위해 timestamp나 경로 문자열을 무분별하게 제거하지 않는다.

### 5. DAE 기존 driver 컴파일 및 생성 코드 점검

```bash
run_on python -u main.py --model dae_toycar_full_pretrained --pattern random \
  --vmode HALF --acc-mask 0 --stop-at compile \
  > "$DAE_PLAN_LOG_DIR/dae.compile.log" 2>&1

run_on python - <<'PY'
import json
from pathlib import Path
d = Path('eval_dir/dae_toycar_full_pretrained_evl.baremetal')
m = json.loads((d / 'build_metadata.json').read_text())
assert m['driver_v2'] is False, m
assert m['imcflow_bugfix'] is True, m
assert m['host_os'] == 'baremetal' and m['host_isa'] == 'x86', m
assert (d / 'lib_graph_system-lib.tar').is_file()
print('existing driver / BUGFIX=on graph executor ready')
PY
```

기존 DAE 결과는 `setup_dir()`에 의해 `.previous.*`로 보존된다.
컴파일 종료 코드 외에 다음을 확인한다.

- 원래 단일 출력 assert가 해소되고 기존 driver의 partition으로 코드 생성이 완료된다.
- `tensor_edge_list.txt`와 `devconfig_state.pkl`에서 quantize의 각 multicast를 식별한다.
  Custom ID는 재생성 시 바뀔 수 있으므로 45라는 값에 고정하지 않는다.
- 각 multicast의 source, 목적지, FIFO, policy가 일치 조건을 만족하고,
  한 번의 전송이 두 소비자로 전달되는 구성인지 확인한다.
- `recv_send_consistency.txt`의 각 edge에서 송신·수신 횟수가 일치한다.
- `build/*/imce.cpp`와 `inode.cpp`에서 QREG 읽기 횟수·순서·SEND 횟수를 확인한다.
- 일반 qconv와 composite 내부 qconv 양쪽에 대해 동기화 코드를 추적하고,
  producer의 STANDBY에 대응하는 SETFLAG가 실행될 수 있는지 확인한다.

**BUGFIX=on에서는 P4의 handshake 자동 검증이 비활성화되므로, 송수신 횟수 일치만으로
deadlock이 없다고 판단하지 않는다.** 생성 코드 리뷰와 아래 RTL 검증으로 확인한다.

### 6. Host binary 빌드: simulation 제외

현재 `--stop-at compile`은 graph executor까지만 진행한다.
Host binary는 `run_simulation()` 안에서 빌드되지만, 아래 명령은
`GRAPH_EXECUTOR`를 명시하므로 빌드 직후 return하며 runner를 시작하지 않는다.
로그에 `RUNNING SIMULATION`이라는 제목이 나오더라도 실제 수행 범위는 빌드뿐이다.

```bash
run_on python - <<'PY' > "$DAE_PLAN_LOG_DIR/dae.host-build.log" 2>&1
from test import run_simulation
from runners.pipeline_options import PipelineOptions, PipelineStage
run_simulation(
    'eval_dir/dae_toycar_full_pretrained_evl.baremetal',
    'x86',
    PipelineOptions(stop_at=PipelineStage.GRAPH_EXECUTOR, use_v2=False),
)
PY
test -x eval_dir/dae_toycar_full_pretrained_evl.baremetal/host_binary_make/build/execute_graph
```

빌드 종료 코드 0과 `Compile only mode: skipping simulation`을 확인한다.
여기까지가 에이전트 측 구현·검증 완료 조건이다.

### 7. 사용자가 직접 실행하는 RTL simulation

아래 명령은 계획 작성·구현 중 자동 실행하지 않는다.
새 shell에서는 1단계의 환경과 `run_on`을 다시 설정한다.
`run_on`에 지정된 license server, BUGFIX=on, worktree 경로를 사용한다.

먼저 공유 RTL build 상태를 확인한다.

```bash
python3 - <<'PY'
import json
from pathlib import Path
r = Path('/root/project/imcflow/pmap/ISA_sim/gem5/tests/imcflow/rtl_runner')
m = json.loads((r / 'build/build_manifest.json').read_text())
assert m['compile']['bugfix_mode'] == 'on', m['compile']
assert (r / 'build/simv_imcflow_gem5').is_file()
print('recorded RTL build mode: on')
PY
```

이는 기록된 모드 확인이며, 소스 업데이트 후 manifest 일치까지 증명하지는 않는다.
RTLRunner의 setup에서 전체 manifest를 확인한다. 재빌드가 필요하면 공유 결과를 보존하고,
Makefile의 clean 의존 타깃까지 확인해 삭제하지 않는 방법으로 준비한다.
무조건 `make clean` / `make clean_all`을 실행하지 않는다.

CPU 참조 출력 비교까지 수행하는 정확성 검증 명령:

```bash
run_on python -u main.py --model dae_toycar_full_pretrained --pattern random \
  --vmode HALF --acc-mask 0 --stop-at compare \
  2>&1 | tee "$DAE_PLAN_LOG_DIR/dae.rtl-compare.log"
```

사용자가 요청한 simulation 단계까지의 실행 명령:

```bash
run_on python -u main.py --model dae_toycar_full_pretrained --pattern random \
  --vmode HALF --acc-mask 0 --stop-at simulate \
  2>&1 | tee "$DAE_PLAN_LOG_DIR/dae.rtl-simulate.log"
```

위 두 명령은 목적에 따라 선택한다. `compare`는 simulation도 포함하므로,
같은 검증을 위해 두 명령을 연속 실행할 필요는 없다.
`simulate`만 실행한 경우 최종 출력 비교에 통과했다고 처리하지 않는다.

확인할 결과:

- gem5/VCS가 timeout이나 fatal 없이 정상 종료하고 모든 region이 완료된다.
- 결과가 `tvm_latency/.../eval_dir/dae_toycar_full_pretrained_evl.baremetal/`에 저장된다.
- `logs/rtl_runner/`의 실행 로그와 `test_outputs/rtl_runner/output.npy`를 보존한다.
- `compare`에서 변환 후 CPU 참조 출력과의 비교에 합격한다.
- latency 해석을 위해 기존 driver, checkpoint hash, VMODE/ACC_MASK, 입력,
  debug 설정, 측정 시작·종료점을 기록한다. 로그 설정이 다른 측정과 혼동하지 않는다.

최종 보고에서는 컴파일·정적 검증 완료와 RTL 실행·출력 비교 완료를 구분하고,
미실행 RTL 검증은 미실행으로 남긴다. BUGFIX=off 동작은 이번에 보장하지 않는다.
