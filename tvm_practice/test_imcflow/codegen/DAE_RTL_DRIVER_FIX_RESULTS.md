# DAE 기존 driver 수정·검증 결과

2026-09-08, `/root/project/tvm_latency`, `chip_acc_measure`.
계획의 에이전트 측 1–6단계를 완료했다. RTL simulation·CPU 출력 비교·latency 측정은
사용자 실행 대상으로 남겨 두었다. `driver-v2`와 `BUGFIX=off`는 실행하지 않았다.
변경 사항은 아직 커밋하지 않았다.

## 수정

- `imce_codeblock.py`에 `classify_minmax_quantize_outputs()`를 추가하고
  handler와 block의 판별 함수를 이 helper에 연결했다.
- 동일 source의 일반 다중 소비자만 허용한다. 출력 없음, 다른 source 혼합,
  여러 출력에 split이 포함된 형태는 `ValueError`로 거부한다.
- 단일 split의 기존 메타데이터 판별은 유지한다. 누락된 메타데이터는 오류로 남긴다.
- depthwise 전용 builder, 단일 split edge 검사, wrapper FIFO/policy 검사,
  partition과 송수신·동기화 코드는 수정하지 않았다.
- `unittests/test_minmax_quantize_multicast.py`에 양쪽 진입점의 회귀 테스트를 추가했다.

## 검증 결과

| 항목 | 결과 |
| --- | --- |
| 계획의 단위 테스트 + 새 회귀 테스트 | 33 passed, 7 deprecation warnings |
| depthwise v2/v3/v4 수정 전후 컴파일 | 6회 모두 종료 코드 0 |
| depthwise 생성 코드 비교 | 모델별 imce.cpp/inode.cpp 2개, 모두 byte 단위 동일 |
| v3/v4 특수 경로 존재 | 두 baseline에서 `dwconv minmaxquant` 확인 |
| DAE 기존 driver 컴파일 | 종료 코드 0, 두 region 생성 |
| DAE 메타데이터 | driver_v2=false, imcflow_bugfix=true, baremetal/x86 |
| DAE graph executor | lib_graph_system-lib.tar 생성 |
| DAE multicast 6곳 | 동일 source, 일반 qconv 소비자, FIFO/policy 주소 일치 |
| DAE 송수신 횟수 | 80개 edge 기록 전부 일치, 불일치 0 |
| QREG/SEND | multicast마다 QREG 0→1→2→3, 공유 multicast SEND 4개 |
| 소비자 코드 | 일반 qconv·composite 내부 qconv 모두 해당 edge의 LOAD_LB 확인 |
| Host binary | execute_graph 생성, 종료 코드 0, simulation 건너뜀 확인 |
| 공유 RTL build | manifest BUGFIX=on 및 simv_imcflow_gem5 존재 확인 |
| git diff --check | 통과 |

현재 host binary:
`eval_dir/dae_toycar_full_pretrained_evl.baremetal/host_binary_make/build/execute_graph`
(약 2.9 MiB). 실행하지 않았다.

## 동기화 검토 및 남은 주의사항

`BUGFIX=on`에서는 `send_recv_sync.py`가 contention-filter의 pair 집합을 비운다.
따라서 두 region의 IMCE 코드에 데이터 경로 STANDBY/SETFLAG가 없으며,
multicast producer와 일반/composite 소비자는 bare SEND/LOAD_LB 경로를 사용한다.
이 경로에서 새로 생긴 unmatched producer STANDBY는 없다. INODE의 별도 전역 barrier는
SET_FLAG(255) → 다른 INODE의 STANDBY(...,255) → SET_FLAG(0) 형태로 남아 있다.
이번 변경은 이 동작을 바꾸지 않았다.

BUGFIX=on에서는 FIFO 횟수 불일치의 compile-time assert와 P4 handshake 검증이 모두
비활성화되어 있다. 따라서 컴파일 성공만으로 판단하지 않고 consistency 파일의
80개 기록을 별도로 검사했다. 그래도 FIFO/backpressure 및 barrier의 실제 진행 가능성,
출력 정확성은 RTL로 확인해야 한다. 정적 검토를 deadlock-free 보장으로 해석하면 안 된다.

Host 빌드에는 `program_scan_reg_kernel.cc` 부재 경고와 `TVM_CRT_LOG_LEVEL` 재정의
경고가 있었다. 전자는 공용 빌드 함수의 파일 존재 검사에서 출력됐고 CMake에서 선택적으로
링크하는 파일이다. 실제 링크는 성공했다. 기존 driver/host 실행 소스에는 해당 커널 호출이
검색되지 않았다. 별도 scan register 프로그래밍이 필요한 실험까지 검증한 것은 아니다.

RTL manifest 확인은 기록된 모드와 바이너리 존재 확인이다. 현재 전체 소스와 build fingerprint의
일치까지 검증한 것은 아니며, 실제 실행 시 RTLRunner setup이 확인한다.
공유 RTL 빌드 변경·clean·시뮬레이터 실행은 이번 작업에서 수행하지 않았다.

## 재현 환경과 증거

- TVM 기준 HEAD: `517a2d00b5374e045319e2bcc4cc5e0c2772997b` + 이번 수정.
- Python: `/root/project/tvm/tvm_practice/tvm_env/bin/python`.
- TVM: `/root/project/tvm_latency/python/tvm/__init__.py`.
- Native library: `/root/project/tvm_latency/build/libtvm.so`.
- Checkpoint: `/root/project/CIM/runs/dae_integration/initial/checkpoint.pth.tar`.
- SHA256: `0ed5ad690f012fee93f0bfd7be152965f27e1fdf70afd3ca8821649fe934fe71`.
- 기존 driver, HALF, acc-mask 0, random 입력, PYTHONHASHSEED=0.
- 나머지 환경은 계획 1단계의 run_on 설정과 동일하다.

증거 디렉터리: `/tmp/dae-driver-bugfix-on.M2vzZr`.
이 디렉터리의 `validate.py`에 실제 실행 환경·명령, `compare_dw.py`에 생성 코드 비교,
`audit_dae.py`에 metadata/route/count/QREG/소비자 검사 로직을 남겼다.
`unit-tests.log`, `one_dwconv_v*.before.log`, `one_dwconv_v*.after.log`,
`dae.compile.log`, `dae.source-audit.log`, `dae.host-build.log`에 실행 결과가 있다.
TVM/imcflow/gem5 revision·status 및 checkpoint hash도 보존했다.
기존 depthwise 결과는 같은 증거 디렉터리로 이동했고, 기존 DAE 결과는
eval_dir의 `.previous.*` 디렉터리로 보존됐다. 기존 결과를 삭제하지 않았다.

## 사용자가 실행할 다음 단계

[계획](DAE_RTL_DRIVER_FIX_PLAN.md)의 1단계 환경을 설정하고 7단계 명령을 실행한다.
`--stop-at compare`는 RTL 실행 후 CPU 참조 출력 비교까지 포함하고,
`--stop-at simulate`는 RTL 실행까지만 수행한다. 두 명령을 연속 실행할 필요는 없다.
`--driver-v2`는 추가하지 않는다. 환경의 checkpoint 파일 지정은 `CKPT_PATH`를 사용한다.

RTL 종료·출력 비교·latency 결과는 아직 없다.
