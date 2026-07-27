# 칩에 올릴 데모 자산 업데이트 가이드

데모 chip 모드가 쓰는 **칩 위의 자산**을 갱신하는 절차. (서버에서 수행 — 노트북 아님.)
이번 디버깅(2026-07-27)에서 규명된 핵심 교훈을 반영한다.

## ⚠️ 가장 중요한 교훈: build 를 스킵하지 마라

칩엔 아래 3종이 올라가야 한다:
1. **ARM 실행 바이너리** — `host_binary_make.dataset/build/execute_graph_for_dataset`
2. **MLF graph/params** — checkpoint 를 compile 한 산출물 (**column-disable 등 HW 설정 반영. 여기가 핵심**)
3. **스테이징 데이터셋** — `dataset/cifar10/_staged/`

**잘못된(옛) MLF 가 칩에 남으면 에러 없이 전부 class 0 만 예측한다(정확도 5%, 조용한 실패).**
`run_dataset_eval.sh -s 1`(build 스킵)로 돌리면 옛 host MLF 를 그대로 전송해 이 함정에 빠진다.
→ **MLF 를 바꿔야 할 땐 반드시 build(Step 1)를 포함해 전체 스텝**으로 돌려라.

## 두 단계 구조

```
[1] compile (main.py --stop-at compile)   checkpoint -> eval_dir/<model>_evl.linux/lib_graph_system-lib.tar
[2] build + 전송 + 실행 (run_dataset_eval.sh)   eval_dir tar -> ARM 바이너리 + host MLF -> 칩 전송 -> scan/warmup -> 추론
```

---

## 경우별 절차

### A. checkpoint 를 바꿀 때 (다른 iter / 다른 학습 결과) — compile 부터

```bash
cd /root/project/tvm/tvm_practice/test_imcflow/codegen
# 환경 (CLAUDE.md 규칙)
source /root/project/tvm/tvm_practice/tvm_env/bin/activate
eval "$(direnv export zsh)"
source ./imcflow-linux.sh
export PYTHONPATH=/root/project/tvm/python:/root/project/tvm/vta/python:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/root/project/tvm/build:${LD_LIBRARY_PATH:-}

# [1] compile — 76.4% fixture 와 동일한 인자 (column-disable 필수!)
CKPT_PATH=<새 checkpoint.pth.tar> \
python3 main.py --model resnet8_subset31_pretrained_orig --acc-mask 1 --driver-v2 \
  --ref-models transformed --fixed-imce-core 0,1 --num-disable-columns 32 \
  --column-disable-config /root/project/CIM/noise/noise_df/B2_out_chip3_bitline/N32/disabled.json \
  --random-seed 42 --stop-at compile
#   => eval_dir/resnet8_subset31_pretrained_orig_evl.linux/lib_graph_system-lib.tar 갱신

# [2] build 포함 전체 스텝: 빌드 -> 전송 -> scan/warmup -> 추론 (build 스킵 금지!)
CKPT=<alias> CKPT_PATH=<새 checkpoint.pth.tar> \
DEBUG_EXE=0 CONSOLE_LOG_LEVEL=INFO PRESERVE_DEBUG_DUMPS=0 \
REMOTE_HOST=147.46.117.99 REMOTE_PORT=1326 REMOTE_USER=root REMOTE_AUTH_METHOD=key \
ACC_MASK=1 DATASET_NAME=cifar10 \
./run_dataset_eval.sh --model resnet8_subset31_pretrained_orig_evl.linux --dataset cifar10 20
#   => Accuracy 확인. ~75-90% 나오면 정상. 5%(class 0)면 MLF/warmup 문제.
```

> **정공법(권장)**: 위 [1]+[2] 를 손으로 하는 대신 orchestrator 로:
> ```bash
> cd /root/project/CIM
> python3 scripts/loop/run_resnet8_chip_noise_loop.py \
>   --config scripts/loop/resnet8_chip_noise_loop.eval500.json --run_only chip_run
> ```
> 단 `--run_only chip_run` 은 compile 을 스킵하므로(`--skip-compile`), **직전에 올바른 compile 이
> 돼 있어야** 한다. 새 checkpoint 면 compile 을 먼저(위 [1]) 하거나 orchestrator 의 compile phase 를 포함.
> eval500.json 에 model/column_disable/checkpoint 인자가 다 정의돼 있어 인자 실수를 막아준다.

### B. 샘플셋만 바꿀 때 (다른 이미지 / 개수) — 데이터셋만 재스테이징

바이너리·MLF 는 그대로. 스테이징만 새로 하고 전송:
```bash
cd /root/project/tvm/tvm_practice/test_imcflow/codegen
# num_samples 또는 --indices 로 스테이징 (run_dataset_eval 이 자동 스테이징함)
./run_dataset_eval.sh -s 1 --model resnet8_subset31_pretrained_orig_evl.linux --dataset cifar10 <N>
#   (이 경우엔 MLF 안 바뀌므로 -s 1 build 스킵 OK. 단 데이터셋 전송은 돼야 함)
```
- 데모 fixture 사진(`fixtures/cifar10_test_images.npy`)은 10000장 전부라 별도 갱신 불필요.
  샘플 순번↔원본 매핑은 `sample_map.json` 이 담당.

### C. 아무것도 안 바꿨는데 칩이 리부트됐을 때 — warmup 만

파일은 칩에 남아있으니 재전송 불필요. **warmup 만 다시** 하면 된다:
```bash
ssh -p 1326 root@147.46.117.99 \
  "cd /home/root/imcflow/xilinx/petalinux-csrc && make clear_time && make warmup"
```
- 데모 runner(`backend/runner.py`)는 이제 **추론 전 warmup 을 자동 포함**하므로,
  chip 모드로 그냥 실행하면 warmup 이 알아서 된다. (수동 warmup 은 확인용.)

---

## 검증 방법 (업데이트 후 항상)

20샘플 정도로 **정확도와 예측 분포**를 확인:
```bash
ssh -p 1326 root@147.46.117.99 "grep 'Accuracy' /tmp/tvm_dataset_results.txt; \
  grep 'Predicted:' /tmp/tvm_dataset_results.txt | sed -E 's/.*Predicted: ([0-9]+).*/\1/' | sort | uniq -c"
```
- **정상**: Accuracy ~75-90%, 예측이 여러 클래스에 분포.
- **비정상**: Accuracy ~5-10%, 예측이 거의 class 0 하나 → MLF 잘못됨(build 스킵?) 또는 warmup 안 됨.

## 왜 이런가 (배경)

- **MLF**: checkpoint 마다 다르고, column-disable(칩 불량 컬럼 회피) 설정이 반영돼야 칩에서 맞는 연산.
  build 스킵 시 옛 host MLF 가 전송돼 어긋남.
- **warmup**: IMC 아날로그 코어는 추론 전 `make warmup` 으로 깨워야 함. 안 하면 전부 class 0.
  칩 리부트하면 warmup 상태가 날아가므로 재실행 필요. (`.last_warmup_time` gating)
- **scan npz 에러**(`imce_0_1.npz Unable to open`)는 무시해도 됨 — 76.4% 정상 실행에도 있던 무해한 에러.
