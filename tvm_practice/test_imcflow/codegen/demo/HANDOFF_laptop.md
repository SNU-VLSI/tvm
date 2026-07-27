# HANDOFF — 노트북에서 칩(petalinux2) 실연결 테스트 이어받기

> 이 문서는 **GPU 서버**에서 진행하던 데모 작업을 **노트북(WSL)** Claude 세션으로 넘기기 위한 것.
> 목표: 노트북 ↔ 칩을 **현재의 petalinux2 공인IP(147.46.117.99:1326) 그대로** SSH 연결해서
> `demo` 웹앱을 **chip 모드**로 돌려, 실제 칩 추론 결과를 실시간 시각화한다.
> (내부 LAN IP 전환은 데모장 직전에 별도로 — 지금은 공인IP가 수정 없이 간단.)

---

## 0. 현재 상태 (서버 세션이 해둔 것)

- ✅ **데모 웹앱 완성 + mock 검증 완료.** `demo_iitp` 브랜치. 노트북에서 mock 정상 실행 확인됨.
- ✅ **칩 바이너리 소수정 불필요** — `print_class_scores` 가 이미 `stdout + 결과파일` 양쪽에
  per-sample `[Sample N] Scores/Predicted/Label` 을 쓴다(debug_execute_graph_for_dataset.c:214).
  → stdout 으로 per-sample 이 실시간으로 나온다. 노트북은 이걸 파싱만 하면 됨.
- ✅ **칩에 올바른 자산 전송 + chip eval 80% 검증 완료** — 서버가 build 포함 전체 스텝으로
  올바른 MLF(76.4% 컴파일)를 칩(`/home/root/tvm/tvm_practice/test_imcflow/codegen`)에 올리고,
  runner 커맨드로 20샘플 **80%(class 0 편향 없음)** 확인. (자세한 경위는 아래 §검증결과)
- ✅ **config chip 프로파일 공인IP 시나리오 맞춤 + warmup 자동화** — `ssh_host: petalinux2`, `ssh_port: 1326`,
  `remote_base_path: /home/root/tvm/.../codegen`, runner 가 추론 전 warmup 자동 포함. config 수정 불필요.
- 📄 **칩 자산 업데이트가 필요할 때** — `UPDATE_chip_assets.md` (checkpoint 변경 등 경우별 절차).

즉 노트북 세션이 할 일은 사실상 **"profile 을 chip 으로 바꾸고 실행 + 결과 육안 확인"** 뿐이다.

---

## 1. 노트북에서 최신 코드 받기

```bash
cd ~/Project/tvm-demo         # 기존 clone 위치 (없으면 아래 clone)
git fetch origin && git checkout demo_iitp && git pull
git lfs pull                  # cifar10_test_images.npy (122MB) 최신화
# 새 clone 이면:
#   git clone --branch demo_iitp --single-branch --depth 1 git@github.com:SNU-VLSI/tvm.git tvm-demo
#   cd tvm-demo && git lfs install && git lfs pull
```

## 2. 칩 SSH 도달성 확인 (노트북 → 칩)

노트북에서 이미 petalinux2 로 붙을 수 있는 상태라고 확인됨. 그래도 chip 모드 전에 한 번 점검:

```bash
# ~/.ssh/config 에 petalinux2 별칭이 있으면:
ssh petalinux2 "echo CHIP_OK; hostname"
#   => CHIP_OK / imcflow  가 나오면 정상.

# 별칭이 없다면 config 에 추가 (서버와 동일):
#   Host petalinux2
#       HostName 147.46.117.99
#       Port 1326
#       User root
#       IdentityFile ~/.ssh/id_ed25519      # 칩 접속 키
```

- **BatchMode 주의**: chip 모드는 `ssh -o BatchMode=yes` (비대화식)로 붙는다.
  최초 접속이면 host key 확인 프롬프트에서 멈추므로, **위 수동 `ssh petalinux2` 를 1회** 해서
  known_hosts 에 지문을 등록해 둘 것. (안 하면 BatchMode 에서 조용히 실패.)
- 키가 아니라 비밀번호로 붙는다면 config/resnet8.yaml 의 `chip.auth_method: password` 로 바꾸고
  `sshpass` 설치 필요 — 하지만 지금은 키 방식 전제.

## 3. chip 모드로 전환

`demo/config/resnet8.yaml` 딱 한 줄:

```yaml
profile: chip      # mock -> chip
```

나머지 chip 블록(ssh_host/port/user, 원격 경로들)은 그대로 두면 된다(공인IP 시나리오에 이미 맞음).
`run.num_samples` 는 처음엔 작게(예: 20) 두고 확인 후 늘리는 걸 권장.

- 실제로 나갈 ssh 커맨드를 미리 보고 싶으면:
  ```bash
  cd demo && python3 -c \
    "import sys,yaml,os; sys.path.insert(0,'backend'); import runner; \
     c=yaml.safe_load(open('config/resnet8.yaml')); c['_demo_root']=os.getcwd(); \
     print(runner.remote_cmd_preview(c))"
  ```

## 4. 실행 + 육안 확인

```bash
cd demo
# chip 모드는 사진 소스도 필요 — full_npy(git-lfs) 기본 유지
DEMO_IMAGE_SOURCE=full_npy ./run_laptop.sh --no-setup    # 이미 venv 있으면 --no-setup
#   => http://127.0.0.1:8079  (Windows 브라우저에서 localhost:8079)
```

- **mock 과 차이**: chip 모드는 sleep 없음 — 칩이 샘플을 실제로 처리하는 속도대로 하나씩 갱신된다.
  (칩 추론은 샘플당 수십~수백 ms MMIO 핸드셰이크라, mock 의 300ms 재생보다 느릴 수도 빠를 수도 있음.)
- 화면에 사진 + top-5(softmax%) + 정답 + running accuracy 가 뜬다. mock 과 렌더링 코드 100% 동일.

---

## 5. 예상 함정 / 트러블슈팅

- **stdout 이 안 흐르고 멈춰 보임**: `chip.console_log_level` 이 INFO 면 filter 가 `[Sample]` 을 먹는다.
  현재 `RAW` 로 돼 있어야 정상. runner.py 는 INFO 가 아니면 raw passthrough.
- **`ssh: BatchMode` 실패 / Permission denied**: known_hosts 미등록(§2) 또는 키 경로 문제.
  `ssh petalinux2` 수동 접속으로 먼저 해결.
- **원격 자산 없음(No such file)**: 칩이 리부트/초기화됐으면 전송이 날아갔을 수 있다.
  서버에서 `run_dataset_eval.sh` 준비 스텝을 재실행해야 함(§준비 재실행).
- **칩이 이미 다른 작업 중**: 서버에서 학습/측정이 칩을 쓰고 있으면 충돌. 칩 사용 중인지 확인 후 실행.
- **SoC wedge**: 깊은 모델에서 CPU0 IRQ 경합으로 SoC 가 먹통될 수 있어 `taskset -c 3` 로 핀돼 있음
  (resnet8 은 얕아 위험 낮음). 만약 wedge 시 칩 리부트 필요.

### 준비 재실행 (칩 자산이 날아갔을 때 — 서버에서)

```bash
cd /root/project/tvm/tvm_practice/test_imcflow/codegen
CKPT=resnet8_demo_iter009 \
CKPT_PATH=/root/project/CIM/runs/chip3/resnet8_run4/n32_signed_sample_loop/iter_009/deploy/2026-Jul-25-12-18-13/checkpoint.pth.tar \
DEBUG_EXE=0 CONSOLE_LOG_LEVEL=DEBUG \
REMOTE_HOST=petalinux2 REMOTE_PORT=1326 REMOTE_USER=root REMOTE_AUTH_METHOD=key \
DATASET_NAME=cifar10 \
./run_dataset_eval.sh -s 1 --model resnet8_subset31_pretrained_orig_evl.linux --dataset cifar10 20
#   -s 1 = build 스킵(로컬 바이너리 재사용), 전송+실행만.
```

---

## 6. 다음 단계 (데모장 — 나중)

- 내부 LAN IP 로 전환: `~/.ssh/config` 의 petalinux2 HostName 을 내부IP로, `chip.ssh_port` 일치.
  (README 이식 가이드 §3 의 3함정 — 별칭 vs REMOTE_HOST / transfer_evl.sh 하드코딩 / known_hosts.)
- 공인IP 로 여기까지 됐으면 내부IP 는 HostName/Port 값만 바꾸는 문제.

---

## 참고 경로

- 데모 루트: `tvm_practice/test_imcflow/codegen/demo/`
- config: `demo/config/resnet8.yaml`  (profile / chip 블록 / image_source)
- runner: `demo/backend/runner.py`  (`_chip_lines` = 실제 ssh 실행, `remote_cmd_preview`)
- best checkpoint: `/root/project/CIM/runs/chip3/resnet8_run4/n32_signed_sample_loop/iter_009/deploy/2026-Jul-25-12-18-13/checkpoint.pth.tar`
- README(전체 설계 + 이식 가이드): `demo/README.md`

## ✅ 검증결과 (서버 20샘플 chip 실행) — chip 모드 재현 성공

**runner 가 생성하는 커맨드(노트북이 쓸 그것 그대로)로 정확도 80%(16/20), 예측 10클래스 분포 확인 (2026-07-27).**
→ chip 모드가 노트북 방식으로 검증됨. class 0 편향 없음.

### 겪은 문제와 원인 (노트북에서도 알아야 함)

처음 chip eval 이 **전부 class 0 예측, 정확도 5%** 로 나왔음. 원인은 **두 가지**였다:

1. **잘못된 MLF (주원인)**: `run_dataset_eval.sh -s 1`(build 스킵)로 돌려서, 칩엔 76.4% 컴파일이 아닌
   **옛 host MLF**가 전송됨. build 를 포함해 eval_dir 의 올바른 tar 로 MLF 를 새로 만들어 전송하니 해결.
   → **build 를 스킵하면 조용히 class 0 만 나온다.** (자세한 절차는 `UPDATE_chip_assets.md`)

2. **warmup 누락**: IMC 아날로그 코어는 추론 전 `make warmup` 으로 깨워야 함. 칩 리부트하면 날아감.
   → runner.py 가 이제 **추론 전 warmup 을 자동 포함**하도록 수정됨(`_build_remote_cmd`).

**무해한 red herring**: `Error loading NPZ file .../imce_0_1.npz` 에러는 76.4% 정상 실행에도 있던 것.
scan npz 는 정확도와 무관하니 무시하라.

### 노트북에서 할 일 (정리)

- 칩엔 지금 **올바른 MLF 가 이미 올라가 있다**(서버가 전체 빌드로 전송함). 노트북은 chip 모드로 바로 실행 가능.
- 단 **정확도를 반드시 확인**하라. ~75-90% 면 정상, ~5-10%(class 0 편향)면 MLF/warmup 문제
  → `UPDATE_chip_assets.md` 참조해 서버에서 재준비.
- 칩 자산을 새로 올려야 할 때(checkpoint 변경 등)의 절차는 **`UPDATE_chip_assets.md`**.

**base path**: 원격 base = `/home/root/tvm/tvm_practice/test_imcflow/codegen` (config 반영 완료).
warmup dir = `/home/root/imcflow/xilinx/petalinux-csrc`.
