# IMCFlow chip inference-only 데모 (ResNet8 / CIFAR-10)

chip3 학습 loop의 **best checkpoint**(ResNet8 `resnet8_run4/iter_009`, chip acc 80%)로
칩 inference-only를 돌리고, 그 결과를 **실시간 웹으로 시각화**하는 데모.

- 샘플 사진 + top-5 추측(softmax %) + 실제 정답 + running accuracy 를 한 화면에.
- 뷰어는 칩 stdout 을 실시간 파싱해 SSE 로 브라우저에 push.
- **파생 계산(softmax·top-5·running acc)은 전부 뷰어**가 raw logit 으로 직접 계산.
  칩 자체 집계(`Progress:`)는 완료 신호 + 최종 크로스체크로만 사용.

## 구조

```
demo/
├─ README.md              # 이 문서 (실행법 + 이식 가이드)
├─ verify.py              # 헤드리스 자동검증 (브라우저 없이 파이프라인 검증)
├─ config/resnet8.yaml    # task/접속/실행 파라미터 전부. mock<->chip 은 profile 만 바꾼다
├─ backend/
│  ├─ app.py              # FastAPI: /config /image /stream(SSE)
│  ├─ runner.py           # 소스 추상화: mock(txt 재생) | chip(ssh subprocess)
│  └─ parser.py           # 칩 stdout 라인 -> 샘플 이벤트 (파생계산 안 함)
├─ frontend/index.html    # SSE 구독 + 사진/top5/정답/running acc 렌더
└─ fixtures/
   ├─ resnet8_iter009_500.txt   # 박제된 canned 결과 (실제 best checkpoint 500샘플, 76.4%)
   ├─ sample_map.json           # staged 순번 -> 원본 CIFAR-10 인덱스
   └─ cifar10_metadata.json     # 클래스명·정규화 파라미터 참조
```

## 실행 (이 서버 / mock 모드)

`config/resnet8.yaml` 의 `profile: mock` 상태(기본)에서:

```bash
# 헤드리스 자동검증 (권장 — 브라우저 없이 전 경로 확인)
cd demo
/root/project/tvm/tvm_practice/tvm_env/bin/python3 verify.py
#  => samples parsed 500 / viewer acc 76.40% == chip acc 76.40% / ✅ PASS

# 웹 서버 (브라우저 육안 데모)
cd demo/backend
/root/project/tvm/tvm_practice/tvm_env/bin/python3 -m uvicorn app:app --host 127.0.0.1 --port 8079
#  => http://127.0.0.1:8079 접속
```

> venv 의 uvicorn 을 쓴다. `.envrc` 가 direnv-blocked 면 셸 PATH 에 uvicorn 이 안 잡히므로
> **절대경로 python 의 `-m uvicorn`** 으로 띄우는 게 안전하다.

mock 은 `fixtures/resnet8_iter009_500.txt` 를 라인별로 재생하되 per-sample 마다
`mock.delay_ms`(기본 300ms) sleep 해 실시간처럼 보여준다. 실제 데모(chip)에서는 sleep 없음.

---

## 노트북(WSL)에서 mock 육안 확인 — 칩 연결 불필요

프론트 렌더링을 노트북 브라우저에서 실제로 눈으로 보는 절차. **칩을 LAN 으로 붙일 필요 없다**(mock).
`demo/` 디렉토리만 노트북으로 옮기면 된다(코드+fixture, ~100KB. 대용량 데이터 불필요 —
사진은 torchvision 이 CIFAR-10 을 자동 다운로드해서 쓴다).

```bash
# 1) demo/ 를 노트북으로 복사 (git pull 또는 scp)
# 2) 실행 (최초 1회는 venv 생성 + 의존성 설치 + CIFAR-10 다운로드)
cd demo
./run_laptop.sh
#   => http://127.0.0.1:8079  를 브라우저(Windows 쪽)에서 연다.
#   => 사진이 300ms 간격으로 하나씩 넘어가며 top-5·정답·running accuracy 가 실시간 갱신.

# 이미 셋업했으면:
./run_laptop.sh --no-setup
```

- **사진 소스**: `run_laptop.sh` 는 `DEMO_IMAGE_SOURCE=torchvision` 을 넣어 노트북 모드로 띄운다.
  torchvision CIFAR-10 test split 을 `sample_map.json` 의 `orig_idx` 로 조회한다(순차 가정 없음).
  최초 실행 시 `demo/cifar_data/` 로 test set(약 170MB)을 자동 다운로드.
- **WSL 브라우저**: 서버는 `127.0.0.1:8079`. WSL2 라면 Windows 브라우저에서 `localhost:8079` 로 바로 열린다.
- **이 서버 검증과의 차이**: 렌더링·SSE·집계 코드는 100% 동일하고 **사진 소스만** 다르다
  (서버=staged npy 역정규화, 노트북=torchvision 원본). 매핑 로직은 양쪽 동일하게 검증됨.

> torchvision 설치가 부담이거나 오프라인이면 `DEMO_IMAGE_SOURCE=staged_npy` 로 띄우고
> `dataset/cifar10/_staged/images.npy` 를 노트북에 함께 두면 사진 없이/역정규화로도 볼 수 있다.

---

## 이식 가이드 — 노트북 연결 + 실제 칩 구동 (mock → chip)

> 이번 세션은 **배선만** 했다. 아래는 실제 데모 당일 노트북에서 칩을 돌릴 때의 절차다.

### 0. 전제: 아키텍처

```
FPGA(칩) ── SSH(내부IP) ── WSL 노트북(백엔드+브라우저)
```

노트북이 직접 칩에 SSH 한다. 서버는 **데모 실행 경로에서 빠진다.**
단, 칩에 올라갈 **ARM 바이너리·데이터셋·스캔레지스터는 서버가 미리 빌드/전송**해 둔 것을 쓴다(B1).

### 1. 준비 (서버에서 1회) — 원격 자산 적재

칩에 실행 바이너리·스테이징 데이터셋·스캔이 올라가 있어야 데모의 "실행만" 경로가 성립한다.
검증된 기존 스크립트를 **그대로** 1회 완주:

```bash
cd /root/project/tvm/tvm_practice/test_imcflow/codegen
CKPT_PATH=/root/project/CIM/runs/chip3/resnet8_run4/n32_signed_sample_loop/iter_009/deploy/2026-Jul-25-12-18-13/checkpoint.pth.tar \
REMOTE_HOST=<내부IP 또는 별칭> REMOTE_PORT=<포트> REMOTE_USER=root REMOTE_AUTH_METHOD=key \
DATASET_NAME=cifar10 \
./run_dataset_eval.sh --model resnet8_subset31_pretrained_orig_evl.linux --dataset cifar10 500
```

이걸로 스텝1~7이 다 돈다(빌드·전송·스캔·실행·회수). 이후 데모는 스텝6(실행)만 반복한다.

### 2. 바이너리 소수정 — per-sample 실시간 stdout (한 줄)

현재 칩 바이너리는 `[Sample N] Scores/Predicted/Label` 를 **결과 파일(`g_result_file`)에만** 쓰고,
stdout 으로는 `Progress:` 만 흘린다. 실시간으로 per-sample top-5 를 뿌리려면 stdout 에도 흘려야 한다.

- 파일: `host_binary_make.dataset/src/debug_execute_graph_for_dataset.c`
- 위치: `print_class_scores(...)` 호출 근처(라인 ~707) — 매 샘플 `printf("Progress:...")+fflush(stdout)` 바로 옆.
- 추가: 같은 자리에서 `[Sample N] Scores: [...]` 와 `Predicted: p, Label: l, Result: O/X` 를
  **`printf` 로(파일이 아니라 stdout)** 한 벌 더 내보내고 `fflush(stdout)`.
  포맷은 파서(`parser.py`)가 먹는 것과 동일해야 한다(현재 파일 포맷 그대로 복제하면 됨).
- 재빌드: 데모용 ARM 바이너리를 한 번 다시 빌드해 칩에 올린다(스텝1~2). checkpoint 하나 고정이라 1회면 됨.

> 속도 영향 없음: 칩 병목은 샘플당 MMIO+UIO 핸드셰이크(수십~수백 ms)라, stdout 한 줄(~108B) 추가는 무시 수준.
> `CONSOLE_LOG_LEVEL` 은 **INFO 가 아닌 값**이어야 `[Sample]` 라인이 `filter_eval_output.sh` 에 안 걸린다.

### 3. 노트북 SSH 설정 (⚠️ 이식 시 가장 헷갈리는 3함정)

**함정 ①: 이 코드/스크립트는 ssh_config 별칭이 아니라 IP+포트를 직접 조립한다.**
`scan_steps.sh` 는 `ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST` 로 만든다.
→ `~/.ssh/config` 의 `petalinux2` 를 내부IP로 바꿔도 스크립트엔 자동 반영 안 됨.
   반드시 **`REMOTE_HOST`/`REMOTE_PORT`/`REMOTE_USER` 를 넘겨야** 한다. 데모 백엔드는
   `config/resnet8.yaml` 의 `chip.ssh_host/ssh_port/ssh_user` 로 이걸 관리한다.
   (`ssh_host` 에 별칭을 써도, 명령행 `-p ssh_port` 가 config Port 를 **덮으므로** 포트는 `ssh_port` 값이 최종이다.
    → config Port 와 `ssh_port` 를 **일치**시켜 둘 것.)

**함정 ②: `transfer_evl.sh` 에 옛 공인IP 가 하드코딩돼 있다.**
`transfer_evl.sh:8-9` → `REMOTE_HOST=${REMOTE_HOST:-147.46.117.99}`, `REMOTE_PORT=${...:-1326}`.
→ 환경변수를 안 넘기는 호출 경로가 하나라도 있으면 **옛 공인IP 로 샌다.**
   준비 스텝(1)에서 항상 `REMOTE_HOST/REMOTE_PORT` 를 명시하거나, 이 기본값을 내부값으로 바꿔 둘 것.

**함정 ③: 키교환 + known_hosts.**
`auth_method: key` → `ssh -o BatchMode=yes`(비대화식, 키 필수). 노트북에서:
1. 노트북 `~/.ssh/` 에 칩 접속 키 배치(키교환 완료).
2. **데모 전 내부IP 로 1회 수동 SSH** 해 known_hosts 에 지문 등록.
   (안 하면 BatchMode 에서 host key 확인 프롬프트 없이 바로 실패한다.)

### 4. mock → chip 전환

`config/resnet8.yaml` 만 수정:

```yaml
profile: chip            # mock -> chip
chip:
  ssh_host: petalinux2   # 노트북 ssh_config 별칭 (내부IP)
  ssh_port: <포트>
  ssh_user: root
  ...                    # 나머지 원격 경로는 준비 스텝(1)의 적재 위치와 일치해야 함
```

그러면 `runner._chip_lines()` 가 스텝6 원격커맨드를 그대로 복제해 `ssh` subprocess 로 실행하고,
그 stdout 을 line-buffered 로 파서에 흘린다. 나머지(SSE·프론트·집계)는 mock 과 **완전히 동일**하다.

- 실제로 나갈 ssh 커맨드 미리보기:
  ```bash
  cd demo && /root/project/tvm/tvm_practice/tvm_env/bin/python3 -c \
    "import sys,yaml,os; sys.path.insert(0,'backend'); import runner; \
     c=yaml.safe_load(open('config/resnet8.yaml')); c['_demo_root']=os.getcwd(); \
     print(runner.remote_cmd_preview(c))"
  ```

### 5. 이미지 소스 (노트북)

실제 데모에서 화면 사진은 **노트북 로컬 원본 CIFAR-10**(torchvision 표준 test split, 10000장, 순서 고정)에서
`sample_map.json["staged_to_original"][staged_idx]` = 원본 인덱스로 조회한다(**순차 가정 금지**).
- 이 서버 검증(mock)에서는 `dataset/cifar10/_staged/images.npy` 를 역정규화해 복원해 쓴다(`app.py /image`).
- 노트북 전환 시 `app.py` 의 이미지 경로를 노트북 원본 배열로 바꾸고, 조회는 `orig_idx` 기준으로 하면 된다.

---

## 검증 상태 (이번 세션)

- ✅ mock end-to-end: 500샘플 파싱, 뷰어 running acc **76.40% == 칩 Progress 76.40%**(크로스체크 일치).
- ✅ softmax argmax == 칩 predicted (전 샘플), orig_idx 매핑 정합.
- ✅ HTTP: `/config` `/` `/image/{n}`(32×32 RGB PNG 복원) `/stream`(300ms SSE) 전부 200.
- ⚠️ chip 프로파일: **배선만.** 실제 SSH 실행은 호출하지 않음. ssh 커맨드 문자열은 원본 스텝6과 일치 확인.
- 다른 task(VWW/KWS)는 `config/*.yaml` 추가 + 이미지 어댑터만으로 확장 가능(구조는 task 플러그형).
