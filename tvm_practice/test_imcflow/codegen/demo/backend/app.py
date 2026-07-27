"""FastAPI 백엔드: 칩(또는 mock) stdout 스트림 -> 파싱 -> SSE push.

엔드포인트:
  GET /               데모 프론트 (frontend/index.html)
  GET /config         task 메타 (title, classes, num_samples, profile)
  GET /image/{staged_idx}   staged 순번의 원본 CIFAR-10 사진 (PNG, 확대 없이 32x32 원본)
  GET /stream         SSE. 샘플이 하나씩 완성될 때마다 이벤트를 push.

파생 계산(softmax %, top-5, running accuracy)은 프론트가 raw logit 으로 직접 한다.
백엔드는 파서가 뽑은 raw 이벤트만 넘긴다. (뷰어 자체 집계 메인, 칩 Progress 는 크로스체크.)

이미지 소스: 실제 데모에서는 노트북 로컬 원본 CIFAR-10 을 쓰지만, 이 서버 검증에서는
codegen/dataset/cifar10/_staged/images.npy 를 역정규화해 원본 사진을 복원해 보여준다
(같은 npy 한 소스라 인덱스 어긋남 없음). staged_idx -> sample_map 으로 원본 인덱스도 함께 표기.
"""

import asyncio
import io
import json
import os
import threading

import numpy as np
import yaml
from fastapi import FastAPI
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
    Response,
)

import runner
from parser import ChipStreamParser

# ── 경로 앵커 ──
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_ROOT = os.path.dirname(BACKEND_DIR)
CODEGEN_ROOT = os.path.dirname(DEMO_ROOT)
CONFIG_PATH = os.path.join(DEMO_ROOT, "config", "resnet8.yaml")

# CIFAR-10 정규화 파라미터 (metadata.json transform). 역정규화해 사진 복원용.
_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(3, 1, 1)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["_demo_root"] = DEMO_ROOT
    return cfg


CFG = load_config()

# sample_map: staged 순번 -> 원본 CIFAR-10 인덱스
with open(os.path.join(DEMO_ROOT, CFG["sample_map"])) as f:
    _SAMPLE_MAP = json.load(f)["staged_to_original"]

# ── 이미지 소스 (config.image_source, 환경변수 DEMO_IMAGE_SOURCE 로 오버라이드 가능) ──
# "full_npy"   : 노트북 기본. fixtures/cifar10_test_images.npy (10000장 정규화 float32, git-lfs 박제)를
#                orig_idx(sample_map) 로 조회 + 역정규화. torchvision 다운로드 불필요.
# "staged_npy" : 서버 검증. dataset/cifar10/_staged/images.npy (500장, staged 순번) 역정규화.
_IMAGE_SOURCE = os.environ.get("DEMO_IMAGE_SOURCE") or CFG.get("image_source", "full_npy")
_IMAGES = None          # 정규화된 [N,3,32,32] float32 (full 또는 staged)
_INDEX_BY = "staged"    # "orig" (full_npy: orig_idx 로 조회) | "staged" (staged_npy: staged 순번)

if _IMAGE_SOURCE == "full_npy":
    _p = os.path.join(DEMO_ROOT, "fixtures", "cifar10_test_images.npy")
    _IMAGES = np.load(_p) if os.path.exists(_p) else None
    _INDEX_BY = "orig"
elif _IMAGE_SOURCE == "staged_npy":
    # worktree 는 코드만 복사되므로 대용량 데이터는 원본 checkout 에서도 탐색.
    _cands = [
        os.path.join(CODEGEN_ROOT, "dataset", "cifar10", "_staged", "images.npy"),
        "/root/project/tvm/tvm_practice/test_imcflow/codegen/dataset/cifar10/_staged/images.npy",
    ]
    _p = next((p for p in _cands if os.path.exists(p)), None)
    _IMAGES = np.load(_p) if _p else None
    _INDEX_BY = "staged"

app = FastAPI(title="IMCFlow chip demo")


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(DEMO_ROOT, "frontend", "index.html")) as f:
        return f.read()


@app.get("/config")
def config():
    t = CFG["task"]
    return JSONResponse(
        {
            "title": t["title"],
            "classes": t["classes"],
            "num_classes": t["num_classes"],
            "num_samples": CFG["run"]["num_samples"],
            "profile": CFG.get("profile", "mock"),
        }
    )


@app.get("/image/{staged_idx}")
def image(staged_idx: int):
    """staged 순번에 해당하는 원본 사진(32x32 PNG). 둘 다 정규화 npy 라 역정규화 복원.

    - full_npy 모드   : sample_map 으로 orig_idx 변환 -> 10000장 npy[orig_idx].
    - staged_npy 모드 : 500장 npy[staged_idx].
    """
    from PIL import Image

    if _IMAGES is None:
        return Response(status_code=404, content=b"image npy not found")

    if _INDEX_BY == "orig":
        idx = _SAMPLE_MAP[staged_idx] if staged_idx < len(_SAMPLE_MAP) else staged_idx
    else:
        idx = staged_idx

    arr = _IMAGES[idx].astype(np.float32)  # [3,32,32] normalized
    if arr.shape[0] == 3:
        arr = arr * _STD + _MEAN  # 역정규화
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8).transpose(1, 2, 0)  # HWC
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


class _RunHub:
    """/stream 의 single-flight 게이트 + 이벤트 브로드캐스터.

    칩은 동시에 한 번만 돌릴 수 있다 — 실행이 `taskset -c 3` 으로 같은 코어에 핀되고
    같은 MMIO 를 잡기 때문에, /stream 연결마다 ssh 를 새로 띄우면 두 추론이 칩에서 경합한다.
    (실제로 그렇게 해서 SoC 가 wedge 돼 보드 전원 재인가가 필요했다. 2026-07-27.)
    브라우저 새로고침이나 탭 두 개만으로도 재현되므로 게이트가 필수다.

    동작:
      - 실행 중이 아니면  : 연결이 실행을 "소유"한다 (워커 스레드 1개 = ssh 1개).
      - 실행 중이면       : 새 ssh 를 띄우지 않고 진행 중인 실행에 붙는다.
                            지금까지의 이벤트를 리플레이받아 화면이 즉시 따라잡는다.
      - 모든 클라이언트가 끊겨도 실행은 계속된다 (원격 프로세스는 어차피 계속 도니,
        중간에 끊고 다시 붙어도 이어볼 수 있어야 한다).
      - 실행이 끝난 뒤 새로 연결하면 새 실행이 시작된다 (기존 동작 유지).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._active = False
        self._history: list = []
        self._subs: set = set()
        self.runs_started = 0   # 실제로 띄운 실행(=ssh) 횟수. 게이트가 도는지 확인용.
        self.loop = None

    def attach_or_start(self):
        """(내가 실행 소유자인가, 내 큐, 리플레이할 과거 이벤트) 반환."""
        with self._lock:
            owner = not self._active
            if owner:
                self._active = True
                self._history = []
                self.runs_started += 1
            q: asyncio.Queue = asyncio.Queue()
            self._subs.add(q)
            # history 복사와 구독 등록이 같은 락 안 → 이벤트 중복/유실 없음
            return owner, q, list(self._history)

    def publish(self, ev: dict):
        """워커 스레드에서 호출 — 이벤트 루프로 넘겨 구독자 전원에게 전달."""
        with self._lock:
            self._history.append(ev)
            subs = list(self._subs)
        for q in subs:
            self.loop.call_soon_threadsafe(q.put_nowait, ev)

    def finish(self):
        with self._lock:
            self._active = False
            subs = list(self._subs)
        for q in subs:
            self.loop.call_soon_threadsafe(q.put_nowait, None)  # 종료 sentinel

    def detach(self, q):
        with self._lock:
            self._subs.discard(q)

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active


_HUB = _RunHub()


def _run_worker(hub: _RunHub):
    """실행 소유자 1명만 도는 워커. blocking 제너레이터라 별도 스레드."""
    try:
        parser = ChipStreamParser()
        for line in runner.iter_lines(CFG):
            ev = parser.feed(line)
            if ev is None:
                continue
            if ev["type"] == "sample":
                orig = _SAMPLE_MAP[ev["idx"]] if ev["idx"] < len(_SAMPLE_MAP) else ev["idx"]
                ev["orig_idx"] = orig
            hub.publish(ev)
    except Exception as e:  # ssh 실패 등 — 조용히 죽지 않게 프론트로 올린다
        hub.publish({"type": "error", "message": f"{type(e).__name__}: {e}"})
    finally:
        hub.publish({"type": "done"})
        hub.finish()


@app.get("/status")
def status():
    """실행 중인지 확인용 (디버그/운영). 칩을 건드리지 않는다."""
    return JSONResponse({"running": _HUB.active, "runs_started": _HUB.runs_started})


@app.get("/stream")
async def stream():
    """SSE. 실행 중이면 새 ssh 를 띄우지 않고 진행 중인 실행에 붙는다(_RunHub 참조)."""
    _HUB.loop = asyncio.get_running_loop()
    owner, q, replay = _HUB.attach_or_start()
    if owner:
        threading.Thread(target=_run_worker, args=(_HUB,), daemon=True).start()

    async def gen():
        try:
            for ev in replay:  # 늦게 붙은 뷰어를 현재 시점까지 즉시 따라잡힘
                yield f"data: {json.dumps(ev)}\n\n"
            while True:
                ev = await q.get()
                if ev is None:
                    break
                yield f"data: {json.dumps(ev)}\n\n"
        finally:
            _HUB.detach(q)

    return StreamingResponse(gen(), media_type="text/event-stream")
