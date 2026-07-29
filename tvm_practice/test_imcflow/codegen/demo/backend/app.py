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
# Workload selection: DEMO_WORKLOAD=resnet8|kws|vww picks config/<workload>.yaml.
# Defaults to resnet8 (original behavior). Each config carries its own task meta,
# chip binary_dir (.resnet8/.kws/.vww warmup-off dirs), and display block.
_WORKLOAD = os.environ.get("DEMO_WORKLOAD", "resnet8")
CONFIG_PATH = os.path.join(DEMO_ROOT, "config", f"{_WORKLOAD}.yaml")

# CIFAR-10 정규화 파라미터 (metadata.json transform). 역정규화해 사진 복원용.
_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(3, 1, 1)

# Small viridis colour-map (8 anchor stops, linearly interpolated) for the KWS
# MFCC heatmap — avoids a matplotlib dependency in the demo venv.
_VIRIDIS_STOPS = np.array([
    [ 68,   1,  84], [ 72,  40, 120], [ 62,  74, 137], [ 49, 104, 142],
    [ 38, 130, 142], [ 31, 158, 137], [ 53, 183, 121], [109, 205,  89],
    [180, 222,  44], [253, 231,  37],
], dtype=np.float32)

def _viridis(norm2d):
    """norm2d: [H,W] in 0..1 -> [H,W,3] uint8 viridis."""
    x = np.clip(norm2d, 0.0, 1.0) * (len(_VIRIDIS_STOPS) - 1)
    lo = np.floor(x).astype(int)
    hi = np.minimum(lo + 1, len(_VIRIDIS_STOPS) - 1)
    frac = (x - lo)[..., None]
    rgb = _VIRIDIS_STOPS[lo] * (1 - frac) + _VIRIDIS_STOPS[hi] * frac
    return rgb.astype(np.uint8)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["_demo_root"] = DEMO_ROOT
    return cfg


CFG = load_config()

# sample_map: staged 순번 -> 원본 데이터셋 인덱스. resnet8(full_npy) 에만 필수이고,
# kws/vww 는 staged 배열을 순번대로 바로 쓰므로 없을 수 있다(없으면 항등 매핑).
_SAMPLE_MAP = None
_sm = CFG.get("sample_map")
if _sm:
    _sm_path = os.path.join(DEMO_ROOT, _sm)
    if os.path.exists(_sm_path):
        with open(_sm_path) as f:
            _SAMPLE_MAP = json.load(f)["staged_to_original"]

# ── 이미지 소스 (config.image_source, 환경변수 DEMO_IMAGE_SOURCE 로 오버라이드 가능) ──
# "full_npy"   : resnet8 노트북 기본. fixtures/cifar10_test_images.npy (10000장) orig_idx 조회.
# "staged_npy" : dataset/<name>/_staged/images.npy (staged 순번). resnet8/kws/vww 공용.
#                config.chip.dataset_name 이 dataset 하위 디렉토리명을 준다(cifar10|kws_sc|vww).
# ── 표시 방식 (config.display.kind) ── 워크로드마다 staged 배열의 물리 의미가 달라 렌더가 다르다:
#   "cifar_denorm" : [3,32,32] CIFAR 정규화 -> mean/std 역정규화 -> RGB 사진.
#   "raw01"        : [3,H,W] 이미 0..1 범위 -> ×255 RGB (vww COCO).
#   "mfcc_heatmap" : [1,49,10] 원시 MFCC(dB) -> per-image min-max 정규화 -> viridis heatmap (kws).
_IMAGE_SOURCE = os.environ.get("DEMO_IMAGE_SOURCE") or CFG.get("image_source", "full_npy")
_DISPLAY = (CFG.get("display") or {}).get("kind", "cifar_denorm")
_DATASET_NAME = (CFG.get("chip") or {}).get("dataset_name") or CFG.get("dataset_name", "cifar10")
_IMAGES = None
_INDEX_BY = "staged"    # "orig" (full_npy: orig_idx 로 조회) | "staged" (staged 순번)

def _find_staged_images(dataset_name):
    """staged 입력 배열 탐색. 우선순위:
      1) config.staged_images_fixture (demo/fixtures 에 박제한 것) — 노트북 기본.
         resnet8 이 cifar10_test_images.npy 를 git-lfs 로 박제하듯, kws/vww staged
         입력도 fixtures 로 박제해 `git lfs pull` 만으로 노트북에서 확보되게 한다.
      2) codegen/dataset/<name>/_staged/images.npy — 서버(원본 checkout) 폴백.
    """
    fx = CFG.get("staged_images_fixture")
    cands = []
    if fx:
        cands.append(os.path.join(DEMO_ROOT, fx))
    cands += [
        os.path.join(CODEGEN_ROOT, "dataset", dataset_name, "_staged", "images.npy"),
        f"/root/project/tvm/tvm_practice/test_imcflow/codegen/dataset/{dataset_name}/_staged/images.npy",
    ]
    p = next((c for c in cands if os.path.exists(c)), None)
    return np.load(p) if p else None

if _IMAGE_SOURCE == "full_npy":
    _p = os.path.join(DEMO_ROOT, "fixtures", "cifar10_test_images.npy")
    _IMAGES = np.load(_p) if os.path.exists(_p) else None
    _INDEX_BY = "orig"
elif _IMAGE_SOURCE == "staged_npy":
    _IMAGES = _find_staged_images(_DATASET_NAME)
    _INDEX_BY = "staged"

app = FastAPI(title="IMCFlow chip demo")


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(DEMO_ROOT, "frontend", "index.html")) as f:
        return f.read()


@app.get("/config")
def config():
    t = CFG["task"]
    disp = CFG.get("display") or {}
    return JSONResponse(
        {
            "title": t["title"],
            "classes": t["classes"],
            "num_classes": t["num_classes"],
            "num_samples": CFG["run"]["num_samples"],
            "profile": CFG.get("profile", "mock"),
            # display hints for the frontend (aspect ratio + pixelation) so the
            # input panel fits MFCC (49x10), COCO (96x96), or CIFAR (32x32).
            "display": {
                "kind": disp.get("kind", "cifar_denorm"),
                "aspect": disp.get("aspect", "1"),
                "pixelated": disp.get("pixelated", True),
            },
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

    if _INDEX_BY == "orig" and _SAMPLE_MAP is not None:
        idx = _SAMPLE_MAP[staged_idx] if staged_idx < len(_SAMPLE_MAP) else staged_idx
    else:
        idx = staged_idx

    arr = _IMAGES[idx].astype(np.float32)

    if _DISPLAY == "mfcc_heatmap":
        # [1,49,10] (or [49,10]) raw MFCC -> per-image min-max -> viridis heatmap.
        m = arr[0] if arr.ndim == 3 else arr           # [49,10]
        lo, hi = float(m.min()), float(m.max())
        norm = (m - lo) / (hi - lo) if hi > lo else np.zeros_like(m)
        rgb = _viridis(norm)                            # [49,10,3] uint8
        img = Image.fromarray(rgb)
    elif _DISPLAY == "raw01":
        # [3,H,W] already in 0..1 -> RGB photo (vww COCO).
        a = np.clip(arr, 0.0, 1.0)
        img = Image.fromarray((a * 255).astype(np.uint8).transpose(1, 2, 0))
    else:  # "cifar_denorm"
        a = arr * _STD + _MEAN
        a = np.clip(a, 0.0, 1.0)
        img = Image.fromarray((a * 255).astype(np.uint8).transpose(1, 2, 0))

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
                if _SAMPLE_MAP is not None and ev["idx"] < len(_SAMPLE_MAP):
                    ev["orig_idx"] = _SAMPLE_MAP[ev["idx"]]
                else:
                    ev["orig_idx"] = ev["idx"]
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
