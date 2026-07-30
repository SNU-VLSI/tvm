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
import time

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
# 시작 워크로드. 실행 중에는 POST /workload/<name> 으로 바꾼다(set_workload 참조).
_WORKLOAD = os.environ.get("DEMO_WORKLOAD", "resnet8")
CONFIG_DIR = os.path.join(DEMO_ROOT, "config")
CONFIG_PATH = os.path.join(CONFIG_DIR, f"{_WORKLOAD}.yaml")


def available_workloads() -> list:
    """config/<name>.yaml 목록. 파일이 곧 워크로드 정의이므로 디렉토리를 그대로 읽는다.
    _tv_test.yaml 처럼 '_' 로 시작하는 건 작업용이라 뺀다."""
    try:
        names = [f[:-5] for f in os.listdir(CONFIG_DIR)
                 if f.endswith(".yaml") and not f.startswith("_")]
    except OSError:
        return [_WORKLOAD]
    # resnet8 을 앞에 두고 나머지는 이름순 — 버튼 순서가 매번 바뀌지 않게.
    return sorted(names, key=lambda n: (n != "resnet8", n))

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


def load_config(path: str = None) -> dict:
    with open(path or CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg["_demo_root"] = DEMO_ROOT
    return cfg


CFG = load_config()

# sample_map: staged 순번 -> 원본 데이터셋 인덱스. resnet8(full_npy) 에만 필수이고,
# kws/vww 는 staged 배열을 순번대로 바로 쓰므로 없을 수 있다(없으면 항등 매핑).
_SAMPLE_MAP = None


def _load_sample_map(cfg):
    sm = cfg.get("sample_map")
    if not sm:
        return None
    p = os.path.join(DEMO_ROOT, sm)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)["staged_to_original"]


_SAMPLE_MAP = _load_sample_map(CFG)

# ── 이미지 소스 (config.image_source, 환경변수 DEMO_IMAGE_SOURCE 로 오버라이드 가능) ──
# "full_npy"   : resnet8 노트북 기본. fixtures/cifar10_test_images.npy (10000장) orig_idx 조회.
# "staged_npy" : dataset/<name>/_staged/images.npy (staged 순번). resnet8/kws/vww 공용.
#                config.chip.dataset_name 이 dataset 하위 디렉토리명을 준다(cifar10|kws_sc|vww).
# ── 표시 방식 (config.display.kind) ── 워크로드마다 staged 배열의 물리 의미가 달라 렌더가 다르다:
#   "cifar_denorm" : [3,32,32] CIFAR 정규화 -> mean/std 역정규화 -> RGB 사진.
#   "raw01"        : [3,H,W] 이미 0..1 범위 -> ×255 RGB (vww COCO).
#   "mfcc_heatmap" : [1,49,10] 원시 MFCC(dB) -> per-image min-max 정규화 -> viridis heatmap (kws).
# 아래 5개는 _load_images(CFG) 가 한 번에 세팅한다(워크로드 전환 때 같이 갈아끼우려고).
#   _IMAGES / _INDEX_BY("orig"|"staged") / _IMAGE_SOURCE / _DISPLAY / _DATASET_NAME

def _find_staged_images(dataset_name, cfg=None):
    """staged 입력 배열 탐색. 우선순위:
      1) config.staged_images_fixture (demo/fixtures 에 박제한 것) — 노트북 기본.
         resnet8 이 cifar10_test_images.npy 를 git-lfs 로 박제하듯, kws/vww staged
         입력도 fixtures 로 박제해 `git lfs pull` 만으로 노트북에서 확보되게 한다.
      2) codegen/dataset/<name>/_staged/images.npy — 서버(원본 checkout) 폴백.
    """
    fx = (cfg or CFG).get("staged_images_fixture")
    cands = []
    if fx:
        cands.append(os.path.join(DEMO_ROOT, fx))
    cands += [
        os.path.join(CODEGEN_ROOT, "dataset", dataset_name, "_staged", "images.npy"),
        f"/root/project/tvm/tvm_practice/test_imcflow/codegen/dataset/{dataset_name}/_staged/images.npy",
    ]
    p = next((c for c in cands if os.path.exists(c)), None)
    return np.load(p) if p else None

def _load_images(cfg):
    """(images 배열, index_by, image_source, display_kind, dataset_name) 를 cfg 로부터 구성."""
    src = os.environ.get("DEMO_IMAGE_SOURCE") or cfg.get("image_source", "full_npy")
    kind = (cfg.get("display") or {}).get("kind", "cifar_denorm")
    name = (cfg.get("chip") or {}).get("dataset_name") or cfg.get("dataset_name", "cifar10")
    if src == "full_npy":
        p = os.path.join(DEMO_ROOT, "fixtures", "cifar10_test_images.npy")
        return (np.load(p) if os.path.exists(p) else None), "orig", src, kind, name
    if src == "staged_npy":
        return _find_staged_images(name, cfg), "staged", src, kind, name
    return None, "staged", src, kind, name


_IMAGES, _INDEX_BY, _IMAGE_SOURCE, _DISPLAY, _DATASET_NAME = _load_images(CFG)

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
            # 워크로드 전환 버튼용. 지금 무엇이 떠 있는지 + 고를 수 있는 목록.
            "workload": _WORKLOAD,
            "workloads": available_workloads(),
            # pipelined 빠른 실행 패널 설정. 없는 워크로드(vww)는 null 이고 프론트가 패널을 숨긴다.
            "pipelined": _pipelined_meta(),
            # display hints for the frontend (aspect ratio + pixelation) so the
            # input panel fits MFCC (49x10), COCO (96x96), or CIFAR (32x32).
            "display": {
                "kind": disp.get("kind", "cifar_denorm"),
                "aspect": disp.get("aspect", "1"),
                "pixelated": disp.get("pixelated", True),
            },
        }
    )


def _pipelined_meta():
    """config.pipelined 를 프론트가 쓸 형태로. 모자이크 PNG 가 실제로 있을 때만 노출한다
    (파일이 없으면 패널이 빈 캔버스로 떠서 고장난 것처럼 보인다)."""
    p = CFG.get("pipelined")
    if not p:
        return None
    rel = p.get("mosaic")
    if not rel or not os.path.exists(os.path.join(DEMO_ROOT, rel)):
        return None
    total = int(p["total_samples"])
    ms = float(p["ms_per_sample"])
    return {
        "ms_per_sample": ms,
        "total_samples": total,
        "duration_s": round(total * ms / 1000.0, 2),   # 재생 길이 = 실제 예상 소요시간
        "cols": int(p.get("mosaic_cols", 100)),
        "rows": int(p.get("mosaic_rows", 20)),
        "mosaic_url": "/mosaic",
    }


@app.get("/tuning")
def tuning():
    """전압 튜닝 스윕의 진행 상태 (dev 패널용).

    ⚠️ 여기서 PS 를 직접 RPC 로 읽지 않는다. 스윕 스크립트가 GPIB 세션을 점유하고 있어
    동시에 붙으면 측정에 간섭한다. 스윕이 원자적으로 갱신하는 상태 파일만 읽는다.
    파일이 없거나 오래됐으면 null — 프론트가 패널을 숨긴다.
    """
    path = os.environ.get("DEMO_TUNING_STATUS", "/tmp/ps_tuning_status.json")
    try:
        with open(path) as f:
            st = json.load(f)
    except (OSError, ValueError):
        return JSONResponse(None)
    # 스윕이 죽어도 파일은 남는다. 오래된 상태를 "진행 중"으로 보여주면 안 된다.
    st["age_s"] = round(time.time() - float(st.get("ts", 0)), 1)
    st["stale"] = st["age_s"] > 120
    return JSONResponse(st)


@app.get("/mosaic")
def mosaic():
    """현재 워크로드의 모자이크 PNG. 워크로드 전환 때 URL 이 같으므로 캐시를 막는다."""
    p = (CFG.get("pipelined") or {}).get("mosaic")
    if not p:
        return Response(status_code=404, content=b"no pipelined mosaic for this workload")
    path = os.path.join(DEMO_ROOT, p)
    if not os.path.exists(path):
        return Response(status_code=404, content=b"mosaic file not found")
    with open(path, "rb") as f:
        return Response(content=f.read(), media_type="image/png",
                        headers={"Cache-Control": "no-store"})


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
        self._proc = None       # chip 실행의 ssh Popen (mock 은 None)
        self._stop = False      # stop 요청 플래그. 워커가 라인마다 확인한다.

    def attach_or_start(self):
        """(내가 실행 소유자인가, 내 큐, 리플레이할 과거 이벤트) 반환."""
        with self._lock:
            owner = not self._active
            if owner:
                self._active = True
                self._history = []
                self.runs_started += 1
                self._stop = False
                self._proc = None
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

    def set_proc(self, proc):
        """runner 가 ssh Popen 을 만들면 콜백으로 받아 둔다 (stop 에서 terminate 하려고)."""
        with self._lock:
            self._proc = proc

    @property
    def stop_requested(self) -> bool:
        with self._lock:
            return self._stop

    def request_stop(self) -> bool:
        """실행 중이면 정지를 건다. 반환값 = 실제로 정지를 걸었는지.

        로컬 ssh terminate 만으로는 원격 프로세스가 남을 수 있어, 호출자가 이어서
        runner.kill_remote() 도 호출해야 한다 (chip 프로파일).
        """
        with self._lock:
            if not self._active:
                return False
            self._stop = True
            proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        return True

    def finish(self):
        with self._lock:
            self._active = False
            self._proc = None
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
        for line in runner.iter_lines(CFG, on_proc=hub.set_proc):
            if hub.stop_requested:   # mock/chip 공통 — 라인 경계에서 빠져나온다
                break
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
        if not hub.stop_requested:   # stop 으로 죽인 ssh 의 예외는 에러가 아니다
            hub.publish({"type": "error", "message": f"{type(e).__name__}: {e}"})
    finally:
        if hub.stop_requested:
            hub.publish({"type": "stopped"})
        hub.publish({"type": "done"})
        hub.finish()


@app.get("/status")
def status():
    """실행 중인지 확인용 (디버그/운영). 칩을 건드리지 않는다."""
    return JSONResponse({"running": _HUB.active, "runs_started": _HUB.runs_started})


def _stop_and_wait(timeout: float = 40.0) -> dict:
    """정지를 걸고 워커가 실제로 끝날 때까지 기다린다.

    끝날 때까지 기다리는 게 핵심: _active 가 False 가 되기 전에 새 연결이 들어오면
    죽어가는 실행에 붙어버린다(restart 가 조용히 실패). 원격 pkill 도 반드시 같이 —
    로컬 ssh 만 죽이면 칩에서 추론이 계속 돌아 다음 실행과 경합한다.
    """
    was_running = _HUB.request_stop()
    killed_remote = False
    if was_running and CFG.get("profile") == "chip":
        killed_remote = runner.kill_remote(CFG)
    deadline = time.monotonic() + timeout
    while _HUB.active and time.monotonic() < deadline:
        time.sleep(0.2)
    return {
        "was_running": was_running,
        "killed_remote": killed_remote,
        "idle": not _HUB.active,
    }


@app.post("/stop")
def stop_run():
    """실행 중인 추론을 중단한다. 실행 중이 아니면 was_running=False 로 무해하게 끝난다."""
    return JSONResponse(_stop_and_wait())


@app.post("/workload/{name}")
def set_workload(name: str):
    """워크로드를 바꾸고 처음부터 다시 시작할 수 있는 상태로 만든다.

    실행 중이면 먼저 정지시킨다 — 칩은 한 번에 하나만 돌릴 수 있고, 특히 워크로드마다
    바이너리(.resnet8/.kws/.vww)가 달라 이전 실행이 살아있는 채로 다른 모델을 띄우면
    칩에서 두 모델이 경합한다.

    실행 시작은 여기서 하지 않는다. 프론트가 /config 를 다시 읽고 /stream 에 재연결하면
    그 연결이 소유자가 되어 시작한다("첫 연결이 소유자" 규칙을 한 곳에만 둔다).
    """
    global CFG, CONFIG_PATH, _WORKLOAD, _SAMPLE_MAP
    global _IMAGES, _INDEX_BY, _IMAGE_SOURCE, _DISPLAY, _DATASET_NAME

    avail = available_workloads()
    if name not in avail:
        return JSONResponse({"error": f"unknown workload {name!r}", "available": avail},
                            status_code=404)

    r = _stop_and_wait()
    if not r["idle"]:
        return JSONResponse({**r, "switched": False,
                             "message": "이전 실행이 아직 정리되지 않았습니다"}, status_code=409)

    path = os.path.join(CONFIG_DIR, f"{name}.yaml")
    try:
        cfg = load_config(path)
        images, index_by, src, kind, dsname = _load_images(cfg)
    except Exception as e:   # config 오타/누락 npy 등 — 기존 워크로드를 유지한 채 실패를 알린다
        return JSONResponse({"switched": False, "workload": _WORKLOAD,
                             "error": f"{type(e).__name__}: {e}"}, status_code=400)

    CFG, CONFIG_PATH, _WORKLOAD = cfg, path, name
    _SAMPLE_MAP = _load_sample_map(cfg)
    _IMAGES, _INDEX_BY, _IMAGE_SOURCE, _DISPLAY, _DATASET_NAME = images, index_by, src, kind, dsname
    return JSONResponse({"switched": True, "workload": name,
                         "images_loaded": images is not None, **r})


@app.post("/restart")
def restart_run():
    """정지 후 새 실행을 띄운다. 프론트는 응답을 받고 EventSource 를 다시 연결한다.

    여기서 실행을 직접 시작하진 않는다 — /stream 재연결이 소유자가 되어 시작하므로
    (실행 소유 = 첫 연결) 게이트 규칙이 한 곳에만 남는다.
    """
    r = _stop_and_wait()
    if not r["idle"]:
        return JSONResponse({**r, "restarted": False,
                             "message": "이전 실행이 아직 정리되지 않았습니다"}, status_code=409)
    return JSONResponse({**r, "restarted": True})


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
