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

import io
import json
import os

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

# staged 이미지 (칩 입력과 같은 소스). 서버 검증용 사진 복원에 사용.
# 실제 데모(노트북)에서는 로컬 원본 CIFAR-10 을 쓰지만, 이 서버 검증에서는 staged images.npy 를 복원한다.
# worktree 는 코드만 복사되고 대용량 데이터는 원본 checkout 에만 있으므로 두 경로를 순서대로 탐색한다.
_IMAGE_CANDIDATES = [
    os.path.join(CODEGEN_ROOT, "dataset", "cifar10", "_staged", "images.npy"),
    "/root/project/tvm/tvm_practice/test_imcflow/codegen/dataset/cifar10/_staged/images.npy",
]
_STAGED_IMAGES_PATH = next((p for p in _IMAGE_CANDIDATES if os.path.exists(p)), None)
_IMAGES = np.load(_STAGED_IMAGES_PATH) if _STAGED_IMAGES_PATH else None

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
    """staged 순번의 원본 사진(32x32 PNG). 역정규화해 8-bit 복원."""
    if _IMAGES is None:
        return Response(status_code=404, content=b"staged images.npy not found")
    from PIL import Image

    arr = _IMAGES[staged_idx].astype(np.float32)  # [3,32,32] normalized
    if arr.shape[0] == 3:
        arr = arr * _STD + _MEAN  # 역정규화
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8).transpose(1, 2, 0)  # HWC
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/stream")
def stream():
    """SSE. 라인 스트림을 파싱해 샘플/진행 이벤트를 순차 push."""

    def gen():
        parser = ChipStreamParser()
        for line in runner.iter_lines(CFG):
            ev = parser.feed(line)
            if ev is None:
                continue
            if ev["type"] == "sample":
                orig = _SAMPLE_MAP[ev["idx"]] if ev["idx"] < len(_SAMPLE_MAP) else ev["idx"]
                ev["orig_idx"] = orig
            yield f"data: {json.dumps(ev)}\n\n"
        yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(gen(), media_type="text/event-stream")
