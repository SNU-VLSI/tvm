#!/usr/bin/env python3
"""pipelined(시뮬레이션) 빠른 실행 패널이 쓸 모자이크 시트를 만든다.

빠른 실행은 정확도를 보여주지 않는다 — 처리 "속도"만 보여주는 패널이라, 입력 데이터셋
전체를 훑는 그림이면 충분하다. 그래서 per-sample 추론 결과(mock txt)가 아니라
입력 배열만 있으면 된다.

산출물은 **완성된 모자이크 PNG 한 장**이다. 재생은 프론트가 이 그림의 타일을 순서대로
드러내는 방식이라, 프레임을 낱장으로 갖고 있을 필요가 없다(스프라이트 시트 불필요).
- 타일 = 전체 데이터셋에서 균등 간격(stride)으로 뽑은 입력 샘플. 앞부분만 쓰면 특정
  클래스에 몰릴 수 있어 반드시 전체에 걸쳐 고르게 뽑는다.
- 재생 길이는 프론트가 total_samples × ms_per_sample 로 계산한다. 타일 수와 무관하므로
  fps 가 떨어져도 총 시간은 정확히 유지된다.

타일 렌더는 backend/app.py 의 image() 와 같은 규칙을 쓴다(cifar_denorm / mfcc_heatmap /
raw01). 화면의 입력 패널과 모자이크가 다르게 보이면 안 되므로 바꿀 때 양쪽을 같이 고칠 것.

사용:
    python3 make_mosaic.py resnet8 --cols 100 --rows 20 --tile 16
    python3 make_mosaic.py kws     --cols 100 --rows 8  --tile-w 10 --tile-h 30
"""

import argparse
import os

import numpy as np
from PIL import Image

DEMO_ROOT = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(DEMO_ROOT, "fixtures")

# backend/app.py 와 동일한 상수 (CIFAR 정규화 / viridis 스톱)
_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(3, 1, 1)
_VIRIDIS_STOPS = np.array([
    [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
    [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
    [180, 222, 44], [253, 231, 37],
], dtype=np.float32)

# 워크로드별 입력 배열 후보 경로 + 렌더 방식.
# kws 전체(12105) 배열은 git-lfs 박제본이 100장뿐이라 칩에서 받아온 로컬 사본을 먼저 본다.
SOURCES = {
    "resnet8": {
        "display": "cifar_denorm",
        "paths": [os.path.join(FIX, "cifar10_test_images.npy")],
    },
    "kws": {
        "display": "mfcc_heatmap",
        "paths": [
            os.path.join(DEMO_ROOT, "dataset", "kws_sc", "images_full.npy"),
            os.path.join(FIX, "kws_staged_images.npy"),
        ],
    },
    "vww": {
        "display": "raw01",
        "paths": [os.path.join(FIX, "vww_staged_images.npy")],
    },
}


def _viridis(norm2d):
    x = np.clip(norm2d, 0.0, 1.0) * (len(_VIRIDIS_STOPS) - 1)
    lo = np.floor(x).astype(int)
    hi = np.minimum(lo + 1, len(_VIRIDIS_STOPS) - 1)
    frac = (x - lo)[..., None]
    rgb = _VIRIDIS_STOPS[lo] * (1 - frac) + _VIRIDIS_STOPS[hi] * frac
    return rgb.astype(np.uint8)


def render_tile(arr, display):
    arr = arr.astype(np.float32)
    if display == "mfcc_heatmap":
        m = arr[0] if arr.ndim == 3 else arr
        lo, hi = float(m.min()), float(m.max())
        norm = (m - lo) / (hi - lo) if hi > lo else np.zeros_like(m)
        return Image.fromarray(_viridis(norm))
    if display == "raw01":
        a = np.clip(arr, 0.0, 1.0)
        return Image.fromarray((a * 255).astype(np.uint8).transpose(1, 2, 0))
    a = np.clip(arr * _STD + _MEAN, 0.0, 1.0)          # cifar_denorm
    return Image.fromarray((a * 255).astype(np.uint8).transpose(1, 2, 0))


def main():
    ap = argparse.ArgumentParser(description="Build the pipelined-run mosaic sheet.")
    ap.add_argument("workload", choices=sorted(SOURCES))
    ap.add_argument("--cols", type=int, default=100)
    ap.add_argument("--rows", type=int, default=20)
    ap.add_argument("--tile", type=int, help="정사각 타일 한 변(px). tile-w/h 를 함께 주면 무시")
    ap.add_argument("--tile-w", type=int)
    ap.add_argument("--tile-h", type=int)
    ap.add_argument("--out", help="출력 PNG 경로 (기본 fixtures/<workload>_mosaic.png)")
    args = ap.parse_args()

    spec = SOURCES[args.workload]
    src = next((p for p in spec["paths"] if os.path.exists(p)), None)
    if src is None:
        raise SystemExit(f"입력 배열을 찾지 못했습니다: {spec['paths']}")

    images = np.load(src, mmap_mode="r")
    total = images.shape[0]
    ntiles = args.cols * args.rows
    if ntiles > total:
        raise SystemExit(f"타일 {ntiles}개 > 샘플 {total}개 — cols/rows 를 줄이세요")

    tw = args.tile_w or args.tile or 16
    th = args.tile_h or args.tile or tw

    # 전체에 걸쳐 균등 간격으로 뽑는다 (앞에서 N개 자르면 클래스 편향이 생긴다)
    idxs = np.linspace(0, total - 1, ntiles).round().astype(int)

    sheet = Image.new("RGB", (args.cols * tw, args.rows * th), (13, 17, 23))  # --bg 와 동일
    for k, i in enumerate(idxs):
        tile = render_tile(np.asarray(images[i]), spec["display"]).resize((tw, th), Image.NEAREST)
        sheet.paste(tile, ((k % args.cols) * tw, (k // args.cols) * th))

    out = args.out or os.path.join(FIX, f"{args.workload}_mosaic.png")
    sheet.save(out, format="PNG", optimize=True)
    kb = os.path.getsize(out) / 1024
    print(f"✅ {out}  ({sheet.width}x{sheet.height}px, {kb:.0f} KB)")
    print(f"   소스 {os.path.relpath(src, DEMO_ROOT)}  샘플 {total}개 중 {ntiles}개 균등 추출")
    print(f"   config 에 넣을 값: mosaic_cols={args.cols}  mosaic_rows={args.rows}  total_samples={total}")


if __name__ == "__main__":
    main()
