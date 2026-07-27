#!/usr/bin/env python3
"""_stagedN 생성기 — 칩에 올릴 N-샘플 입력(images+labels+sample_map)을 만든다.

fixtures 의 git-lfs 박제 npy 에서 앞 N 개를 잘라 staged 디렉토리를 만든다.
- images: fixtures/cifar10_test_images.npy[0:N]  (10000장 정규화 float32)
- labels: fixtures/cifar10_test_labels.npy[0:N]  (10000장 int64) ← 500 상한 없음
- sample_map.json: staged_to_original = [0,1,...,N-1] (순차 슬라이스)

칩 dump 는 순번(sample_0..N-1)으로 남고, 뷰어는 sample_map[staged_idx]=orig_idx 로
사진/라벨을 조회한다. 순차 슬라이스이므로 staged_idx == orig_idx 가 되어 정합이 자명하다.

사용:
    python3 make_staged.py 100            # demo/dataset/cifar10/_staged100/ 생성
    python3 make_staged.py 500 --out /path/to/dataset/cifar10   # 출력 루트 지정
    python3 make_staged.py 10000          # 전체 (밤샘 배치용)

생성 후 칩으로 전송하는 법은 --help 하단 참고. (이 스크립트는 로컬 파일만 만든다.)
"""

import argparse
import json
import os

import numpy as np

DEMO_ROOT = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(DEMO_ROOT, "fixtures")


def main():
    ap = argparse.ArgumentParser(description="Build _stagedN input dir for chip demo.")
    ap.add_argument("n", type=int, help="샘플 수 (1..10000)")
    ap.add_argument(
        "--out",
        default=os.path.join(DEMO_ROOT, "dataset", "cifar10"),
        help="staged 디렉토리를 놓을 루트 (default: demo/dataset/cifar10). "
        "칩 전송 시엔 codegen/dataset/cifar10 을 가리키게 하는 게 편하다.",
    )
    args = ap.parse_args()
    n = args.n

    images = np.load(os.path.join(FIX, "cifar10_test_images.npy"))   # [10000,3,32,32]
    labels = np.load(os.path.join(FIX, "cifar10_test_labels.npy"))   # [10000]
    total = images.shape[0]
    if not (1 <= n <= total):
        raise SystemExit(f"n must be in 1..{total} (got {n})")
    if labels.shape[0] != total:
        raise SystemExit(f"labels len {labels.shape[0]} != images {total}")

    staged_dir = os.path.join(args.out, f"_staged{n}")
    os.makedirs(staged_dir, exist_ok=True)

    np.save(os.path.join(staged_dir, "images.npy"), images[:n])
    np.save(os.path.join(staged_dir, "labels.npy"), labels[:n])
    sample_map = {
        "source_dir": args.out,
        "source_num_samples": int(total),
        "num_staged": int(n),
        "staged_to_original": list(range(n)),   # 순차 슬라이스
    }
    with open(os.path.join(staged_dir, "sample_map.json"), "w") as f:
        json.dump(sample_map, f, indent=2)

    print(f"✅ {staged_dir}")
    print(f"   images.npy {images[:n].shape}  labels.npy {labels[:n].shape}")
    print(f"   labels head {labels[:min(n,10)].tolist()}")
    print()
    print("칩 전송 (예):")
    print(f"   scp -P 1326 -r {staged_dir} \\")
    print(f"     root@147.46.117.99:/home/root/tvm/tvm_practice/test_imcflow/codegen/dataset/cifar10/")
    print("그리고 config/resnet8.yaml 의 chip.images_path/labels_path 를")
    print(f"   dataset/cifar10/_staged{n}/images.npy (labels.npy) 로,")
    print(f"   run.num_samples 를 {n} 로 맞춘다.")


if __name__ == "__main__":
    main()
