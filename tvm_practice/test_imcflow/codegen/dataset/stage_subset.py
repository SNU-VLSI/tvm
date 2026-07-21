#!/usr/bin/env python3
"""Stage only the samples a chip run will actually evaluate.

run_dataset_eval.sh used to scp the whole dataset/ dir to the chip every run.
For a deep model like VWW that is 1.2GB (10961 samples) even though the loop
only evaluates ~100 of them. This slices images.npy/labels.npy down to just the
requested rows, writing them 0-based-remapped into a staging dir, so the
transfer and the on-chip load touch only the samples in play. Works for both
num_samples mode (first K rows) and explicit-indices mode (those rows).

The C dataset binary indexes images.npy by ABSOLUTE row, so once we drop rows we
must renumber: the staged file holds the selected rows in order 0..K-1 and the
caller passes num_samples=K (not the original indices). sample_map.json records
staged_row -> original_row so results stay traceable back to the source dataset.

Usage:
    stage_subset.py --src-dir dataset/vww --out-dir dataset/vww/_staged \
        --num-samples 100
    stage_subset.py --src-dir dataset/vww --out-dir dataset/vww/_staged \
        --indices 1147,6158,4318
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _parse_indices(text: str) -> list[int]:
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-dir", required=True, help="dir holding images.npy + labels.npy")
    ap.add_argument("--out-dir", required=True, help="staging dir to write the subset into")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--num-samples", type=int, help="take the first K rows")
    grp.add_argument("--indices", type=str, help="comma-separated original row indices")
    args = ap.parse_args()

    images_path = os.path.join(args.src_dir, "images.npy")
    labels_path = os.path.join(args.src_dir, "labels.npy")
    for p in (images_path, labels_path):
        if not os.path.exists(p):
            print(f"stage_subset: missing {p}", file=sys.stderr)
            return 2

    # mmap so we never pull a 1.2GB array fully into RAM just to slice it.
    images = np.load(images_path, mmap_mode="r")
    labels = np.load(labels_path)
    n = images.shape[0]
    if labels.shape[0] != n:
        print(f"stage_subset: {n} images but {labels.shape[0]} labels", file=sys.stderr)
        return 2

    if args.indices is not None:
        original = _parse_indices(args.indices)
        for i in original:
            if i < 0 or i >= n:
                print(f"stage_subset: index {i} out of range [0,{n})", file=sys.stderr)
                return 2
    else:
        k = args.num_samples
        if k <= 0:
            print(f"stage_subset: --num-samples must be > 0 (got {k})", file=sys.stderr)
            return 2
        k = min(k, n)
        original = list(range(k))

    # np fancy-indexing on the mmap materializes only the selected rows.
    sel_images = np.ascontiguousarray(images[original])
    sel_labels = np.ascontiguousarray(labels[original])

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "images.npy"), sel_images)
    np.save(os.path.join(args.out_dir, "labels.npy"), sel_labels)
    with open(os.path.join(args.out_dir, "sample_map.json"), "w") as f:
        json.dump(
            {
                "source_dir": os.path.abspath(args.src_dir),
                "source_num_samples": int(n),
                "num_staged": len(original),
                # staged_row (list position) -> original_row in the source dataset
                "staged_to_original": original,
            },
            f,
            indent=2,
        )

    # stdout is the staged count; run_dataset_eval.sh reads it to set SAMPLES_ARG.
    print(len(original))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
