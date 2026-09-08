#!/usr/bin/env python3
"""Generate / swap the max-TOPS-kernel activation input: ones <-> uniform uint4.

The board's host binary loads test_inputs/conv_input.bin (raw uint8 NCHW,
1x256x127x127) from the eval_dir ON THE BOARD, so no recompile is needed --
only this file changes. QUADRU(Q1) feeds 4 bitplanes per conversion, so the
hardware consumes bits[3:0]: uniform 0..15 (uint4 full range) gives ~50%%
average bit activity per plane, matching the int4 full-range uniform weights.

Local artifacts (next to the original, never overwriting it):
  conv_input.rand4.bin / .npy   uniform uint4, fixed seed 42
  conv_input.ones.bin           copy of the original all-ones input

Board swap (explicit flags; each scp's then verifies via md5):
  --push rand4 | --push ones    replace board conv_input.bin
  (original board file is first backed up once to conv_input.ones.bin)

Usage:
  python3 experiments/gen_rand4_input.py            # generate local files only
  python3 experiments/gen_rand4_input.py --push rand4
  python3 experiments/gen_rand4_input.py --push ones     # restore
"""
import argparse, hashlib, os, subprocess, sys

import numpy as np

CODEGEN = "/root/project/tvm/.claude/worktrees/step-freerun-interleave/tvm_practice/test_imcflow/codegen"
EVL = "one_1x1_quant_16imce_max127_evl.linux.bugfixoff"
TI = os.path.join(CODEGEN, "eval_dir", EVL, "test_inputs")
BOARD_TI = f"/home/root/tvm/tvm_practice/test_imcflow/codegen/eval_dir/{EVL}/test_inputs"
SSH = ["ssh", "-o", "BatchMode=yes", "-p", "1326", "root@147.46.117.99"]
SCP = ["scp", "-O", "-o", "BatchMode=yes", "-P", "1326"]
SHAPE, SEED = (1, 256, 127, 127), 42

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

def generate():
    orig = os.path.join(TI, "conv_input.bin")
    ones_bak = os.path.join(TI, "conv_input.ones.bin")
    rand_bin = os.path.join(TI, "conv_input.rand4.bin")
    a = np.load(os.path.join(TI, "conv_input.npy"))
    assert a.shape == SHAPE and a.dtype == np.uint8, (a.shape, a.dtype)
    if not os.path.exists(ones_bak):
        with open(orig, "rb") as s, open(ones_bak, "wb") as d: d.write(s.read())
    rng = np.random.default_rng(SEED)
    r = rng.integers(0, 16, size=SHAPE, dtype=np.uint8)   # uint4 full range
    r.tofile(rand_bin)
    np.save(os.path.join(TI, "conv_input.rand4.npy"), r)
    assert os.path.getsize(rand_bin) == os.path.getsize(orig)
    print(f"rand4: min={r.min()} max={r.max()} mean={r.mean():.3f} "
          f"(ideal 7.5)  zero%={100*(r==0).mean():.2f} (ideal 6.25)")
    print(f"bit-activity per plane: " + " ".join(
        f"b{b}={100*((r>>b)&1).mean():.1f}%" for b in range(4)))
    print("local files ready:", rand_bin)
    return rand_bin, ones_bak

def push(which, rand_bin, ones_bak):
    src = rand_bin if which == "rand4" else ones_bak
    # one-time board-side backup of the pristine ones input
    subprocess.run(SSH + [f"[ -f {BOARD_TI}/conv_input.ones.bin ] || "
                          f"cp {BOARD_TI}/conv_input.bin {BOARD_TI}/conv_input.ones.bin"],
                   check=True, timeout=30)
    subprocess.run(SCP + [src, f"root@147.46.117.99:{BOARD_TI}/conv_input.bin"],
                   check=True, timeout=120)
    want = md5(src)
    r = subprocess.run(SSH + [f"md5sum {BOARD_TI}/conv_input.bin"],
                       capture_output=True, text=True, timeout=60)
    got = r.stdout.split()[0] if r.stdout else "?"
    if got != want:
        sys.exit(f"md5 mismatch after push: board={got} local={want}")
    print(f"board conv_input.bin <- {which} (md5 {got[:12]} ok)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", choices=["rand4", "ones"])
    args = ap.parse_args()
    rand_bin, ones_bak = generate()
    if args.push:
        push(args.push, rand_bin, ones_bak)
