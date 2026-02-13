"""Generate scan register NPZ files for IMCFlow program_scan_reg.

This creates a directory containing per-IMCE NPZ files:
  scan_reg_files/
    imce_0_1.npz
    ...
    imce_3_4.npz

Each NPZ contains:
  - key: arr_0
  - value: numpy array of shape (64,), dtype=uint8

The host-side `program_scan_reg` loader expects exactly 64 bytes per IMCE and
performs a bit-level reversal to derive two short16 packets.

So the *byte pattern you write here is not the final register values*; it’s the
preimage under that decode. For testing “different per IMCE”, we just need
unique 64-byte payloads per IMCE.

Usage:
  python3 gen_scan_reg_npz.py --out-dir scan_reg_files

Common knobs:
  --pattern increment   : each IMCE gets 0..63 offset by a per-IMCE base
  --pattern constant    : each IMCE gets all bytes = (base & 0xff)
  --pattern random      : deterministic RNG using --seed

"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class ImceCoord:
    h: int
    w: int

    @property
    def name(self) -> str:
        return f"imce_{self.h}_{self.w}"


def iter_default_imces() -> Iterable[ImceCoord]:
    # 4x4 grid, w=1..4 (matches existing codegen conventions)
    for h in range(4):
        for w in range(1, 5):
            yield ImceCoord(h=h, w=w)


def _parse_manual_nibbles(manual: str) -> np.ndarray:
    """Parse a manual 64-byte (nibble-valued) payload.

    Accepts either:
      - underscore-separated: "09_08_0f_00_..." (64 tokens)
      - space/comma separated: "09 08 0f 00 ..." (64 tokens)
      - plain hex string: "09080f00..." (length 128)

    Returns:
      uint8 array of shape (64,), with all values 0..15.
    """
    s = manual.strip()
    if not s:
        raise ValueError("manual value string is empty")

    tokens: List[str]

    # 128 hex chars -> 64 bytes
    if all(c in "0123456789abcdefABCDEF" for c in s) and len(s) == 128:
        tokens = [s[i : i + 2] for i in range(0, 128, 2)]
    else:
        for sep in ["_", ",", " ", "\n", "\t"]:
            if sep in s:
                s = s.replace(sep, " ")
        tokens = [t for t in s.split(" ") if t]

    if len(tokens) != 64:
        raise ValueError(f"manual payload must have 64 bytes, got {len(tokens)}")

    out = np.empty((64,), dtype=np.uint8)
    for i, t in enumerate(tokens):
        t = t.strip()
        if t.lower().startswith("0x"):
            t = t[2:]
        if len(t) != 2:
            raise ValueError(f"token {i} must be 2 hex chars (00..0f), got '{t}'")
        v = int(t, 16)
        if not (0 <= v < 16):
            raise ValueError(f"token {i} out of range (00..0f): '{t}'")
        out[i] = np.uint8(v)
    return out


def make_payload(
    coord: ImceCoord,
    *,
    pattern: str,
    seed: int,
) -> np.ndarray:
    """Return uint8[64] payload for one IMCE."""

    # A stable, unique base per IMCE.
    # (h,w) ∈ {(0..3),(1..4)} -> idx 0..15
    idx = coord.h * 4 + (coord.w - 1)

    # Constrain all bytes to 0..15 (nibbles), like existing scan NPZ naming patterns
    # such as 09_08_..._0f_....
    if pattern == "increment":
        base = (idx * 3) & 0xF
        data = (np.arange(64, dtype=np.uint8) + np.uint8(base)) & np.uint8(0x0F)
        return data

    if pattern == "constant":
        base = (idx + 1) & 0xF
        return np.full((64,), np.uint8(base), dtype=np.uint8)

    if pattern == "random":
        # Deterministic per-IMCE stream.
        rng = np.random.default_rng(seed + idx)
        return rng.integers(0, 16, size=(64,), dtype=np.uint8)

    raise ValueError(f"Unknown pattern: {pattern}")


def load_scan_from_json(json_path: str) -> dict[ImceCoord, np.ndarray]:
    """Load scan values from JSON file.

    JSON format:
      {
        "0_1": {"h_id": 0, "w_id": 1, "pairs": {"0": val, "1": val, ...}},
        ...
      }

    Returns:
      dict mapping ImceCoord -> uint8[64] array
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    result = {}
    for key, entry in data.items():
        h_id = entry["h_id"]
        w_id = entry["w_id"]
        coord = ImceCoord(h=h_id, w=w_id)

        pairs = entry["pairs"]
        payload = np.zeros(64, dtype=np.uint8)
        for pair_idx, val in pairs.items():
            payload[int(pair_idx)] = np.uint8(val)

        result[coord] = payload

    return result


def write_npz(out_dir: str, coord: ImceCoord, payload: np.ndarray) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{coord.name}.npz")
    # Use arr_0 key (required by both numpy and cnpy loaders in this repo)
    np.savez(path, payload)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-IMCE scan NPZ files")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "scan_reg_files"),
        help="Output directory (default: utils/scan_reg_files)",
    )
    parser.add_argument(
        "--pattern",
        choices=["increment", "constant", "random"],
        default="increment",
        help="How to generate distinct 64-byte payloads per IMCE",
    )
    parser.add_argument(
        "--manual",
        type=str,
        default="",
        help=(
            "Manually specify the 64-byte payload as hex tokens (00..0f). "
            "Examples: '09_08_05_..._0f' or '090805...0f' (128 hex chars). "
            "If set, --pattern is ignored."
        ),
    )
    parser.add_argument(
        "--only-imce",
        type=str,
        default="",
        help=(
            "If set, only write one NPZ for this IMCE id (e.g. imce_2_3). "
            "By default writes all 16 IMCE files."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for --pattern random (default: 0)",
    )
    parser.add_argument(
        "--json",
        type=str,
        default="",
        help="JSON file containing per-IMCE scan values (overrides --pattern and --manual)",
    )

    args = parser.parse_args()

    # Select output IMCEs
    imces = list(iter_default_imces())
    if args.only_imce:
        imces = [c for c in imces if c.name == args.only_imce]
        if not imces:
            raise SystemExit(f"Unknown --only-imce '{args.only_imce}' (expected like imce_0_1)")

    json_payloads = None
    if args.json:
        json_payloads = load_scan_from_json(args.json)
        # JSON에 있는 IMCE만 처리하도록 필터링 (--only-imce가 없을 때)
        if not args.only_imce:
            imces = [c for c in imces if c in json_payloads]

    manual_payload = None
    if args.manual:
        manual_payload = _parse_manual_nibbles(args.manual)

    written: List[str] = []
    for coord in imces:
        if json_payloads is not None:
            payload = json_payloads[coord]
        elif manual_payload is not None:
            payload = manual_payload
        else:
            payload = make_payload(coord, pattern=args.pattern, seed=args.seed)
        if payload.shape != (64,) or payload.dtype != np.uint8:
            raise RuntimeError("Internal error: payload must be uint8[64]")
        written.append(write_npz(args.out_dir, coord, payload))

    print(f"Wrote {len(written)} NPZ files to: {os.path.abspath(args.out_dir)}")
    print("Example:")
    for p in written[:4]:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
