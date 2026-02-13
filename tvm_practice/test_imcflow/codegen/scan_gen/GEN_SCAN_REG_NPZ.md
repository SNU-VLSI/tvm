# `gen_scan_reg_npz.py` — Generate Scan Register NPZ Files

This script generates the **per-IMCE** NPZ files consumed by the host-side scan register programming kernel (`program_scan_reg`).

It writes files like:

```text
scan_reg_files/
  imce_0_1.npz
  imce_0_2.npz
  ...
  imce_3_4.npz
```

Each NPZ contains:

- key: `arr_0`
- value: `uint8[64]`

> Important: the runtime loader (`program_scan_reg`) will **bit-reverse** and repack these bytes into two `short16` scan packets. So the bytes in `arr_0` are the *encoded input*, not the final register values.

---

## Requirements / format

- Exactly **16** files by default (one per IMCE in a 4×4 grid)
- Naming convention matches the rest of the repo:
  - `imce_{h}_{w}.npz` where `h=0..3`, `w=1..4`
- `arr_0` must be:
  - shape `(64,)`
  - dtype `uint8`
- **Recommended**: keep each byte `0..15` (`00..0f`) to match existing scan NPZ patterns like:
  - `09_08_05_07_..._0f_... .npz`

---

## Quick start

From `tvm_practice/test_imcflow/codegen/utils`:

```bash
python3 gen_scan_reg_npz.py --out-dir scan_reg_files --pattern increment
```

This creates all 16 files.

---

## Auto-generated patterns

### 1) Increment pattern (default)

```bash
python3 gen_scan_reg_npz.py --out-dir scan_reg_files --pattern increment
```

- Each IMCE gets a different 64-byte pattern.
- All values are constrained to `0..15`.

### 2) Constant pattern

```bash
python3 gen_scan_reg_npz.py --out-dir scan_reg_files --pattern constant
```

- Each IMCE gets a constant nibble value (still unique per IMCE).

### 3) Random pattern (deterministic)

```bash
python3 gen_scan_reg_npz.py --out-dir scan_reg_files --pattern random --seed 123
```

- Each IMCE gets deterministic pseudo-random nibbles.
- Different IMCEs use different streams derived from `seed`.

---

## Manual values (write exactly what you want)

Use `--manual` to specify the 64 bytes directly.

### Accepted manual formats

All formats must describe **64 bytes**, each in the range `00..0f` (value < 16).

#### A) Underscore separated (most similar to the repo’s example style)

```bash
python3 gen_scan_reg_npz.py \
  --out-dir scan_reg_files \
  --manual "09_08_05_07_05_08_06_07_0e_0d_0e_0c_0f_0f_0f_0c_...(64 tokens total)..."
```

#### B) Space / comma separated

```bash
python3 gen_scan_reg_npz.py --out-dir scan_reg_files --manual "09 08 05 07 ..."
python3 gen_scan_reg_npz.py --out-dir scan_reg_files --manual "09,08,05,07,..."
```

#### C) Raw hex string (128 hex chars)

```bash
python3 gen_scan_reg_npz.py --out-dir scan_reg_files --manual "09080507050806070e0d0e0c0f0f0f0c...(128 hex chars total)..."
```

### Apply manual values to only one IMCE

```bash
python3 gen_scan_reg_npz.py \
  --out-dir scan_reg_files \
  --only-imce imce_0_1 \
  --manual "09_08_05_07_..."
```

This overwrites only `scan_reg_files/imce_0_1.npz`.

---

## Sanity-checking generated files

You can quickly inspect one file:

```bash
python3 - <<'PY'
import numpy as np
arr = np.load('scan_reg_files/imce_0_1.npz')['arr_0']
print(arr.shape, arr.dtype)
print('min/max:', int(arr.min()), int(arr.max()))
print('head:', [int(x) for x in arr[:16]])
PY
```

---

## Notes

- If `--manual` is provided, `--pattern` is ignored.
- If `--only-imce` is not provided, the script writes all 16 `imce_h_w.npz` files.
