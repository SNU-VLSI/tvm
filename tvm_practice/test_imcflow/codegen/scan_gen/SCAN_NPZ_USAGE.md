# Scan Register Code Generation with NPZ Files

## Overview

The `scan_codegen.py` script supports loading scan register values from NPZ files, matching the format used in the imcflow simulator (`acim.py:ScanData` and `testing.py:parse_scan`).

**NEW**: Each IMCE now receives unique scan values from its own NPZ file!

## NPZ File Format

Each NPZ file contains scan data for **one IMCE**:
- **Key**: `arr_0`
- **Value**: 64-element numpy array (uint8/int8)
- **Layout**: 64 bytes = 2 scan registers = 2 packets of 32 bytes each
  - Packet 0 (scan reg 0): bytes 0-31 (16 int16 values, little-endian)
  - Packet 1 (scan reg 1): bytes 32-63 (16 int16 values, little-endian)

## Usage Patterns

### 1. Single NPZ File (Same Values for All IMCEs)

All 16 IMCEs receive the same scan values from one NPZ file:

```bash
python scan_codegen.py --run-policy-gen \
    --scan-npz /path/to/scan_values.npz
```

**Result**: 32 packets total (2 packets × 16 IMCEs), all IMCEs get identical values from the same NPZ file

---

### 2. Directory with Per-IMCE NPZ Files

Each IMCE gets different scan values from its own NPZ file:

```bash
python scan_codegen.py --run-policy-gen \
    --scan-npz-dir /path/to/scan_npz_directory
```

**Directory structure required**:
```
scan_npz_directory/
├── imce_0_1.npz  # IMCE at (h=0, w=1)
├── imce_0_2.npz  # IMCE at (h=0, w=2)
├── imce_0_3.npz  # IMCE at (h=0, w=3)
├── imce_0_4.npz  # IMCE at (h=0, w=4)
├── imce_1_1.npz  # IMCE at (h=1, w=1)
├── ...
└── imce_3_4.npz  # IMCE at (h=3, w=4)
```

**Result**: 32 packets total (2 packets × 16 IMCEs), with different values per IMCE

---

### 3. Comma-Separated List of NPZ Files

Specify exactly 16 NPZ files (one per IMCE):

```bash
python scan_codegen.py --run-policy-gen \
    --scan-npz "file1.npz,file2.npz,file3.npz,...,file16.npz"
```

**Result**: 32 packets total, with values loaded from the specified files in order

---

### 4. Default Generated Values (No NPZ)

Generate test values automatically:

```bash
python scan_codegen.py --run-policy-gen --scan-count 32
```

**Result**: Each packet contains `[i % 256] * 16` where i is the packet index

---

## Example: Testing with Real NPZ File

```bash
# Use a sample NPZ file from the imcflow test suite
python scan_codegen.py --run-policy-gen \
    --scan-npz /root/project/imcflow/pmap/ISA_sim/multi_core/test/test_compiler/scan_npz/06_03_04_06_00_06_06_01_04_0d_05_08_04_08_01_0a_04_02_02_0b_0f_05_09_01_01_0b_04_04_08_02_04_03_02_02_01_01_00_01_00_00_00_03_02_02_05_00_04_01_00_00_05_04_00_01_04_0b_03_04_02_04_0a_05_0b_01.npz
```

**Generated files**:
- `build/scan_reg/imce.cpp` - IMCE code (each receives 2 packets)
- `build/scan_reg/inode.cpp` - INode code (sends 2 packets to each IMCE group)
- `build/scan_reg/scan_reg_kernel.cc` - Host kernel with loaded NPZ values
- Policy table binaries for all nodes

---

## Verification

Check the loaded values in the generated kernel:

```bash
grep -A 5 "scan_reg_scan_data\[\]" build/scan_reg/scan_reg_kernel.cc
```

Example output:
```c
static const short16 scan_reg_scan_data[] = {
    { 774, 1540, 1536, 262, 3332, 2053, 2052, 2561, 516, 2818, 1295, 265, 2817, 1028, 520, 772 },  // packet 0
    { 514, 257, 256, 0, 768, 514, 5, 260, 0, 1029, 256, 2820, 1027, 1026, 1290, 267 },  // packet 1
    ...
};
```

---

## Important Notes

1. **Per-IMCE Addressing**: Each IMCE receives unique scan values
   - Each INode sends 8 packets total (4 IMCEs × 2 packets each)
   - imce_0_1 gets packets from offset 0 (bytes 0-63)
   - imce_0_2 gets packets from offset 64 (bytes 64-127)
   - imce_0_3 gets packets from offset 128 (bytes 128-191)
   - ...and so on for all 16 IMCEs

2. **Packet Count**: When using NPZ files, the total packet count is always 32:
   - 16 IMCEs × 2 packets per IMCE = 32 packets total = 1024 bytes

3. **Memory Layout**:
   - Scan data is stored in a contiguous 1024-byte block
   - Each IMCE's data occupies 64 bytes (2 × 32-byte packets)
   - Packets are organized sequentially by IMCE index (0-15)

---

## Python API

You can also use the loading functions programmatically:

```python
from scan_codegen import load_scan_values_from_npz, load_scan_values_from_directory

# Load from single file
scan_values = load_scan_values_from_npz("scan_data.npz", imce_count=16)

# Load from directory
scan_values = load_scan_values_from_directory("/path/to/npz_dir")

# Load from list of files
scan_values = load_scan_values_from_npz([
    "imce_0_1.npz", "imce_0_2.npz", ..., "imce_3_4.npz"
], imce_count=16)
```

Each function returns a list of scan packets (32 packets for 16 IMCEs, each packet is a list of 16 int16 values).
