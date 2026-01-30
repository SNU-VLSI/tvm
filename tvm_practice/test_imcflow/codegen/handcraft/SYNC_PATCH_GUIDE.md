# Sync Patch Application Guide for IMCFlow imce.cpp

This guide documents the methodology for applying sync-related code patches from `before_pnr` to current imce.cpp files when node mappings have changed.

## Overview

When hardware node mappings change between versions, sync primitives (`SETFLAG`, `STANDBY`) need to be re-applied with updated node IDs. This guide provides a systematic approach to identify, map, and apply these patches.

## Background

### IMCE/INODE ID Mapping
```
inode_0_0 = 0      imce_0_1 = 1      imce_0_2 = 2      imce_0_3 = 3      imce_0_4 = 4
inode_1_0 = 5      imce_1_1 = 6      imce_1_2 = 7      imce_1_3 = 8      imce_1_4 = 9
inode_2_0 = 10     imce_2_1 = 11     imce_2_2 = 12     imce_2_3 = 13     imce_2_4 = 14
inode_3_0 = 15     imce_3_1 = 16     imce_3_2 = 17     imce_3_3 = 18     imce_3_4 = 19
```

### Sync Primitives
- `__builtin_IMCE_SETFLAG(1)` / `SETFLAG(0)`: Mark critical sections
- `__builtin_IMCE_STANDBY(N, 1)`: Wait for node with ID=N
  - The number N changes based on which IMCE node is being waited for

## Step-by-Step Methodology

### Phase 1: Analysis and Mapping

#### 1.1 Identify Sync Patterns in before_pnr

**Use grep to find all sync primitives:**
```bash
cd before_pnr/resnet8_subset31_pretrained_orig_evl/build/tvmgen_default_tvmgen_default_imcflow_main_46_round_imcflow_region1_main_0/

# Find all SETFLAG usage
grep -n "SETFLAG" imce.cpp

# Find all STANDBY usage with context
grep -n -B2 -A2 "STANDBY" imce.cpp
```

**Expected patterns:**
1. **Shared input sync**: `SETFLAG(1)` + `STANDBY(inode_id, 1)` + `SETFLAG(0)` before RECV, then `STANDBY(imce_id, 1)` before MM_QUANT
2. **Multi-RECV sync**: `SETFLAG(1)` before first RECV, `SETFLAG(0)` after last RECV, `STANDBY(imce_id, 1)` before MM_QUANT
3. **SEND sync**: `STANDBY(imce_id, 1)` before SEND operations

#### 1.2 Find Which Nodes Have Sync Patches

**Use grep with context to identify nodes:**
```bash
# Find which imce nodes have SETFLAG
grep -B10 "SETFLAG" imce.cpp | grep "else if.*imce_"

# Find which imce nodes have STANDBY
grep -B20 "STANDBY" imce.cpp | grep "else if.*imce_"
```

**Document findings:**
```
Example from our work:
- imce_3_4: SETFLAG + STANDBY(0,1) + STANDBY(18,1) pattern
- imce_3_2: SETFLAG + STANDBY(16,1) pattern
- imce_3_3: Multiple STANDBY(17,1) before SEND operations
```

#### 1.3 Identify TensorEdge Comments for Mapping

**Key insight**: TensorEdge comments are reliable identifiers that survive node remapping.

**Use grep to find TensorEdge patterns:**
```bash
# Find specific TensorEdge that identifies a sync location
grep -n "TensorEdge((-13, odata), (38, data))" imce.cpp

# Find batch_norm outputs (common sync points)
grep -n "TensorEdge(((39, 36), odata), (40, data))" imce.cpp
```

**Create mapping table:**
```
| before_pnr Node | TensorEdge Pattern | Functionality |
|-----------------|-------------------|---------------|
| imce_3_4 | (-13, odata), (38, data) | MinmaxQuant receiving shared input |
| imce_3_2 | ((39, 36), odata), (40, data) | MinmaxQuant receiving from conv |
| imce_3_3 | ((39, 36), odata), (40, data) | Conv + BatchNorm outputs |
```

#### 1.4 Find Corresponding Locations in Current Version

**Use same TensorEdge patterns to find new locations:**
```bash
cd ../../resnet8_subset31_pretrained_orig_evl/build/tvmgen_default_tvmgen_default_imcflow_main_46_round_imcflow_region1_main_0/

# Find same TensorEdge in current version
grep -n "TensorEdge((-13, odata), (38, data))" imce.cpp
# Output: Line 72: ... inode_0_0 -> imce_0_1
# This means imce_3_4 (before_pnr) → imce_0_1 (current)

grep -n "TensorEdge(((39, 36), odata), (40, data))" imce.cpp
# Check which imce nodes are receivers vs senders
```

**Build complete mapping:**
```
before_pnr → current
imce_3_4 → imce_0_1  (TensorEdge: (-13, odata), (38, data))
imce_3_2 → imce_2_1  (TensorEdge: ((39, 36), odata), (40, data), receiver)
imce_3_3 → imce_1_1  (TensorEdge: ((39, 36), odata), (40, data), sender)
```

### Phase 2: STANDBY ID Calculation

**Critical**: When node mappings change, STANDBY IDs must be updated.

#### 2.1 Identify What Each STANDBY Waits For

**In before_pnr, check comments and context:**
```bash
# Look at STANDBY with surrounding context
grep -B5 -A5 "STANDBY(18, 1)" before_pnr/.../imce.cpp
```

**Example analysis:**
- `STANDBY(18, 1)` in imce_3_4 → waiting for imce_3_3 (ID=18)
- `STANDBY(17, 1)` in imce_3_3 → waiting for imce_3_2 (ID=17)
- `STANDBY(16, 1)` in imce_3_2 → waiting for imce_3_1 (ID=16)
- `STANDBY(0, 1)` → waiting for inode_0_0 (ID=0, doesn't change)

#### 2.2 Calculate New STANDBY IDs

**Apply node mapping to STANDBY targets:**
```
Old: STANDBY(18, 1) waiting for imce_3_3
New: imce_3_3 → imce_1_1 (ID=6)
Result: STANDBY(6, 1)

Old: STANDBY(17, 1) waiting for imce_3_2
New: imce_3_2 → imce_2_1 (ID=11)
Result: STANDBY(11, 1)

Old: STANDBY(16, 1) waiting for imce_3_1
New: imce_3_1 → imce_3_1 (ID=16, unchanged)
Result: STANDBY(16, 1)
```

### Phase 3: Applying Patches

#### 3.1 Pattern 1: Shared Input Sync (e.g., imce_0_1)

**Grep command to find exact location:**
```bash
grep -n -A5 "TensorEdge((-13, odata), (38, data))" imce.cpp | grep -A5 "RECV"
```

**Patch to apply:**
```cpp
// BEFORE the RECV line, add:
__builtin_IMCE_SETFLAG(1); // sync one of imce_X_Y, and imce_A_B
__builtin_IMCE_STANDBY(0, 1);
__builtin_IMCE_SETFLAG(0);

// BEFORE the MM_QUANT line, add:
__builtin_IMCE_STANDBY(N, 1); // N = ID of sender node
```

**Tip**: Use Read tool with line numbers from grep to get exact context for Edit tool.

#### 3.2 Pattern 2: Multi-RECV Sync (e.g., imce_2_1)

**Grep command to find 4x RECV pattern:**
```bash
grep -n -A3 "TensorEdge(((39, 36), odata), (40, data))" imce.cpp | grep "RECV" | head -4
```

**Patch to apply:**
```cpp
// BEFORE first RECV:
__builtin_IMCE_SETFLAG(1);

// AFTER last (4th) RECV:
__builtin_IMCE_SETFLAG(0);

// BEFORE MM_QUANT:
__builtin_IMCE_STANDBY(N, 1);
```

#### 3.3 Pattern 3: Multiple SEND Sync (e.g., imce_1_1)

**Grep command to find all batch_norm + SEND locations:**
```bash
# Find all batch_norm blocks
grep -n "endgenerate: batch_norm" imce.cpp

# Check if followed by SEND
grep -n -A1 "endgenerate: batch_norm" imce.cpp | grep "SEND"
```

**Count locations:**
```bash
# Count how many batch_norm → SEND patterns exist
grep -A1 "endgenerate: batch_norm" imce.cpp | grep -c "SEND"
```

**Patch to apply (at each location):**
```cpp
// endgenerate: batch_norm
__builtin_IMCE_STANDBY(N, 1);  // N = ID of receiver node
__builtin_IMCE_SEND(...);
```

**Tip**: Use grep with row_group patterns to identify each location:
```bash
grep -n "row_group0_col_group0" imce.cpp
grep -n "row_group0_col_group1" imce.cpp
grep -n "row_group0_col_group2" imce.cpp
grep -n "row_group1_col_group0" imce.cpp
grep -n "row_group1_col_group1" imce.cpp
grep -n "row_group1_col_group2" imce.cpp
```

### Phase 4: Verification

#### 4.1 Verify All Patches Applied

**Check sync primitives count:**
```bash
# Count SETFLAG in current vs before_pnr
grep -c "SETFLAG" imce.cpp
grep -c "SETFLAG" before_pnr/.../imce.cpp

# Count STANDBY in current vs before_pnr
grep -c "STANDBY" imce.cpp
grep -c "STANDBY" before_pnr/.../imce.cpp
```

**Note**: Counts may differ due to node mapping changes, but should be close.

#### 4.2 Verify STANDBY IDs

**Check all STANDBY IDs are valid:**
```bash
# List all STANDBY calls with line numbers
grep -n "STANDBY(" imce.cpp

# Verify IDs are in valid range (0-19)
grep "STANDBY([0-9]\+," imce.cpp | sed 's/.*STANDBY(\([0-9]\+\).*/\1/' | sort -u
```

#### 4.3 Use view_diff.py for Comparison

**Compare with original (before sync patches):**
```bash
cd ../..  # Go back to codegen directory
python view_diff.py resnet8_subset31_pretrained_orig_evl imce --region region1
```

## Efficient Workflow Tips

### Tip 1: Use Grep Pipelines for Quick Analysis

```bash
# Find all imce blocks with their line ranges
grep -n "else if (hid == [0-9] && wid == [0-9])" imce.cpp

# Find all TensorEdge patterns with node info
grep -o "TensorEdge([^)]*)" imce.cpp | sort -u

# Find specific patterns across multiple files
grep -r "SETFLAG" before_pnr/ --include="imce.cpp" -n
```

### Tip 2: Create a Mapping Spreadsheet First

Before editing, create a table:
| before_pnr node | TensorEdge identifier | current node | old STANDBY ID | new STANDBY ID |
|-----------------|----------------------|--------------|----------------|----------------|
| imce_3_4 | (-13, odata), (38) | imce_0_1 | 18 | 6 |
| imce_3_2 | ((39,36), odata), (40) recv | imce_2_1 | 16 | 16 |
| imce_3_3 | ((39,36), odata), (40) send | imce_1_1 | 17 | 11 |

### Tip 3: Use Read Tool Line Ranges Strategically

Instead of reading entire file multiple times:
1. Use grep to find line numbers
2. Read only ±20 lines around target location
3. Apply Edit with unique context

```bash
# Example: Find line number first
grep -n "specific_pattern" imce.cpp
# Output: 145:...

# Then use Read tool with offset=125, limit=40 to see lines 125-165
```

### Tip 4: Batch Similar Edits by Pattern Type

1. Apply all Pattern 1 edits (shared input sync)
2. Apply all Pattern 2 edits (multi-RECV sync)
3. Apply all Pattern 3 edits (SEND sync)

This reduces context switching and makes verification easier.

### Tip 5: Leverage Comment Patterns

Sync patches often have descriptive comments:
```cpp
// sync one of imce_2_4, and imce_3_4
// STANDBY is not inserted before SEND but before MM_QUNAT, because of overwritten QREGs
```

Update these comments with new node names for future reference.

### Tip 6: Handle Duplicate Patterns Carefully

When Edit tool reports "found 2 matches":
- Add more unique context from surrounding code
- Use distinctive comments or loop structures
- Check for row_group/col_group labels

## Common Pitfalls

1. **Forgetting to update STANDBY IDs**: Always recalculate based on node mapping
2. **Missing some SEND sync locations**: Use grep to count and verify all locations
3. **Incorrect TensorEdge matching**: Distinguish between sender and receiver nodes
4. **Not preserving indentation**: Match exact whitespace from Read tool output
5. **Applying patches to wrong node**: Always verify TensorEdge comments match

## Region-Specific Notes

Different regions may have different patterns:
- **region1**: Typically has conv + batch_norm patterns
- **region2**: May have different operation types
- **region3**: Check for unique patterns

Always analyze each region independently before applying patches.

## Quick Reference: Common Grep Commands

```bash
# Find all imce node blocks
grep -n "else if.*imce_[0-9]_[0-9]" imce.cpp

# Find all sync primitives with context
grep -n -B3 -A3 "SETFLAG\|STANDBY" imce.cpp

# Find specific TensorEdge pattern
grep -n "TensorEdge([^)]*38[^)]*)" imce.cpp

# Count batch_norm blocks
grep -c "endgenerate: batch_norm" imce.cpp

# Find MM_QUANT locations
grep -n "MM_QUANT" imce.cpp

# Find all RECV operations
grep -n "__builtin_IMCE_RECV(" imce.cpp

# Find all SEND operations
grep -n "__builtin_IMCE_SEND(" imce.cpp
```

## Automation Potential

For future work, consider scripting:
1. TensorEdge pattern extraction and mapping
2. STANDBY ID calculation based on node mapping
3. Automated patch generation (with manual verification)

However, manual verification is always recommended due to:
- Complex context dependencies
- Hardware-specific timing requirements
- Potential for subtle bugs in synchronization

## Example Session Workflow

1. **Setup** (5 min)
   ```bash
   cd handcraft/
   grep -n "SETFLAG\|STANDBY" before_pnr/.../imce.cpp > sync_patterns.txt
   ```

2. **Analysis** (10 min)
   - Identify all sync patterns in before_pnr
   - Map TensorEdge patterns between versions
   - Calculate new STANDBY IDs
   - Create mapping table

3. **Application** (15-30 min)
   - Apply Pattern 1 patches
   - Apply Pattern 2 patches
   - Apply Pattern 3 patches (multiple locations)

4. **Verification** (5 min)
   - Count sync primitives
   - Verify STANDBY IDs
   - Run view_diff.py if available

**Total time**: ~30-50 minutes per region (after learning the methodology)

---

*Last updated: 2026-01-29*
*Based on sync patch application for region1 imce.cpp*
