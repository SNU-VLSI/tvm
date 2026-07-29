# B1 Core 0,1 Noise Workflow Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the B1 `(0,1)` noise artifact workflow after the already-running BulkBitLinePlanner finishes, then update the Visual Wake Words chip-noise loop configuration with the produced artifact paths.

**Architecture:** Treat the active BulkBitLinePlanner output as the input boundary and do not move or rerun it. Back up only stale downstream output directories, execute workflow stages 3–8 exactly as documented, validate every stage boundary, and finally change only path fields in the loop JSON.

**Tech Stack:** Bash, direnv, Python 3, imcflow measurement planners, TVM codegen, CIM noise CSV utilities, JSON

## Global Constraints

- Do not interrupt or duplicate the active BulkBitLinePlanner process.
- Do not use `rm`, `rm -r`, or `rm -rf`.
- Preserve unrelated working-tree changes.
- Source workflow: `/root/project/imcflow/xilinx/measurement/B1_CORE_0_1_NOISE_WORKFLOW.md`.
- Final configuration: `/root/project/CIM/scripts/loop/visual_wake_words_chip_noise_loop.jjy.json`.

---

### Task 1: Wait for and validate BulkBitLinePlanner

**Files:**
- Read: `/root/project/imcflow/xilinx/measurement/workspace/BulkBitLinePlanner_for_csv_0_1/results/result.pkl`

**Interfaces:**
- Consumes: active `run_planner_all_nodes.sh` / `run_planner.py` process for node `(0,1)`
- Produces: non-empty `result.pkl` for the converter

- [ ] **Step 1: Poll the planner process without interrupting it**

Run:

```bash
ps -eo pid,ppid,etime,state,args | rg -i 'run_planner_all_nodes|run_planner.py' | rg 'bitline_for_csv|BulkBitLinePlanner'
```

Expected: the active process remains visible until it exits normally.

- [ ] **Step 2: Validate the completed result**

Run:

```bash
test -s /root/project/imcflow/xilinx/measurement/workspace/BulkBitLinePlanner_for_csv_0_1/results/result.pkl
```

Expected: exit status 0.

### Task 2: Back up downstream outputs and build disabled-column artifacts

**Files:**
- Read: `/root/project/imcflow/xilinx/measurement/workspace/BulkBitLinePlanner_for_csv_0_1/results/result.pkl`
- Create: `/root/project/CIM/noise/noise_df/B1_cache_0_1/disabled.json`
- Create: `/root/project/CIM/noise/noise_df/B1_out_0_1/N32/disabled.json`
- Create: `/root/project/CIM/noise/noise_df/B1_out_0_1/N32/B1_noise_matrix_per_ch_concat.csv`
- Create: `/root/project/CIM/noise/noise_df/B1_out_0_1/N32/concat_per_core.json`

**Interfaces:**
- Consumes: completed BulkBitLinePlanner result
- Produces: disabled-column configuration and concatenated bitline-noise artifacts for planner and TVM stages

- [ ] **Step 1: Move only existing downstream output directories to timestamped backups**

Use `mv` with one shared `RUN_TS` for these exact targets when present:

```text
/root/project/imcflow/xilinx/measurement/workspace/TargetSignedWeightPsumNoisePlanner_b1_random_sample_nd32_0_1
/root/project/CIM/noise/noise_df/B1_cache_0_1
/root/project/CIM/noise/noise_df/B1_out_0_1
/root/project/CIM/noise/noise_df/B1_vww_mobilenet_signed_weight_ref_cache_random_sample_nd32_0_1_N32
/root/project/CIM/noise/noise_df/B1_vww_mobilenet_signed_weight_ref_out_random_sample_nd32_0_1
```

Expected: each existing target is preserved as `<target>.backup.<RUN_TS>` and each original path is free for fresh output.

- [ ] **Step 2: Convert the planner result**

Run from `/root/project/CIM`:

```bash
python3 noise/noise_csv_util.py convert \
  /root/project/imcflow/xilinx/measurement/workspace/BulkBitLinePlanner_for_csv_0_1/results/result.pkl \
  --output-dir noise/noise_df/B1_cache_0_1 \
  --num-disable 32 \
  --cores 0,1
test -s noise/noise_df/B1_cache_0_1/disabled.json
```

Expected: converter succeeds and `disabled.json` is non-empty.

- [ ] **Step 3: Aggregate and concatenate**

First add the per-core count-cache layout required by the aggregation scripts
while retaining the converter-produced `disabled.json` in the same directory:

```bash
python3 noise/noise_csv_util.py cache \
  /root/project/imcflow/xilinx/measurement/workspace/BulkBitLinePlanner_for_csv_0_1/results/result.pkl \
  --cache-dir noise/noise_df/B1_cache_0_1 \
  --round-bins
test -s noise/noise_df/B1_cache_0_1/manifest.json
```

Run from `/root/project/CIM`:

```bash
./noise/noise_df/aggregate.sh \
  --board B1 \
  --cache-dir /root/project/CIM/noise/noise_df/B1_cache_0_1 \
  --col-order /root/project/CIM/noise/noise_df/B1_cache_0_1/disabled.json \
  --output-dir /root/project/CIM/noise/noise_df/B1_out_0_1 \
  32
./noise/noise_df/concatenate.sh \
  --board B1 \
  --cache-dir /root/project/CIM/noise/noise_df/B1_cache_0_1 \
  --col-order /root/project/CIM/noise/noise_df/B1_cache_0_1/disabled.json \
  --output-dir /root/project/CIM/noise/noise_df/B1_out_0_1 \
  32
```

Expected: the three Task 2 output files under `B1_out_0_1/N32` are non-empty.

### Task 3: Measure signed-weight psum noise and compile the TVM column map

**Files:**
- Create: `/root/project/imcflow/xilinx/measurement/workspace/TargetSignedWeightPsumNoisePlanner_b1_random_sample_nd32_0_1/results/result.pkl`
- Create: `/root/project/tvm/tvm_practice/test_imcflow/codegen/eval_dir/mobilenet_v1_vww_full_pretrained_evl.linux/psum_imcu_column_map.npz`

**Interfaces:**
- Consumes: `/root/project/CIM/noise/noise_df/B1_out_0_1/N32/disabled.json`
- Produces: signed-weight measurement result and matching TVM psum column map

- [ ] **Step 1: Run TargetSignedWeightPsumNoisePlanner**

Run the exact environment and command from workflow section 5 with `BOARD=B1`, `TAG=b1_random_sample_nd32_0_1`, `NODE=(0,1)`, five samples per target/chunk, and one repeat.

Expected: the planner exits successfully and its `result.pkl` is non-empty.

- [ ] **Step 2: Compile the Visual Wake Words model**

Run the exact activation, environment, checkpoint, and `main.py --stop-at compile` command from workflow section 6.

Expected: `eval_dir/mobilenet_v1_vww_full_pretrained_evl.linux/psum_imcu_column_map.npz` is non-empty.

### Task 4: Assemble and validate signed-weight reference artifacts

**Files:**
- Create: `/root/project/CIM/noise/noise_df/B1_vww_mobilenet_signed_weight_ref_out_random_sample_nd32_0_1/N32/B1_noise_matrix_per_ch_concat_signed_weight_ref.csv`
- Create: `/root/project/CIM/noise/noise_df/B1_vww_mobilenet_signed_weight_ref_out_random_sample_nd32_0_1/N32/concat_per_core.json`
- Create: `/root/project/CIM/noise/noise_df/B1_vww_mobilenet_signed_weight_ref_out_random_sample_nd32_0_1/N32/psum_imcu_column_map.npz`

**Interfaces:**
- Consumes: signed-weight planner `result.pkl`, `disabled.json`, and TVM column map
- Produces: validated, deployable B1 `(0,1)` signed-weight noise directory

- [ ] **Step 1: Generate the signed-weight reference CSV**

Run `/root/project/CIM/noise/run_signed_weight_ref_concat_nd32.sh` with the exact environment values from workflow section 7.

Expected: the signed-weight CSV and `concat_per_core.json` are created under the final `N32` directory.

- [ ] **Step 2: Copy the TVM column map**

Run the exact `cp` command from workflow section 8.

Expected: the final `psum_imcu_column_map.npz` is non-empty.

- [ ] **Step 3: Validate all three artifacts**

Run:

```bash
/root/project/CIM/scripts/common/check_signed_weight_ref_artifacts.sh \
  /root/project/CIM/noise/noise_df/B1_vww_mobilenet_signed_weight_ref_out_random_sample_nd32_0_1/N32/B1_noise_matrix_per_ch_concat_signed_weight_ref.csv \
  /root/project/CIM/noise/noise_df/B1_vww_mobilenet_signed_weight_ref_out_random_sample_nd32_0_1/N32/concat_per_core.json \
  /root/project/CIM/noise/noise_df/B1_vww_mobilenet_signed_weight_ref_out_random_sample_nd32_0_1/N32/psum_imcu_column_map.npz \
  0_1
```

Expected: validation exits with status 0.

### Task 5: Update and verify the chip-noise loop configuration

**Files:**
- Modify: `/root/project/CIM/scripts/loop/visual_wake_words_chip_noise_loop.jjy.json`

**Interfaces:**
- Consumes: freshly validated disabled-column and signed-weight artifact paths
- Produces: valid loop JSON pointing at this workflow's B1 `(0,1)` outputs

- [ ] **Step 1: Identify path-valued fields that refer to noise workflow inputs**

Inspect the JSON and retain all non-path settings unchanged.

Expected: `training.initial_noise_dir` and `tvm.column_disable_config` are updated; any additional path field is changed only if its current value is superseded by a Task 4 artifact.

- [ ] **Step 2: Apply the exact generated paths**

Set:

```json
{
  "training.initial_noise_dir": "/root/project/CIM/noise/noise_df/B1_vww_mobilenet_signed_weight_ref_out_random_sample_nd32_0_1/N32",
  "tvm.column_disable_config": "/root/project/CIM/noise/noise_df/B1_out_0_1/N32/disabled.json"
}
```

Expected: only the intended JSON string values change.

- [ ] **Step 3: Validate JSON and referenced paths**

Run:

```bash
python3 -m json.tool /root/project/CIM/scripts/loop/visual_wake_words_chip_noise_loop.jjy.json
test -d /root/project/CIM/noise/noise_df/B1_vww_mobilenet_signed_weight_ref_out_random_sample_nd32_0_1/N32
test -s /root/project/CIM/noise/noise_df/B1_out_0_1/N32/disabled.json
```

Expected: JSON parsing and both path checks exit with status 0.
