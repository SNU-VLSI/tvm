# TVM Compiler - IMCFlow Extensions

This is the TVM compiler extended to map high-level neural network operations onto imcflow in-memory computing hardware.

## Agent Usage

**ALWAYS use the `imcflow-compiler-expert` agent when working in this directory. Refer to /root/.claude/agents/imcflow-compiler-expert.md**

This agent has deep expertise in:
- Mapping tvm.relay operations to imcflow hardware primitives
- TVM codegen and schedule optimization for imcflow
- Debugging relay graph transformations
- BYOC (Bring Your Own Codegen) integration with imcflow backend
- Cross-compiling for gem5 simulation targets

## Key Areas

- **Relay IR**: High-level operation definitions and graph transformations
- **Codegen**: Code generation targeting imcflow ISA (inode/imce instructions)
- **BYOC Integration**: Custom backend integration for offloading to imcflow
- **Schedule/Compute**: Tensor operation scheduling and optimization
- **Testing**: ResNet8 and other model layers mapped to imcflow

## Build System

- Uses `direnv` for environment setup
- Build directory: `tvm/build/`
- Logs generated in: `eval_dir/logs/` when invoked from codegen

## Debugging vs. speed (driver-v2 `compile_for_imcflow_v2`)

Two debug-only outputs dominate compile time on deep models (e.g. VWW MobileNet's
13 blocks) and should be turned OFF for fast iteration / structure checks:

- **`save_intermediate` (default True)**: dumps per-stage relay (`0N_after_*.txt/.pdf`)
  AND the NoC visualizations (matplotlib PNG/PDF, ~30% of compile time). Keep it ON
  only when you actually need to inspect a specific transform stage or the NoC
  routing/placement (debugging layout, partitioning, or PnR). Pass
  `save_intermediate=False` otherwise.
- **`skip_codegen` (default False)**: set `skip_codegen=True` to stop after
  transform+codegen and skip the slow CPU `relay.build` (the hw-accurate qconv
  compute). Use for verifying that transform/layout legalization succeeds (e.g.
  bringing up a new model) without paying for the full graph-executor build.

For numerical/bit-exact checks you DO need the build (and `--ref-models`); for "does
it compile / does layout legalize" checks, `skip_codegen=True, save_intermediate=False`
is far faster. NoC visualization is only meaningful when debugging hardware
placement/routing — leave it off otherwise.

## When to Use the Agent

Use `imcflow-compiler-expert` for ANY task including:
- Understanding TVM codegen flow
- Debugging compilation failures
- Adding new relay operation support
- Optimizing hardware mappings
- Investigating generated LLVM IR or assembly
- Integration issues with LLVM backend or gem5

## compilation and simulation

- invoke python at `/root/project/tvm/tvm_practice/test_imcflow/codegen`
- source ~/.zshrc
- use `.envrc` with `direnv`

## IMCFLOW_BUGFIX master knob (BUGFIX-on vs BUGFIX-off RTL) — READ FIRST

The RTL has two compile configurations: **BUGFIX-on** (the original,
chip/tapeout HW model) and **BUGFIX-off** (BUGFIX_STEP/DWCONV/OVERFLOW/ROUTER
macros omitted). BUGFIX-off deadlocks unless codegen emits extra NoC sync
(rendezvous/barrier); BUGFIX-on works with the original a8af sync. ONE env var
switches BOTH codegen and the RTL runner consistently:

- **`IMCFLOW_BUGFIX` unset or `=off` (DEFAULT)** → codegen = 934+P0-P3+P4
  sync; the shared RTL runner compiles without the BUGFIX_* defines;
  overflow-SW fix default ON; eval_dir gets a `.bugfixoff` suffix so its
  artifacts (codegen/fsdb/output) do not overwrite an explicit BUGFIX-on run.
- **`IMCFLOW_BUGFIX=on`** → codegen = a8af (no new sync, byte-identical to
  pristine); the same RTL runner compiles with the BUGFIX_* defines;
  `IMCFLOW_BUGFIX_OVERFLOW_SW` defaults OFF.

Helper: `tvm.contrib.imcflow.bugfix_off_mode()` (default True). Gated at
all sync call sites in send_recv_sync.py / imce_codeblock.py / inode_codeblock.py
/ codegen.py. The RTL runner uses `build/build_manifest.json` to rebuild when
the mode, defines/options, source/include inputs, compile paths, or VCS identity
changes. `IMCFLOW_RTL_RUNNER_DIR` still overrides the shared runner directory.
Background & design: `DWCONV_SYNC_GRANULARITY_DESIGN.md`, `P4_HANDOFF.md`.

## Running the BUGFIX-off RTL co-simulation (deadlock-check path)

Not the chip path — this drives the VCS RTL runner (`--stop-at simulate`).
Model names: `resnet8_subset31_pretrained_orig`, `ds_cnn_subset08_pretrained`
(fast), `ds_cnn_full_pretrained`. Required env (from the codegen dir):

```bash
export IMCFLOW_BUGFIX=off            # optional: this is the default
export IMCFLOW_RUNNER=rtl IMCFLOW_HOST_OS=baremetal IMCFLOW_HOST_ISA=x86
export IMCFLOW_DIR=/root/project/imcflow
export SNPSLMD_LICENSE_FILE=1727@147.46.168.128   # VCS/XProp license
export CKPT=n32_signed_sample        # resnet8 (b2_half.json); ds_cnn -> kws_dscnn_base
# PYTHONPATH/TVM_HOME must point at THIS checkout (worktree-aware)
python -u main.py --model <model> --stop-at simulate
```

- **Pass = `POLLING ERROR 0` + `SIMULATION COMPLETED SUCCESSFULLY`.** A deadlock
  shows as poll count climbing to 20000 then `POLLING ERROR` (host polls the
  accelerator until it returns IDLE; 20000 = hang). Healthy runs finish in
  hundreds–thousands of polls.
- fsim per-node logs: `eval_dir/<model>_evl.baremetal[.bugfixoff]/logs/rtl_runner/fsim_logs/`.
- GOTCHA: the shared BUGFIX-off runner `run.sh` hardcodes a `TVM_BUILD_DIR`; from
  a different worktree set `IMCFLOW_TVM_CODEGEN_DIR=<this codegen dir>` (a
  backward-compatible override was added). `direnv allow` the worktree + gem5
  `.envrc` if the C build step reports `direnv blocked`.
- ★GOTCHA (worktree + inputs): gem5's `run_imcflow_rtl.py` resolves the model
  inputs against the **main** codegen tree (`/root/project/tvm/tvm_practice/test_imcflow/codegen/eval_dir/...`),
  NOT the worktree. If that main-tree `eval_dir/<name>/test_inputs/` lacks
  `model_input.{bin,meta.txt}`, gem5 logs `No inputs loaded!` and exits WITHOUT
  running — yet the host binary still prints `POLLING ERROR 0` /
  `SIMULATION COMPLETED SUCCESSFULLY` (misleading: it only means clean exit, not
  that the accelerator computed). Verify real work ran: `imcflow_state_o` should
  rise to 1 (fsdb has busy pulses), and vcs_sim.log should have
  `Processing READ/WRITE` + `resuming normal operation` markers. A ~800 KB fsdb
  (vs tens of MB) or a truncated vcs_sim.log ending at "gem5 connected..." means
  the workload never ran. Fix: copy `model_input.*` into the MAIN-tree
  `eval_dir/<name>/test_inputs/` before the run. (This surfaced when the
  `.bugfixoff` eval_dir suffix created a new name with no matching main-tree
  inputs; `rtl_region_cycles.py` now prints this specific diagnosis instead of a
  bare "no region markers".)

## Measuring accelerator busy cycles (imcflow_state_o)

`tools/rtl_region_cycles.py <eval_dir> [--method fsdb|poll] [--json]` reports
per-region cycles where the accelerator was actually computing (`imcflow_state_o=1`
in the .fsdb; host poll/transfer overhead excluded), at 100 MHz. `--method fsdb`
(default, accurate) needs Verdi on PATH (`/tool/Program/synopsys/verdi/V-2023.12-SP2-4/bin`);
`--method poll` estimates from vcs_sim.log poll spans (within ~1% of fsdb).
Use it to compare BUGFIX-on vs BUGFIX-off execute cycles for the same model — run
each with its knob into its own eval_dir, then point the tool at each. See memory
`[[rtl-region-busy-cycles]]`.
