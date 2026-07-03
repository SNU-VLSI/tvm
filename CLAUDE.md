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