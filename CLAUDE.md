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