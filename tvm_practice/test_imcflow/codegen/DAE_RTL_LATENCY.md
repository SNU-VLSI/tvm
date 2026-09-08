# DAE RTL latency worktree

Run from `/root/project/tvm_latency/tvm_practice/test_imcflow/codegen`.
The worktree uses its own TVM Python sources and native libraries, with the
existing Python environment linked at `../../tvm_env`.

```sh
cd /root/project/tvm_latency/tvm_practice/test_imcflow/codegen
direnv exec . env \
  IMCFLOW_RUNNER=rtl \
  IMCFLOW_DIR=/root/project/imcflow \
  IMCFLOW_BUGFIX=on \
  CKPT_PATH=/root/project/CIM/runs/dae_integration/initial/checkpoint.pth.tar \
  python -u main.py --model dae_toycar_full_pretrained --driver-v2 --stop-at simulate
```

Set `SNPSLMD_LICENSE_FILE` to the license server in your shell if needed.
Replace `CKPT_PATH` with your deploy checkpoint file. `CKPT=initial` instead
selects an alias from the CIM checkpoint registry; `CKPT` does not accept a
file path. With no dataset option, the pipeline uses random model input.
Use `--driver-v2`: the default driver currently fails on this full DAE with
`AssertionError: Only one output edge is expected` during code generation.

Use `--stop-at compile` to build without simulation. Output is under
`eval_dir/dae_toycar_full_pretrained_evl.baremetal/`; RTL logs are under
`logs/rtl_runner/` within that directory.

The shared gem5 checkout's `tests/imcflow/rtl_runner/run.sh` and
`configs/imcflow/run_imcflow_rtl.py` accept `IMCFLOW_TVM_CODEGEN_DIR` to select
this worktree. Those compatibility edits are local changes in the gem5 repo.
The RTL simulator build is shared; switching BUGFIX mode rebuilds it.
