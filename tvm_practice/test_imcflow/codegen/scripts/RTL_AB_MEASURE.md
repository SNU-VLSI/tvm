# RTL A/B measurement scripts (BUGFIX-off co-sim)

Reproducible run scripts for the residual-in-region / packing measurements on
branch `feat/imcflow-residual-in-region`. All run the full pipeline
(`--stop-at compare`, bit-exact check vs CPU golden) with a wedge watcher
(`tools/rtl_wedge_watch.py --kill --max-polls 12000`) attached, and are
worktree-safe (resolve their own codegen dir; symlink the eval_dir into the
main tree for the gem5 input-resolution gotcha).

| script | levers | measured accelerator busy @100MHz |
|---|---|---|
| `run_resnet8_resid_on_rtl.sh` | PACK_BN_MINMAX + RESIDUAL_IN_REGION (+INODE_BUFFER, OC=64) + REGION_MERGE=2 | 2 regions, **95,744 cyc = 957.4 us** |
| `run_resnet8_resid_off_rtl.sh` | all off | 4 regions, **132,682 cyc = 1,326.8 us** |
| `run_dscnn_baseline_rtl.sh` | all off | 2 regions (24 IMCE), **32,370 cyc = 323.7 us** |

Both resnet8 variants (and the DS-CNN baseline) use the same enlarged-memory
simv (`IMCFLOW_BIG_IMEM=1`, runner dir `rtl_runner_bigimem`) so only the
codegen levers differ.

Measure busy cycles (`imcflow_state_o=1` regions in the .fsdb) after a PASS:

```bash
export PATH=$PATH:/tool/Program/synopsys/verdi/V-2023.12-SP2-4/bin
python tools/rtl_region_cycles.py eval_dir/<name> --method fsdb   # or --method poll (~1%)
```

Pass criterion: `[POLLING] Operation complete!` for every launch +
`SIMULATION COMPLETED SUCCESSFULLY` + the compare stage reporting bit-exact.
A poll count climbing toward 20000 is a wedge (the watcher kills it early).
