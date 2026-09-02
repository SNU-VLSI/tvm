#!/bin/zsh
# ResNet8 residual-in-region ON (combined levers + balanced 2-region merge) —
# BUGFIX-off RTL co-sim, full pipeline (--stop-at compare -> bit-exact check).
#
# Measured result (2026-08, feat/imcflow-residual-in-region): 2 regions,
# accelerator busy 95,744 cyc = 957.4 us @100MHz (region1 77,587 + region2
# 18,157) vs lever-OFF 4-region baseline 132,682 cyc = 1,326.8 us (-28%).
# Baseline counterpart: run_resnet8_resid_off_rtl.sh. Measure busy cycles with:
#   PATH=$PATH:/tool/Program/synopsys/verdi/V-2023.12-SP2-4/bin \
#   python tools/rtl_region_cycles.py eval_dir/<name> --method fsdb
#
# Historical eval suffix ".cutsel2r" = balanced merge-cut selection, 2-region.
set -e
CODE="$(cd "$(dirname "$0")/.." && pwd)"   # this codegen dir (worktree-safe)
MAIN=/root/project/tvm/tvm_practice/test_imcflow/codegen  # gem5 resolves inputs here
cd $CODE
source ~/.zshrc
eval "$(direnv export zsh)" 2>/dev/null || true

# residual-in-region combined levers
export IMCFLOW_PACK_BN_MINMAX=1
export IMCFLOW_RESIDUAL_IN_REGION=1
export IMCFLOW_RESID_INODE_BUFFER=1
export IMCFLOW_RESID_INREGION_OC=64
export IMCFLOW_REGION_MERGE=2            # balanced 2-region cut
export IMCFLOW_BIG_IMEM=1                # sim-only enlarged imem/dmem (needs bigimem simv)
export IMCFLOW_EVAL_SUFFIX=.cutsel2r
export IMCFLOW_DEBUG=0 IMCFLOW_BUGFIX=off
export IMCFLOW_RUNNER=rtl IMCFLOW_HOST_OS=baremetal IMCFLOW_HOST_ISA=x86
export IMCFLOW_DIR=/root/project/imcflow
export IMCFLOW_RTL_RUNNER_DIR=/root/project/imcflow/pmap/ISA_sim/gem5/tests/imcflow/rtl_runner_bigimem
export SNPSLMD_LICENSE_FILE=1727@147.46.168.128
export CKPT=n32_signed_sample
export IMCFLOW_TVM_CODEGEN_DIR="$PWD"    # worktree gotcha: shared run.sh hardcodes main tree
PY=/root/project/tvm/tvm_practice/tvm_env/bin/python
NAME=resnet8_subset31_pretrained_orig_evl.baremetal.bugfixoff.cutsel2r
# gem5 gotcha: model inputs are resolved against the MAIN codegen tree.
mkdir -p $MAIN/eval_dir
if [ "$CODE" != "$MAIN" ]; then
  rm -rf $MAIN/eval_dir/$NAME 2>/dev/null || true
  ln -s $CODE/eval_dir/$NAME $MAIN/eval_dir/$NAME 2>/dev/null || true
fi
WATCHLOG=eval_dir/wedge_watch_resnet8_resid_on_$(date +%m%d_%H%M%S).log
# factor-1 region1 is a LONG single-tile compute (~7500 polls) -> headroom 12000.
echo "[rtl] START $(date) resnet8 resid-ON (MERGE=2) + watcher(log: $PWD/$WATCHLOG)"
( sleep 60; $PY tools/rtl_wedge_watch.py "eval_dir/$NAME" --kill --max-polls 12000 > "$WATCHLOG" 2>&1 ) &
WATCH_PID=$!
$PY -u main.py --model resnet8_subset31_pretrained_orig --stop-at compare 2>&1
kill $WATCH_PID 2>/dev/null || true
echo "[rtl] DONE $(date)"
