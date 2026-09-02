#!/bin/zsh
# ResNet8 lever-OFF baseline — BUGFIX-off RTL co-sim (--stop-at compare).
# Counterpart of run_resnet8_resid_on_rtl.sh for the residual-in-region A/B:
# 4 regions, accelerator busy 132,682 cyc = 1,326.8 us @100MHz.
# Same bigimem simv as the ON run so only the codegen levers differ.
set -e
CODE="$(cd "$(dirname "$0")/.." && pwd)"
MAIN=/root/project/tvm/tvm_practice/test_imcflow/codegen
cd $CODE
source ~/.zshrc
eval "$(direnv export zsh)" 2>/dev/null || true
unset IMCFLOW_PACK_BN_MINMAX IMCFLOW_RESIDUAL_IN_REGION IMCFLOW_RESID_INODE_BUFFER \
      IMCFLOW_RESID_INREGION_OC IMCFLOW_REGION_MERGE
export IMCFLOW_BIG_IMEM=1
export IMCFLOW_EVAL_SUFFIX=.offbase
export IMCFLOW_DEBUG=0 IMCFLOW_BUGFIX=off
export IMCFLOW_RUNNER=rtl IMCFLOW_HOST_OS=baremetal IMCFLOW_HOST_ISA=x86
export IMCFLOW_DIR=/root/project/imcflow
export IMCFLOW_RTL_RUNNER_DIR=/root/project/imcflow/pmap/ISA_sim/gem5/tests/imcflow/rtl_runner_bigimem
export SNPSLMD_LICENSE_FILE=1727@147.46.168.128
export CKPT=n32_signed_sample
export IMCFLOW_TVM_CODEGEN_DIR="$PWD"
PY=/root/project/tvm/tvm_practice/tvm_env/bin/python
NAME=resnet8_subset31_pretrained_orig_evl.baremetal.bugfixoff.offbase
mkdir -p $MAIN/eval_dir
if [ "$CODE" != "$MAIN" ]; then
  rm -rf $MAIN/eval_dir/$NAME 2>/dev/null || true
  ln -s $CODE/eval_dir/$NAME $MAIN/eval_dir/$NAME 2>/dev/null || true
fi
WATCHLOG=eval_dir/wedge_watch_resnet8_resid_off_$(date +%m%d_%H%M%S).log
echo "[rtl] START $(date) resnet8 lever-OFF baseline + watcher(log: $PWD/$WATCHLOG)"
( sleep 60; $PY tools/rtl_wedge_watch.py "eval_dir/$NAME" --kill --max-polls 12000 > "$WATCHLOG" 2>&1 ) &
WATCH_PID=$!
$PY -u main.py --model resnet8_subset31_pretrained_orig --stop-at compare 2>&1
kill $WATCH_PID 2>/dev/null || true
echo "[rtl] DONE $(date)"
