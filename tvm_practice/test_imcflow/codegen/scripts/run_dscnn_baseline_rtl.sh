#!/bin/zsh
# DS-CNN full lever-OFF baseline — BUGFIX-off RTL co-sim (busy-cycle reference).
# Measured (2026-08): 2 regions (16+8=24 IMCE), accelerator busy 32,370 cyc
# = 323.7 us @100MHz (region1 17,248 + region2 15,122).
# Comparison target for the DS-CNN ideal-mapping (D1) run once it lands.
set -e
CODE="$(cd "$(dirname "$0")/.." && pwd)"
MAIN=/root/project/tvm/tvm_practice/test_imcflow/codegen
cd $CODE
source ~/.zshrc
eval "$(direnv export zsh)" 2>/dev/null || true
unset IMCFLOW_PACK_BN_MINMAX IMCFLOW_RESIDUAL_IN_REGION IMCFLOW_RESID_INODE_BUFFER \
      IMCFLOW_RESID_INREGION_OC IMCFLOW_REGION_MERGE
export IMCFLOW_BIG_IMEM=1
export IMCFLOW_EVAL_SUFFIX=.dscnnoff
export IMCFLOW_DEBUG=0 IMCFLOW_BUGFIX=off
export IMCFLOW_RUNNER=rtl IMCFLOW_HOST_OS=baremetal IMCFLOW_HOST_ISA=x86
export IMCFLOW_DIR=/root/project/imcflow
export IMCFLOW_RTL_RUNNER_DIR=/root/project/imcflow/pmap/ISA_sim/gem5/tests/imcflow/rtl_runner_bigimem
export SNPSLMD_LICENSE_FILE=1727@147.46.168.128
export CKPT=kws_dscnn_base
export IMCFLOW_TVM_CODEGEN_DIR="$PWD"
PY=/root/project/tvm/tvm_practice/tvm_env/bin/python
NAME=ds_cnn_full_pretrained_evl.baremetal.bugfixoff.dscnnoff
mkdir -p $MAIN/eval_dir
if [ "$CODE" != "$MAIN" ]; then
  rm -rf $MAIN/eval_dir/$NAME 2>/dev/null || true
  ln -s $CODE/eval_dir/$NAME $MAIN/eval_dir/$NAME 2>/dev/null || true
fi
WATCHLOG=eval_dir/wedge_watch_dscnn_baseline_$(date +%m%d_%H%M%S).log
echo "[rtl] START $(date) ds_cnn_full lever-OFF baseline + watcher(log: $PWD/$WATCHLOG)"
( sleep 60; $PY tools/rtl_wedge_watch.py "eval_dir/$NAME" --kill --max-polls 12000 > "$WATCHLOG" 2>&1 ) &
WATCH_PID=$!
$PY -u main.py --model ds_cnn_full_pretrained --stop-at compare 2>&1
kill $WATCH_PID 2>/dev/null || true
echo "[rtl] DONE $(date)"
