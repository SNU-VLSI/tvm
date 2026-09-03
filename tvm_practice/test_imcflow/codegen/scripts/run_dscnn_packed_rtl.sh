#!/bin/zsh
# DS-CNN full IDEAL mapping (Path C: concat-push packing, 1-region 16-IMCE,
# standalone bn/mm = 0) RTL gate + bit-exact compare.
set -e
CODE="$(cd "$(dirname "$0")/.." && pwd)"
MAIN=/root/project/tvm/tvm_practice/test_imcflow/codegen
cd $CODE
source ~/.zshrc
eval "$(direnv export zsh)" 2>/dev/null || true
unset IMCFLOW_RESIDUAL_IN_REGION IMCFLOW_RESID_INODE_BUFFER IMCFLOW_RESID_INREGION_OC IMCFLOW_REGION_MERGE
export IMCFLOW_PACK_BN_MINMAX=1
export IMCFLOW_BIG_IMEM=1 IMCFLOW_BUGFIX=off IMCFLOW_DEBUG=0
export IMCFLOW_EVAL_SUFFIX=.dscnn2region
export IMCFLOW_RUNNER=rtl IMCFLOW_HOST_OS=baremetal IMCFLOW_HOST_ISA=x86
export IMCFLOW_DIR=/root/project/imcflow
export IMCFLOW_RTL_RUNNER_DIR=/root/project/imcflow/pmap/ISA_sim/gem5/tests/imcflow/rtl_runner_bigimem
export SNPSLMD_LICENSE_FILE=1727@147.46.168.128
export CKPT=kws_dscnn_base
export IMCFLOW_TVM_CODEGEN_DIR="$PWD"
PY=/root/project/tvm/tvm_practice/tvm_env/bin/python
NAME=ds_cnn_full_pretrained_evl.baremetal.bugfixoff.dscnn2region
mkdir -p $MAIN/eval_dir
rm -rf $MAIN/eval_dir/$NAME 2>/dev/null || true
ln -s $CODE/eval_dir/$NAME $MAIN/eval_dir/$NAME 2>/dev/null || true
WATCHLOG=eval_dir/wedge_watch_dscnn_packed_$(date +%H%M%S).log
echo "[rtl] START $(date) ds_cnn_full PACKED (2-region ideal mapping) (--stop-at compare) + watcher(log: $PWD/$WATCHLOG)"
# Wait for the sim to actually start (codegen+build can take 20+ min; a fixed
# sleep made the watcher declare "run ended" before vcs_sim.log even existed).
( while [ ! -f "eval_dir/$NAME/logs/rtl_runner/vcs_sim.log" ]; do sleep 30; done; \
  $PY tools/rtl_wedge_watch.py "eval_dir/$NAME" --kill --max-polls 12000 > "$WATCHLOG" 2>&1 ) &
WATCH_PID=$!
$PY -u main.py --model ds_cnn_full_pretrained --stop-at compare 2>&1
kill $WATCH_PID 2>/dev/null || true
echo "[rtl] DONE $(date)"
