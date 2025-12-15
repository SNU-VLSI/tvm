#!/bin/bash

python_dbg="python -m debugpy --listen $TVM_DEBUG_PORT --wait-for-client"

run_test() {
  local model=$1
  local pattern=${2:-random}  # 기본값 random
  local dbg=${3:-0}      # 기본값 0
  echo "Running test: $model with pattern $pattern"

  if [ "$dbg" -eq 1 ]; then
    cmd="$python_dbg main.py -p $pattern -m $model 2>&1 | tee ${model}_${pattern}.log"
  else
    cmd="python main.py -p $pattern -m $model 2>&1 | tee ${model}_${pattern}.log"
  fi
  echo $cmd
  eval $cmd
}

# run_test "one_relu" "random"
# run_test "one_conv" "random"
# run_test "one_mmquant" "random"
# run_test "one_conv_quant" "random"
# run_test "one_fused_bn" "random"
# run_test "one_conv_bn" "random"
# run_test "big_conv" "random"
# run_test "big_conv_rparam" "random"

# run_test "super_big_conv_rev1" "ones"
# run_test "super_big_conv_rev1" "random"

# run_test "super_big_conv_rparam_rev1" "ones"
# run_test "super_big_conv_rparam_rev1" "random"

# run_test "super_big_conv_rev2" "ones"
# run_test "super_big_conv_rev2" "linear"
# run_test "super_big_conv_rev2" "random"

# run_test "super_big_conv_rev3" "ones"
# run_test "super_big_conv_rev3" "linear"
# run_test "super_big_conv_rev3" "random"

# run_test "super_big_conv_rev4" "ones"
# run_test "super_big_conv_rev4" "linear"
# run_test "super_big_conv_rev4" "random"

# run_test "super_big_conv_rev5" "ones"
# run_test "super_big_conv_rev5" "linear"
# run_test "super_big_conv_rev5" "random"

# run_test "super_big_conv_bn_quant_rev1" "ones"
# run_test "super_big_conv_bn_quant_rev1" "linear"
# run_test "super_big_conv_bn_quant_rev1" "random"

# run_test "super_big_conv_bn_quant_rev2" "ones"
# run_test "super_big_conv_bn_quant_rev2" "linear"
# run_test "super_big_conv_bn_quant_rev2" "random"

# run_test "super_big_conv_bn_quant_rev6" "ones"
# run_test "super_big_conv_bn_quant_rev6" "linear"
# run_test "super_big_conv_bn_quant_rev6" "random"

run_test "resnet8_small_pretrained" "random" 1