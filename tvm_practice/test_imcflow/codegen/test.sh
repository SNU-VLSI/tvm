#!/bin/bash

python3 main.py -p random -m one_relu        2>&1 |  tee one_relu.log
python3 main.py -p random -m one_conv        2>&1 |  tee one_conv.log
python3 main.py -p random -m one_mmquant     2>&1 |  tee one_mmquant.log
python3 main.py -p random -m one_conv_quant  2>&1 |  tee one_conv_quant.log
python3 main.py -p random -m one_fused_bn    2>&1 |  tee one_fused_bn.log
python3 main.py -p random -m one_conv_bn     2>&1 |  tee one_conv_bn.log
python3 main.py -p random -m big_conv        2>&1 |  tee big_conv.log
python3 main.py -p random -m big_conv_rparam 2>&1 |  tee big_conv_rparam.log
python3 main.py -p ones -m super_big_conv  2>&1 |  tee super_big_conv.log
