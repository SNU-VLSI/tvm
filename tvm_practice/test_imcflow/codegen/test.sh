#!/bin/bash
# python test.py -k "test_resnet8_from_pretrained"       -s 2>&1  | tee resnet.log
# python test.py -k "ds_cnn"       -s 2>&1 | tee ds_cnn.log
# python test.py -k "autoencoder"  -s 2>&1 | tee autoencoder.log
# python test.py -k "mobilenet"    -s 2>&1 | tee mobilenet.log

python test.py -k "test_one_relu_evl" -s 2>&1 | tee one_relu.log
# python test.py -k "test_one_conv_evl" -s 2>&1 | tee test_one_conv_evl.log