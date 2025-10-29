#!/bin/bash
python test.py -k "resnet"       -s &> resnet.log
python test.py -k "ds_cnn"       -s &> ds_cnn.log
python test.py -k "autoencoder"  -s &> autoencoder.log
python test.py -k "mobilenet"    -s &> mobilenet.log