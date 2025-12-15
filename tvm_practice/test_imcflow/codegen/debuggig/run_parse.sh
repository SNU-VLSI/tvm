#!/bin/bash

grep -ir "imce.3.0" super_big_conv_rev4_evl/logs/now.debug.log > imce3_0.log
grep -ir "imce.3.1" super_big_conv_rev4_evl/logs/now.debug.log > imce3_1.log
grep -ir "imce.3.2" super_big_conv_rev4_evl/logs/now.debug.log > imce3_2.log
grep -ir "imce.3.3" super_big_conv_rev4_evl/logs/now.debug.log > imce3_3.log
grep -ir "imce.3.4" super_big_conv_rev4_evl/logs/now.debug.log > imce3_4.log