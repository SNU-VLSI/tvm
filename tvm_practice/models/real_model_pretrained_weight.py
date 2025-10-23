import pytest
import itertools
import numpy as np
import sys
import subprocess
import math
import collections
import os

import torch
from tvm.relay.backend import te_compiler
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.op.contrib import imcflow
import tvm.testing
from tvm.contrib import utils, graph_executor
from tvm import runtime as tvm_runtime

from tvm.relay.qnn.op.qnn import imcflow_min_max_quantize, imcflow_nu_quantize
from tvm.relay.op.nn import imcflow_batch_norm, imcflow_qconv2d

def getModel(): 
  # Load checkpoint
  checkpoint_path = '/root/project/tvm/tvm_practice/models/model_best.pth.tar'
  checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
  model_dict = checkpoint['state_dict']
  
  # Input matching the checkpoint model (3 channels for RGB input)
  input_ = relay.var("input", shape=(1, 3, 32, 32))

  # First conv layer: 3 -> 16 channels
  y = relay.nn.conv2d(
      input_,
      relay.var("conv1_weight", shape=(16, 3, 3, 3)),
      channels=16,
      in_channels=3,
      kernel_size=(3, 3),
      padding=(1, 1),
  )
  y = relay.nn.bias_add(y, relay.var("conv1_bias", shape=(16,)))
  
  # First batch norm
  y = relay.nn.batch_norm(y, 
                         relay.var("bn1_gamma", shape=(16,), dtype="float32"), 
                         relay.var("bn1_beta", shape=(16,), dtype="float32"), 
                         relay.var("bn1_moving_mean", shape=(16,), dtype="float32"), 
                         relay.var("bn1_moving_var", shape=(16,), dtype="float32"))[0]
  y = relay.nn.relu(y)

  # Layer1.0.conv1: 16 -> 16 channels
  y = relay.nn.conv2d(
      y,
      relay.var("layer1_0_conv1_weight", shape=(16, 16, 3, 3)),
      channels=16,
      in_channels=16,
      kernel_size=(3, 3),
      padding=(1, 1),
  )
  
  # Layer1.0.bn1
  y = relay.nn.batch_norm(y, 
                         relay.var("layer1_0_bn1_gamma", shape=(16,), dtype="float32"), 
                         relay.var("layer1_0_bn1_beta", shape=(16,), dtype="float32"), 
                         relay.var("layer1_0_bn1_moving_mean", shape=(16,), dtype="float32"), 
                         relay.var("layer1_0_bn1_moving_var", shape=(16,), dtype="float32"))[0]
  y = relay.nn.relu(y)

  # Layer1.0.conv2: 16 -> 16 channels
  y = relay.nn.conv2d(
      y,
      relay.var("layer1_0_conv2_weight", shape=(16, 16, 3, 3)),
      channels=16,
      in_channels=16,
      kernel_size=(3, 3),
      padding=(1, 1),
  )

  # Extract real weights from checkpoint
  param_dict = {
    # Conv1 layer (only has weight and bias, no quantization params)
    "conv1_weight": model_dict['module.conv1.weight'].numpy(),
    "conv1_bias": model_dict['module.conv1.bias'].numpy(),
    
    # BN1 layer
    "bn1_gamma": model_dict['module.bn1.weight'].numpy(),
    "bn1_beta": model_dict['module.bn1.bias'].numpy(),
    "bn1_moving_mean": model_dict['module.bn1.running_mean'].numpy(),
    "bn1_moving_var": model_dict['module.bn1.running_var'].numpy(),
    
    # Layer1.0.conv1 (includes all conv-related parameters)
    "layer1_0_conv1_weight": model_dict['module.layer1.0.conv1.weight'].numpy(),
    "layer1_0_conv1_group_in_offset": model_dict['module.layer1.0.conv1.group_in_offset'].numpy(),
    "layer1_0_conv1_quant_func_s": model_dict['module.layer1.0.conv1.quant_func.s'].numpy(),
    "layer1_0_conv1_quant_func_init_state": model_dict['module.layer1.0.conv1.quant_func.init_state'].numpy(),
    
    # Layer1.0.bn1
    "layer1_0_bn1_gamma": model_dict['module.layer1.0.bn1.weight'].numpy(),
    "layer1_0_bn1_beta": model_dict['module.layer1.0.bn1.bias'].numpy(),
    "layer1_0_bn1_moving_mean": model_dict['module.layer1.0.bn1.running_mean'].numpy(),
    "layer1_0_bn1_moving_var": model_dict['module.layer1.0.bn1.running_var'].numpy(),
    
    # Layer1.0.conv2 (includes all conv-related parameters)
    "layer1_0_conv2_weight": model_dict['module.layer1.0.conv2.weight'].numpy(),
    "layer1_0_conv2_group_in_offset": model_dict['module.layer1.0.conv2.group_in_offset'].numpy(),
    "layer1_0_conv2_quant_func_s": model_dict['module.layer1.0.conv2.quant_func.s'].numpy(),
    "layer1_0_conv2_quant_func_init_state": model_dict['module.layer1.0.conv2.quant_func.init_state'].numpy(),
  }

  out = tvm.IRModule.from_expr(y)

  return out, param_dict

def getModelV2():
  input_ = relay.var("input", shape=(1, 28, 16, 16))

  y = relay.nn.conv2d(
      input_,
      relay.var("weight1", shape=(28, 28, 3, 3)),
      channels=28,
      in_channels=28,
      kernel_size=(3, 3),
      padding=(1, 1),
  )

  y = relay.nn.bias_add(y, relay.var("bias1", shape=(28,)))
  y = relay.nn.batch_norm(y, relay.var("gamma", shape=(28,), dtype="float32"), 
                             relay.var("beta", shape=(28,), dtype="float32"), 
                             relay.var("moving_mean", shape=(28,), dtype="float32"), 
                             relay.var("moving_var", shape=(28,), dtype="float32"))[0]
  y = relay.nn.relu(y)

  y = relay.nn.conv2d(
    y,
    relay.var("weight2_0", shape=(64,28,3,3)),
    channels=64,
    in_channels=28,
    kernel_size=(3, 3),
    padding=(1, 1),
  )

  y = relay.nn.batch_norm(y, relay.var("gamma2", shape=(64,), dtype="float32"),
                             relay.var("beta2", shape=(64,), dtype="float32"),
                             relay.var("moving_mean2", shape=(64,), dtype="float32"),
                             relay.var("moving_var2", shape=(64,), dtype="float32"))[0]
  y = relay.nn.relu(y)

  param_dict = {
    "weight1": np.random.rand(28, 28, 3, 3).astype("float32"),
    "bias1"  : np.random.rand(28).astype("float32"),
    "weight2_0": np.random.rand(64,28,3,3).astype("float32"),
    "gamma": np.random.rand(28).astype("float32"),
    "beta": np.random.rand(28).astype("float32"),
    "moving_mean": np.random.rand(28).astype("float32"),
    "moving_var": np.random.rand(28).astype("float32"),
    "gamma2": np.random.rand(64).astype("float32"),
    "beta2": np.random.rand(64).astype("float32"),
    "moving_mean2": np.random.rand(64).astype("float32"),
    "moving_var2": np.random.rand(64).astype("float32"),
  }

  out = tvm.IRModule.from_expr(y)

  return out, param_dict

def getOneConvModel():
  N, IC, IH, IW = 2, 32, 4, 4
  OC, KH, KW = 128, 3, 3
  padding = (0, 0)
  stride = (1, 1)
  OH, OW = (IH - KH + 2 * padding[0]) // stride[0] + 1, (IW - KW + 2 * padding[1]) // stride[1] + 1

  atom_IC = math.floor(256/(KH*KW))
  atom_OC = 64
  ic_gnum = math.ceil(IC/atom_IC)
  oc_gnum = math.ceil(OC/atom_OC)

  input = relay.var("input", shape=(N,ic_gnum,IH,IW,4,8), dtype="int32")
  y = imcflow_qconv2d(
    input,
    relay.var("weight", shape=(oc_gnum,ic_gnum,256,8), dtype="int32"),
    channels=OC,
    in_channels=IC,
    kernel_size=(KH, KW),
    padding=padding,
    strides=stride,
    out_dtype="int16"
  )

  weight_numpy = np.random.rand(oc_gnum,ic_gnum,256,8).astype("int32")
  print(weight_numpy.dtype)
  param_dict = {
    "weight": weight_numpy
  }

  out = tvm.IRModule.from_expr(y)

  return out, param_dict

def getOneConvQuantModel():
  N, IC, IH, IW = 2, 32, 4, 4
  OC, KH, KW = 128, 3, 3
  padding = (0, 0)
  stride = (1, 1)
  OH, OW = (IH - KH + 2 * padding[0]) // stride[0] + 1, (IW - KW + 2 * padding[1]) // stride[1] + 1

  atom_IC = math.floor(256/(KH*KW))
  atom_OC = 64
  ic_gnum = math.ceil(IC/atom_IC)
  oc_gnum = math.ceil(OC/atom_OC)

  input = relay.var("input", shape=(N,ic_gnum,IH,IW,4,8), dtype="int32")
  y = imcflow_qconv2d(
    input,
    relay.var("weight", shape=(oc_gnum,ic_gnum,256,8), dtype="int32"),
    channels=OC,
    in_channels=IC,
    kernel_size=(KH, KW),
    padding=padding,
    strides=stride,
    out_dtype="int16"
  )
  y = imcflow_min_max_quantize(y, relay.const(0, "int16"), relay.const(1, "int16"), 1, "int16", "int16")

  weight_numpy = np.random.rand(oc_gnum,ic_gnum,256,8).astype("int32")
  print(weight_numpy.dtype)
  param_dict = {
    "weight": weight_numpy
  }

  out = tvm.IRModule.from_expr(y)

  return out, param_dict

def getOneReluModel():
  # input_ = relay.var("input", shape=(1, 28, 4, 4))
  input_ = relay.var("input", shape=(1,28,4,4), dtype="int16")
  y = relay.nn.pad(input_, pad_width=((0, 0),(0, 0),(1, 1),(1,1)), pad_value=0)
  y = relay.nn.relu(y)

  param_dict = {
    # "quant_scale": np.random.rand(28).astype("float32"),
    # "quant_zp": np.random.randint(0, 255, 28).astype("int"),
    # "weight": np.random.rand(28,28,3,3).astype("float32")
  }


  out = tvm.IRModule.from_expr(y)

  return out, param_dict