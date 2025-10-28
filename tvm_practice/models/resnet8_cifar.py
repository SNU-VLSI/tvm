import pytest
import itertools
import numpy as np
import sys
import subprocess
import math
import collections
import os

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

from tvm.relay.backend.contrib.imcflow.acim_util import ConfigData

from .utils import get_param_info_from_relay_func

def get_height(H, KH, padding, stride):
    pad_h = padding
    out_h = (H + 2 * pad_h - KH) // stride + 1
    return out_h

def get_width(W, KW, padding, stride):
    pad_w = padding
    out_w = (W + 2 * pad_w - KW) // stride + 1
    return out_w

def getModel_(input_shape):
  input = relay.var("input", shape=input_shape, dtype="float32")
  N, IC, H, W = input_shape

  y = relay.nn.conv2d(
      input,
      relay.var("weight1", shape=(16, 3, 3, 3), dtype="float32"),
      in_channels=3,
      channels=16,
      kernel_size=(3, 3),
      padding=(1, 1),
  )

  N, IC, H, W = (N, 16, get_height(H, 3, 1, 1), get_width(W, 3, 1, 1))

  y = relay.nn.batch_norm(y, 
                          relay.var("bn_gamma", shape=(16,), dtype="float32"), relay.var("bn_beta", shape=(16,), dtype="float32"), 
                          relay.var("bn_moving_mean", shape=(16,), dtype="float32"), relay.var("bn_moving_var", shape=(16,), dtype="float32"))[0]
  
  y = y * relay.var("x_f_1", shape=(1,), dtype="float32")
  y = relay.cast(y, dtype="int16")

  # basic block 1
  residual = y
  y = imcflow_min_max_quantize(y, relay.var("quant_min_1", shape=(), dtype="int16"), relay.var("quant_max_1", shape=(), dtype="int16"), axis=1, out_dtype="uint8", channel=16)
  y = imcflow_qconv2d(
    y,
    relay.var("weight2_1", shape=(16,16,3,3), dtype="int8"),
    ConfigData((N, IC, H, W), (16,16,3,3), padding=1, stride=1).get_as_const_tensor(),
    in_channels=16,
    channels=16,
    kernel_size=(3, 3),
    padding=(1, 1),
    out_dtype="int16"
  )
  IC, H, W = (16, get_height(H, 3, 1, 1), get_width(W, 3, 1, 1))

  y = imcflow_batch_norm(y, relay.var("fused_scale1", shape=(16,), dtype="int16"), relay.var("fused_bias1", shape=(16,), dtype="int16"))
  y = imcflow_min_max_quantize(y, relay.var("quant_min_2", shape=(), dtype="int16"), relay.var("quant_max_2", shape=(), dtype="int16"), axis=1, out_dtype="uint8", channel=16)
  y = imcflow_qconv2d(
    y,
    relay.var("weight2_2", shape=(16,16,3,3), dtype="int8"),
    ConfigData((N, IC, H, W), (16,16,3,3), padding=1, stride=1).get_as_const_tensor(),
    in_channels=16,
    channels=16,
    kernel_size=(3, 3),
    padding=(1, 1),
    out_dtype="int16"
  )
  IC, H, W = (16, get_height(H, 3, 1, 1), get_width(W, 3, 1, 1))

  y = imcflow_batch_norm(y, relay.var("fused_scale2", shape=(16,), dtype="int16"), relay.var("fused_bias2", shape=(16,), dtype="int16"))
  y = y + residual * relay.var("y_f_1", shape=(1,), dtype="int16")

  # basic block 2
  residual = y
  IC_res, H_res, W_res = IC, H, W
  y = imcflow_min_max_quantize(y, relay.var("quant_min_3", shape=(), dtype="int16"), relay.var("quant_max_3", shape=(), dtype="int16"), axis=1, out_dtype="uint8", channel=16)
  y = imcflow_qconv2d(
    y,
    relay.var("weight3_1", shape=(32,16,3,3), dtype="int8"),
    ConfigData((N, IC, H, W), (32,16,3,3), padding=1, stride=2).get_as_const_tensor(),
    in_channels=16,
    channels=32,
    kernel_size=(3, 3),
    padding=(1, 1),
    strides=(2,2),
    out_dtype="int16"
  )
  IC, H, W = (32, get_height(H, 3, 1, 2), get_width(W, 3, 1, 2))

  y = imcflow_batch_norm(y, relay.var("fused_scale3", shape=(32,), dtype="int16"), relay.var("fused_bias3", shape=(32,), dtype="int16"))
  y = imcflow_min_max_quantize(y, relay.var("quant_min_4", shape=(), dtype="int16"), relay.var("quant_max_4", shape=(), dtype="int16"), axis=1, out_dtype="uint8", channel=32)
  y = imcflow_qconv2d(
    y,
    relay.var("weight3_2", shape=(32,32,3,3), dtype="int8"),
    ConfigData((N, IC, H, W), (32,32,3,3), padding=1, stride=1).get_as_const_tensor(),
    in_channels=32,
    channels=32,
    kernel_size=(3, 3),
    padding=(1, 1),
    out_dtype="int16"
  )
  IC, H, W = (32, get_height(H, 3, 1, 1), get_width(W, 3, 1, 1))

  y = imcflow_batch_norm(y, relay.var("fused_scale4", shape=(32,), dtype="int16"), relay.var("fused_bias4", shape=(32,), dtype="int16"))

  y_residual = imcflow_min_max_quantize(residual, relay.var("quant_min_4_2", shape=(), dtype="int16"), relay.var("quant_max_4_2", shape=(), dtype="int16"), axis=1, out_dtype="uint8", channel=32)
  y_residual = imcflow_qconv2d(
    y_residual,
    relay.var("weight3_0", shape=(32,16,1,1), dtype="int8"),
    ConfigData((N, IC_res, H_res, W_res), (32,16,1,1), padding=0, stride=2).get_as_const_tensor(),
    in_channels=16,
    channels=32,
    kernel_size=(1, 1),
    strides=(2,2),
    out_dtype="int16"
  )
  y_residual = imcflow_batch_norm(y_residual, relay.var("fused_scale4_2", shape=(32,), dtype="int16"), relay.var("fused_bias4_2", shape=(32,), dtype="int16"))

  y_residual = relay.var("bn_out_f_1", shape=(32,1,1), dtype="int16") * y_residual + relay.var("bn_out_f_0", shape=(32,1,1), dtype="int16")
  y = y + y_residual

  # basic block 3
  residual = y
  IC_res, H_res, W_res = IC, H, W
  y = imcflow_min_max_quantize(y, relay.var("quant_min_5", shape=(), dtype="int16"), relay.var("quant_max_5", shape=(), dtype="int16"), axis=1, out_dtype="uint8", channel=32)
  y = imcflow_qconv2d(
    y,
    relay.var("weight4_1", shape=(64,32,3,3), dtype="int8"),
    ConfigData((N, IC, H, W), (64,32,3,3), padding=1, stride=2).get_as_const_tensor(),
    in_channels=32,
    channels=64,
    kernel_size=(3, 3),
    padding=(1, 1),
    strides=(2,2),
    out_dtype="int16"
  )
  IC, H, W = (64, get_height(H, 3, 1, 2), get_width(W, 3, 1, 2))

  y = imcflow_batch_norm(y, relay.var("fused_scale5", shape=(64,), dtype="int16"), relay.var("fused_bias5", shape=(64,), dtype="int16"))
  y = imcflow_min_max_quantize(y, relay.var("quant_min_6", shape=(), dtype="int16"), relay.var("quant_max_6", shape=(), dtype="int16"), axis=1, out_dtype="uint8", channel=64)
  y = imcflow_qconv2d(
    y,
    relay.var("weight4_2", shape=(64,64,3,3), dtype="int8"),
    ConfigData((N, IC, H, W), (64,64,3,3), padding=1, stride=1).get_as_const_tensor(),
    in_channels=64,
    channels=64,
    kernel_size=(3, 3),
    padding=(1, 1),
    out_dtype="int16"
  )
  IC, H, W = (64, get_height(H, 3, 1, 1), get_width(W, 3, 1, 1))

  y = imcflow_batch_norm(y, relay.var("fused_scale6", shape=(64,), dtype="int16"), relay.var("fused_bias6", shape=(64,), dtype="int16"))

  y_residual = imcflow_min_max_quantize(residual, relay.var("quant_min_6_2", shape=(), dtype="int16"), relay.var("quant_max_6_2", shape=(), dtype="int16"), axis=1, out_dtype="uint8", channel=64)
  y_residual = imcflow_qconv2d(
    y_residual,
    relay.var("weight4_0", shape=(64,32,1,1), dtype="int8"),
    ConfigData((N, IC_res, H_res, W_res), (64,32,1,1), padding=0, stride=2).get_as_const_tensor(),
    in_channels=32,
    channels=64,
    kernel_size=(1, 1),
    strides=(2,2),
    out_dtype="int16"
  )
  y_residual = imcflow_batch_norm(y_residual, relay.var("fused_scale6_2", shape=(64,), dtype="int16"), relay.var("fused_bias6_2", shape=(64,), dtype="int16"))

  y_residual = relay.var("bn_out_f_3", shape=(64,1,1), dtype="int16") * y_residual + relay.var("bn_out_f_2", shape=(64,1,1), dtype="int16")

  y = y + y_residual

  # post process
  y = relay.cast(y,dtype="float32") * relay.var("post_f_inv", shape=(1,), dtype="float32")
  y = relay.nn.relu(y)
  y = relay.nn.adaptive_avg_pool2d(y, output_size=(1,1))
  y = relay.nn.batch_flatten(y) 
  y = relay.nn.dense(y, relay.var("dense_weight", shape=(10, 64), dtype="float32"))
  y = relay.nn.bias_add(y, relay.var("dense_bias", shape=(10,), dtype="float32"))

  # Collect parameter vars from the graph (exclude the input var)
  free_vars = relay.analysis.free_vars(y)
  var_info = {}
  for v in free_vars:
    if v is input:
      continue
    name = v.name_hint
    # Deduplicate by name in case of separately-constructed Vars with the same name
    if name in var_info:
      continue
    ttype = v.type_annotation
    if isinstance(ttype, relay.ty.TensorType):
      # Convert TVM shape (IntImm / PrimExpr) to Python ints when possible
      shape = []
      for dim in ttype.shape:
        try:
          shape.append(int(dim))
        except Exception:
          # Fallback if dynamic: leave as-is
          shape.append(dim)
      var_info[name] = {"shape": tuple(shape), "dtype": ttype.dtype}
    else:
      # If no TensorType annotation, skip or set defaults
      continue

  out = tvm.IRModule.from_expr(y)

  return out, var_info

def getModel_2(input_shape):
  input = relay.var("input", shape=input_shape, dtype="float32")

  y = relay.nn.conv2d(
      input,
      relay.var("weight1", shape=(16, 3, 3, 3), dtype="float32"),
      in_channels=3,
      channels=16,
      kernel_size=(3, 3),
      padding=(1, 1),
  )

  y = relay.nn.batch_norm(y, 
                          relay.var("bn_gamma", shape=(16,), dtype="float32"), relay.var("bn_beta", shape=(16,), dtype="float32"), 
                          relay.var("bn_moving_mean", shape=(16,), dtype="float32"), relay.var("bn_moving_var", shape=(16,), dtype="float32"))[0]
  
  y = y * relay.var("x_f_1", shape=(1,), dtype="float32")
  y = relay.cast(y, dtype="int16")

  # basic block 1
  y = imcflow_min_max_quantize(y, relay.var("quant_min_1", shape=(), dtype="int16"), relay.var("quant_max_1", shape=(), dtype="int16"), axis=1, out_dtype="int8", channel=16)
  y = imcflow_qconv2d(
    y,
    relay.var("weight2_1", shape=(16,16,3,3), dtype="int8"),
    in_channels=16,
    channels=16,
    kernel_size=(3, 3),
    padding=(1, 1),
    out_dtype="int16"
  )
  y = imcflow_batch_norm(y, relay.var("fused_scale1", shape=(16,), dtype="int16"), relay.var("fused_bias1", shape=(16,), dtype="int16"))

  # post process
  y = relay.cast(y,dtype="float32") / relay.var("post_f", shape=(1,), dtype="float32")
  y = relay.nn.relu(y)
  y = relay.nn.adaptive_avg_pool2d(y, output_size=(1,1))
  y = relay.nn.batch_flatten(y) 
  y = relay.nn.dense(y, relay.var("dense_weight", shape=(10, 64), dtype="float32"))
  y = relay.nn.bias_add(y, relay.var("dense_bias", shape=(10,), dtype="float32"))

  # Collect parameter vars from the graph (exclude the input var)
  free_vars = relay.analysis.free_vars(y)
  var_info = {}
  for v in free_vars:
    if v is input:
      continue
    name = v.name_hint
    # Deduplicate by name in case of separately-constructed Vars with the same name
    if name in var_info:
      continue
    ttype = v.type_annotation
    if isinstance(ttype, relay.ty.TensorType):
      # Convert TVM shape (IntImm / PrimExpr) to Python ints when possible
      shape = []
      for dim in ttype.shape:
        try:
          shape.append(int(dim))
        except Exception:
          # Fallback if dynamic: leave as-is
          shape.append(dim)
      var_info[name] = {"shape": tuple(shape), "dtype": ttype.dtype}
    else:
      # If no TensorType annotation, skip or set defaults
      continue

  out = tvm.IRModule.from_expr(y)

  return out, var_info

def getModel():
  out, var_dict = getModel_([1, 3, 32, 32])
  # out, var_dict = getModel_([1, 3, 8, 8])

  def _rand_tensor(dtype: str, shape):
    # Handle common dtypes with appropriate ranges
    if dtype in ("float32", "float16", "float64"):
      return np.random.uniform(-1, 1, shape).astype(dtype)
    if dtype.startswith("int"):
      # Parse bit width if available (e.g., int4, int8, int16, int32)
      try:
        bits = int(dtype.replace("int", ""))
      except Exception:
        bits = 32
      if bits == 4:
        # No native int4 in numpy; store in int8 within valid int4 range
        return np.random.randint(-8, 8, size=shape, dtype=np.int8)
      if bits == 8:
        return np.random.randint(-128, 128, size=shape, dtype=np.int8)
      if bits == 16:
        return np.random.randint(-32768, 32768, size=shape, dtype=np.int16)
      if bits == 32:
        return np.random.randint(-2**31, 2**31, size=shape, dtype=np.int32)
      if bits == 64:
        return np.random.randint(-2**63, 2**63 - 1, size=shape, dtype=np.int64)
      # Fallback: use int32
      return np.random.randint(-2**31, 2**31, size=shape, dtype=np.int32)
    if dtype.startswith("uint"):
      try:
        bits = int(dtype.replace("uint", ""))
      except Exception:
        bits = 32
      if bits == 4:
        return np.random.randint(0, 16, size=shape, dtype=np.uint8)
      if bits == 8:
        return np.random.randint(0, 256, size=shape, dtype=np.uint8)
      if bits == 16:
        return np.random.randint(0, 2**16, size=shape, dtype=np.uint16)
      if bits == 32:
        return np.random.randint(0, 2**32, size=shape, dtype=np.uint32)
      if bits == 64:
        # numpy uint64 randint high is exclusive and must be <= 2**64-1
        return np.random.randint(0, np.iinfo(np.uint64).max, size=shape, dtype=np.uint64)
      return np.random.randint(0, 2**32, size=shape, dtype=np.uint32)
    # Default float32 if unrecognized
    return np.random.uniform(-1, 1, shape).astype("float32")

  params_dict = {}
  # Sort by name for determinism
  for name in sorted(var_dict.keys()):
    info = var_dict[name]
    params_dict[name] = _rand_tensor(info["dtype"], info["shape"])

  return out, params_dict

def getModel_from_pretrained_weight():
  import torch
  import sys
  import os
  # Add the models directory to path to import safe_convert
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  from safe_convert import to_int16, to_int4
  
  out, var_dict = getModel_([1, 3, 32, 32])
  
  # Load checkpoint
  checkpoint_path = '/root/project/tvm/tvm_practice/models/model_best.pth.tar'
  checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
  model_dict = checkpoint['state_dict']
  
  # Define adjustment factors (same as in main.py)
  adjust_factors = {
      'x_f_1': 36.0,
      'bn1_f_1': 72.0,
      'bn2_f_1': 36.0,
      'x_f_2': 36.0,  # needs to be the same with bn2_f_1
      'bn1_f_2': 150.0,
      'bn2_f_2': 15.0,
      'x_f_3': 10.0,
      'bn1_f_3': 50.0,
      'bn2_f_3': 500.0,
  }

  def _get_pretrained_weight(name, dtype: str, shape):
    """
    Map TVM Relay parameter names to PyTorch checkpoint weights.
    Applies necessary transformations for quantization and batch norm fusion.
    """
    # Helper function to extract layer state similar to make_layer_state_dict
    def get_layer_components(layer_idx):
      s_dict = model_dict
      layer_key = f"module.layer{layer_idx}.0"
      
      # Get conv weights and scales
      conv1_weight = s_dict[f"{layer_key}.conv1.weight"].cpu()
      conv1_weight_s = s_dict[f"{layer_key}.conv1.quant_func.s"].cpu()
      conv2_weight = s_dict[f"{layer_key}.conv2.weight"].cpu()
      conv2_weight_s = s_dict[f"{layer_key}.conv2.quant_func.s"].cpu()
      
      # Get batch norm parameters
      bn1_var = s_dict[f"{layer_key}.bn1.running_var"].cpu()
      bn1_gamma = s_dict[f"{layer_key}.bn1.weight"].cpu()
      bn1_mean = s_dict[f"{layer_key}.bn1.running_mean"].cpu()
      bn1_beta = s_dict[f"{layer_key}.bn1.bias"].cpu()
      bn2_var = s_dict[f"{layer_key}.bn2.running_var"].cpu()
      bn2_gamma = s_dict[f"{layer_key}.bn2.weight"].cpu()
      bn2_mean = s_dict[f"{layer_key}.bn2.running_mean"].cpu()
      bn2_beta = s_dict[f"{layer_key}.bn2.bias"].cpu()
      
      # Get activation scales
      act1_s = s_dict[f"{layer_key}.act1.s"].cpu()
      act2_s = s_dict[f"{layer_key}.act2.s"].cpu()
      
      # Compute fused batch norm parameters
      bn1_scale = bn1_gamma / torch.sqrt(bn1_var)
      bn1_bias = bn1_beta - bn1_gamma * bn1_mean / torch.sqrt(bn1_var)
      bn2_scale = bn2_gamma / torch.sqrt(bn2_var)
      bn2_bias = bn2_beta - bn2_gamma * bn2_mean / torch.sqrt(bn2_var)
      
      # Check for downsample
      downsample_key = f"{layer_key}.downsample.1.weight"
      has_downsample = downsample_key in s_dict
      
      if has_downsample:
        downsample_weight = s_dict[downsample_key].cpu()
        downsample_weight_s = s_dict[f"{layer_key}.downsample.1.quant_func.s"].cpu()
        downsample_act_s = s_dict[f"{layer_key}.downsample.0.s"].cpu()
      else:
        downsample_weight = None
        downsample_weight_s = None
        downsample_act_s = None
      
      return {
        'conv1_weight': conv1_weight,
        'conv1_weight_s': conv1_weight_s,
        'conv2_weight': conv2_weight,
        'conv2_weight_s': conv2_weight_s,
        'bn1_scale': bn1_scale,
        'bn1_bias': bn1_bias,
        'bn2_scale': bn2_scale,
        'bn2_bias': bn2_bias,
        'act1_s': act1_s,
        'act2_s': act2_s,
        'downsample_weight': downsample_weight,
        'downsample_weight_s': downsample_weight_s,
        'downsample_act_s': downsample_act_s,
      }
    
    # ========== Conv1 and BN1 (first layer, not quantized) ==========
    if name == "weight1":
      return model_dict["module.conv1.weight"].cpu().numpy()
    elif name == "bn_gamma":
      return model_dict["module.bn1.weight"].cpu().numpy()
    elif name == "bn_beta":
      return model_dict["module.bn1.bias"].cpu().numpy()
    elif name == "bn_moving_mean":
      return model_dict["module.bn1.running_mean"].cpu().numpy()
    elif name == "bn_moving_var":
      return model_dict["module.bn1.running_var"].cpu().numpy()
    elif name == "x_f_1":
      return np.array([adjust_factors['x_f_1']], dtype=np.float32)
    
    # ========== Layer 1 (Basic Block 1) ==========
    elif name.startswith("weight2_") or name.startswith("fused_scale1") or name.startswith("fused_bias1") or name.startswith("quant_min_1") or name.startswith("quant_max_1") or name.startswith("quant_min_2") or name.startswith("quant_max_2") or name.startswith("fused_scale2") or name.startswith("fused_bias2") or name == "y_f_1":
      layer1 = get_layer_components(1)
      
      if name == "weight2_1":
        # Quantize conv1 weight to int8
        weight_q = torch.round(torch.clamp(
            layer1['conv1_weight'] / layer1['conv1_weight_s'], -8, 7))
        return to_int4(weight_q).numpy().astype(np.int8)
      elif name == "weight2_2":
        # Quantize conv2 weight to int8
        weight_q = torch.round(torch.clamp(
            layer1['conv2_weight'] / layer1['conv2_weight_s'], -8, 7))
        return to_int4(weight_q).numpy().astype(np.int8)
      elif name == "fused_scale1":
        # Fused scale for BN1: act1_s * conv1_weight_s * bn1_scale * bn1_f
        fused = layer1['act1_s'] * layer1['conv1_weight_s'] * layer1['bn1_scale'] * adjust_factors['bn1_f_1']
        return to_int16(fused, assert_range=True).numpy().astype(np.int16)
      elif name == "fused_bias1":
        # Fused bias for BN1: bn1_bias * bn1_f
        fused = layer1['bn1_bias'] * adjust_factors['bn1_f_1']
        return to_int16(fused, assert_range=True).numpy().astype(np.int16)
      elif name == "fused_scale2":
        # Fused scale for BN2: act2_s * conv2_weight_s * bn2_scale * bn2_f
        fused = layer1['act2_s'] * layer1['conv2_weight_s'] * layer1['bn2_scale'] * adjust_factors['bn2_f_1']
        return to_int16(fused, assert_range=True).numpy().astype(np.int16)
      elif name == "fused_bias2":
        # Fused bias for BN2: bn2_bias * bn2_f
        fused = layer1['bn2_bias'] * adjust_factors['bn2_f_1']
        return to_int16(fused, assert_range=True).numpy().astype(np.int16)
      elif name == "quant_min_1":
        # Min for act1: -0.5 * act1_s * x_f
        val = -0.5 * layer1['act1_s'] * adjust_factors['x_f_1']
        return to_int16(val, assert_range=True).numpy().astype(np.int16)
      elif name == "quant_max_1":
        # Max for act1: 15.5 * act1_s * x_f
        val = 15.5 * layer1['act1_s'] * adjust_factors['x_f_1']
        return to_int16(val, assert_range=True).numpy().astype(np.int16)
      elif name == "quant_min_2":
        # Min for act2: -0.5 * act2_s * bn1_f
        val = -0.5 * layer1['act2_s'] * adjust_factors['bn1_f_1']
        return to_int16(val, assert_range=True).numpy().astype(np.int16)
      elif name == "quant_max_2":
        # Max for act2: 15.5 * act2_s * bn1_f
        val = 15.5 * layer1['act2_s'] * adjust_factors['bn1_f_1']
        return to_int16(val, assert_range=True).numpy().astype(np.int16)
      elif name == "y_f_1":
        # Residual scaling factor: bn2_f / x_f
        val = adjust_factors['bn2_f_1'] / adjust_factors['x_f_1']
        return np.array([to_int16(val, assert_range=True).numpy().astype(np.int16)], dtype=np.int16)
    
    # ========== Layer 2 (Basic Block 2) ==========
    elif name.startswith("weight3_") or name.startswith("fused_scale3") or name.startswith("fused_bias3") or name.startswith("quant_min_3") or name.startswith("quant_max_3") or name.startswith("quant_min_4") or name.startswith("quant_max_4") or name.startswith("fused_scale4") or name.startswith("fused_bias4") or name.startswith("bn_out_f_0") or name.startswith("bn_out_f_1") or name.startswith("quant_min_4_2") or name.startswith("quant_max_4_2") or name.startswith("fused_scale4_2") or name.startswith("fused_bias4_2"):
      layer2 = get_layer_components(2)
      
      if name == "weight3_0":
        # Downsample weight (1x1 conv)
        weight_q = torch.round(torch.clamp(
            layer2['downsample_weight'] / layer2['downsample_weight_s'], -8, 7))
        return to_int4(weight_q).numpy().astype(np.int8)
      elif name == "weight3_1":
        # Conv1 weight
        weight_q = torch.round(torch.clamp(
            layer2['conv1_weight'] / layer2['conv1_weight_s'], -8, 7))
        return to_int4(weight_q).numpy().astype(np.int8)
      elif name == "weight3_2":
        # Conv2 weight
        weight_q = torch.round(torch.clamp(
            layer2['conv2_weight'] / layer2['conv2_weight_s'], -8, 7))
        return to_int4(weight_q).numpy().astype(np.int8)
      elif name == "fused_scale3":
        fused = layer2['act1_s'] * layer2['conv1_weight_s'] * layer2['bn1_scale'] * adjust_factors['bn1_f_2']
        return to_int16(fused, assert_range=True).numpy().astype(np.int16)
      elif name == "fused_bias3":
        fused = layer2['bn1_bias'] * adjust_factors['bn1_f_2']
        return to_int16(fused, assert_range=True).numpy().astype(np.int16)
      elif name == "fused_scale4":
        fused = layer2['act2_s'] * layer2['conv2_weight_s'] * layer2['bn2_scale'] * adjust_factors['bn2_f_2']
        return to_int16(fused, assert_range=True).numpy().astype(np.int16)
      elif name == "fused_bias4":
        fused = layer2['bn2_bias'] * adjust_factors['bn2_f_2']
        return to_int16(fused, assert_range=True).numpy().astype(np.int16)
      elif name == "fused_scale4_2":
        # Downsample batch norm scale
        if layer2['downsample_weight_s'] is not None and layer2['downsample_act_s'] is not None:
          downsample_scale = torch.ones_like(layer2['bn1_scale']) * layer2['downsample_weight_s'] * layer2['downsample_act_s'] * adjust_factors['bn2_f_2']
        else:
          # If no downsample, use ones
          downsample_scale = torch.ones(32)
        return to_int16(downsample_scale, assert_range=True).numpy().astype(np.int16)
      elif name == "fused_bias4_2":
        # Downsample batch norm bias (zeros)
        downsample_bias = torch.zeros(32)
        return to_int16(downsample_bias, assert_range=True).numpy().astype(np.int16)
      elif name == "quant_min_3":
        val = -0.5 * layer2['act1_s'] * adjust_factors['x_f_2']
        return to_int16(val, assert_range=True).numpy().astype(np.int16)
      elif name == "quant_max_3":
        val = 15.5 * layer2['act1_s'] * adjust_factors['x_f_2']
        return to_int16(val, assert_range=True).numpy().astype(np.int16)
      elif name == "quant_min_4":
        val = -0.5 * layer2['act2_s'] * adjust_factors['bn1_f_2']
        return to_int16(val, assert_range=True).numpy().astype(np.int16)
      elif name == "quant_max_4":
        val = 15.5 * layer2['act2_s'] * adjust_factors['bn1_f_2']
        return to_int16(val, assert_range=True).numpy().astype(np.int16)
      elif name == "quant_min_4_2":
        # Downsample activation min
        val = -0.5 * layer2['downsample_act_s'] * adjust_factors['x_f_2']
        return to_int16(val, assert_range=True).numpy().astype(np.int16)
      elif name == "quant_max_4_2":
        # Downsample activation max
        val = 15.5 * layer2['downsample_act_s'] * adjust_factors['x_f_2']
        return to_int16(val, assert_range=True).numpy().astype(np.int16)
      elif name == "bn_out_f_0":
        # Constant term for residual adjustment (zeros with shape matching output channels)
        return np.zeros((32, 1, 1), dtype=np.int16)
      elif name == "bn_out_f_1":
        # Scale term for residual adjustment (ones)
        return np.ones((32, 1, 1), dtype=np.int16)
    
    # ========== Layer 3 (Basic Block 3) ==========
    elif name.startswith("weight4_") or name.startswith("fused_scale5") or name.startswith("fused_bias5") or name.startswith("quant_min_5") or name.startswith("quant_max_5") or name.startswith("quant_min_6") or name.startswith("quant_max_6") or name.startswith("fused_scale6") or name.startswith("fused_bias6") or name.startswith("bn_out_f_2") or name.startswith("bn_out_f_3") or name.startswith("quant_min_6_2") or name.startswith("quant_max_6_2") or name.startswith("fused_scale6_2") or name.startswith("fused_bias6_2"):
      layer3 = get_layer_components(3)
      
      if name == "weight4_0":
        # Downsample weight (1x1 conv)
        weight_q = torch.round(torch.clamp(
            layer3['downsample_weight'] / layer3['downsample_weight_s'], -8, 7))
        return to_int4(weight_q).numpy().astype(np.int8)
      elif name == "weight4_1":
        # Conv1 weight
        weight_q = torch.round(torch.clamp(
            layer3['conv1_weight'] / layer3['conv1_weight_s'], -8, 7))
        return to_int4(weight_q).numpy().astype(np.int8)
      elif name == "weight4_2":
        # Conv2 weight
        weight_q = torch.round(torch.clamp(
            layer3['conv2_weight'] / layer3['conv2_weight_s'], -8, 7))
        return to_int4(weight_q).numpy().astype(np.int8)
      elif name == "fused_scale5":
        fused = layer3['act1_s'] * layer3['conv1_weight_s'] * layer3['bn1_scale'] * adjust_factors['bn1_f_3']
        return to_int16(fused, assert_range=True).numpy().astype(np.int16)
      elif name == "fused_bias5":
        fused = layer3['bn1_bias'] * adjust_factors['bn1_f_3']
        return to_int16(fused, assert_range=True).numpy().astype(np.int16)
      elif name == "fused_scale6":
        fused = layer3['act2_s'] * layer3['conv2_weight_s'] * layer3['bn2_scale'] * adjust_factors['bn2_f_3']
        return to_int16(fused, assert_range=True).numpy().astype(np.int16)
      elif name == "fused_bias6":
        fused = layer3['bn2_bias'] * adjust_factors['bn2_f_3']
        return to_int16(fused, assert_range=True).numpy().astype(np.int16)
      elif name == "fused_scale6_2":
        # Downsample batch norm scale
        if layer3['downsample_weight_s'] is not None and layer3['downsample_act_s'] is not None:
          downsample_scale = torch.ones_like(layer3['bn1_scale']) * layer3['downsample_weight_s'] * layer3['downsample_act_s'] * adjust_factors['bn2_f_3']
        else:
          downsample_scale = torch.ones(64)
        return to_int16(downsample_scale, assert_range=True).numpy().astype(np.int16)
      elif name == "fused_bias6_2":
        # Downsample batch norm bias (zeros)
        downsample_bias = torch.zeros(64)
        return to_int16(downsample_bias, assert_range=True).numpy().astype(np.int16)
      elif name == "quant_min_5":
        val = -0.5 * layer3['act1_s'] * adjust_factors['x_f_3']
        return to_int16(val, assert_range=True).numpy().astype(np.int16)
      elif name == "quant_max_5":
        val = 15.5 * layer3['act1_s'] * adjust_factors['x_f_3']
        return to_int16(val, assert_range=True).numpy().astype(np.int16)
      elif name == "quant_min_6":
        val = -0.5 * layer3['act2_s'] * adjust_factors['bn1_f_3']
        return to_int16(val, assert_range=True).numpy().astype(np.int16)
      elif name == "quant_max_6":
        val = 15.5 * layer3['act2_s'] * adjust_factors['bn1_f_3']
        return to_int16(val, assert_range=True).numpy().astype(np.int16)
      elif name == "quant_min_6_2":
        # Downsample activation min
        val = -0.5 * layer3['downsample_act_s'] * adjust_factors['x_f_3']
        return to_int16(val, assert_range=True).numpy().astype(np.int16)
      elif name == "quant_max_6_2":
        # Downsample activation max
        val = 15.5 * layer3['downsample_act_s'] * adjust_factors['x_f_3']
        return to_int16(val, assert_range=True).numpy().astype(np.int16)
      elif name == "bn_out_f_2":
        # Constant term for residual adjustment (zeros with shape matching output channels)
        return np.zeros((64, 1, 1), dtype=np.int16)
      elif name == "bn_out_f_3":
        # Scale term for residual adjustment (ones)
        return np.ones((64, 1, 1), dtype=np.int16)
    
    # ========== Post-processing and FC layer ==========
    elif name == "post_f_inv":
      # Inverse of bn2_f_3 for dequantization
      return np.array([1.0 / adjust_factors['bn2_f_3']], dtype=np.float32)
    elif name == "dense_weight":
      return model_dict["module.fc.weight"].cpu().numpy()
    elif name == "dense_bias":
      return model_dict["module.fc.bias"].cpu().numpy()
    
    else:
      raise ValueError(f"Unknown parameter name: {name} with dtype={dtype}, shape={shape}")

  params_dict = {}
  # Sort by name for determinism
  for name in sorted(var_dict.keys()):
    # Skip the input variable - it's not a parameter
    if name == "input":
      continue
    info = var_dict[name]
    param = _get_pretrained_weight(name, info["dtype"], info["shape"])
    
    # Sanity check: verify shape and dtype
    expected_shape = info["shape"]
    expected_dtype = info["dtype"]
    
    if not isinstance(param, np.ndarray):
      raise TypeError(f"Parameter '{name}' should be numpy array, got {type(param)}")
    
    if param.shape != expected_shape:
      raise ValueError(f"Parameter '{name}' shape mismatch: expected {expected_shape}, got {param.shape}")
    
    # Check dtype compatibility (handle numpy dtype naming variations)
    param_dtype_str = str(param.dtype)
    if expected_dtype.startswith("int") or expected_dtype.startswith("uint"):
      # For integer types, check the base type matches
      if not param_dtype_str.startswith(expected_dtype.split("int")[0] + "int"):
        raise TypeError(f"Parameter '{name}' dtype mismatch: expected {expected_dtype}, got {param_dtype_str}")
    elif expected_dtype.startswith("float"):
      # For float types, check the base type matches
      if not param_dtype_str.startswith("float"):
        raise TypeError(f"Parameter '{name}' dtype mismatch: expected {expected_dtype}, got {param_dtype_str}")
    else:
      # Exact match for other types
      if param_dtype_str != expected_dtype:
        raise TypeError(f"Parameter '{name}' dtype mismatch: expected {expected_dtype}, got {param_dtype_str}")
    
    params_dict[name] = param

  return out, params_dict

  