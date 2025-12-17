import numpy as np

import tvm
from tvm import relay

from tvm.relay.backend.contrib.imcflow.acim_util import ConfigData, AccMask
from tvm.relay.qnn.op.qnn import imcflow_min_max_quantize, imcflow_nu_quantize
from tvm.relay.op.nn import imcflow_batch_norm, imcflow_qconv2d
import numpy as np
import tvm
from tvm import relay
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


class EarlyStopException(Exception):
    """Exception raised to stop model construction early"""
    def __init__(self, y, input_var):
        self.y = y
        self.input_var = input_var
        super().__init__()


class RelayOpCounter:
    """Helper class to track relay operations and stop early if needed"""
    def __init__(self, until_relay, input_var):
        self.count = 0
        self.until_relay = until_relay
        self.input_var = input_var

    def check(self, y):
        """Increment counter and check if we should stop"""
        self.count += 1
        if self.until_relay is not None and self.count > self.until_relay:
            raise EarlyStopException(y, self.input_var)
        return y

def getModel_(input_shape, until_relay: int = None):
  """
  Get ResNet8 model with optional early stopping at specific relay operation.

  Args:
    input_shape: Input tensor shape (N, C, H, W)
    until_relay: Optional index to stop at (0-based). Stops after the Nth 'y = ...' assignment.
                 If None, returns full model.
  """
  input = relay.var("model_input", shape=input_shape, dtype="float32")
  N, IC, H, W = input_shape

  # Create counter that will automatically check and throw exception if needed
  c = RelayOpCounter(until_relay, input)

  try:
    y = c.check(relay.nn.conv2d(
        input,
        relay.var("weight1", shape=(16, 3, 3, 3), dtype="float32"),
        in_channels=3,
        channels=16,
        kernel_size=(3, 3),
        padding=(1, 1),
    ))

    N, IC, H, W = (N, 16, get_height(H, 3, 1, 1), get_width(W, 3, 1, 1))

    y = c.check(relay.nn.batch_norm(y,
                            relay.var("bn_gamma", shape=(16,), dtype="float32"), relay.var("bn_beta", shape=(16,), dtype="float32"),
                            relay.var("bn_moving_mean", shape=(16,), dtype="float32"), relay.var("bn_moving_var", shape=(16,), dtype="float32"))[0])

    y = c.check(y * relay.var("x_f_1", shape=(1,), dtype="float32"))
    y = c.check(relay.cast(y, dtype="int16"))

    # basic block 1
    residual = y
    y = c.check(imcflow_min_max_quantize(y, relay.var("quant_min_1", shape=(), dtype="int16"), relay.var("quant_max_1", shape=(), dtype="int16"), axis=1, out_dtype="uint8", channel=16))
    y = c.check(imcflow_qconv2d(
      y,
      relay.var("weight2_1", shape=(16,16,3,3), dtype="int8"),
      ConfigData((N, IC, H, W), (16,16,3,3), padding=1, stride=1).get_as_const_tensor(),
      in_channels=16,
      channels=16,
      kernel_size=(3, 3),
      padding=(1, 1),
      out_dtype="int16"
    ))
    IC, H, W = (16, get_height(H, 3, 1, 1), get_width(W, 3, 1, 1))

    y = c.check(imcflow_batch_norm(y, relay.var("fused_scale1", shape=(16,), dtype="int16"), relay.var("fused_bias1", shape=(16,), dtype="int16")))
    y = c.check(imcflow_min_max_quantize(y, relay.var("quant_min_2", shape=(), dtype="int16"), relay.var("quant_max_2", shape=(), dtype="int16"), axis=1, out_dtype="uint8", channel=16))
    y = c.check(imcflow_qconv2d(
      y,
      relay.var("weight2_2", shape=(16,16,3,3), dtype="int8"),
      ConfigData((N, IC, H, W), (16,16,3,3), padding=1, stride=1).get_as_const_tensor(),
      in_channels=16,
      channels=16,
      kernel_size=(3, 3),
      padding=(1, 1),
      out_dtype="int16"
    ))
    IC, H, W = (16, get_height(H, 3, 1, 1), get_width(W, 3, 1, 1))

    y = c.check(imcflow_batch_norm(y, relay.var("fused_scale2", shape=(16,), dtype="int16"), relay.var("fused_bias2", shape=(16,), dtype="int16")))
    y = c.check(y + residual * relay.var("y_f_1", shape=(1,), dtype="int16"))

    # basic block 2
    residual = y
    IC_res, H_res, W_res = IC, H, W
    y = c.check(imcflow_min_max_quantize(y, relay.var("quant_min_3", shape=(), dtype="int16"), relay.var("quant_max_3", shape=(), dtype="int16"), axis=1, out_dtype="uint8", channel=16))
    y = c.check(imcflow_qconv2d(
      y,
      relay.var("weight3_1", shape=(32,16,3,3), dtype="int8"),
      ConfigData((N, IC, H, W), (32,16,3,3), padding=1, stride=2).get_as_const_tensor(),
      in_channels=16,
      channels=32,
      kernel_size=(3, 3),
      padding=(1, 1),
      strides=(2,2),
      out_dtype="int16"
    ))
    IC, H, W = (32, get_height(H, 3, 1, 2), get_width(W, 3, 1, 2))

    y = c.check(imcflow_batch_norm(y, relay.var("fused_scale3", shape=(32,), dtype="int16"), relay.var("fused_bias3", shape=(32,), dtype="int16")))
    y = c.check(imcflow_min_max_quantize(y, relay.var("quant_min_4", shape=(), dtype="int16"), relay.var("quant_max_4", shape=(), dtype="int16"), axis=1, out_dtype="uint8", channel=32))
    y = c.check(imcflow_qconv2d(
      y,
      relay.var("weight3_2", shape=(32,32,3,3), dtype="int8"),
      ConfigData((N, IC, H, W), (32,32,3,3), padding=1, stride=1).get_as_const_tensor(),
      in_channels=32,
      channels=32,
      kernel_size=(3, 3),
      padding=(1, 1),
      out_dtype="int16"
    ))
    IC, H, W = (32, get_height(H, 3, 1, 1), get_width(W, 3, 1, 1))

    y = c.check(imcflow_batch_norm(y, relay.var("fused_scale4", shape=(32,), dtype="int16"), relay.var("fused_bias4", shape=(32,), dtype="int16")))

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
    y = c.check(y + y_residual)

    # basic block 3
    residual = y
    IC_res, H_res, W_res = IC, H, W
    y = c.check(imcflow_min_max_quantize(y, relay.var("quant_min_5", shape=(), dtype="int16"), relay.var("quant_max_5", shape=(), dtype="int16"), axis=1, out_dtype="uint8", channel=32))
    y = c.check(imcflow_qconv2d(
      y,
      relay.var("weight4_1", shape=(64,32,3,3), dtype="int8"),
      ConfigData((N, IC, H, W), (64,32,3,3), padding=1, stride=2).get_as_const_tensor(),
      in_channels=32,
      channels=64,
      kernel_size=(3, 3),
      padding=(1, 1),
      strides=(2,2),
      out_dtype="int16"
    ))
    IC, H, W = (64, get_height(H, 3, 1, 2), get_width(W, 3, 1, 2))

    y = c.check(imcflow_batch_norm(y, relay.var("fused_scale5", shape=(64,), dtype="int16"), relay.var("fused_bias5", shape=(64,), dtype="int16")))
    y = c.check(imcflow_min_max_quantize(y, relay.var("quant_min_6", shape=(), dtype="int16"), relay.var("quant_max_6", shape=(), dtype="int16"), axis=1, out_dtype="uint8", channel=64))
    y = c.check(imcflow_qconv2d(
      y,
      relay.var("weight4_2", shape=(64,64,3,3), dtype="int8"),
      ConfigData((N, IC, H, W), (64,64,3,3), padding=1, stride=1).get_as_const_tensor(),
      in_channels=64,
      channels=64,
      kernel_size=(3, 3),
      padding=(1, 1),
      out_dtype="int16"
    ))
    IC, H, W = (64, get_height(H, 3, 1, 1), get_width(W, 3, 1, 1))

    y = c.check(imcflow_batch_norm(y, relay.var("fused_scale6", shape=(64,), dtype="int16"), relay.var("fused_bias6", shape=(64,), dtype="int16")))

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

    y = c.check(y + y_residual)

    # post process
    y = c.check(relay.cast(y,dtype="float32") * relay.var("post_f_inv", shape=(1,), dtype="float32"))
    y = c.check(relay.nn.relu(y))
    y = c.check(relay.nn.adaptive_avg_pool2d(y, output_size=(1,1)))
    y = c.check(relay.nn.batch_flatten(y))
    y = c.check(relay.nn.dense(y, relay.var("dense_weight", shape=(10, 64), dtype="float32")))
    y = c.check(relay.nn.bias_add(y, relay.var("dense_bias", shape=(10,), dtype="float32")))

    # Full model - finalize
    return _finalize_model(y, input)

  except EarlyStopException as e:
    # Early stop triggered, return model at current point
    return _finalize_model(e.y, e.input_var)


def _finalize_model(y, input):
  """Helper function to finalize the model by collecting parameters and creating IRModule"""
  # Collect parameter vars from the graph (exclude the input var)
  free_vars = relay.analysis.free_vars(y)
  var_info = {}
  for v in free_vars:
    if v == input:
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


def getModel_from_pretrained_weight(small_debug=False, until_relay=None):
  """
  Get ResNet8 subset model with pretrained weights loaded from checkpoint.

  Args:
    small_debug: If True, use smaller input shape (8x8 instead of 32x32)
    until_relay: Optional relay operation index to stop at (0-based). If None, returns full model.

  Returns:
    (out, params_dict): IRModule and dictionary of parameter tensors
  """
  import torch
  import re

  if small_debug:
    out, var_dict = getModel_([1, 3, 8, 8], until_relay=until_relay)
  else:
    out, var_dict = getModel_([1, 3, 32, 32], until_relay=until_relay)

  # Load checkpoint
  checkpoint_path = '/root/project/tvm/tvm_practice/models/checkpoint.pth.tar'
  checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
  model_dict = checkpoint['state_dict']
  adjust_factors = checkpoint['adjust_factors']

  def _get_tensor_from_checkpoint(name, dtype, shape):
    """
    Get parameter tensor from checkpoint matching the given name.

    Args:
        name: TVM Relay parameter name
        dtype: Expected dtype (e.g., 'float32', 'int8', 'int16')
        shape: Expected shape tuple

    Returns:
        numpy.ndarray with the parameter data

    Raises:
        ValueError: If no matching parameter found in checkpoint
    """
    # Direct mappings for initial conv and bn layers
    direct_mappings = {
        'weight1': 'conv1.weight',
        'bn_gamma': 'bn1.weight',
        'bn_beta': 'bn1.bias',
        'bn_moving_mean': 'bn1.running_mean',
        'bn_moving_var': 'bn1.running_var',
        'dense_weight': 'fc.weight',
        'dense_bias': 'fc.bias',
    }

    if name in direct_mappings:
      key = direct_mappings[name]
      if key in model_dict:
        tensor = model_dict[key].cpu().numpy().astype(dtype)
        if tensor.shape != shape:
          raise ValueError(f"Shape mismatch for {name}: expected {shape}, got {tensor.shape}")
        return tensor
      else:
        raise ValueError(f"Key {key} not found in checkpoint for parameter {name}")

    # Handle scaling factors from adjust_factors
    if name == 'x_f_1':
      return np.array([adjust_factors['x_f_1']], dtype=dtype)

    if name == 'post_f_inv':
      # post_f_inv = 1.0 / bn2_f_3
      return np.array([1.0 / adjust_factors['bn2_f_3']], dtype=dtype)

    # Handle layer-specific parameters using regex patterns
    # Pattern for weight{2,3,4}_{0,1,2}
    weight_pattern = re.match(r'weight(\d)_(\d)', name)
    if weight_pattern:
      block_num = int(weight_pattern.group(1)) - 1  # weight2->layer1, weight3->layer2, weight4->layer3
      conv_idx = int(weight_pattern.group(2))

      layer_name = f"layer{block_num}"
      if conv_idx == 0:
        # Downsample conv
        key = f"{layer_name}.block_int16.downsample.1.weight"
      else:
        # Regular conv
        key = f"{layer_name}.block_int16.conv{conv_idx}.weight"

      if key in model_dict:
        tensor = model_dict[key].cpu().numpy().astype(dtype)
        if tensor.shape != shape:
          raise ValueError(f"Shape mismatch for {name}: expected {shape}, got {tensor.shape}")
        return tensor
      else:
        raise ValueError(f"Key {key} not found in checkpoint for parameter {name}")

    # Pattern for fused_scale{1-6} and fused_bias{1-6}
    fused_pattern = re.match(r'fused_(scale|bias)(\d+)(_2)?', name)
    if fused_pattern:
      param_type = fused_pattern.group(1)  # 'scale' or 'bias'
      idx = int(fused_pattern.group(2))
      is_downsample = fused_pattern.group(3) is not None

      # Map index to layer and bn
      mapping = {
          1: ('layer1', 'bn1'),
          2: ('layer1', 'bn2'),
          3: ('layer2', 'bn1'),
          4: ('layer2', 'bn2'),
          5: ('layer3', 'bn1'),
          6: ('layer3', 'bn2'),
      }

      layer, bn = mapping[idx]
      if is_downsample:
        key = f"{layer}.block_int16.downsample.2.{param_type}"
      else:
        key = f"{layer}.block_int16.{bn}.{param_type}"

      if key in model_dict:
        tensor = model_dict[key].cpu().numpy().astype(dtype)
        if tensor.shape != shape:
          raise ValueError(f"Shape mismatch for {name}: expected {shape}, got {tensor.shape}")
        return tensor
      else:
        raise ValueError(f"Key {key} not found in checkpoint for parameter {name}")

    # Pattern for quant_min_{1-6} and quant_max_{1-6}
    quant_pattern = re.match(r'quant_(min|max)_(\d+)(_2)?', name)
    if quant_pattern:
      param_type = quant_pattern.group(1)  # 'min' or 'max'
      idx = int(quant_pattern.group(2))
      is_downsample = quant_pattern.group(3) is not None

      # Map index to layer and act
      mapping = {
          1: ('layer1', 'act1'),
          2: ('layer1', 'act2'),
          3: ('layer2', 'act1'),
          4: ('layer2', 'act2'),
          5: ('layer3', 'act1'),
          6: ('layer3', 'act2'),
      }

      layer, act = mapping[idx]
      if is_downsample:
        key = f"{layer}.block_int16.downsample.0.{param_type}"
      else:
        key = f"{layer}.block_int16.{act}.{param_type}"

      if key in model_dict:
        tensor = model_dict[key].cpu().numpy()
        # Scalar values need to be converted to proper shape
        if shape == ():
          # Scalar
          return tensor.astype(dtype) if tensor.shape == () else np.array(tensor.item(), dtype=dtype)
        elif shape == (1,):
          # Single-element array
          return np.array([tensor.item()], dtype=dtype) if tensor.shape == () else tensor.astype(dtype)
        else:
          raise ValueError(f"Unexpected shape {shape} for scalar parameter {name}")
      else:
        raise ValueError(f"Key {key} not found in checkpoint for parameter {name}")

    # Pattern for y_f_{1,2,3}
    y_f_pattern = re.match(r'y_f_(\d+)', name)
    if y_f_pattern:
      idx = int(y_f_pattern.group(1))
      # y_f_1 = bn2_f_1 / x_f_1
      x_f_key = f'x_f_{idx}'
      bn2_f_key = f'bn2_f_{idx}'

      if x_f_key in adjust_factors and bn2_f_key in adjust_factors:
        value = adjust_factors[bn2_f_key] / adjust_factors[x_f_key]
        if dtype.startswith('int'):
          value = int(round(value))
        if shape == (1,):
          return np.array([value], dtype=dtype)
        else:
          return np.array(value, dtype=dtype)
      else:
        raise ValueError(f"Missing adjust_factors for computing {name}")

    # Pattern for bn_out_f_{0,1,2,3}
    bn_out_f_pattern = re.match(r'bn_out_f_(\d+)', name)
    if bn_out_f_pattern:
      idx = int(bn_out_f_pattern.group(1))
      if idx % 2 == 0:
        # bn_out_f_0, bn_out_f_2 (bias terms) -> zeros
        return np.zeros(shape, dtype=dtype)
      else:
        # bn_out_f_1, bn_out_f_3 (scale terms) -> ones
        return np.ones(shape, dtype=dtype)

    # If no pattern matched, raise an error
    raise ValueError(f"No mapping found for parameter: {name} with dtype={dtype}, shape={shape}")


  params_dict = {}
  # Sort by name for determinism
  for name in sorted(var_dict.keys()):
    if name == "model_input":
      continue
    info = var_dict[name]
    params_dict[name] = _get_tensor_from_checkpoint(name, info["dtype"], info["shape"])

  return out, params_dict
