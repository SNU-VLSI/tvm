"""
DS-CNN Subset Models for IMCFlow hardware target with pretrained weight loading.

This module provides:
1. DS-CNN model definition in TVM Relay IR for IMCFlow
2. Pretrained weight loading from checkpoint
3. Subset functionality (early stopping at specific relay operation)
"""

import numpy as np

import tvm
from tvm import relay

from tvm.relay.backend.contrib.imcflow.acim_util import ConfigData
from tvm.relay.qnn.op.qnn import imcflow_min_max_quantize
from tvm.relay.op.nn import imcflow_batch_norm, imcflow_qconv2d, imcflow_qdwconv2d


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


def getModel_(input_shape, until_relay: int = None, replicate_factor: int = 1):
    """
    Get DS-CNN model with optional early stopping at specific relay operation.

    Args:
        input_shape: Input tensor shape (N, C, H, W) - e.g., (1, 1, 49, 10)
        until_relay: Optional index to stop at (0-based). Stops after the Nth 'y = ...' assignment.
                     If None, returns full model.
        replicate_factor: Replication factor for IMCFlow operations (default: 1)

    Returns:
        (IRModule, var_info): TVM IRModule and dictionary of parameter info
    """
    input = relay.var("model_input", shape=input_shape, dtype="float32")
    N, IC, H, W = input_shape
    filters = 64

    # Create counter that will automatically check and throw exception if needed
    c = RelayOpCounter(until_relay, input)

    try:
        # ============== Stem (FP32 - CPU) ==============
        # Conv2d: (N, 1, H, W) -> (N, 64, H', W')
        # kernel=(10,4), stride=(2,2), padding=(4,1)
        y = c.check(relay.nn.conv2d(
            input,
            relay.var("stem_weight", shape=(filters, IC, 10, 4), dtype="float32"),
            in_channels=IC,
            channels=filters,
            kernel_size=(10, 4),
            strides=(2, 2),
            padding=(4, 1),
        ))

        N, IC, H, W = (N, filters, get_height(H, 10, 4, 2), get_width(W, 4, 1, 2))

        y = c.check(relay.nn.batch_norm(
            y,
            relay.var("stem_bn_gamma", shape=(filters,), dtype="float32"),
            relay.var("stem_bn_beta", shape=(filters,), dtype="float32"),
            relay.var("stem_bn_mean", shape=(filters,), dtype="float32"),
            relay.var("stem_bn_var", shape=(filters,), dtype="float32")
        )[0])

        # Convert to int16 for IMCFlow processing
        y = c.check(y * relay.var("x_f_1", shape=(1,), dtype="float32"))
        y = relay.clip(y, a_min=-32768.0, a_max=32767.0)
        y = c.check(relay.cast(y, dtype="int16"))

        # ============== Block 1 ==============
        # Depthwise path
        y = c.check(imcflow_min_max_quantize(
            y,
            relay.var("quant_dw_min_1", shape=(), dtype="int16"),
            relay.var("quant_dw_max_1", shape=(), dtype="int16"),
            axis=1, out_dtype="uint8", channel=filters,
            replicate_factor=replicate_factor
        ))
        y = c.check(imcflow_qdwconv2d(
            y,
            relay.var("weight_dw_1", shape=(filters, 1, 3, 3), dtype="int8"),
            ConfigData((N, IC, H, W), (filters, 1, 3, 3), padding=1, stride=1).get_as_const_tensor(),
            in_channels=filters,
            channels=filters,
            kernel_size=(3, 3),
            padding=(1, 1),
            groups=filters,
            out_dtype="int16"
        ))
        IC, H, W = (filters, get_height(H, 3, 1, 1), get_width(W, 3, 1, 1))

        y = c.check(imcflow_batch_norm(
            y,
            relay.var("fused_scale_dw_1", shape=(filters,), dtype="int16"),
            relay.var("fused_bias_dw_1", shape=(filters,), dtype="int16")
        ))

        # Pointwise path
        y = c.check(imcflow_min_max_quantize(
            y,
            relay.var("quant_pw_min_1", shape=(), dtype="int16"),
            relay.var("quant_pw_max_1", shape=(), dtype="int16"),
            axis=1, out_dtype="uint8", channel=filters,
            replicate_factor=replicate_factor
        ))
        y = c.check(imcflow_qconv2d(
            y,
            relay.var("weight_pw_1", shape=(filters, filters, 1, 1), dtype="int8"),
            ConfigData((N, IC, H, W), (filters, filters, 1, 1), padding=0, stride=1).get_as_const_tensor(),
            in_channels=filters,
            channels=filters,
            kernel_size=(1, 1),
            out_dtype="int16",
            replicate_factor=replicate_factor
        ))

        y = c.check(imcflow_batch_norm(
            y,
            relay.var("fused_scale_pw_1", shape=(filters,), dtype="int16"),
            relay.var("fused_bias_pw_1", shape=(filters,), dtype="int16")
        ))

        # ============== Block 2 ==============
        # Depthwise path
        y = c.check(imcflow_min_max_quantize(
            y,
            relay.var("quant_dw_min_2", shape=(), dtype="int16"),
            relay.var("quant_dw_max_2", shape=(), dtype="int16"),
            axis=1, out_dtype="uint8", channel=filters,
            replicate_factor=replicate_factor
        ))
        y = c.check(imcflow_qdwconv2d(
            y,
            relay.var("weight_dw_2", shape=(filters, 1, 3, 3), dtype="int8"),
            ConfigData((N, IC, H, W), (filters, 1, 3, 3), padding=1, stride=1).get_as_const_tensor(),
            in_channels=filters,
            channels=filters,
            kernel_size=(3, 3),
            padding=(1, 1),
            groups=filters,
            out_dtype="int16"
        ))
        IC, H, W = (filters, get_height(H, 3, 1, 1), get_width(W, 3, 1, 1))

        y = c.check(imcflow_batch_norm(
            y,
            relay.var("fused_scale_dw_2", shape=(filters,), dtype="int16"),
            relay.var("fused_bias_dw_2", shape=(filters,), dtype="int16")
        ))

        # Pointwise path
        y = c.check(imcflow_min_max_quantize(
            y,
            relay.var("quant_pw_min_2", shape=(), dtype="int16"),
            relay.var("quant_pw_max_2", shape=(), dtype="int16"),
            axis=1, out_dtype="uint8", channel=filters,
            replicate_factor=replicate_factor
        ))
        y = c.check(imcflow_qconv2d(
            y,
            relay.var("weight_pw_2", shape=(filters, filters, 1, 1), dtype="int8"),
            ConfigData((N, IC, H, W), (filters, filters, 1, 1), padding=0, stride=1).get_as_const_tensor(),
            in_channels=filters,
            channels=filters,
            kernel_size=(1, 1),
            out_dtype="int16",
            replicate_factor=replicate_factor
        ))

        y = c.check(imcflow_batch_norm(
            y,
            relay.var("fused_scale_pw_2", shape=(filters,), dtype="int16"),
            relay.var("fused_bias_pw_2", shape=(filters,), dtype="int16")
        ))

        # ============== Block 3 ==============
        # Depthwise path
        y = c.check(imcflow_min_max_quantize(
            y,
            relay.var("quant_dw_min_3", shape=(), dtype="int16"),
            relay.var("quant_dw_max_3", shape=(), dtype="int16"),
            axis=1, out_dtype="uint8", channel=filters,
            replicate_factor=replicate_factor
        ))
        y = c.check(imcflow_qdwconv2d(
            y,
            relay.var("weight_dw_3", shape=(filters, 1, 3, 3), dtype="int8"),
            ConfigData((N, IC, H, W), (filters, 1, 3, 3), padding=1, stride=1).get_as_const_tensor(),
            in_channels=filters,
            channels=filters,
            kernel_size=(3, 3),
            padding=(1, 1),
            groups=filters,
            out_dtype="int16"
        ))
        IC, H, W = (filters, get_height(H, 3, 1, 1), get_width(W, 3, 1, 1))

        y = c.check(imcflow_batch_norm(
            y,
            relay.var("fused_scale_dw_3", shape=(filters,), dtype="int16"),
            relay.var("fused_bias_dw_3", shape=(filters,), dtype="int16")
        ))

        # Pointwise path
        y = c.check(imcflow_min_max_quantize(
            y,
            relay.var("quant_pw_min_3", shape=(), dtype="int16"),
            relay.var("quant_pw_max_3", shape=(), dtype="int16"),
            axis=1, out_dtype="uint8", channel=filters,
            replicate_factor=replicate_factor
        ))
        y = c.check(imcflow_qconv2d(
            y,
            relay.var("weight_pw_3", shape=(filters, filters, 1, 1), dtype="int8"),
            ConfigData((N, IC, H, W), (filters, filters, 1, 1), padding=0, stride=1).get_as_const_tensor(),
            in_channels=filters,
            channels=filters,
            kernel_size=(1, 1),
            out_dtype="int16",
            replicate_factor=replicate_factor
        ))

        y = c.check(imcflow_batch_norm(
            y,
            relay.var("fused_scale_pw_3", shape=(filters,), dtype="int16"),
            relay.var("fused_bias_pw_3", shape=(filters,), dtype="int16")
        ))

        # ============== Block 4 ==============
        # Depthwise path
        y = c.check(imcflow_min_max_quantize(
            y,
            relay.var("quant_dw_min_4", shape=(), dtype="int16"),
            relay.var("quant_dw_max_4", shape=(), dtype="int16"),
            axis=1, out_dtype="uint8", channel=filters,
            replicate_factor=replicate_factor
        ))
        y = c.check(imcflow_qdwconv2d(
            y,
            relay.var("weight_dw_4", shape=(filters, 1, 3, 3), dtype="int8"),
            ConfigData((N, IC, H, W), (filters, 1, 3, 3), padding=1, stride=1).get_as_const_tensor(),
            in_channels=filters,
            channels=filters,
            kernel_size=(3, 3),
            padding=(1, 1),
            groups=filters,
            out_dtype="int16"
        ))
        IC, H, W = (filters, get_height(H, 3, 1, 1), get_width(W, 3, 1, 1))

        y = c.check(imcflow_batch_norm(
            y,
            relay.var("fused_scale_dw_4", shape=(filters,), dtype="int16"),
            relay.var("fused_bias_dw_4", shape=(filters,), dtype="int16")
        ))

        # Pointwise path
        y = c.check(imcflow_min_max_quantize(
            y,
            relay.var("quant_pw_min_4", shape=(), dtype="int16"),
            relay.var("quant_pw_max_4", shape=(), dtype="int16"),
            axis=1, out_dtype="uint8", channel=filters,
            replicate_factor=replicate_factor
        ))
        y = c.check(imcflow_qconv2d(
            y,
            relay.var("weight_pw_4", shape=(filters, filters, 1, 1), dtype="int8"),
            ConfigData((N, IC, H, W), (filters, filters, 1, 1), padding=0, stride=1).get_as_const_tensor(),
            in_channels=filters,
            channels=filters,
            kernel_size=(1, 1),
            out_dtype="int16",
            replicate_factor=replicate_factor
        ))

        y = c.check(imcflow_batch_norm(
            y,
            relay.var("fused_scale_pw_4", shape=(filters,), dtype="int16"),
            relay.var("fused_bias_pw_4", shape=(filters,), dtype="int16")
        ))

        # ============== Post-process (FP32 - CPU) ==============
        y = c.check(relay.cast(y, dtype="float32") * relay.var("post_f_inv", shape=(1,), dtype="float32"))
        y = c.check(relay.nn.relu(y))
        y = c.check(relay.nn.adaptive_avg_pool2d(y, output_size=(1, 1)))
        y = c.check(relay.nn.batch_flatten(y))
        y = c.check(relay.nn.dense(y, relay.var("dense_weight", shape=(12, filters), dtype="float32")))
        y = c.check(relay.nn.bias_add(y, relay.var("dense_bias", shape=(12,), dtype="float32")))

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


def getModel_from_pretrained_weight(iH=10, iW=10, until_relay=None, replicate_factor=1):
    """
    Get DS-CNN subset model with pretrained weights loaded from checkpoint.

    Args:
        iH: Input height (default 10 for resized MFCC features)
        iW: Input width (default 10 for resized time frames)
        until_relay: Optional relay operation index to stop at (0-based). If None, returns full model.
        replicate_factor: Replication factor for IMCFlow operations (default: 1)

    Returns:
        (out, params_dict): IRModule and dictionary of parameter tensors
    """
    import torch
    import re

    out, var_dict = getModel_([1, 1, iH, iW], until_relay=until_relay, replicate_factor=replicate_factor)

    # Load checkpoint
    checkpoint_path = '/root/project/CIM/trained_models/keyword_spotting/NAT/prange_full_psum_duplication_1/2025-Dec-06-16-45-17/imcflow/2026-Jan-30-19-49-06/checkpoint.pth.tar'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
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
        # Direct mappings for stem and fc layers
        direct_mappings = {
            'stem_weight': 'stem_conv.weight',
            'stem_bn_gamma': 'stem_bn.weight',
            'stem_bn_beta': 'stem_bn.bias',
            'stem_bn_mean': 'stem_bn.running_mean',
            'stem_bn_var': 'stem_bn.running_var',
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
            # post_f_inv = 1.0 / bn_pw_f_4
            return np.array([1.0 / adjust_factors['bn_pw_f_4']], dtype=dtype)

        # Handle block-specific parameters using regex patterns

        # Pattern for weight_dw_{1-4}
        weight_dw_pattern = re.match(r'weight_dw_(\d+)', name)
        if weight_dw_pattern:
            block_idx = int(weight_dw_pattern.group(1))
            key = f"block{block_idx}.block_int16.dw.weight"
            if key in model_dict:
                tensor = model_dict[key].cpu().numpy().astype(dtype)
                if tensor.shape != shape:
                    raise ValueError(f"Shape mismatch for {name}: expected {shape}, got {tensor.shape}")
                return tensor
            else:
                raise ValueError(f"Key {key} not found in checkpoint for parameter {name}")

        # Pattern for weight_pw_{1-4}
        weight_pw_pattern = re.match(r'weight_pw_(\d+)', name)
        if weight_pw_pattern:
            block_idx = int(weight_pw_pattern.group(1))
            key = f"block{block_idx}.block_int16.pw.weight"
            if key in model_dict:
                tensor = model_dict[key].cpu().numpy().astype(dtype)
                if tensor.shape != shape:
                    raise ValueError(f"Shape mismatch for {name}: expected {shape}, got {tensor.shape}")
                return tensor
            else:
                raise ValueError(f"Key {key} not found in checkpoint for parameter {name}")

        # Pattern for fused_scale_dw_{1-4} and fused_bias_dw_{1-4}
        fused_dw_pattern = re.match(r'fused_(scale|bias)_dw_(\d+)', name)
        if fused_dw_pattern:
            param_type = fused_dw_pattern.group(1)  # 'scale' or 'bias'
            block_idx = int(fused_dw_pattern.group(2))
            key = f"block{block_idx}.block_int16.bn_dw.{param_type}"
            if key in model_dict:
                tensor = model_dict[key].cpu().numpy().astype(dtype)
                if tensor.shape != shape:
                    raise ValueError(f"Shape mismatch for {name}: expected {shape}, got {tensor.shape}")
                return tensor
            else:
                raise ValueError(f"Key {key} not found in checkpoint for parameter {name}")

        # Pattern for fused_scale_pw_{1-4} and fused_bias_pw_{1-4}
        fused_pw_pattern = re.match(r'fused_(scale|bias)_pw_(\d+)', name)
        if fused_pw_pattern:
            param_type = fused_pw_pattern.group(1)  # 'scale' or 'bias'
            block_idx = int(fused_pw_pattern.group(2))
            key = f"block{block_idx}.block_int16.bn_pw.{param_type}"
            if key in model_dict:
                tensor = model_dict[key].cpu().numpy().astype(dtype)
                if tensor.shape != shape:
                    raise ValueError(f"Shape mismatch for {name}: expected {shape}, got {tensor.shape}")
                return tensor
            else:
                raise ValueError(f"Key {key} not found in checkpoint for parameter {name}")

        # Pattern for quant_dw_min_{1-4} and quant_dw_max_{1-4}
        quant_dw_pattern = re.match(r'quant_dw_(min|max)_(\d+)', name)
        if quant_dw_pattern:
            param_type = quant_dw_pattern.group(1)  # 'min' or 'max'
            block_idx = int(quant_dw_pattern.group(2))
            key = f"block{block_idx}.block_int16.act_dw.{param_type}"
            if key in model_dict:
                tensor = model_dict[key].cpu().numpy()
                # Scalar values need to be converted to proper shape
                if shape == ():
                    return tensor.astype(dtype) if tensor.shape == () else np.array(tensor.item(), dtype=dtype)
                elif shape == (1,):
                    return np.array([tensor.item()], dtype=dtype) if tensor.shape == () else tensor.astype(dtype)
                else:
                    raise ValueError(f"Unexpected shape {shape} for scalar parameter {name}")
            else:
                raise ValueError(f"Key {key} not found in checkpoint for parameter {name}")

        # Pattern for quant_pw_min_{1-4} and quant_pw_max_{1-4}
        quant_pw_pattern = re.match(r'quant_pw_(min|max)_(\d+)', name)
        if quant_pw_pattern:
            param_type = quant_pw_pattern.group(1)  # 'min' or 'max'
            block_idx = int(quant_pw_pattern.group(2))
            key = f"block{block_idx}.block_int16.act_pw.{param_type}"
            if key in model_dict:
                tensor = model_dict[key].cpu().numpy()
                # Scalar values need to be converted to proper shape
                if shape == ():
                    return tensor.astype(dtype) if tensor.shape == () else np.array(tensor.item(), dtype=dtype)
                elif shape == (1,):
                    return np.array([tensor.item()], dtype=dtype) if tensor.shape == () else tensor.astype(dtype)
                else:
                    raise ValueError(f"Unexpected shape {shape} for scalar parameter {name}")
            else:
                raise ValueError(f"Key {key} not found in checkpoint for parameter {name}")

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
