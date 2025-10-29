"""
 @file   mobilenet_imcflow.py
 @brief  MobileNetV1 model for IMCFlow hardware target
 @author Converted from original mobilenet.py
"""

import numpy as np
import tvm
from tvm import relay
from tvm.relay.qnn.op.qnn import imcflow_min_max_quantize, imcflow_nu_quantize
from tvm.relay.op.nn import imcflow_batch_norm, imcflow_qconv2d, imcflow_qdwconv2d
from tvm.relay.backend.contrib.imcflow.acim_util import ConfigData
from .utils import get_param_info_from_relay_func, _rand_tensor


def get_height(H, KH, padding, stride):
    pad_h = padding
    out_h = (H + 2 * pad_h - KH) // stride + 1
    return out_h


def get_width(W, KW, padding, stride):
    pad_w = padding
    out_w = (W + 2 * pad_w - KW) // stride + 1
    return out_w


def getModel_(input_shape):
    """
    Define the IMCFlow version of MobileNetV1
    First conv2d uses CPU (float32), middle layers use IMCFlow ops (depthwise + pointwise),
    last dense layer uses CPU (float32)
    
    Args:
        input_shape: tuple (N, C, H, W) in NCHW format
                    e.g., (1, 3, 96, 96) for person detection
    """
    input = relay.var("model_input", shape=input_shape, dtype="float32")
    N, IC, H, W = input_shape
    num_filters = 8  # alpha=0.25 per EEMBC requirement (normally 32)
    
    # First layer - pure conv2d (CPU)
    y = relay.nn.conv2d(
        input,
        relay.var("weight1", shape=(num_filters, IC, 3, 3), dtype="float32"),
        in_channels=IC,
        channels=num_filters,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding=(1, 1),
    )
    N, IC, H, W = (N, num_filters, get_height(H, 3, 1, 2), get_width(W, 3, 1, 2))
    
    y = relay.nn.batch_norm(y,
                           relay.var("bn_gamma1", shape=(num_filters,), dtype="float32"),
                           relay.var("bn_beta1", shape=(num_filters,), dtype="float32"),
                           relay.var("bn_moving_mean1", shape=(num_filters,), dtype="float32"),
                           relay.var("bn_moving_var1", shape=(num_filters,), dtype="float32"))[0]
    y = relay.nn.relu(y)
    
    # Convert to int16 for IMCFlow processing
    y = y * relay.var("scale_f1", shape=(1,), dtype="float32")
    y = relay.cast(y, dtype="int16")
    
    layer_idx = 2
    quant_idx = 1
    
    # Helper function for depthwise separable conv block
    def add_depthwise_separable_block(y, N, IC, H, W, in_filters, out_filters, stride, 
                                     layer_idx, quant_idx):
        # Depthwise conv
        y = imcflow_min_max_quantize(y, 
                                      relay.var(f"quant_min{quant_idx}", shape=(), dtype="int16"),
                                      relay.var(f"quant_max{quant_idx}", shape=(), dtype="int16"),
                                      axis=1, out_dtype="uint8", channel=in_filters)
        
        y = imcflow_qdwconv2d(
            y,
            relay.var(f"weight_dw{layer_idx}", shape=(in_filters, 1, 3, 3), dtype="int8"),
            ConfigData((N, IC, H, W), (in_filters, 1, 3, 3), padding=1, stride=stride).get_as_const_tensor(),
            in_channels=in_filters,
            channels=in_filters,
            kernel_size=(3, 3),
            strides=(stride, stride),
            padding=(1, 1),
            groups=in_filters,
            out_dtype="int16"
        )
        IC, H, W = (in_filters, get_height(H, 3, 1, stride), get_width(W, 3, 1, stride))
        
        y = imcflow_batch_norm(y, 
                               relay.var(f"fused_scale{layer_idx}_dw", shape=(in_filters,), dtype="int16"),
                               relay.var(f"fused_bias{layer_idx}_dw", shape=(in_filters,), dtype="int16"))
        
        quant_idx += 1
        
        # Pointwise conv (1x1)
        y = imcflow_min_max_quantize(y,
                                      relay.var(f"quant_min{quant_idx}", shape=(), dtype="int16"),
                                      relay.var(f"quant_max{quant_idx}", shape=(), dtype="int16"),
                                      axis=1, out_dtype="uint8", channel=in_filters)
        
        y = imcflow_qconv2d(
            y,
            relay.var(f"weight_pw{layer_idx}", shape=(out_filters, in_filters, 1, 1), dtype="int8"),
            ConfigData((N, IC, H, W), (out_filters, in_filters, 1, 1), padding=0, stride=1).get_as_const_tensor(),
            in_channels=in_filters,
            channels=out_filters,
            kernel_size=(1, 1),
            out_dtype="int16"
        )
        IC = out_filters
        
        y = imcflow_batch_norm(y,
                               relay.var(f"fused_scale{layer_idx}_pw", shape=(out_filters,), dtype="int16"),
                               relay.var(f"fused_bias{layer_idx}_pw", shape=(out_filters,), dtype="int16"))
        
        quant_idx += 1
        return y, IC, H, W, quant_idx
    
    # 2nd layer: depthwise separable conv
    y, IC, H, W, quant_idx = add_depthwise_separable_block(
        y, N, IC, H, W, num_filters, num_filters * 2, 1, layer_idx, quant_idx)
    num_filters *= 2
    layer_idx += 1
    
    # 3rd layer: depthwise separable conv with stride 2
    y, IC, H, W, quant_idx = add_depthwise_separable_block(
        y, N, IC, H, W, num_filters, num_filters * 2, 2, layer_idx, quant_idx)
    num_filters *= 2
    layer_idx += 1
    
    # 4th layer: depthwise separable conv
    y, IC, H, W, quant_idx = add_depthwise_separable_block(
        y, N, IC, H, W, num_filters, num_filters, 1, layer_idx, quant_idx)
    layer_idx += 1
    
    # 5th layer: depthwise separable conv with stride 2
    y, IC, H, W, quant_idx = add_depthwise_separable_block(
        y, N, IC, H, W, num_filters, num_filters * 2, 2, layer_idx, quant_idx)
    num_filters *= 2
    layer_idx += 1
    
    # 6th layer: depthwise separable conv
    y, IC, H, W, quant_idx = add_depthwise_separable_block(
        y, N, IC, H, W, num_filters, num_filters, 1, layer_idx, quant_idx)
    layer_idx += 1
    
    # 7th layer: depthwise separable conv with stride 2
    y, IC, H, W, quant_idx = add_depthwise_separable_block(
        y, N, IC, H, W, num_filters, num_filters * 2, 2, layer_idx, quant_idx)
    num_filters *= 2
    layer_idx += 1
    
    # 8th-12th layers: identical depthwise separable convs (5 layers)
    for i in range(5):
        y, IC, H, W, quant_idx = add_depthwise_separable_block(
            y, N, IC, H, W, num_filters, num_filters, 1, layer_idx, quant_idx)
        layer_idx += 1
    
    # 13th layer: depthwise separable conv with stride 2
    y, IC, H, W, quant_idx = add_depthwise_separable_block(
        y, N, IC, H, W, num_filters, num_filters * 2, 2, layer_idx, quant_idx)
    num_filters *= 2
    layer_idx += 1
    
    # 14th layer: depthwise separable conv
    y, IC, H, W, quant_idx = add_depthwise_separable_block(
        y, N, IC, H, W, num_filters, num_filters, 1, layer_idx, quant_idx)
    
    # Convert back to float32 for final layers (CPU)
    y = relay.cast(y, dtype="float32") * relay.var("post_f_inv", shape=(1,), dtype="float32")
    
    # Average pooling - use the actual spatial dimensions
    y = relay.nn.avg_pool2d(y, pool_size=(H, W))
    
    # Flatten and Dense (CPU)
    y = relay.nn.batch_flatten(y)
    y = relay.nn.dense(y, relay.var("dense_weight", shape=(2, num_filters), dtype="float32"))
    y = relay.nn.bias_add(y, relay.var("dense_bias", shape=(2,), dtype="float32"))
    y = relay.nn.softmax(y)

    var_info = get_param_info_from_relay_func(y)
    out = tvm.IRModule.from_expr(y)
    return out, var_info


def getModel(small_debug=False):
    """
    Create a test model for IMCFlow MobileNetV1
    Input shape: (1, 3, 96, 96) - batch=1, RGB channels=3, H=96, W=96
    """
    if small_debug:
      input_shape = (1, 3, 32, 32)  # NCHW format
    else:
      input_shape = (1, 3, 96, 96)  # NCHW format
    out, var_dict = getModel_(input_shape)
    params_dict={}
    for name in sorted(var_dict.keys()):
      info = var_dict[name]
      params_dict[name] = _rand_tensor(info["dtype"], info["shape"])
    
    return out, params_dict