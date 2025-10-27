"""
 @file   ds_cnn_imcflow.py
 @brief  DS-CNN (Depthwise Separable CNN) model for IMCFlow hardware target
 @author Converted from original ds_cnn.py
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
    Define the IMCFlow version of DS-CNN (td_cnn variant)
    First two conv2d layers (time-domain and regular) use CPU (float32),
    middle depthwise separable layers use IMCFlow ops,
    last dense layer uses CPU (float32)
    
    Args:
        input_shape: tuple (N, C, H, W) in NCHW format
                    For td_cnn: (N, 1, 16000, 1) - raw time-domain samples
    """
    input = relay.var("input", shape=input_shape, dtype="float32")
    N, IC, H, W = input_shape
    filters = 64
    
    # First layer - time-domain conv (CPU)
    # Input: (N, 1, 16000, 1), kernel: (512, 1), stride: (384, 1), padding: valid
    y = relay.nn.conv2d(
        input,
        relay.var("weight_td", shape=(filters, IC, 512, 1), dtype="float32"),
        in_channels=IC,
        channels=filters,
        kernel_size=(512, 1),
        strides=(384, 1),
        padding=(0, 0),  # 'valid' padding
    )
    # Output shape after conv: ((16000 - 512) / 384 + 1, 1) = (41, 1)
    N, IC, H, W = (N, filters, (H - 512) // 384 + 1, 1)
    
    y = relay.nn.batch_norm(y,
                           relay.var("bn_gamma_td", shape=(filters,), dtype="float32"),
                           relay.var("bn_beta_td", shape=(filters,), dtype="float32"),
                           relay.var("bn_moving_mean_td", shape=(filters,), dtype="float32"),
                           relay.var("bn_moving_var_td", shape=(filters,), dtype="float32"))[0]
    y = relay.nn.relu(y)
    
    # Reshape: (N, 64, 41, 1) -> (N, 1, 41, 64)
    # This is done to prepare for the next conv layer
    y = relay.reshape(y, newshape=(N, 1, H, filters))
    IC, H, W = 1, H, filters
    
    # Second layer - regular conv2d (CPU)
    # Input: (N, 1, 41, 64), kernel: (10, 4), stride: (2, 2), padding: same
    pad_h = 5  # 'same' padding for kernel 10, stride 2
    pad_w = 2  # 'same' padding for kernel 4, stride 2
    y = relay.nn.conv2d(
        y,
        relay.var("weight1", shape=(filters, IC, 10, 4), dtype="float32"),
        in_channels=IC,
        channels=filters,
        kernel_size=(10, 4),
        strides=(2, 2),
        padding=(pad_h, pad_w),
    )
    N, IC, H, W = (N, filters, get_height(H, 10, pad_h, 2), get_width(W, 4, pad_w, 2))
    
    y = relay.nn.batch_norm(y,
                           relay.var("bn_gamma1", shape=(filters,), dtype="float32"),
                           relay.var("bn_beta1", shape=(filters,), dtype="float32"),
                           relay.var("bn_moving_mean1", shape=(filters,), dtype="float32"),
                           relay.var("bn_moving_var1", shape=(filters,), dtype="float32"))[0]
    y = relay.nn.relu(y)
    
    # Convert to int16 for IMCFlow processing
    y = y * relay.var("scale_f1", shape=(1,), dtype="float32")
    y = relay.cast(y, dtype="int16")
    
    # First separable depthwise conv layer (IMCFlow)
    # Depthwise conv
    y = imcflow_min_max_quantize(y, relay.var("quant_min1", shape=(), dtype="int16"),
                                  relay.var("quant_max1", shape=(), dtype="int16"),
                                  axis=1, out_dtype="uint8", channel=filters)
    y = imcflow_qdwconv2d(
        y,
        relay.var("weight_dw1", shape=(filters, 1, 3, 3), dtype="int8"),
        ConfigData((N, IC, H, W), (filters, 1, 3, 3), padding=1, stride=1).get_as_const_tensor(),
        in_channels=filters,
        channels=filters,
        kernel_size=(3, 3),
        padding=(1, 1),
        groups=filters,  # depthwise
        out_dtype="int16"
    )
    IC, H, W = (filters, get_height(H, 3, 1, 1), get_width(W, 3, 1, 1))
    
    y = imcflow_batch_norm(y, relay.var("fused_scale1", shape=(filters,), dtype="int16"),
                           relay.var("fused_bias1", shape=(filters,), dtype="int16"))
    
    # Pointwise conv (1x1)
    y = imcflow_min_max_quantize(y, relay.var("quant_min2", shape=(), dtype="int16"),
                                  relay.var("quant_max2", shape=(), dtype="int16"),
                                  axis=1, out_dtype="uint8", channel=filters)
    y = imcflow_qconv2d(
        y,
        relay.var("weight_pw1", shape=(filters, filters, 1, 1), dtype="int8"),
        ConfigData((N, IC, H, W), (filters, filters, 1, 1), padding=0, stride=1).get_as_const_tensor(),
        in_channels=filters,
        channels=filters,
        kernel_size=(1, 1),
        out_dtype="int16"
    )
    
    y = imcflow_batch_norm(y, relay.var("fused_scale2", shape=(filters,), dtype="int16"),
                           relay.var("fused_bias2", shape=(filters,), dtype="int16"))
    
    # Second separable depthwise conv layer (IMCFlow)
    # Depthwise conv
    y = imcflow_min_max_quantize(y, relay.var("quant_min3", shape=(), dtype="int16"),
                                  relay.var("quant_max3", shape=(), dtype="int16"),
                                  axis=1, out_dtype="uint8", channel=filters)
    y = imcflow_qdwconv2d(
        y,
        relay.var("weight_dw2", shape=(filters, 1, 3, 3), dtype="int8"),
        ConfigData((N, IC, H, W), (filters, 1, 3, 3), padding=1, stride=1).get_as_const_tensor(),
        in_channels=filters,
        channels=filters,
        kernel_size=(3, 3),
        padding=(1, 1),
        groups=filters,
        out_dtype="int16"
    )
    
    y = imcflow_batch_norm(y, relay.var("fused_scale3", shape=(filters,), dtype="int16"),
                           relay.var("fused_bias3", shape=(filters,), dtype="int16"))
    
    # Pointwise conv (1x1)
    y = imcflow_min_max_quantize(y, relay.var("quant_min4", shape=(), dtype="int16"),
                                  relay.var("quant_max4", shape=(), dtype="int16"),
                                  axis=1, out_dtype="uint8", channel=filters)
    y = imcflow_qconv2d(
        y,
        relay.var("weight_pw2", shape=(filters, filters, 1, 1), dtype="int8"),
        ConfigData((N, IC, H, W), (filters, filters, 1, 1), padding=0, stride=1).get_as_const_tensor(),
        in_channels=filters,
        channels=filters,
        kernel_size=(1, 1),
        out_dtype="int16"
    )
    
    y = imcflow_batch_norm(y, relay.var("fused_scale4", shape=(filters,), dtype="int16"),
                           relay.var("fused_bias4", shape=(filters,), dtype="int16"))
    
    # Third separable depthwise conv layer (IMCFlow)
    # Depthwise conv
    y = imcflow_min_max_quantize(y, relay.var("quant_min5", shape=(), dtype="int16"),
                                  relay.var("quant_max5", shape=(), dtype="int16"),
                                  axis=1, out_dtype="uint8", channel=filters)
    y = imcflow_qdwconv2d(
        y,
        relay.var("weight_dw3", shape=(filters, 1, 3, 3), dtype="int8"),
        ConfigData((N, IC, H, W), (filters, 1, 3, 3), padding=1, stride=1).get_as_const_tensor(),
        in_channels=filters,
        channels=filters,
        kernel_size=(3, 3),
        padding=(1, 1),
        groups=filters,
        out_dtype="int16"
    )
    
    y = imcflow_batch_norm(y, relay.var("fused_scale5", shape=(filters,), dtype="int16"),
                           relay.var("fused_bias5", shape=(filters,), dtype="int16"))
    
    # Pointwise conv (1x1)
    y = imcflow_min_max_quantize(y, relay.var("quant_min6", shape=(), dtype="int16"),
                                  relay.var("quant_max6", shape=(), dtype="int16"),
                                  axis=1, out_dtype="uint8", channel=filters)
    y = imcflow_qconv2d(
        y,
        relay.var("weight_pw3", shape=(filters, filters, 1, 1), dtype="int8"),
        ConfigData((N, IC, H, W), (filters, filters, 1, 1), padding=0, stride=1).get_as_const_tensor(),
        in_channels=filters,
        channels=filters,
        kernel_size=(1, 1),
        out_dtype="int16"
    )
    
    y = imcflow_batch_norm(y, relay.var("fused_scale6", shape=(filters,), dtype="int16"),
                           relay.var("fused_bias6", shape=(filters,), dtype="int16"))
    
    # Fourth separable depthwise conv layer (IMCFlow)
    # Depthwise conv
    y = imcflow_min_max_quantize(y, relay.var("quant_min7", shape=(), dtype="int16"),
                                  relay.var("quant_max7", shape=(), dtype="int16"),
                                  axis=1, out_dtype="uint8", channel=filters)
    y = imcflow_qdwconv2d(
        y,
        relay.var("weight_dw4", shape=(filters, 1, 3, 3), dtype="int8"),
        ConfigData((N, IC, H, W), (filters, 1, 3, 3), padding=1, stride=1).get_as_const_tensor(),
        in_channels=filters,
        channels=filters,
        kernel_size=(3, 3),
        padding=(1, 1),
        groups=filters,
        out_dtype="int16"
    )
    
    y = imcflow_batch_norm(y, relay.var("fused_scale7", shape=(filters,), dtype="int16"),
                           relay.var("fused_bias7", shape=(filters,), dtype="int16"))
    
    # Pointwise conv (1x1)
    y = imcflow_min_max_quantize(y, relay.var("quant_min8", shape=(), dtype="int16"),
                                  relay.var("quant_max8", shape=(), dtype="int16"),
                                  axis=1, out_dtype="uint8", channel=filters)
    y = imcflow_qconv2d(
        y,
        relay.var("weight_pw4", shape=(filters, filters, 1, 1), dtype="int8"),
        ConfigData((N, IC, H, W), (filters, filters, 1, 1), padding=0, stride=1).get_as_const_tensor(),
        in_channels=filters,
        channels=filters,
        kernel_size=(1, 1),
        out_dtype="int16"
    )
    
    y = imcflow_batch_norm(y, relay.var("fused_scale8", shape=(filters,), dtype="int16"),
                           relay.var("fused_bias8", shape=(filters,), dtype="int16"))
    
    # Convert back to float32 for final layers (CPU)
    y = relay.cast(y, dtype="float32") * relay.var("post_f_inv", shape=(1,), dtype="float32")
    y = relay.nn.relu(y)
    
    # Average pooling
    final_pool_size = (int(H / 2), int(W / 2))
    y = relay.nn.avg_pool2d(y, pool_size=final_pool_size)
    
    # Flatten and Dense (CPU)
    y = relay.nn.batch_flatten(y)
    y = relay.nn.dense(y, relay.var("dense_weight", shape=(12, filters), dtype="float32"))
    y = relay.nn.bias_add(y, relay.var("dense_bias", shape=(12,), dtype="float32"))
    y = relay.nn.softmax(y)

    var_info = get_param_info_from_relay_func(y)
    out = tvm.IRModule.from_expr(y)

    return out, var_info


def getModel():
    """
    Create a test model for IMCFlow DS-CNN (td_cnn variant)
    Input shape: (1, 1, 16000, 1) - batch=1, channels=1, time_samples=16000, width=1
    For td_samples feature type with raw audio input
    """
    input_shape = (1, 1, 16000, 1)  # NCHW format for time-domain samples
    out, var_dict = getModel_(input_shape)
    params_dict = {}
    for name in sorted(var_dict.keys()):
      info = var_dict[name]
      params_dict[name] = _rand_tensor(info["dtype"], info["shape"])
    
    return out, params_dict
