"""
 @file   deep_autoencoder_imcflow.py
 @brief  Deep autoencoder model for IMCFlow hardware target
 @author Converted from original deep_autoencoder.py
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

import numpy as np
import tvm
from tvm import relay
from tvm.relay.qnn.op.qnn import imcflow_min_max_quantize, imcflow_nu_quantize
from tvm.relay.op.nn import imcflow_batch_norm, imcflow_qconv2d
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
    Define the IMCFlow version of deep autoencoder
    First dense layer uses CPU (float32), middle layers use IMCFlow ops as 1x1 conv,
    last dense layer uses CPU (float32)
    Structure: 128*128*128*128*8*128*128*128*128
    
    Dense layers are represented as 1x1 convolutions by reshaping input to (N, C, 1, 1)
    """
    input = relay.var("model_input", shape=input_shape, dtype="float32")
    N, inputDim = input_shape
    
    # First Dense layer (CPU) - keep as float32
    y = relay.nn.dense(input, relay.var("weight1", shape=(128, inputDim), dtype="float32"))
    y = relay.nn.batch_norm(y,
                           relay.var("bn_gamma1", shape=(128,), dtype="float32"),
                           relay.var("bn_beta1", shape=(128,), dtype="float32"),
                           relay.var("bn_moving_mean1", shape=(128,), dtype="float32"),
                           relay.var("bn_moving_var1", shape=(128,), dtype="float32"))[0]
    y = relay.nn.relu(y)
    
    # Convert to int16 for IMCFlow processing
    y = y * relay.var("scale_f1", shape=(1,), dtype="float32")
    y = relay.cast(y, dtype="int16")
    
    # Reshape to (N, 128, 1, 1) for 1x1 conv representation
    y = relay.reshape(y, newshape=(N, 128, 1, 1))
    IC, H, W = 128, 1, 1
    
    # Second Dense layer (IMCFlow) - as 1x1 conv
    y = imcflow_min_max_quantize(y, relay.var("quant_min1", shape=(), dtype="int16"),
                                  relay.var("quant_max1", shape=(), dtype="int16"),
                                  axis=1, out_dtype="uint8", channel=128)
    y = imcflow_qconv2d(
        y,
        relay.var("weight2", shape=(128, 128, 1, 1), dtype="int8"),
        ConfigData((N, IC, H, W), (128, 128, 1, 1), padding=0, stride=1).get_as_const_tensor(),
        in_channels=128,
        channels=128,
        kernel_size=(1, 1),
        out_dtype="int16"
    )
    IC = 128
    y = imcflow_batch_norm(y, relay.var("fused_scale1", shape=(128,), dtype="int16"),
                           relay.var("fused_bias1", shape=(128,), dtype="int16"))
    
    # Third Dense layer (IMCFlow) - as 1x1 conv
    y = imcflow_min_max_quantize(y, relay.var("quant_min2", shape=(), dtype="int16"),
                                  relay.var("quant_max2", shape=(), dtype="int16"),
                                  axis=1, out_dtype="uint8", channel=128)
    y = imcflow_qconv2d(
        y,
        relay.var("weight3", shape=(128, 128, 1, 1), dtype="int8"),
        ConfigData((N, IC, H, W), (128, 128, 1, 1), padding=0, stride=1).get_as_const_tensor(),
        in_channels=128,
        channels=128,
        kernel_size=(1, 1),
        out_dtype="int16"
    )
    IC = 128
    y = imcflow_batch_norm(y, relay.var("fused_scale2", shape=(128,), dtype="int16"),
                           relay.var("fused_bias2", shape=(128,), dtype="int16"))
    
    # Fourth Dense layer (IMCFlow) - as 1x1 conv
    y = imcflow_min_max_quantize(y, relay.var("quant_min3", shape=(), dtype="int16"),
                                  relay.var("quant_max3", shape=(), dtype="int16"),
                                  axis=1, out_dtype="uint8", channel=128)
    y = imcflow_qconv2d(
        y,
        relay.var("weight4", shape=(128, 128, 1, 1), dtype="int8"),
        ConfigData((N, IC, H, W), (128, 128, 1, 1), padding=0, stride=1).get_as_const_tensor(),
        in_channels=128,
        channels=128,
        kernel_size=(1, 1),
        out_dtype="int16"
    )
    IC = 128
    y = imcflow_batch_norm(y, relay.var("fused_scale3", shape=(128,), dtype="int16"),
                           relay.var("fused_bias3", shape=(128,), dtype="int16"))
    
    # Fifth Dense layer (IMCFlow) - bottleneck to 8 features as 1x1 conv
    y = imcflow_min_max_quantize(y, relay.var("quant_min4", shape=(), dtype="int16"),
                                  relay.var("quant_max4", shape=(), dtype="int16"),
                                  axis=1, out_dtype="uint8", channel=128)
    y = imcflow_qconv2d(
        y,
        relay.var("weight5", shape=(8, 128, 1, 1), dtype="int8"),
        ConfigData((N, IC, H, W), (8, 128, 1, 1), padding=0, stride=1).get_as_const_tensor(),
        in_channels=128,
        channels=8,
        kernel_size=(1, 1),
        out_dtype="int16"
    )
    IC = 8
    y = imcflow_batch_norm(y, relay.var("fused_scale4", shape=(8,), dtype="int16"),
                           relay.var("fused_bias4", shape=(8,), dtype="int16"))
    
    # Sixth Dense layer (IMCFlow) - expand from 8 to 128 as 1x1 conv
    y = imcflow_min_max_quantize(y, relay.var("quant_min5", shape=(), dtype="int16"),
                                  relay.var("quant_max5", shape=(), dtype="int16"),
                                  axis=1, out_dtype="uint8", channel=8)
    y = imcflow_qconv2d(
        y,
        relay.var("weight6", shape=(128, 8, 1, 1), dtype="int8"),
        ConfigData((N, IC, H, W), (128, 8, 1, 1), padding=0, stride=1).get_as_const_tensor(),
        in_channels=8,
        channels=128,
        kernel_size=(1, 1),
        out_dtype="int16"
    )
    IC = 128
    y = imcflow_batch_norm(y, relay.var("fused_scale5", shape=(128,), dtype="int16"),
                           relay.var("fused_bias5", shape=(128,), dtype="int16"))
    
    # Seventh Dense layer (IMCFlow) - as 1x1 conv
    y = imcflow_min_max_quantize(y, relay.var("quant_min6", shape=(), dtype="int16"),
                                  relay.var("quant_max6", shape=(), dtype="int16"),
                                  axis=1, out_dtype="uint8", channel=128)
    y = imcflow_qconv2d(
        y,
        relay.var("weight7", shape=(128, 128, 1, 1), dtype="int8"),
        ConfigData((N, IC, H, W), (128, 128, 1, 1), padding=0, stride=1).get_as_const_tensor(),
        in_channels=128,
        channels=128,
        kernel_size=(1, 1),
        out_dtype="int16"
    )
    IC = 128
    y = imcflow_batch_norm(y, relay.var("fused_scale6", shape=(128,), dtype="int16"),
                           relay.var("fused_bias6", shape=(128,), dtype="int16"))
    
    # Eighth Dense layer (IMCFlow) - as 1x1 conv
    y = imcflow_min_max_quantize(y, relay.var("quant_min7", shape=(), dtype="int16"),
                                  relay.var("quant_max7", shape=(), dtype="int16"),
                                  axis=1, out_dtype="uint8", channel=128)
    y = imcflow_qconv2d(
        y,
        relay.var("weight8", shape=(128, 128, 1, 1), dtype="int8"),
        ConfigData((N, IC, H, W), (128, 128, 1, 1), padding=0, stride=1).get_as_const_tensor(),
        in_channels=128,
        channels=128,
        kernel_size=(1, 1),
        out_dtype="int16"
    )
    IC = 128
    y = imcflow_batch_norm(y, relay.var("fused_scale7", shape=(128,), dtype="int16"),
                           relay.var("fused_bias7", shape=(128,), dtype="int16"))
    
    # Convert back to float32 for final layer (CPU)
    y = relay.cast(y, dtype="float32") * relay.var("post_f_inv", shape=(1,), dtype="float32")
    
    # Reshape back to (N, 128) for final dense layer
    y = relay.reshape(y, newshape=(N, 128))
    
    # Final Dense layer (CPU) - output reconstruction
    y = relay.nn.dense(y, relay.var("dense_weight_final", shape=(inputDim, 128), dtype="float32"))

    var_info = get_param_info_from_relay_func(y)
    out = tvm.IRModule.from_expr(y)

    return out, var_info


def getModel(small_debug=False):
    """
    Create a test model for IMCFlow deep autoencoder
    """
    if small_debug:
      input_shape = (1, 640)  # batch_size=1, inputDim=640
    else:
      input_shape = (1, 640)  # batch_size=1, inputDim=640
    out, var_dict = getModel_(input_shape)
    params_dict={}
    for name in sorted(var_dict.keys()):
      info = var_dict[name]
      params_dict[name] = _rand_tensor(info["dtype"], info["shape"])
    
    return out, params_dict
