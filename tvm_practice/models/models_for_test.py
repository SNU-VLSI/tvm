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

def rand_tensor(dtype: str, shape):
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

def one_tensor(dtype: str, shape):
  # Handle common dtypes with appropriate ranges
  if dtype in ("float32", "float16", "float64"):
    return np.ones(shape).astype(dtype)
  if dtype.startswith("int"):
    # Parse bit width if available (e.g., int4, int8, int16, int32)
    try:
      bits = int(dtype.replace("int", ""))
    except Exception:
      bits = 32
    if bits == 4:
      # No native int4 in numpy; store in int8 within valid int4 range
      return np.ones(shape, dtype=np.int8)
    if bits == 8:
      return np.ones(shape, dtype=np.int8)
    if bits == 16:
      return np.ones(shape, dtype=np.int16)
    if bits == 32:
      return np.ones(shape, dtype=np.int32)
    if bits == 64:
      return np.ones(shape, dtype=np.int64)
    # Fallback: use int32
    return np.ones(shape, dtype=np.int32)
  if dtype.startswith("uint"):
    try:
      bits = int(dtype.replace("uint", ""))
    except Exception:
      bits = 32
    if bits == 4:
      return np.ones(shape, dtype=np.uint8)
    if bits == 8:
      return np.ones(shape, dtype=np.uint8)
    if bits == 16:
      return np.ones(shape, dtype=np.uint16)
    if bits == 32:
      return np.ones(shape, dtype=np.uint32)
    if bits == 64:
      # numpy uint64 with ones
      return np.ones(shape, dtype=np.uint64)
    return np.ones(shape, dtype=np.uint32)
  # Default float32 if unrecognized
  return np.ones(shape).astype("float32")

def getOneReluModel():
  N, C, H, W = 1, 28, 4, 4
  input_ = relay.var("input", shape=(N,C,H,W), dtype="int16")
  y = relay.nn.relu(input_)

  param_dict = { }

  out = tvm.IRModule.from_expr(y)

  return out, param_dict

def getOneConvModel():
  N, IC, H, W = 1, 28, 4, 4
  OC = 64
  KH, KW = 3, 3
  stride, padding = 1, 1
  # input = relay.var("conv_input", shape=(N,math.ceil(IC/256),H,W,4,8), dtype="int32")
  input = relay.var("conv_input", shape=(N,IC,H,W), dtype="uint8")

  y = imcflow_qconv2d(
    input,
    relay.var("conv_weight", shape=(OC,IC,KH,KW), dtype="int8"),
    ConfigData((N, IC, H, W), (OC, IC, KH, KW), padding=padding, stride=stride).get_as_const_tensor(),
    in_channels=IC,
    channels=OC,
    kernel_size=(KH, KW),
    padding=(padding, padding),
    out_dtype="int16"
  )

  # param_dict = {"conv_weight": np.ones((OC,IC,KH,KW), dtype="int8")}
  param_dict = {
    "conv_weight": np.random.randint(-8, 8, size=(OC,IC,KH,KW), dtype=np.int8),
    # "conv_weight": np.random.randint(-1, 0, size=(OC,IC,KH,KW), dtype=np.int8),
  }

  out = tvm.IRModule.from_expr(y)

  return out, param_dict

def getOneMMQuantModel():
  N, C, H, W = 1, 28, 4, 4
  input = relay.var("input", shape=(N,C,H,W), dtype="int16")
  y = imcflow_min_max_quantize(
    input, 
    relay.var("quant_min", shape=(), dtype="int16"), 
    relay.var("quant_max", shape=(), dtype="int16"), 
    axis=1, out_dtype="uint8", channel=16)
  
  # param_dict = {
  #   "quant_min": np.array(-128, dtype="int16"),
  #   "quant_max": np.array(127, dtype="int16"),
  # }
  param_dict = {
    "quant_min": np.array(-2**15, dtype="int16"),
    "quant_max": np.array(2**15-1, dtype="int16"),
  }

  out = tvm.IRModule.from_expr(y)
  return out, param_dict

def getOneConvQuantModel():
  N, IC, H, W = 1, 28, 4, 4
  OC = 64
  KH, KW = 3, 3
  stride, padding = 1, 1
  input = relay.var("conv_input", shape=(N,IC,H,W), dtype="uint8")

  y = imcflow_qconv2d(
    input,
    relay.var("conv_weight", shape=(OC,IC,KH,KW), dtype="int8"),
    ConfigData((N, IC, H, W), (OC, IC, KH, KW), padding=padding, stride=stride).get_as_const_tensor(),
    in_channels=IC,
    channels=OC,
    kernel_size=(KH, KW),
    padding=(padding, padding),
    out_dtype="int16"
  )
  y = imcflow_min_max_quantize(
    y, 
    relay.var("quant_min", shape=(), dtype="int16"), 
    relay.var("quant_max", shape=(), dtype="int16"), 
    axis=1, out_dtype="uint8", channel=16)

  param_dict = {
    "conv_weight": np.random.randint(-8, 8, size=(OC,IC,KH,KW), dtype=np.int8),
    "quant_min": np.array(-128, dtype="int16"),
    "quant_max": np.array(127, dtype="int16"),
    }

  out = tvm.IRModule.from_expr(y)

  return out, param_dict

def getOneFusedBNModel():
  N, C, H, W = 1, 16, 4, 4
  input = relay.var("input", shape=(N,C,H,W), dtype="int16")
  y = imcflow_batch_norm(
    input,
    relay.var("fused_scale", shape=(C,), dtype="int16"),
    relay.var("fused_bias", shape=(C,), dtype="int16"),
  )

  param_dict = {
    "fused_scale": np.ones((C,), dtype="int16"),
    "fused_bias" : np.zeros((C,), dtype="int16"),
  }

  out = tvm.IRModule.from_expr(y)
  return out, param_dict

def getConvBNQuantModel():
  N, IC, H, W = 1, 16, 8, 8
  OC = 32
  KH, KW = 3, 3
  stride, padding = 2, 1
  input = relay.var("conv_input", shape=(N,IC,H,W), dtype="uint8")

  y = imcflow_qconv2d(
    input,
    relay.var("conv_weight", shape=(OC,IC,KH,KW), dtype="int8"),
    ConfigData((N, IC, H, W), (OC, IC, KH, KW), padding=padding, stride=stride).get_as_const_tensor(),
    in_channels=IC,
    channels=OC,
    kernel_size=(KH, KW),
    strides=(stride, stride),
    padding=(padding, padding),
    out_dtype="int16"
  )

  y = imcflow_batch_norm(
    y,
    relay.var("fused_scale", shape=(OC,), dtype="int16"),
    relay.var("fused_bias", shape=(OC,), dtype="int16"),
  )

  y = imcflow_min_max_quantize(y, relay.var("quant_min", shape=(), dtype="int16"), relay.var("quant_max", shape=(), dtype="int16"), axis=1, out_dtype="uint8", channel=16)

  param_dict = {
    "conv_weight": np.random.randint(-8, 7, size=(OC,IC,KH,KW), dtype=np.int8),
    "fused_scale": np.ones((OC,), dtype="int16"),
    "fused_bias" : np.zeros((OC,), dtype="int16"),
    "quant_min": np.array(-128, dtype="int16"),
    "quant_max": np.array(127, dtype="int16"),
  }
  out = tvm.IRModule.from_expr(y)
  return out, param_dict

def getConvBNMultAddModel():
  N, IC, H, W = 1, 16, 8, 8
  OC = 32
  KH, KW = 3, 3
  stride, padding = 2, 1
  input = relay.var("conv_input", shape=(N,IC,H,W), dtype="uint8")

  y = imcflow_qconv2d(
    input,
    relay.var("conv_weight", shape=(OC,IC,KH,KW), dtype="int8"),
    ConfigData((N, IC, H, W), (OC, IC, KH, KW), padding=padding, stride=stride).get_as_const_tensor(),
    in_channels=IC,
    channels=OC,
    kernel_size=(KH, KW),
    strides=(stride, stride),
    padding=(padding, padding),
    out_dtype="int16"
  )

  y = imcflow_batch_norm(
    y,
    relay.var("fused_scale", shape=(OC,), dtype="int16"),
    relay.var("fused_bias", shape=(OC,), dtype="int16"),
  )

  y = y * relay.var("multiplier", shape=(OC,1,1), dtype="int16")

  y = y + relay.var("adder", shape=(OC,1,1), dtype="int16")

  param_dict = {
    "conv_weight": np.random.randint(-8, 7, size=(OC,IC,KH,KW), dtype=np.int8),
    "fused_scale": np.ones((OC,), dtype="int16"),
    "fused_bias" : np.zeros((OC,), dtype="int16"),
    "multiplier": np.random.randint(-8, 7, size=(OC,1,1), dtype="int16"),
    "adder": np.random.randint(-8, 7, size=(OC,1,1), dtype="int16"),
  }
  out = tvm.IRModule.from_expr(y)
  return out, param_dict

def getConvQuantConvModel():
  N, IC, H, W = 1, 16, 1, 1
  y = relay.var("input", shape=(N,IC,H,W), dtype="uint8")
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

  y = imcflow_min_max_quantize(y, relay.var("quant_min_2", shape=(), dtype="int16"), relay.var("quant_max_2", shape=(), dtype="int16"), axis=1, out_dtype="uint8", channel=16)
  y = imcflow_qconv2d(
    y,
    relay.var("weight2_2", shape=(16,16,3,3), dtype="int8"),
    ConfigData((N, IC, H, W), (16,16,3,3), padding=1, stride=1, acc_mask=AccMask.BM_1111).get_as_const_tensor(),
    in_channels=16,
    channels=16,
    kernel_size=(3, 3),
    padding=(1, 1),
    out_dtype="int16"
  )
  IC, H, W = (16, get_height(H, 3, 1, 1), get_width(W, 3, 1, 1))

  param_dict = {
    "weight2_1"  : np.random.randint(-8, 7, (16,16,3,3), dtype="int8"),
    "weight2_2"  : np.random.randint(-8, 7, (16,16,3,3), dtype="int8"),
    "quant_min_2": np.array(-256, dtype="int16"),
    "quant_max_2": np.array(256, dtype="int16"),
  }

  out = tvm.IRModule.from_expr(y)
  return out, param_dict

def getResidualModel(random=False):
  N, IC, H, W = 1, 16, 1, 1
  y = relay.var("input", shape=(N,IC,H,W), dtype="uint8")
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
  residual = y
  IC, H, W = (16, get_height(H, 3, 1, 1), get_width(W, 3, 1, 1))

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

  y = y + residual

  if random == False:
    param_dict = {
      "weight2_1"  : np.ones((16,16,3,3), dtype="int8"),
      "weight2_2"  : np.ones((16,16,3,3), dtype="int8"),
      "quant_min_2": np.array(-128, dtype="int16"),
      "quant_max_2": np.array(127, dtype="int16"),
    }
  else:
    param_dict = {
      "weight2_1"  : np.random.randint(-8, 7, (16,16,3,3), dtype="int8"),
      "weight2_2"  : np.random.randint(-8, 7, (16,16,3,3), dtype="int8"),
      "quant_min_2": np.array(-256, dtype="int16"),
      "quant_max_2": np.array(256, dtype="int16"),
    }

  out = tvm.IRModule.from_expr(y)
  return out, param_dict

def getResnetCifar10Small_(input_shape):
  input = relay.var("model_input", shape=input_shape, dtype="float32")
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

  """
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
  """

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


def getResnetCifar10Small(small_debug=False, init_only_one=False):
  if small_debug:
    out, var_dict = getResnetCifar10Small_([1, 3, 8, 8])
  else:
    out, var_dict = getResnetCifar10Small_([1, 3, 32, 32])

  params_dict = {}
  # Sort by name for determinism
  for name in sorted(var_dict.keys()):
    info = var_dict[name]
    if init_only_one:
      params_dict[name] = one_tensor(info["dtype"], info["shape"])
    else:
      params_dict[name] = rand_tensor(info["dtype"], info["shape"])
  
  # swap min, max pair if min is greater than max
  min_max_pairs = {}
  for name, value in params_dict.items():
    if "quant_min" in name:
      base_name = name.replace("quant_min", "")
      if base_name not in min_max_pairs:
        min_max_pairs[base_name] = [value, None]
      else:
        min_max_pairs[base_name][0] = value
    elif "quant_max" in name:
      base_name = name.replace("quant_max", "")
      if base_name not in min_max_pairs:
        min_max_pairs[base_name] = [None, value]
      else:
        min_max_pairs[base_name][1] = value
    
  for base_name, (min_val, max_val) in min_max_pairs.items():
    if min_val is not None and max_val is not None:
      if np.any(min_val > max_val):
        # swap
        params_dict[f"quant_min{base_name}"], params_dict[f"quant_max{base_name}"] = max_val, min_val

  return out, params_dict

def getResnetCifar10SmallPretrained(small_debug=False):
  import torch
  import re
  
  if small_debug:
    out, var_dict = getResnetCifar10Small_([1, 3, 8, 8])
  else:
    out, var_dict = getResnetCifar10Small_([1, 3, 32, 32])
  
  # Load checkpoint
  checkpoint_path = '/root/project/tvm/tvm_practice/models/checkpoint.pth.tar' # with int16 conversion, CIM/tree/deploy/models_checkpoint/A4W4%2BPS6/2025-Sep-24-01-20-40/imcflow/2025-Oct-28-17-49-32
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
      # fused_scale1/bias1 -> layer1.bn1
      # fused_scale2/bias2 -> layer1.bn2
      # fused_scale3/bias3 -> layer2.bn1
      # fused_scale4/bias4 -> layer2.bn2
      # fused_scale5/bias5 -> layer3.bn1
      # fused_scale6/bias6 -> layer3.bn2
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
      # quant_min_1/max_1 -> layer1.act1
      # quant_min_2/max_2 -> layer1.act2
      # quant_min_3/max_3 -> layer2.act1
      # quant_min_4/max_4 -> layer2.act2
      # quant_min_5/max_5 -> layer3.act1
      # quant_min_6/max_6 -> layer3.act2
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
      # Based on the adjust_factors structure
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
      # These are used for downsample residual adjustment
      # Based on the code in deploy_modules.py line 129-130:
      # y_residual = bn_out_f_1 * y_residual + bn_out_f_0
      # This suggests bn_out_f_0 should be 0 and bn_out_f_1 should be 1 (or appropriate scaling)
      # However, looking at the model definition in resnet8_cifar.py, these are used as:
      # y_residual = relay.var("bn_out_f_1", shape=(32,1,1), dtype="int16") * y_residual + relay.var("bn_out_f_0", shape=(32,1,1), dtype="int16")
      
      # For proper implementation, we need to compute the adjustment between main path and downsample path
      # For now, using zeros for bn_out_f_0 (bias term) and computing scale from adjust_factors
      if idx % 2 == 0:
        # bn_out_f_0, bn_out_f_2 (bias terms) -> zeros
        return np.zeros(shape, dtype=dtype)
      else:
        # bn_out_f_1, bn_out_f_3 (scale terms)
        # Need to compute the ratio between downsample path and main path output scales
        # For simplicity, using ones (identity scaling)
        # A more accurate implementation would compute: downsample_output_scale / main_path_output_scale
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
  
def getOneConvBnModel():
  """
  
  """
  N, IC, H, W = 1, 28, 1, 1
  OC = 64
  KH, KW = 3, 3
  stride, padding = 1, 1

  input = relay.var("input", shape=(N,IC,H,W), dtype="uint8")

  y = imcflow_qconv2d(
    input,
    relay.var("conv_weight", shape=(OC,IC,KH,KW), dtype="int8"),
    ConfigData((N, IC, H, W), (OC, IC, KH, KW), padding=padding, stride=stride).get_as_const_tensor(),
    in_channels=IC,
    channels=OC,
    kernel_size=(KH, KW),
    padding=(padding, padding),
    out_dtype="int16"
  )

  y = imcflow_batch_norm(
    y,
    relay.var("fused_scale", shape=(OC,), dtype="int16"),
    relay.var("fused_bias", shape=(OC,), dtype="int16"),
    axis=1,
    in_channels=OC
  )

  param_dict = {
    "conv_weight": rand_tensor("int8", (OC,IC,KH,KW)),
    "quant_min" : np.array(-128, dtype="int16"),
    "quant_max" : np.array(127, dtype="int16"),
    "fused_scale": one_tensor("int16", (OC,)),
    "fused_bias" : rand_tensor("int16", (OC,))
  }

  out = tvm.IRModule.from_expr(y)

  return out, param_dict

def getBigConvModel():
  N, IC, H, W = 1, 64, 4, 4
  OC = 64
  KH, KW = 3, 3
  stride, padding = 1, 1
  # input = relay.var("conv_input", shape=(N,math.ceil(IC/256),H,W,4,8), dtype="int32")
  input = relay.var("conv_input", shape=(N,IC,H,W), dtype="uint8")

  y = imcflow_qconv2d(
    input,
    relay.var("conv_weight", shape=(OC,IC,KH,KW), dtype="int8"),
    ConfigData((N, IC, H, W), (OC, IC, KH, KW), padding=padding, stride=stride).get_as_const_tensor(),
    in_channels=IC,
    channels=OC,
    kernel_size=(KH, KW),
    padding=(padding, padding),
    out_dtype="int16"
  )

  param_dict = {
    # "conv_weight": np.random.randint(-8, 8, size=(OC,IC,KH,KW), dtype=np.int8),
    "conv_weight": np.ones((OC,IC,KH,KW), dtype="int8"),
  }

  out = tvm.IRModule.from_expr(y)

  return out, param_dict


def getResnetCifar10SmallManualParam_(input_shape):
  input = relay.var("model_input", shape=input_shape, dtype="float32")
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

  """
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
  """
  out = tvm.IRModule.from_expr(y)

  params_dict = {
    "weight1": np.ones((16, 3, 3, 3), dtype="float32"),
    "bn_gamma": np.ones((16,), dtype="float32"),
    "bn_beta": np.ones((16,), dtype="float32"),
    "bn_moving_mean": np.zeros((16,), dtype="float32"),
    "bn_moving_var": np.ones((16,), dtype="float32"),
    "quant_min_1": np.array(-64, dtype="int16"),
    "quant_max_1": np.array(64, dtype="int16"),
    "x_f_1": np.array([2.0], dtype="float32"),
    "weight2_1" : np.ones((16,16,3,3), dtype="int8"),
  }

  params_dict['weight1'][0,0,0,1] = 2
  params_dict['weight1'][0,0,1,0] = 2
  params_dict['weight1'][0,1,0,0] = 2
  params_dict['weight1'][1,0,0,0] = 2

  params_dict['weight2_1'][0,0,0,1] = 2
  params_dict['weight2_1'][0,0,1,0] = 2
  params_dict['weight2_1'][0,1,0,0] = 2
  params_dict['weight2_1'][1,0,0,0] = 2

  return out, params_dict

def getResnetCifar10SmallManualParam(small_debug=False):
  if small_debug:
    out, param_dict = getResnetCifar10SmallManualParam_([1, 3, 8, 8])
  else:
    out, param_dict = getResnetCifar10SmallManualParam_([1, 3, 32, 32])

  return out, param_dict