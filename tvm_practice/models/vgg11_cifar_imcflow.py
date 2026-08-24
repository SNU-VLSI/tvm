"""VGG-11 (Verma JSSC'22 Table V topology) for CIFAR-10, built for the imcflow
BYOC codegen flow.

This mirrors the hand-written-relay pattern used by ``resnet8_cifar.py`` and
``ds_cnn_imcflow.py``: the network is constructed directly as a relay graph with
imcflow quantized ops (``imcflow_min_max_quantize`` -> ``imcflow_qconv2d`` ->
``imcflow_batch_norm``) so it drops straight into the same partition / PnR /
codegen pipeline that ``main.py`` drives via ``MODEL_REGISTRY``.

Topology (Verma JSSC'22 Table V, CIFAR-10 32x32x3, 4-bit weights/acts):
  conv3x3   3->128  @32x32 (pad 1)   + BN + ReLU
  conv3x3 128->128  @32x32 (pad 1)   + BN + ReLU
  conv3x3 128->128  @32x32 (pad 1)   + BN + ReLU
  conv3x3 128->128  @32x32 (pad 1)   + BN + ReLU
  maxpool 2x2  -> 16x16
  conv3x3 128->256  @16x16 (pad 1)   + BN + ReLU
  conv3x3 256->256  @16x16 (pad 1)   + BN + ReLU
  maxpool 2x2  -> 8x8
  conv3x3 256->256  @8x8   (pad 1)    + BN + ReLU
  conv3x3 256->256  @8x8   (pad 1)    + BN + ReLU
  flatten (8*8*256 = 16384) -> dense 1024 (ReLU) -> dense 1024 (ReLU) -> dense 10

Weights are randomly initialised (no imcflow-quantized checkpoint exists for
VGG-11). The random-init path mirrors ``resnet8_cifar.getModel``.

Registry entry (see codegen/test.py MODEL_REGISTRY):
    "vgg11_cifar_rnd": (lambda: vgg11_cifar_imcflow.getModel(), "ones")
"""

import numpy as np

import tvm
from tvm import relay

from tvm.relay.qnn.op.qnn import imcflow_min_max_quantize
from tvm.relay.op.nn import imcflow_batch_norm, imcflow_qconv2d

from tvm.relay.backend.contrib.imcflow.acim_util import ConfigData


def get_height(H, KH, padding, stride):
  return (H + 2 * padding - KH) // stride + 1


def get_width(W, KW, padding, stride):
  return (W + 2 * padding - KW) // stride + 1


# Monotonically increasing suffix so every conv/bn/quant var gets a unique name.
class _NameGen:
  def __init__(self):
    self.n = 0

  def next(self):
    self.n += 1
    return self.n


def _qconv_block(y, ng, N, IC, H, W, OC, KH=3, KW=3, pad=1, stride=1,
                 relu=True):
  """One (min_max_quantize -> imcflow_qconv2d -> imcflow_batch_norm [-> relu])
  block, exactly mirroring the per-conv pattern in resnet8_cifar.getModel_.

  Returns (y, OC, OH, OW) so the caller can keep shape bookkeeping like the
  reference model does.
  """
  idx = ng.next()
  y = imcflow_min_max_quantize(
      y,
      relay.var(f"quant_min_{idx}", shape=(), dtype="int16"),
      relay.var(f"quant_max_{idx}", shape=(), dtype="int16"),
      axis=1, out_dtype="uint8", channel=IC,
  )
  y = imcflow_qconv2d(
      y,
      relay.var(f"weight_{idx}", shape=(OC, IC, KH, KW), dtype="int8"),
      ConfigData((N, IC, H, W), (OC, IC, KH, KW), padding=pad,
                 stride=stride).get_as_const_tensor(),
      in_channels=IC,
      channels=OC,
      kernel_size=(KH, KW),
      padding=(pad, pad),
      strides=(stride, stride),
      out_dtype="int16",
  )
  OH = get_height(H, KH, pad, stride)
  OW = get_width(W, KW, pad, stride)
  y = imcflow_batch_norm(
      y,
      relay.var(f"fused_scale_{idx}", shape=(OC,), dtype="int16"),
      relay.var(f"fused_bias_{idx}", shape=(OC,), dtype="int16"),
  )
  if relu:
    # ReLU is applied in the int16 domain (same op the resnet8 post-process
    # uses); the imcflow relay->imce lowering supports nn.relu.
    y = relay.nn.relu(y)
  return y, OC, OH, OW


def getModel_(input_shape):
  """Build the VGG-11 relay graph. Returns (IRModule, var_info dict)."""
  input = relay.var("model_input", shape=input_shape, dtype="float32")
  N, IC, H, W = input_shape
  ng = _NameGen()

  # --- Front float conv1 3->128 + BN, then quantize into int16 domain. ---
  # Like resnet8, the very first conv sees the raw float input and is kept as a
  # plain float nn.conv2d + nn.batch_norm; it will run on CPU (the imcflow flow
  # only offloads the imcflow_qconv2d ops). Everything after the initial scale
  # is int.
  y = relay.nn.conv2d(
      input,
      relay.var("stem_weight", shape=(128, IC, 3, 3), dtype="float32"),
      in_channels=IC, channels=128, kernel_size=(3, 3), padding=(1, 1),
  )
  y = relay.nn.batch_norm(
      y,
      relay.var("stem_bn_gamma", shape=(128,), dtype="float32"),
      relay.var("stem_bn_beta", shape=(128,), dtype="float32"),
      relay.var("stem_bn_mean", shape=(128,), dtype="float32"),
      relay.var("stem_bn_var", shape=(128,), dtype="float32"),
  )[0]
  y = relay.nn.relu(y)
  y = y * relay.var("x_f_1", shape=(1,), dtype="float32")
  y = relay.clip(y, a_min=-32768.0, a_max=32767.0)
  y = relay.cast(y, dtype="int16")
  IC, H, W = 128, get_height(H, 3, 1, 1), get_width(W, 3, 1, 1)

  # --- VGG conv stack (all quantized imcflow convs). ---
  # block A: 3 more convs at 128ch @32x32 (conv1 above is the 4th 128ch conv).
  y, IC, H, W = _qconv_block(y, ng, N, IC, H, W, 128)
  y, IC, H, W = _qconv_block(y, ng, N, IC, H, W, 128)
  y, IC, H, W = _qconv_block(y, ng, N, IC, H, W, 128)
  # maxpool 2x2 -> 16x16 (pooling runs on CPU, like resnet8's avgpool).
  y = relay.cast(y, dtype="float32")
  y = relay.nn.max_pool2d(y, pool_size=(2, 2), strides=(2, 2))
  y = relay.cast(y, dtype="int16")
  H, W = H // 2, W // 2

  # block B: 128->256, 256->256 @16x16
  y, IC, H, W = _qconv_block(y, ng, N, IC, H, W, 256)
  y, IC, H, W = _qconv_block(y, ng, N, IC, H, W, 256)
  y = relay.cast(y, dtype="float32")
  y = relay.nn.max_pool2d(y, pool_size=(2, 2), strides=(2, 2))
  y = relay.cast(y, dtype="int16")
  H, W = H // 2, W // 2

  # block C: 256->256, 256->256 @8x8
  y, IC, H, W = _qconv_block(y, ng, N, IC, H, W, 256)
  y, IC, H, W = _qconv_block(y, ng, N, IC, H, W, 256)

  # --- Classifier: flatten (8*8*256=16384) -> 1024 -> 1024 -> 10, on CPU. ---
  y = relay.cast(y, dtype="float32") * relay.var("post_f_inv", shape=(1,),
                                                 dtype="float32")
  y = relay.nn.batch_flatten(y)                          # (N, 16384)
  y = relay.nn.dense(y, relay.var("fc1_weight", shape=(1024, IC * H * W),
                                  dtype="float32"))
  y = relay.nn.bias_add(y, relay.var("fc1_bias", shape=(1024,),
                                     dtype="float32"))
  y = relay.nn.relu(y)
  y = relay.nn.dense(y, relay.var("fc2_weight", shape=(1024, 1024),
                                  dtype="float32"))
  y = relay.nn.bias_add(y, relay.var("fc2_bias", shape=(1024,),
                                     dtype="float32"))
  y = relay.nn.relu(y)
  y = relay.nn.dense(y, relay.var("fc3_weight", shape=(10, 1024),
                                  dtype="float32"))
  y = relay.nn.bias_add(y, relay.var("fc3_bias", shape=(10,), dtype="float32"))

  # --- Collect param var info (mirrors resnet8_cifar.getModel_). ---
  var_info = {}
  for v in relay.analysis.free_vars(y):
    if v == input:
      continue
    name = v.name_hint
    if name in var_info:
      continue
    ttype = v.type_annotation
    if isinstance(ttype, relay.ty.TensorType):
      shape = []
      for dim in ttype.shape:
        try:
          shape.append(int(dim))
        except Exception:
          shape.append(dim)
      var_info[name] = {"shape": tuple(shape), "dtype": ttype.dtype}

  out = tvm.IRModule.from_expr(y)
  return out, var_info


def _rand_tensor(dtype, shape):
  """Random initialiser, copied from resnet8_cifar.getModel so int4/int8/int16
  ranges match the reference random-init path."""
  if dtype in ("float32", "float16", "float64"):
    return np.random.uniform(-1, 1, shape).astype(dtype)
  if dtype.startswith("int"):
    try:
      bits = int(dtype.replace("int", ""))
    except Exception:
      bits = 32
    if bits == 4:
      return np.random.randint(-8, 8, size=shape, dtype=np.int8)
    if bits == 8:
      return np.random.randint(-128, 128, size=shape, dtype=np.int8)
    if bits == 16:
      return np.random.randint(-32768, 32768, size=shape, dtype=np.int16)
    if bits == 32:
      return np.random.randint(-2**31, 2**31, size=shape, dtype=np.int32)
    return np.random.randint(-2**31, 2**31, size=shape, dtype=np.int32)
  if dtype.startswith("uint"):
    try:
      bits = int(dtype.replace("uint", ""))
    except Exception:
      bits = 32
    if bits == 8:
      return np.random.randint(0, 256, size=shape, dtype=np.uint8)
    if bits == 16:
      return np.random.randint(0, 2**16, size=shape, dtype=np.uint16)
    if bits == 32:
      return np.random.randint(0, 2**32, size=shape, dtype=np.uint32)
    return np.random.randint(0, 2**32, size=shape, dtype=np.uint32)
  return np.random.uniform(-1, 1, shape).astype("float32")


def getModel(small_debug=False):
  """Random-init VGG-11 for CIFAR-10. Returns (IRModule, params_dict).

  small_debug shrinks the spatial input to 8x8 for a fast partition smoke test
  (mirrors resnet8_cifar.getModel's small_debug switch).
  """
  if small_debug:
    out, var_dict = getModel_([1, 3, 8, 8])
  else:
    out, var_dict = getModel_([1, 3, 32, 32])

  params_dict = {}
  for name in sorted(var_dict.keys()):
    if name == "model_input":
      continue
    info = var_dict[name]
    params_dict[name] = _rand_tensor(info["dtype"], info["shape"])

  # Ensure quant_min <= quant_max (same fix-up resnet8_cifar.getModel does).
  min_max_pairs = {}
  for name, value in params_dict.items():
    if "quant_min" in name:
      base = name.replace("quant_min", "")
      min_max_pairs.setdefault(base, [None, None])[0] = value
    elif "quant_max" in name:
      base = name.replace("quant_max", "")
      min_max_pairs.setdefault(base, [None, None])[1] = value
  for base, (mn, mx) in min_max_pairs.items():
    if mn is not None and mx is not None and np.any(mn > mx):
      params_dict[f"quant_min{base}"], params_dict[f"quant_max{base}"] = mx, mn

  return out, params_dict
