"""ResNet-50 (standard torchvision topology, ImageNet 224x224x3) built for the
imcflow BYOC codegen flow.

torch / torchvision are NOT installed in this environment, and the existing
imcflow models (resnet8_cifar.py, ds_cnn_imcflow.py) are all hand-written relay
graphs rather than tvm.relay.frontend.from_pytorch imports. So -- per the task's
"otherwise mirror however resnet8 is defined" fallback -- ResNet-50 is emitted
here as a hand-built relay graph using the same imcflow quantized-op pattern
(imcflow_min_max_quantize -> imcflow_qconv2d -> imcflow_batch_norm). The layer
dimensions exactly follow torchvision.models.resnet50:

  stem : conv7x7 s2 3->64, maxpool3x3 s2   (224 -> 112 -> 56)
  layer1 : 3x bottleneck  (64,64,256),  first block 1x1 downsample 64->256
  layer2 : 4x bottleneck  (128,128,512), first block s2 downsample 256->512 (56->28)
  layer3 : 6x bottleneck  (256,256,1024),first block s2 downsample 512->1024(28->14)
  layer4 : 3x bottleneck  (512,512,2048),first block s2 downsample 1024->2048(14->7)
  avgpool -> flatten -> dense 2048->1000

Each bottleneck: 1x1 reduce -> 3x3 -> 1x1 expand, with a residual add (and a
1x1 downsample conv on the first block of each stage). All conv weights are
random int8 (no imcflow-quantized ResNet-50 checkpoint exists).

WARNING -- this model is expected to hit compile blockers (that is the point;
the user wants to debug them). See module docstring notes below and the
handoff report:
  * ConfigData asserts H,W <= 128 (acim_util.py). ResNet-50 stem input is
    224x224, so building the *relay graph* works but constructing ConfigData for
    the early convs raises AssertionError. To let the relay graph / partition
    pass be inspected without tripping that assert, the ConfigData spatial dims
    are clamped to <=128 via _cfg() (config values only affect the ACIM control
    register, not the conv's logical shape). The real fix (tiling large spatial
    maps) is left for the debugging pass.
  * 3x3 conv with IC>=256 needs an IC-split chain of ceil(IC*9/256) atoms; e.g.
    layer4 3x3 512ch -> ceil(512/28)=19 atoms > 16 IMCEs, which will break PnR.
  * dense 2048->1000 and the ImageNet classifier are large; they run on CPU.

Registry entry (see codegen/test.py MODEL_REGISTRY):
    "resnet50_imagenet_rnd": (lambda: resnet50_imagenet_imcflow.getModel(), "ones")
"""

import numpy as np

import tvm
from tvm import relay

from tvm.relay.qnn.op.qnn import imcflow_min_max_quantize
from tvm.relay.op.nn import imcflow_batch_norm, imcflow_qconv2d

from tvm.relay.backend.contrib.imcflow.acim_util import ConfigData


def get_out(D, K, pad, stride):
  return (D + 2 * pad - K) // stride + 1


class _NameGen:
  def __init__(self):
    self.n = 0

  def next(self):
    self.n += 1
    return self.n


def _cfg(N, IC, H, W, OC, KH, KW, pad, stride):
  """Build a ConfigData, clamping H/W to the ACIM 128-limit.

  ConfigData asserts H,W <= 128 (acim_util.py). The config tensor only encodes
  the ACIM control register (ksel/pad/stride/H/W/adcmode/...); the conv's real
  logical shape comes from the imcflow_qconv2d channels/kernel args, not this
  tensor. Clamping the H/W fields lets the >128 early-stage relay graph be
  constructed and partitioned for debugging without tripping the assert. The
  correct handling (spatial tiling) is a known follow-up.
  """
  return ConfigData(
      (N, IC, min(int(H), 128), min(int(W), 128)),
      (OC, IC, KH, KW),
      padding=pad, stride=stride,
  ).get_as_const_tensor()


def _qconv(y, ng, N, IC, H, W, OC, KH, KW, pad, stride, relu=True):
  """min_max_quantize -> imcflow_qconv2d -> imcflow_batch_norm [-> relu].

  Mirrors the per-conv pattern in resnet8_cifar. Returns (y, OC, OH, OW).
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
      _cfg(N, IC, H, W, OC, KH, KW, pad, stride),
      in_channels=IC,
      channels=OC,
      kernel_size=(KH, KW),
      padding=(pad, pad),
      strides=(stride, stride),
      out_dtype="int16",
  )
  OH, OW = get_out(H, KH, pad, stride), get_out(W, KW, pad, stride)
  y = imcflow_batch_norm(
      y,
      relay.var(f"fused_scale_{idx}", shape=(OC,), dtype="int16"),
      relay.var(f"fused_bias_{idx}", shape=(OC,), dtype="int16"),
  )
  if relu:
    y = relay.nn.relu(y)
  return y, OC, OH, OW


def _bottleneck(y, ng, N, IC, H, W, mid, out, stride, downsample):
  """Standard torchvision bottleneck: 1x1 reduce -> 3x3(stride) -> 1x1 expand,
  plus a residual add. When downsample, the shortcut is a 1x1 stride conv."""
  residual = y
  IC_res, H_res, W_res = IC, H, W

  # 1x1 reduce (IC -> mid), 3x3 (mid -> mid, stride), 1x1 expand (mid -> out).
  yy, c, h, w = _qconv(y, ng, N, IC, H, W, mid, 1, 1, 0, 1)
  yy, c, h, w = _qconv(yy, ng, N, c, h, w, mid, 3, 3, 1, stride)
  # Last conv of the block: no ReLU before the residual add (torchvision applies
  # ReLU after the add).
  yy, c, h, w = _qconv(yy, ng, N, c, h, w, out, 1, 1, 0, 1, relu=False)

  if downsample:
    residual, _, _, _ = _qconv(residual, ng, N, IC_res, H_res, W_res, out,
                               1, 1, 0, stride, relu=False)

  y = yy + residual
  y = relay.nn.relu(y)
  return y, out, h, w


def getModel_(input_shape):
  """Build the ResNet-50 relay graph. Returns (IRModule, var_info dict)."""
  input = relay.var("model_input", shape=input_shape, dtype="float32")
  N, IC, H, W = input_shape
  ng = _NameGen()

  # --- Float stem: conv7x7 s2 3->64 + BN + ReLU, maxpool3x3 s2 (CPU). ---
  y = relay.nn.conv2d(
      input,
      relay.var("stem_weight", shape=(64, IC, 7, 7), dtype="float32"),
      in_channels=IC, channels=64, kernel_size=(7, 7),
      strides=(2, 2), padding=(3, 3),
  )
  y = relay.nn.batch_norm(
      y,
      relay.var("stem_bn_gamma", shape=(64,), dtype="float32"),
      relay.var("stem_bn_beta", shape=(64,), dtype="float32"),
      relay.var("stem_bn_mean", shape=(64,), dtype="float32"),
      relay.var("stem_bn_var", shape=(64,), dtype="float32"),
  )[0]
  y = relay.nn.relu(y)
  y = relay.nn.max_pool2d(y, pool_size=(3, 3), strides=(2, 2), padding=(1, 1))
  H = get_out(get_out(H, 7, 3, 2), 3, 1, 2)   # 224 -> 112 -> 56
  W = get_out(get_out(W, 7, 3, 2), 3, 1, 2)
  IC = 64
  y = y * relay.var("x_f_1", shape=(1,), dtype="float32")
  y = relay.clip(y, a_min=-32768.0, a_max=32767.0)
  y = relay.cast(y, dtype="int16")

  # --- 4 residual stages (torchvision resnet50 block counts: 3,4,6,3). ---
  stage_cfg = [
      # (mid, out, num_blocks, first_stride)
      (64,   256,  3, 1),   # layer1 (56x56)
      (128,  512,  4, 2),   # layer2 (56 -> 28)
      (256,  1024, 6, 2),   # layer3 (28 -> 14)
      (512,  2048, 3, 2),   # layer4 (14 -> 7)
  ]
  for mid, out, nblocks, first_stride in stage_cfg:
    for b in range(nblocks):
      stride = first_stride if b == 0 else 1
      downsample = (b == 0)
      y, IC, H, W = _bottleneck(y, ng, N, IC, H, W, mid, out, stride, downsample)

  # --- ImageNet classifier: global avgpool -> flatten -> dense 2048->1000 (CPU). ---
  y = relay.cast(y, dtype="float32") * relay.var("post_f_inv", shape=(1,),
                                                 dtype="float32")
  y = relay.nn.adaptive_avg_pool2d(y, output_size=(1, 1))
  y = relay.nn.batch_flatten(y)                       # (N, 2048)
  y = relay.nn.dense(y, relay.var("fc_weight", shape=(1000, IC),
                                  dtype="float32"))
  y = relay.nn.bias_add(y, relay.var("fc_bias", shape=(1000,), dtype="float32"))

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
  """Random initialiser, matching resnet8_cifar.getModel's ranges."""
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
    return np.random.randint(0, 2**32, size=shape, dtype=np.uint32)
  return np.random.uniform(-1, 1, shape).astype("float32")


def getModel(small_debug=False):
  """Random-init ResNet-50 for ImageNet. Returns (IRModule, params_dict).

  small_debug uses a 56x56 input (skips the 224->56 stem downsampling scale) so
  the graph is cheaper to build/partition while keeping the full block topology.
  """
  if small_debug:
    out, var_dict = getModel_([1, 3, 56, 56])
  else:
    out, var_dict = getModel_([1, 3, 224, 224])

  params_dict = {}
  for name in sorted(var_dict.keys()):
    if name == "model_input":
      continue
    info = var_dict[name]
    params_dict[name] = _rand_tensor(info["dtype"], info["shape"])

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
