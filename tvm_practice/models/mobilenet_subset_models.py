"""
VWW MobileNetV1 Subset Models for IMCFlow hardware target with pretrained weights.

Visual Wake Words MobileNetV1 (alpha=0.25, 96x96, 2-class), 13 depthwise-separable
blocks. Only the pointwise 1x1 convs run on the IMC array (imcflow_qconv2d); the
depthwise convs run off-array (imcflow_qdwconv2d, use_imcu=0). Stem conv+bn and the
final dense stay FP off-chip. This mirrors ds_cnn_subset_models.py (4-block DS-CNN)
generalized to 13 blocks with per-block IC!=OC and per-block depthwise stride, plus
TF-'same' padding (asymmetric for stride-2 layers) to match the deploy model.

Provides:
1. MobileNetV1 VWW definition in TVM Relay IR for IMCFlow
2. Pretrained weight loading from an imcflow-format checkpoint (via CKPT registry)
3. Subset functionality (early stopping at a specific relay operation)
"""

import json as _json
import os

import numpy as np

import tvm
from tvm import relay

from tvm.relay.backend.contrib.imcflow.acim_util import ConfigData
from tvm.relay.qnn.op.qnn import imcflow_min_max_quantize
from tvm.relay.op.nn import imcflow_batch_norm, imcflow_qconv2d, imcflow_qdwconv2d

_last_checkpoint_path = None
_last_checkpoint_alias = None

# VWW (MobileNetV1) depthwise-separable block configuration.
# (in_channels=dw_ic=pw_ic, out_channels=pw_oc, dw_stride). alpha=0.25.
# Spatial after stem (96->48). Matches deploy VWW_BLOCK_CONFIG.
VWW_BLOCK_CONFIG = [
    # (IC,  OC,  dw_stride)
    (8,    16,  1),   # block1
    (16,   32,  2),   # block2
    (32,   32,  1),   # block3
    (32,   64,  2),   # block4
    (64,   64,  1),   # block5
    (64,   128, 2),   # block6
    (128,  128, 1),   # block7
    (128,  128, 1),   # block8
    (128,  128, 1),   # block9
    (128,  128, 1),   # block10
    (128,  128, 1),   # block11
    (128,  256, 2),   # block12
    (256,  256, 1),   # block13
]

STEM_OUT = 8
NUM_CLASS = 2


# VWW checkpoint registry: loaded from CIM/checkpoints/b2_half_vww.json (single
# source of truth). Mirrors b2_half_kws.json but for visual_wake_words checkpoints.
# Falls back to CIM_DIR env var, then to /root/project/CIM. BOARD/vmode do not
# affect VWW checkpoint selection.

def _load_vww_checkpoints():
    cim_dir = os.environ.get("CIM_DIR", "/root/project/CIM")
    registry_path = os.path.join(cim_dir, "checkpoints", "b2_half_vww.json")
    if not os.path.isfile(registry_path):
        # Registry not created yet (no VWW imcflow checkpoint trained). Allow the
        # module to import; getModel_from_pretrained_weight will error clearly if
        # called without a checkpoint available.
        return {}, ""
    with open(registry_path) as f:
        reg = _json.load(f)
    base = os.path.join(cim_dir, reg["_base"])
    checkpoints = {
        k: os.path.normpath(os.path.join(base, v, "checkpoint.pth.tar"))
        for k, v in reg["entries"].items()
    }
    return checkpoints, reg.get("default", "")


VWW_CHECKPOINTS, VWW_DEFAULT_CKPT = _load_vww_checkpoints()


def get_last_checkpoint_path():
    """Return the checkpoint path used by the most recent getModel_from_pretrained_weight() call."""
    return _last_checkpoint_path


def get_last_checkpoint_alias():
    """Return the checkpoint alias used by the most recent getModel_from_pretrained_weight() call."""
    return _last_checkpoint_alias


def get_height(H, KH, padding, stride):
    out_h = (H + 2 * padding - KH) // stride + 1
    return out_h


def get_width(W, KW, padding, stride):
    out_w = (W + 2 * padding - KW) // stride + 1
    return out_w


def _tf_same_pad_amounts(in_size, kernel, stride):
    """TF/Keras 'same' total pad and (before, after) split for one spatial dim.

    Matches CIM deploy _same_pad: F.pad uses [pad//2, pad - pad//2] = (before, after).
    """
    import math
    pad = max(0, (math.ceil(in_size / stride) - 1) * stride + kernel - in_size)
    before = pad // 2
    after = pad - before
    return before, after


def _apply_tf_same_pad(y, H, W, kernel, stride):
    """Apply TF-'same' spatial padding to an NCHW int/float tensor via relay.nn.pad.

    Returns (padded_y, padded_H, padded_W). The subsequent conv must use padding=0.
    Symmetric pads collapse to the conv's own padding by the caller if desired, but
    we always emit an explicit pad here so stride-2 (asymmetric) layers are exact.
    """
    t, b = _tf_same_pad_amounts(H, kernel, stride)
    l, r = _tf_same_pad_amounts(W, kernel, stride)
    y = relay.nn.pad(y, pad_width=((0, 0), (0, 0), (t, b), (l, r)))
    return y, H + t + b, W + l + r


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
        self.count += 1
        if self.until_relay is not None and self.count > self.until_relay:
            raise EarlyStopException(y, self.input_var)
        return y


def getModel_(input_shape, until_relay: int = None, replicate_factor: int = 1):
    """
    Get VWW MobileNetV1 model with optional early stopping at a specific relay op.

    Args:
        input_shape: Input tensor shape (N, C, H, W) - e.g. (1, 3, 96, 96)
        until_relay: Optional index to stop at (0-based). Stops after the Nth
                     c.check() relay op. If None, returns the full model.
        replicate_factor: Replication factor for IMCFlow ops (default 1)

    Returns:
        (IRModule, var_info)
    """
    input = relay.var("model_input", shape=input_shape, dtype="float32")
    N, IC, H, W = input_shape

    c = RelayOpCounter(until_relay, input)

    try:
        # ============== Stem (FP32 - CPU) ==============
        # Conv2d 3->8, k=3, s=2, TF-'same' padding (asymmetric for even input).
        y, H, W = _apply_tf_same_pad(input, H, W, kernel=3, stride=2)
        y = c.check(relay.nn.conv2d(
            y,
            relay.var("stem_weight", shape=(STEM_OUT, IC, 3, 3), dtype="float32"),
            in_channels=IC,
            channels=STEM_OUT,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=(0, 0),
        ))
        IC = STEM_OUT
        H = get_height(H, 3, 0, 2)
        W = get_width(W, 3, 0, 2)

        y = c.check(relay.nn.batch_norm(
            y,
            relay.var("stem_bn_gamma", shape=(STEM_OUT,), dtype="float32"),
            relay.var("stem_bn_beta", shape=(STEM_OUT,), dtype="float32"),
            relay.var("stem_bn_mean", shape=(STEM_OUT,), dtype="float32"),
            relay.var("stem_bn_var", shape=(STEM_OUT,), dtype="float32"),
        )[0])
        # NOTE: deploy stem has NO relu after BN (relu lives in the head).

        # Convert to int16 for IMCFlow processing (block-1 FP input boundary).
        y = c.check(y * relay.var("x_f_1", shape=(1,), dtype="float32"))
        y = relay.clip(y, a_min=-32768.0, a_max=32767.0)
        y = c.check(relay.cast(y, dtype="int16"))

        # ============== 13 depthwise-separable blocks ==============
        for b, (blk_ic, blk_oc, dw_stride) in enumerate(VWW_BLOCK_CONFIG, start=1):
            assert blk_ic == IC, f"block{b} IC mismatch: expected {IC}, config {blk_ic}"

            # ----- Depthwise path (off-array, use_imcu=0) -----
            y = c.check(imcflow_min_max_quantize(
                y,
                relay.var(f"quant_dw_min_{b}", shape=(), dtype="int16"),
                relay.var(f"quant_dw_max_{b}", shape=(), dtype="int16"),
                axis=1, out_dtype="uint8", channel=blk_ic,
                replicate_factor=replicate_factor,
            ))

            # TF-'same' pad before the depthwise conv, then conv with padding=0.
            yp, Hp, Wp = _apply_tf_same_pad(y, H, W, kernel=3, stride=dw_stride)
            y = c.check(imcflow_qdwconv2d(
                yp,
                relay.var(f"weight_dw_{b}", shape=(blk_ic, 1, 3, 3), dtype="int8"),
                ConfigData((N, blk_ic, Hp, Wp), (blk_ic, 1, 3, 3),
                           padding=0, stride=dw_stride, use_imcu=0).get_as_const_tensor(),
                in_channels=blk_ic,
                channels=blk_ic,
                kernel_size=(3, 3),
                strides=(dw_stride, dw_stride),
                padding=(0, 0),
                groups=blk_ic,
                out_dtype="int16",
            ))
            H = get_height(Hp, 3, 0, dw_stride)
            W = get_width(Wp, 3, 0, dw_stride)

            y = c.check(imcflow_batch_norm(
                y,
                relay.var(f"fused_scale_dw_{b}", shape=(blk_ic,), dtype="int16"),
                relay.var(f"fused_bias_dw_{b}", shape=(blk_ic,), dtype="int16"),
            ))

            # ----- Pointwise path (on-array IMC MVM) -----
            y = c.check(imcflow_min_max_quantize(
                y,
                relay.var(f"quant_pw_min_{b}", shape=(), dtype="int16"),
                relay.var(f"quant_pw_max_{b}", shape=(), dtype="int16"),
                axis=1, out_dtype="uint8", channel=blk_ic,
                replicate_factor=replicate_factor,
            ))
            y = c.check(imcflow_qconv2d(
                y,
                relay.var(f"weight_pw_{b}", shape=(blk_oc, blk_ic, 1, 1), dtype="int8"),
                ConfigData((N, blk_ic, H, W), (blk_oc, blk_ic, 1, 1),
                           padding=0, stride=1).get_as_const_tensor(),
                in_channels=blk_ic,
                channels=blk_oc,
                kernel_size=(1, 1),
                out_dtype="int16",
                replicate_factor=replicate_factor,
            ))
            IC = blk_oc

            y = c.check(imcflow_batch_norm(
                y,
                relay.var(f"fused_scale_pw_{b}", shape=(blk_oc,), dtype="int16"),
                relay.var(f"fused_bias_pw_{b}", shape=(blk_oc,), dtype="int16"),
            ))

        # ============== Head (FP32 - CPU) ==============
        # int16 -> FP exit (divide by last block's bn_pw_f, per-channel).
        y = c.check(relay.cast(y, dtype="float32")
                    * relay.var("post_f_inv", shape=(IC, 1, 1), dtype="float32"))
        y = c.check(relay.nn.relu(y))
        y = c.check(relay.nn.adaptive_avg_pool2d(y, output_size=(1, 1)))
        y = c.check(relay.nn.batch_flatten(y))
        y = c.check(relay.nn.dense(
            y, relay.var("dense_weight", shape=(NUM_CLASS, IC), dtype="float32")))
        y = c.check(relay.nn.bias_add(
            y, relay.var("dense_bias", shape=(NUM_CLASS,), dtype="float32")))

        return _finalize_model(y, input)

    except EarlyStopException as e:
        return _finalize_model(e.y, e.input_var)


def _finalize_model(y, input):
    """Collect parameter vars from the graph and create the IRModule."""
    free_vars = relay.analysis.free_vars(y)
    var_info = {}
    for v in free_vars:
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
        else:
            continue

    out = tvm.IRModule.from_expr(y)
    return out, var_info


def getModel_from_pretrained_weight(iH=96, iW=96, until_relay=None, replicate_factor=1):
    """
    Get VWW MobileNetV1 subset model with pretrained weights from an imcflow checkpoint.

    Checkpoint selection: CKPT_PATH > CKPT alias > registry default. BOARD/vmode are
    intentionally ignored for VWW (no effect on this model).

    Returns:
        (out, params_dict)
    """
    import torch
    import re

    out, var_dict = getModel_([1, 3, iH, iW], until_relay=until_relay,
                              replicate_factor=replicate_factor)

    direct_checkpoint_path = os.getenv("CKPT_PATH", "").strip()
    direct_checkpoint_alias = os.getenv("CKPT", "").strip() or None

    ckpt_key = None
    if direct_checkpoint_path:
        checkpoint_path = os.path.abspath(os.path.expanduser(direct_checkpoint_path))
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"CKPT_PATH does not exist or is not a file: {checkpoint_path}")
        ckpt_key = direct_checkpoint_alias
        print(f"[INFO] Loading VWW checkpoint from CKPT_PATH={checkpoint_path}")
    else:
        ckpt_key = direct_checkpoint_alias
        if not ckpt_key:
            ckpt_key = VWW_DEFAULT_CKPT
            if not ckpt_key:
                raise ValueError(
                    "No VWW checkpoint available. The imcflow checkpoint has not been "
                    "generated yet (CIM/checkpoints/b2_half_vww.json missing or empty). "
                    "Set CKPT_PATH to an imcflow-format checkpoint.pth.tar, or create the "
                    "registry once the checkpoint exists.")
            print(f"\033[93m[WARNING] CKPT not set. Defaulting to '{ckpt_key}'. "
                  f"Available: {list(VWW_CHECKPOINTS.keys())}\033[0m")
        if ckpt_key not in VWW_CHECKPOINTS:
            raise ValueError(f"Unknown CKPT='{ckpt_key}'. Available VWW checkpoints: "
                             f"{list(VWW_CHECKPOINTS.keys())}")
        checkpoint_path = VWW_CHECKPOINTS[ckpt_key]

    global _last_checkpoint_path, _last_checkpoint_alias
    _last_checkpoint_path = checkpoint_path
    _last_checkpoint_alias = ckpt_key

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
    model_dict = checkpoint['state_dict']
    adjust_factors = checkpoint['adjust_factors']

    def _find_key(candidates):
        """Return the first key present in model_dict from a list of candidates."""
        for k in candidates:
            if k in model_dict:
                return k
        return None

    def _get_tensor_from_checkpoint(name, dtype, shape):
        # Stem / fc direct mappings. The imcflow checkpoint nests stem/fc under
        # the _StemHead submodule (_sh.*); accept both with and without the prefix.
        direct_groups = {
            'stem_weight':  ['_sh.stem_conv.weight', 'stem_conv.weight'],
            'stem_bn_gamma': ['_sh.stem_bn.weight', 'stem_bn.weight'],
            'stem_bn_beta':  ['_sh.stem_bn.bias', 'stem_bn.bias'],
            'stem_bn_mean':  ['_sh.stem_bn.running_mean', 'stem_bn.running_mean'],
            'stem_bn_var':   ['_sh.stem_bn.running_var', 'stem_bn.running_var'],
            'dense_weight':  ['_sh.fc.weight', 'fc.weight'],
            'dense_bias':    ['_sh.fc.bias', 'fc.bias'],
        }
        if name in direct_groups:
            key = _find_key(direct_groups[name])
            if key is None:
                raise ValueError(f"No checkpoint key for {name} (tried {direct_groups[name]})")
            tensor = model_dict[key].cpu().numpy().astype(dtype)
            if tensor.shape != shape:
                raise ValueError(f"Shape mismatch for {name}: expected {shape}, got {tensor.shape}")
            return tensor

        # FP->int16 entry scale (scalar).
        if name == 'x_f_1':
            return np.array([adjust_factors['x_f_1']], dtype=dtype)

        # int16->FP exit: post_f_inv = 1 / bn_pw_f_13 (per-channel OC=256 or scalar).
        if name == 'post_f_inv':
            bn_pw_f_last = np.asarray(adjust_factors[f'bn_pw_f_{len(VWW_BLOCK_CONFIG)}'],
                                     dtype='float64').flatten()
            inv = 1.0 / bn_pw_f_last
            return np.broadcast_to(inv.reshape(-1, 1, 1), shape).astype(dtype).copy()

        # Per-block weights / fused bn / quant ranges.
        m = re.match(r'weight_dw_(\d+)', name)
        if m:
            b = int(m.group(1))
            key = f"blocks.{b-1}.block_int16.dw.weight"
            return _block_tensor(name, key, dtype, shape)

        m = re.match(r'weight_pw_(\d+)', name)
        if m:
            b = int(m.group(1))
            key = f"blocks.{b-1}.block_int16.pw.weight"
            return _block_tensor(name, key, dtype, shape)

        m = re.match(r'fused_(scale|bias)_dw_(\d+)', name)
        if m:
            ptype, b = m.group(1), int(m.group(2))
            key = f"blocks.{b-1}.block_int16.bn_dw.{ptype}"
            return _block_tensor(name, key, dtype, shape)

        m = re.match(r'fused_(scale|bias)_pw_(\d+)', name)
        if m:
            ptype, b = m.group(1), int(m.group(2))
            key = f"blocks.{b-1}.block_int16.bn_pw.{ptype}"
            return _block_tensor(name, key, dtype, shape)

        m = re.match(r'quant_dw_(min|max)_(\d+)', name)
        if m:
            ptype, b = m.group(1), int(m.group(2))
            key = f"blocks.{b-1}.block_int16.act_dw.{ptype}"
            return _scalar_tensor(name, key, dtype, shape)

        m = re.match(r'quant_pw_(min|max)_(\d+)', name)
        if m:
            ptype, b = m.group(1), int(m.group(2))
            key = f"blocks.{b-1}.block_int16.act_pw.{ptype}"
            return _scalar_tensor(name, key, dtype, shape)

        raise ValueError(f"No mapping found for parameter: {name} (dtype={dtype}, shape={shape})")

    def _block_tensor(name, key, dtype, shape):
        if key not in model_dict:
            raise ValueError(f"Key {key} not found in checkpoint for parameter {name}")
        tensor = model_dict[key].cpu().numpy().astype(dtype)
        if tensor.shape != shape:
            raise ValueError(f"Shape mismatch for {name}: expected {shape}, got {tensor.shape}")
        return tensor

    def _scalar_tensor(name, key, dtype, shape):
        if key not in model_dict:
            raise ValueError(f"Key {key} not found in checkpoint for parameter {name}")
        tensor = model_dict[key].cpu().numpy()
        if shape == ():
            return tensor.astype(dtype) if tensor.shape == () else np.array(tensor.item(), dtype=dtype)
        elif shape == (1,):
            return np.array([tensor.item()], dtype=dtype) if tensor.shape == () else tensor.astype(dtype)
        raise ValueError(f"Unexpected shape {shape} for scalar parameter {name}")

    params_dict = {}
    for name in sorted(var_dict.keys()):
        if name == "model_input":
            continue
        info = var_dict[name]
        params_dict[name] = _get_tensor_from_checkpoint(name, info["dtype"], info["shape"])

    return out, params_dict
