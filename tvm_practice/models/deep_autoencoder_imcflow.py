"""
Deep autoencoder for IMCFlow: FP front/head and eight 1x1 IMC Dense blocks.

Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""
import json
import os
from pathlib import Path

import numpy as np
import tvm
from tvm import relay
from tvm.relay.qnn.op.qnn import imcflow_min_max_quantize
from tvm.relay.op.nn import imcflow_batch_norm, imcflow_qconv2d
from tvm.relay.backend.contrib.imcflow.acim_util import ConfigData
from .utils import get_param_info_from_relay_func


DAE_BLOCKS = ("enc2", "enc3", "enc4", "bottleneck", "dec1", "dec2", "dec3", "dec4")
DAE_BLOCK_CHANNELS = ((128, 128), (128, 128), (128, 128), (128, 8),
                      (8, 128), (128, 128), (128, 128), (128, 128))
DAE_LAYER_IDX_TO_RELAY_WEIGHT_NAME = {i: f"weight{i + 2}" for i in range(8)}
_last_checkpoint_path = None
_last_checkpoint_alias = None


def get_last_checkpoint_path():
    return _last_checkpoint_path


def get_last_checkpoint_alias():
    return _last_checkpoint_alias


class EarlyStopException(Exception):
    def __init__(self, y):
        self.y = y
        super().__init__()


class RelayOpCounter:
    """Zero-based assignment index, matching the existing subset builders."""
    def __init__(self, until_relay):
        if until_relay is not None and (not isinstance(until_relay, int) or until_relay < 0):
            raise ValueError("until_relay must be a non-negative integer")
        self.count = 0
        self.until_relay = until_relay

    def check(self, y):
        index = self.count
        self.count += 1
        if self.until_relay == index:
            raise EarlyStopException(y)
        return y


def getModel_(input_shape, until_relay=None):
    """Build [N,640] -> [N,640], optionally stopping at a Relay assignment."""
    if len(input_shape) != 2 or input_shape[1] != 640 or input_shape[0] < 1:
        raise ValueError("DAE expects input shape (N, 640), N >= 1")
    n, input_dim = input_shape
    x = relay.var("model_input", shape=input_shape, dtype="float32")
    c = RelayOpCounter(until_relay)
    try:
        y = c.check(relay.nn.dense(
            x, relay.var("weight1", shape=(128, input_dim), dtype="float32")))
        y = c.check(relay.nn.batch_norm(
            y,
            relay.var("bn_gamma1", shape=(128,), dtype="float32"),
            relay.var("bn_beta1", shape=(128,), dtype="float32"),
            relay.var("bn_moving_mean1", shape=(128,), dtype="float32"),
            relay.var("bn_moving_var1", shape=(128,), dtype="float32"),
            epsilon=1e-5)[0])
        # Signed front: deploy has no ReLU here; clamp before truncating.
        y = c.check(y * relay.var("scale_f1", shape=(1,), dtype="float32"))
        y = c.check(relay.clip(y, -32768, 32767))
        y = c.check(relay.cast(y, "int16"))
        y = c.check(relay.reshape(y, (n, 128, 1, 1)))

        for i, (ic, oc) in enumerate(DAE_BLOCK_CHANNELS, start=1):
            y = c.check(imcflow_min_max_quantize(
                y, relay.var(f"quant_min{i}", shape=(), dtype="int16"),
                relay.var(f"quant_max{i}", shape=(), dtype="int16"),
                axis=1, out_dtype="uint8", channel=ic))
            y = c.check(imcflow_qconv2d(
                y, relay.var(f"weight{i + 1}", shape=(oc, ic, 1, 1), dtype="int8"),
                ConfigData((n, ic, 1, 1), (oc, ic, 1, 1),
                           padding=0, stride=1).get_as_const_tensor(),
                in_channels=ic, channels=oc, kernel_size=(1, 1), out_dtype="int16"))
            y = c.check(imcflow_batch_norm(
                y, relay.var(f"fused_scale{i}", shape=(oc,), dtype="int16"),
                relay.var(f"fused_bias{i}", shape=(oc,), dtype="int16")))

        y = c.check(relay.cast(y, "float32") *
                    relay.var("post_f_inv", shape=(1,), dtype="float32"))
        y = c.check(relay.reshape(y, (n, 128)))
        y = c.check(relay.nn.relu(y))
        y = c.check(relay.nn.dense(
            y, relay.var("dense_weight_final", shape=(input_dim, 128), dtype="float32")))
        y = c.check(relay.nn.bias_add(
            y, relay.var("dense_bias_final", shape=(input_dim,), dtype="float32"), axis=1))
    except EarlyStopException as stop:
        y = stop.y
    return tvm.IRModule.from_expr(y), get_param_info_from_relay_func(y)


def _make_synthetic_param(name, dtype, shape, rng):
    """Deterministic, finite synthetic parameters, without changing global RNG."""
    if name in ("bn_gamma1", "bn_moving_var1"):
        return np.ones(shape, dtype=dtype)
    if name in ("bn_beta1", "bn_moving_mean1", "dense_bias_final"):
        return np.zeros(shape, dtype=dtype)
    constants = {"scale_f1": 64.0, "post_f_inv": 1.0 / 64.0}
    if name in constants:
        return np.full(shape, constants[name], dtype=dtype)
    for prefix, value in (("quant_min", -512), ("quant_max", 511),
                          ("fused_scale", 1), ("fused_bias", 0)):
        if name.startswith(prefix):
            return np.full(shape, value, dtype=dtype)
    if name in ("weight1", "dense_weight_final"):
        return rng.uniform(-0.125, 0.125, size=shape).astype(dtype)
    if name.startswith("weight"):
        return rng.integers(-2, 3, size=shape, dtype=np.dtype(dtype))
    raise ValueError(f"No synthetic DAE initializer for {name!r}")


def getModel(small_debug=False, seed=1234, until_relay=None):
    """Synthetic eight-block smoke model; small_debug retains API compatibility."""
    out, var_dict = getModel_((1, 640), until_relay=until_relay)
    rng = np.random.default_rng(seed)
    return out, {name: _make_synthetic_param(name, info["dtype"], info["shape"], rng)
                 for name, info in sorted(var_dict.items())}


def _checkpoint_path():
    direct = os.environ.get("CKPT_PATH", "").strip()
    alias = os.environ.get("CKPT", "").strip() or None
    if direct:
        path = Path(direct).expanduser().resolve()
    else:
        board = os.environ.get("BOARD", "B1").upper()
        cim = Path(os.environ.get("CIM_DIR", "/root/project/CIM"))
        registry = cim / "checkpoints" / f"{board.lower()}_half_ad.json"
        with registry.open() as stream:
            reg = json.load(stream)
        alias = alias or reg.get("default")
        if alias not in reg["entries"]:
            raise ValueError(f"Unknown DAE CKPT={alias!r}; available: {list(reg['entries'])}")
        path = (cim / reg["_base"] / reg["entries"][alias] / "checkpoint.pth.tar").resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DAE deploy checkpoint does not exist: {path}")
    return str(path), alias


def _validate_factors(factors):
    values = {}
    for i in range(1, 9):
        for prefix in ("x_f", "bn_f"):
            key = f"{prefix}_{i}"
            value = factors[key]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            scalar = np.asarray(value)
            if scalar.shape != () or not np.isfinite(scalar) or scalar <= 0:
                raise ValueError(f"{key} must be a positive finite scalar")
            values[key] = float(scalar)
    for i in range(1, 8):
        if values[f"bn_f_{i}"] != values[f"x_f_{i + 1}"]:
            raise ValueError(f"DAE factor tie violated: bn_f_{i} != x_f_{i + 1}")
    return values


def getModel_from_pretrained_weight(until_relay=None):
    """Load a deploy export (not a raw training checkpoint), preserving int values."""
    import torch

    global _last_checkpoint_path, _last_checkpoint_alias
    _last_checkpoint_path = _last_checkpoint_alias = None
    path, alias = _checkpoint_path()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    factors = _validate_factors(checkpoint["adjust_factors"])
    out, var_info = getModel_((1, 640), until_relay=until_relay)
    mappings = {
        "weight1": "_fh.linear1.weight",
        "bn_gamma1": "_fh.bn1.weight",
        "bn_beta1": "_fh.bn1.bias",
        "bn_moving_mean1": "_fh.bn1.running_mean",
        "bn_moving_var1": "_fh.bn1.running_var",
        "dense_weight_final": "_fh.out.weight",
        "dense_bias_final": "_fh.out.bias",
    }
    for i in range(1, 9):
        prefix = f"blocks.{i - 1}.block_int16"
        mappings.update({
            f"weight{i + 1}": f"{prefix}.linear.weight",
            f"quant_min{i}": f"{prefix}.act.min",
            f"quant_max{i}": f"{prefix}.act.max",
            f"fused_scale{i}": f"{prefix}.bn.scale",
            f"fused_bias{i}": f"{prefix}.bn.bias",
        })
    params = {}
    for name, info in var_info.items():
        dtype, shape = info["dtype"], tuple(info["shape"])
        if name == "scale_f1":
            value = np.asarray([factors["x_f_1"]])
        elif name == "post_f_inv":
            value = np.asarray([1.0 / factors["bn_f_8"]])
        else:
            value = state[mappings[name]]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            value = np.asarray(value)
            if shape == () and value.size == 1:
                value = value.reshape(())
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"Invalid {name}: expected finite shape {shape}, got {value.shape}")
        target = np.dtype(dtype)
        if np.issubdtype(target, np.integer):
            low, high = (np.iinfo(target).min, np.iinfo(target).max)
            if name.startswith("weight"):
                low, high = -8, 7
            if np.any(value != np.trunc(value)) or np.any(value < low) or np.any(value > high):
                raise ValueError(f"{name} is not representable in [{low}, {high}]")
        with np.errstate(over="ignore"):
            converted = value.astype(target)
        if not np.all(np.isfinite(converted)):
            raise ValueError(f"{name} overflows {dtype}")
        params[name] = converted
    for i in range(1, 9):
        if f"quant_max{i}" in params and params[f"quant_min{i}"] >= params[f"quant_max{i}"]:
            raise ValueError(f"Invalid quantization bounds for {DAE_BLOCKS[i - 1]}")
    if "bn_moving_var1" in params and np.any(params["bn_moving_var1"] < 0):
        raise ValueError("Negative front BN running variance")
    _last_checkpoint_path, _last_checkpoint_alias = path, alias
    return out, params
