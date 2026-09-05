"""Regression tests for deterministic metadata from split_conv_to_atomic."""

from collections import OrderedDict
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "python"))

import tvm
from tvm import relay
from tvm.contrib.imcflow import ImcflowDeviceConfig
from tvm.relay.backend.contrib.imcflow import transform
from tvm.relay.op.nn import imcflow_qconv2d


class _OrderedModule:
  """Minimal module wrapper whose function iteration order is controllable."""

  def __init__(self, functions):
    self.functions = OrderedDict(functions)

  def __setitem__(self, global_var, func):
    self.functions[global_var] = func


def _make_qconv_function(symbol, weight_value):
  data = relay.var(f"data_{symbol}", shape=(1, 8, 4, 4), dtype="int16")
  weight_array = np.full((8, 8, 1, 1), weight_value, dtype="int8")
  weight = relay.const(weight_array)
  config = relay.const(np.array([0, 0], dtype="int32"))
  conv = imcflow_qconv2d(
      data,
      weight,
      config,
      channels=8,
      in_channels=8,
      kernel_size=(1, 1),
      out_dtype="int16",
  )
  attrs = tvm.ir.make_node(
      "DictAttrs", global_symbol=symbol, Compiler="imcflow"
  )
  return relay.Function([data], conv, attrs=attrs), weight_array


def test_orig_conv_ids_follow_global_symbol_order():
  func_z, weight_z = _make_qconv_function("imcflow_z", 7)
  func_a, weight_a = _make_qconv_function("imcflow_a", 3)
  global_z = tvm.ir.GlobalVar("imcflow_z")
  global_a = tvm.ir.GlobalVar("imcflow_a")
  mod = _OrderedModule([(global_z, func_z), (global_a, func_a)])

  config = ImcflowDeviceConfig()
  config.AtomicSplitInfo = {}
  config.OrigConvNameMap = {
      transform._weight_bytes_hash(tvm.nd.array(weight_z)): "weight_z",
      transform._weight_bytes_hash(tvm.nd.array(weight_a)): "weight_a",
  }

  transform.split_conv_to_atomic(mod, {}, effective_oc=32)

  entries = [
      entry
      for bucket in config.AtomicSplitInfo.values()
      for entry in bucket
  ]
  name_to_id = {
      entry["orig_conv_name"]: entry["orig_conv_id"] for entry in entries
  }
  assert name_to_id == {"weight_a": 0, "weight_z": 1}


if __name__ == "__main__":
  test_orig_conv_ids_follow_global_symbol_order()
