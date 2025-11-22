"""
Sanity test for LayoutPropagationContext using a small ResNet8-inspired fragment.
"""

import sys
from pathlib import Path
import numpy as np

# Ensure TVM python path is available
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "python"))

import tvm
from tvm import relay
from tvm.relay.qnn.op.qnn import imcflow_min_max_quantize
from tvm.relay.op.nn import imcflow_qconv2d

from tvm.relay.backend.contrib.imcflow.transform import (
    LayoutPropagationContext,
    LayoutType,
)


def _const(shape, dtype):
  return relay.const(np.zeros(shape, dtype=dtype))


def build_resnetish_fragment():
  """A two-path block inspired by resnet8_cifar.py but simplified for testing."""
  data = relay.var("data", shape=(1, 16, 8, 8), dtype="int16")
  skip = relay.var("skip", shape=(1, 16, 8, 8), dtype="int16")

  # Quantization path producing qconv-friendly layout
  q = imcflow_min_max_quantize(
    data,
    relay.const(0, dtype="int16"),
    relay.const(255, dtype="int16"),
    axis=1,
    out_dtype="uint8",
    channel=16,
  )

  # qconv block
  weight = _const((16, 16, 3, 3), "int8")
  kernel_cfg = relay.const(np.array([0], dtype="int32"))
  conv = imcflow_qconv2d(
    q,
    weight,
    kernel_cfg,
    in_channels=16,
    channels=16,
    kernel_size=(3, 3),
    padding=(1, 1),
    strides=(1, 1),
    out_dtype="int16",
  )

  out = relay.add(conv, skip)
  attrs = tvm.ir.make_node("DictAttrs", Compiler="imcflow", Primitive=1, Inline=1)
  func= relay.Function([data, skip], out, attrs=attrs)
  print("Built ResNetish fragment:")
  print(func)
  return func


def test_layout_propagation_context_resnetish():
  func = build_resnetish_fragment()
  ctx = LayoutPropagationContext(func)
  combos = ctx.build_var_layout_combinations()

  assert combos, "Expected at least one layout combination"
  print(f"Found {len(combos)} layout combinations.")
  print("Layout combinations:")
  for combo in combos:
    by_name = {var.name_hint: layout for var, layout in combo.items()}
    print(by_name)
  # Two parameters expected: data and skip
  assert all(len(combo) == 2 for combo in combos)

  expected_data_layout = LayoutType.NCHW16C  # from min_max_quantize input rule
  expected_skip_layouts = {LayoutType.NCHW16C, LayoutType.NCHW64C, LayoutType.SCALAR}  # from add patterns

  for combo in combos:
    by_name = {var.name_hint: layout for var, layout in combo.items()}
    assert by_name["data"] == expected_data_layout
    assert by_name["skip"] in expected_skip_layouts


if __name__ == "__main__":
  test_layout_propagation_context_resnetish()
  print("LayoutPropagationContext resnetish test passed.")
