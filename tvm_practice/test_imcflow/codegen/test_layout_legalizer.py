import sys
from pathlib import Path

import numpy as np

# Ensure TVM python path is available
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT / "python"))

import tvm
from tvm import relay

from tvm.relay.backend.contrib.imcflow.transform import (
    get_required_layout_from_op,
    get_valid_output_layout_of_node,
    LayoutType,
)


def _const(shape, dtype):
  return relay.const(np.zeros(shape, dtype=dtype))


def _make_add_call():
  a = relay.var("a", shape=(1, 16, 8, 8))
  b = relay.var("b", shape=(1, 16, 8, 8))
  return relay.add(a, b)

def _flatten_layouts(layouts):
  flat = []
  for l in layouts:
    if isinstance(l, (list, tuple)):
      flat.extend(_flatten_layouts(l))
    else:
      flat.append(l)
  return flat


def _print_header(name, expr):
  print(f"\n=== {name} ===")
  print("expr:")
  print(expr)

def _binary_vars():
  a = relay.var("a", shape=(1, 16, 8, 8))
  b = relay.var("b", shape=(1, 16, 8, 8))
  return a, b


def test_builtin_call_inputs_outputs():
  call = _make_add_call()
  _print_header("builtin add", call)
  inputs_layouts = get_required_layout_from_op(call, "inputs", 0)
  outputs_layouts = get_required_layout_from_op(call, "outputs", 0)

  print("inputs layouts:", inputs_layouts)
  print("outputs layouts:", outputs_layouts)
  assert LayoutType.NCHW16C in inputs_layouts
  assert LayoutType.NCHW16C in outputs_layouts


def test_composite_input_param_mapping():
  p0 = relay.var("p0", shape=(1, 16, 8, 8))
  p1 = relay.var("p1", shape=(1, 16, 8, 8))
  body = relay.add(p0, p1)
  comp = relay.Function([p0, p1], body, attrs=tvm.ir.make_node("DictAttrs", Composite="test.comp"))
  call = relay.Call(comp, [_const((1, 16, 8, 8), "float32"), _const((1, 16, 8, 8), "float32")])

  _print_header("composite input mapping", call)
  layouts = get_required_layout_from_op(call, "inputs", 0)
  print("layouts for input 0:", layouts)
  assert LayoutType.NCHW16C in layouts


def test_composite_param_multiple_users():
  p0 = relay.var("p0", shape=(1, 16, 8, 8))
  p1 = relay.var("p1", shape=(1, 16, 8, 8))
  add_call = relay.add(p0, p1)
  mul_call = relay.multiply(p0, p1)
  body = relay.Tuple([add_call, mul_call])
  comp = relay.Function([p0, p1], body, attrs=tvm.ir.make_node("DictAttrs", Composite="test.comp"))
  call = relay.Call(comp, [_const((1, 16, 8, 8), "float32"), _const((1, 16, 8, 8), "float32")])

  _print_header("composite param multiple users", call)
  layouts = get_required_layout_from_op(call, "inputs", 0)
  print("layouts for input 0:", layouts)
  assert LayoutType.NCHW16C in layouts


def test_composite_output_tuple_uses_builtin_rule():
  p0 = relay.var("p0", shape=(1, 16, 8, 8))
  p1 = relay.var("p1", shape=(1, 16, 8, 8))
  add_call = relay.add(p0, p1)
  mul_call = relay.multiply(p0, p1)
  body = relay.Tuple([add_call, mul_call])
  comp = relay.Function([p0, p1], body, attrs=tvm.ir.make_node("DictAttrs", Composite="test.comp"))
  call = relay.Call(comp, [_const((1, 16, 8, 8), "float32"), _const((1, 16, 8, 8), "float32")])

  _print_header("composite output tuple shallow", call)
  outputs_layouts = get_required_layout_from_op(call, "outputs", 0)
  print("output layouts for field 0:", outputs_layouts)
  assert LayoutType.NCHW16C in _flatten_layouts(outputs_layouts)

def test_composite_output_tuple_uses_builtin_rule_deeper():
  p0 = relay.var("p0", shape=(1, 16, 8, 8))
  p1 = relay.var("p1", shape=(1, 16, 8, 8))
  add_call = relay.add(p0, p1)
  mul_call = relay.multiply(p0, p1)
  add_call2 = relay.add(add_call, mul_call)
  mul_call2 = relay.multiply(add_call, mul_call)
  body = relay.Tuple([add_call2, mul_call2])
  comp = relay.Function([p0, p1], body, attrs=tvm.ir.make_node("DictAttrs", Composite="test.comp"))
  call = relay.Call(comp, [_const((1, 16, 8, 8), "float32"), _const((1, 16, 8, 8), "float32")])

  _print_header("composite output tuple deep", call)
  outputs_layouts = get_required_layout_from_op(call, "outputs", 0)
  print("output layouts for field 0:", outputs_layouts)
  assert LayoutType.NCHW16C in _flatten_layouts(outputs_layouts)

def test_get_valid_output_layout_builtin():
  a, b = _binary_vars()
  call = relay.add(a, b)
  layout = get_valid_output_layout_of_node(call, [LayoutType.NCHW16C, LayoutType.NCHW16C])
  print("\n=== valid layout builtin ===")
  print("layout:", layout)
  assert layout == LayoutType.NCHW16C

def test_get_valid_output_layout_tuple():
  a, b = _binary_vars()
  add_call = relay.add(a, b)
  mul_call = relay.multiply(a, b)
  tup = relay.Tuple([add_call, mul_call])
  layout = get_valid_output_layout_of_node(tup, [LayoutType.NCHW16C, LayoutType.NCHW16C])
  print("\n=== valid layout tuple ===")
  print("layout:", layout)
  assert layout == (LayoutType.NCHW16C, LayoutType.NCHW16C)

def test_get_valid_output_layout_composite():
  p0 = relay.var("p0", shape=(1, 16, 8, 8))
  p1 = relay.var("p1", shape=(1, 16, 8, 8))
  body = relay.add(p0, p1)
  comp = relay.Function([p0, p1], body, attrs=tvm.ir.make_node("DictAttrs", Composite="test.comp"))
  call = relay.Call(comp, [_const((1, 16, 8, 8), "float32"), _const((1, 16, 8, 8), "float32")])
  layout = get_valid_output_layout_of_node(call, [LayoutType.NCHW16C, LayoutType.NCHW16C])
  print("\n=== valid layout composite ===")
  print("layout:", layout)
  assert layout == LayoutType.NCHW16C

def test_get_valid_output_layout_tgi():
  a, b = _binary_vars()
  tup = relay.Tuple([relay.add(a, b), relay.multiply(a, b)])
  tgi = relay.TupleGetItem(tup, 1)
  layout = get_valid_output_layout_of_node(tgi, [LayoutType.NCHW16C, LayoutType.NCHW16C])
  print("\n=== valid layout tuple_getitem ===")
  print("layout:", layout)
  assert layout == LayoutType.NCHW16C


if __name__ == "__main__":
  test_builtin_call_inputs_outputs()
  test_composite_input_param_mapping()
  test_composite_param_multiple_users()
  test_composite_output_tuple_uses_builtin_rule()
  test_composite_output_tuple_uses_builtin_rule_deeper()
  test_get_valid_output_layout_builtin()
  test_get_valid_output_layout_tuple()
  test_get_valid_output_layout_composite()
  test_get_valid_output_layout_tgi()
  print("layout legalizer tests passed.")
