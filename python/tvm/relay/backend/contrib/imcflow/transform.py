import tvm
from tvm import relay
from tvm.relay import transform, op
from tvm.relay.ty import TupleType, TensorType
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.function import Function, FunctionWithFields
from tvm.relay.expr import (Call, GlobalVar, TupleGetItem, const, Let, Var, If, Tuple, Constant)
from tvm.relay import expr as _expr
from tvm.relay.expr import RefCreate, RefRead, RefWrite
from tvm.relay.adt import Constructor, Match, Clause
from tvm.contrib.imcflow import ImcflowDeviceConfig, TensorEdge, TensorID, NodeID, TensorEdgeInfo, InstEdgeInfo, RouterEntry, DataBlock, MemoryLayout, MemoryRegion
from tvm.ir import Op
from tvm.relay.op.contrib.imcflow import HashToCustomID, CustomIDToName, CustomIDInFunc, CustomIDToNode

# Debug logging utility controlled by IMCFLOW_DEBUG environment variable
# Usage:
#   export IMCFLOW_DEBUG=1  # Enable all debug messages
#   export IMCFLOW_DEBUG=0  # Disable all debug messages
_DEBUG_ENABLED = None

def _is_debug_enabled():
    """Check if debug logging is enabled via IMCFLOW_DEBUG environment variable"""
    global _DEBUG_ENABLED
    if _DEBUG_ENABLED is None:
        debug_var = os.environ.get('IMCFLOW_DEBUG', '0')
        _DEBUG_ENABLED = debug_var == '1' or debug_var.lower() == 'true'
    return _DEBUG_ENABLED

def debug_print(*args, **kwargs):
    """Print debug message only if IMCFLOW_DEBUG is enabled"""
    if _is_debug_enabled():
        print(*args, **kwargs)


from tvm.contrib.relay_viz import RelayVisualizer, DotPlotter, DotVizParser
from tvm.relay import pretty_print
def printModel(result_dir, mod, param_dict, mod_name):
  RelayVisualizer(
      relay_mod=mod,
      relay_param=param_dict,
      plotter=DotPlotter(),
      parser=DotVizParser(),
  ).render(f"{result_dir}/{mod_name}")

  with open(f"{result_dir}/{mod_name}.txt", "w") as f:
    f.write(pretty_print(mod))

# Operator groups for predicate reuse
ELEMENT_WISE_OPS = {
  op.get("add"),
  op.get("multiply"),
  op.get("divide"),
  op.get("subtract"),
  op.get("clip"),
  op.get("nn.relu"),
}

def skip_element_wise_predicate(call, _idx):
  return isinstance(call, relay.Call) and isinstance(call.op, tvm.ir.Op) and call.op in ELEMENT_WISE_OPS

def skip_composite_predicate(call, _idx):
  return isinstance(call, relay.Call) and isinstance(call.op, relay.Function) and call.op.attrs and "Composite" in call.op.attrs.keys()
from tvm.relay.op.nn import imcflow_batch_norm, imcflow_qconv2d, imcflow_qdwconv2d
from tvm.relay.qnn.op.qnn import imcflow_min_max_quantize, imcflow_nu_quantize
from tvm.relay.op.transform import imcflow_packing, imcflow_unpacking, imcflow_4d_to_qconv_input, imcflow_mmquant_out_to_4d
import numpy as np
from tvm.relay.op.contrib import imcflow
from tvm.relay.backend.contrib.imcflow.acim_util import *

import math
from copy import deepcopy
import collections
import re
import itertools
from dataclasses import dataclass
from enum import Enum
import json
import pprint
import os

#TODO:
# 1. memory allocation is not optimal. we can split tensor to multiple inode. current implementation assign only one inode for entire tensor.

# Layout requirements for each IMCFLOW-friendly op.
# These drive packing/unpacking instead of ad-hoc shape checks.
class LayoutType(Enum):
  NCHW = "NCHW"
  NCHW16C = "NCHW16c"
  NCHW64C = "NCHW64c"
  QCONV_INPUT = "QCONV_INPUT"   # Packed activation layout used by qconv input path
  QCONV_WEIGHT = "QCONV_WEIGHT" # Packed filter layout for qconv
  QDCONV_WEIGHT = "QDCONV_WEIGHT"
  SCALAR = "SCALAR"
  C="C"
  MK="MK"

def is_layout_compatible_with_type(layout, ttype):
  if layout == LayoutType.SCALAR:
    if not isinstance(ttype, TensorType):
      return False
    rank = len(ttype.shape)
    if rank == 0:
      return True
    if (rank == 1 and ttype.shape[0] == 1) or (rank == 1 and ttype.shape[0] == 8 and ttype.dtype == "uint32"): # conv config
      return True
    return False
  return True

def _deduce_layout_from_op_const(call, index, not_const_layouts):
  if not isinstance(call, relay.Call):
    raise ValueError("Input must be a relay.Call")
  if not isinstance(call.op, tvm.ir.Op):
    raise ValueError("call.op must be a tvm.ir.Op")

  op_name = call.op.name
  const = call.args[index]
  ttype = const.checked_type
  if op_name == "nn.imcflow_qconv":
    if index == 1:
      return LayoutType.QCONV_WEIGHT
    elif index == 2:
      return LayoutType.SCALAR
  elif op_name == "nn.imcflow_qdwconv":
    if index == 1:
      return LayoutType.QDCONV_WEIGHT
    elif index == 2:
      return LayoutType.SCALAR
  elif op_name == "qnn.imcflow_min_max_quantize":
    if index in [1, 2]:
      return LayoutType.SCALAR
  elif op_name == "qnn.imcflow_nu_quantize":
    if index in [1, 2]:
      return LayoutType.SCALAR
  elif op_name == "imcflow.fused_batch_norm":
    if index in [1,2]:
      return LayoutType.C
  elif op_name == "nn.bias_add":
    if index == 1:
      return LayoutType.C
  elif op_name in ["add", "multiply", "divide"]:
    if len(ttype.shape) == 0 or (len(ttype.shape) == 1 and ttype.shape[0] == 1):
      return LayoutType.SCALAR
    else:
      if LayoutType.NCHW16C in not_const_layouts and (LayoutType.NCHW64C not in not_const_layouts):
        return LayoutType.NCHW16C
      elif LayoutType.NCHW64C in not_const_layouts and (LayoutType.NCHW16C not in not_const_layouts):
        return LayoutType.NCHW64C
      else:
        raise ValueError("Cannot deduce constant layout for binary op with ambiguous input layouts.")
  
  raise ValueError(f"Cannot deduce layout from op {op_name} at index {index}")

def get_op_name_of_call(call):
  if not isinstance(call, relay.Call):
    raise ValueError("Input must be a relay.Call")
  
  if not isinstance(call.op, tvm.ir.Op):
    raise ValueError("call.op must be a tvm.ir.Op or relay.GlobalVar")

  op_name = call.op.name
  
  return op_name

def find_first_builtin_call(self, expr):
  """DFS to find the first Call with builtin Op in the expression."""
  if isinstance(expr, relay.Call) and isinstance(expr.op, tvm.ir.Op):
    return expr
  elif isinstance(expr, relay.Call) and isinstance(expr.op, relay.Function):
    return self.find_first_builtin_call(expr.op.body)
  elif isinstance(expr, relay.Tuple):
    for field in expr.fields:
      found = self.find_first_builtin_call(field)
      if found:
        return found
  elif isinstance(expr, relay.TupleGetItem):
    return self.find_first_builtin_call(expr.tuple_value)
  return None


# Layout requirements per op.
# Each op maps to a list of rules. A rule is a tuple:
#   (inputs_options, output_layout)
# - inputs_options: list of possible input layout lists (one list per permutatio.
#   Example: [[a, b], [b, a]] means two valid input orderings.
#   If a layout list has length 1 and the call has more args, the single layout applies to all args.
# - output_layout: single layout applied to all outputs of the op.
REQUIRED_OP_LAYOUTS = {
  "nn.imcflow_qconv": [
    (
      [
        [LayoutType.QCONV_INPUT, LayoutType.QCONV_WEIGHT, LayoutType.SCALAR],
      ],
      LayoutType.NCHW64C,
    ),
  ],
  "nn.imcflow_qdwconv": [
    (
      [
        [LayoutType.QCONV_INPUT, LayoutType.QDCONV_WEIGHT, LayoutType.SCALAR],
      ],
      LayoutType.NCHW16C,
    ),
  ],
  "qnn.imcflow_min_max_quantize": [
    (
      [
        [LayoutType.NCHW16C, LayoutType.SCALAR, LayoutType.SCALAR],
      ],
      LayoutType.QCONV_INPUT,
    ),
    (
      [
        [LayoutType.NCHW64C, LayoutType.SCALAR, LayoutType.SCALAR],
      ],
      LayoutType.QCONV_INPUT,
    ),
  ],
  "qnn.imcflow_nu_quantize": [ #TODO: refine
    (
      [
        [LayoutType.NCHW16C, LayoutType.SCALAR, LayoutType.SCALAR],
      ],
      LayoutType.QCONV_INPUT,
    ),
    (
      [
        [LayoutType.NCHW64C, LayoutType.SCALAR, LayoutType.SCALAR],
      ],
      LayoutType.QCONV_INPUT,
    ),
  ],
  "nn.bias_add": [
    (
      [
        [LayoutType.NCHW16C, LayoutType.C],
      ],
      LayoutType.NCHW16C,
    ),
    (
      [
        [LayoutType.NCHW64C, LayoutType.C],
      ],
      LayoutType.NCHW64C,
    ),
    (
      [
        [LayoutType.NCHW, LayoutType.C],
      ],
      LayoutType.NCHW,
    ),
  ],
  "nn.relu": [
    (
      [
        [LayoutType.NCHW16C],
      ],
      LayoutType.NCHW16C,
    ),
    (
      [
        [LayoutType.NCHW64C],
      ],
      LayoutType.NCHW64C,
    ),
    (
      [
        [LayoutType.NCHW],
      ],
      LayoutType.NCHW,
    )
  ],
  "imcflow.fused_batch_norm": [
    (
      [
        [LayoutType.NCHW16C, LayoutType.C, LayoutType.C],
      ],
      LayoutType.NCHW16C,
    ),
    (
      [
        [LayoutType.NCHW64C, LayoutType.C, LayoutType.C],
      ],
      LayoutType.NCHW64C,
    ),
  ],
  "add": [
    (
      [
        [LayoutType.NCHW16C, LayoutType.NCHW16C],
      ],
      LayoutType.NCHW16C,
    ),
    (
      [
        [LayoutType.NCHW64C, LayoutType.NCHW64C],
      ],
      LayoutType.NCHW64C,
    ),
    (
      [
        [LayoutType.NCHW16C, LayoutType.SCALAR],
        [LayoutType.SCALAR, LayoutType.NCHW16C],
      ],
      LayoutType.NCHW16C,
    ),
    (
      [
        [LayoutType.NCHW64C, LayoutType.SCALAR],
        [LayoutType.SCALAR, LayoutType.NCHW64C],
      ],
      LayoutType.NCHW64C,
    ),
    (
      [
        [LayoutType.NCHW16C, LayoutType.NCHW64C],
        [LayoutType.NCHW64C, LayoutType.NCHW16C],
      ],
      LayoutType.NCHW64C,
    ),
    (
      [
        [LayoutType.NCHW, LayoutType.NCHW],
        [LayoutType.NCHW, LayoutType.SCALAR],
        [LayoutType.SCALAR, LayoutType.NCHW],
      ],
      LayoutType.NCHW,
    )
  ],
  "multiply": [
    (
      [
        [LayoutType.NCHW16C, LayoutType.NCHW16C],
        [LayoutType.NCHW16C, LayoutType.SCALAR],
        [LayoutType.SCALAR, LayoutType.NCHW16C],
      ],
      LayoutType.NCHW16C,
    ),
    (
      [
        [LayoutType.NCHW64C, LayoutType.NCHW64C],
        [LayoutType.NCHW64C, LayoutType.SCALAR],
        [LayoutType.SCALAR, LayoutType.NCHW64C],
      ],
      LayoutType.NCHW64C,
    ),
    (
      [
        [LayoutType.NCHW16C, LayoutType.NCHW64C],
        [LayoutType.NCHW64C, LayoutType.NCHW16C],
      ],
      LayoutType.NCHW64C,
    ),
    (
      [
        [LayoutType.NCHW, LayoutType.NCHW],
        [LayoutType.NCHW, LayoutType.SCALAR],
        [LayoutType.SCALAR, LayoutType.NCHW],
      ],
      LayoutType.NCHW,
    )
  ],
  "divide": [
    (
      [
        [LayoutType.NCHW16C, LayoutType.NCHW16C],
        [LayoutType.NCHW16C, LayoutType.SCALAR],
        [LayoutType.SCALAR, LayoutType.NCHW16C],
      ],
      LayoutType.NCHW16C,
    ),
    (
      [
        [LayoutType.NCHW16C, LayoutType.NCHW64C],
        [LayoutType.NCHW64C, LayoutType.NCHW16C],
      ],
      LayoutType.NCHW64C,
    ),
    (
      [
        [LayoutType.NCHW, LayoutType.NCHW],
        [LayoutType.NCHW, LayoutType.SCALAR],
        [LayoutType.SCALAR, LayoutType.NCHW],
      ],
      LayoutType.NCHW,
    )
  ],
  "split": [
    (
      [
        [LayoutType.QCONV_INPUT],
      ],
      LayoutType.QCONV_INPUT,
    ),
    (
      [
        [LayoutType.NCHW16C],
      ],
      LayoutType.NCHW16C,
    ),
    (
      [
        [LayoutType.NCHW64C],
      ],
      LayoutType.NCHW64C,
    ),
  ],
  "concatenate": [
    (
      [
        [LayoutType.QCONV_INPUT],
      ],
      LayoutType.QCONV_INPUT,
    ),
    (
      [
        [LayoutType.NCHW16C],
      ],
      LayoutType.NCHW16C,
    ),
    (
      [
        [LayoutType.NCHW64C],
      ],
      LayoutType.NCHW64C,
    ),
  ],
  "nn.conv2d" : [
    (
      [
        [LayoutType.NCHW, LayoutType.NCHW]
      ],
      LayoutType.NCHW,
    )
  ],
  "nn.batch_norm" : [
    (
      [
        [LayoutType.NCHW, LayoutType.C, LayoutType.C, LayoutType.C, LayoutType.C]
      ],
      LayoutType.NCHW,
    )
  ],
  "cast" : [
    (
      [
        [LayoutType.NCHW16C],
      ],
      LayoutType.NCHW16C,
    ),
    (
      [
        [LayoutType.NCHW64C],
      ],
      LayoutType.NCHW64C,
    ),
    (
      [
        [LayoutType.NCHW],
      ],
      LayoutType.NCHW,
    )
  ],
  "nn.bitpack" : [
    (
      [
        [LayoutType.NCHW],
      ],
      LayoutType.QCONV_INPUT,
    ),
  ],
  "nn.bitunpack" : [
    (
      [
        [LayoutType.QCONV_INPUT],
      ],
      LayoutType.NCHW,
    ),
  ],
  "nn.dense" : [
    (
      [
        [LayoutType.MK, LayoutType.MK],
      ],
      LayoutType.MK,
    ),
  ],
  "nn.batch_flatten": [
    (
      [
        [LayoutType.NCHW],
      ],
      LayoutType.MK,
    )
  ]
}

def get_required_layout_from_op(call, io_kind, index, cpu_node=False):
  """
  Lightweight accessor for layout rules.

  Parameters
  ----------
  call : relay.Call
    call node. it can be built-in op or composite function.
  io_kind : str
    "inputs" or "outputs".
  index : int
    Input or output index.
  """
  if not isinstance(call, relay.Call):
    raise ValueError("Input must be a relay.Call")

  # Resolve op name (builtin or from composite body)
  if isinstance(call.op, tvm.ir.Op):
    op_names = [call.op.name]
    index_list = [index]
    layout_set = set([LayoutType.SCALAR, LayoutType.NCHW, LayoutType.NCHW16C, LayoutType.NCHW64C, LayoutType.QCONV_INPUT, LayoutType.QCONV_WEIGHT, LayoutType.QDCONV_WEIGHT])
    for op_name, index in zip(op_names, index_list):
      # rules = REQUIRED_OP_LAYOUTS.get(op_name, None)
      rules = get_required_layout_rules(call, cpu_node=cpu_node)

      if rules is None:
        raise ValueError(f"Layout requirement not defined for op {op_name}")

      required_layouts_ = []
      for rule in rules:
        inputs_options, outputs_layout = rule
        if io_kind == "inputs":
          required_layouts_.extend([opt[index] for opt in inputs_options])
        elif io_kind == "outputs":
          required_layouts_.append(outputs_layout)
        else:
          raise ValueError(f"Unknown io_kind {io_kind}")
      
      layout = required_layouts_
      layout_set = layout_set.intersection(set(layout))

    return list(layout_set)
  elif isinstance(call.op, relay.Function):
    # Traverse into composite function to find first builtin call
    def first_builtin_pred(_ctx, expr):
      curr_is_builtin = isinstance(expr, relay.Call) and isinstance(expr.op, tvm.ir.Op)
      stack = _ctx["stack"]
      already_meet = any([isinstance(s, relay.Call) and isinstance(s.op, tvm.ir.Op) for s in stack])
      return curr_is_builtin and not already_meet
    collector = NodeCollector(predicates=[first_builtin_pred])
    collected = collector.collect(call.op.body)
    if not collected:
      raise ValueError("Composite function does not contain builtin call for layout lookup")

    if io_kind == "inputs":
      # Map param index to actual arg usage in first builtin call
      target_param = call.op.params[index]
      # Find user of target_param inside composite body
      use_def = UseDefChainParser()
      use_def.visit(call.op.body)
      users = use_def.get_users(target_param)
      if not users:
        raise ValueError("Composite param has no users in body for layout lookup")

      builtin_users = []
      builtin_idxs = []
      for user, arg_idx in users:
        if isinstance(user, relay.Call) and isinstance(user.op, tvm.ir.Op):
          builtin_users.append(user)
          builtin_idxs.append(arg_idx)
      
      op_names = [builtin_user.op.name for builtin_user in builtin_users]
      index_list = builtin_idxs

      layout_set = set([LayoutType.SCALAR, LayoutType.NCHW, LayoutType.NCHW16C, LayoutType.NCHW64C, LayoutType.QCONV_INPUT, LayoutType.QCONV_WEIGHT, LayoutType.QDCONV_WEIGHT])
      for op_name, index, builtin_user in zip(op_names, index_list, builtin_users):
        # rules = REQUIRED_OP_LAYOUTS.get(op_name, None)
        rules = get_required_layout_rules(builtin_user, cpu_node=cpu_node)

        if rules is None:
          raise ValueError(f"Layout requirement not defined for op {op_name}")

        required_layouts_ = []
        for rule in rules:
          inputs_options, outputs_layout = rule
          if io_kind == "inputs":
            required_layouts_.extend([opt[index] for opt in inputs_options])
          elif io_kind == "outputs":
            required_layouts_.append(outputs_layout)
          else:
            raise ValueError(f"Unknown io_kind {io_kind}")
        
        layout = required_layouts_
        layout_set = layout_set.intersection(set(layout))

      return list(layout_set)
    elif io_kind == "outputs":
      if len(collected) == 1:
        builtin_call = collected[0]
        op_name = builtin_call.op.name
        index_list = 0  # Assume single output for now
      elif len(collected) > 1:
        builtin_calls = collected
        op_name = tuple([c.op.name for c in collected])
        index_list = tuple([0 for _ in collected])

      if isinstance(op_name, tuple): # output is tuple
        required_layouts_per_output = {key:[] for key in op_name} # container of layout candidates per tuple field
        for op_name_, index_, builtin_call_ in zip(op_name, index_list, builtin_calls):
          # rules = REQUIRED_OP_LAYOUTS.get(op_name_, None)
          rules = get_required_layout_rules(builtin_call_, cpu_node=cpu_node)
          if rules is None:
            raise ValueError(f"Layout requirement not defined for op {op_name_}")
          
          for rule in rules:
            inputs_options, outputs_layout = rule
            if outputs_layout not in required_layouts_per_output[op_name_]:
              required_layouts_per_output[op_name_].append(outputs_layout)
        
        required_layouts = []
        for comb in itertools.product(*required_layouts_per_output.values()):
          layout_candidate = tuple(comb)
          required_layouts.append(layout_candidate)
          
        return required_layouts
      else:
        # rules = REQUIRED_OP_LAYOUTS.get(op_name, None)
        rules = get_required_layout_rules(builtin_call, cpu_node=cpu_node)

        if rules is None:
          raise ValueError(f"Layout requirement not defined for op {op_name}")

        required_layouts_ = []
        for rule in rules:
          inputs_options, outputs_layout = rule
          if outputs_layout not in required_layouts_:
            required_layouts_.append(outputs_layout)
        
        return required_layouts_
    else:
      raise ValueError(f"Unknown io_kind {io_kind}")
  else:
    raise ValueError("call.op must be a tvm.ir.Op or relay.Function")

def get_required_layout_rules(call, cpu_node=False):
  """
  get layout requirement rule
  """
  assert isinstance(call, relay.Call) and isinstance(call.op, tvm.ir.Op), "call must be relay.Call with built-in op."
  op_name = call.op.name
  call_attr = call.attrs

  rules = None
  if op_name == "layout_transform":
    src_layout = LayoutType(call_attr.src_layout)
    dst_layout = LayoutType(call_attr.dst_layout)
    rules = [(
      [[src_layout]],
      dst_layout,
    )]
  elif op_name in REQUIRED_OP_LAYOUTS.keys():
    rules = REQUIRED_OP_LAYOUTS.get(op_name, None)
    if rules is None:
      raise ValueError(f"Layout requirement not defined for op {op_name}")
  else:
    debug_print(f"[get_required_layout_rules] op_name {op_name} not found in REQUIRED_OP_LAYOUTS.")
    debug_print(f"[get_required_layout_rules] fallback NCHW layout.")
    in_num = len(call.args)
    rules = [(
      [[LayoutType.NCHW] * in_num],
      LayoutType.NCHW,
    )]
  
  if not rules: raise ValueError(f"Layout requirement not defined for op {op_name} @ before cpu_node filter.")

  if not cpu_node:
    new_rules = []
    for rule in rules:
      new_rule = deepcopy(rule)
      delete_input_options = []
      inputs_options, outputs_layout = rule
      if LayoutType.NCHW == outputs_layout:
        continue
      for input_option in inputs_options:
        if LayoutType.NCHW in input_option:
          delete_input_options.append(input_option)
      
      for input_option in delete_input_options:
        inputs_options.remove(input_option)
      if inputs_options:
        new_rules.append((inputs_options, outputs_layout))
    if new_rules:
      rules = new_rules
  else:
    new_rules = []
    for rule in rules:
      new_rule = deepcopy(rule)
      delete_input_options = []
      inputs_options, outputs_layout = rule
      if outputs_layout not in [LayoutType.NCHW, LayoutType.SCALAR, LayoutType.MK, LayoutType.C]:
        continue
      for input_option in inputs_options:
        if any([layout not in [LayoutType.NCHW, LayoutType.SCALAR, LayoutType.MK, LayoutType.C] for layout in input_option]):
          delete_input_options.append(input_option)
      
      for input_option in delete_input_options:
        inputs_options.remove(input_option)
      if inputs_options:
        new_rules.append((inputs_options, outputs_layout))
    if new_rules:
      rules = new_rules
  
  if not rules: raise ValueError(f"Layout requirement not defined for op {op_name} @ after cpu_node filter.")
  return rules

def get_valid_output_layout_of_node(node, input_layouts, mod, cpu_node=False):
  """
  get valid output layout of call node based on input layouts.
  call node can be built-in op or composite function or global var.
  If call is built-in op, use REQUIRED_OP_LAYOUTS to get output layout.
  if call is composite function or global var, we apply input layouts to function params and
  propagate layouts inside function body to get output layout.
  """

  debug_print(f"[get_valid_output_layout_of_node] node: {getNodeDebugID(node)}, input_layouts: {input_layouts}. cpu_node: {cpu_node}")

  # if not isinstance(node, (relay.Call, relay.Function)): raise ValueError("node must be relay.Call or relay.Function")

  if isinstance(node, relay.Call) and isinstance(node.op, tvm.ir.Op):
    def _layout_match(actual, expected):
      if isinstance(actual, (tuple, list)):
        return all(_layout_match(a, expected) for a in actual)
      return actual == expected

    op_name = node.op.name
    # rules = REQUIRED_OP_LAYOUTS.get(op_name, None)
    rules = get_required_layout_rules(node, cpu_node=cpu_node)
    if rules is None: raise ValueError(f"Layout requirement not defined for op {op_name}")

    valid_outputs_layout = None
    for rule in rules:
      inputs_options, outputs_layout = rule
      for option in inputs_options:
        _option = option
        _inputs = input_layouts

        is_tuple_input = False
        if len(option) == 1 and len(input_layouts) == 1 and isinstance(input_layouts[0], tuple):
          _inputs = list(input_layouts[0])
          _option = [option[0]] * len(_inputs)
          is_tuple_input = True

        # check if input_layouts match option
        if len(_option) == len(_inputs):
          if not is_tuple_input:
            arg_types = [_get_type(mod, arg) for arg in node.args]
          else:
            arg_types = _get_type(mod, node.args[0]).fields # input of concat
          match = True
          for i in range(len(_inputs)):
            if (not _layout_match(_inputs[i], _option[i])) or (not is_layout_compatible_with_type(_inputs[i], arg_types[i])):
              match = False
              break
          if match:
            if valid_outputs_layout: raise ValueError("multiple valid output layouts found")
            valid_outputs_layout = outputs_layout
    
    debug_print(f"[get_valid_output_layout_of_node] node: {getNodeDebugID(node)}, valid_outputs_layout: {valid_outputs_layout}")
    return valid_outputs_layout
  elif (isinstance(node, relay.Call) and (isinstance(node.op, relay.Function) or isinstance(node.op, relay.GlobalVar))) or (isinstance(node, relay.Function)):
    if isinstance(node, relay.Function):
      func = node
    else:
      if isinstance(node.op, relay.GlobalVar):
        func = mod[node.op.name_hint]
      else:
        func = node.op

    # topological sort of function body to visit nodes in order
    use_def = UseDefChainParser()
    use_def.visit(func.body)
    call_nodes_topological = use_def.topological_call_order(call_only=False)

    #- initalize layout dict for function params
    layout_dict = {param: layout for param, layout in zip(func.params, input_layouts)}

    def _get_layout(expr, call=None, idx=None, not_const_layouts=None):
      if isinstance(expr, relay.Constant):
        assert call is not None and idx is not None, "call and idx must be provided for constant layout deduction."
        assert not_const_layouts is not None, "not_const_layouts must be provided for constant layout deduction."
        debug_print(f"[get_valid_output_layout_of_node] node: {getNodeDebugID(node)}, constant layout deduced.")
        return _deduce_layout_from_op_const(call, idx, not_const_layouts)
      else:
        assert expr in layout_dict, "input layout not found for expr in layout dict."
        return layout_dict[expr]

    for _node in call_nodes_topological:
      node_input_layouts = []
      if isinstance(_node, relay.Var):
        continue
      elif isinstance(_node, relay.Call):
        const_args_indices = [i for i, arg in enumerate(_node.args) if isinstance(arg, relay.Constant)]
        args_sorted = sorted(range(len(_node.args)), key=lambda k: k in const_args_indices)
        not_const_layouts = []
        for idx in args_sorted:
          arg = _node.args[idx]
          layout = _get_layout(arg, _node, idx, not_const_layouts=not_const_layouts)
          node_input_layouts.append(layout)
          if not isinstance(arg, relay.Constant):
            not_const_layouts.append(layout)

      elif isinstance(_node, relay.Tuple):
        for field in _node.fields:
          node_input_layouts.append(_get_layout(field))
      elif isinstance(_node, relay.TupleGetItem):
        assert isinstance(_node.tuple_value, relay.Tuple) or (isinstance(_node.tuple_value, relay.Call) and _node.tuple_value.op == op.get("split")), "tuple get item input must be tuple or split call."
        tuple_layout = _get_layout(_node.tuple_value)
        if isinstance(_node.tuple_value, relay.Tuple):
          node_input_layouts.append(tuple_layout[_node.index])
        else:
          node_input_layouts.append(tuple_layout)
      else:
        continue

      #- get output layout
      is_multi_candidate = any(isinstance(l, list) for l in node_input_layouts)
      if not is_multi_candidate:
        node_output_layout = get_valid_output_layout_of_node(_node, node_input_layouts, mod, cpu_node=cpu_node)
      else:
        # multiple input layout candidates, try all combinations
        input_layouts_candidates = []
        for l in node_input_layouts:
          if isinstance(l, list):
            input_layouts_candidates.append(l)
          else:
            input_layouts_candidates.append([l])
        
        output_layout_candidates = set()
        for comb in itertools.product(*input_layouts_candidates):
          out_layout = get_valid_output_layout_of_node(_node, list(comb), mod, cpu_node=cpu_node)
          if out_layout:
            output_layout_candidates.add(out_layout)
        
        if len(output_layout_candidates) == 1:
          node_output_layout = output_layout_candidates.pop()
        else:
          debug_print(f"[get_valid_output_layout_of_node] node: {getNodeDebugID(node)}, multiple output layout candidates found: {output_layout_candidates}.")
          node_output_layout = output_layout_candidates[-1]

      if node_output_layout is None:
        return None
      layout_dict[_node] = node_output_layout
    
    def _resolve(expr):
      if isinstance(expr, relay.Tuple):
        return tuple(_resolve(f) for f in expr.fields)
      if isinstance(expr, relay.TupleGetItem):
        assert isinstance(_node.tuple_value, relay.Tuple) or (isinstance(_node.tuple_value, relay.Call) and _node.tuple_value.op == op.get("split")), "tuple get item input must be tuple or split call."
        tuple_layout = _get_layout(_node.tuple_value)
        if isinstance(_node.tuple_value, relay.Tuple):
          return tuple_layout[expr.index]
        else:
          return tuple_layout
      return _get_layout(expr)

    output = _resolve(func.body) 
    debug_print(f"[get_valid_output_layout_of_node] node: {getNodeDebugID(node)}, output layout: {output}")
    return output
  elif isinstance(node, relay.Tuple):
    debug_print(f"[get_valid_output_layout_of_node] node: {getNodeDebugID(node)}, output layout: {tuple(input_layouts)}")
    return tuple(input_layouts)
  elif isinstance(node, relay.TupleGetItem):
    if isinstance(input_layouts, (list, tuple)):
      if len(input_layouts) == 1:
        debug_print(f"[get_valid_output_layout_of_node] node: {getNodeDebugID(node)}, output layout: {input_layouts[0]}")
        return input_layouts[0]
      if len(input_layouts) > node.index:
        debug_print(f"[get_valid_output_layout_of_node] node: {getNodeDebugID(node)}, output layout: {input_layouts[node.index]}")
        return input_layouts[node.index]
    debug_print(f"[get_valid_output_layout_of_node] node: {getNodeDebugID(node)}, output layout: {input_layouts}")
    return input_layouts

def getNodeID(node) -> int:
  id_dict = HashToCustomID()
  if int(hash(node)) in id_dict:
    return id_dict[int(hash(node))]
  else:
    return None

def getNodeDebugID(node):
  if isinstance(node, relay.Call):
    if isinstance(node.op, tvm.ir.Op):
      indicator = str(node.op.name)
    elif isinstance(node.op, relay.Function) and "Composite" in node.op.attrs: # composite node
      indicator = str(node.op.attrs["Composite"])
    elif isinstance(node.op, relay.GlobalVar):
      indicator = str(node.op.name_hint)
    else:
      indicator = "imcflow_func_impl"
  elif isinstance(node, relay.Function):
    if "Composite" in node.attrs:
      indicator = str(node.attrs["Composite"])
    else:
      indicator = "func"
  else:
    node_id = getNodeID(node)
    if node_id is not None:
      indicator = f"node_{node_id}"
    else:
      indicator = "node_unknown"
  return indicator

def getInnerNodeID(node):
  if isinstance(node, tuple):
    return node[1]
  else:
    return node

def getOuterNodeID(node):
  if isinstance(node, tuple):
    return node[0]
  else:
    return node

def _get_type(parent_mod, node):
    """A method to infer the type of a relay expression."""

    try:
      out_type = node.checked_type
      debug_print(f"node {getNodeDebugID(node)} already has checked_type: {out_type}")
      return out_type
    except:
      debug_print(f"node {getNodeDebugID(node)} has no checked_type, inferring...")
      pass

    if isinstance(node, relay.Call) and isinstance(node.op, tvm.ir.Op):
      out_type = relay.transform.InferTypeLocal(node)
    elif isinstance(node, relay.Call) and isinstance(node.op, relay.Function):
      # out_type = node.op.body.checked_type
      out_type = relay.transform.InferTypeLocal(node.op.body)
    elif isinstance(node, relay.Call) and isinstance(node.op, relay.GlobalVar):
      out_type = _get_type(parent_mod, parent_mod[node.op.name_hint].body)
    elif isinstance(node, relay.Function):
      out_type = relay.transform.InferTypeLocal(node.body)
    elif isinstance(node, relay.Var):
      # out_type = node.checked_type
      out_type = node.type_annotation
    elif isinstance(node, relay.TupleGetItem):
      # For TupleGetItem, get the type of the tuple and extract the field type
      tuple_type = _get_type(parent_mod, node.tuple_value)
      if isinstance(tuple_type, relay.TupleType):
        out_type = tuple_type.fields[node.index]
      else:
        raise RuntimeError(f"TupleGetItem node has non-tuple parent type: {tuple_type}")
    elif isinstance(node, relay.Tuple):
      # For Tuple, infer the type of each field and construct a TupleType
      field_types = [_get_type(parent_mod, field) for field in node.fields]
      out_type = relay.TupleType(field_types)
    elif isinstance(node, relay.Constant):
      out_type = node.checked_type
    else:
      raise RuntimeError(f"can't infer type for node {node}")

    debug_print("----------------------------------------------------")
    debug_print(f"[_get_type] node {getNodeDebugID(node)} -> out_type: {out_type}")
    debug_print("----------------------------------------------------")
    return out_type

def getInputNodesOfFunc(func):
  InNodes = []

  class _Visitor(tvm.relay.ExprVisitor):
    def visit_function(self, func):
      for param in func.params:
        InNodes.append(param)

  _Visitor().visit(func)
  return InNodes

class NodeCollector(relay.ExprVisitor):
  """
  Collects expr nodes matching given predicates (default: all).

  Parameters
  ----------
  predicates : list of callables(ctx, expr) -> bool
    If empty, all nodes are collected. If provided, a node is collected if any predicate returns True.
  ctx : any (optional)
    Carried into predicates; defaults to internal dict with traversal flags.
  """
  def __init__(self, predicates=None):
    super().__init__()
    self.ctx = {
      "in_sub_func" : False,
      "sub_func_level" : 0,
      "stack" : []
    }
    self.predicates = predicates or []
    self.collected = []

  def _should_collect(self, expr):
    if not self.predicates:
      return True
    return any(pred(self.ctx, expr) for pred in self.predicates)

  def visit_call(self, call):
    # Collect current call if it matches predicates
    if self._should_collect(call):
      self.collected.append(call)
    
    self.ctx["stack"].append(call)

    # If call is a composite function, traverse into its body
    if isinstance(call.op, relay.Function):
      self.ctx["in_sub_func"] = True
      self.ctx["sub_func_level"] += 1
      self.visit(call.op.body)
      self.ctx["sub_func_level"] -= 1
      if self.ctx["sub_func_level"] == 0:
        self.ctx["in_sub_func"] = False
    else:
      super().visit_call(call)
    
    self.ctx["stack"].pop()

  def visit_var(self, var):
    if self._should_collect(var):
      self.collected.append(var)
    
    self.ctx["stack"].append(var)
    super().visit_var(var)
    self.ctx["stack"].pop()

  def visit_tuple(self, tup):
    if self._should_collect(tup):
      self.collected.append(tup)
    
    self.ctx["stack"].append(tup)
    super().visit_tuple(tup)
    self.ctx["stack"].pop()

  def visit_tuple_getitem(self, tgi):
    if self._should_collect(tgi):
      self.collected.append(tgi)

    self.ctx["stack"].append(tgi)
    super().visit_tuple_getitem(tgi)
    self.ctx["stack"].pop()

  def collect(self, expr, ctx=None):
    if ctx is not None:
      self.ctx = ctx
    self.collected = []
    self.visit(expr)
    return self.collected

def getConstNodesOfFunc(func):
  InNodes = []

  class _Visitor(tvm.relay.ExprVisitor):
    def visit_constant(self, const):
      InNodes.append(const)
      super().visit_constant(const)

  _Visitor().visit(func)
  return InNodes

class UseDefChainParser(relay.ExprVisitor):
  """
  Parse Use-Def chain for expressions.
  Builds a mapping from each expression to its users (consumers).
  """
  def __init__(self):
    super().__init__()
    self.users = {}  # {expr : [list of (user_call, arg_index)]}
    self.uses = {}  # {expr : [list of operand exprs]}
    self.call_nodes = []
    self._call_set = set()

  def _record_use(self, user, operand, tag):
    if operand not in self.users:
      self.users[operand] = []
    self.users[operand].append((user, tag))

  def _register_uses(self, user, operands, tag_fn=None):
    self.uses[user] = list(operands)
    for idx, operand in enumerate(operands):
      tag = tag_fn(idx, operand) if tag_fn else idx
      self._record_use(user, operand, tag)

  def visit_call(self, call):
    if call not in self._call_set:
      self._call_set.add(call)
      self.call_nodes.append(call)
    self._register_uses(call, call.args)

    # Continue visiting child nodes
    for arg in call.args:
      self.visit(arg)

  def visit_tuple(self, tup):
    # Record tuple field usage
    self._register_uses(tup, tup.fields)

    super().visit_tuple(tup)

  def visit_tuple_getitem(self, tgi):
    # Record TupleGetItem usage
    self._register_uses(
      tgi,
      [tgi.tuple_value],
      tag_fn=lambda idx, _: tgi.index if idx == 0 else idx,
    )

    super().visit_tuple_getitem(tgi)

  def get_users(self, expr):
    """Get all users (consumers) of an expression"""
    return self.users.get(expr, [])

  def get_uses(self, expr):
    """Get operands (dependencies) of an expression"""
    return self.uses.get(expr, [])

  def _find_call_inputs(self, call):
    deps = set()
    stack = list(self.uses.get(call, []))
    while stack:
      expr = stack.pop()
      if isinstance(expr, relay.Call):
        deps.add(expr)
      elif isinstance(expr, relay.TupleGetItem):
        stack.append(expr.tuple_value)
      elif isinstance(expr, relay.Tuple):
        stack.extend(expr.fields)
    return deps

  def topological_call_order(self, call_only=False):
    """
    Return nodes in topological order (producers before consumers).
    If call_only is True, only call nodes are included. Otherwise tuples,
    tuple_get_item, and var nodes are also part of the ordering.
    """
    if call_only:
      nodes = list(self.call_nodes)
      dep_fn = lambda n: self._find_call_inputs(n)
    else:
      node_set = set(self.call_nodes)
      for expr, deps in self.uses.items():
        if isinstance(expr, (relay.Tuple, relay.TupleGetItem, relay.Var)):
          node_set.add(expr)
        for dep in deps:
          if isinstance(dep, (relay.Call, relay.Tuple, relay.TupleGetItem, relay.Var)):
            node_set.add(dep)
      nodes = list(node_set)
      dep_fn = lambda n: [
        op for op in self.uses.get(n, [])
        if isinstance(op, (relay.Call, relay.Tuple, relay.TupleGetItem, relay.Var))
      ]

    indegree = {n: 0 for n in nodes}
    adj = collections.defaultdict(list)

    for node in nodes:
      for dep in dep_fn(node):
        adj[dep].append(node)
        indegree[node] += 1

    queue = collections.deque([n for n, deg in indegree.items() if deg == 0])
    ordered = []

    while queue:
      curr = queue.popleft()
      ordered.append(curr)
      for nxt in adj.get(curr, []):
        indegree[nxt] -= 1
        if indegree[nxt] == 0:
          queue.append(nxt)

    if len(ordered) != len(nodes):
      raise ValueError("Cycle detected in node graph")
    return ordered

class LayoutPropagationContext:
  """
  Helper to build candidate layout combinations for function variables using use-def chains.

  The context parses the use-def chain of the function body and walks from inputs toward outputs
  to gather all possible required layouts for each parameter. It returns the Cartesian product
  of these per-parameter candidate sets.
  """
  def __init__(self, func):
    self.func = func
    self.use_def_chain = UseDefChainParser()
    self.use_def_chain.visit(func.body)

  def build_var_layout_combinations(self):
    candidate_lists = []
    for param in self.func.params:
      layouts = self._collect_candidate_layouts(param, self.use_def_chain, set())
      if not layouts:
        fallback = LayoutType.NCHW
        layouts = {fallback}
      candidate_lists.append(sorted(list(layouts), key=lambda l: l.name))

    combinations = []
    for product in itertools.product(*candidate_lists):
      combo = {}
      for param, layout in zip(self.func.params, product):
        combo[param] = layout
      combinations.append(combo)
    return combinations

  def _collect_candidate_layouts(self, expr, parser, visited):
    key = (expr, id(parser))
    if key in visited:
      return set()
    visited.add(key)

    layouts = set()
    for user, arg_index in parser.get_users(expr):
      if isinstance(user, relay.Call):
        if isinstance(user.op, tvm.ir.Op):
          layouts.update(self._layouts_from_op(user, arg_index))
        elif isinstance(user.op, relay.Function):
          inner_parser = UseDefChainParser()
          inner_parser.visit(user.op.body)
          param_var = user.op.params[arg_index]
          layouts.update(self._collect_candidate_layouts(param_var, inner_parser, visited))
        elif isinstance(user.op, relay.GlobalVar):
          # Without module context, conservatively fall back to default layout later.
          continue
      elif isinstance(user, relay.TupleGetItem):
        layouts.update(self._collect_candidate_layouts(user, parser, visited))
      elif isinstance(user, relay.Tuple):
        layouts.update(self._collect_candidate_layouts(user, parser, visited))
    return layouts

  def _layouts_from_op(self, call, arg_index):
    layouts = set()
    rules = None
    if isinstance(call.op, tvm.ir.Op):
      # rules = REQUIRED_OP_LAYOUTS.get(call.op.name, None)
      rules = get_required_layout_rules(call)
    elif isinstance(call.op, relay.Function):
      inner_call = self._find_first_builtin_call(call.op.body)
      if inner_call is None or not isinstance(inner_call.op, tvm.ir.Op):
        raise ValueError("Composite function does not contain builtin call for layout deduction")
      # rules = REQUIRED_OP_LAYOUTS.get(inner_call.op.name, None)
      rules = get_required_layout_rules(inner_call)
    if rules is None:
      raise ValueError(f"Layout requirement not defined for op {call.op}")
    for inputs_options, _ in rules:
      for pattern in inputs_options:
        if len(pattern) == 1:
          candidate = pattern[0]
        elif arg_index < len(pattern):
          candidate = pattern[arg_index]
        else:
          continue
        if candidate is not None:
          layouts.add(candidate)
    return layouts

class ConsumerFinder:
  """
  Find consumers of a node and arg idx for the node. consumer is Call node.
  If current target node is A, and consumer node is B and B = func(A, C), then return (B, 0) where 0 is the arg index of A in B.

  parameters:
    - use_def_chain_parser: helper to query direct users of an expr
    - skip_predicates     : list of callables(call, arg_index) -> bool.
                            If any predicate returns True for a consumer call,
                            we skip recording that call and recurse downstream.

  returns:
    - list of (consumer_call, arg_index)
  """
  def __init__(self, use_def_chain_parser, skip_predicates=None):
    self.use_def_chain_parser = use_def_chain_parser
    self.skip_predicates = skip_predicates or []
    self.visited = set()  # Track visited expressions to avoid cycles

  def find_consumers_of_node(self, node):
    """Find all final consumers for an expr node (skipping via predicate)"""
    self.visited.clear()
    consumers_found = []
    self._find_consumers_recursive(node, consumers_found)
    return consumers_found

  def _find_consumers_recursive(self, expr, consumers):
    """Recursively find consumers, skipping element-wise operations"""

    # Avoid infinite loops
    if expr in self.visited:
      return
    self.visited.add(expr)

    # Get direct users of this expression
    users = self.use_def_chain_parser.get_users(expr)

    for user, arg_index in users:
      if isinstance(user, relay.Call):
        should_recurse = any(pred(user, arg_index) for pred in self.skip_predicates) if self.skip_predicates else False
        if should_recurse:
          before = len(consumers)
          if isinstance(user.op, relay.Function):
            _recurse_use_def = UseDefChainParser()
            _recurse_use_def.visit(user.op.body)
            recursive_consumer_finder = ConsumerFinder(_recurse_use_def, self.skip_predicates)
            param_var = user.op.params[arg_index]
            param_consumers = recursive_consumer_finder.find_consumers_of_node(param_var)
            for cons in param_consumers:
              consumers.append(cons)
          else:
            # Skip recording this call and continue downstream
            self._find_consumers_recursive(user, consumers)
          # If skipping did not yield any downstream consumers (e.g., leaf element-wise op),
          # keep this call as the consumer so the variable is not lost.
          if len(consumers) == before:
            consumers.append((user, arg_index))
        else:
          if isinstance(user.op, tvm.ir.Op):
            op_name = user.op.name
          elif isinstance(user.op, relay.GlobalVar):
            op_name = user.op.name_hint
          else:
            op_name = None
          consumers.append((user, arg_index))
      elif isinstance(user, relay.TupleGetItem):
        self._find_consumers_recursive(user, consumers)
      elif isinstance(user, relay.Tuple):
        self._find_consumers_recursive(user, consumers)
      else:
        raise ValueError("Unsupported type")

class ProducerFinder:
  """
  Find producers for an expression.

  Returns list of (producer_call, output_index).
  If the producer is a TupleGetItem path, the output_index is the tuple index used.

  parameters:
    - skip_predicates: list of callables(call, output_index) -> bool.
                       If any predicate returns True for a producer Call, we skip
                       recording that call and recurse into its inputs instead.
  """
  def __init__(self, skip_predicates=None):
    self.skip_predicates = skip_predicates or []
    self.visited = set()

  def find_producers_of_node(self, node):
    self.visited.clear()
    producers = []
    self._find_producers_recursive(node, [], producers)
    return producers

  def _normalize_index_for_key(self, out_index):
    if isinstance(out_index, list):
      return tuple(out_index)
    return out_index

  def _push_index(self, current_index, new_index):
    if isinstance(current_index, list):
      stack = list(current_index)
    elif current_index == 0:
      stack = []
    else:
      stack = [current_index]
    stack.append(new_index)
    return stack

  def _materialize_index(self, index_stack):
    if isinstance(index_stack, list):
      if len(index_stack) == 0:
        return 0
      if len(index_stack) == 1:
        return index_stack[0]
      return index_stack
    return index_stack

  def _find_producers_recursive(self, expr, out_index, producers):
    key = (expr, self._normalize_index_for_key(out_index))
    if key in self.visited:
      return
    self.visited.add(key)

    if isinstance(expr, relay.TupleGetItem):
      # Track tuple index as output index
      next_index = self._push_index(out_index, expr.index)
      self._find_producers_recursive(expr.tuple_value, next_index, producers)
    elif isinstance(expr, relay.Tuple):
      # Traverse every tuple field, tracking the nested output indices like a stack
      for idx, field in enumerate(expr.fields):
        next_index = self._push_index(out_index, idx)
        self._find_producers_recursive(field, next_index, producers)
    elif isinstance(expr, relay.Call):
      materialized_index = self._materialize_index(out_index)
      # If calling a relay.Function, recurse into its body to find real producers
      if isinstance(expr.op, relay.Function):
        self._find_producers_recursive(expr.op.body, out_index, producers)
        return

      should_recurse = any(pred(expr, materialized_index) for pred in self.skip_predicates) if self.skip_predicates else False
      if should_recurse:
        # Skip this call and recurse into its inputs
        for arg in expr.args:
          self._find_producers_recursive(arg, out_index, producers)
      else:
        producers.append((expr, materialized_index))
    else:
      # For vars/constants etc., no Call producer to record
      return

def getOutputNodeOfFunc(func):
  return func.body

def isImcflowFunc(func, mod):
  if isinstance(func, relay.Function):
    if "Compiler" in func.attrs and func.attrs["Compiler"] == "imcflow":
      return True
    else:
      return False
  elif isinstance(func, relay.GlobalVar):
    target_func = mod[func.name_hint]
    if "Compiler" in target_func.attrs and target_func.attrs["Compiler"] == "imcflow":
      return True
    else:
      return False
  else:
    return False

def get_imcflow_supported_regions(mod, include_first_conv=False):
  """
  Traverse the graph and find regions of imcflow-supported operators.
  A region is a list of consecutive nodes that are supported. This function
  finds the maximal connected subgraphs of supported operators.

  This function should be called with a module containing only the main function.

  Parameters
  ----------
  mod : tvm.IRModule
    The module to be processed.

  Returns
  -------
  list[list[tvm.relay.expr.Call]]
    A list of regions, where each region is a list of supported Call nodes.
  """
  # A set of imcflow-supported primitive operators.
  # This list should be updated based on the actual capabilities of the imcflow backend.
  #TODO: do we need it seperately?
  _SUPPORTED_OPS = {
    "nn.imcflow_qconv",
    'nn.imcflow_qdwconv',
    "qnn.imcflow_min_max_quantize",
    "imcflow.fused_batch_norm",
    "nn.relu",
    "nn.bias_add",
    "add",
    "multiply",
  }

  def is_first_conv(call):
    return isinstance(call.op, Op) and call.op.name == "nn.conv2d" and not meet_first_conv

  def is_supported(call):
    return isinstance(call.op, Op) and call.op.name in _SUPPORTED_OPS

  class NodeCollector(ExprVisitor):
    """Collects all call nodes in the expression."""

    def __init__(self):
      super().__init__()
      self.call_nodes = []

    def visit_call(self, call):
      super().visit_call(call)
      self.call_nodes.append(call)

  # 1. Run type inference to ensure checked_type is available, then collect call nodes.
  # typed_mod = relay.transform.InferType()(mod)
  typed_mod = mod
  main_func = typed_mod["main"]
  collector = NodeCollector()
  collector.visit(main_func)

  # Helper to determine if a dtype string is integer type
  def _is_int_dtype(dt: str) -> bool:
    return isinstance(dt, str) and (dt.startswith("int") or dt.startswith("uint"))

  # Fetch dtype for an expression if it is a TensorType; otherwise None
  def _expr_tensor_dtype(e):
    try:
      ty = e.checked_type
    except Exception:
      return None
    if isinstance(ty, relay.ty.TensorType):
      return ty.dtype
    return None

  # Check that all tensor inputs to a call are integer-typed
  def _inputs_are_int(call: Call) -> bool:
    for arg in call.args:
      # For tuples, check each field if tensor
      if isinstance(arg.checked_type, relay.ty.TupleType):
        # Tuple inputs are rare for these ops, but handle gracefully
        for field_ty in arg.checked_type.fields:
          if isinstance(field_ty, relay.ty.TensorType):
            if not _is_int_dtype(field_ty.dtype):
              return False
        continue
      dt = _expr_tensor_dtype(arg)
      if dt is None:
        # Non-tensor inputs (e.g., attrs) or unknown types are ignored
        continue
      if not _is_int_dtype(dt):
        return False
    return True

  # 2. Filter for supported nodes.
  supported_calls = []
  meet_first_conv = False
  for call in collector.call_nodes:
    if is_first_conv(call):
      meet_first_conv = True
      if not include_first_conv:
        continue
    if is_supported(call) and _inputs_are_int(call):
      supported_calls.append(call)
  supported_set = set(supported_calls)

  if not supported_calls:
    return []

  # 3. Build the graph of supported nodes.
  # Map each supported call node to a unique integer ID.
  node_to_id = {node: i for i, node in enumerate(supported_calls)}
  adj = [[] for _ in range(len(supported_calls))]

  memo = {}
  tuple_get_nodes = {}

  def get_producer(expr):
    """Trace back through expressions to find the producing supported call node."""
    if expr in memo:
      return memo[expr]

    if expr in supported_set:
      memo[expr] = expr
      return expr

    if isinstance(expr, TupleGetItem):
      producer = get_producer(expr.tuple_value)
      memo[expr] = producer
      return producer

    memo[expr] = None
    return None

  for i, call_node in enumerate(supported_calls):
    for arg in call_node.args:
      producer = get_producer(arg)
      if producer:
        # producer is already guaranteed to be in supported_set by get_producer
        j = node_to_id[producer]
        adj[i].append(j)
        adj[j].append(i)

        if isinstance(arg, TupleGetItem):
          # record related tuple get item node
          tuple_get_nodes[producer] = arg

  # 4. Find connected components (these are the maximal regions).
  regions = []
  visited = set()
  for i in range(len(supported_calls)):
    node = supported_calls[i]
    if node not in visited:
      component = []
      q = [node]
      visited.add(node)
      head = 0
      while head < len(q):
        u = q[head]
        head += 1
        component.append(u)
        if u in tuple_get_nodes:
          component.append(tuple_get_nodes[u])
        u_idx = node_to_id[u]
        for v_idx in adj[u_idx]:
          v = supported_calls[v_idx]
          if v not in visited:
            visited.add(v)
            q.append(v)
      regions.append(component)

  return regions

def partitionImcflowSubGraph(mod):
  mod = relay.transform.InferType()(mod)
  region_list = get_imcflow_supported_regions(mod)
  mod = imcflow.ImcflowAnnotationPass(region_list)(mod)
  mod = transform.MergeCompilerRegions()(mod)
  mod = imcflow.ImcflowCleanRegionTag()(mod)
  mod = transform.PartitionGraph()(mod)
  # mod = clearCompilerAttr(mod)
  # mod = clearPrimitiveTag(mod)
  return mod

def split_conv_to_atomic(mod, OldParamDict):
    class Worker:
      def __init__(self, OldParamDict):
        self.OldParamDict = OldParamDict
        self.NewParamDict = {}

      def transform_function(self, func, mod):
        class _RedundantTupleRemover(tvm.relay.ExprMutator):
          def __init__(self):
            super().__init__()

          def visit_tuple_getitem(self, op):
            TupleValue = op.tuple_value
            if isinstance(TupleValue, relay.Tuple):
              if len(TupleValue.fields) == 1:
                return super().visit(TupleValue.fields[0])
              else:
                return super().visit_tuple_getitem(op)
            else:
              return super().visit_tuple_getitem(op)

        class Spliter(tvm.relay.ExprMutator):
          """Split large conv2d into smaller conv2d, split, concat, add, etc"""

          def __init__(self, OldParamDict):
            super().__init__()
            self.OldParamDict = OldParamDict
            self.NewParamDict = {k:v for k,v in OldParamDict.items()}
            self.DeleteArgs = []
            self.AddArgs = []
            self.PostProcess = []
            # self.IsSplitedPostNode = []

          def removeSplitedArg(self, node):
            if isinstance(node, relay.Var):
              self.NewParamDict.pop(node.name_hint)
            self.DeleteArgs.append(node)

          def addParamVar(self, Var, Data):
            self.NewParamDict[Var.name_hint] = Data
            self.AddArgs.append(Var)

          def split_and_optimize_conv2d(self, expr, mod, PostProcess):

            def _get_type(node):
                """A method to infer the type of a relay expression."""
                mod = tvm.IRModule.from_expr(node)
                mod = relay.transform.InferType()(mod)
                entry = mod["main"]

                infer_out = entry if isinstance(node, relay.Function) else entry.body
                out_type = infer_out._checked_type_

                if isinstance(out_type, TensorType):
                    # Single tensor, get the shape directly
                    shapes = [int(dim) for dim in out_type.shape]
                elif isinstance(out_type, TupleType):
                    # Tuple of tensors, get the shape of each tensor in the tuple
                    shapes = [int(field) for field in out_type.fields]
                else:
                    raise RuntimeError(f"Unsupported output type {type(out_type)} in operator {node.op.name}")

                return shapes

            # Extract input and kernel shapes
            _, IC, IH, IW = _get_type(expr.args[0])  # Input shape
            OC, _, KH, KW = _get_type(expr.args[1])  # Kernel shape
            padding = expr.attrs.padding
            strides = expr.attrs.strides

            if not ImcflowDeviceConfig.is_supported_kernel(KH, KW):
              return expr

            #TODO: add, multiply can be here. but one operand should constant (adjust scaling)
            for PostNode in PostProcess:
              assert PostNode.op in [op.get("nn.bias_add"), op.get("nn.relu"), op.get("imcflow.fused_batch_norm"), op.get("divide"),
                                    op.get("qnn.imcflow_min_max_quantize"), op.get("qnn.imcflow_nu_quantize")], "Unsupported post process node"

            groups = expr.attrs.groups
            assert (groups == 1 or groups == IC), "Grouped convolutions are not supported"

            IsDepthWise = (groups == IC)

            # Set limits for in and out channels
            in_ch_limit = math.floor(256 / (KH * KW)) if not IsDepthWise else 32
            out_ch_limit = 64 if not IsDepthWise else 32

            if (IC <= in_ch_limit) and (OC <= out_ch_limit):
                return expr  # Return original if no splitting is needed

            # Determine split counts
            ic_split_num = math.ceil(IC / in_ch_limit)
            oc_split_num = math.ceil(OC / out_ch_limit)
            IsICSplited = ic_split_num > 1
            IsOCSplited = oc_split_num > 1

            # Split the input and weights
            ic_sections = [i*in_ch_limit for i in range(1, ic_split_num)]
            oc_sections = [i*out_ch_limit for i in range(1, oc_split_num)]

            # input splitting
            split_inputs = relay.op.transform.split(expr.args[0], indices_or_sections=ic_sections, axis=1) if IsICSplited else [expr.args[0]]

            # split weight and make New params
            split_conv_weights = [[None for _ in range(ic_split_num if (not IsDepthWise) else 1)] for _ in range(oc_split_num)]
            if isinstance(expr.args[1], relay.Var):
              self.removeSplitedArg(expr.args[1])
            for oc_id in range(oc_split_num):
              oc_size = out_ch_limit if (oc_id * out_ch_limit) + out_ch_limit - 1 < OC else OC % out_ch_limit
              for ic_id in range(ic_split_num if not IsDepthWise else 1):
                if IsDepthWise:
                  ic_size = 1
                else:
                  ic_size = in_ch_limit if (ic_id * in_ch_limit) + in_ch_limit - 1 < IC else IC % in_ch_limit

                if isinstance(expr.args[1], relay.Var):
                  SplitParam = relay.Var(f"{expr.args[1].name_hint}_oc{oc_id}_ic{ic_id}", relay.TensorType([oc_size, ic_size, KH, KW], dtype=expr.args[1].type_annotation.dtype))
                elif isinstance(expr.args[1], relay.Constant):
                  nd_array = expr.args[1].data.numpy()[oc_id*out_ch_limit:(oc_id*out_ch_limit)+oc_size, ic_id*in_ch_limit:(ic_id*in_ch_limit)+ic_size, :, :]
                  SplitParam = relay.Constant(tvm.nd.array(nd_array))
                else:
                  raise RuntimeError("Unsupported weight node type for splitting")

                split_conv_weights[oc_id][ic_id] = SplitParam

                if isinstance(expr.args[1], relay.Var):
                  OldParam = self.OldParamDict[expr.args[1].name_hint]
                  if isinstance(OldParam, tvm.nd.NDArray):
                    NewData = OldParam.numpy()[oc_id*out_ch_limit:(oc_id*out_ch_limit)+oc_size, ic_id*in_ch_limit:(ic_id*in_ch_limit)+ic_size, :, :]
                    self.addParamVar(SplitParam, tvm.nd.array(NewData, device=OldParam.device))
                  else:
                    NewData = OldParam[oc_id*out_ch_limit:(oc_id*out_ch_limit)+oc_size, ic_id*in_ch_limit:(ic_id*in_ch_limit)+ic_size, :, :]
                    self.addParamVar(SplitParam, tvm.nd.array(NewData))

            # Create conv2d calls for each input-output channel slice
            conv_nodes = {}
            for oc_id in range(oc_split_num):
                oc_size = out_ch_limit if (oc_id * out_ch_limit) + out_ch_limit - 1 < OC else OC % out_ch_limit
                for ic_id in range(ic_split_num if not IsDepthWise else 1):
                    ic_size = in_ch_limit if (ic_id * in_ch_limit) + in_ch_limit - 1 < IC else IC % in_ch_limit

                    # Get input shape for this slice
                    input_node = split_inputs[ic_id] if (not IsDepthWise) else split_inputs[oc_id]
                    N, IC_slice, IH_slice, IW_slice = _get_type(input_node)

                    # Create config data
                    from tvm.relay.backend.contrib.imcflow.acim_util import ConfigData
                    config_data = ConfigData(
                        data_shape=(N, IC_slice, IH_slice, IW_slice),
                        weight_shape=(oc_size, ic_size, KH, KW),
                        padding=padding[0] if isinstance(padding, (list, tuple)) else padding,
                        stride=strides[0] if isinstance(strides, (list, tuple)) else strides
                    )

                    if not IsDepthWise:
                      conv_nodes[(oc_id, ic_id)] = imcflow_qconv2d(
                        input_node,
                        split_conv_weights[oc_id][ic_id],
                        config_data.get_as_const_tensor(),
                        in_channels=ic_size,
                        channels=oc_size,
                        kernel_size=(KH, KW),
                        padding=padding,
                        strides=strides,
                        groups=1,
                        out_dtype="int16"
                      )
                    else:
                      conv_nodes[(oc_id, ic_id)] = imcflow_qdwconv2d(
                        input_node,
                        split_conv_weights[oc_id][ic_id],
                        config_data.get_as_const_tensor(),
                        in_channels=1,
                        channels=oc_size,
                        kernel_size=(KH, KW),
                        padding=padding,
                        strides=strides,
                        groups=oc_size,
                        out_dtype="int16"
                      )

            # If input channels were split, sum the resulting conv2d outputs for each out channel slice
            if IsICSplited and (not IsDepthWise):
                add_nodes = {}
                for oc_id in range(oc_split_num):
                    add_nodes[oc_id] = conv_nodes[(oc_id, 0)]
                    for ic_id in range(1, ic_split_num):
                        add_nodes[oc_id] = relay.op.add(add_nodes[oc_id], conv_nodes[(oc_id, ic_id)])
            else:
                add_nodes = {oc_id: conv_nodes[(oc_id, 0)] for oc_id in range(oc_split_num)}

            # If output channels were split
            #  1. split post-process nodes
            #  2. concatenate along the output axis
            if IsOCSplited:
                # split post-process nodes
                post_nodes = {oc_id: None for oc_id in range(oc_split_num)}

                for oc_id in range(oc_split_num):
                  post_nodes[oc_id] = add_nodes[oc_id]

                # RemoveTargets.extend(PostProcess)
                # self.IsSplitedPostNode.extend([True for _ in range(len(PostProcess))])
                for PostNode in PostProcess[::-1]:
                  setattr(PostNode, "ShouldDelete", True)
                  if PostNode.op == op.get("nn.bias_add") and isinstance(PostNode.args[1], relay.Var):
                    self.removeSplitedArg(PostNode.args[1])
                  elif PostNode.op == op.get("nn.batch_norm"):
                    for i in range(1, 5):
                      if isinstance(PostNode.args[i], relay.Var):
                        self.removeSplitedArg(PostNode.args[i])
                  elif PostNode.op == op.get("imcflow.fused_batch_norm"):
                    for i in range(1, 3):
                      if isinstance(PostNode.args[i], relay.Var):
                        self.removeSplitedArg(PostNode.args[i])

                  for oc_id in range(oc_split_num):
                    oc_size = out_ch_limit if (oc_id * out_ch_limit) + out_ch_limit - 1 < OC else OC % out_ch_limit
                    if PostNode.op == op.get("nn.bias_add"):
                      if isinstance(PostNode.args[1], relay.Var):
                        ParamOldName = PostNode.args[1].name_hint
                        ParamNewName = f"{ParamOldName}_oc{oc_id}"
                        ParamNewType = relay.TensorType([oc_size], dtype=PostNode.args[1].type_annotation.dtype)
                        SplitParam = relay.Var(ParamNewName, ParamNewType)
                        OldParam = self.OldParamDict[ParamOldName]
                        if isinstance(OldParam, tvm.nd.NDArray):
                          NewData = OldParam.numpy()[oc_id*out_ch_limit:(oc_id*out_ch_limit)+oc_size]
                          self.addParamVar(SplitParam, tvm.nd.array(NewData, device=OldParam.device))
                        else:
                          NewData = OldParam[oc_id*out_ch_limit:(oc_id*out_ch_limit)+oc_size]
                          self.addParamVar(SplitParam, tvm.nd.array(NewData))
                      else:
                        assert isinstance(PostNode.args[1], relay.Constant), "PostNode.args[0] must be a Var or Constant"
                        nd_array = PostNode.args[1].data.numpy()[oc_id*out_ch_limit:(oc_id*out_ch_limit)+oc_size]
                        SplitParam = relay.Constant(tvm.nd.array(nd_array))
                      post_nodes[oc_id] = relay.nn.bias_add(post_nodes[oc_id], SplitParam, PostNode.attrs.axis)
                    elif PostNode.op == op.get("nn.relu"):
                      post_nodes[oc_id] = relay.nn.relu(post_nodes[oc_id])
                    elif PostNode.op == op.get("nn.batch_norm"):
                      NewParams = []
                      for i in range(1, 5):
                        if isinstance(PostNode.args[i], relay.Var):
                          ParamOldName = PostNode.args[i].name_hint
                          ParamNewName = f"{ParamOldName}_oc{oc_id}"
                          ParamNewType = relay.TensorType([oc_size], dtype=PostNode.args[i].type_annotation.dtype)
                          SplitParam = relay.Var(ParamNewName, ParamNewType)
                          OldParam = self.OldParamDict[ParamOldName]
                          if isinstance(OldParam, tvm.nd.NDArray):
                            NewData = OldParam.numpy()[oc_id*out_ch_limit:(oc_id*out_ch_limit)+oc_size]
                            self.addParamVar(SplitParam, tvm.nd.array(NewData, device=OldParam.device))
                          else:
                            NewData = OldParam[oc_id*out_ch_limit:(oc_id*out_ch_limit)+oc_size]
                            self.addParamVar(SplitParam, tvm.nd.array(NewData))
                        else:
                          assert isinstance(PostNode.args[i], relay.Constant), "PostNode.args[i] must be a Var or Constant"
                          nd_array = PostNode.args[i].data.numpy()[oc_id*out_ch_limit:(oc_id*out_ch_limit)+oc_size]
                          SplitParam = relay.Constant(tvm.nd.array(nd_array))
                        NewParams.append(SplitParam)
                      post_nodes[oc_id] = relay.nn.batch_norm(post_nodes[oc_id], *NewParams)[0]
                    elif PostNode.op == op.get("imcflow.fused_batch_norm"):
                      NewParams = []
                      for i in range(1, 3):
                        if isinstance(PostNode.args[i], relay.Var):
                          ParamOldName = PostNode.args[i].name_hint
                          ParamNewName = f"{ParamOldName}_oc{oc_id}"
                          ParamNewType = relay.TensorType([oc_size], dtype=PostNode.args[i].type_annotation.dtype)
                          SplitParam = relay.Var(ParamNewName, ParamNewType)
                          OldParam = self.OldParamDict[ParamOldName]
                          if isinstance(OldParam, tvm.nd.NDArray):
                            NewData = OldParam.numpy()[oc_id*out_ch_limit:(oc_id*out_ch_limit)+oc_size]
                            self.addParamVar(SplitParam, tvm.nd.array(NewData, device=OldParam.device))
                          else:
                            NewData = OldParam[oc_id*out_ch_limit:(oc_id*out_ch_limit)+oc_size]
                            self.addParamVar(SplitParam, tvm.nd.array(NewData))
                        else:
                          assert isinstance(PostNode.args[i], relay.Constant), "PostNode.args[i] must be a Var or Constant"
                          nd_array = PostNode.args[i].data.numpy()[oc_id*out_ch_limit:(oc_id*out_ch_limit)+oc_size]
                          SplitParam = relay.Constant(tvm.nd.array(nd_array))
                        NewParams.append(SplitParam)
                      post_nodes[oc_id] = imcflow_batch_norm(post_nodes[oc_id], *NewParams)

                concat_node = relay.op.concatenate([post_nodes[oc_id] for oc_id in range(oc_split_num)], axis=1)
            else:
                concat_node = add_nodes[0]

            return concat_node

          def visit_call(self, call):
            if call.op == op.get("nn.imcflow_qconv") or call.op == op.get("nn.imcflow_qdwconv"):
              PostProcess = self.PostProcess[:]
              self.PostProcess = []
              NewCall = super().visit_call(call)
              NewCall = self.split_and_optimize_conv2d(NewCall, mod, PostProcess)
              return NewCall
            elif call.op in [op.get("nn.bias_add"), op.get("nn.relu"), op.get("nn.batch_norm"), op.get("imcflow.fused_batch_norm")]:
              self.PostProcess.append(call)
              NewCall = super().visit_call(call)
              if hasattr(call, "ShouldDelete"):
                if call.op == op.get("nn.batch_norm"):
                  return relay.Tuple([NewCall.args[0]])
                else:
                  return NewCall.args[0]
              else:
                return NewCall
            else:
              # self.IsSplitedPostNode.extend([False for _ in range(len(self.PostProcess))])
              self.PostProcess = []
              return super().visit_call(call)

        Spliter_ = Spliter(self.OldParamDict)
        NewFunc = Spliter_.visit(func)
        OldArgs = func.params
        NewArgs = OldArgs[:]
        for arg in Spliter_.DeleteArgs:
          NewArgs.remove(arg)
        for arg in Spliter_.AddArgs:
          NewArgs.append(arg)
        self.NewParamDict = Spliter_.NewParamDict

        NewFunc = relay.Function(NewArgs, NewFunc.body, attrs=func.attrs)
        NewFunc = _RedundantTupleRemover().visit(NewFunc)

        return NewFunc

    worker = Worker(OldParamDict)
    for global_var, func in mod.functions.items():
      # if isinstance(func, relay.Function) and "Compiler" in func.attrs and re.match(r"imcflow.*", func.attrs["Compiler"]):
      if isinstance(func, relay.Function) and "global_symbol" in func.attrs and "imcflow" in func.attrs["global_symbol"]:
        mod[global_var] = worker.transform_function(func, mod)

    return mod, worker.NewParamDict

def merge_composite_ops(mod):
    for global_var, func in mod.functions.items():
        # if isinstance(func, relay.Function) and "Compiler" in func.attrs and re.match(r"imcflow.*", func.attrs["Compiler"]):
        if isinstance(func, relay.Function) and "global_symbol" in func.attrs and "imcflow" in func.attrs["global_symbol"]:
            attr_record = func.attrs
            func_no_attr = relay.Function(func.params, func.body) # no global_symbols attr
            target_mod = tvm.IRModule.from_expr(func_no_attr)
            transformed = transform.MergeComposite(imcflow.pattern_table())(target_mod)
            _, transformed_func = transformed.functions.items()[0]
            transformed_func = relay.Function(transformed_func.params, transformed_func.body,
                                              ret_type=transformed_func.ret_type, attrs=attr_record)
            mod[global_var] = transformed_func
    return mod

    # transformed = transform.MergeComposite(imcflow.pattern_table())(mod["tvmgen_default_imcflow_main_0"])
    transformed = transform.MergeComposite(imcflow.pattern_table())(mod)
    return transformed

@relay.transform.function_pass(opt_level=0)
class DenseToConv:
    def __init__(self):
      pass

    def transform_function(self, func, mod, ctx):
      class _Mutator(tvm.relay.ExprMutator):
        """convert dense to conv2d with kernel size 1x1"""

        def transform(self, expr, mod):

          def _get_type(node):
              """A method to infer the type of a relay expression."""
              mod = tvm.IRModule.from_expr(node)
              mod = relay.transform.InferType()(mod)
              entry = mod["main"]

              infer_out = entry if isinstance(node, relay.Function) else entry.body
              out_type = infer_out._checked_type_

              if isinstance(out_type, TensorType):
                  # Single tensor, get the shape directly
                  shapes = [int(dim) for dim in out_type.shape]
              elif isinstance(out_type, TupleType):
                  # Tuple of tensors, get the shape of each tensor in the tuple
                  shapes = [int(field) for field in out_type.fields]
              else:
                  raise RuntimeError(f"Unsupported output type {type(out_type)} in operator {node.op.name}")

              return shapes

          # Extract input and kernel shapes
          _, K = _get_type(expr.args[0])  # Input shape
          N, _ = _get_type(expr.args[1])  # Kernel shape

          KH, KW = 1, 1
          IC = K
          OC = N
          stride = 1
          padding = 0

          # reshape input
          x = relay.op.transform.reshape(expr.args[0], newshape=(1, IC, 1, 1))

          # reshape weight
          w = relay.op.transform.reshape(expr.args[1], newshape=(OC, IC, KH, KW))

          # convert dense to conv2d
          y = relay.nn.conv2d(
              x,
              w,
              channels=OC,
              kernel_size=(KH, KW),
              strides=(stride, stride),
              padding=(padding, padding),
          )

          y = relay.op.transform.reshape(y, newshape=(1, N))

          return y

        def visit_call(self, call):
          if call.op == op.get("nn.dense"):
            NewCall = super().visit_call(call)
            NewCall = self.transform(NewCall, mod)
            return NewCall
          else:
            return super().visit_call(call)

      return _Mutator().visit(func)

def getFirstInCalls(expr):
  InCalls = []

  class _Visitor(tvm.relay.ExprVisitor):
    def visit_call(self, call):
      # if int(hash(expr)) != int(hash(call)):
      if getNodeID(expr) != getNodeID(call):
        InCalls.append(call)
      super().visit_call(call)

  _Visitor().visit(expr)

  pass

def getFirstOutCall(func, expr):
  pass

def makeSplitConcatDepsRegions(mod):
  for global_var, func in mod.functions.items():
    # if isinstance(func, relay.Function) and "Compiler" in func.attrs and re.match(r"imcflow.*", func.attrs["Compiler"]):
    if isinstance(func, relay.Function) and "global_symbol" in func.attrs and "imcflow" in func.attrs["global_symbol"]:
      SplitConcatRegions = getSplitConcatDepsRegionsImpl(func)
      func_attr = func.attrs
      target_mod = tvm.IRModule.from_expr(relay.Function(func.params, func.body, ret_type=func.ret_type))
      target_mod = imcflow.ImcflowAnnotationPass(SplitConcatRegions, "split_concat_")(target_mod)
      # printModel(".", target_mod, {}, "split_concat_deps_before_partition")
      target_mod = transform.MergeCompilerRegions()(target_mod)
      target_mod = convert_compiler_regions_to_composite(target_mod)
      transformed_func = target_mod.functions.items()[0][1]
      transformed_func = transformed_func.with_attr({k:v for k,v in func_attr.items()})
      mod[global_var] = transformed_func

      # target_mod = imcflow.ImcflowCleanRegionTag()(target_mod)
      # target_mod = transform.PartitionGraph()(target_mod)
      # for new_gv, new_func in target_mod.functions.items():
      #   if new_gv.name_hint == "main":
      #     new_func = new_func.with_attr({k:v for k,v in func_attr.items()})
      #     mod[global_var] = new_func
      #   else:
      #     sub_func_gv = relay.GlobalVar(f"{global_var.name_hint}_{new_gv.name_hint}")
      #     mod[sub_func_gv] = new_func

  return mod

def getSplitConcatDepsRegionsImpl(func):
  """
  Traverse the graph and find split/concat dependent regions using Use-Def chain.

  For each split node, find all its consumers (nodes that use split outputs).
  For each concat node, find all its producers (nodes that produce concat inputs).
  Create regions containing these nodes and merge overlapping regions.
  """

  # Step 1: Build Use-Def chain (def -> users mapping)
  class _UseDefChainBuilder(relay.ExprVisitor):
    def __init__(self):
      super().__init__()
      self.def_to_users = {}  # expr -> [users]
      self.split_nodes = []
      self.concat_nodes = []

    def add_user(self, definition, user):
      """Add user to the definition's user list"""
      if definition not in self.def_to_users:
        self.def_to_users[definition] = []
      if user not in self.def_to_users[definition]:
        self.def_to_users[definition].append(user)

    def visit_call(self, call):
      # Check if this is split or concat
      if isinstance(call.op, tvm.ir.Op):
        if call.op == op.get("split"):
          self.split_nodes.append(call)
          debug_print(f"Split node detected: {getNodeID(call)}")
        elif call.op == op.get("concatenate"):
          self.concat_nodes.append(call)
          debug_print(f"Concat node detected: {getNodeID(call)}")

      # Register all args as definitions used by this call
      for arg in call.args:
        self.add_user(arg, call)
        self.visit(arg)

    def visit_tuple(self, tup):
      # Register tuple fields as definitions used by the tuple
      for field in tup.fields:
        self.add_user(field, tup)
        self.visit(field)

    def visit_tuple_getitem(self, tgi):
      # Register the tuple as definition used by tuple_getitem
      self.add_user(tgi.tuple_value, tgi)
      self.visit(tgi.tuple_value)

    def get_users(self, expr):
      """Get all users of an expression"""
      return self.def_to_users.get(expr, [])

  # Build the Use-Def chain
  debug_print("Building Use-Def chain...")
  builder = _UseDefChainBuilder()
  builder.visit(func)

  debug_print(f"Found {len(builder.split_nodes)} split nodes and {len(builder.concat_nodes)} concat nodes")

  # Step 2: For each split node, find all direct consumers (BFS from split node)
  def find_consumers(start_node, def_to_users):
    """Find direct Call consumers, skipping through Tuple/TupleGetItem"""
    consumers = set()
    queue = [start_node]
    visited = {start_node}

    while queue:
      current = queue.pop(0)

      # Get direct users of current node
      users = def_to_users.get(current, [])
      for user in users:
        if user not in visited:
          visited.add(user)

          if isinstance(user, relay.Call):
            # Found a Call node - add it and STOP searching this branch
            consumers.add(user)
          else:
            # Tuple or TupleGetItem - continue BFS through them
            consumers.add(user)
            queue.append(user)

    return consumers

  # Step 3: For each concat node, find all direct producers (traverse args)
  def find_producers(concat_node):
    """Find direct Call producers, skipping through Tuple/TupleGetItem"""
    producers = set()

    def trace_back(expr):
      """Trace back to find first Call nodes, stopping at them"""
      if isinstance(expr, relay.Call):
        # Found a Call node - add it and STOP searching this branch
        producers.add(expr)
      elif isinstance(expr, relay.Tuple):
        # Trace through tuple fields
        producers.add(expr)
        for field in expr.fields:
          trace_back(field)
      elif isinstance(expr, relay.TupleGetItem):
        producers.add(expr)
        # Trace through the tuple source
        trace_back(expr.tuple_value)
      # Var and Constant are leaves, stop here

    # Concat typically has a single Tuple argument
    for arg in concat_node.args:
      trace_back(arg)

    return producers

  # Step 4: Build regions
  Results = {}

  # Process split nodes
  for split_node in builder.split_nodes:
    consumers = find_consumers(split_node, builder.def_to_users)
    if consumers:
      Results[split_node] = list(consumers)
      debug_print(f"Split node {getNodeID(split_node)} has {len(consumers)} consumers")

  # Process concat nodes
  for concat_node in builder.concat_nodes:
    producers = find_producers(concat_node)
    if producers:
      Results[concat_node] = list(producers)
      debug_print(f"Concat node {getNodeID(concat_node)} has {len(producers)} producers")

  # Step 5: Create regions (include the split/concat node itself and its related nodes)
  Regions = []
  for key, related_nodes in Results.items():
    Region = [key] + related_nodes
    Regions.append(Region)

  debug_print(f"Split-Concat dependent regions: {len(Regions)} regions created")
  for i, region in enumerate(Regions):
    debug_print(f"  Region {i}: {len(region)} nodes")

  # Step 6: Merge regions if they have any intersection
  Changed = True
  while Changed:
    Changed = False
    for i in range(len(Regions)):
      for j in range(i+1, len(Regions)):
        if len(set(Regions[i]) & set(Regions[j])) > 0:
          # Merge region j into region i
          Regions[i] = list(set(Regions[i]) | set(Regions[j]))
          Regions.pop(j)
          Changed = True
          debug_print(f"Merged region {j} into region {i}")
          break
      if Changed:
        break

  debug_print(f"Final merged regions: {len(Regions)} regions")
  return Regions

def getInputNodes(expr, recursive=False):
  InNodes = []

  class _Visitor(tvm.relay.ExprVisitor):
    def visit_call(self, call):
      for arg in call.args:
        InNodes.append(arg)
      if recursive:
        super().visit_call(call)

    def visit_tuple_getitem(self, op):
      InNodes.append(op.tuple_value)
      if recursive:
        super().visit_tuple_getitem(op)

    def visit_tuple(self, op):
      for field in op.fields:
        InNodes.append(field)
      if recursive:
        super().visit_tuple(op)

  if isinstance(expr, list):
    for node in expr:
      _Visitor().visit(node)
  else:
    _Visitor().visit(expr)

  return list(set(InNodes))

def getOutputNodes(expr, recursive=False):
  OutNodes = []

  class _Visitor(tvm.relay.ExprVisitor):
    def visit_call(self, call):
      for arg in call.args:
        # if int(hash(expr)) == int(hash(arg)):
        if getNodeID(expr) == getNodeID(arg):
          OutNodes.append(call)
      if recursive:
        super().visit_call(call)

    def visit_tuple_getitem(self, op):
      OutNodes.append(op)
      if recursive:
        super().visit_tuple_getitem(op)

    def visit_tuple(self, op):
      for field in op.fields:
        OutNodes.append(field)
      if recursive:
        super().visit_tuple(op)

  _Visitor().visit(expr)
  return OutNodes

class AnnotGenerator:
    def __init__(self):
      self.RegionList = []

    def createRegion(self, mod):
      assert len(mod.functions.items()) == 1, "only one function is allowed in the module"
      target_func = list(mod.functions.items())[0][1]
      self.visit_function(target_func, mod)
      return self.RegionList

    def visit_function(self, func, mod):
      RegionList = []

      class _Annotator(tvm.relay.ExprVisitor):
        """
          Target Operators:
            conv2d, bias_add, batch_norm, relu, add and fused versions
              + min_max_quant, nu_quant, div
            split, concat
        """
        def createRegion(self):
          Region = []
          RegionList.append(Region)
          return Region

        def addToRegion(self, Region, Node):
          if Node not in Region:
            Region.append(Node)
          return Region

        def getRegionSize(self, Region):
          Cost = 0
          for Node in Region:
            Cost = Cost + self.getCost(Node)
          return Cost

        def getRegion(self, Node):
          Regions = []
          if isinstance(Node, list):
            for n in Node:
              for Region in RegionList:
                if n in Region:
                  if Region not in Regions:
                    Regions.append(Region)
            return Regions
          else:
            for Region in RegionList:
              if Node in Region:
                return Region
            return None

        def isComposite(self, call):
          return isinstance(call.op, relay.Function) and "Composite" in call.op.attrs and re.match(r"imcflow.*", call.op.attrs["Composite"])

        def isSupportedOp(self, call):
          return isinstance(call.op, tvm.ir.Op) and call.op.name in ImcflowDeviceConfig.SUPPORTED_OPS

        def isSuperNode(self, call):
          return isinstance(call.op, relay.Function) and "Composite" in call.op.attrs and re.match(r"split_concat_imcflow.*", call.op.attrs["Composite"])

        def isNoCostCall(self, call):
          return isinstance(call.op, tvm.ir.Op) and call.op.name in ImcflowDeviceConfig.NO_COST_OPS

        def getCost(self, call):
          if not isinstance(call, Call):
             return 0

          IsComposite = self.isComposite(call)
          IsSupportedOp = self.isSupportedOp(call)
          IsSuperNode = self.isSuperNode(call)
          IsNoCostCall = self.isNoCostCall(call)

          class _CostVisitor(tvm.relay.ExprVisitor):
            def __init__(self, getCostFunc):
              super().__init__()
              self.Cost = 0
              self.getCost = getCostFunc

            def isSuperNode(self, call):
              return isinstance(call.op, relay.Function) and "Composite" in call.op.attrs and re.match(r"split_concat_imcflow.*", call.op.attrs["Composite"])

            def visit(self, expr):
              self.Cost = self.Cost + self.getCost(expr)
              super().visit(expr)

            def visit_call(self, call):
              # if isinstance(call.op, relay.GlobalVar) and re.match(r"imcflow_.*", mod[call.op].attrs["Compiler"]):
              if self.isSuperNode(call):
                self.visit(call.op)
              for a in call.args:
                self.visit(a)

          if IsNoCostCall:
            return 0

          if IsComposite or IsSupportedOp:
            return 1

          if IsSuperNode:
            obj = _CostVisitor(self.getCost)
            obj.visit(call.op.body)
            return obj.Cost

          debug_print(f"Warning: Unsupported node found in cost calculation: {call}")
          raise NotImplementedError()

        def visit_call(self, call):
          # post DFS search
          for a in call.args:
              self.visit(a)

          # check this node is for imcflow
          IsComposite = self.isComposite(call)
          IsSupportedOp = self.isSupportedOp(call)
          IsSuperNode = self.isSuperNode(call)

          if IsComposite or IsSupportedOp or IsSuperNode:
            # check possibility
            if self.getCost(call) > ImcflowDeviceConfig.IMCE_NUM:
              raise ValueError("Cost of node is too high")

            # get possible region list
            InputNodes = getInputNodes(call)
            InputRegions = self.getRegion(InputNodes)
            CandidateRegions = InputRegions[:]

            ## cycle dependency check
            for InputRegion in InputRegions:
              for InputNode in [x for x in InputNodes if x not in InputRegion]:
                RecurInputRegions = self.getRegion(getInputNodes(InputNode, True))
                if InputRegion in RecurInputRegions:
                  try:
                    CandidateRegions.remove(InputRegion)
                  except:
                    pass

            ## capacity check
            Deletes = []
            for CandidateRegion in CandidateRegions:
              if self.getRegionSize(CandidateRegion) + self.getCost(call) > ImcflowDeviceConfig.IMCE_NUM:
                Deletes.append(CandidateRegion)
            for Delete in Deletes:
              if Delete in CandidateRegions:
                CandidateRegions.remove(Delete)

            ## select region
            #TODO: select optimal region. curently, select first region
            if len(CandidateRegions) > 0:
              Region = CandidateRegions[0]
            else:
              Region = self.createRegion()
            Region = self.addToRegion(Region, call)

        def visit_tuple_getitem(self, op):
          super().visit_tuple_getitem(op)
          TupleValueRegion = self.getRegion(op.tuple_value)
          TupleValueRegion = self.addToRegion(TupleValueRegion, op)
          # TupleValueRegion = self.addToRegion(TupleValueRegion, -1)

        def visit_tuple(self, op):
          super().visit_tuple(op)

          # get possible region list
          InputNodes = getInputNodes(op)
          InputRegions = self.getRegion(InputNodes)
          CandidateRegions = InputRegions[:]

          ## cycle dependency check
          for InputRegion in InputRegions:
            for InputNode in [x for x in InputNodes if x not in InputRegion]:
              RecurInputRegions = self.getRegion(getInputNodes(InputNode, True))
              if InputRegion in RecurInputRegions:
                try:
                  CandidateRegions.pop(InputRegion)
                except:
                  pass

          ## select region
          #TODO: select optimal region. curently, select first region
          if len(CandidateRegions) > 0:
            Region = CandidateRegions[0]
          else:
            Region = self.createRegion()

          # add node to region
          Region = self.addToRegion(Region, op)
          # Region = self.addToRegion(Region, -1)

      # find all regions
      _Annotator().visit(func)

      self.RegionList = RegionList

    def createRegionBFS(self, mod):
      """Build regions using a BFS-style (topological, Kahn) traversal.
      Processes producers before consumers (post-style w.r.t. inputs), but in breadth-first order.
      """
      assert len(mod.functions.items()) == 1, "only one function is allowed in the module"
      func = list(mod.functions.items())[0][1]

      RegionList = []

      class _AnnotatorBFS:
        def __init__(self, outer_self):
          self.RegionList = RegionList
          self.outer = outer_self
          # Track most recently assigned region to attach nodes with no input regions
          self.last_assigned_region = None

        def createRegion(self):
          Region = []
          self.RegionList.append(Region)
          return Region

        def addToRegion(self, Region, Node):
          if Region is None:
            Region = self.createRegion()
          if Node not in Region:
            Region.append(Node)
          # Update last assigned region so subsequent nodes with no inputs can piggyback
          self.last_assigned_region = Region
          return Region

        def getRegionSize(self, Region):
          Cost = 0
          for Node in Region:
            Cost = Cost + self.getCost(Node)
          return Cost

        def getRegion(self, Node):
          Regions = []
          if isinstance(Node, list):
            for n in Node:
              for Region in self.RegionList:
                if n in Region and Region not in Regions:
                  Regions.append(Region)
            return Regions
          else:
            for Region in self.RegionList:
              if Node in Region:
                return Region
            return None

        def isComposite(self, call):
          return isinstance(call.op, relay.Function) and "Composite" in call.op.attrs and re.match(r"imcflow.*", call.op.attrs["Composite"])

        def isSupportedOp(self, call):
          return isinstance(call.op, tvm.ir.Op) and call.op.name in ImcflowDeviceConfig.SUPPORTED_OPS

        def isSuperNode(self, call):
          return isinstance(call.op, relay.Function) and "Composite" in call.op.attrs and re.match(r"split_concat_imcflow.*", call.op.attrs["Composite"])

        def isNoCostCall(self, call):
          return isinstance(call.op, tvm.ir.Op) and call.op.name in ImcflowDeviceConfig.NO_COST_OPS

        def getCost(self, call):
          if not isinstance(call, Call):
            return 0
          IsComposite = self.isComposite(call)
          IsSupportedOp = self.isSupportedOp(call)
          IsSuperNode = self.isSuperNode(call)
          IsNoCostCall = self.isNoCostCall(call)

          class _CostVisitor(tvm.relay.ExprVisitor):
            def __init__(self, getCostFunc):
              super().__init__()
              self.Cost = 0
              self.getCost = getCostFunc

            def isSuperNode(self, call):
              return isinstance(call.op, relay.Function) and "Composite" in call.op.attrs and re.match(r"split_concat_imcflow.*", call.op.attrs["Composite"])

            def visit(self, expr):
              self.Cost = self.Cost + self.getCost(expr)
              super().visit(expr)

            def visit_call(self, call):
              if self.isSuperNode(call):
                self.visit(call.op)
              for a in call.args:
                self.visit(a)

          if IsNoCostCall:
            return 0
          if IsComposite or IsSupportedOp:
            return 1
          if IsSuperNode:
            obj = _CostVisitor(self.getCost)
            obj.visit(call.op.body)
            return obj.Cost
          # Unsupported node: cost 0 so it doesn't affect capacity, but it's not placed anyway
          return 0

        class GraphBuilder(tvm.relay.ExprVisitor):
          def __init__(self):
            super().__init__()
            self.nodes = []  # Call/Tuple/TupleGetItem
            self.edges = collections.defaultdict(list)  # src -> [dst]
            self.rev_edges = collections.defaultdict(list)  # dst -> [src]
            self.in_degree = collections.defaultdict(int)

          def _add_node(self, n):
            if n not in self.nodes:
              self.nodes.append(n)

          def _connect(self, src, dst):
            self.edges[src].append(dst)
            self.rev_edges[dst].append(src)
            self.in_degree[dst] += 1

          def visit_call(self, call):
            self._add_node(call)
            for a in call.args:
              self.visit(a)
              if isinstance(a, (Call, Tuple, TupleGetItem)):
                self._add_node(a)
                self._connect(a, call)

          def visit_tuple(self, tup):
            self._add_node(tup)
            for f in tup.fields:
              self.visit(f)
              if isinstance(f, (Call, Tuple, TupleGetItem)):
                self._add_node(f)
                self._connect(f, tup)

          def visit_tuple_getitem(self, tgi):
            self._add_node(tgi)
            self.visit(tgi.tuple_value)
            tv = tgi.tuple_value
            if isinstance(tv, (Call, Tuple, TupleGetItem)):
              self._add_node(tv)
              self._connect(tv, tgi)

        def _topo_bfs_order(self, fn):
          gb = self.GraphBuilder()
          gb.visit(fn)
          # Initialize queue with zero in-degree nodes
          from collections import deque
          q = deque()
          indeg = dict(gb.in_degree)
          for n in gb.nodes:
            if indeg.get(n, 0) == 0:
              q.append(n)
          order = []
          while q:
            u = q.popleft()
            order.append(u)
            for v in gb.edges.get(u, []):
              indeg[v] = indeg.get(v, 0) - 1
              if indeg[v] == 0:
                q.append(v)
          return order, gb.edges, gb.rev_edges

        def run(self, fn):
          order, edges, rev_edges = self._topo_bfs_order(fn)
          for node in order:
            if isinstance(node, Call):
              IsComposite = self.isComposite(node)
              IsSupportedOp = self.isSupportedOp(node)
              IsSuperNode = self.isSuperNode(node)
              if IsComposite or IsSupportedOp or IsSuperNode:
                if self.getCost(node) > ImcflowDeviceConfig.IMCE_NUM:
                  raise ValueError("Cost of node is too high")

                # Determine predecessor nodes that belong to regions
                preds = rev_edges.get(node, [])
                input_nodes = [p for p in preds if isinstance(p, (Call, Tuple, TupleGetItem))]
                input_regions = self.getRegion(input_nodes)
                candidate_regions = input_regions[:]

                # Cycle check: remove regions that would introduce cycles
                for in_region in input_regions:
                  for in_node in [x for x in input_nodes if x not in in_region]:
                    recur_regions = self.getRegion(getInputNodes(in_node, True))
                    if in_region in recur_regions:
                      if in_region in candidate_regions:
                        candidate_regions.remove(in_region)
                        debug_print(f"cycle detected. current node {node}. cycle region : {in_region}")

                # Capacity check
                deletes = []
                for cand in candidate_regions:
                  if self.getRegionSize(cand) + self.getCost(node) > ImcflowDeviceConfig.IMCE_NUM:
                    deletes.append(cand)
                    debug_print(f"candidate size : {self.getRegionSize(cand)}. current node size : {self.getCost(node)}. too big node!!")
                for d in deletes:
                  if d in candidate_regions:
                    candidate_regions.remove(d)

                # Selection policy: if multiple distinct input regions, just use first one
                uniq = list({id(r): r for r in candidate_regions}.values())
                if len(uniq) == 1:
                  Region = uniq[0]
                  self.addToRegion(Region, node)
                elif len(uniq) > 1:
                  Region = uniq[0]
                  # Region = self.createRegion()
                  self.addToRegion(Region, node)
                else:
                  # No input region (inputs likely Var/Const). Prefer previous node's region if available.
                  #TODO: just traverse call node..(?) no need to consider Var and Const node..
                  Region = None
                  if self.last_assigned_region is not None:
                    # Capacity gate when attaching to previous region
                    if self.getRegionSize(self.last_assigned_region) + self.getCost(node) <= ImcflowDeviceConfig.IMCE_NUM:
                      Region = self.last_assigned_region
                  if Region is None:
                    Region = self.createRegion()
                  self.addToRegion(Region, node)

            elif isinstance(node, TupleGetItem):
              # Attach to tuple region; create one if absent
              Region = self.getRegion(node.tuple_value)
              Region = self.addToRegion(Region, node)

            elif isinstance(node, Tuple):
              preds = rev_edges.get(node, [])
              input_nodes = [p for p in preds if isinstance(p, (Call, Tuple, TupleGetItem))]
              input_regions = self.getRegion(input_nodes)
              candidate_regions = input_regions[:]

              for in_region in input_regions:
                for in_node in [x for x in input_nodes if x not in in_region]:
                  recur_regions = self.getRegion(getInputNodes(in_node, True))
                  if in_region in recur_regions:
                    if in_region in candidate_regions:
                      candidate_regions.remove(in_region)

              if len(candidate_regions) == 1:
                Region = candidate_regions[0]
              else:
                Region = self.createRegion()
              self.addToRegion(Region, node)

          # No second pass needed; nodes with no inputs were attached to previous region when possible

      annot = _AnnotatorBFS(self)
      annot.run(func)
      self.RegionList = RegionList
      return self.RegionList

def partitionRound(mod):
  for global_var, func in mod.functions.items():
    if isinstance(func, relay.Function) and "Compiler" in func.attrs and re.match(r"imcflow.*", func.attrs["Compiler"]):
      name = global_var.name_hint
      func_attr = func.attrs
      annotator = AnnotGenerator()
      target_mod = tvm.IRModule.from_expr(relay.Function(func.params, func.body, ret_type=func.ret_type))
      # RegionList = annotator.createRegion(target_mod)
      RegionList = annotator.createRegionBFS(target_mod)
      target_mod = imcflow.ImcflowAnnotationPass(RegionList, f"{name}_round_")(target_mod)
      target_mod = transform.MergeCompilerRegions()(target_mod)
      target_mod = imcflow.ImcflowCleanRegionTag()(target_mod)
      # printModel("resnet8_evl", target_mod, {}, f"{name}_round_partitioned")
      target_mod = transform.PartitionGraph()(target_mod)

      for new_gv, new_func in target_mod.functions.items():
        if new_gv.name_hint == "main":
          new_func = new_func.with_attr({k:v for k,v in func_attr.items()})
          mod[global_var] = new_func
        else:
          mod[new_gv] = new_func

  return mod

@relay.transform.function_pass(opt_level=0)
class NodeMapper:
    def __init__(self):
      # self.MappingDict_2D = {}
      self.MappingDict = {}

    def run_(self, func):
      class _UseDefChainBuilder(relay.ExprVisitor):
        """Build use-def chain: expr -> [users of expr]"""
        def __init__(self):
          super().__init__()
          self.def_to_users = {}  # expr -> [users]
        
        def add_user(self, definition, user):
          """Add user to the definition's user list"""
          if definition not in self.def_to_users:
            self.def_to_users[definition] = []
          if user not in self.def_to_users[definition]:
            self.def_to_users[definition].append(user)
        
        def visit_call(self, call):
          # Register all args as definitions used by this call
          for arg in call.args:
            self.add_user(arg, call)
            self.visit(arg)
          
          # Visit the operator (for composite functions)
          if isinstance(call.op, relay.Function):
            self.visit(call.op)
        
        def visit_tuple(self, tup):
          # Register tuple fields as definitions used by the tuple
          for field in tup.fields:
            self.add_user(field, tup)
            self.visit(field)
        
        def visit_tuple_getitem(self, tgi):
          # Register the tuple as definition used by tuple_getitem
          self.add_user(tgi.tuple_value, tgi)
          self.visit(tgi.tuple_value)
        
        def get_users(self, expr):
          """Get all users of an expression"""
          return self.def_to_users.get(expr, [])

      class _Nodemapper(tvm.relay.ExprVisitor):
        """
          Assign hardware node ID to func, var, const, call nodes.
          Current implementation just assign hardware node ID interleavly.

          function node -> assign to inode 
          var node      -> assign to inode
          constant node -> assign to inode

          call node:
            split -> inode or imce
            other -> imce
          
          call nodes in composite function -> assign to the composite function's node ID

          Target Operators:
            conv2d, bias_add, batch_norm, relu, add and fused versions
            split, concat
          
          We assign var and constant node to consumer node to avoid sync overhead because some edges have hard order constraints.
          For example, 2d conv inputs are config and data. In this case, config should be arrived before data.
          
          Assumption:
            - concat node doesn't have args which is Var Node.
          
          TODO:
            - locality between producer and consumers
        """
        def __init__(self, use_def_chain_builder):
            super().__init__()
            self.MappingDict ={}
            self.imce_index = ImcflowDeviceConfig.IMCE_NUM - 1
            self.inode_index = ImcflowDeviceConfig.INODE_NUM - 1
            self.in_composite = False
            self.curr_composite_node_id = None
            self.vars = []
            self.consts = []
            self.remaining_splits = []
            self.use_def_builder = use_def_chain_builder
            self._split_prod_cons_map = {}

            self.undetermined_callnode_exists = False
            self.undetermined_callnode = None

        def traverse_func(self, func):
            self.visit(func)
            
            # assign var and constant nodes to consumer nodes
            self._assign_nodes_same_as_consumer(self.remaining_splits)
            self._assign_nodes_same_as_consumer(self.vars)
            self._assign_nodes_same_as_consumer(self.consts)
            return self.MappingDict
        
        def _assign_nodes_same_as_consumer(self, node_list):
            """Assign remaining split nodes to their consumer's hardware node"""
            for node in node_list:
                consumers = self.use_def_builder.get_users(node)
                if consumers:
                    # Find the first consumer that has been assigned
                    consumer_node_id = None
                    for consumer in consumers:
                        # Skip tuple and tuple_getitem nodes, find actual call nodes
                        actual_consumer = self._find_actual_consumer(consumer)
                        if actual_consumer and getNodeID(actual_consumer) in self.MappingDict:
                            consumer_node_id = self.MappingDict[getNodeID(actual_consumer)]
                            break
                    
                    if consumer_node_id is not None:
                      if consumer_node_id.is_imce():
                        self.MappingDict[getNodeID(node)] = consumer_node_id.master()
                      else:
                        self.MappingDict[getNodeID(node)] = consumer_node_id
                    else:
                      raise ValueError(f"No assigned consumer found for {node} node")
                else:
                  raise ValueError(f"{node} node has no consumers")
      
        def _find_actual_consumer(self, expr):
            """
            Find the actual consumer call node by traversing through tuple/tuple_getitem nodes.
            Returns the first Call node found, or None.
            """
            if isinstance(expr, relay.Call):
                return expr
            elif isinstance(expr, relay.Tuple):
                # Tuple is used by something else, find its users
                users = self.use_def_builder.get_users(expr)
                for user in users:
                    result = self._find_actual_consumer(user)
                    if result:
                        return result
            elif isinstance(expr, relay.TupleGetItem):
                # TupleGetItem is used by something else, find its users
                users = self.use_def_builder.get_users(expr)
                for user in users:
                    result = self._find_actual_consumer(user)
                    if result:
                        return result
            raise ValueError("No valid consumer Call node found in the chain")
        
        def visit_function(self, fn):
          if self.in_composite: 
            self.MappingDict[getNodeID(fn)] = self.curr_composite_node_id
          else:
            self.MappingDict[getNodeID(fn)] = NodeID.from_inode_coord(self.inode_index)
            self.inode_index -= 1
          super().visit_function(fn)
        
        def visit_var(self, var):
          if not self.in_composite:
            self.vars.append(var)
            # self.MappingDict[getNodeID(var)] = NodeID.from_inode_coord(self.inode_index)
            # self.inode_index -= 1
        
        def visit_constant(self, const):
          self.consts.append(const)
          # self.MappingDict[getNodeID(const)] = NodeID.from_inode_coord(self.inode_index)
          # self.inode_index -= 1

        def visit_call(self, call):
          # post DFS search
          # traverse child node

          # If we are already in a composite function, just traverse args without assigning
          # we need to find constant node only
          if self.in_composite:
            assert isinstance(call.op, tvm.ir.Op), "not built-in operator found in composite function"

          for a in call.args:
              self.visit(a)
          
          if not self.in_composite:
            IsConcat = isinstance(call.op, tvm.ir.Op) and call.op.name in ["concatenate"]
            IsSplit = isinstance(call.op, tvm.ir.Op) and call.op.name in ["split"]
            if IsConcat:
                self.MappingDict[getNodeID(call)] = self.MappingDict[getNodeID(call.args[-1].fields[-1])]
            elif IsSplit:
              producer_node_id = getNodeID(call.args[-1])
              if producer_node_id in self.MappingDict.keys():
                self.MappingDict[getNodeID(call)] = self.MappingDict[producer_node_id]
              else:
                self.remaining_splits.append(call)
            else:
                if self.imce_index < 0:
                    raise ValueError("too many compute nodes for available hardware nodes")
                self.MappingDict[getNodeID(call)] = NodeID.from_imce_coord(self.imce_index)
                self.imce_index -= 1
          else:
            # inside composite function, assign all nodes to the composite function's node ID
            self.MappingDict[getNodeID(call)] = self.curr_composite_node_id

          if isinstance(call.op, relay.Function) and "Composite" in call.op.attrs and re.match(r"imcflow.*", call.op.attrs["Composite"]):
            self.in_composite = True
            self.curr_composite_node_id = self.MappingDict[getNodeID(call)]
            self.visit(call.op)
            self.curr_composite_node_id = None
            self.in_composite = False
          else:
            self.visit(call.op)

        def visit_tuple_getitem(self, op):
          super().visit_tuple_getitem(op)

        def visit_tuple(self, op):
          super().visit_tuple(op)

      # First build use-def chain
      use_def_builder = _UseDefChainBuilder()
      use_def_builder.visit(func)
      
      # Then run node mapper with use-def chain
      return _Nodemapper(use_def_builder).traverse_func(func)

    def run(self, mod):
      imcflow_func_map = ImcflowDeviceConfig().ImcflowFuncMap
      for global_var, func in mod.functions.items():
        if isinstance(func, relay.Function) and "Compiler" in func.attrs and re.match(r"imcflow.*", func.attrs["Compiler"]):
          func_info = imcflow_func_map[global_var.name_hint]
          mapping_dict = self.run_(func_info.func_node)
          ImcflowDeviceConfig().HWNodeMap.update(mapping_dict)
      return mod

class ConcatDistributor:
  """
  Distribute concat operations with more than max_inputs into a binary tree structure.

  Hardware constraint: Each IMCE can handle at most max_inputs inputs.
  When a concat has more than max_inputs, we need to split it into a tree of smaller concats.

  This should run AFTER makeSplitConcatDepsRegions but BEFORE constructUsefulMappings.
  At this point, CustomIDs are not yet assigned, so we can freely add new nodes.

  Example with max_inputs=8 and 12 inputs (a,b,c,d,e,f,g,h,i,j,k,l):
    Original: concat(a,b,c,d,e,f,g,h,i,j,k,l)
    Tree form: concat(
                 concat(
                   concat(a,b,c,d),
                   concat(e,f,g,h)
                 ),
                 concat(i,j,k,l)
               )

  Node mapping: After NodeMapper runs, each intermediate concat will be mapped
  to the last input node's HW mapping (this happens automatically by NodeMapper logic).
  """
  def __init__(self, max_inputs=8):
    self.max_inputs = max_inputs

  def run(self, mod):
    """
    Transform the module by distributing large concat operations.
    This should run after makeSplitConcatDepsRegions.
    """
    for global_var, func in mod.functions.items():
      if isinstance(func, relay.Function) and "global_symbol" in func.attrs and "imcflow" in func.attrs["global_symbol"]:
        # Transform the function
        new_func = self._transform_function(func)

        # Update module if function changed
        if new_func != func:
          mod[global_var] = new_func
          debug_print(f"ConcatDistributor: Transformed function {global_var.name_hint}")

    return mod

  def _transform_function(self, func):
    """Transform a single function by distributing concat operations."""
    mutator = self._ConcatMutator(self.max_inputs)
    new_body = mutator.visit(func.body)

    if new_body == func.body:
      return func

    return relay.Function(
      func.params,
      new_body,
      func.ret_type,
      func.type_params,
      func.attrs
    )

  class _ConcatMutator(relay.ExprMutator):
    """Mutator that replaces large concat operations with binary trees."""

    def __init__(self, max_inputs):
      super().__init__()
      self.max_inputs = max_inputs

    def visit_call(self, call):
      # First, recursively transform children
      new_call = super().visit_call(call)

      # Check if this is a concat operation
      if isinstance(new_call.op, tvm.ir.Op) and new_call.op.name == "concatenate":
        # Get the tuple of inputs
        if len(new_call.args) > 0 and isinstance(new_call.args[0], relay.Tuple):
          inputs = new_call.args[0].fields

          # If we have more than max_inputs, we need to distribute
          if len(inputs) > self.max_inputs:
            debug_print(f"  ConcatDistributor: Found concat with {len(inputs)} inputs (max={self.max_inputs})")

            # Get concat attributes (axis, etc.)
            attrs = new_call.attrs

            # Build binary tree of concats
            new_concat = self._build_concat_tree(list(inputs), attrs)
            debug_print(f"  ConcatDistributor: Distributed concat into tree structure")

            return new_concat

      return new_call

    def _build_concat_tree(self, inputs, attrs):
      """
      Build a binary tree of concat operations.

      Args:
        inputs: List of input nodes
        attrs: Concat attributes (axis, etc.)

      Returns:
        A concat call node representing the tree
      """
      if len(inputs) <= self.max_inputs:
        # Base case: small enough to fit in one concat
        return relay.concatenate(relay.Tuple(inputs), axis=attrs.axis)

      # Recursive case: split into two halves
      mid = len(inputs) // 2

      # Handle the case where mid == 0 (shouldn't happen, but be safe)
      if mid == 0:
        mid = 1

      left_inputs = inputs[:mid]
      right_inputs = inputs[mid:]

      # Recursively build left and right subtrees
      left_concat = self._build_concat_tree(left_inputs, attrs)
      right_concat = self._build_concat_tree(right_inputs, attrs)

      # Combine them
      return relay.concatenate(relay.Tuple([left_concat, right_concat]), axis=attrs.axis)

def constructTensorEdgeList(mod):
  """
  make tensor edge list.
  output edge -> (last_node, func_node). odata and func_out tag attached
    if last_node is tuple, go into each field recursively and find first call
    if last_node is composite node, go into body and find first call. use (func_node, body_node) as dst_node

  input edge -> (var_node, dst_node)
                (const_node, dst_node)
      we use "var" tag for var_node
  
  Tuple and TupleGetItem nodes are not included in edge list. When we detect tuple and tgi node,
  we search var or const or call node they have and make the edges between (const, var, call) <-> (const, var, call)
  """
  @dataclass
  class TensorIDPair:
    graph_node_id : int
    split_idx : None | int
  class _Visitor(tvm.relay.ExprVisitor):

    def __init__(self):
        super().__init__()
        # self.MappingDict = ImcflowDeviceConfig().HWNodeMap
        self.TensorEdgeList = []
        self.InSubFunction = False
        self.IsSrcUnpacking = False
        # self.SubFunctionMapping = None
        self.SubFunctionNodeID = None
        self.VarProperties = {}

    def getCustomID(self, node):
      if isinstance(node, Function):
          return getNodeID(node)
      if isinstance(node, Call):
        if isinstance(node.op, relay.Function) and "Composite" in node.op.attrs and re.match(r"imcflow\..*", node.op.attrs["Composite"]):
          return (getNodeID(node), getNodeID(node.op.body))
        else:
          return getNodeID(node)
      elif isinstance(node, Tuple):
        result = []
        for b in node.fields:
          result.append(self.getCustomID(b))
        return result
      elif isinstance(node, TupleGetItem):
          return self.getCustomID(node.tuple_value)
      elif isinstance(node, Var):
          return getNodeID(node)
      elif isinstance(node, Constant):
          return getNodeID(node)

    def getInputGraphNodeSplitIndex(self, node):
      if isinstance(node, TupleGetItem):
        return node.index
      else:
        return None

    # def getInodePlaceHolderInputVar(self):
    #   return TensorIDPair(VAR_NODE_ID, 'inode_placeholder')

    # def getInodePlaceHolderInputConstant(self):
    #   return TensorIDPair(CONST_NODE_ID, 'inode_placeholder')

    def appendToTensorEdgeList(self, SrcGraphNodeIDs, DstGraphNodeID, SrcTag, DstTag, SplitIdx=None):
      if isinstance(SrcGraphNodeIDs, list):
        for SrcGraphNodeID in SrcGraphNodeIDs:
          self.appendToTensorEdgeList(SrcGraphNodeID, DstGraphNodeID, SrcTag, DstTag, SplitIdx)
      elif isinstance(SrcGraphNodeIDs, (int, tuple)):
        SrcGraphNodeID = SrcGraphNodeIDs
        self.TensorEdgeList.append(
          TensorEdge(TensorID(SrcGraphNodeID, SrcTag),
                     TensorID(DstGraphNodeID, DstTag),
                     SplitIdx)
        )
      else:
        raise ValueError("Invalid input tensor id pair")

    def visit_function(self, fn):
      # append to TensorEdgeList if fn is the entrance node of whole subgraph function
      if hasattr(fn.attrs, "Compiler") and fn.attrs["Compiler"]=="imcflow":
        #TODO: if fn.body is tuple, custom id will return list. it is not intuitivie..
        InputGraphNodeID = self.getCustomID(fn.body)
        DstGraphNodeID = self.getCustomID(fn)
        SrcTag = "odata"
        DstTag = "func_out"
        self.appendToTensorEdgeList(InputGraphNodeID, DstGraphNodeID, SrcTag, DstTag, None)

      if self.InSubFunction:
        self.VarProperties = {}
        for x in fn.params:
          self.VarProperties[x] = {}
          self.visit(x)
        self.visit(fn.body)
      else:
        super().visit_function(fn)

    def visit_call(self, call):
        # current_node_id = int(hash(call))  # Unique identifier for the current node
        # DstGraphNodeID = getNodeID(call)
        DstGraphNodeID = getNodeID(call) if not self.InSubFunction else (self.SubFunctionNodeID, getNodeID(call))
        # current_mapping = self.MappingDict[current_node_id] if not self.InSubFunction else self.SubFunctionMapping
        # DstNodeProperty = (current_mapping, current_node_id) if not self.InSubFunction else (current_mapping, (self.SubFunctionNodeID, current_node_id))
        # if not self.InSubFunction:
        #   DstNodeProperty = DstNode(current_mapping[0], current_node_id, current_mapping[1])
        # else:
        #   DstNodeProperty = DstNode(current_mapping[0], (self.SubFunctionNodeID, current_node_id), getNodeDebugID(call) + "_in_"  + current_mapping[1])

        # if current_mapping is None:
        #     return  # Skip nodes not included in the mapping

        IsComposite = isinstance(call.op, relay.Function) and "Composite" in call.op.attrs and re.match(r"imcflow\..*", call.op.attrs["Composite"])
        IsSupportedOp = isinstance(call.op, tvm.ir.Op) and call.op.name in ImcflowDeviceConfig.SUPPORTED_OPS

        if not IsComposite and not IsSupportedOp:
          debug_print(call)
          raise ValueError("Unsupported operator detected. please check.")

        # visit composite function
        # we will collect Var Nodes usage and its properties
        def _processInputNode(SrcGraphNode, SrcTag, DstGraphNodeID, DstTag, SplitIdx):
          if not self.InSubFunction:
            InputGraphNodeID = self.getCustomID(SrcGraphNode)
            self.appendToTensorEdgeList(InputGraphNodeID, DstGraphNodeID, SrcTag, DstTag, SplitIdx)
            return True
          else:
              if isinstance(SrcGraphNode, Var):
                # self.VarProperties[SrcGraphNode]["src_tag"] = SrcTag
                self.VarProperties[SrcGraphNode]["src_tag"] = "var"
                self.VarProperties[SrcGraphNode]["dst_tag"] = DstTag
                self.VarProperties[SrcGraphNode]["dst_graph_node_id"] = DstGraphNodeID
              if isinstance(SrcGraphNode, Constant):
                InputGraphNodeID = (self.SubFunctionNodeID, self.getCustomID(SrcGraphNode))
                self.appendToTensorEdgeList(InputGraphNodeID, DstGraphNodeID, SrcTag, DstTag, SplitIdx)
                return True
              # if self.IsSrcUnpacking is True:
              #   # append edge if (src: unpacking -> dst: qconv)
              #   InputGraphNodeID = (self.SubFunctionNodeID, self.getCustomID(SrcGraphNode))
              #   self.appendToTensorEdgeList(InputGraphNodeID, DstGraphNodeID, SrcTag, DstTag, SplitIdx)
              #   self.IsSrcUnpacking = False

        if IsComposite:
          self.InSubFunction = True
          # self.SubFunctionMapping = current_mapping
          self.SubFunctionNodeID = DstGraphNodeID
          self.visit(call.op)
          self.InSubFunction = False
          ParamToArg = {x: y for x, y in zip(call.op.params, call.args)}
          for var, arg in ParamToArg.items():
            # print(f"var: {var}, arg: {arg}, var_properties: {self.VarProperties[var]}")
            _processInputNode(arg, self.VarProperties[var]["src_tag"],
                              self.VarProperties[var]["dst_graph_node_id"], self.VarProperties[var]["dst_tag"],
                              self.getInputGraphNodeSplitIndex(arg))
        elif IsSupportedOp:
          if call.op == op.get("split"):
            _processInputNode(call.args[0], "odata", DstGraphNodeID, "data", self.getInputGraphNodeSplitIndex(call.args[0]))
          elif call.op == op.get("concatenate"):
            _processInputNode(call.args[0], "odata", DstGraphNodeID, "data", self.getInputGraphNodeSplitIndex(call.args[0]))
          elif call.op == op.get("add"):
            _processInputNode(call.args[0], "odata", DstGraphNodeID, "lhs", self.getInputGraphNodeSplitIndex(call.args[0]))
            _processInputNode(call.args[1], "odata", DstGraphNodeID, "rhs", self.getInputGraphNodeSplitIndex(call.args[1]))
          elif call.op == op.get("divide"):
            ScaleNode = 0 if isinstance(call.args[0], Constant) else 1
            InputNode = 1 if ScaleNode == 0 else 0
            _processInputNode(call.args[InputNode], "odata", DstGraphNodeID, "lhs", self.getInputGraphNodeSplitIndex(call.args[InputNode]))
            _processInputNode(call.args[ScaleNode], "scale", DstGraphNodeID, "rhs", None)
          elif call.op == op.get("multiply"):
            #TODO multiply input node can be constant.
            _processInputNode(call.args[0], "odata", DstGraphNodeID, "lhs", self.getInputGraphNodeSplitIndex(call.args[0]))
            _processInputNode(call.args[1], "odata", DstGraphNodeID, "rhs", self.getInputGraphNodeSplitIndex(call.args[1]))
          elif call.op == op.get("nn.conv2d"):
            _processInputNode(call.args[0], "odata", DstGraphNodeID, "data", self.getInputGraphNodeSplitIndex(call.args[0]))
            _processInputNode(call.args[1], "weight", DstGraphNodeID, "weight", None)
          elif call.op == op.get("nn.bias_add"):
            _processInputNode(call.args[0], "odata", DstGraphNodeID, "data", self.getInputGraphNodeSplitIndex(call.args[0]))
            _processInputNode(call.args[1], "bias", DstGraphNodeID, "bias", None)
          elif call.op == op.get("nn.relu"):
            _processInputNode(call.args[0], "odata", DstGraphNodeID, "data", self.getInputGraphNodeSplitIndex(call.args[0]))
          elif call.op == op.get("nn.imcflow_qconv") or call.op == op.get("nn.imcflow_qdwconv"):
            _processInputNode(call.args[0], "odata", DstGraphNodeID, "data", self.getInputGraphNodeSplitIndex(call.args[0]))
            _processInputNode(call.args[1], "weight", DstGraphNodeID, "weight", None)
            _processInputNode(call.args[2], "config", DstGraphNodeID, "config", None)
          elif call.op == op.get("imcflow.fused_batch_norm"):
            _processInputNode(call.args[0], "odata", DstGraphNodeID, "data", self.getInputGraphNodeSplitIndex(call.args[0]))
            _processInputNode(call.args[1], "fused_scale", DstGraphNodeID, "fused_scale", None)
            _processInputNode(call.args[2], "fused_bias", DstGraphNodeID, "fused_bias", None)
          elif call.op == op.get("qnn.imcflow_min_max_quantize"):
            _processInputNode(call.args[0], "odata", DstGraphNodeID, "data", self.getInputGraphNodeSplitIndex(call.args[0]))
            _processInputNode(call.args[1], "min", DstGraphNodeID, "min", None)
            _processInputNode(call.args[2], "max", DstGraphNodeID, "max", None)
          elif call.op == op.get("qnn.imcflow_nu_quantize"):
            _processInputNode(call.args[0], "odata", DstGraphNodeID, "data", self.getInputGraphNodeSplitIndex(call.args[0]))
            _processInputNode(call.args[1], "threshold", DstGraphNodeID, "threshold", None)
          # elif call.op == op.get("imcflow_packing"):
          #   _processInputNode(call.args[0], "odata", DstGraphNodeID, "data", self.getInputGraphNodeSplitIndex(call.args[0]))
          # elif call.op == op.get("imcflow_unpacking"):
          #   _processInputNode(call.args[0], "odata", DstGraphNodeID, "data", self.getInputGraphNodeSplitIndex(call.args[0]))
          else:
            raise ValueError("Unsupported operator detected. please check.")

        #Pre DFS search: Traverse child nodes
        for a in call.args:
            self.visit(a)

    def visit_tuple_getitem(self, op):
      super().visit_tuple_getitem(op)

    def visit_tuple(self, op):
      super().visit_tuple(op)

    def getTensorEdgeList(self, func_name, func):
      self.visit(func)
      return self.TensorEdgeList

  imcflow_func_map = ImcflowDeviceConfig().ImcflowFuncMap
  for func_name_var, func in mod.functions.items():
    if func_name_var.name_hint == "main": continue
    elif func.attrs["Compiler"]=="imcflow":
      func_info = imcflow_func_map[func_name_var.name_hint]
      ImcflowDeviceConfig().TensorEdgeListDict[func_name_var.name_hint] = _Visitor().getTensorEdgeList(func_name_var, func_info.func_node)
      ImcflowDeviceConfig().TensorEdgeList.extend(ImcflowDeviceConfig().TensorEdgeListDict[func_name_var.name_hint])

def constructActiveIMCEDict(mod):
  for func_name_var, func in mod.functions.items():
    if func_name_var.name_hint == "main": continue
    elif func.attrs["Compiler"]=="imcflow":
      GraphNodeIDs = CustomIDInFunc()[func_name_var.name_hint]
      ActiveIMCEs = set()
      for GraphNodeID in GraphNodeIDs:
        if GraphNodeID in ImcflowDeviceConfig().HWNodeMap and ImcflowDeviceConfig().HWNodeMap[GraphNodeID].is_imce():
          ActiveIMCEs.add(ImcflowDeviceConfig().HWNodeMap[GraphNodeID])
      ImcflowDeviceConfig().ActiveIMCEPerFunc[func_name_var.name_hint] = list(ActiveIMCEs)

def constructNoCPathDict(mod):
  """
  Make NoC path dict from tensor edge list.
  Plus, add instruction path from inode to imce nodes.
  """

  HwMapping = ImcflowDeviceConfig().HWNodeMap
  NocPaths = ImcflowDeviceConfig().NoCPaths
  IMCECOL = ImcflowDeviceConfig.IMCE_W_NUM
  for func_name_var, func in mod.functions.items():
    if func_name_var.name_hint == "main": continue
    elif func.attrs["Compiler"]=="imcflow":
      NocPaths[func_name_var.name_hint] = {}
      # tensor edge to path entry
      TensorEdgeList_ = ImcflowDeviceConfig().TensorEdgeListDict[func_name_var.name_hint]
      for tensor_edge in TensorEdgeList_:
        SrcTensorID = tensor_edge.src_id
        DstTensorID = tensor_edge.dst_id
        SplitIdx = tensor_edge.split_idx
        SrcGraphNode = CustomIDToNode()[getInnerNodeID(SrcTensorID.graph_node_id)]
        DstGraphNode = CustomIDToNode()[getInnerNodeID(DstTensorID.graph_node_id)]
        NocPaths[func_name_var.name_hint][tensor_edge] = (
          (HwMapping[getInnerNodeID(SrcTensorID.graph_node_id)], HwMapping[getInnerNodeID(DstTensorID.graph_node_id)], SplitIdx)
        )
        # if isinstance(SrcGraphNode, (Var, Constant)):
        #   # else, map src node into inode
        #   SrcHwNodeID = HwMapping[getOuterNodeID(SrcTensorID.graph_node_id)]
        #   DstHwNodeID = HwMapping[getOuterNodeID(DstTensorID.graph_node_id)]
        #   NocPaths[func_name_var.name_hint][tensor_edge] = (
        #     (SrcHwNodeID, DstHwNodeID, SplitIdx)
        #   )
        #   # # if "inode" not in DstHwNodeID:
        #   # if not DstHwNodeID.is_inode():
        #   #   InodeID = NodeID.from_inode_coord(NodeID.to_coord(DstHwNodeID)[0])
        #   #   NocPaths[func_name_var.name_hint][tensor_edge] = (
        #   #     (InodeID, DstHwNodeID, SplitIdx)
        #   #   )
        #   #   HwMapping[SrcTensorID.graph_node_id] = InodeID
        # elif hasattr(DstGraphNode, "attrs") and hasattr(DstGraphNode.attrs, "Compiler") and DstGraphNode.attrs["Compiler"] == "imcflow" :
        #   # if this tensoredge is the final edge directly connected to host (= if destination is function)
        #   SrcHwNodeID = HwMapping[getOuterNodeID(SrcTensorID.graph_node_id)]
        #   DstHwNodeID = HwMapping[getOuterNodeID(DstTensorID.graph_node_id)]
        #   NocPaths[func_name_var.name_hint][tensor_edge] = (
        #     (SrcHwNodeID, DstHwNodeID, SplitIdx)
        #   )
        #   # InodeID = NodeID.from_inode_coord(NodeID.to_coord(SrcHwNodeID)[0])
        #   # NocPaths[func_name_var.name_hint][tensor_edge] = (
        #   #   (HwMapping[getOuterNodeID(SrcTensorID.graph_node_id)], InodeID, SplitIdx)
        #   # )
        #   # HwMapping[DstTensorID.graph_node_id] = InodeID
        # else:
        #   NocPaths[func_name_var.name_hint][tensor_edge] = (
        #     (HwMapping[getOuterNodeID(SrcTensorID.graph_node_id)], HwMapping[getOuterNodeID(DstTensorID.graph_node_id)], SplitIdx)
        #   )

      # instruction path
      for DstHwNodeID in NodeID.imces():
        InodeID = NodeID.from_inode_coord(NodeID.to_coord(DstHwNodeID)[0])
        NocPaths[func_name_var.name_hint][DstHwNodeID] = (
          (InodeID, DstHwNodeID, None)
        )

def constructTensorIDToTensorEdgeDict():
  TensorEdgeList = ImcflowDeviceConfig().TensorEdgeList
  TensorEdgeMap = ImcflowDeviceConfig().TensorIDtoEdge
  def _add(tensor_id_, tensor_edge_):
    if tensor_id_ not in TensorEdgeMap.keys():
      TensorEdgeMap[tensor_id_] = tensor_edge_
    elif isinstance(TensorEdgeMap[tensor_id_], list):
      TensorEdgeMap[tensor_id_].append(tensor_edge_)
    else:
      TensorEdgeMap[tensor_id_] = [TensorEdgeMap[tensor_id_], tensor_edge_]
  for tensor_edge in TensorEdgeList:
    SrcID = tensor_edge.src_id
    DstID = tensor_edge.dst_id
    _add(SrcID, tensor_edge)
    _add(DstID, tensor_edge)

class MemoryAllocator:
    """
    Allocate memory block to var, constant, function output.
    Target Operators:
      conv2d, bias_add, batch_norm, relu, add and fused versions
      split, concat
    
    Assumption:
      no edge from inode to inode directly
    """
    def run_(self, func, func_name, ttype_map):
      class _MemoryAllocator(tvm.relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.TensorEdgeList = ImcflowDeviceConfig().TensorEdgeList
            # self.DataBlockDict ={edge: DataBlock(edge.dst_id, None) for edge in self.TensorEdgeList}
            self.DataBlockDict ={}

            self.imce_index = ImcflowDeviceConfig.IMCE_NUM - 1
            self.inode_index = ImcflowDeviceConfig.INODE_NUM - 1

            self.id_dict = HashToCustomID()
            self.name_dict = CustomIDToName()
            self.data = CustomIDToNode()
            self.hwnodemap = ImcflowDeviceConfig().HWNodeMap

            self.func_name = None
            self.ttype_map = None

        def traverse_func(self, func, func_name, ttype_map):
            self.func_name = func_name
            self.ttype_map = ttype_map
            self.visit(func)
            self.allocate()
            return self.DataBlockDict

        def visit_function(self, fn):
          super().visit_function(fn)

          if hasattr(fn.attrs, "Compiler") and fn.attrs["Compiler"]=="imcflow":
            edges = self.find_in_edge_from_list(fn)
            for edge in edges:
              self.add_to_block_dict(edge, fn)
        
        def visit_var(self, var):
          super().visit_var(var)
          edges = self.find_out_edge_from_list(var)
          for edge in edges:
            self.add_to_block_dict(edge, var)
        
        def visit_constant(self, const):
          super().visit_constant(const)
          edges = self.find_out_edge_from_list(const)
          for edge in edges:
            self.add_to_block_dict(edge, const)
          
        def add_to_block_dict(self, edge, node):
            size = self.get_size(edge, node)
            if size > 0:
              datablock = DataBlock(edge, None)
              datablock.set_size(size)
              self.DataBlockDict[edge] = datablock
            else:
              raise ValueError("edge has zero size.")

        def visit_tuple_getitem(self, op):
          super().visit_tuple_getitem(op)

        def visit_tuple(self, op):
          super().visit_tuple(op)

        def is_inode_in_edge(self, edge):
          dst_hw_node_id = None
          src_hw_node_id = None
          is_inode = False
          inode_tensorid = None

          #dst id
          if getInnerNodeID(edge.dst_id.graph_node_id) in self.hwnodemap:
            dst_hw_node_id = self.hwnodemap[getInnerNodeID(edge.dst_id.graph_node_id)]
            if dst_hw_node_id.name.startswith("inode"):
              # determine whether inode is included in the edge and which id it is.
              is_inode = True
              inode_tensorid = edge.dst_id

          #src id
          if getInnerNodeID(edge.src_id.graph_node_id) in self.hwnodemap:
            src_hw_node_id = self.hwnodemap[getInnerNodeID(edge.src_id.graph_node_id)]
            if src_hw_node_id.name.startswith("inode"):
              # determine whether inode is included in the edge and which id it is.
              is_inode = True
              inode_tensorid = edge.src_id

          return is_inode, inode_tensorid
        
        def find_out_edge_from_list(self, call, to_only_inode=False):
          tensor_edge_list = self.TensorEdgeList
          graph_node_id = getNodeID(call)

          def matches_node_id(node_id):
            if isinstance(node_id, (int, tvm.tir.expr.IntImm)):
              return node_id == graph_node_id
            elif isinstance(node_id, tuple):
              return graph_node_id in node_id
            return False

          edges = []
          for edge in tensor_edge_list:
            if matches_node_id(getInnerNodeID(edge.src_id.graph_node_id)) and (not to_only_inode or self.is_inode_in_edge(edge)[0]):
              edges.append(edge)

          return edges

        def find_in_edge_from_list(self, call, from_only_inode=False):
          tensor_edge_list = self.TensorEdgeList
          graph_node_id = getNodeID(call)

          def matches_node_id(node_id):
            if isinstance(node_id, (int, tvm.tir.expr.IntImm)):
              return node_id == graph_node_id
            elif isinstance(node_id, tuple):
              return graph_node_id in node_id
            return False

          edges = []
          for edge in tensor_edge_list:
            if matches_node_id(getInnerNodeID(edge.dst_id.graph_node_id)) and (not from_only_inode or self.is_inode_in_edge(edge)[0]):
              edges.append(edge)

          return edges

        def allocate(self):
          """
          Two-phase memory allocation:
          Phase 1: Collect information about input/output tensors per inode
          Phase 2: Calculate tiling factor and perform actual allocation
          """
          # Phase 1: Collect information
          # {inode_name: {'input': [], 'output': [], 'weight': [], 'other': []}}
          inode_tensors = {}
          
          for edge, mem_block in self.DataBlockDict.items():
            if mem_block.size is None:
              raise ValueError("Memory size cannot be none.")

            _, inode_tensorid = self.is_inode_in_edge(edge)
            hw_node_id = self.hwnodemap[getInnerNodeID(inode_tensorid.graph_node_id)]
            inode_name = hw_node_id.name  # ex) inode_3
            
            if inode_name not in inode_tensors:
              inode_tensors[inode_name] = {
                'input': [],
                'output': [],
                'weight': [],
                'other': []
              }
            
            # Classify tensor type
            tensor_type = inode_tensorid.tensor_type
            
            if tensor_type == "weight":
              inode_tensors[inode_name]['weight'].append((edge, mem_block, inode_tensorid))
            elif tensor_type == "data" or tensor_type == "odata" or tensor_type == "func_out" or tensor_type == "var":
              # Check if this is function input or output
              src_node = self.data.get(getInnerNodeID(edge.src_id.graph_node_id))
              dst_node = self.data.get(getInnerNodeID(edge.dst_id.graph_node_id))
              
              if isinstance(src_node, relay.Var):
                # Function input
                inode_tensors[inode_name]['input'].append((edge, mem_block, inode_tensorid))
              elif isinstance(dst_node, relay.Function):
                # Function output
                inode_tensors[inode_name]['output'].append((edge, mem_block, inode_tensorid))
              else:
                # Intermediate tensor
                inode_tensors[inode_name]['other'].append((edge, mem_block, inode_tensorid))
            else:
              # Other types (bias, min, max, etc.)
              inode_tensors[inode_name]['other'].append((edge, mem_block, inode_tensorid))
          
          # Phase 2: Calculate tiling factor for this function
          tiling_factor = 1
          
          for inode_name, tensors in inode_tensors.items():
            # Calculate total size of input/output tensors for this inode
            input_output_total = 0
            
            for edge, mem_block, _ in tensors['input']:
              input_output_total += mem_block.size
            
            for edge, mem_block, _ in tensors['output']:
              input_output_total += mem_block.size
            
            # Check if tiling is needed
            if input_output_total > ImcflowDeviceConfig.INODE_DATA_MEM_SIZE:
              required_factor = math.ceil(input_output_total / ImcflowDeviceConfig.INODE_DATA_MEM_SIZE)
              tiling_factor = max(tiling_factor, required_factor)
              debug_print(f"  [{self.func_name}] {inode_name}: input/output total = {input_output_total} bytes")
              debug_print(f"    > Memory capacity = {ImcflowDeviceConfig.INODE_DATA_MEM_SIZE} bytes")
              debug_print(f"    > Required tiling factor = {required_factor}")
          
          # Store tiling factor in FunctionInfo
          func_info = ImcflowDeviceConfig().ImcflowFuncMap[self.func_name]
          func_info.tiling_factor = tiling_factor
          
          if tiling_factor > 1:
            debug_print(f"  [{self.func_name}] Tiling factor = {tiling_factor}")
          
          # Phase 3: Perform actual allocation with tiling
          for inode_name, tensors in inode_tensors.items():
            # Allocate weight tensors (no tiling, allow overlap)
            for edge, mem_block, inode_tensorid in tensors['weight']:
              ImcflowDeviceConfig().add_tensor_edge_info(edge, TensorEdgeInfo(data_block=mem_block))
              ImcflowDeviceConfig().MemLayout[self.func_name][f"{inode_name}_data"].allocate(mem_block, phase="init")
            
            # Allocate input tensors (with tiling if needed)
            for edge, mem_block, inode_tensorid in tensors['input']:
              if tiling_factor > 1:
                # Apply tiling: divide size by tiling factor
                # This represents height-wise tiling (axis=2)
                tiled_size = math.ceil(mem_block.size / tiling_factor)
                mem_block.set_size(tiled_size)
                debug_print(f"    Input tensor tiled: {mem_block.size} -> {tiled_size} bytes")
              
              ImcflowDeviceConfig().add_tensor_edge_info(edge, TensorEdgeInfo(data_block=mem_block))
              ImcflowDeviceConfig().MemLayout[self.func_name][f"{inode_name}_data"].allocate(mem_block, phase="exec")
            
            # Allocate output tensors (with tiling if needed)
            for edge, mem_block, inode_tensorid in tensors['output']:
              if tiling_factor > 1:
                # Apply tiling: divide size by tiling factor
                tiled_size = math.ceil(mem_block.size / tiling_factor)
                mem_block.set_size(tiled_size)
                debug_print(f"    Output tensor tiled: {mem_block.size} -> {tiled_size} bytes")
              
              ImcflowDeviceConfig().add_tensor_edge_info(edge, TensorEdgeInfo(data_block=mem_block))
              ImcflowDeviceConfig().MemLayout[self.func_name][f"{inode_name}_data"].allocate(mem_block, phase="exec")
            
            # Allocate other tensors (no tiling)
            for edge, mem_block, inode_tensorid in tensors['other']:
              ImcflowDeviceConfig().add_tensor_edge_info(edge, TensorEdgeInfo(data_block=mem_block))
              ImcflowDeviceConfig().MemLayout[self.func_name][f"{inode_name}_data"].allocate(mem_block, phase="init")

          return

        def get_arg_idx(self, edge, call):
          # edge is input edge to call node
          # find arg index from call by comparing edge's tensorid
          idx = None
          shape = None
          arg_dtype = None
          for i, arg in enumerate(call.args):
            # Determine the source ID based on the type of `arg`
            if isinstance(arg, TupleGetItem):
                src_id = getNodeID(arg.tuple_value)
            else:
                src_id = getNodeID(arg)

            dst_id = getNodeID(call)

            # Check if `src_id` matches the source node in `edge`
            if isinstance(edge.src_id.graph_node_id, tuple):
              if src_id in edge.src_id.graph_node_id:
                idx = i
                shape = call.type_args[idx].shape
                arg_dtype = call.type_args[idx].dtype
            else:
              if src_id == edge.src_id.graph_node_id:
                idx = i
                shape = call.type_args[idx].shape
                arg_dtype = call.type_args[idx].dtype

            # Check if `dst_id` matches the source node in `edge`
            # this is only for the case where src node is Var node, because customID of Var node in subfunction is not the same one in tensoredge.
            if isinstance(edge.dst_id.graph_node_id, tuple):
              if dst_id in edge.dst_id.graph_node_id and isinstance(arg, Var):
                idx = i
                shape = call.type_args[idx].shape
                arg_dtype = call.type_args[idx].dtype

          return idx, shape, arg_dtype

        def get_op_from_id(self, node_id):
            if isinstance(node_id, (int, tvm.tir.expr.IntImm)):
                return self.name_dict[node_id]
            elif isinstance(node_id, tuple):
                return self.name_dict[node_id[1]]
            else:
              raise ValueError("CustomIDToName does not have this node id.")

        def get_size(self, edge, call):
            size = None
            arg_shape = None
            arg_dtype = None

            if isinstance(call, Function): # output edge of function
              size = None
              arg_node = call.body
              if isinstance(arg_node, Tuple):
                # find field of current edge
                target_idx = -1
                for i, field in enumerate(arg_node.fields):
                  if isinstance(edge.src_id.graph_node_id, tuple):
                    if getNodeID(field) in edge.src_id.graph_node_id:
                      target_idx = i
                      break
                      # arg_node = field
                      # func_ret_type = call.ret_type.fields[i]
                  else:
                    if getNodeID(field) == edge.src_id.graph_node_id:
                      target_idx = i
                      break
                      # arg_node = field
                      # func_ret_type = call.ret_type.fields[i]
                assert target_idx != -1, "Cannot find target field index in function return tuple."
                arg_ttype = self.ttype_map[self.func_name][target_idx]
              else:
                arg_ttype = self.ttype_map[self.func_name]
              arg_shape = arg_ttype[0]
              arg_dtype = arg_ttype[1]
            else:
              src_op = self.get_op_from_id(edge.src_id.graph_node_id)

              #find which argument index this edge correspond to find corresponding shape by type_args.shape
              src_node = self.data[getInnerNodeID(edge.src_id.graph_node_id)]
              if isinstance(src_node, relay.Var):
                arg_ttype = self.ttype_map[src_node.name_hint]
                arg_shape, arg_dtype = arg_ttype[0], arg_ttype[1]
              elif isinstance(src_node, relay.Constant):
                arg_shape, arg_dtype = list(src_node.data.shape), str(src_node.data.dtype)
                # _, arg_shape, arg_dtype = self.get_arg_idx(edge, call)
              else:
                raise ValueError("Source node is neither Var nor Constant.")

              # if src_op == "Op(split)":
              #   # when first node of subgraph is split, memoryblock is already allocated by (src: var -> dst: split) case.
              #   arg_shape = -1
              #   raise ValueError("Split operator output edge should not be allocated here.")

            # calculate size for inode memory allocation
            # if arg_shape == -1:
            #   size = -1
            # else:
            size = math.prod(arg_shape)
            if arg_dtype == "int32" or arg_dtype == "uint32":
              size = size * 32 // 8 # dtype is int32 and unit is byte
            elif arg_dtype == "int16" or arg_dtype == "uint16":
              size = size * 16 // 8 # dtype is int16 and unit is byte
            elif arg_dtype == "int8" or arg_dtype == "uint8":
              size = size * 8 // 8 # dtype is int8 and unit is byte
            elif arg_dtype == "uint4":
              size = size * 4 // 8 # dtype is int4 and unit is byte
            else:
              #sanity check
              raise ValueError(f"Unsupported dtype {arg_dtype} in function return type.")

            if size is None:
              raise ValueError("Size cannot be none.")

            return size

      _MemoryAllocator().traverse_func(func, func_name, ttype_map)
      return func

    def run(self, mod, ttype_map):
      imcflow_func_map = ImcflowDeviceConfig().ImcflowFuncMap
      for gv, func in mod.functions.items():
        if isinstance(func, relay.Function) and hasattr(func.attrs, "Compiler") and func.attrs["Compiler"]=="imcflow":
          func_info = imcflow_func_map[gv.name_hint]
          self.run_(func_info.func_node, gv.name_hint, ttype_map[gv.name_hint])

@relay.transform.function_pass(opt_level=0)
class PolicyTableGenerator:
    def __init__(self, NoCPaths):
      self.NoCPaths = NoCPaths
      self.PolicyTable_2D = {}

    def transform_function(self, func, mod, ctx):
      class _PolicyTableGenerator(tvm.relay.ExprVisitor):
        def __init__(self, NoCPaths):
            super().__init__()
            self.NoCPaths = NoCPaths
            self.router_entry_list_temp = {}
            self.Policytable = []
            self.explored_router_list = {}

            # Dictionary to store initial addresses for each source-index pair
            self.start_addr_dict = {}  # {(source, data type): start_address}

            self.table_capacity = 32
            self.InSubFunction = False
            self.SubFunctionMapping = None
            self.SubFunctionNodeID = None
            self.VarProperties = {}

        def generate_policy_table(self):
            # Initialize policy tables for all nodes using NodeID as keys
            policy_tables = {node_id: [] for node_id in NodeID}

            def get_direction(source_coord, dest_coord):
                if source_coord[1] < dest_coord[1]:
                    return "East"
                elif source_coord[1] > dest_coord[1]:
                    return "West"
                elif source_coord[0] < dest_coord[0]:
                    return "South"
                elif source_coord[0] > dest_coord[0]:
                    return "North"
                return None

            def check_path_capacity(path_coords, explored_router_list):
                """Check if all nodes in the path have available capacity"""
                for coord in path_coords:
                    node = NodeID.from_coord(coord[0],coord[1])
                    if len(policy_tables[node]) >= self.table_capacity:
                        if explored_router_list is not None and coord in explored_router_list:
                            continue
                        else:
                            return False
                return True

            def get_path_coords(source_coord, dest_coord, is_xy_routing=True, explored_router_list=None):
                """Get list of coordinates for the path"""
                path_coords = []
                current_coord = source_coord

                if is_xy_routing:
                    # Move horizontally first (X)
                    while current_coord[1] != dest_coord[1]:
                        next_coord = (current_coord[0],
                                    current_coord[1] + (1 if current_coord[1] < dest_coord[1] else -1))
                        path_coords.append(next_coord)
                        current_coord = next_coord

                    # Then vertically (Y)
                    while current_coord[0] != dest_coord[0]:
                        next_coord = (current_coord[0] + (1 if current_coord[0] < dest_coord[0] else -1),
                                    current_coord[1])
                        path_coords.append(next_coord)
                        current_coord = next_coord
                else:
                    # Move vertically first (Y)
                    while current_coord[0] != dest_coord[0]:
                        next_coord = (current_coord[0] + (1 if current_coord[0] < dest_coord[0] else -1),
                                    current_coord[1])
                        path_coords.append(next_coord)
                        current_coord = next_coord

                    # Then horizontally (X)
                    while current_coord[1] != dest_coord[1]:
                        next_coord = (current_coord[0],
                                    current_coord[1] + (1 if current_coord[1] < dest_coord[1] else -1))
                        path_coords.append(next_coord)
                        current_coord = next_coord

                # check policy table's capacity along the designated routing path
                if not check_path_capacity(path_coords, explored_router_list):
                    # If X-Y fails, try Y-X routing
                    path_coords = get_path_coords(source_coord, dest_coord, False, explored_router_list)
                    if not check_path_capacity(path_coords, explored_router_list):
                        raise ValueError("Routing failed for both X-Y and Y-X!")

                #TODO: there may be cases that X-Y and Y-X both fails!!!!!

                return path_coords

            def handle_single_path(edge, mapping_info, init_addr_save=True, router_entry_list=None):
                """Append new entries to policy tables for a single destination"""
                source_node = mapping_info[0]
                dest_node = mapping_info[1]
                dest_index = mapping_info[2]
                if isinstance(edge, NodeID):
                  source_node_data_type = f"instruction_{edge.name}"
                else:
                  source_node_data_type = edge.src_id.tensor_type

                source_coord = NodeID.to_coord(source_node)
                dest_coord = NodeID.to_coord(dest_node)
                entry_addr = len(policy_tables[source_node])

                if router_entry_list is None: # initial handling
                    router_entry_list= []
                    if source_coord == dest_coord: # if same node, return
                        return
                    # check if there's previous path with same source and same tensor type, which means multicast
                    elif (source_node, source_node_data_type) in self.start_addr_dict:
                        handle_multicast(edge, mapping_info)
                        return
                    else:
                        self.start_addr_dict[(source_node, source_node_data_type)] = entry_addr # each source can have several tensor type

                # Try X-Y routing first
                path_coords = get_path_coords(source_coord, dest_coord, True)
                if (source_node, source_node_data_type) not in self.explored_router_list:
                    self.explored_router_list[(source_node, source_node_data_type)] = path_coords
                else:
                    self.explored_router_list[(source_node, source_node_data_type)].extend(path_coords)

                current_coord = source_coord
                current_node = source_node
                # Apply the successful path to tables
                for next_coord in path_coords:
                    direction = get_direction(current_coord, next_coord)
                    next_node = NodeID.from_coord(next_coord[0], next_coord[1])

                    #append entry to router's policy table
                    entry = {"Local": {"enable": False, "chunk_index": 0, "addr": 0}, \
                      "North": {"enable": False, "addr": 0}, \
                      "South": {"enable": False, "addr": 0}, \
                      "East": {"enable": False, "addr": 0},  \
                      "West": {"enable": False, "addr": 0}}

                    target_addr = len(policy_tables[next_node])
                    entry[direction]["addr"] = target_addr
                    entry[direction]["enable"] = True
                    policy_tables[current_node].append(entry)

                    #create RouterEntry and append to router_entry_list
                    router_entry_list.append((current_node, len(policy_tables[current_node])-1))

                    #switch to next node
                    current_coord = next_coord
                    current_node = NodeID.from_coord(current_coord[0], current_coord[1])

                # insert entry for destination node
                entry = {"Local": {"enable": True, "chunk_index": dest_index, "addr": 0}, \
                  "North": {"enable": False, "addr": 0}, \
                  "South": {"enable": False, "addr": 0}, \
                  "East": {"enable": False, "addr": 0},  \
                  "West": {"enable": False, "addr": 0}}

                policy_tables[dest_node].append(entry)

                #create RouterEntry and append to RouterEntry_list
                router_entry_list.append((dest_node, len(policy_tables[dest_node])-1))

                # temporary saving. Final saving is done after whole paths finish.
                self.router_entry_list_temp[edge] = router_entry_list

            def handle_multicast(edge, mapping_info):
                """Handle multiple destinations with potential path sharing"""
                source_node = mapping_info[0]
                dest_node = mapping_info[1]
                # dest_index = mapping_info[2]
                if isinstance(edge, NodeID):
                  source_node_data_type = f"instruction_{edge.name}"
                else:
                  source_node_data_type = edge.src_id.tensor_type

                router_entry_list= []

                if source_node == dest_node: # if same node, return
                    return

                # Follow existing path and modify at divergence point
                entry_addr = self.start_addr_dict[(source_node, source_node_data_type)]
                current_node = source_node
                current_coord = NodeID.to_coord(current_node)
                dest_coord = NodeID.to_coord(dest_node)
                next_coord = None

                while current_coord != dest_coord:
                    entry = policy_tables[current_node][entry_addr] # current policy table entry

                    # Find which direction to go next.
                    path_coords = get_path_coords(current_coord, dest_coord, self.explored_router_list[(source_node, source_node_data_type)])
                    next_coord = path_coords[0]
                    next_node = NodeID.from_coord(next_coord[0],next_coord[1])
                    direction = get_direction(current_coord, next_coord)

                    # If direction is different from previous path, diverge!
                    if entry[direction]["enable"] is False:
                        # modify entry
                        target_addr = len(policy_tables[next_node])
                        policy_tables[current_node][entry_addr][direction]["addr"] = target_addr
                        policy_tables[current_node][entry_addr][direction]["enable"] = True

                        #create RouterEntry and append to router_entry_list
                        router_entry_list.append((current_node, entry_addr))

                        # diverge into new path
                        new_mapping = (next_node, mapping_info[1], mapping_info[2])
                        handle_single_path(edge, new_mapping, init_addr_save=False, router_entry_list=router_entry_list)
                        break
                    else:
                        # create RouterEntry and append to router_entry_list
                        router_entry_list.append((current_node, entry_addr))

                        # keep following the previous path
                        current_coord = next_coord
                        current_node = next_node
                        entry_addr = entry[direction]["addr"]

                        if current_node == dest_node: # if same node, return
                            policy_tables[dest_node][entry_addr]["Local"]["enable"] = True
                            # create RouterEntry and append to router_entry_list
                            router_entry_list.append((current_node, entry_addr))
                            # temporary saving. Final saving is done after whole paths finish.
                            self.router_entry_list_temp[edge] = router_entry_list
                            break

            # Main logic
            for edge, mapping_info in self.NoCPaths.items():
                handle_single_path(edge, mapping_info)

            self.Policytable = policy_tables
            ImcflowDeviceConfig().PolicyTableDict = policy_tables

        def add_EdgeInfo(self):
            # def get_meminfo(edge):
            #     if isinstance(edge.src_id, tuple):
            #         id = edge.src_id[1]
            #     else:
            #         id = edge.src_id

            #     size = self.DataBlockDict[id]["size"]
            #     offset = self.DataBlockDict[id]["offset"]
            #     base_address = self.DataBlockDict[id]["base_address"]
            #     meminfo = DataBlock(id, size)

            #     meminfo.set_offset(offset)
            #     meminfo.set_base_address(base_address)

            #     return meminfo

            # after policy table entry generation finished, add to TensorEdgeToInfo
            fifo_id_cnt = {node_id: 2 for node_id in NodeID}
            ID_dict = CustomIDToName()
            for edge, mapping_info in self.NoCPaths.items():
              # if tensoredge, save to TensorEdgetoInfo
              dest_node = mapping_info[1]
              router_entry_list=[]
              if edge in self.router_entry_list_temp:
                  for entry_tuple in self.router_entry_list_temp[edge]:
                      router_entry_list.append(RouterEntry(entry_tuple[0], entry_tuple[1], self.Policytable[entry_tuple[0]][entry_tuple[1]]))

                  if isinstance(edge, TensorEdge): # TensorEdge
                      # find mem_info
                      # meminfo = get_meminfo(edge) # decided to erase MemoryBlock in EdgeInfo

                      # FIFO ID assign
                      # 0: conv input
                      # 1: const (including weight)
                      # 2~6: rest
                      # edgeinfo = ImcflowDeviceConfig().get_tensor_edge_info(edge)
                      # edgeinfo.set_policy_info(router_entry_list)

                      if edge.src_id.tensor_type in ["odata", "var"]:
                        # get src node name from CustomIDToName
                        dst_node_name = ID_dict[getInnerNodeID(edge.dst_id.graph_node_id)]

                        if dst_node_name == "nn.imcflow_qconv":
                          # if src is input of qconv, FIFO ID = 0
                          # edgeinfo.set_fifo_id(0)
                          edgeinfo = TensorEdgeInfo(router_entry_list, None, 0)
                          ImcflowDeviceConfig().add_tensor_edge_info(edge, edgeinfo)
                        else:
                          # if not, FIFO ID = 2~6
                          # edgeinfo.set_fifo_id(fifo_id_cnt[dest_node])
                          edgeinfo = TensorEdgeInfo(router_entry_list, None, fifo_id_cnt[dest_node])
                          ImcflowDeviceConfig().add_tensor_edge_info(edge, edgeinfo)

                          fifo_id_cnt[dest_node] = fifo_id_cnt[dest_node] + 1
                          if fifo_id_cnt[dest_node] >= 8:
                            raise ValueError("FIFO ID cannot be over 7!")

                      elif edge.src_id.tensor_type in ["odata", "weight", "bias", "fused_scale", "fused_bias", "min", "max", "threshold", "scale", "config"]:
                        # if const, FIFO ID = 1
                        edgeinfo = TensorEdgeInfo(router_entry_list, None, 1)
                        ImcflowDeviceConfig().add_tensor_edge_info(edge, edgeinfo)
                      else:
                        raise ValueError("Wrong tensor type!")

                  else: # Instruction edge
                      # meminfo = get_meminfo(edge) # decided to erase MemoryBlock in EdgeInfo
                      edgeinfo = InstEdgeInfo(router_entry_list, None)
                      ImcflowDeviceConfig().add_inst_edge_info(edge, edgeinfo)

        def allocate(self, func_name):
          # Allocate memory for policy tables
          for node_id, policy_table in self.Policytable.items():
            if len(policy_table) == 0:
                continue
            mem_size = len(policy_table) * 32
            mem_block = DataBlock(f"{node_id.name}_policy", mem_size)
            inode_id = node_id.master() if node_id.is_imce() else node_id
            ImcflowDeviceConfig().MemLayout[func_name][f"{inode_id.name}_data"].allocate(mem_block, phase="init")

        def update_device_config(self, func_name):
            # traverse input function by visit() to make PathDict and generate policy table for it
            self.generate_policy_table()
            self.add_EdgeInfo()
            self.allocate(func_name)
            return self.Policytable

      # Returns list of (GlobalVar, Function) pairs sorted alphabetically by function name
      for gv, func in mod.functions.items():
        if isinstance(func, relay.Function) and hasattr(func.attrs, "Compiler") and func.attrs["Compiler"]=="imcflow":
          self.PolicyTable_2D[gv.name_hint] = _PolicyTableGenerator(self.NoCPaths[gv.name_hint]).update_device_config(gv.name_hint)
          for x in self.PolicyTable_2D[gv.name_hint]:
            print(x)

      return func

class TensorPathVisualizer:
    """
    Visualizes tensor routing paths in the 2D mesh NoC topology.
    
    For each imcflow function, generates an image showing:
    - 2D mesh grid with inodes and imces as labeled squares
    - Tensor paths as colored lines between nodes
    - Each tensor gets a unique color
    """
    
    def __init__(self, output_dir="noc_visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Import visualization libraries
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.collections import LineCollection
            self.plt = plt
            self.patches = patches
            self.LineCollection = LineCollection
        except ImportError:
            raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")
    
    def visualize_all_functions(self, mod):
        """
        Generate visualizations for all imcflow functions in the module.
        
        Parameters
        ----------
        mod : tvm.IRModule
            The module containing imcflow functions
        """
        for gv, func in mod.functions.items():
            if isinstance(func, relay.Function) and hasattr(func.attrs, "Compiler") and func.attrs["Compiler"] == "imcflow":
                func_name = gv.name_hint
                debug_print(f"Generating visualization for function: {func_name}")
                self.visualize_function(func_name)
    
    def visualize_function(self, func_name):
        """
        Generate visualization for a single imcflow function.
        Creates separate images for each tensor type (odata, weight, bias, etc.)
        
        Parameters
        ----------
        func_name : str
            Name of the imcflow function
        """
        # Get NoC paths for this function
        if func_name not in ImcflowDeviceConfig().NoCPaths:
            debug_print(f"No NoC paths found for function {func_name}")
            return
        
        noc_paths = ImcflowDeviceConfig().NoCPaths[func_name]
        tensor_edge_list = ImcflowDeviceConfig().TensorEdgeListDict.get(func_name, [])
        
        # Create subdirectory for this function
        func_output_dir = os.path.join(self.output_dir, func_name)
        os.makedirs(func_output_dir, exist_ok=True)
        
        # Group NoC paths by tensor type
        paths_by_type = {}
        for edge, mapping_info in noc_paths.items():
            if isinstance(edge, TensorEdge):
                tensor_type = edge.src_id.tensor_type
                if tensor_type not in paths_by_type:
                    paths_by_type[tensor_type] = {}
                paths_by_type[tensor_type][edge] = mapping_info
        
        # Create a visualization for each tensor type
        if not paths_by_type:
            debug_print(f"No tensor edges found for function {func_name}")
            return
        
        # Create individual visualizations for each tensor type
        for tensor_type, type_paths in sorted(paths_by_type.items()):
            debug_print(f"  Creating visualization for {tensor_type}: {len(type_paths)} paths")
            
            # Create the visualization
            fig, ax = self._create_mesh_grid(title=f"{func_name} - {tensor_type} Paths")
            
            # Draw tensor paths for this type only
            self._draw_tensor_paths(ax, type_paths, tensor_edge_list)
            
            # Save the figure
            output_path = os.path.join(func_output_dir, f"{tensor_type}.png")
            self.plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.plt.close(fig)
            
            debug_print(f"    Saved: {output_path}")
        
        # Also create an overview image with all tensor types
        debug_print(f"  Creating overview with all {len(paths_by_type)} tensor types")
        fig, ax = self._create_mesh_grid(title=f"{func_name} - All Tensor Paths (Overview)")
        
        # Collect all tensor edges
        all_type_paths = {}
        for type_paths in paths_by_type.values():
            all_type_paths.update(type_paths)
        
        self._draw_tensor_paths(ax, all_type_paths, tensor_edge_list)
        
        overview_path = os.path.join(func_output_dir, "00_overview_all_types.png")
        self.plt.savefig(overview_path, dpi=300, bbox_inches='tight')
        self.plt.close(fig)
        
        debug_print(f"    Saved: {overview_path}")
        debug_print(f"Completed visualization for {func_name}: {len(paths_by_type)} tensor types")
    
    def _create_mesh_grid(self, title="NoC Tensor Routing Paths"):
        """
        Create the 2D mesh grid with nodes.
        
        Parameters
        ----------
        title : str, optional
            Title for the visualization
        
        Returns
        -------
        fig, ax : matplotlib figure and axis
        """
        # Grid dimensions
        rows = ImcflowDeviceConfig.INODE_NUM  # 4 rows
        cols = ImcflowDeviceConfig.NODE_COL_NUM  # 5 columns (1 inode + 4 imces)
        
        # Node size and spacing
        node_size = 1.0
        spacing = 0.5
        
        # Calculate figure size
        fig_width = cols * (node_size + spacing) + spacing
        fig_height = rows * (node_size + spacing) + spacing
        
        fig, ax = self.plt.subplots(figsize=(fig_width * 2, fig_height * 2))
        
        # Draw each node
        for node_id in NodeID:
            coord = NodeID.to_coord(node_id)
            row, col = coord
            
            # Calculate position (flip y-axis so row 0 is at top)
            x = col * (node_size + spacing) + spacing
            y = (rows - 1 - row) * (node_size + spacing) + spacing
            
            # Determine node color
            if node_id.is_inode():
                color = 'lightblue'
                edgecolor = 'darkblue'
            else:
                color = 'lightgreen'
                edgecolor = 'darkgreen'
            
            # Draw node as rectangle
            rect = self.patches.Rectangle((x, y), node_size, node_size, 
                                         linewidth=2, edgecolor=edgecolor, 
                                         facecolor=color, alpha=0.7)
            ax.add_patch(rect)
            
            # Add node label
            ax.text(x + node_size/2, y + node_size/2, node_id.name,
                   ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Set axis properties
        ax.set_xlim(0, fig_width)
        ax.set_ylim(0, fig_height)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        return fig, ax
    
    def _draw_tensor_paths(self, ax, noc_paths, tensor_edge_list):
        """
        Draw tensor paths on the mesh grid.
        
        Parameters
        ----------
        ax : matplotlib axis
            The axis to draw on
        noc_paths : dict
            Dictionary mapping edges to (source_node, dest_node, dest_index) tuples
        tensor_edge_list : list
            List of TensorEdge objects for this function
        """
        # Get unique colors for each tensor edge
        num_tensor_edges = len([e for e in noc_paths.keys() if isinstance(e, TensorEdge)])
        colors = self._generate_colors(num_tensor_edges)
        
        # Node size and spacing (must match _create_mesh_grid)
        node_size = 1.0
        spacing = 0.5
        rows = ImcflowDeviceConfig.INODE_NUM
        
        # Track which tensor edges we've drawn
        tensor_edge_idx = 0
        legend_entries = []
        
        # Track edge segments to add offsets for overlapping paths
        segment_usage = {}  # ((x1,y1), (x2,y2)) -> count
        
        for edge, mapping_info in noc_paths.items():
            # Only visualize TensorEdge (not instruction edges)
            if not isinstance(edge, TensorEdge):
                continue
            
            source_node = mapping_info[0]
            dest_node = mapping_info[1]
            
            # Get color for this edge
            color = colors[tensor_edge_idx % len(colors)]
            tensor_edge_idx += 1
            
            # Get the full path by looking up router entries
            if edge in ImcflowDeviceConfig().TensorEdgetoInfo:
                edge_info = ImcflowDeviceConfig().TensorEdgetoInfo[edge]
                if edge_info.policy_info:
                    # Extract path from router entries
                    path_nodes = [entry.router_id for entry in edge_info.policy_info]
                    
                    # Convert path to coordinates and draw
                    path_coords = []
                    for node_id in path_nodes:
                        coord = NodeID.to_coord(node_id)
                        row, col = coord
                        x = col * (node_size + spacing) + spacing + node_size/2
                        y = (rows - 1 - row) * (node_size + spacing) + spacing + node_size/2
                        path_coords.append((x, y))
                    
                    # Draw the path with offsets to avoid overlap
                    if len(path_coords) > 1:
                        offset_coords = []
                        for i, (x, y) in enumerate(path_coords):
                            if i > 0:
                                # Calculate offset based on segment usage
                                prev_pt = path_coords[i-1]
                                curr_pt = (x, y)
                                segment = (prev_pt, curr_pt)
                                segment_rev = (curr_pt, prev_pt)
                                
                                # Count usage (consider both directions as same segment)
                                if segment in segment_usage:
                                    offset_idx = segment_usage[segment]
                                    segment_usage[segment] += 1
                                elif segment_rev in segment_usage:
                                    offset_idx = segment_usage[segment_rev]
                                    segment_usage[segment_rev] += 1
                                else:
                                    offset_idx = 0
                                    segment_usage[segment] = 1
                                
                                # Apply perpendicular offset
                                dx = curr_pt[0] - prev_pt[0]
                                dy = curr_pt[1] - prev_pt[1]
                                length = (dx**2 + dy**2)**0.5
                                if length > 0:
                                    # Perpendicular direction
                                    perp_x = -dy / length
                                    perp_y = dx / length
                                    # Offset amount (alternate positive/negative)
                                    offset_amount = 0.08 * (offset_idx + 1) * (1 if offset_idx % 2 == 0 else -1)
                                    x_offset = x + perp_x * offset_amount
                                    y_offset = y + perp_y * offset_amount
                                    offset_coords.append((x_offset, y_offset))
                                else:
                                    offset_coords.append((x, y))
                            else:
                                offset_coords.append((x, y))
                        
                        xs, ys = zip(*offset_coords)
                        line = ax.plot(xs, ys, color=color, linewidth=2.5, alpha=0.8, 
                                      marker='o', markersize=5, markeredgecolor='white', 
                                      markeredgewidth=0.5, zorder=10)
                        
                        # Add arrow at the end
                        if len(offset_coords) >= 2:
                            dx = offset_coords[-1][0] - offset_coords[-2][0]
                            dy = offset_coords[-1][1] - offset_coords[-2][1]
                            length = (dx**2 + dy**2)**0.5
                            if length > 0:
                                ax.arrow(offset_coords[-2][0], offset_coords[-2][1], dx*0.6, dy*0.6,
                                       head_width=0.2, head_length=0.15, fc=color, ec=color, 
                                       alpha=0.9, linewidth=1.5, zorder=11)
                        
                        # Create legend entry with NodeID and CustomID information
                        src_node_name = source_node.name
                        dst_node_name = dest_node.name
                        tensor_type = edge.src_id.tensor_type
                        
                        # Get custom IDs from the tensor edge
                        src_custom_id = edge.src_id.graph_node_id
                        dst_custom_id = edge.dst_id.graph_node_id
                        
                        # Format custom IDs (handle tuples for composite functions)
                        src_id_str = f"{src_custom_id[1]}" if isinstance(src_custom_id, tuple) else f"{src_custom_id}"
                        dst_id_str = f"{dst_custom_id[1]}" if isinstance(dst_custom_id, tuple) else f"{dst_custom_id}"
                        
                        tensor_label = f"{src_node_name} → {dst_node_name} | ID:{src_id_str}→{dst_id_str} ({tensor_type})"
                        if edge.split_idx is not None:
                            tensor_label += f"[{edge.split_idx}]"
                        legend_entries.append((line[0], tensor_label))
            else:
                # Fallback: draw direct line from source to dest
                src_coord = NodeID.to_coord(source_node)
                dst_coord = NodeID.to_coord(dest_node)
                
                src_x = src_coord[1] * (node_size + spacing) + spacing + node_size/2
                src_y = (rows - 1 - src_coord[0]) * (node_size + spacing) + spacing + node_size/2
                dst_x = dst_coord[1] * (node_size + spacing) + spacing + node_size/2
                dst_y = (rows - 1 - dst_coord[0]) * (node_size + spacing) + spacing + node_size/2
                
                line = ax.plot([src_x, dst_x], [src_y, dst_y], 
                             color=color, linewidth=2.5, alpha=0.8, marker='o', 
                             markersize=5, markeredgecolor='white', markeredgewidth=0.5, zorder=10)
                
                # Add arrow
                dx = dst_x - src_x
                dy = dst_y - src_y
                length = (dx**2 + dy**2)**0.5
                if length > 0:
                    ax.arrow(src_x, src_y, dx*0.6, dy*0.6,
                           head_width=0.2, head_length=0.15, fc=color, ec=color, 
                           alpha=0.9, linewidth=1.5, zorder=11)
                
                # Create legend entry with NodeID and CustomID information
                src_node_name = source_node.name
                dst_node_name = dest_node.name
                tensor_type = edge.src_id.tensor_type
                
                # Get custom IDs from the tensor edge
                src_custom_id = edge.src_id.graph_node_id
                dst_custom_id = edge.dst_id.graph_node_id
                
                # Format custom IDs (handle tuples for composite functions)
                src_id_str = f"{src_custom_id[1]}" if isinstance(src_custom_id, tuple) else f"{src_custom_id}"
                dst_id_str = f"{dst_custom_id[1]}" if isinstance(dst_custom_id, tuple) else f"{dst_custom_id}"
                
                tensor_label = f"{src_node_name} → {dst_node_name} | ID:{src_id_str}→{dst_id_str} ({tensor_type})"
                if edge.split_idx is not None:
                    tensor_label += f"[{edge.split_idx}]"
                legend_entries.append((line[0], tensor_label))
        
        # Add legend if there are paths
        if legend_entries:
            lines, labels = zip(*legend_entries)
            ax.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.02, 1), 
                     fontsize=7, framealpha=0.95, edgecolor='gray', fancybox=True)
    
    def _generate_colors(self, n):
        """
        Generate n distinct colors for visualization.
        
        Parameters
        ----------
        n : int
            Number of colors to generate
            
        Returns
        -------
        list of color strings
        """
        if n == 0:
            return []
        
        # Use a colormap to generate distinct colors
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        if n <= 10:
            # Use tab10 for small number of colors
            cmap = cm.get_cmap('tab10')
            return [cmap(i) for i in range(n)]
        elif n <= 20:
            # Use tab20 for medium number
            cmap = cm.get_cmap('tab20')
            return [cmap(i) for i in range(n)]
        else:
            # Use hsv for large number
            cmap = cm.get_cmap('hsv')
            return [cmap(i/n) for i in range(n)]

def generateNoCVisualizations(mod, output_dir="noc_visualizations"):
    """
    Generate NoC path visualizations for all imcflow functions in the module.
    
    This function should be called after PolicyTableGenerator has run and
    populated ImcflowDeviceConfig with NoC paths and tensor edge information.
    
    For each imcflow function, creates:
    - A subdirectory named after the function
    - Separate images for each tensor type (odata.png, weight.png, bias.png, etc.)
    - An overview image showing all tensor types together (00_overview_all_types.png)
    
    Parameters
    ----------
    mod : tvm.IRModule
        The module containing imcflow functions
    output_dir : str, optional
        Base directory to save visualization images (default: "noc_visualizations")
    
    Output Structure
    ----------------
    noc_visualizations/
        function_name_1/
            00_overview_all_types.png
            odata.png
            weight.png
            bias.png
            ...
        function_name_2/
            00_overview_all_types.png
            odata.png
            ...
    
    Example
    -------
    >>> # After running PolicyTableGenerator
    >>> generateNoCVisualizations(mod, "my_visualizations")
    """
    visualizer = TensorPathVisualizer(output_dir=output_dir)
    visualizer.visualize_all_functions(mod)
    debug_print(f"NoC visualizations saved to: {output_dir}")

def clearPrimitiveTag(mod):
  class _Visitor(tvm.relay.ExprMutator):
    def visit_function(self, fn):
      fn = super().visit_function(fn)

      NewAttrs = {}
      for key in fn.attrs.keys():
        NewAttrs[key] = fn.attrs.get_str(key)
      if "Primitive" in NewAttrs.keys():
        del NewAttrs["Primitive"]

      return FunctionWithFields(fn, list(fn.params), fn.body, fn.ret_type, fn.type_params, tvm.ir.make_node("DictAttrs", **NewAttrs))

    def visit_call(self, call):
      if isinstance(call.op, relay.Function) and "Composite" in call.op.attrs and re.match(r"imcflow\..*", call.op.attrs["Composite"]):
        var_map = {}
        for arg, param in zip(call.args, call.op.params):
          var_map[param] = super().visit(arg)
        new_body = relay.bind(super().visit(call.op.body), var_map)
        return new_body
      else:
        return super().visit_call(call)

  for func_name in mod.functions:
    mod[func_name] = _Visitor().visit(mod[func_name])

  return mod

def clearCompilerAttr(mod):
  class _Visitor(tvm.relay.ExprMutator):
    def visit_function(self, fn):
      fn = super().visit_function(fn)

      NewAttrs = {}
      for key in fn.attrs.keys():
        NewAttrs[key] = fn.attrs.get_str(key)
      if "Compiler" in NewAttrs.keys():
        del NewAttrs["Compiler"]

      return FunctionWithFields(fn, list(fn.params), fn.body, fn.ret_type, fn.type_params, tvm.ir.make_node("DictAttrs", **NewAttrs))

  for func_name in mod.functions:
    mod[func_name] = _Visitor().visit(mod[func_name])

  return mod

def constructImcflowFuncMap(mod):
  from tvm.contrib.imcflow import FunctionInfo
  
  imcflow_func_map = {}
  class FirstFuncVisitor(tvm.relay.ExprVisitor):
    def __init__(self):
      super().__init__()
      self.first_func = None
    def visit_call(self, call):
      if isinstance(call.op, relay.Function):
        if self.first_func is None:
          self.first_func = call.op
      super().visit_call(call)

  for func_name in mod.functions:
    if "imcflow" in func_name.name_hint:
      wrap_func = mod[func_name.name_hint]
      visitor = FirstFuncVisitor()
      visitor.visit(wrap_func)
      imcflow_func_map[func_name.name_hint] = FunctionInfo(
        func_node=visitor.first_func,
        tiling_factor=1  # Initial value
      )

  ImcflowDeviceConfig().ImcflowFuncMap = imcflow_func_map

def annotateCustomId(mod):
  class _Visitor(tvm.relay.ExprMutator):
    def __init__(self):
      super().__init__()
      self.cnt = 0

    def update_attrs(self, origin_attrs, updates):
      new_attr_dict = {}
      if origin_attrs is not None:
        for key in origin_attrs.keys():
          attr_value = origin_attrs[key]

          if isinstance(attr_value, tvm.ir.container.Array):
            # Convert array to tuple for proper handling
            new_attr_dict[str(key)] = tuple(attr_value)
          else:
            new_attr_dict[str(key)] = attr_value

      new_attr_dict.update(updates)

      if isinstance(origin_attrs, tvm.ir.attrs.DictAttrs):
        attr_type = "DictAttrs"
      elif origin_attrs is not None:
        attr_type = str(origin_attrs).split("(")[0]
      else:
        attr_type = "DictAttrs"
        # return None

      return tvm.ir.make_node(attr_type, **new_attr_dict)

    def visit_call(self, call):
      new_call = super().visit_call(call)
      self.cnt = self.cnt + 1
      origin_attrs = new_call.attrs
      new_attrs = self.update_attrs(origin_attrs, {"custom_id": self.cnt})
      return _expr.CallWithFields(new_call, new_call.op, new_call.args, new_attrs, new_call.type_args, new_call.span)

    def visit_function(self, fn):
      new_fn = super().visit_function(fn)
      self.cnt = self.cnt + 1
      origin_attrs = new_fn.attrs
      new_attrs = self.update_attrs(origin_attrs, {"custom_id": self.cnt})
      return FunctionWithFields(new_fn, list(new_fn.params), new_fn.body, new_fn.ret_type, new_fn.type_params, new_attrs)

  visitor = _Visitor()
  for func_name in mod.functions:
    mod[func_name] = visitor.visit(mod[func_name])

  return mod

def constructUsefulMappings(mod):
  id_dict = HashToCustomID()
  name_dict = CustomIDToName()
  data = CustomIDToNode()
  class _Visitor(tvm.relay.ExprVisitor):
    def __init__(self):
      super().__init__()
      self.Cnt = -2

    def visit_call(self, call):
      # id_dict[int(hash(call))] = self.Cnt
      # name_dict[self.Cnt] = getNodeDebugID(call)
      # data[id_dict[int(hash(call))]] = call
      # self.Cnt = self.Cnt + 1

      id_dict[int(hash(call))] = int(call.attrs["custom_id"])
      name_dict[call.attrs["custom_id"]] = getNodeDebugID(call)
      data[id_dict[int(hash(call))]] = call

      super().visit_call(call)

    def visit_function(self, call):
      id_dict[int(hash(call))] = int(call.attrs["custom_id"])
      name_dict[call.attrs["custom_id"]] = "Function"
      data[id_dict[int(hash(call))]] = call

      # id_dict[int(hash(call))] = self.Cnt
      # name_dict[self.Cnt] = "Function"
      # data[id_dict[int(hash(call))]] = call
      # self.Cnt = self.Cnt + 1

      super().visit_function(call)

    def visit_var(self, var):
      # id_dict[int(hash(var))] = self.Cnt
      # name_dict[self.Cnt] = var.name_hint
      # data[id_dict[int(hash(var))]] = var
      # self.Cnt = self.Cnt + 1

      id_dict[int(hash(var))] = self.Cnt
      name_dict[self.Cnt] = var.name_hint
      data[id_dict[int(hash(var))]] = var
      self.Cnt = self.Cnt - 1

      super().visit_var(var)

    def visit_constant(self, const):
      # id_dict[int(hash(const))] = self.Cnt
      # name_dict[self.Cnt] = "Const"
      # data[id_dict[int(hash(const))]] = const
      # self.Cnt = self.Cnt + 1

      id_dict[int(hash(const))] = self.Cnt
      name_dict[self.Cnt] = "Const"
      data[id_dict[int(hash(const))]] = const
      self.Cnt = self.Cnt - 1

      super().visit_constant(const)

  vis = _Visitor()
  for func_name in mod.functions:
    vis.visit(mod[func_name])

def constructCustomIDInFunc(mod):
  data = CustomIDInFunc()
  class _Visitor(tvm.relay.ExprVisitor):
    def __init__(self, func_name):
      super().__init__()
      self.func_name = func_name
      data[func_name] = []

    def visit_call(self, call):
      data[self.func_name].append(getNodeID(call))
      super().visit_call(call)

  for func_name in mod.functions:
    if "imcflow" in func_name.name_hint: _Visitor(func_name.name_hint).visit(mod[func_name.name_hint])

#TODO: DataBlock -> TVM name. consider difference between function parameter, constant, instruction
class CodeWriter:
    def __init__(self, indent_str="  "):
        self.lines = []
        self.indent_str = indent_str
        self.indent_level = 0

    def getIndent(self):
      return self.indent_level

    def setIndent(self, indent_level):
      self.indent_level = indent_level

    def applyIndent(self, indent_level):
      for idx, line in enumerate(self.lines):
        line_ = indent_level * self.indent_str + line.lsstrip()
        self.lines[idx] = line_

    def nextIndent(self):
      self.indent_level += 1
      return self

    def prevIndent(self):
      self.indent_level -= 1
      return self

    def write(self, line=""):
        for line_ in line.split("\n"):
          if len(line_) > 0:
            self.lines.append(f"{self.indent_str * self.indent_level}{line_}")

    def get_code(self):
        return "\n".join(self.lines)

    def __str__(self):
        return self.get_code()

    def __add__(self, other):
      if isinstance(other, CodeWriter):
        self.lines.extend(other.lines)
        return self
      elif isinstance(other, str):
        self.write(other)
        return self

def dtype_to_cpp(dtype: str) -> str:
    mapping = {
        "float32": "float",
        "float": "float",
        "int32": "int32_t",
        "int16" : "int16_t",
        "int8": "int8_t",
        "uint8": "uint8_t",
        "float64": "double",
    }

    # if dtype not in mapping: print(dtype)
    return mapping.get(dtype, "unknown_type")

def getConstantIdx(func, node_id):
  node_id_to_constant_id = {}
  class _Visitor(tvm.relay.ExprVisitor):
    def __init__(self):
      super().__init__()
      self.Cnt = 0

    def visit_constant(self, const):
      node_id_to_constant_id[getNodeID(const)] = self.Cnt
      self.Cnt = self.Cnt + 1
      super().visit_constant(const)

  _Visitor().visit(func)
  return node_id_to_constant_id[node_id]

def generateConstantArrayDecl(func_name, func):
  return ""

def generateConstantArrayInit(func_name, func):
  return CodeWriter()

def makeWrapper(func, func_name):
  # parameter spec -> number of params and types
  params = func.params
  proto_list = []
  cast_list = []
  for i, param in enumerate(params):
    # Use the existing var name; if absent, fall back to "arg{i}"
    param_name = param.name_hint if param.name_hint else f"arg{i}"
    # Default dtype is float32, override if checked_type is present and is a TensorType.
    dtype = "float32"
    if hasattr(param, "checked_type") and isinstance(param.checked_type, TensorType):
      dtype = param.checked_type.dtype
      cpp_type = dtype_to_cpp(dtype)
      proto_list.append(f"DLTensor* {param_name}")
      cast_list.append(f"static_cast<{cpp_type}*>({param_name}->data)")

  output_node = getOutputNodeOfFunc(func)
  output_node_type = output_node.checked_type
  proto_list.append(f"DLTensor* out0")
  cast_list.append(f"static_cast<{dtype_to_cpp(output_node_type.dtype)}*>(out0->data)")

  args_proto_type = ", ".join(proto_list)
  args_type_cast = ", ".join(cast_list)

  code = CodeWriter()
  code += f'extern "C" void {func_name}_wrapper({args_proto_type}) {{\n'
  code += f'  {func_name}_kernel('
  code += f'    {args_type_cast}'
  code += f'  );\n'
  code += '}\n'

  return code

def convert_compiler_regions_to_composite(mod):
  """Convert compiler_begin/compiler_end regions to composite functions."""

  class _CompositeConverter(tvm.relay.ExprMutator):
    def __init__(self):
      super().__init__()
      self.composite_counter = 0
      # State used during single region extraction
      self._begin_to_param = None
      self._params = None
      self._inputs = None

    def _infer_type(self, expr):
      # Try to obtain checked_type if already available; otherwise try local inference
      try:
        if hasattr(expr, "checked_type") and expr.checked_type is not None:
          return expr.checked_type
      except Exception:
        pass
      try:
        return relay.transform.InferTypeLocal(expr)
      except Exception:
        return None

    def _extract_region(self, expr, compiler_name):
      """Rewrite expr by cutting at compiler_begin for the given compiler.
      Returns (region_body, params, inputs).
      """
      self._begin_to_param = {}
      self._params = []
      self._inputs = []
      memo = {}

      def rewrite(e):
        # Preserve DAG structure: reuse previously rewritten node
        if e in memo:
          return memo[e]
        if isinstance(e, Call):
          # Strip nested compiler_end of the same compiler
          if e.op == op.get("annotation.compiler_end") and e.attrs.compiler == compiler_name:
            res = rewrite(e.args[0])
            memo[e] = res
            return res
          # Cut at compiler_begin of the same compiler and create or reuse a param
          if e.op == op.get("annotation.compiler_begin") and e.attrs.compiler == compiler_name:
            begin_node = e
            if begin_node in self._begin_to_param:
              res = self._begin_to_param[begin_node]
              memo[e] = res
              return res
            input_expr = begin_node.args[0]
            in_ty = self._infer_type(input_expr)
            name_hint = f"input_{len(self._params)}"
            param = relay.Var(name_hint, in_ty) if in_ty is not None else relay.Var(name_hint)
            self._begin_to_param[begin_node] = param
            self._params.append(param)
            self._inputs.append(input_expr)
            memo[e] = param
            return param
          new_args = [rewrite(a) for a in e.args]
          res = Call(e.op, new_args, e.attrs, e.type_args, e.span)
          memo[e] = res
          return res
        if isinstance(e, Tuple):
          res = Tuple([rewrite(f) for f in e.fields])
          memo[e] = res
          return res
        if isinstance(e, TupleGetItem):
          res = TupleGetItem(rewrite(e.tuple_value), e.index)
          memo[e] = res
          return res
        # Var/Constant/others pass through
        memo[e] = e
        return e

      body = rewrite(expr)
      params = list(self._params)
      inputs = list(self._inputs)
      # Clear state to avoid leakage between regions
      self._begin_to_param = None
      self._params = None
      self._inputs = None
      return body, params, inputs

    def visit_call(self, call):
      # We handle compiler_end as the anchoring point of a region.
      if call.op == op.get("annotation.compiler_end"):
        compiler_name = call.attrs.compiler
        # Extract region spanning from begins (as params) to this end
        region_body, params, inputs = self._extract_region(call.args[0], compiler_name)

        # Build composite function and call
        composite_func = relay.Function(params, region_body)
        composite_func = composite_func.with_attr("Composite", f"{compiler_name}.region_{self.composite_counter}")
        self.composite_counter += 1

        # Visit inputs so that upstream regions get converted too
        visited_inputs = [self.visit(arg) for arg in inputs]
        return relay.Call(composite_func, visited_inputs)

      # Strip standalone compiler_begin by visiting through
      if call.op == op.get("annotation.compiler_begin"):
        return self.visit(call.args[0])

      # Default: recursively transform children
      new_args = [self.visit(a) for a in call.args]
      return Call(call.op, new_args, call.attrs, call.type_args, call.span)

  converter = _CompositeConverter()

  for global_var, func in mod.functions.items():
    if isinstance(func, relay.Function):
      new_body = converter.visit(func.body)
      new_func = relay.Function(func.params, new_body, func.ret_type, func.type_params, func.attrs)
      mod[global_var] = new_func

  return mod


def modify_call_node_attrs(call_node, in_node=None, out_node=None, const_packed_node=None):
  """
  Modify the attributes of a Call node by setting in_node and/or out_node flags.

  Parameters
  ----------
  call_node : relay.Call
      The call node to modify
  in_node : bool, optional
      Set the in_node flag. If None, the original value is preserved.
  out_node : bool, optional
      Set the out_node flag. If None, the original value is preserved.

  Returns
  -------
  relay.Call
      A new Call node with modified attributes
  """
  if not isinstance(call_node, relay.Call):
    raise ValueError("Input must be a relay.Call node")

  # Create a dictionary to hold all attribute key-value pairs
  new_attr_dict = {}

  # Copy existing attributes if they exist
  if call_node.attrs is not None:
    for key in call_node.attrs.keys():
      attr_value = call_node.attrs[key]
      # Skip copying in_node and out_node since we'll set them explicitly
      if str(key) in ["in_node", "out_node"]:
        continue

      if isinstance(attr_value, tvm.ir.container.Array):
        # Convert array to tuple for proper handling
        new_attr_dict[str(key)] = tuple(attr_value)
      else:
        new_attr_dict[str(key)] = attr_value

    attr_type = str(call_node.attrs).split("(")[0]
  else:
    attr_type = "DictAttrs"

  # Set the new in_node and out_node values
  if in_node is not None:
    new_attr_dict["in_node"] = in_node
  if out_node is not None:
    new_attr_dict["out_node"] = out_node
  if const_packed_node is not None:
    new_attr_dict["const_packed_node"] = const_packed_node

  new_attrs = tvm.ir.make_node(attr_type, **new_attr_dict)

  return Call(call_node.op, call_node.args, new_attrs, call_node.type_args, call_node.span)

def expand_pattern(pattern, num_args):
  if len(pattern) == num_args:
    return pattern
  if len(pattern) == 1:
    return [pattern[0]] * num_args
  return None

def calculate_imcflow_func_type(func, func_name=None):
  """
  imcflow function need real tensor type with real layout.
  This function calculates the real parameter and return types of the given imcflow function.
  For example, qconv2d input parameter will be changed from float32[N,C,H,W] to uint32[N,C//256,H,W,4,8] type.

  arguments:
    func: relay.Function
          target function.
  return:
    param_types: list of relay.Type
                  updated parameter types.
    param_layouts: list of LayoutType
                    layout type for each parameter.
    ret_type: relay.Type (can be tupleType)
              updated return type.
    ret_layout: LayoutType
                layout type for return value.
  """

  class _ImcflowFunctionParamUpdater(relay.ExprMutator):
    def __init__(self, mod, func_name=None):
      super().__init__()
      self.mod = mod
      self.func_name = func_name

    def run(self, func):
      temp_mod = tvm.IRModule.from_expr(func)
      temp_mod = relay.transform.InferType()(temp_mod)
      gv = list(temp_mod.get_global_vars())[0]
      inferred_func = temp_mod[gv]

      layout_context = LayoutPropagationContext(inferred_func)
      param_layout_combinations = layout_context.build_var_layout_combinations()
      debug_print(f"Parameter layout combinations: {param_layout_combinations}")

      valid_layouts = []
      for combo in param_layout_combinations:
        output_layout = get_valid_output_layout_of_node(inferred_func, [l for l in combo.values()], self.mod)
        if output_layout is not None:
          debug_print(f"[layout] combo succeeded: {combo} -> output layout: {output_layout}")
          valid_layouts.append((combo, output_layout))
        else:
          debug_print(f"[layout] combo failed: {combo}")

      if len(valid_layouts) == 0:
        raise ValueError("No valid layout cases found for function")

      for idx, (param_layouts, output_layout) in enumerate(valid_layouts):
        debug_print(f"[layout] valid case {idx}: params {param_layouts}, output {output_layout}")

      chosen_layout = valid_layouts[-1]
      param_types, param_layouts = self._build_param_types(inferred_func, chosen_layout)
      ret_type, ret_layout = self._build_return_type(inferred_func, chosen_layout)
      return param_types, param_layouts, ret_type, ret_layout

    def _build_param_types(self, func, layout):
      param_types = []
      param_layouts = []
      input_layouts = layout[0]
      for idx, param in enumerate(func.params):
        layout = input_layouts[param]
        param_layouts.append(layout)
        param_types.append(self._apply_layout_to_type(param.checked_type, layout))
      return param_types, param_layouts

    def _build_return_type(self, func, layout):
      body = func.body
      output_layout = layout[1]
      if isinstance(body, relay.Tuple):
        ret_layouts = []
        ret_types = []
        for idx, field in enumerate(body.fields):
          field_type = getattr(field, "checked_type", None)
          if field_type is None and isinstance(body.checked_type, relay.TupleType):
            field_type = body.checked_type.fields[idx]
          layout = output_layout[idx]
          ret_layouts.append(layout)
          ret_types.append(self._apply_layout_to_type(field_type, layout))
        return relay.TupleType(ret_types), ret_layouts
      else:
        layout = output_layout
        ret_type = self._apply_layout_to_type(body.checked_type, layout)
        return ret_type, layout

    def _apply_layout_to_type(self, original_type, layout_type):
      """
      Apply layout to tensor type. it reshape tensor.
      args:
        - original_type: relay.TensorType
                         original tensor type.
        - layout_type: LayoutType
                       layout type to be applied.
      return:
        - new_type: relay.TensorType
                    new tensor type with applied layout.
      """
      if not isinstance(original_type, TensorType):
        raise ValueError("Variable type must be TensorType")

      if layout_type == LayoutType.NCHW:
        assert isinstance(original_type, TensorType) and len(original_type.shape) == 4, "NCHW layout requires 4D tensor"
        return original_type
      
      if layout_type == LayoutType.SCALAR:
        assert len(original_type.shape) == 0 or (len(original_type.shape) == 1 and original_type.shape[0] == 1) , f"SCALAR layout requires 0D tensor or shape [1]. input : {original_type}"
        return original_type

      original_shape = original_type.shape
      original_dtype = original_type.dtype

      if layout_type == LayoutType.NCHW16C:
        if len(original_shape) != 4:
          raise ValueError(f"Unsupported shape for NCHW16c layout: {original_shape}")
        N, C, H, W = original_shape
        C_ceil = (C + 15) // 16
        new_shape = [N, C_ceil, H, W, 16]
        new_dtype = original_dtype
      elif layout_type == LayoutType.NCHW64C:
        if len(original_shape) != 4:
          raise ValueError(f"Unsupported shape for NCHW64c layout: {original_shape}")
        N, C, H, W = original_shape
        C_ceil = (C + 63) // 64
        new_shape = [N, C_ceil, H, W, 64]
        new_dtype = original_dtype
      elif layout_type == LayoutType.QCONV_INPUT:
        if len(original_shape) != 4:
          raise ValueError(f"Unsupported shape for qconv_input layout: {original_shape}")
        N, C, H, W = original_shape
        C_ceil = (C + 255) // 256
        IB = 4
        new_shape = [N, C_ceil, H, W, IB, 8]
        new_dtype = "uint32"
      elif layout_type == LayoutType.QCONV_WEIGHT:
        if len(original_shape) != 4:
          raise ValueError(f"Unsupported shape for qconv_weight layout: {original_shape}")
        out_channels, in_channels, kh, kw = original_shape
        ic = 256 // (kh * kw)
        new_shape = [
          (out_channels + 63) // 64,
          (in_channels + ic - 1) // ic,
          256,
          8,
        ]
        new_dtype = "int32"
      elif layout_type == LayoutType.QDCONV_WEIGHT:
        if len(original_shape) != 4:
          raise ValueError(f"Unsupported shape for qdwconv_weight layout: {original_shape}")
        out_channels, _, kh, kw = original_shape
        # mirror packing logic used in qdwconv path (uint32)
        new_shape = [math.ceil(out_channels / 16), 8, 8]
        new_dtype = "uint32"
      else:
        raise ValueError(f"Unknown layout type: {layout_type}")

      return relay.TensorType(new_shape, new_dtype)

  updater = _ImcflowFunctionParamUpdater(func_name)
  return updater.run(func)

def create_wrap_func(func, func_name, new_param_type, new_param_layout, new_ret_type, new_ret_layout):
    """
    param_specs: [(name, shape, dtype), ...]
    """
    ttype_map = {}
    param_layouts = new_param_layout
    ret_layouts = new_ret_layout

    def _block_from_layout(layout_type):
      if layout_type == LayoutType.NCHW16C:
        return 16
      if layout_type == LayoutType.NCHW64C:
        return 64
      raise ValueError(f"Layout type {layout_type} does not have block size")
    
    def _unpack_input_value(param, old_type, new_type, layout_type):
      if layout_type == LayoutType.QCONV_INPUT:
        arg = imcflow_mmquant_out_to_4d(param, old_type.shape[1])
      elif layout_type in (LayoutType.NCHW16C, LayoutType.NCHW64C):
        block = _block_from_layout(layout_type)
        arg = relay.op.layout_transform(param, layout_type.value, "NCHW")
        N, CG, H, W, _ = new_type.shape
        c_converted = CG * block
        if c_converted > old_type.shape[1]:
          arg = relay.op.strided_slice(arg, begin=[0,0,0,0], end=[N, old_type.shape[1], H, W])
      else:
        assert layout_type == LayoutType.NCHW, "Only NCHW layout is supported for input"
        arg = params[i]
      return arg

    def _pack_output_value(expr, layout_type):
      if layout_type == LayoutType.QCONV_INPUT:
        return imcflow_4d_to_qconv_input(expr)
      if layout_type == LayoutType.NCHW16C:
        return relay.op.layout_transform(expr, "NCHW", "NCHW16c")
      if layout_type == LayoutType.NCHW64C:
        return relay.op.layout_transform(expr, "NCHW", "NCHW64c")
      
      assert layout_type == LayoutType.NCHW, "Only NCHW layout is supported for packing"
      return expr

    params = []
    old_param_names = [p.name_hint for p in func.params]
    for i, typ in enumerate(new_param_type):
        name  = f"{old_param_names[i]}_wrap"
        shape = typ.shape
        dtype = typ.dtype
        params.append(relay.var(name, shape=shape, dtype=dtype))

    old_params = func.params
    old_ret_type = func.ret_type

    #- check old param and new param shape and make args
    args = []
    for i in range(len(old_params)):
        old_type = old_params[i].checked_type
        new_type = new_param_type[i]
        if i >= len(param_layouts): raise ValueError("param layout is not enough")
        layout_type = param_layouts[i]
        arg = _unpack_input_value(params[i], old_type, new_type, layout_type)
        args.append(arg)
        ttype_map[old_params[i].name_hint] = (new_type.shape, new_type.dtype, old_type.shape, old_type.dtype)

    #- make function body
    new_attr = tvm.ir.make_node("DictAttrs", Composite=f"{func_name}_impl", Compiler="imcflow")
    func_no_attr = relay.Function(func.params, func.body, func.ret_type, attrs=new_attr)
    body = func_no_attr(*args)

    #- check old ret and new ret shape and make return value
    if isinstance(new_ret_type, relay.TupleType):
      outs = []
      for i, field_type in enumerate(new_ret_type.fields):
        if i >= len(ret_layouts): raise ValueError("return layout is not enough")
        layout_type = ret_layouts[i]
        gti = relay.TupleGetItem(body, i)
        ret_field = _pack_output_value(gti, layout_type)
        outs.append(ret_field)
      body = relay.Tuple(outs)
    elif isinstance(new_ret_type, relay.TensorType):
      target_layout = ret_layouts if isinstance(ret_layouts, LayoutType) else (ret_layouts[0] if ret_layouts else LayoutType.NCHW)
      body = _pack_output_value(body, target_layout)

    if isinstance(old_ret_type, relay.TupleType):
      temp = []
      for i, field_type in enumerate(old_ret_type.fields):
        old_type = field_type
        new_type = new_ret_type.fields[i]
        temp.append((new_type.shape, new_type.dtype, old_type.shape, old_type.dtype))
      ttype_map[func_name] = temp
    else:
      ttype_map[func_name] = (new_ret_type.shape, new_ret_type.dtype, old_ret_type.shape, old_ret_type.dtype)

    return relay.Function(params, body, new_ret_type, attrs=func.attrs), ttype_map

class ImcflowLayoutLegalizer:
  """
  A pass that identifies boundary nodes between IMCFLOW and CPU execution domains.

  This pass traverses the graph to find Function nodes with "Compiler" attribute set to "imcflow",
  then:
  1. Marks the first Call node inside the function as in_node=True
  2. Marks the last Call node inside the function as out_node=True
  3. Inserts packing nodes before calls to imcflow functions
  4. Inserts unpacking nodes after calls to imcflow functions
  """

  def __init__(self):
    self.input_call_dict = {}
    self.output_call_dict = {}
    self.imcflow_func_layout_map = {}
    self.layout_map = ImcflowDeviceConfig().LayoutMap

  def transform_mod(self, mod):
    """
    Transform the function to mark boundary nodes and insert packing/unpacking.

    This iterates through all functions in the module and processes IMCFLOW functions.
    """

    # Get all function items from the module
    items = mod.functions_items()
    function_names = [item[0].name_hint for item in items]

    # Process each IMCFLOW function to mark internal boundaries
    num_func = len(function_names)
    new_gv_map = {}

    real_tensor_type_map = {}

    for i in range(num_func):
      if isImcflowFunc(mod[function_names[i]], mod):
        print('--------------------Legalize---------------------------')
        print(mod[function_names[i]])
        print('-------------------------------------------------------')
        param_type, param_layout, ret_type, ret_layout = calculate_imcflow_func_type(mod[function_names[i]], function_names[i])
        self.imcflow_func_layout_map[function_names[i]] = (param_layout, ret_layout)
        print("Created wrapper function for", function_names[i])
        print("  Param Types and Layouts:")
        print(param_type)
        print(param_layout)
        print("  Return Type and Layout:")
        print(ret_type)
        print(ret_layout)
        mod[function_names[i]] = self._mark_and_transform_imcflow_qconv(mod[function_names[i]])
        wrap_func, ttype_map = create_wrap_func(mod[function_names[i]], function_names[i], param_type, param_layout, ret_type, ret_layout)
        real_tensor_type_map[function_names[i]] = ttype_map
        old_gv = mod.get_global_var(function_names[i])
        func_type = relay.FuncType([x.type_annotation for x in wrap_func.params], wrap_func.ret_type)
        new_gv = relay.GlobalVar(function_names[i], type_annot=func_type)
        del mod[old_gv]
        mod[new_gv] = wrap_func
        new_gv_map[old_gv] = new_gv

        print("Wrap function created for", function_names[i])
        print(wrap_func)

    # printModel(".", mod, {}, "after_imcflow_layout_legalizer")

    mod = self.replace_imcflow_gv(mod, new_gv_map)
    mod = self._insert_packing_unpacking(mod, real_tensor_type_map, self.imcflow_func_layout_map)

    #- make layout map of main and imcflow_impl functions. exclude imcflow_wrap function's local node.
    #- they are just glue nodes.
    # self._build_layout_map(mod)

    # return transformed_func
    return mod, real_tensor_type_map

  def replace_imcflow_gv(self, mod, new_gv_map):
    class _GVReplacer(tvm.relay.ExprMutator):
      def __init__(self, new_gv_map):
        super().__init__()
        self.new_gv_map = new_gv_map

      def visit_global_var(self, gvar):
        if gvar in self.new_gv_map:
          return self.new_gv_map[gvar]
        return gvar

    mod['main'] = _GVReplacer(new_gv_map).visit(mod['main'])

    return mod
  
  def _build_layout_map(self, mod):
    """
    Build node -> output_layout map for main and imcflow_impl functions.
    Skip layouts for imcflow_wrap glue nodes while still traversing them to reach impl bodies.
    Result is stored in ImcflowDeviceConfig().LayoutMap with NodeID keys.
    """
    imcflow_layout_map = {}

    class _LayoutCollector(relay.ExprVisitor):
      def __init__(self, module, layout_dst, imcflow_func_layout_map):
        super().__init__()
        self.module = module
        self.layout_dst = layout_dst
        self.imcflow_func_layout_map = imcflow_func_layout_map
        self.node_layouts = {}
        self.collect_stack = []
        self.main_func = module["main"]

      def _infer_layout_from_type(self, ttype):
        if isinstance(ttype, TupleType):
          return tuple(self._infer_layout_from_type(f) for f in ttype.fields)
        if not isinstance(ttype, TensorType):
          return None
        rank = len(ttype.shape)
        if rank == 0 or (rank == 1 and ttype.shape[0] == 1):
          return LayoutType.SCALAR
        if rank == 1:
          return LayoutType.C
        if rank == 3:
          return LayoutType.NCHW
        if rank == 4:
          return LayoutType.NCHW
        if rank == 5:
          block = int(ttype.shape[4])
          if block == 16:
            return LayoutType.NCHW16C
          if block == 64:
            return LayoutType.NCHW64C
        if rank == 6:
          return LayoutType.QCONV_INPUT
        return None

      def _record(self, expr, layout):
        self.node_layouts[expr] = layout
        if self.collect_stack and self.collect_stack[-1]:
          node_id = getNodeID(expr)
          if node_id is not None:
            self.layout_dst[node_id] = layout

      def _should_collect_fn(self, fn):
        if fn == self.main_func:
          return True
        if fn.attrs and "Composite" in fn.attrs:
          comp = str(fn.attrs["Composite"])
          if comp.endswith("_impl"):
            return True
        return False

      def visit_var(self, var):
        layout = self._infer_layout_from_type(var.type_annotation) or LayoutType.NCHW
        self._record(var, layout)

      def visit_constant(self, const):
        layout = self._infer_layout_from_type(const.checked_type) or LayoutType.SCALAR
        self._record(const, layout)

      def visit_tuple(self, tup):
        for field in tup.fields:
          self.visit(field)
        layout = tuple(self.node_layouts.get(f, None) for f in tup.fields)
        self._record(tup, layout)

      def visit_tuple_getitem(self, tgi):
        self.visit(tgi.tuple_value)
        tuple_layout = self.node_layouts.get(tgi.tuple_value, None)
        layout = tuple_layout[tgi.index] if isinstance(tuple_layout, (tuple, list)) else tuple_layout
        self._record(tgi, layout)

      def visit_call(self, call):
        for arg in call.args:
          self.visit(arg)
        arg_layouts = []
        for arg in call.args:
          layout = self.node_layouts.get(arg, None)
          if layout is None:
            layout = self._infer_layout_from_type(_get_type(self.module, arg))
          arg_layouts.append(layout)

        if isinstance(call.op, relay.GlobalVar) and call.op.name_hint in self.imcflow_func_layout_map:
          out_layout = self.imcflow_func_layout_map[call.op.name_hint][1]
        else:
          out_layout = get_valid_output_layout_of_node(call, arg_layouts, self.module, True)
        if out_layout is None:
          out_layout = self._infer_layout_from_type(_get_type(self.module, call))
        call_type = _get_type(self.module, call)
        if isinstance(call_type, TupleType) and not isinstance(out_layout, (tuple, list)):
          out_layout = tuple(out_layout for _ in call_type.fields)
        self._record(call, out_layout)

      def visit_function(self, fn):
        collect = self._should_collect_fn(fn)
        self.collect_stack.append(collect)
        self.visit(fn.body)
        body_layout = self.node_layouts.get(fn.body, self._infer_layout_from_type(fn.ret_type))
        self._record(fn, body_layout)
        self.collect_stack.pop()

    collector = _LayoutCollector(mod, imcflow_layout_map, self.imcflow_func_layout_map)

    # Visit main first to populate map for runtime visible graph
    collector.visit(mod["main"])
    # Visit remaining functions to collect layouts for inline *_impl bodies
    for gv, func in mod.functions_items():
      if gv.name_hint != "main":
        collector.visit(func)

    ImcflowDeviceConfig().LayoutMap.clear()
    ImcflowDeviceConfig().LayoutMap.update(imcflow_layout_map)

  def _mark_and_transform_imcflow_qconv(self, func):
    """
    Mark the imcflow_qconv call nodes in an IMCFLOW function.
    """
    def qconv_weight_transform(call):
      """
      Transform the weight argument of imcflow_qconv
      Original weight: int8 4D tensor (out_channels, in_channels, kh, kw)
      Transformed weight: int32 4D tensor (ceil(out_channels/64), ceil(in_channels/ic), 256, 8), where
      - int32 contains 8 int4 values
      - ic = floor(256/kh/kw) => 256 = ic * kh * kw
      Why (ceil(out_channels/64), ceil(in_channels/ic), 256, 8) int32?
      - 8 int32 values means 64 output channels (that's why ceil(out_channels/64))
      - 256 = ic * kh * kw means each block contains ic input channels. 256 is internally ordered by (ic, kh, kw).
      - ceil(in_channels/ic) means the number of input channel blocks, each block contains ic input channels
      """
      # Transform the weight argument of imcflow_qconv to int8
      if call.op == op.get("nn.imcflow_qconv"):
        OriginWeight = call.args[1].data.asnumpy()

        # Original weight shape: (out_channels, in_channels, kh, kw)
        out_channels, in_channels, kh, kw = OriginWeight.shape

        # Calculate ic: number of input channels per block (floor division)
        ic = 256 // (kh * kw)

        # Calculate actual spatial elements per input channel block
        spatial_elements = ic * kh * kw  # This might be < 256

        # Calculate padded dimensions
        out_blocks = (out_channels + 63) // 64  # ceil(out_channels / 64)
        in_blocks = (in_channels + ic - 1) // ic  # ceil(in_channels / ic)

        # Pad output channels to multiple of 64
        padded_out_channels = out_blocks * 64
        # Pad input channels to multiple of ic
        padded_in_channels = in_blocks * ic

        # Create padded weight array
        PaddedWeight = np.zeros((padded_out_channels, padded_in_channels, kh, kw), dtype=np.int8)
        PaddedWeight[:out_channels, :in_channels, :, :] = OriginWeight

        # Reshape to group by blocks
        # First reshape to (out_blocks, 64, in_blocks, ic, kh, kw)
        Reshaped = PaddedWeight.reshape(out_blocks, 64, in_blocks, ic, kh, kw)

        # Transpose to (out_blocks, in_blocks, ic, kh, kw, 64)
        # This groups the spatial elements (ic*kh*kw) together with 64 output channels
        Transposed = Reshaped.transpose(0, 2, 3, 4, 5, 1)

        # Flatten spatial dimensions: (out_blocks, in_blocks, spatial_elements, 64)
        Flattened = Transposed.reshape(out_blocks, in_blocks, spatial_elements, 64)

        # Pad spatial dimension to 256 if needed
        if spatial_elements < 256:
          padding = 256 - spatial_elements
          Padded = np.pad(Flattened, ((0, 0), (0, 0), (0, padding), (0, 0)), mode='constant', constant_values=0)
        else:
          Padded = Flattened

        # Now Padded has shape (out_blocks, in_blocks, 256, 64)

        # Now we need to pack 8 int4 values into each int32
        # Each group of 8 output channels (64/8 = 8 groups) becomes one int32
        # Reshape to (out_blocks, in_blocks, 256, 8, 8) where last dim is 8 int4 values to pack
        ToPack = Padded.reshape(out_blocks, in_blocks, 256, 8, 8)

        # Pack 8 int4 values into int32
        # Each int4 occupies 4 bits in the int32
        Packed = np.zeros((out_blocks, in_blocks, 256, 8), dtype=np.uint32)
        for i in range(8):
          # Shift each int4 value to its position (4 bits per value)
          # Mask to 4 bits (0xF) to ensure int4 range
          Packed += ((ToPack[:, :, :, :, i].astype(np.uint32) & 0xF) << (i * 4))

        NewWeight = relay.Constant(tvm.nd.array(Packed))
        new_args = [call.args[0], NewWeight, call.args[2]]  # Include config as third argument
        new_type_args = [call.type_args[0], relay.TensorType(NewWeight.data.shape, "uint32"), call.type_args[2]]

        return Call(call.op, new_args, call.attrs, new_type_args, call.span)
      elif call.op == op.get("nn.imcflow_qdwconv"):
        OriginWeight = call.args[1].data.asnumpy()
        out_channels, in_channels, kh, kw = OriginWeight.shape
        new_weight = np.zeros((math.ceil(out_channels/16), 8, 8), dtype=np.uint32)

        for c in range(out_channels):
          for kh_ in range(kh):
            for kw_ in range(kw):
              for wb in range(8):
                oc_block = c // 16
                oc_offset = c % 16
                khkw_index = kh_ * kw + kw_
                origin_val = OriginWeight[c, 0, kh_, kw_]
                bit_val = (origin_val >> wb) & 0x1
                word_idx = (oc_offset*16 + khkw_index) // 32
                word_offset = (oc_offset*16 + khkw_index) % 32
                new_weight[oc_block, wb, word_idx] |= (bit_val << word_offset)
        NewWeight = relay.Constant(tvm.nd.array(new_weight))
        new_args = [call.args[0], NewWeight, call.args[2]]
        new_type_args = [call.type_args[0], relay.TensorType(NewWeight.data.shape, "uint32"), call.type_args[2]]
        return Call(call.op, new_args, call.attrs, new_type_args, call.span)
      else:
        return call

    class _BoundaryMarker(relay.ExprMutator):
      def visit_call(self, call):
        new_call = super().visit_call(call)

        # Mark imcflow_qconv calls as both input and output nodes
        if isinstance(call.op, tvm.ir.Op) and (call.op == op.get("nn.imcflow_qconv") or call.op == op.get("nn.imcflow_qdwconv")):
          # new_call = modify_call_node_attrs(new_call, const_packed_node=True, in_node=True, out_node=True)
          new_call = modify_call_node_attrs(new_call, const_packed_node=True)
          new_call = qconv_weight_transform(new_call)
          return new_call

        return new_call

    marker = _BoundaryMarker()
    new_body = marker.visit(func.body)
    return relay.Function(func.params, new_body, func.ret_type, func.type_params, func.attrs)

  def _mark_imcflow_function_boundaries(self, func):
    """
    Mark the first and last Call nodes in an IMCFLOW function.
    The first call is the one that directly uses function parameters as input.
    The last call is the one that directly produces the function's output.
    """

    # Collect all Call nodes in the function
    call_nodes = []

    class _CallCollector(relay.ExprVisitor):
      def visit_call(self, call):
        call_nodes.append(call)
        super().visit_call(call)

    collector = _CallCollector()
    collector.visit(func.body)

    if not call_nodes:
      return func

    # Find the first call node that uses function parameters
    input_calls = self._find_input_call(func, call_nodes)
    self.input_call_dict[func] = input_calls

    # Find the output call node - the one that directly produces the function's return
    output_calls = self._find_output_call(func.body)
    self.output_call_dict[func] = output_calls

    class _BoundaryMarker(relay.ExprMutator):
      def visit_call(self, call):
        new_call = super().visit_call(call)

        # Handle both single call and list of calls
        if (isinstance(output_calls, list) and call in output_calls) or call == output_calls:
          return modify_call_node_attrs(new_call, in_node=None, out_node=True)
        if (isinstance(input_calls, list) and call in input_calls) or call == input_calls:
          return modify_call_node_attrs(new_call, in_node=True, out_node=None)
        return new_call

        # return modify_call_node_attrs(new_call, in_node=True, out_node=True)

    marker = _BoundaryMarker()
    new_body = marker.visit(func.body)
    return relay.Function(func.params, new_body, func.ret_type, func.type_params, func.attrs)

  def _find_input_call(self, func, call_nodes):
    """
    Find the first Call node that directly uses function parameters as input.
    """
    # Create set of function parameter variables for quick lookup
    param_vars = set(func.params)

    input_calls = []

    # Check each call node to see if it directly uses function parameters
    for call in call_nodes:
      # Check if any of the call's arguments are function parameters
      for arg in call.args:
        if isinstance(arg, relay.Var) and arg in param_vars:
          input_calls.append(call)

    return input_calls

  def _find_output_call(self, body):
    """
    Find the Call node that directly produces the function's output.
    This traverses the body expression to find the root Call node.
    """
    # Handle different body types
    if isinstance(body, relay.Call):
      # If body is a Call to a composite function, we need to look inside
      if hasattr(body.op, "attrs"):
        if hasattr(body.op.attrs, "Composite"):
          return self._find_output_call(body.op.body)
      # If body is directly a Call, that's our output call
      return body
    elif isinstance(body, relay.TupleGetItem):
      # If body is TupleGetItem, find the call that produces the tuple
      return self._find_output_call(body.tuple_value)
    elif isinstance(body, relay.Tuple):
      # If body is a Tuple, we need to find the calls that produce each field
      # For now, we'll just return the first Call we find in the fields
      output_calls = []
      for field in body.fields:
        output_call = self._find_output_call(field)
        output_calls.append(output_call)
      return output_calls
    else:
      raise ValueError("Unsupported body type for finding output call")

  def update_imcflow_func_params(self, func):
    class _ImcflowFunctionParamUpdater(relay.ExprMutator):
      def __init__(self):
        super().__init__()
        self.var_consumers = {}  # {var : [(consumer_call, arg_index), ...]}

      def gather_var_consumers(self, func):
        """
        Traverse the function to gather consumer node descriptions of function parameters.
        Returns {var : [(consumer_call, arg_index), ...]}
        """
        self.var_consumers = {param: [] for param in func.params}

        # Build use-def chain for the function body
        use_def_parser = UseDefChainParser()
        use_def_parser.visit(func.body)

        # Skip element-wise ops and recurse through them to reach meaningful consumers
        finder = ConsumerFinder(use_def_parser, skip_predicates=[skip_element_wise_predicate])

        for param in func.params:
          consumers = finder.find_consumers_of_node(param)
          self.var_consumers[param] = consumers

        return self.var_consumers

      def get_required_layout(self, consumer_node_desc):
        """
        given a consumer node description, return the required layout for the function parameter.
        Returns tuple: (layout_type, is_packed)
        - layout_type: "qconv_input", "qconv_output", "vector", "scalar"
        - is_packed: True if already packed, False if needs packing
        """
        consumer_node, arg_index = consumer_node_desc
        assert isinstance(consumer_node, relay.Call), "Consumer node must be a Call node"

        if consumer_node.op == op.get("split"):
          return ("qconv_input", True)
        elif consumer_node.op == op.get("concatenate"):
          return ("vector", True)
        elif consumer_node.op == op.get("nn.imcflow_qconv") or consumer_node.op == op.get("nn.imcflow_qdwconv"):
          # arg_index: 0=input, 1=weight
          if arg_index == 0:
            return ("qconv_input", True)
          elif arg_index == 1:
            return ("qconv_output", True)
        elif consumer_node.op == op.get("nn.bias_add"):
          return ("vector", True)
        elif consumer_node.op == op.get("nn.relu"):
          return ("vector", True)
        elif consumer_node.op == op.get("imcflow.fused_batch_norm"):
          return ("vector", True)
        elif consumer_node.op == op.get("qnn.imcflow_min_max_quantize"):
          if arg_index == 0:
            return ("vector", True)
          else:
            return ("scalar", False)
        elif consumer_node.op == op.get("add") or consumer_node.op == op.get("divide") or consumer_node.op == op.get("multiply"):
          return ("vector", True)
        elif consumer_node.op == op.get("qnn.imcflow_nu_quantize"):
          raise ValueError("nu_quantize should not be consumer nodes of function parameters")
        elif consumer_node.op == op.get("nn.conv2d"):
          raise ValueError("conv2d should not be consumer nodes of function parameters")
        else:
          raise ValueError(f"Unsupported operator detected: {consumer_node.op}. please check.")

      def update_param(self, var):
        """
        Gather required layouts from all consumer nodes of the variable.
        If more than one consumer node exists, check compatibility of them.
        If compatible, calculate new shape corresponding to the layout.

        vector : NCHW16c
        qconv_input : [N, ceil(C/256), H, W, IB, 8] int32

        Returns updated variable with new type, or original if no update needed.
        """
        if var not in self.var_consumers or len(self.var_consumers[var]) == 0:
          # No consumers, keep original
          return var

        consumers = self.var_consumers[var]

        # Gather all required layouts
        required_layouts = []
        for consumer_desc in consumers:
          layout_info = self.get_required_layout(consumer_desc)
          required_layouts.append(layout_info)

        # Check compatibility - all consumers should require the same layout
        if len(required_layouts) == 0:
          return var

        first_layout = required_layouts[0]
        for layout_info in required_layouts[1:]:
          if layout_info[0] != first_layout[0]:
            raise ValueError(f"Incompatible layouts required for parameter {var.name_hint}: "
                           f"{first_layout[0]} vs {layout_info[0]}")

        layout_type, is_packed = first_layout

        # If not packed, keep original (scalar values like min/max)
        if not is_packed:
          return var

        # Calculate new shape based on layout type
        original_type = var.checked_type
        if not isinstance(original_type, TensorType):
          return var

        original_shape = original_type.shape
        original_dtype = original_type.dtype

        # Calculate new shape based on layout type
        if layout_type == "vector":
          # NCHW -> NCHW16c
          if len(original_shape) == 4:
            N, C, H, W = original_shape
            C_ceil = (C + 15) // 16
            new_shape = [N, C_ceil, H, W, 16]
            new_dtype = original_dtype
          else:
            raise ValueError(f"Unsupported shape for vector layout: {original_shape}")
        elif layout_type == "qconv_input":
          # NCHW -> [N, ceil(C/256), H, W, IB, 8] int32
          if len(original_shape) == 4:
            N, C, H, W = original_shape
            C_ceil = (C + 255) // 256
            IB = 4  # Fixed value for qconv_input
            new_shape = [N, C_ceil, H, W, IB, 8]
            new_dtype = "uint32"
          else:
            raise ValueError(f"Unsupported shape for qconv_input layout: {original_shape}")
        elif layout_type == "qconv_output":
          # NCHW -> [N, ceil(C/256), H, W, IB, 8] int32 (same as qconv_input for weights)
          if len(original_shape) == 4:
            N, C, H, W = original_shape
            C_ceil = (C + 255) // 256
            IB = 4  # Fixed value for qconv_output
            new_shape = [N, C_ceil, H, W, IB, 8]
            new_dtype = "int32"
          else:
            raise ValueError(f"Unsupported shape for qconv_output layout: {original_shape}")
        else:
          raise ValueError(f"Unknown layout type: {layout_type}")

        # Create new variable with updated type
        new_type = relay.TensorType(new_shape, new_dtype)
        new_var = relay.Var(var.name_hint, new_type)

        return new_var

      def visit_function(self, fn):
        """
        update the parameters of imcflow functions to match the packed layout
        Scan function argument nodes and check layout.
        if function param layout is different from argument node layout, update the function param layout

        This also recursively updates local functions within the function body.
        """
        # First gather consumers for all parameters
        self.gather_var_consumers(fn)

        # Update each parameter based on its consumers
        new_params = []
        param_map = {}  # Map from old var to new var

        for param in fn.params:
          new_param = self.update_param(param)
          new_params.append(new_param)
          if new_param != param:
            param_map[param] = new_param
            print(f"  Updated parameter {param.name_hint}: {param.checked_type} -> {new_param.type_annotation}")

        # Recursively visit the function body to update local functions
        # This will also apply variable substitution if params were updated
        if len(param_map) == 0:
          # No parameter updates, but still need to visit body for local functions
          new_body = self.visit(fn.body)
        else:
          # Apply parameter substitution first, then visit for local functions
          substituted_body = relay.bind(fn.body, param_map)
          new_body = self.visit(substituted_body)

        # Check if anything changed (params or body)
        if len(param_map) == 0 and new_body == fn.body:
          return fn

        # Create temporary function with updated parameters and body (but old return type)
        temp_func = relay.Function(
          new_params,
          new_body,
          None,
          fn.type_params,
          fn.attrs
        )

        # Wrap the function in an IRModule and run InferType to get the actual return type
        temp_mod = tvm.IRModule.from_expr(temp_func)
        print(temp_mod)
        temp_mod = relay.transform.InferType()(temp_mod)

        # Get the inferred function with updated types
        gv = list(temp_mod.get_global_vars())[0]
        inferred_func = temp_mod[gv]

        # Get the inferred return type from the function body
        new_ret_type = inferred_func.body.checked_type

        print(f"  Updated return type: {fn.ret_type} -> {new_ret_type}")

        # Create final function with updated parameters AND updated return type
        new_func = relay.Function(
          new_params,
          new_body,
          new_ret_type,
          fn.type_params,
          fn.attrs
        )

        return new_func

      def run(self, func):
        return self.visit(func)

    updater = _ImcflowFunctionParamUpdater()
    return updater.run(func)

  def _insert_packing_unpacking(self, mod, real_tensor_type_map, imcflow_func_layout_map):
    """
    Insert layout related nodes like layout transform, imcflow_mmquant_out_to_4d, imcflow_4d_to_qconv_input, stride or padding.
    We traverse the main function to find calls to imcflow functions, and insert layout related nodes around them.
    We assume if cpu has computation nodees between imcflow function calls, cpu can handle only NCHW.
    If the args of cpu computation nodes are not NCHW, we need to insert layout transform nodes before and after cpu computation nodes as well.

    We always transform layout of last node of function to NCHW if it is not NCHW.
    """
    class _LayoutTransformer(relay.ExprMutator):
      def __init__(self, module, ttype_map, imcflow_func_layout_map):
        super().__init__()
        self.module = module
        self.ttype_map = ttype_map
        self.imcflow_func_layout_map = imcflow_func_layout_map
        self.layout_map = {}

      def _infer_layout_from_type(self, ttype):
        if isinstance(ttype, TupleType):
          return tuple(self._infer_layout_from_type(f) for f in ttype.fields)
        if not isinstance(ttype, TensorType):
          return None
        rank = len(ttype.shape)
        if rank == 0 or (rank == 1 and ttype.shape[0] == 1):
          return LayoutType.SCALAR
        if rank == 1:
          return LayoutType.C
        if rank == 2:
          return LayoutType.MK
        if rank == 3:
          return LayoutType.NCHW
        if rank == 4:
          return LayoutType.NCHW
        if rank == 5:
          block = int(ttype.shape[4])
          if block == 16:
            return LayoutType.NCHW16C
          if block == 64:
            return LayoutType.NCHW64C
        if rank == 6:
          return LayoutType.QCONV_INPUT
        return None

      def _layout_equal(self, a, b):
        if a is None or b is None:
          return False
        if isinstance(a, (tuple, list)) and not isinstance(b, (tuple, list)):
          return all(self._layout_equal(x, b) for x in a)
        if isinstance(b, (tuple, list)) and not isinstance(a, (tuple, list)):
          return all(self._layout_equal(a, x) for x in b)
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
          return len(a) == len(b) and all(self._layout_equal(x, y) for x, y in zip(a, b))
        return a == b

      def _layout_to_str(self, layout):
        mapping = {
          LayoutType.NCHW: "NCHW",
          LayoutType.NCHW16C: "NCHW16c",
          LayoutType.NCHW64C: "NCHW64c",
        }
        return mapping.get(layout, None)

      def _channels_from_map(self, func_name, index=None):
        if func_name not in self.ttype_map:
          return None
        info = self.ttype_map[func_name].get(func_name, None)
        if info is None:
          return None
        if isinstance(info, list):
          if index is None or index >= len(info):
            index = 0
          _, _, old_shape, _ = info[index]
        else:
          _, _, old_shape, _ = info
        if len(old_shape) > 1:
          return int(old_shape[1])
        return None

      def _channels_from_expr(self, expr, index=None):
        if isinstance(expr, relay.TupleGetItem):
          return self._channels_from_expr(expr.tuple_value, expr.index)
        if isinstance(expr, relay.Call) and isinstance(expr.op, relay.GlobalVar):
          return self._channels_from_map(expr.op.name_hint, index)
        ttype = _get_type(self.module, expr)
        if isinstance(ttype, TensorType):
          if len(ttype.shape) == 4:
            return int(ttype.shape[1])
          if len(ttype.shape) == 5:
            block = int(ttype.shape[4])
            return int(ttype.shape[1]) * block
          if len(ttype.shape) == 6:
            return int(ttype.shape[1]) * 256
        return None

      def _convert_layout(self, expr, curr_layout, target_layout, channel_hint=None):
        if target_layout is None or curr_layout is None or self._layout_equal(curr_layout, target_layout):
          return expr, curr_layout if curr_layout is not None else target_layout
        if isinstance(curr_layout, (tuple, list)) or isinstance(target_layout, (tuple, list)):
          if self._layout_equal(curr_layout, target_layout):
            return expr, target_layout
          raise ValueError("Tuple layout mismatch; cannot convert composite layout automatically.")

        if curr_layout == LayoutType.NCHW and target_layout in (LayoutType.NCHW16C, LayoutType.NCHW64C):
          layout_str = self._layout_to_str(target_layout)
          expr = relay.op.layout_transform(expr, "NCHW", layout_str)
          return expr, target_layout
        if curr_layout in (LayoutType.NCHW16C, LayoutType.NCHW64C) and target_layout == LayoutType.NCHW:
          layout_str = self._layout_to_str(curr_layout)
          expr = relay.op.layout_transform(expr, layout_str, "NCHW")
          if channel_hint is not None:
            ttype = _get_type(self.module, expr)
            if isinstance(ttype, TensorType) and len(ttype.shape) == 4:
              N, C, H, W = [int(x) for x in ttype.shape]
              if C > channel_hint:
                expr = relay.op.strided_slice(expr, begin=[0, 0, 0, 0], end=[N, channel_hint, H, W])
          return expr, target_layout
        if curr_layout == LayoutType.NCHW and target_layout == LayoutType.QCONV_INPUT:
          expr = imcflow_4d_to_qconv_input(expr)
          return expr, target_layout
        if curr_layout == LayoutType.QCONV_INPUT and target_layout == LayoutType.NCHW:
          channels = channel_hint if channel_hint is not None else self._channels_from_expr(expr)
          channels = channels if channels is not None else 0
          expr = imcflow_mmquant_out_to_4d(expr, channels)
          return expr, target_layout
        return expr, curr_layout

      def visit_var(self, var):
        layout = self._infer_layout_from_type(var.type_annotation)
        if not layout: raise ValueError(f"Cannot infer layout from var type: {var.type_annotation}")
        self.layout_map[var] = layout
        return var

      def visit_constant(self, const):
        layout = self._infer_layout_from_type(const.checked_type)
        if not layout: raise ValueError(f"Cannot infer layout from constant type: {const.checked_type}")
        self.layout_map[const] = layout
        return const

      def visit_tuple(self, tup):
        new_fields = [self.visit(f) for f in tup.fields]
        layout = tuple(self.layout_map[f] for f in new_fields)
        new_tup = relay.Tuple(new_fields)
        self.layout_map[new_tup] = layout
        return new_tup

      def visit_tuple_getitem(self, tgi):
        new_val = self.visit(tgi.tuple_value)
        val_layout = self.layout_map[new_val]
        layout = val_layout[tgi.index] if isinstance(val_layout, (tuple, list)) else val_layout
        new_tgi = relay.TupleGetItem(new_val, tgi.index)
        self.layout_map[new_tgi] = layout
        return new_tgi

      def visit_call(self, call):
        new_args = [self.visit(arg) for arg in call.args]
        arg_layouts = [self.layout_map[arg] for arg in new_args]
        transformed_args = []
        target_arg_layouts = list(arg_layouts)

        if isinstance(call.op, relay.GlobalVar) and isImcflowFunc(self.module[call.op.name_hint], self.module):
          target_func = self.module[call.op.name_hint]
          expected_layouts = self.imcflow_func_layout_map[call.op.name_hint][0]
          for idx, (arg, curr_layout, tgt_layout) in enumerate(zip(new_args, arg_layouts, expected_layouts)):
            new_arg, new_layout = self._convert_layout(arg, curr_layout, tgt_layout)
            transformed_args.append(new_arg)
            target_arg_layouts[idx] = new_layout
        elif isinstance(call.op, tvm.ir.Op):
          for i, (arg, curr_layout) in enumerate(zip(new_args, arg_layouts)):
            req_layouts = get_required_layout_from_op(call, "inputs", i, True)
            tgt_layout = None
            for r in req_layouts:
              if self._layout_equal(curr_layout, r):
                tgt_layout = r
                break
            if tgt_layout is None and req_layouts:
              tgt_layout = req_layouts[0]
            new_arg, new_layout = self._convert_layout(arg, curr_layout, tgt_layout)
            transformed_args.append(new_arg)
            target_arg_layouts[i] = new_layout
        else:
          for i, (arg, curr_layout) in enumerate(zip(new_args, arg_layouts)):
            tgt_layout = LayoutType.NCHW if isinstance(curr_layout, LayoutType) else curr_layout
            new_arg, new_layout = self._convert_layout(arg, curr_layout, tgt_layout)
            transformed_args.append(new_arg)
            target_arg_layouts[i] = new_layout

        new_call = relay.Call(call.op, transformed_args, call.attrs)
        if isinstance(call.op, relay.GlobalVar) and isImcflowFunc(self.module[call.op.name_hint], self.module):
          out_layout = self.imcflow_func_layout_map[call.op.name_hint][1]
        else:
          out_layout = get_valid_output_layout_of_node(new_call, target_arg_layouts, self.module, True)
        call_type = _get_type(self.module, new_call)
        if isinstance(call_type, TupleType) and not isinstance(out_layout, (tuple, list)):
          out_layout = tuple(out_layout for _ in call_type.fields)
        debug_print(f"[insert_packing_unpacking] Call: {new_call.op} | In layouts: {target_arg_layouts} | Out layout: {out_layout}")
        self.layout_map[new_call] = out_layout
        return new_call

      def visit_function(self, fn):
        new_body = self.visit(fn.body)
        ret_layout = self.layout_map[new_body]
        if not self._layout_equal(ret_layout, LayoutType.NCHW):
          if isinstance(ret_layout, (tuple, list)):
            # best-effort: transform tuple fields individually
            new_fields = []
            for idx, field in enumerate(new_body.fields if isinstance(new_body, relay.Tuple) else [new_body]):
              layout = ret_layout[idx] if isinstance(ret_layout, (tuple, list)) else ret_layout
              converted, _ = self._convert_layout(field, layout, LayoutType.NCHW, self._channels_from_expr(field, idx))
              new_fields.append(converted)
            new_body = relay.Tuple(new_fields) if isinstance(new_body, relay.Tuple) else new_fields[0]
          else:
            new_body, _ = self._convert_layout(new_body, ret_layout, LayoutType.NCHW, self._channels_from_expr(new_body))
        self.layout_map[fn] = ret_layout
        return relay.Function(fn.params, new_body, None, fn.type_params, fn.attrs)

    inserter = _LayoutTransformer(mod, real_tensor_type_map, imcflow_func_layout_map)
    new_main = inserter.visit(mod["main"])
    ImcflowDeviceConfig().LayoutMap.update(inserter.layout_map)
    mod.update_func(mod.get_global_var("main"), new_main)
    mod["main"] = relay.Function(new_main.params, new_main.body, None, new_main.type_params, new_main.attrs)
    mod = relay.transform.InferType()(mod)
    return mod

class ImcflowFuncInOutOrderSetup:
  """
  setup order of input and output nodes of imcflow functions.
  IMCE has specific order of inputs. For example, 
  - input data : input arguments of imcflow function
    - some of them should be interleaved send.
  
  - output data : return values of imcflow function
    - some of them should be interleaved receive.

  - constant with CMD 
    - conv weight
      - weight : pushed by cmd. order is not important

  - constant
    - conv configuration : it should has priorify over conv input
    - minmax params
    - batch norm params
    - if a IMCE have multiple nodes which has constant node, receive order of the IMCE is from first op to last op in topological order. 

  order format:
    - 2D list format. outer list is order groups. inner list is interleaving group.
      e.g., [[input1, input2], [input3]] means input1 and input2 are interleaved send first, then input3 is sent.
    
  """

def constructDataBlockDict(mod):
  imcflow_func_map = ImcflowDeviceConfig().ImcflowFuncMap
  for func_name_var, func in mod.functions.items():
    if func_name_var.name_hint == "main": continue
    elif isinstance(func, relay.Function) and hasattr(func.attrs, "Compiler") and func.attrs["Compiler"]=="imcflow" :
      target_func = imcflow_func_map[func_name_var.name_hint]
      input_node_ids = [getNodeID(n) for n in getInputNodesOfFunc(target_func.func_node)]
      output_node_id = getNodeID(target_func.func_node)
      const_node_ids = [getNodeID(n) for n in getConstNodesOfFunc(target_func.func_node)]
      ImcflowDeviceConfig().get_data_block_dict(target_func, func_name_var.name_hint, input_node_ids, output_node_id, const_node_ids)
