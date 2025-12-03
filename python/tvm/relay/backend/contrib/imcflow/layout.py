from tvm.relay.ty import TupleType, TensorType
import tvm
from tvm import relay
from tvm.relay import transform, op
from tvm.relay.expr import (Call, GlobalVar, TupleGetItem, const, Let, Var, If, Tuple, Constant)
from tvm.relay.backend.contrib.imcflow.transform_utils import (
  debug_print, getNodeID, getNodeDebugID, UseDefChainParser, NodeCollector, get_type, get_shape, 
  isImcflowFunc
)
from tvm.contrib.imcflow import (
  ImcflowDeviceConfig
)
from tvm.relay.op.transform import imcflow_packing, imcflow_unpacking, imcflow_4d_to_qconv_input, imcflow_mmquant_out_to_4d

from enum import Enum
import itertools
from copy import deepcopy
import math
import numpy as np

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

  def all():
    return [
      LayoutType.NCHW,
      LayoutType.NCHW16C,
      LayoutType.NCHW64C,
      LayoutType.QCONV_INPUT,
      LayoutType.QCONV_WEIGHT,
      LayoutType.QDCONV_WEIGHT,
      LayoutType.SCALAR,
      LayoutType.C,
      LayoutType.MK,
    ]

# Layout requirements per op.
# Each op maps to a list of rules. A rule is a tuple:
#   (inputs_options, output_layout)
# - inputs_options: list of possible input layout lists (one list per permutatio.
#   Example: [[a, b], [b, a]] means two valid input orderings.
#   If a layout list has length 1 and the call has more args, the single layout applies to all args.
# - output_layout: single layout applied to all outputs of the op.
IMCFLOW_REQUIRED_OP_LAYOUTS = {
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
    (
      [
        [LayoutType.MK, LayoutType.C],
      ],
      LayoutType.MK,
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
    ),
    (
      [
        [LayoutType.MK],
      ],
      LayoutType.MK,
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
}

CPU_REQUIRED_OP_LAYOUTS = {
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
    (
      [
        [LayoutType.MK, LayoutType.C],
      ],
      LayoutType.MK,
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
    ),
    (
      [
        [LayoutType.MK],
      ],
      LayoutType.MK,
    )
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
    ),
    (
      [
        [LayoutType.MK, LayoutType.MK],
        [LayoutType.MK, LayoutType.SCALAR],
        [LayoutType.SCALAR, LayoutType.MK],
      ],
      LayoutType.MK,
    ),
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
    ),
    (
      [
        [LayoutType.MK, LayoutType.MK],
        [LayoutType.MK, LayoutType.SCALAR],
        [LayoutType.SCALAR, LayoutType.MK],
      ],
      LayoutType.MK,
    ),
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
    ),
    (
      [
        [LayoutType.MK, LayoutType.MK],
        [LayoutType.MK, LayoutType.SCALAR],
        [LayoutType.SCALAR, LayoutType.MK],
      ],
      LayoutType.MK,
    ),
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
    ),
    (
      [
        [LayoutType.MK, LayoutType.C, LayoutType.C, LayoutType.C, LayoutType.C]
      ],
      LayoutType.MK,
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
    ),
    (
      [
        [LayoutType.MK],
      ],
      LayoutType.MK,
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
  ],
  "nn.softmax": [
    (
      [
        [LayoutType.MK],
      ],
      LayoutType.MK,
    )
  ],
  "nn.avg_pool2d": [
    (
      [
        [LayoutType.NCHW],
      ],
      LayoutType.NCHW
    )
  ],
  "nn.adaptive_avg_pool2d": [
    (
      [
        [LayoutType.NCHW],
      ],
      LayoutType.NCHW
    )
  ]
}


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
    # layout_set = set([LayoutType.SCALAR, LayoutType.NCHW, LayoutType.NCHW16C, LayoutType.NCHW64C, LayoutType.QCONV_INPUT, LayoutType.QCONV_WEIGHT, LayoutType.QDCONV_WEIGHT])
    layout_set = set(LayoutType.all())
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
  elif op_name == "reshape":
    new_shape = call.attrs['newshape']
    
    if len(new_shape) == 4:
      output_layout = LayoutType.NCHW
    elif len(new_shape) == 2:
      output_layout = LayoutType.MK
    else:
      raise ValueError(f"reshape new shape rank {len(new_shape)} not supported.")
    rules = [(
      [[LayoutType.MK],
       [LayoutType.NCHW],
       [LayoutType.NCHW16C],
       [LayoutType.NCHW64C],
      ],
      output_layout,
    )]
  elif cpu_node and op_name in CPU_REQUIRED_OP_LAYOUTS.keys():
    rules = CPU_REQUIRED_OP_LAYOUTS.get(op_name, None)
    if rules is None:
      raise ValueError(f"Layout requirement not defined for op {op_name} at CPU")
  elif (not cpu_node) and op_name in IMCFLOW_REQUIRED_OP_LAYOUTS.keys():
    rules = IMCFLOW_REQUIRED_OP_LAYOUTS.get(op_name, None)
    if rules is None:
      raise ValueError(f"Layout requirement not defined for op {op_name} at IMCFLOW")
  else:
    raise ValueError(f"Layout requirement not defined for op {op_name}.")
  
  if not rules: raise ValueError(f"Layout requirement not defined for op {op_name}.")

  return rules

def get_valid_output_layout_of_node(node, input_layouts, mod, cpu_node=False, layout_results=None):
  """
  get valid output layout of call node based on input layouts.
  call node can be built-in op or composite function or global var.
  If call is built-in op, use REQUIRED_OP_LAYOUTS to get output layout.
  if call is composite function or global var, we apply input layouts to function params and
  propagate layouts inside function body to get output layout.
  """

  debug_print(f"[get_valid_output_layout_of_node] node: {getNodeDebugID(node)}, input_layouts: {input_layouts}. cpu_node: {cpu_node}")
  if layout_results is None:
    layout_results = {}

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
            arg_types = [get_type(mod, arg) for arg in node.args]
          else:
            arg_types = get_type(mod, node.args[0]).fields # input of concat
          match = True
          for i in range(len(_inputs)):
            if (not _layout_match(_inputs[i], _option[i])) or (not is_layout_compatible_with_type(_inputs[i], arg_types[i])):
              match = False
              break
          if match:
            if valid_outputs_layout: raise ValueError("multiple valid output layouts found")
            valid_outputs_layout = outputs_layout
    
    if layout_results is not None:
      layout_results[node] = valid_outputs_layout
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
    layout_results.update(layout_dict)

    def _get_layout(expr, call=None, idx=None, not_const_layouts=None):
      if isinstance(expr, relay.Constant):
        assert call is not None and idx is not None, "call and idx must be provided for constant layout deduction."
        assert not_const_layouts is not None, "not_const_layouts must be provided for constant layout deduction."
        debug_print(f"[get_valid_output_layout_of_node] node: {getNodeDebugID(node)}, constant layout deduced.")
        const_layout = _deduce_layout_from_op_const(call, idx, not_const_layouts)
        layout_results[expr] = const_layout
        return const_layout
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
        node_output_layout = get_valid_output_layout_of_node(_node, node_input_layouts, mod, cpu_node=cpu_node, layout_results=layout_results)
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
          out_layout = get_valid_output_layout_of_node(_node, list(comb), mod, cpu_node=cpu_node, layout_results=layout_results)
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
      if layout_results is not None:
        layout_results[_node] = node_output_layout
    
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
    layout_results[node] = output
    debug_print(f"[get_valid_output_layout_of_node] node: {getNodeDebugID(node)}, output layout: {output}")
    return output
  elif isinstance(node, relay.Tuple):
    layout_results[node] = tuple(input_layouts)
    debug_print(f"[get_valid_output_layout_of_node] node: {getNodeDebugID(node)}, output layout: {tuple(input_layouts)}")
    return tuple(input_layouts)
  elif isinstance(node, relay.TupleGetItem):
    if isinstance(input_layouts, (list, tuple)):
      if len(input_layouts) == 1:
        layout_results[node] = input_layouts[0]
        debug_print(f"[get_valid_output_layout_of_node] node: {getNodeDebugID(node)}, output layout: {input_layouts[0]}")
        return input_layouts[0]
      elif len(input_layouts) > node.index:
        layout_results[node] = input_layouts[node.index]
        debug_print(f"[get_valid_output_layout_of_node] node: {getNodeDebugID(node)}, output layout: {input_layouts[node.index]}")
        return input_layouts[node.index]
      else:
        raise ValueError(f"input_layouts length {len(input_layouts)} is less than index {node.index} for TupleGetItem.")
    else:
      raise ValueError("input_layouts must be list or tuple for TupleGetItem node.")

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

      debug_print("Sort parameter combinations with scores.")
      scores = []
      for combo in param_layout_combinations:
        score = 0
        for layout in combo.values():
          score += ImcflowLayoutLegalizer.get_layout_priority()[layout]
        scores.append(score)
      sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
      sorted_param_layout_combinations = [param_layout_combinations[i] for i in sorted_indices]
      param_layout_combinations = sorted_param_layout_combinations
      debug_print(f"Sorted parameter layout combinations by scores: {scores}")

      valid_layouts = []
      for combo in param_layout_combinations:
        layout_results = {}
        output_layout = get_valid_output_layout_of_node(inferred_func, [l for l in combo.values()], self.mod, layout_results=layout_results)
        if output_layout is not None:
          debug_print(f"[layout] combo succeeded: {combo} -> output layout: {output_layout}")
          valid_layouts.append((combo, output_layout, layout_results))
          break
        else:
          debug_print(f"[layout] combo failed: {combo}")

      if len(valid_layouts) == 0:
        raise ValueError("No valid layout cases found for function")

      chosen_inputs, chosen_output, chosen_layout_results = valid_layouts[-1]
      chosen_layout = (chosen_inputs, chosen_output)
      param_types, param_layouts = self._build_param_types(inferred_func, chosen_layout)
      ret_type, ret_layout = self._build_return_type(inferred_func, chosen_layout)
      return param_types, param_layouts, ret_type, ret_layout, chosen_layout_results

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

  @staticmethod
  def infer_cpu_layout_from_type(ttype):
    if isinstance(ttype, TupleType):
      return tuple(ImcflowLayoutLegalizer.infer_cpu_layout_from_type(f) for f in ttype.fields)
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

  @staticmethod
  def get_layout_priority():
    return {
      LayoutType.SCALAR: 4,
      LayoutType.NCHW: 3,
      LayoutType.MK: 3,
      LayoutType.NCHW16C: 2,
      LayoutType.NCHW64C: 1,
      LayoutType.QCONV_INPUT: 0,
    }
  
  @staticmethod
  def get_high_priority_layout(layout1, layout2):
    priority = ImcflowLayoutLegalizer.get_layout_priority()
    p1 = priority.get(layout1, 0)
    p2 = priority.get(layout2, 0)
    return layout1 if p1 >= p2 else layout2

  def __init__(self):
    self.input_call_dict = {}
    self.output_call_dict = {}
    self.imcflow_func_interface_layout_map = {}
    self.layout_map = ImcflowDeviceConfig().LayoutMap
    self.layout_results = {}

  def _dump_layout_results(self, func, layout_results, annotate=False):
    """Pretty-print layout results following graph structure via use-def."""
    def _dump(parser_, node, indent, seen, annotate=False):
      indent_str = "  " * indent
      layout = layout_results.get(node, "unknown")
      debug_print(f"{indent_str}{getNodeDebugID(node)}: {layout}")
      if annotate:
        setattr(node, "debug_layout", str(layout))
      
      # if composite call, dump recursively
      if isinstance(node, relay.Call) and isinstance(node.op, relay.Function) and node.op.attrs and node.op.attrs['Composite']:
        inner_func = node.op
        debug_print(f"{indent_str}  ------------ Entering composite function {inner_func.attrs['Composite']}: ------------")
        parser_inner = UseDefChainParser()
        parser_inner.visit(inner_func.body)
        _dump(parser_inner, inner_func.body, indent + 1, seen, annotate)
        debug_print(f"{indent_str}  ------------ Exiting composite function {inner_func.attrs['Composite']}. ------------")

      # if not traverse producers(childrean)
      for child in parser_.get_uses(node):
        if isinstance(child, (relay.Call, relay.Tuple, relay.TupleGetItem, relay.Var, relay.Constant)):
          _dump(parser_, child, indent + 1, seen, annotate)

    parser = UseDefChainParser()
    parser.visit(func.body)
    _dump(parser, func.body, 0, set(), annotate)

  def create_wrap_func(self, func, func_name, new_param_type, new_param_layout, new_ret_type, new_ret_layout):
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
      for func_impl_param, layout in zip(func_no_attr.params, new_param_layout):
        self.layout_results[func_impl_param] = layout

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
        debug_print('--------------------Legalize---------------------------')
        debug_print(mod[function_names[i]])
        debug_print('-------------------------------------------------------')
        param_type, param_layout, ret_type, ret_layout, layout_results = calculate_imcflow_func_type(mod[function_names[i]], function_names[i])
        self.imcflow_func_interface_layout_map[function_names[i]] = (param_layout, ret_layout)
        self.layout_results[function_names[i]] = layout_results
        debug_print("Created wrapper function for", function_names[i])
        debug_print("  Param Types and Layouts:", param_type,param_layout)
        debug_print("  Return Type and Layout:", ret_type,ret_layout)
        debug_print("  Layout Results:")
        self._dump_layout_results(mod[function_names[i]], layout_results)
        mod[function_names[i]] = self._mark_and_transform_imcflow_qconv(mod[function_names[i]])
        wrap_func, ttype_map = self.create_wrap_func(mod[function_names[i]], function_names[i], param_type, param_layout, ret_type, ret_layout)
        temp_mod = tvm.IRModule.from_expr(wrap_func)
        temp_mod = relay.transform.InferType()(temp_mod)
        wrap_func = temp_mod[list(temp_mod.get_global_vars())[0]]
        real_tensor_type_map[function_names[i]] = ttype_map
        old_gv = mod.get_global_var(function_names[i])
        func_type = relay.FuncType([x.type_annotation for x in wrap_func.params], wrap_func.ret_type)
        new_gv = relay.GlobalVar(function_names[i], type_annot=func_type)
        del mod[old_gv]
        mod[new_gv] = wrap_func
        new_gv_map[old_gv] = new_gv

        debug_print("Wrap function created for", function_names[i])
        debug_print(wrap_func)

    debug_print("-"*40)
    debug_print("imcflow function interface update results:")
    debug_print(mod.astext())

    mod = self.replace_imcflow_gv(mod, new_gv_map)
    mod = self._insert_packing_unpacking(mod, real_tensor_type_map, self.imcflow_func_interface_layout_map)
    self.construct_layout_map(mod)

    # dump layout with graph structure
    debug_print("[FINAL LAYOUT RESULTS] Dump layout results with graph structure:")
    self.dump_mod(mod)

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
        return ImcflowLayoutLegalizer.infer_cpu_layout_from_type(ttype)

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
        ttype = get_type(self.module, expr)
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
          self.layout_map[expr] = target_layout
          return expr, target_layout

        if curr_layout in (LayoutType.NCHW16C, LayoutType.NCHW64C) and target_layout == LayoutType.NCHW:
          layout_str = self._layout_to_str(curr_layout)
          expr = relay.op.layout_transform(expr, layout_str, "NCHW")
          self.layout_map[expr] = target_layout
          if channel_hint is not None:
            ttype = get_type(self.module, expr)
            if isinstance(ttype, TensorType) and len(ttype.shape) == 4:
              N, C, H, W = [int(x) for x in ttype.shape]
              if C > channel_hint:
                expr = relay.op.strided_slice(expr, begin=[0, 0, 0, 0], end=[N, channel_hint, H, W])
                self.layout_map[expr] = target_layout
          return expr, target_layout

        if curr_layout == LayoutType.NCHW and target_layout == LayoutType.QCONV_INPUT:
          new_expr = imcflow_4d_to_qconv_input(expr)
          if new_expr == expr:
             debug_print(f"Warning: imcflow_4d_to_qconv_input returned identity for {getNodeDebugID(expr)}")
          self.layout_map[new_expr] = target_layout
          return new_expr, target_layout

        if curr_layout == LayoutType.QCONV_INPUT and target_layout == LayoutType.NCHW:
          channels = channel_hint if channel_hint is not None else self._channels_from_expr(expr)
          channels = channels if channels is not None else 0
          new_expr = imcflow_mmquant_out_to_4d(expr, channels)
          self.layout_map[new_expr] = target_layout
          return new_expr, target_layout

        if curr_layout == LayoutType.NCHW16C and target_layout == LayoutType.NCHW64C:
          expr = relay.op.layout_transform(expr, "NCHW16c", "NCHW64c")
          self.layout_map[expr] = target_layout
          return expr, target_layout

        if curr_layout == LayoutType.NCHW64C and target_layout == LayoutType.NCHW16C:
          expr = relay.op.layout_transform(expr, "NCHW64c", "NCHW16c")
          self.layout_map[expr] = target_layout
          return expr, target_layout

        raise ValueError(f"node : {getNodeDebugID(expr)}. Unsupported layout conversion from {curr_layout} to {target_layout}")

      def visit_var(self, var):
        layout = self._infer_layout_from_type(var.type_annotation)
        if not layout: raise ValueError(f"Cannot infer layout from var type: {var.type_annotation}")
        self.layout_map[var] = layout
        return var

      def visit_constant(self, const):
        layout = self._infer_layout_from_type(const.checked_type)
        if not layout: raise ValueError(f"Cannot infer layout from constant type: {const.checked_type}")
        self.layout_map[const] = layout
        debug_print("[insert_packing_unpacking] Constant: layout:", layout)
        return const

      def visit_tuple(self, tup):
        new_fields = [self.visit(f) for f in tup.fields]
        layout = tuple(self.layout_map[f] for f in new_fields)
        new_tup = relay.Tuple(new_fields)
        self.layout_map[new_tup] = layout
        debug_print("[insert_packing_unpacking] Tuple: In layouts:", [self.layout_map[f] for f in new_fields], " | Out layout:", layout)
        return new_tup

      def visit_tuple_getitem(self, tgi):
        new_val = self.visit(tgi.tuple_value)
        val_layout = self.layout_map[new_val]
        
        if isinstance(val_layout, (tuple, list)):
            if tgi.index < len(val_layout):
                layout = val_layout[tgi.index]
            else:
                # Fallback if index out of bounds (should not happen for valid IR)
                layout = val_layout[0]
        else:
            # If val_layout is a single layout, assume it applies to all fields
            # This handles cases where a Tuple might be treated as having a single layout (e.g. QCONV_INPUT)
            layout = val_layout

        new_tgi = relay.TupleGetItem(new_val, tgi.index)
        if new_tgi in self.layout_map:
          # raise ValueError("Layout already exists for TupleGetItem node.")
          pass # Allow revisiting if memoization didn't catch it (though it should)
        
        self.layout_map[new_tgi] = layout
        debug_print("[insert_packing_unpacking] TupleGetItem: index", tgi.index, " | In layout:", val_layout, " | Out layout:", layout)
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
              #TODO: if multiple required layouts, choose the best one
              if LayoutType.MK in req_layouts and LayoutType.NCHW in req_layouts:
                tgt_layout = req_layouts[req_layouts.index(LayoutType.NCHW)]
              else:
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
        relay.transform.InferType()(tvm.IRModule.from_expr(new_call))
        if isinstance(call.op, relay.GlobalVar) and isImcflowFunc(self.module[call.op.name_hint], self.module):
          out_layout = self.imcflow_func_layout_map[call.op.name_hint][1]
        else:
          out_layout = get_valid_output_layout_of_node(new_call, target_arg_layouts, self.module, True)
        call_type = get_type(self.module, new_call)
        if isinstance(call_type, TupleType) and not isinstance(out_layout, (tuple, list)):
          out_layout = tuple(out_layout for _ in call_type.fields)
        debug_print(f"[insert_packing_unpacking] Call: {new_call.op} | In layouts: {target_arg_layouts} | Out layout: {out_layout}")
        self.layout_map[new_call] = out_layout
        return new_call

      def visit_function(self, fn):
        new_body = self.visit(fn.body)
        ret_layout = self.layout_map[new_body]
        if not self._layout_equal(ret_layout, LayoutType.NCHW) and ret_layout in [LayoutType.NCHW16C, LayoutType.NCHW64C, LayoutType.QCONV_INPUT]:
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

  def _find_impl_func(self, func):
    class _Visitor(relay.ExprVisitor):
      def __init__(self):
        super().__init__()
        self.impl_func = None

      def visit_call(self, call):
        if isinstance(call.op, relay.Function): 
          self.impl_func = call.op
        else:
          super().visit_call(call)
    visitor = _Visitor()
    visitor.visit(func.body)
    return visitor.impl_func
  
  def _find_composite(self, func):
    class _Visitor(relay.ExprVisitor):
      def __init__(self):
        super().__init__()
        self.composite_calls = []

      def visit_call(self, call):
        if isinstance(call.op, tvm.ir.Op):
          attrs = call.attrs
          if attrs and "Composite" in attrs.keys():
            self.composite_calls.append(call)
        super().visit_call(call)
    visitor = _Visitor()
    visitor.visit(func.body)
    return visitor.composite_name
  
  def dump_mod(self, mod):
    """
    dump layout map with graph structure by using _dump_layout_results.
    dump main function first and loop global vars
    """ 

    debug_print("[FINAL LAYOUT RESULTS] main")
    self._dump_layout_results(mod["main"], self.layout_map, True)
    for gv in mod.get_global_vars():
      if gv.name_hint == "main":
        continue
      func = mod[gv]
      if isImcflowFunc(func, mod):
        debug_print(f"[FINAL LAYOUT RESULTS] {gv.name_hint}")
        impl_func = self._find_impl_func(func)
        self._dump_layout_results(impl_func, self.layout_map, True)

  def construct_layout_map(self, mod):
    """
    Construct layout map of module.
    We traverse imcflow functions first using imcflow_func_interface_layout_map.
    We apply the input layouts to imcflow function params and construct layout map 
    by propagating through the graph using get_valid_output_layout_of_node.
    After that, we traverse main function (without recursing into global function calls).
    Finally, clear and update ImcflowDeviceConfig().LayoutMap with the new map.
    """
    new_layout_map = {}

    class LayoutPropagator(relay.ExprVisitor):
      def __init__(self, layout_map, module, imcflow_func_layout_map):
        super().__init__()
        self.layout_map = layout_map
        self.module = module
        self.imcflow_func_layout_map = imcflow_func_layout_map
        self.skip_global_calls = False  # Flag to skip recursing into global function calls

      def visit_var(self, var):
        if self.skip_global_calls:
          layout = ImcflowLayoutLegalizer.infer_cpu_layout_from_type(var.type_annotation)
          self.layout_map[var] = layout
        else:
          if var not in self.layout_map:
            raise ValueError(f"Imcflow Variable layout not found in layout map: {var.name_hint}")

      def visit_constant(self, const):
        if self.skip_global_calls:
          layout = ImcflowLayoutLegalizer.infer_cpu_layout_from_type(const.checked_type)
          self.layout_map[const] = layout
        else:
          pass

      def visit_call(self, call):
        # Visit arguments first
        for arg in call.args:
          self.visit(arg)

        # Handle global function calls (imcflow functions)
        if isinstance(call.op, relay.GlobalVar):
          if self.skip_global_calls:
            # In main function: use interface layout map
            if call.op.name_hint in self.imcflow_func_layout_map:
              out_layout = self.imcflow_func_layout_map[call.op.name_hint][1]  # return layout
              self.layout_map[call] = out_layout
        else:
          # Determine output layout based on operation and input layouts
          arg_layouts = [self.layout_map[arg] for arg in call.args]
          out_layout = get_valid_output_layout_of_node(call, arg_layouts, self.module, True, self.layout_map)
          if out_layout is None: raise ValueError(f"Cannot determine output layout for call: {call.op}")
          self.layout_map[call] = out_layout

      def visit_tuple(self, tup):
        for field in tup.fields:
          self.visit(field)
        # Tuple layout is tuple of field layouts
        field_layouts = tuple(self.layout_map[f] for f in tup.fields)
        self.layout_map[tup] = field_layouts

      def visit_tuple_getitem(self, tgi):
        self.visit(tgi.tuple_value)
        tuple_layout = self.layout_map[tgi.tuple_value]
        if isinstance(tuple_layout, (tuple, list)) and tgi.index < len(tuple_layout):
          self.layout_map[tgi] = tuple_layout[tgi.index]
        else:
          self.layout_map[tgi] = tuple_layout

      def visit_function(self, fn):
        # Visit parameters
        for param in fn.params:
          self.visit(param)
        # Visit body
        self.visit(fn.body)
        # Function layout is body layout
        body_layout = self.layout_map[fn.body]
        self.layout_map[fn] = body_layout

    # Step 1: Process imcflow functions with known interface layouts
    for gv in mod.get_global_vars():
      if gv.name_hint == "main":
        continue
      func = mod[gv]
      if isImcflowFunc(func, mod) and gv.name_hint in self.imcflow_func_interface_layout_map:
        param_layouts, ret_layout = self.imcflow_func_interface_layout_map[gv.name_hint]
        impl_func = self._find_impl_func(func)
        if impl_func is None: raise ValueError(f"Cannot find implementation function inside imcflow function: {gv.name_hint}")
        for i, param in enumerate(impl_func.params):
          layout = param_layouts[i]
          new_layout_map[param] = layout
        out_layout = get_valid_output_layout_of_node(impl_func, param_layouts, mod, False, new_layout_map)
        if isinstance(out_layout, tuple): out_layout = list(out_layout)
        if out_layout != ret_layout: raise ValueError(f"Output layout mismatch for function {gv.name_hint}: expected {ret_layout}, got {out_layout}")

    # Step 2: Process main function (skip recursing into global calls)
    main_func = mod["main"]
    propagator = LayoutPropagator(new_layout_map, mod, self.imcflow_func_interface_layout_map)
    propagator.skip_global_calls = True  # Don't recurse into global function calls
    propagator.visit(main_func)
    debug_print(f"[construct_layout_map] Processed main function")

    # Step 3: Update global layout map
    ImcflowDeviceConfig().LayoutMap.clear()
    ImcflowDeviceConfig().LayoutMap.update(new_layout_map)
    self.layout_map = new_layout_map
    debug_print(f"[construct_layout_map] Updated global LayoutMap with {len(new_layout_map)} entries")