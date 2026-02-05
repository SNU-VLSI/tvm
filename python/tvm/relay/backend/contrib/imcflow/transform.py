import pickle
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
from tvm.contrib.imcflow import ImcflowDeviceConfig, TensorEdge, TensorID, NodeID, TensorEdgeInfo, InstEdgeInfo, RouterEntry, DataBlock, MemoryLayout, MemoryRegion, BlockTileInfo
from tvm.ir import Op
from tvm.relay.op.contrib.imcflow import HashToCustomID, CustomIDToName, CustomIDInFunc, CustomIDToNode

from tvm.relay.backend.contrib.imcflow.layout import ImcflowLayoutLegalizer, apply_layout_to_type
from tvm.relay.backend.contrib.imcflow import layout as imcflow_layout
from tvm.relay.backend.contrib.imcflow import transform_utils
from tvm.relay.backend.contrib.imcflow.transform_utils import getInodePktCntForEdge, NodeCollector
from itertools import cycle
from collections import defaultdict

# Debug logging utility controlled by IMCFLOW_DEBUG environment variable
# Usage:
#   export IMCFLOW_DEBUG=1  # Enable all debug messages
#   export IMCFLOW_DEBUG=0  # Disable all debug messages
_DEBUG_ENABLED = None

INODE_DMEM_FOR_TILE = ImcflowDeviceConfig.INODE_DATA_MEM_SIZE

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
from typing import Dict, List, Any, Optional
import json
import pprint
import os

#TODO:
# 1. memory allocation is not optimal. we can split tensor to multiple inode. current implementation assign only one inode for entire tensor.

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


def getNodeID(node) -> int:
  id_dict = HashToCustomID()
  if int(hash(node)) in id_dict:
    return id_dict[int(hash(node))]
  else:
    return None

def getNodeDebugID(node):
  if isinstance(node, relay.Call):
    if isinstance(node.op, (tvm.ir.Op, tvm.ir.op.Op)):
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
  elif isinstance(node, relay.Constant):
    indicator = "const"
  elif isinstance(node, relay.Var):
    indicator = node.name_hint
  elif isinstance(node, relay.TupleGetItem):
    indicator = f"tuple_get_item_{node.index}"
  elif isinstance(node, relay.Tuple):
    indicator = "tuple"
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

    try:
      if isinstance(node, relay.Call) and isinstance(node.op, tvm.ir.Op):
        out_type = relay.transform.InferTypeLocal(node)
      elif isinstance(node, relay.Call) and isinstance(node.op, relay.Function):
        # out_type = node.op.body.checked_type
        out_type = relay.transform.InferTypeLocal(node.op.body)
      elif isinstance(node, relay.Call) and isinstance(node.op, relay.GlobalVar):
        # out_type = _get_type(parent_mod, parent_mod[node.op.name_hint].body)
        out_type = node.op.checked_type.ret_type
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
    except Exception as e:
      debug_print(f"Type inference failed for node {getNodeDebugID(node)}: {e}")
      debug_print(f"node:")
      debug_print(node)
      raise e

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

    #- we never include min_max_quant as conv2d post op. min_max_quant never be split into multiple nodes.
    post_op_candidates = [op.get("nn.bias_add"), op.get("nn.relu"), op.get("nn.batch_norm"), op.get("imcflow.fused_batch_norm")]
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
                        stride=strides[0] if isinstance(strides, (list, tuple)) else strides,
                        use_imcu=True if (not IsDepthWise) else False
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
            elif call.op in post_op_candidates:
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


def merge_composite_for_partition(mod):
    """
    Merge composite ops before partitionRound for simpler converge point detection.

    This function merges ops into composites using the imcflow pattern table,
    making the graph simpler for region partitioning.

    After partitionRound, use unmerge_composite to inline the composites back.
    """
    for global_var, func in mod.functions.items():
        if isinstance(func, relay.Function) and "Compiler" in func.attrs and re.match(r"imcflow.*", func.attrs["Compiler"]):
            attr_record = func.attrs
            func_no_attr = relay.Function(func.params, func.body)
            target_mod = tvm.IRModule.from_expr(func_no_attr)
            transformed = transform.MergeComposite(imcflow.pattern_table())(target_mod)
            _, transformed_func = transformed.functions.items()[0]
            transformed_func = relay.Function(transformed_func.params, transformed_func.body,
                                              ret_type=transformed_func.ret_type, attrs=attr_record)
            mod[global_var] = transformed_func
    mod = relay.transform.InferType()(mod)
    return mod


def unmerge_composite(mod):
    """
    Unmerge (inline) composites after partitionRound.

    This function inlines all composite functions back to their original ops,
    so that subsequent transforms (split_conv_to_atomic, etc.) can process them.

    Uses a custom ExprMutator to inline composite function calls by substituting
    the function body with actual arguments.
    """

    class CompositeInliner(relay.ExprMutator):
        """Inline composite function calls by substituting function body with args."""

        def visit_call(self, call):
            # First visit args recursively
            new_args = [self.visit(arg) for arg in call.args]

            # Check if this is a composite function call
            if isinstance(call.op, relay.Function) and "Composite" in call.op.attrs:
                composite_func = call.op
                composite_name = composite_func.attrs["Composite"]
                debug_print(f"[CompositeInliner] Inlining composite: {composite_name}")

                # Create substitution map: param -> arg
                subst_map = {}
                for param, arg in zip(composite_func.params, new_args):
                    subst_map[param] = arg

                # Substitute parameters in the function body
                class ParamSubstitutor(relay.ExprMutator):
                    def __init__(self, subst_map):
                        super().__init__()
                        self.subst_map = subst_map

                    def visit_var(self, var):
                        if var in self.subst_map:
                            return self.subst_map[var]
                        return var

                substitutor = ParamSubstitutor(subst_map)
                inlined_body = substitutor.visit(composite_func.body)

                # Recursively inline any nested composites
                return self.visit(inlined_body)

            # Not a composite, just update args if changed
            if new_args != list(call.args):
                return relay.Call(call.op, new_args, call.attrs, call.type_args, call.span)
            return call

    for global_var, func in mod.functions.items():
        # Process round functions (created by partitionRound)
        if isinstance(func, relay.Function) and "Compiler" in func.attrs:
            compiler = func.attrs["Compiler"]
            # Match round functions like "tvmgen_default_imcflow_main_47_round_imcflow_region1"
            if re.match(r".*_round_imcflow_region\d+", compiler) or re.match(r"imcflow.*", compiler):
                attr_record = func.attrs

                # Inline composites
                inliner = CompositeInliner()
                new_body = inliner.visit(func.body)

                # Create new function with inlined body
                new_func = relay.Function(func.params, new_body, ret_type=func.ret_type, attrs=attr_record)
                mod[global_var] = new_func

    mod = relay.transform.InferType()(mod)
    return mod

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


# ==========================================
# Converge Point Detection & Branch Analysis
# ==========================================

class LatencyThroughputCalculator:
    """
    Calculate latency and throughput for operations.

    Latency: Total cycles from input arrival to output ready
    Throughput: Operations per cycle (inverse of initiation interval)

    These values depend on operation type and attributes (kernel size, channels, etc.)
    """

    def __init__(self, module=None):
        """
        Args:
            module: The TVM IRModule containing the functions being analyzed.
                    Used to resolve types and access function definitions.
        """
        self.module = module

    def get_op_latency(self, op_call) -> int:
        """
        Get latency for a single operation.

        Args:
            op_call: relay.Call node (can be inside composite or standalone)

        Returns:
            Latency in cycles

        Test values for development:
        - Conv ops: high latency (10-20 cycles) - represents MAC operations
        - Element-wise ops: low latency (1 cycle)
        - Quantize ops: medium latency (2-3 cycles)

        TODO: Replace with actual hardware-based formulas
        """
        if not isinstance(op_call, relay.Call):
            return 0

        if isinstance(op_call.op, tvm.ir.Op):
            op_name = op_call.op.name
            attrs = op_call.attrs

            if op_name in ["nn.imcflow_qdwconv", "nn.imcflow_qconv", "nn.conv2d"]:
                # Conv2D: high latency due to linebuffer fill delay
                input_tensor_type = transform_utils.get_type(self.module, op_call.args[0])
                input_shape = input_tensor_type.shape
                N, IC, IH, IW = list(input_shape)
                kernel_h = int(attrs.kernel_size[0]) if attrs.kernel_size else 3
                kernel_w = int(attrs.kernel_size[1]) if attrs.kernel_size else 3
                padding, strides = attrs.padding, attrs.strides

                latency = IW * max(0, (kernel_h - padding[0])) + max(0, (kernel_w - padding[1]))
                return latency

            elif op_name in ["nn.relu", "nn.bias_add"]:
                # Element-wise ops: very fast
                return 1

            elif op_name == "add":
                # Add operation: fast element-wise
                return 1

            elif op_name == "qnn.imcflow_min_max_quantize":
                # Quantization: medium latency (comparison + shift)
                return 1

            elif op_name == "qnn.imcflow_nu_quantize":
                # Non-uniform quantization: slightly higher (LUT lookup)
                return 1

            elif op_name == "imcflow.fused_batch_norm":
                # Fused batch norm: multiply + add
                return 1

            elif op_name in ["split", "concatenate"]:
                # Data movement ops: depends on data size, use small value
                return 1

            elif op_name in ["multiply", "divide"]:
                # Arithmetic ops
                return 1

            else:
                debug_print(f"[LatencyThroughputCalculator] Unknown op: {op_name}")
                return 1

        return 0

    def get_op_throughput(self, op_call) -> int:
        """
        Get throughput for a single operation.

        Args:
            op_call: relay.Call node

        Returns:
            Throughput (higher is better, represents data elements per cycle)
            Lower value = bottleneck

        Test values for development:
        - Conv ops: low throughput (limited by MAC units)
        - Element-wise ops: high throughput (fully pipelined)
        - Quantize ops: medium throughput

        TODO: Replace with actual hardware-based formulas
        """
        if not isinstance(op_call, relay.Call):
            return 100  # Non-ops don't limit throughput

        if isinstance(op_call.op, tvm.ir.Op):
            op_name = op_call.op.name
            attrs = op_call.attrs

            if op_name in ["nn.imcflow_qdwconv", "nn.imcflow_qconv", "nn.conv2d"]:
                # Conv2D: throughput limited by MAC array
                # Lower output channels = lower throughput
                strides = attrs.strides
                stride_val = int(strides[1]) if strides else 1
                return 1.0 / stride_val

            elif op_name in ["nn.relu", "nn.bias_add", "add", "multiply", "divide"]:
                # Element-wise: very high throughput (pipelined)
                return 1

            elif op_name == "qnn.imcflow_min_max_quantize":
                # Quantization: medium-high throughput
                return 1

            elif op_name == "qnn.imcflow_nu_quantize":
                # Non-uniform quantization: medium (LUT access)
                return 1

            elif op_name == "imcflow.fused_batch_norm":
                # Fused batch norm: high throughput
                return 1

            elif op_name in ["split", "concatenate"]:
                # Data movement: high throughput
                return 1

            else:
                return 1

        return 1

    def calculate_branch_latency(self, ops: list) -> int:
        """
        Calculate total latency for a branch (sum of all op latencies).

        Args:
            ops: List of relay.Call nodes in the branch

        Returns:
            Total latency in cycles
        """
        return sum(self.get_op_latency(op) for op in ops)

    def calculate_branch_throughput(self, ops: list) -> int:
        """
        Calculate effective throughput for a branch (bottleneck).

        Args:
            ops: List of relay.Call nodes in the branch

        Returns:
            Minimum throughput (bottleneck determines overall throughput)
        """
        if not ops:
            return 1
        throughputs = [self.get_op_throughput(op) for op in ops]
        return min(throughputs) if throughputs else 1


class BranchAnalyzer:
    """
    Analyze diverge-converge patterns in the computation graph.

    Detects converge points where multiple branches merge, extracts branch
    operations (including inside composites), and provides branch metrics.
    """

    def __init__(self, edges, rev_edges):
        """
        Args:
            edges: dict mapping node -> list of successor nodes
            rev_edges: dict mapping node -> list of predecessor nodes
        """
        self.edges = edges
        self.rev_edges = rev_edges

    def is_converge_point(self, node) -> bool:
        """
        Check if a node is a converge point.

        A converge point is where 2+ branches from a common diverge point merge.
        This includes cases where one predecessor IS the diverge point (skip connection).

        Args:
            node: The node to check

        Returns:
            True if this is a converge point
        """
        debug_print(f"\n{'='*60}")
        debug_print(f"[BranchAnalyzer.is_converge_point] Checking node:")
        debug_print(f"  ID: {getNodeDebugID(node)}")
        debug_print(f"  Expr:\n{node}")
        debug_print(f"{'='*60}")

        # Get predecessors
        preds = self.rev_edges.get(node, [])
        debug_print(f"  Number of predecessors: {len(preds)}")
        for i, p in enumerate(preds):
            debug_print(f"    Pred {i}: {getNodeDebugID(p)}")

        if len(preds) < 2:
            debug_print(f"  Result: False (< 2 preds)")
            return False

        # Find ancestors for each predecessor (including the predecessor itself)
        pred_ancestor_sets = []
        for i, pred in enumerate(preds):
            ancestors = self._get_all_ancestors(pred)
            ancestors.add(pred)  # Include the predecessor itself
            pred_ancestor_sets.append(ancestors)
            debug_print(f"  Pred {i} ancestors count: {len(ancestors)}")
            for anc in list(ancestors)[:5]:  # Show first 5
                debug_print(f"    - {getNodeDebugID(anc)}")
            if len(ancestors) > 5:
                debug_print(f"    ... and {len(ancestors) - 5} more")

        # Check if there's a common ancestor (diverge point)
        # Also check if one predecessor is ancestor of another (skip connection case)
        if len(pred_ancestor_sets) >= 2:
            common = pred_ancestor_sets[0]
            for ancestors in pred_ancestor_sets[1:]:
                common = common & ancestors

            debug_print(f"  Common ancestors count: {len(common)}")
            for anc in list(common)[:5]:
                debug_print(f"    - {getNodeDebugID(anc)}")

            # If there's a common ancestor, it's a converge point
            if common:
                debug_print(f"  Result: True (has common ancestor)")
                return True

        debug_print(f"  Result: False (no common ancestor)")
        return False

    def _get_all_ancestors(self, node, max_depth=50) -> set:
        """Get all ancestor nodes up to max_depth."""
        ancestors = set()
        queue = [(node, 0)]
        visited = {node}

        while queue:
            curr, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            for pred in self.rev_edges.get(curr, []):
                if pred not in visited:
                    visited.add(pred)
                    ancestors.add(pred)
                    queue.append((pred, depth + 1))

        return ancestors

    def find_diverge_point(self, converge_node):
        """
        Find the diverge point (common ancestor) for a converge node.

        Args:
            converge_node: The converge point node

        Returns:
            The diverge point node, or None if not found
        """
        debug_print(f"\n{'='*60}")
        debug_print(f"[BranchAnalyzer.find_diverge_point] Finding diverge for:")
        debug_print(f"  Converge node: {getNodeDebugID(converge_node)}")
        debug_print(f"{'='*60}")

        preds = self.rev_edges.get(converge_node, [])
        debug_print(f"  Predecessors: {len(preds)}")
        for i, p in enumerate(preds):
            debug_print(f"    Pred {i}: {getNodeDebugID(p)}")

        if len(preds) < 2:
            debug_print(f"  Result: None (< 2 preds)")
            return None

        # Get ancestors for each predecessor with path tracking
        # Include the predecessor itself in case it IS the diverge point (skip connection)
        pred_ancestors = []
        for i, pred in enumerate(preds):
            ancestors = self._get_all_ancestors(pred)
            ancestors.add(pred)  # Include pred itself for skip connection detection
            pred_ancestors.append(ancestors)
            debug_print(f"  Pred {i} ({getNodeDebugID(pred)}) ancestors: {len(ancestors)}")

        # Find common ancestors
        if len(pred_ancestors) < 2:
            debug_print(f"  Result: None (< 2 ancestor sets)")
            return None

        common = pred_ancestors[0]
        for ancestors in pred_ancestors[1:]:
            common = common & ancestors

        debug_print(f"  Common ancestors: {len(common)}")
        if not common:
            debug_print(f"  Result: None (no common ancestors)")
            return None

        # Find the closest common ancestor (diverge point)
        # by finding the one with shortest max distance from all preds
        min_dist = float('inf')
        diverge_point = None

        for ancestor in common:
            max_dist = 0
            for pred in preds:
                dist = self._distance_to_ancestor(pred, ancestor)
                if dist is not None:
                    max_dist = max(max_dist, dist)

            debug_print(f"    Candidate: {getNodeDebugID(ancestor)}, max_dist={max_dist}")
            if max_dist < min_dist:
                min_dist = max_dist
                diverge_point = ancestor

        debug_print(f"  Selected diverge point: {getNodeDebugID(diverge_point)}")
        debug_print(f"  Diverge point expr:\n{diverge_point}")
        debug_print(f"{'='*60}\n")

        return diverge_point

    def _distance_to_ancestor(self, node, ancestor, max_depth=50) -> int:
        """Calculate distance from node to ancestor."""
        if node == ancestor:
            return 0

        queue = [(node, 0)]
        visited = {node}

        while queue:
            curr, dist = queue.pop(0)
            if dist >= max_depth:
                continue

            for pred in self.rev_edges.get(curr, []):
                if pred == ancestor:
                    return dist + 1
                if pred not in visited:
                    visited.add(pred)
                    queue.append((pred, dist + 1))

        return None

    def extract_branches(self, diverge_node, converge_node) -> list:
        """
        Extract operations in each branch from diverge to converge point.

        Args:
            diverge_node: The common ancestor where branches split
            converge_node: The node where branches merge

        Returns:
            List of lists, where each inner list contains ops in one branch
            Operations are extracted at the operation level (inside composites)
        """
        debug_print(f"\n{'='*60}")
        debug_print(f"[BranchAnalyzer.extract_branches]")
        debug_print(f"  Diverge: {getNodeDebugID(diverge_node)}")
        debug_print(f"  Converge: {getNodeDebugID(converge_node)}")
        debug_print(f"{'='*60}")

        preds = self.rev_edges.get(converge_node, [])
        branches = []

        for i, pred in enumerate(preds):
            branch_ops = []
            self._collect_branch_ops(pred, diverge_node, branch_ops, set())
            branches.append(branch_ops)
            debug_print(f"  Branch {i} (from {getNodeDebugID(pred)}): {len(branch_ops)} ops")
            for j, op in enumerate(branch_ops):
                if isinstance(op, relay.Call) and isinstance(op.op, tvm.ir.Op):
                    debug_print(f"    Op {j}: {op.op.name}")
                else:
                    debug_print(f"    Op {j}: {getNodeDebugID(op)}")

        debug_print(f"{'='*60}\n")
        return branches

    def _collect_branch_ops(self, node, stop_node, ops_list, visited):
        """
        Recursively collect operations from node back to stop_node.
        Flattens composite functions to get individual operations.
        """
        if node == stop_node or node in visited:
            return

        visited.add(node)

        if isinstance(node, relay.Call):
            # Flatten composite to get individual ops
            flattened = self.flatten_composite_ops(node)
            ops_list.extend(flattened)

        # Continue to predecessors
        for pred in self.rev_edges.get(node, []):
            self._collect_branch_ops(pred, stop_node, ops_list, visited)

    def flatten_composite_ops(self, call) -> list:
        """
        Extract individual operations from a composite function call.

        Args:
            call: relay.Call node (may be composite or regular op)

        Returns:
            List of relay.Call nodes representing individual operations
        """
        if not isinstance(call, relay.Call):
            return []

        # Check if this is a composite function
        if isinstance(call.op, relay.Function) and hasattr(call.op.attrs, "Composite"):
            # Traverse the composite body to extract ops
            ops = []
            self._extract_ops_from_expr(call.op.body, ops)
            return ops
        elif isinstance(call.op, tvm.ir.Op):
            # Regular operation
            return [call]

        return []

    def _extract_ops_from_expr(self, expr, ops_list):
        """Recursively extract Call nodes from an expression."""
        if isinstance(expr, relay.Call):
            if isinstance(expr.op, tvm.ir.Op):
                ops_list.append(expr)
            # Continue into args
            for arg in expr.args:
                self._extract_ops_from_expr(arg, ops_list)
        elif isinstance(expr, relay.Tuple):
            for field in expr.fields:
                self._extract_ops_from_expr(field, ops_list)
        elif isinstance(expr, relay.TupleGetItem):
            self._extract_ops_from_expr(expr.tuple_value, ops_list)


class CompositeSplitter:
    """
    Split composite functions at converge points.

    Uses pattern.partition() to create new composite functions with
    different boundaries.
    """

    @staticmethod
    def find_converge_op_in_composite(composite_call) -> relay.Call:
        """
        Find the converge operation (typically 'add') inside a composite.

        Args:
            composite_call: The composite function call

        Returns:
            The converge operation (add node) or None
        """
        if not isinstance(composite_call.op, relay.Function):
            return None

        body = composite_call.op.body

        # Look for 'add' operation that has multiple input branches
        class ConvergeOpFinder(relay.ExprVisitor):
            def __init__(self):
                super().__init__()
                self.converge_op = None
                self.param_set = set()

            def visit_call(self, call):
                if isinstance(call.op, tvm.ir.Op) and call.op.name == "add":
                    # Check if args come from different sources
                    # Valid converge point: one arg is Call (from internal op) and
                    # other arg is either Call or Var (function parameter for external input)
                    arg0_is_call = isinstance(call.args[0], relay.Call)
                    arg1_is_call = isinstance(call.args[1], relay.Call)
                    arg0_is_param = isinstance(call.args[0], relay.Var) and call.args[0] in self.param_set
                    arg1_is_param = isinstance(call.args[1], relay.Var) and call.args[1] in self.param_set

                    # Converge point if: (Call, Call) or (Call, Param) or (Param, Call)
                    has_call = arg0_is_call or arg1_is_call
                    has_param_or_call = (arg0_is_call or arg0_is_param) and (arg1_is_call or arg1_is_param)
                    if has_call and has_param_or_call:
                        self.converge_op = call
                super().visit_call(call)

        finder = ConvergeOpFinder()
        finder.param_set = set(composite_call.op.params)
        finder.visit(body)

        return finder.converge_op

    @staticmethod
    def _collect_ops_before_converge(body, converge_op):
        """
        Collect the chain of operations from output back to converge_op (exclusive).
        Returns list of op names in output-to-input order.
        """
        post_ops = []
        if isinstance(body, relay.Call):
            curr = body
            while isinstance(curr, relay.Call) and isinstance(curr.op, tvm.ir.Op):
                op_name = curr.op.name
                if curr == converge_op or op_name == converge_op.op.name:
                    break
                post_ops.append(op_name)
                if curr.args:
                    curr = curr.args[0]
                else:
                    break
        return post_ops

    @staticmethod
    def _collect_ops_in_branch(start_expr, stop_at_param=True):
        """
        Collect ops in a branch from start_expr back to inputs.
        Returns list of (op_name, has_const_args) tuples.
        """
        ops = []
        visited = set()

        def traverse(expr):
            if expr in visited:
                return
            visited.add(expr)

            if isinstance(expr, relay.Call) and isinstance(expr.op, tvm.ir.Op):
                op_name = expr.op.name
                has_const = any(isinstance(arg, relay.Constant) for arg in expr.args)
                ops.append((op_name, has_const))
                for arg in expr.args:
                    traverse(arg)
            elif isinstance(expr, relay.Var):
                return  # Stop at function params
            elif isinstance(expr, relay.Constant):
                return  # Stop at constants

        traverse(start_expr)
        return ops

    @staticmethod
    def split_composite_at_converge(composite_call, converge_op, composite_name_prefix="imcflow"):
        """
        Split a composite function at the converge point into two composites.

        Strategy:
        1. Inline the composite function body
        2. Post-BFS traverse the graph to build patterns dynamically
        3. Build pre-converge pattern (input → converge_op inputs)
        4. Build post-converge pattern (converge_op → output)
        5. Apply partition with generated patterns

        Args:
            composite_call: The original composite function call
            converge_op: The converge operation inside the composite (add node)
            composite_name_prefix: Prefix for new composite names

        Returns:
            Dict with split info or None if split not possible:
            {
                "result_expr": The transformed expression,
                "pre_composite_name": Name of pre-converge composite,
                "post_composite_name": Name of post-converge composite,
            }
        """
        from tvm.relay.dataflow_pattern import wildcard, is_op, is_constant

        if not isinstance(composite_call.op, relay.Function):
            return None

        func = composite_call.op
        body = func.body
        param_set = set(func.params)

        # Step 1: Inline the composite - create var_map and bind
        var_map = {}
        for param, arg in zip(func.params, composite_call.args):
            var_map[param] = arg

        inlined_body = relay.bind(body, var_map)

        # Step 2: Post-BFS traverse to collect nodes and their order
        # We need to identify: pre-converge nodes, converge node, post-converge nodes
        class GraphAnalyzer(relay.ExprVisitor):
            def __init__(self, converge_op, param_set):
                super().__init__()
                self.converge_op = converge_op
                self.param_set = param_set
                self.node_order = []  # Post-order traversal result
                self.converge_idx = -1
                self.visited = set()

            def visit(self, expr):
                if expr in self.visited:
                    return
                self.visited.add(expr)
                super().visit(expr)
                # Post-order: add after visiting children
                if isinstance(expr, relay.Call):
                    self.node_order.append(expr)
                    # Check if this is the converge op
                    if self._is_same_op(expr, self.converge_op):
                        self.converge_idx = len(self.node_order) - 1

            def _is_same_op(self, expr, target):
                """Check if expr matches target converge op structure."""
                if not isinstance(expr, relay.Call) or not isinstance(target, relay.Call):
                    return False
                if not isinstance(expr.op, tvm.ir.Op) or not isinstance(target.op, tvm.ir.Op):
                    return False
                return expr.op.name == target.op.name == "add"

        analyzer = GraphAnalyzer(converge_op, param_set)
        analyzer.visit(body)

        debug_print(f"[CompositeSplitter] Graph analysis:")
        debug_print(f"  Total nodes: {len(analyzer.node_order)}")
        debug_print(f"  Converge idx: {analyzer.converge_idx}")
        for i, node in enumerate(analyzer.node_order):
            op_name = node.op.name if isinstance(node.op, tvm.ir.Op) else "fn"
            marker = " <-- CONVERGE" if i == analyzer.converge_idx else ""
            debug_print(f"    [{i}] {op_name}{marker}")

        if analyzer.converge_idx < 0:
            debug_print(f"[CompositeSplitter] Converge op not found in traversal")
            return None

        # Step 3: Build patterns dynamically by traversing the graph
        def build_pattern_for_expr(expr, stop_at=None, pattern_cache=None):
            """
            Build dataflow pattern by traversing expression.
            Args:
                expr: Expression to build pattern for
                stop_at: If not None, stop traversal and return wildcard() when reaching this expr
                pattern_cache: Cache for already built patterns
            Returns:
                DFPattern for the expression
            """
            if pattern_cache is None:
                pattern_cache = {}

            if expr in pattern_cache:
                return pattern_cache[expr]

            # Stop condition
            if stop_at is not None and expr is stop_at:
                pattern = wildcard()
                pattern_cache[expr] = pattern
                return pattern

            if isinstance(expr, relay.Var):
                # Var -> wildcard
                pattern = wildcard()
                pattern_cache[expr] = pattern
                return pattern

            elif isinstance(expr, relay.Constant):
                # Constant -> is_constant
                pattern = is_constant()
                pattern_cache[expr] = pattern
                return pattern

            elif isinstance(expr, relay.Call):
                if isinstance(expr.op, tvm.ir.Op):
                    op_name = expr.op.name

                    # Build arg patterns recursively
                    arg_patterns = []
                    for arg in expr.args:
                        arg_pattern = build_pattern_for_expr(arg, stop_at, pattern_cache)
                        arg_patterns.append(arg_pattern)

                    pattern = is_op(op_name)(*arg_patterns)
                    pattern_cache[expr] = pattern
                    return pattern
                else:
                    # Function call - treat as wildcard
                    pattern = wildcard()
                    pattern_cache[expr] = pattern
                    return pattern

            elif isinstance(expr, relay.TupleGetItem):
                # TupleGetItem - build pattern for tuple and get item
                tuple_pattern = build_pattern_for_expr(expr.tuple_value, stop_at, pattern_cache)
                # For TupleGetItem, we need to match the whole thing
                # This is tricky with dataflow patterns, use wildcard for now
                pattern = wildcard()
                pattern_cache[expr] = pattern
                return pattern

            else:
                pattern = wildcard()
                pattern_cache[expr] = pattern
                return pattern

        # Step 4: Build post-converge pattern (from converge_op to output)
        # Post pattern: everything from converge_op to body output
        debug_print(f"[CompositeSplitter] Building post-converge pattern...")
        post_pattern = build_pattern_for_expr(body, stop_at=None)

        # Find where converge_op is in the pattern and replace everything before it with wildcard
        # Actually, we want: converge_op and everything after it
        # So we build pattern from body, but stop at converge_op's inputs (replace with wildcard)
        post_pattern_cache = {}
        for arg in converge_op.args:
            post_pattern_cache[arg] = wildcard()

        post_pattern = build_pattern_for_expr(body, stop_at=None, pattern_cache=post_pattern_cache)
        debug_print(f"[CompositeSplitter] Post-converge pattern built")

        # Step 5: Apply partition for post-converge
        post_composite_name = f"{composite_name_prefix}.post_converge"
        try:
            result_expr = post_pattern.partition(
                inlined_body,
                {"Composite": post_composite_name}
            )
            debug_print(f"[CompositeSplitter] Post-converge partition successful")
        except Exception as e:
            debug_print(f"[CompositeSplitter] Post-converge partition failed: {e}")
            return None

        # Step 6: Build pre-converge pattern for each branch that is a Call (not Var/Constant)
        pre_composite_name = f"{composite_name_prefix}.pre_converge"

        # Find the main branch (the one that has Call nodes, not just Var)
        for i, arg in enumerate(converge_op.args):
            if isinstance(arg, relay.Call):
                # This is a branch with computations - build pattern for it
                debug_print(f"[CompositeSplitter] Building pre-converge pattern for arg[{i}]...")

                # Build pattern for this branch (stop at function params)
                pre_pattern_cache = {}
                for param in param_set:
                    pre_pattern_cache[param] = wildcard()

                pre_pattern = build_pattern_for_expr(arg, stop_at=None, pattern_cache=pre_pattern_cache)

                try:
                    result_expr = pre_pattern.partition(
                        result_expr,
                        {"Composite": pre_composite_name}
                    )
                    debug_print(f"[CompositeSplitter] Pre-converge partition successful for arg[{i}]")
                except Exception as e:
                    debug_print(f"[CompositeSplitter] Pre-converge partition failed for arg[{i}]: {e}")
                    # Continue without pre-converge composite for this branch

        return {
            "result_expr": result_expr,
            "pre_composite_name": pre_composite_name,
            "post_composite_name": post_composite_name,
        }

    @staticmethod
    def create_pre_converge_composite(expr, branch_ops, composite_name_prefix="imcflow"):
        """
        Create a composite function for pre-converge operations using pattern matching.

        Args:
            expr: The expression to transform
            branch_ops: List of operations to include in the composite
            composite_name_prefix: Prefix for composite name

        Returns:
            Transformed expression with pre-converge composite
        """
        from tvm.relay.dataflow_pattern import wildcard, is_op, is_constant

        if not branch_ops:
            return expr

        # Build pattern based on the operations in the branch
        # Start with the first op and chain subsequent ops
        pattern = None

        for op_call in reversed(branch_ops):
            if not isinstance(op_call, relay.Call) or not isinstance(op_call.op, tvm.ir.Op):
                continue

            op_name = op_call.op.name

            if pattern is None:
                # First op - use wildcards for inputs
                if op_name in ["nn.imcflow_qconv", "nn.imcflow_qdwconv"]:
                    pattern = is_op(op_name)(wildcard(), is_constant(), is_constant())
                elif op_name == "nn.bias_add":
                    pattern = is_op(op_name)(wildcard(), is_constant())
                elif op_name == "nn.relu":
                    pattern = is_op(op_name)(wildcard())
                elif op_name == "add":
                    pattern = is_op(op_name)(wildcard(), wildcard())
                else:
                    pattern = is_op(op_name)(wildcard())
            else:
                # Chain with previous pattern
                if op_name == "nn.relu":
                    pattern = is_op(op_name)(pattern)
                elif op_name == "nn.bias_add":
                    pattern = is_op(op_name)(pattern, is_constant())
                elif op_name in ["qnn.imcflow_min_max_quantize"]:
                    pattern = is_op(op_name)(pattern, is_constant(), is_constant())

        if pattern is None:
            return expr

        pre_composite_name = f"{composite_name_prefix}.pre_converge"
        try:
            result_expr = pattern.partition(
                expr,
                {"Composite": pre_composite_name}
            )
            return result_expr
        except Exception as e:
            debug_print(f"[CompositeSplitter] Failed to create pre-converge composite: {e}")
            return expr


class AnnotGenerator:
    def __init__(self, handle_branch_from_var_converge=True):
      """
      Args:
          handle_branch_from_var_converge: If True, treat branch_from_var cases (converge points
              where diverge_node is None because one branch originates from Var inputs outside
              the function) as unbalanced branches and force new region creation.
              If False, these converge points are handled with normal selection policy.
              Default is True.
      """
      self.RegionList = []
      self.handle_branch_from_var_converge = handle_branch_from_var_converge

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
        def __init__(self, outer_self, target_mod, handle_branch_from_var_converge=True):
          self.RegionList = RegionList
          self.outer = outer_self
          self.target_mod = target_mod  # Store module for latency/throughput calculations
          self.handle_branch_from_var_converge = handle_branch_from_var_converge
          # Track most recently assigned region to attach nodes with no input regions
          self.last_assigned_region = None
          # Track composites that need to be split (deferred until after region assignment)
          # Format: {original_composite: {"converge_op": op, "pre_region": region, "post_region": region}}
          self.split_pending = {}
          # Track converge point summary for debugging
          # Format: [{"node": node, "diverge_node": diverge_node, "branches": [...], "is_unbalanced": bool, "action": str}]
          self.converge_point_summary = []

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
          """
          Calculate cost for a call node based on conv IC/OC dimensions.

          Cost calculation:
          - conv or conv-starting composite: ceil(IC/atom_IC) * ceil(OC/64)
            where atom_IC = floor(256/(KH*KW)) for regular conv, 32 for depthwise
          - composite without conv: cost = 1
          - other supported ops: cost = 1
          """
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

          def _get_conv_cost(conv_call):
            """Calculate cost for a conv op based on IC/OC dimensions."""
            try:
              # Get kernel shape for KH, KW
              weight = conv_call.args[1]
              if hasattr(weight, 'checked_type') and weight.checked_type is not None:
                weight_shape = [int(d) for d in weight.checked_type.shape]
              elif isinstance(weight, relay.Constant):
                weight_shape = list(weight.data.shape)
              else:
                return 1  # Fallback to cost 1 if shape unknown

              # Get input shape for IC
              input_arg = conv_call.args[0]
              if hasattr(input_arg, 'checked_type') and input_arg.checked_type is not None:
                input_shape = [int(d) for d in input_arg.checked_type.shape]
                IC = input_shape[1]  # NCHW format
              else:
                return 1  # Fallback

              # Extract dimensions based on conv type
              if conv_call.op.name in ["nn.conv2d", "nn.imcflow_qconv"]:
                OC, _, KH, KW = weight_shape  # OIHW format
              elif conv_call.op.name in ["nn.imcflow_qdwconv"]:
                # Depthwise: weight shape is (IC, 1, KH, KW)
                _, _, KH, KW = weight_shape
                OC = IC  # Depthwise: OC == IC
              else:
                return 1

              # Check if depthwise
              groups = conv_call.attrs.groups if hasattr(conv_call.attrs, 'groups') else 1
              is_depthwise = (groups == IC) if groups > 1 else False

              # Calculate atom_IC
              if is_depthwise:
                atom_IC = 32
                atom_OC = 32
              else:
                atom_IC = math.floor(256 / (KH * KW)) if (KH * KW) > 0 else 32
                atom_OC = 64

              # Calculate cost
              cost = math.ceil(IC / atom_IC) * math.ceil(OC / atom_OC)
              return max(1, cost)
            except Exception as e:
              debug_print(f"[getCost] Error calculating conv cost: {e}")
              return 1

          def _find_conv_in_composite(composite_func):
            """Find conv call in composite function body."""
            class ConvFinder(tvm.relay.ExprVisitor):
              def __init__(self):
                super().__init__()
                self.conv_call = None

              def visit_call(self, call):
                if isinstance(call.op, tvm.ir.Op):
                  if call.op.name in ["nn.conv2d", "nn.imcflow_qconv", "nn.imcflow_qdwconv"]:
                    self.conv_call = call
                super().visit_call(call)

            finder = ConvFinder()
            finder.visit(composite_func.body)
            return finder.conv_call

          if IsNoCostCall:
            return 0

          # Handle composite functions
          if IsComposite:
            composite_func = call.op
            conv_call = _find_conv_in_composite(composite_func)
            if conv_call is not None:
              return _get_conv_cost(conv_call)
            else:
              # Composite without conv
              return 1

          # Handle supported ops directly
          if IsSupportedOp:
            if call.op.name in ["nn.conv2d", "nn.imcflow_qconv", "nn.imcflow_qdwconv"]:
              return _get_conv_cost(call)
            else:
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

        def _is_converge_point(self, node, rev_edges):
          """Check if node is a converge point (2+ inputs from common diverge point)."""
          preds = rev_edges.get(node, [])
          call_preds = [p for p in preds if isinstance(p, (Call, TupleGetItem))]

          debug_print(f"\n{'='*60}")
          debug_print(f"[_is_converge_point] Checking node:")
          debug_print(f"  ID: {getNodeDebugID(node)}")
          debug_print(f"  Expr: {node}")
          debug_print(f"  Total preds: {len(preds)}")
          debug_print(f"  Call/TupleGetItem preds: {len(call_preds)}")
          for i, pred in enumerate(call_preds):
            debug_print(f"    Pred {i}: {getNodeDebugID(pred)}")
            debug_print(f"      Expr: {pred}")
          debug_print(f"  Result: {len(call_preds) >= 2}")
          debug_print(f"{'='*60}\n")

          return len(call_preds) >= 2

        def _branches_unbalanced(self, branch_metrics):
          """Check if branches have different latency or throughput (threshold=0)."""
          if len(branch_metrics) < 2:
            return False
          lats = [m[0] for m in branch_metrics]
          thrs = [m[1] for m in branch_metrics]
          return len(set(lats)) > 1 or len(set(thrs)) > 1

        def _get_branch_last_nodes_regions(self, node, rev_edges):
          """Get the regions of the last nodes in each branch before converge point."""
          preds = rev_edges.get(node, [])
          call_preds = [p for p in preds if isinstance(p, (Call, TupleGetItem))]
          regions = []
          for pred in call_preds:
            region = self.getRegion(pred)
            if region is not None:
              regions.append(region)
          return regions

        def _record_composite_split(self, node, candidate_regions):
          """
          Record split_pending info for a composite at a converge point.

          Returns:
            needs_split: True if composite was recorded for split, False otherwise
          """
          converge_op = CompositeSplitter.find_converge_op_in_composite(node)
          if converge_op is None:
            debug_print(f"[ConvergeCheck] No converge op found in composite")
            return False

          # Determine pre_region: use candidate_region if available, else create new
          if len(candidate_regions) > 0:
            pre_region = candidate_regions[0]
          else:
            pre_region = self.createRegion()
          # post_region: always create new region for the converge part
          post_region = self.createRegion()

          # Record split info for deferred processing
          self.split_pending[node] = {
            "converge_op": converge_op,
            "pre_region": pre_region,
            "post_region": post_region,
            "pre_composite": None,   # Will be filled after mutation
            "post_composite": None,  # Will be filled after mutation
          }
          debug_print(f"[ConvergeCheck] Recorded split_pending for {getNodeDebugID(node)}")
          return True

        def _record_converge_summary(self, node, diverge_node, converge_type, branch_info_list, is_unbalanced, action):
          """Record converge point info to summary for debugging."""
          self.converge_point_summary.append({
            "node": getNodeDebugID(node),
            "diverge_node": getNodeDebugID(diverge_node) if diverge_node else None,
            "type": converge_type,
            "branches": branch_info_list,
            "is_unbalanced": is_unbalanced,
            "action": action
          })

        def _handle_unbalanced_converge(self, node, rev_edges, IsComposite, candidate_regions,
                                        converge_type, diverge_node, branch_info_list):
          """
          Handle unbalanced converge point - check for deadlock risk and record split if needed.

          Args:
            node: The converge point node
            rev_edges: Reverse edges from graph
            IsComposite: Whether node is a composite
            candidate_regions: Available regions for assignment
            converge_type: "diverge-converge" or "branch_from_var"
            diverge_node: The diverge point node (None for branch_from_var)
            branch_info_list: List of branch info dicts for summary

          Returns:
            (force_new_region, needs_split): Tuple of booleans
          """
          force_new_region = False
          needs_split = False

          debug_print(f"[ConvergeCheck] Branches are UNBALANCED - checking region assignment")

          # Check if input branches are in the same region (deadlock risk)
          branch_regions = self._get_branch_last_nodes_regions(node, rev_edges)
          debug_print(f"[ConvergeCheck] Branch regions count: {len(branch_regions)}")
          for i, reg in enumerate(branch_regions):
            region_idx = self.RegionList.index(reg) if reg in self.RegionList else -1
            debug_print(f"    Branch {i} region idx: {region_idx}, size: {len(reg)}")
          unique_regions = list({id(r): r for r in branch_regions}.values())
          debug_print(f"[ConvergeCheck] Unique regions: {len(unique_regions)}")

          if len(unique_regions) == 1:
            # All branches in same region -> deadlock risk
            debug_print(f"[ConvergeCheck] All branches in same region - DEADLOCK RISK")

            if IsComposite:
              debug_print(f"[ConvergeCheck] Recording composite for deferred split")
              needs_split = self._record_composite_split(node, candidate_regions)
              if needs_split:
                action = "split_composite (deadlock risk)"
              else:
                force_new_region = True
                action = "force_new_region (deadlock risk, no internal converge op)"
            else:
              force_new_region = True
              action = "force_new_region (deadlock risk)"

            self._record_converge_summary(node, diverge_node, converge_type,
                                          branch_info_list, True, action)
          else:
            debug_print(f"[ConvergeCheck] Branches in different regions - no deadlock risk")
            debug_print(f"{'#'*70}\n")
            self._record_converge_summary(node, diverge_node, converge_type,
                                          branch_info_list, True, "no_action (different regions)")

          return force_new_region, needs_split

        def _extract_branches_for_converge(self, node, rev_edges, branch_analyzer, diverge_node=None):
          """
          Extract branches leading to a converge point.

          Args:
            node: The converge point node
            rev_edges: Reverse edges from graph
            branch_analyzer: BranchAnalyzer instance
            diverge_node: If provided, extract branches between diverge and converge.
                          If None, extract all ancestor ops for each predecessor.

          Returns:
            branches: List of branch ops lists
            from_var_flags: List of booleans indicating if each branch originates from Var/Const
          """
          preds = rev_edges.get(node, [])
          call_preds = [p for p in preds if isinstance(p, (Call, TupleGetItem))]

          if diverge_node is not None:
            # Use existing extract_branches for diverge-converge case
            branches = branch_analyzer.extract_branches(diverge_node, node)
            from_var_flags = [False] * len(branches)
          else:
            # Trace back from each predecessor to collect all ancestor ops
            branches = []
            from_var_flags = []
            for pred in call_preds:
              branch_ops, is_from_var = self._trace_branch_to_input(pred, branch_analyzer)
              branches.append(branch_ops)
              from_var_flags.append(is_from_var)

          return branches, from_var_flags

        def _trace_branch_to_input(self, pred, branch_analyzer, max_depth=50):
          """
          Trace back from pred to collect all ancestor ops until hitting Var/Const.

          Returns:
            (branch_ops, is_from_var): List of ops and whether branch has no conv
          """
          visited = set()
          queue = [pred]
          branch_ops = []
          has_conv = False
          depth = 0

          while queue and depth < max_depth:
            curr = queue.pop(0)
            if curr in visited:
              continue
            visited.add(curr)
            depth += 1

            if isinstance(curr, Call):
              # Flatten composite ops
              flattened = branch_analyzer.flatten_composite_ops(curr)
              branch_ops.extend(flattened)

              # Check for conv
              if isinstance(curr.op, tvm.ir.Op):
                if curr.op.name in ["nn.imcflow_qconv", "nn.imcflow_qdwconv", "nn.conv2d"]:
                  has_conv = True

              # Continue to predecessors
              for arg in curr.args:
                if isinstance(arg, (Call, TupleGetItem)):
                  queue.append(arg)

          return branch_ops, not has_conv  # is_from_var = not has_conv

        def _calculate_branch_metrics(self, branches, lat_calc):
          """
          Calculate latency and throughput for each branch.

          Args:
            branches: List of branch ops lists
            lat_calc: LatencyThroughputCalculator instance

          Returns:
            List of (latency, throughput) tuples
          """
          branch_metrics = []
          for i, branch_ops in enumerate(branches):
            debug_print(f"\n  --- Branch {i} details ---")
            for j, op in enumerate(branch_ops):
              op_lat = lat_calc.get_op_latency(op)
              op_thr = lat_calc.get_op_throughput(op)
              op_name = op.op.name if isinstance(op, relay.Call) and isinstance(op.op, tvm.ir.Op) else getNodeDebugID(op)
              debug_print(f"    Op {j}: {op_name}, lat={op_lat}, thr={op_thr}")
            lat = lat_calc.calculate_branch_latency(branch_ops)
            thr = lat_calc.calculate_branch_throughput(branch_ops)
            branch_metrics.append((lat, thr))
            debug_print(f"  Branch {i} TOTAL: latency={lat}, throughput={thr}, ops_count={len(branch_ops)}")
          return branch_metrics

        def _build_branch_info_list(self, branches, branch_metrics, from_var_flags=None):
          """
          Build branch info list for converge point summary.

          Args:
            branches: List of branch ops lists
            branch_metrics: List of (latency, throughput) tuples
            from_var_flags: Optional list of from_var flags (None = all False)

          Returns:
            List of branch info dicts
          """
          branch_info_list = []
          for i, branch_ops in enumerate(branches):
            ops_info = []
            for op in branch_ops:
              op_name = op.op.name if isinstance(op, relay.Call) and isinstance(op.op, tvm.ir.Op) else getNodeDebugID(op)
              ops_info.append(op_name)
            lat, thr = branch_metrics[i]
            info = {
              "ops": ops_info,
              "latency": lat,
              "throughput": thr,
              "ops_count": len(branch_ops),
              "is_branch_from_var": from_var_flags[i] if from_var_flags else False
            }
            branch_info_list.append(info)
          return branch_info_list

        def _handle_converge_point(self, node, rev_edges, branch_analyzer, lat_calc,
                                   IsComposite, candidate_regions):
          """
          Handle converge point detection and processing.

          Returns:
            (force_new_region, needs_split): Tuple of booleans
          """
          force_new_region = False
          needs_split = False

          debug_print(f"\n{'#'*70}")
          debug_print(f"[ConvergeCheck] CONVERGE POINT DETECTED")
          debug_print(f"{'#'*70}")
          debug_print(f"  Node ID: {getNodeDebugID(node)}")
          debug_print(f"  Node Expr:\n{node}")
          debug_print(f"{'#'*70}")

          # Find diverge point
          diverge_node = branch_analyzer.find_diverge_point(node)

          if diverge_node is not None:
            debug_print(f"\n[ConvergeCheck] Diverge point found: {getNodeDebugID(diverge_node)}")
            debug_print(f"  Diverge Expr:\n{diverge_node}")
            converge_type = "diverge-converge"
          else:
            debug_print(f"[ConvergeCheck] No diverge point found for this converge point")
            debug_print(f"[ConvergeCheck] handle_branch_from_var_converge={self.handle_branch_from_var_converge}")

            if not self.handle_branch_from_var_converge:
              debug_print(f"[ConvergeCheck] Branch from var handling DISABLED - using normal selection")
              debug_print(f"{'#'*70}\n")
              return force_new_region, needs_split
            converge_type = "branch_from_var"

          # Extract branches (unified for both cases)
          branches, from_var_flags = self._extract_branches_for_converge(
            node, rev_edges, branch_analyzer, diverge_node
          )
          debug_print(f"\n[ConvergeCheck] Extracted {len(branches)} branches")

          # For branch_from_var, check if at least one branch actually originates from Var
          if diverge_node is None:
            has_branch_from_var = any(from_var_flags)
            debug_print(f"[ConvergeCheck] Has at least one branch_from_var: {has_branch_from_var}")
            for i, flag in enumerate(from_var_flags):
              debug_print(f"  Branch {i}: is_branch_from_var={flag}")

            if not has_branch_from_var:
              debug_print(f"[ConvergeCheck] No branch_from_var found - skipping special handling")
              debug_print(f"{'#'*70}\n")
              return force_new_region, needs_split

          # Calculate branch metrics (unified)
          branch_metrics = self._calculate_branch_metrics(branches, lat_calc)

          debug_print(f"\n[ConvergeCheck] Branch metrics summary:")
          for i, (lat, thr) in enumerate(branch_metrics):
            debug_print(f"    Branch {i}: latency={lat}, throughput={thr}")

          # Check if branches are unbalanced
          is_unbalanced = self._branches_unbalanced(branch_metrics)
          debug_print(f"\n[ConvergeCheck] Branches unbalanced? {is_unbalanced}")

          # Build branch info list (unified with from_var field)
          branch_info_list = self._build_branch_info_list(branches, branch_metrics, from_var_flags)

          if not is_unbalanced:
            debug_print(f"[ConvergeCheck] Branches are BALANCED - no special handling needed")
            debug_print(f"{'#'*70}\n")
            self._record_converge_summary(node, diverge_node, converge_type,
                                          branch_info_list, False, "no_action (balanced)")
          else:
            force_new_region, needs_split = self._handle_unbalanced_converge(
              node, rev_edges, IsComposite, candidate_regions,
              converge_type, diverge_node, branch_info_list
            )

          return force_new_region, needs_split

        def run(self, fn):
          order, edges, rev_edges = self._topo_bfs_order(fn)

          # Initialize branch analyzer for converge point detection
          branch_analyzer = BranchAnalyzer(edges, rev_edges)
          lat_calc = LatencyThroughputCalculator(self.target_mod)

          debug_print(f"\n{'='*70}")
          debug_print(f"[AnnotatorBFS] Starting BFS traversal with {len(order)} nodes")
          debug_print(f"{'='*70}\n")

          for node_idx, node in enumerate(order):
            if isinstance(node, Call):
              IsComposite = self.isComposite(node)
              IsSupportedOp = self.isSupportedOp(node)
              IsSuperNode = self.isSuperNode(node)

              debug_print(f"\n{'~'*50}")
              debug_print(f"[AnnotatorBFS] Processing node {node_idx}/{len(order)}:")
              debug_print(f"  ID: {getNodeDebugID(node)}")
              debug_print(f"  IsComposite: {IsComposite}")
              debug_print(f"  IsSupportedOp: {IsSupportedOp}")
              debug_print(f"  IsSuperNode: {IsSuperNode}")

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

                # ====================================================
                # Converge point check: detect branch latency mismatch
                # ====================================================
                force_new_region = False
                needs_split = False

                if self._is_converge_point(node, rev_edges):
                  force_new_region, needs_split = self._handle_converge_point(
                    node, rev_edges, branch_analyzer, lat_calc,
                    IsComposite, candidate_regions
                  )

                # ====================================================
                # Selection policy with converge point handling
                # ====================================================
                if needs_split:
                  # Composite will be split - add original to post_region for now
                  # After mutation, this will be replaced with pre_composite
                  # post_composite will be added to post_region
                  post_region = self.split_pending[node]["post_region"]
                  self.addToRegion(post_region, node)
                  debug_print(f"[ConvergeCheck] Added original composite to post_region (will be split later)")
                elif force_new_region:
                  # Non-composite converge point with unbalanced branches -> new region
                  Region = self.createRegion()
                  self.addToRegion(Region, node)
                  debug_print(f"[ConvergeCheck] Created new region for converge point (region {self.RegionList.index(Region)})")
                else:
                  # Normal selection policy
                  uniq = list({id(r): r for r in candidate_regions}.values())
                  debug_print(f"  [Selection] candidate_regions: {len(candidate_regions)}, unique: {len(uniq)}")
                  if len(uniq) == 1:
                    Region = uniq[0]
                    region_idx = self.RegionList.index(Region) if Region in self.RegionList else -1
                    self.addToRegion(Region, node)
                    debug_print(f"  [Selection] Assigned to existing region {region_idx}")
                  elif len(uniq) > 1:
                    Region = uniq[0]
                    region_idx = self.RegionList.index(Region) if Region in self.RegionList else -1
                    # Region = self.createRegion()
                    self.addToRegion(Region, node)
                    debug_print(f"  [Selection] Multiple regions, assigned to region {region_idx}")
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
                      debug_print(f"  [Selection] Created new region (no input regions)")
                    else:
                      region_idx = self.RegionList.index(Region) if Region in self.RegionList else -1
                      debug_print(f"  [Selection] Using last assigned region {region_idx}")
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

      annot = _AnnotatorBFS(self, mod, self.handle_branch_from_var_converge)
      annot.run(func)
      self.RegionList = RegionList
      self.split_pending = annot.split_pending

      # Print final region summary
      debug_print(f"\n{'='*70}")
      debug_print(f"[AnnotGenerator.createRegionBFS] FINAL REGION SUMMARY")
      debug_print(f"{'='*70}")
      debug_print(f"Total regions: {len(self.RegionList)}")
      for region_idx, region in enumerate(self.RegionList):
        debug_print(f"\n  Region {region_idx} ({len(region)} nodes):")
        for node in region:
          debug_print(f"    - {getNodeDebugID(node)}")

      # Print converge point summary
      debug_print(f"\n{'-'*70}")
      debug_print(f"CONVERGE POINT SUMMARY")
      debug_print(f"{'-'*70}")
      debug_print(f"Total converge points detected: {len(annot.converge_point_summary)}")
      for cp_idx, cp_info in enumerate(annot.converge_point_summary):
        debug_print(f"\n  Converge Point {cp_idx}:")
        debug_print(f"    Node: {cp_info['node']}")
        debug_print(f"    Type: {cp_info['type']}")
        if cp_info['diverge_node']:
          debug_print(f"    Diverge Node: {cp_info['diverge_node']}")
        debug_print(f"    Is Unbalanced: {cp_info['is_unbalanced']}")
        debug_print(f"    Action: {cp_info['action']}")
        debug_print(f"    Branches ({len(cp_info['branches'])}):")
        for br_idx, br_info in enumerate(cp_info['branches']):
          if cp_info['type'] == 'branch_from_var':
            from_var = br_info.get('is_branch_from_var', 'N/A')
            debug_print(f"      Branch {br_idx}: latency={br_info['latency']}, throughput={br_info['throughput']}, from_var={from_var}")
            debug_print(f"        Ops: {br_info['ops']}")
          else:
            debug_print(f"      Branch {br_idx}: latency={br_info['latency']}, throughput={br_info['throughput']}, ops_count={br_info.get('ops_count', len(br_info['ops']))}")
            debug_print(f"        Ops: {br_info['ops']}")

      debug_print(f"\nSplit pending: {len(self.split_pending)} composites")
      debug_print(f"{'='*70}\n")

      # Return func as well to ensure the same Python object is used in _apply_composite_splits
      return self.RegionList, self.split_pending, func


class CompositeGraphMutator(relay.ExprMutator):
    """
    Mutator that replaces original composites with split composites.

    After region assignment, this mutator is used to:
    1. Split composites at converge points
    2. Update the graph with new composite calls
    """

    def __init__(self, split_pending):
        super().__init__()
        self.split_pending = split_pending
        # Maps original composite -> split info
        self.split_results = {}
        # Maps old_node -> new_node for nodes whose args changed during mutation
        self.node_replacements = {}

    def visit_call(self, call):
        # First visit args
        new_args = [self.visit(arg) for arg in call.args]

        if call in self.split_pending:
          split_info = self.split_pending[call]
          debug_print(f"[CompositeGraphMutator] Splitting composite: {getNodeDebugID(call)}")

          # Create call with updated args before splitting
          # This ensures that if any args were mutated (e.g., another composite was split),
          # we split the correct version with updated dependencies
          call_to_split = call
          if new_args != list(call.args):
            call_to_split = relay.Call(call.op, new_args, call.attrs, call.type_args, call.span)

          # Perform actual split using CompositeSplitter
          converge_op = split_info["converge_op"]
          result = CompositeSplitter.split_composite_at_converge(
            call_to_split, converge_op, "imcflow"
          )

          if result is not None:
            # Result is now a dict with result_expr, pre_composite_name, post_composite_name
            new_expr = result["result_expr"]
            pre_composite_name = result["pre_composite_name"]
            post_composite_name = result["post_composite_name"]

            debug_print(f"[CompositeGraphMutator] Split successful: pre={pre_composite_name}, post={post_composite_name}")

            # Extract the IMMEDIATE pre and post composite calls from this split's result
            # The structure of new_expr is: Let(bindings, post_call(pre_call(...), other_branch))
            # We only want the composites created by THIS split, not nested ones from previous splits
            pre_calls_from_split = []
            post_calls_from_split = []

            def get_composite_info(call_node, let_bindings):
                """Get composite name if call_node is a composite call."""
                func = call_node.op
                if isinstance(func, relay.Var) and func in let_bindings:
                    func = let_bindings[func]
                if isinstance(func, relay.Function) and func.attrs and "Composite" in func.attrs:
                    return str(func.attrs["Composite"])
                return None

            def extract_immediate_composites(expr, let_bindings=None):
                """Extract only the immediate pre and post composites, not nested ones."""
                if let_bindings is None:
                    let_bindings = {}

                # Handle Let expressions - collect bindings
                if isinstance(expr, relay.Let):
                    if isinstance(expr.value, relay.Function):
                        let_bindings[expr.var] = expr.value
                    return extract_immediate_composites(expr.body, let_bindings)

                # Handle Call expressions - this should be the post_converge call
                if isinstance(expr, relay.Call):
                    comp_name = get_composite_info(expr, let_bindings)
                    if comp_name == post_composite_name:
                        post_calls_from_split.append(expr)
                        # Now look for pre_converge in the IMMEDIATE args (not recursively)
                        for arg in expr.args:
                            # Traverse through Lets to find the Call
                            inner_expr = arg
                            inner_bindings = dict(let_bindings)
                            while isinstance(inner_expr, relay.Let):
                                if isinstance(inner_expr.value, relay.Function):
                                    inner_bindings[inner_expr.var] = inner_expr.value
                                inner_expr = inner_expr.body
                            if isinstance(inner_expr, relay.Call):
                                arg_comp = get_composite_info(inner_expr, inner_bindings)
                                if arg_comp == pre_composite_name:
                                    pre_calls_from_split.append(inner_expr)

            extract_immediate_composites(new_expr)
            debug_print(f"[CompositeGraphMutator] Extracted from split: {len(pre_calls_from_split)} pre, {len(post_calls_from_split)} post")

            # Store split result for region update
            self.split_results[call] = {
              "new_expr": new_expr,
              "split_info": split_info,
              "pre_composite_name": pre_composite_name,
              "post_composite_name": post_composite_name,
              "pre_calls": pre_calls_from_split,
              "post_calls": post_calls_from_split,
            }
            return new_expr
          else:
            debug_print(f"[CompositeGraphMutator] Split failed, keeping original")
            # Fall through to normal processing

        # Normal call - just update args if changed
        if new_args != list(call.args):
          new_call = relay.Call(call.op, new_args, call.attrs, call.type_args, call.span)
          # Track the replacement so RegionList can be updated
          self.node_replacements[call] = new_call
          return new_call
        return call


def _find_composites_by_name(expr, target_names):
    """
    Find composite function calls in an expression by their Composite attribute name.
    Handles both direct function calls and let-bound function calls.

    Args:
        expr: The expression to search
        target_names: Set of composite names to find

    Returns:
        Dict mapping composite name to list of Call nodes
    """
    found = {name: [] for name in target_names}

    class CompositeFinder(relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            # Track let-bound functions: Var -> Function
            self.let_bindings = {}

        def visit_let(self, let):
            # Track let bindings where value is a Function
            if isinstance(let.value, relay.Function):
                self.let_bindings[let.var] = let.value
            self.visit(let.value)
            self.visit(let.body)

        def visit_call(self, call):
            # Get the actual function (either direct or let-bound)
            func = call.op
            if isinstance(func, relay.Var) and func in self.let_bindings:
                func = self.let_bindings[func]

            if isinstance(func, relay.Function):
                attrs = func.attrs
                if attrs and "Composite" in attrs:
                    comp_name = str(attrs["Composite"])
                    if comp_name in target_names:
                        found[comp_name].append(call)
                        debug_print(f"[_find_composites_by_name] Found composite '{comp_name}': {getNodeDebugID(call)}")
            super().visit_call(call)

    CompositeFinder().visit(expr)

    # Debug: report what was found
    for name in target_names:
        debug_print(f"[_find_composites_by_name] Target '{name}': found {len(found[name])} calls")

    return found


def _apply_composite_splits(target_mod, split_pending, RegionList, main_func):
    """
    Apply composite splits and update region list.

    Args:
        target_mod: The module to transform
        split_pending: Dict of composites to split
        RegionList: List of regions to update
        main_func: The function object from createRegionBFS (must be the same Python object)

    Returns:
        (transformed_mod, updated_RegionList)
    """
    if not split_pending:
        return target_mod, RegionList

    debug_print(f"[_apply_composite_splits] Applying {len(split_pending)} composite splits")

    # Apply mutation
    mutator = CompositeGraphMutator(split_pending)
    new_body = mutator.visit(main_func.body)

    # Create new function with mutated body
    new_func = relay.Function(
        main_func.params,
        new_body,
        main_func.ret_type,
        main_func.type_params,
        main_func.attrs
    )

    # Update module
    new_mod = tvm.IRModule.from_expr(new_func)

    # Update RegionList: replace original composites with split results
    # Use the pre_calls and post_calls extracted during mutation (specific to each split)
    for original_composite, result_info in mutator.split_results.items():
        split_info = result_info["split_info"]
        pre_region = split_info["pre_region"]
        post_region = split_info["post_region"]

        pre_region_idx = RegionList.index(pre_region) if pre_region in RegionList else -1
        post_region_idx = RegionList.index(post_region) if post_region in RegionList else -1
        debug_print(f"[_apply_composite_splits] Processing split for {getNodeDebugID(original_composite)}")
        debug_print(f"  pre_region_idx={pre_region_idx}, post_region_idx={post_region_idx}")

        # Remove original composite from its region
        removed_from_region = -1
        for region_idx, region in enumerate(RegionList):
            if original_composite in region:
                region.remove(original_composite)
                removed_from_region = region_idx
                break
        debug_print(f"  Removed original from region {removed_from_region}")

        # Add new composites to their respective regions using the extracted calls
        # These are specific to THIS split, not all composites with the same name
        pre_calls = result_info.get("pre_calls", [])
        post_calls = result_info.get("post_calls", [])

        debug_print(f"  pre_composite: {len(pre_calls)} calls from this split")
        for pre_call in pre_calls:
            if pre_call not in pre_region:
                pre_region.append(pre_call)
                debug_print(f"    Added pre_call {getNodeDebugID(pre_call)} to pre_region")

        debug_print(f"  post_composite: {len(post_calls)} calls from this split")
        for post_call in post_calls:
            if post_call not in post_region:
                post_region.append(post_call)
                debug_print(f"    Added post_call {getNodeDebugID(post_call)} to post_region")

    # Update RegionList to use NEW nodes from the mutated graph
    # The mutation may have created new Python objects for nodes whose args changed
    # We need to map OLD nodes to NEW nodes using structural equality
    def _collect_all_calls(expr):
        """Collect all Call nodes from an expression"""
        calls = []
        class CallCollector(relay.ExprVisitor):
            def visit_call(self, call):
                calls.append(call)
                super().visit_call(call)
        CallCollector().visit(expr)
        return calls

    # Build updated RegionList using the node_replacements mapping from mutation
    # This is safe because we track exactly which old node maps to which new node
    updated_RegionList = []
    for region in RegionList:
        updated_region = []
        for old_node in region:
            # Check if this node was replaced during mutation
            if old_node in mutator.node_replacements:
                new_node = mutator.node_replacements[old_node]
                if new_node not in updated_region:
                    updated_region.append(new_node)
                debug_print(f"[_apply_composite_splits] Replaced: {getNodeDebugID(old_node)} -> {getNodeDebugID(new_node)}")
            else:
                # Node was not mutated, keep as is
                if old_node not in updated_region:
                    updated_region.append(old_node)
        updated_RegionList.append(updated_region)

    # Debug: print updated region summary
    debug_print(f"\n[_apply_composite_splits] UPDATED REGION SUMMARY")
    debug_print(f"Total regions: {len(updated_RegionList)}")
    for region_idx, region in enumerate(updated_RegionList):
        debug_print(f"  Region {region_idx} ({len(region)} nodes):")
        for node in region:
            debug_print(f"    - {getNodeDebugID(node)}")

    return new_mod, updated_RegionList


def partitionRound(mod, handle_branch_from_var_converge=True):
  """
  Partition IMCFlow functions into regions for hardware mapping.

  Args:
      mod: The TVM IRModule to partition
      handle_branch_from_var_converge: If True, treat branch_from_var cases (converge points
          where diverge_node is None because one branch originates from Var inputs outside
          the function) as unbalanced branches and force new region creation.
          If False, these converge points are handled with normal selection policy.
          Default is True.

  Returns:
      Partitioned IRModule
  """
  for global_var, func in mod.functions.items():
    if isinstance(func, relay.Function) and "Compiler" in func.attrs and re.match(r"imcflow.*", func.attrs["Compiler"]):
      name = global_var.name_hint
      func_attr = func.attrs
      annotator = AnnotGenerator(handle_branch_from_var_converge=handle_branch_from_var_converge)
      target_mod = tvm.IRModule.from_expr(relay.Function(func.params, func.body, ret_type=func.ret_type))

      # Step 1: Create regions (with split_pending for deferred splits)
      RegionList, split_pending, region_func = annotator.createRegionBFS(target_mod)

      # Step 2: Apply composite splits if any
      if split_pending:
        debug_print(f"[partitionRound] Applying {len(split_pending)} composite splits")
        target_mod, RegionList = _apply_composite_splits(target_mod, split_pending, RegionList, region_func)

      # Step 3: Annotate and partition with updated graph and regions
      target_mod = imcflow.ImcflowAnnotationPass(RegionList, f"{name}_round_")(target_mod)
      # printModel("resnet8_evl", target_mod, {}, f"{name}_round_annotated")
      target_mod = transform.MergeCompilerRegions()(target_mod)
      target_mod = imcflow.ImcflowCleanRegionTag()(target_mod)
      # printModel("resnet8_evl", target_mod, {}, f"{name}_round_merged")
      target_mod = transform.PartitionGraph()(target_mod)
      # printModel("resnet8_evl", target_mod, {}, f"{name}_round_partitioned")

      for new_gv, new_func in target_mod.functions.items():
        if new_gv.name_hint == "main":
          new_func = new_func.with_attr({k:v for k,v in func_attr.items()})
          mod[global_var] = new_func
        else:
          mod[new_gv] = new_func

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

def legalizeImcflowLayout(mod):
  layout_legalizer = ImcflowLayoutLegalizer()
  eval_mod, ttype_map = layout_legalizer.transform_mod(mod)
  return eval_mod, ttype_map

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
      if isinstance(SrcGraphNodeIDs, (int, tuple)):
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
        DstGraphNodeID = self.getCustomID(fn)

        if isinstance(fn.body, relay.Tuple):
          for idx, field in enumerate(fn.body.fields):
            InputGraphNodeID = self.getCustomID(field)
            self.appendToTensorEdgeList(InputGraphNodeID, DstGraphNodeID, "odata", f"func_out{idx}", idx)
        else:
          InputGraphNodeID = self.getCustomID(fn.body)
          self.appendToTensorEdgeList(InputGraphNodeID, DstGraphNodeID, "odata", "func_out0")

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
              if isinstance(SrcGraphNode, Tuple):
                for field in SrcGraphNode.fields:
                  _processInputNode(field, SrcTag, DstGraphNodeID, DstTag, self.getInputGraphNodeSplitIndex(field))
                return True

              if isinstance(SrcGraphNode, TupleGetItem):
                _processInputNode(SrcGraphNode.tuple_value, SrcTag, DstGraphNodeID, DstTag, SplitIdx)
                return True

              if isinstance(SrcGraphNode, Var):
                self.VarProperties[SrcGraphNode]["src_tag"] = SrcTag
                # self.VarProperties[SrcGraphNode]["src_tag"] = "var"
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
          # elif call.op == op.get("imcflow_packing"):python_dbg main.py -p random -m super_big_conv  2>&1 | tee main.log
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
  HwMapping = ImcflowDeviceConfig().HWNodeMap
  for func_name_var, func in mod.functions.items():
    if func_name_var.name_hint == "main": continue
    elif func.attrs["Compiler"]=="imcflow":
      GraphNodeIDs = CustomIDInFunc()[func_name_var.name_hint]
      ActiveIMCEs = set()
      for GraphNodeID in GraphNodeIDs:
        if GraphNodeID in HwMapping and isinstance(HwMapping[GraphNodeID], NodeID) and HwMapping[GraphNodeID].is_imce():
          ActiveIMCEs.add(HwMapping[GraphNodeID])
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

        src_hwnode = HwMapping[getInnerNodeID(SrcTensorID.graph_node_id)]
        if isinstance(HwMapping[getInnerNodeID(DstTensorID.graph_node_id)], tuple):
          # TODO: Note that the split idx is just to recover from hwnode tuple (see constructTensorEdgeList)
          dst_hwnode = HwMapping[getInnerNodeID(DstTensorID.graph_node_id)][SplitIdx]
          split_idx = None
        else:
           dst_hwnode = HwMapping[getInnerNodeID(DstTensorID.graph_node_id)]
           split_idx = SplitIdx

        NocPaths[func_name_var.name_hint][tensor_edge] = (src_hwnode, dst_hwnode, split_idx)

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
    def run_(self, mod, func, func_name, ttype_map):
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
          split_or_tuple_idx = edge.split_idx

          #dst id
          # if getInnerNodeID(edge.dst_id.graph_node_id) in self.hwnodemap:
          if ImcflowDeviceConfig().is_in_hw_node(edge.dst_id.graph_node_id):
            # dst_hw_node_id = self.hwnodemap[getInnerNodeID(edge.dst_id.graph_node_id)]
            dst_hw_node_id = ImcflowDeviceConfig().get_hw_node(edge.dst_id.graph_node_id, split_or_tuple_idx)
            # determine whether inode is included in the edge and which id it is.
            if isinstance(dst_hw_node_id, tuple):
              # use first tuple element to determine
              _dst_hw_node_id = dst_hw_node_id[0]
            else:
              _dst_hw_node_id = dst_hw_node_id

            if _dst_hw_node_id.name.startswith("inode"):
              is_inode = True
              inode_tensorid = edge.dst_id

          #src id
          if ImcflowDeviceConfig().is_in_hw_node(edge.src_id.graph_node_id):
            src_hw_node_id = ImcflowDeviceConfig().get_hw_node(edge.src_id.graph_node_id)
            # determine whether inode is included in the edge and which id it is.
            if src_hw_node_id.name.startswith("inode"):
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

        def _trace_all_paths_to_inputs(self, expr, param_map=None):
          """
          Trace all paths from output expr to input Vars, collecting conv operation parameters
          for each path separately.

          Args:
              expr: relay expression to trace
              param_map: dict mapping inner function params to outer args (for composite functions)

          Returns:
              dict: {var_name: conv_params_list} where conv_params_list is the list of
                    (k, s, p_top, p_bottom) tuples for convs on the path from that input to output
          """
          paths = {}  # var_name -> list of conv_params on path to this var
          if param_map is None:
            param_map = {}

          def is_qconv(node):
            if isinstance(node, relay.Call) and isinstance(node.op, tvm.ir.Op):
              return node.op == op.get("nn.imcflow_qconv") or node.op == op.get("nn.imcflow_qdwconv")
            return False

          def extract_conv_params(node):
            k = node.attrs['kernel_size'][0].value
            s = node.attrs['strides'][0].value
            p = node.attrs['padding']
            p_top = p[0].value if hasattr(p[0], 'value') else int(p[0])
            p_bottom = p[2].value if hasattr(p[2], 'value') else int(p[2])
            return (k, s, p_top, p_bottom)

          def trace_path(node, current_conv_params, local_param_map):
            """Recursively trace paths, accumulating conv params"""
            if isinstance(node, relay.Var):
              # Check if this var is mapped to an outer argument
              if node in local_param_map:
                # Continue tracing through the outer argument
                trace_path(local_param_map[node], current_conv_params, local_param_map)
              else:
                # Reached an input variable
                var_name = node.name_hint
                # current_conv_params is in output->input order, keep it that way
                if var_name not in paths:
                  paths[var_name] = list(current_conv_params)
              return

            elif isinstance(node, relay.Call):
              if isinstance(node.op, relay.Function):
                # Composite function - build mapping from inner params to outer args
                inner_func = node.op
                new_param_map = dict(local_param_map)
                for param, arg in zip(inner_func.params, node.args):
                  new_param_map[param] = arg
                # Traverse into body with the new mapping
                trace_path(inner_func.body, current_conv_params, new_param_map)
              elif is_qconv(node):
                # Conv operation - add to path and continue
                conv_param = extract_conv_params(node)
                new_params = current_conv_params + [conv_param]
                # Continue tracing the data input (first arg)
                trace_path(node.args[0], new_params, local_param_map)
              elif isinstance(node.op, tvm.ir.Op):
                # Other ops (add, cast, etc.) - traverse all args
                for arg in node.args:
                  trace_path(arg, current_conv_params, local_param_map)
              else:
                # Unknown call type
                for arg in node.args:
                  trace_path(arg, current_conv_params, local_param_map)

            elif isinstance(node, relay.TupleGetItem):
              trace_path(node.tuple_value, current_conv_params, local_param_map)

            elif isinstance(node, relay.Tuple):
              for field in node.fields:
                trace_path(field, current_conv_params, local_param_map)

            elif isinstance(node, relay.Constant):
              # Constants don't lead to input vars
              pass

          trace_path(expr, [], param_map)
          return paths

        def _compute_input_tile_from_output(self, out_base, out_size, conv_params):
          """
          Compute required input tile range from output tile range.

          Args:
              out_base: output tile start position
              out_size: output tile size
              conv_params: List of (kernel_size, stride, padding_top, padding_bottom)
                          in output→input order

          Returns:
              (input_base, input_size) tuple
          """
          curr_base = out_base
          curr_size = out_size

          for k, s, p_top, p_bottom in conv_params:
            # Backward calculation: output → input
            # input_base = output_base * stride - padding_top
            # input_size = (output_size - 1) * stride + kernel_size
            new_base = curr_base * s - p_top
            new_size = (curr_size - 1) * s + k

            curr_base = new_base
            curr_size = new_size

          return curr_base, curr_size

        def apply_input_sub_tiling(self, output_tile_specs, trimmed_input_tiles,
                                   input_tensor_info, output_tensor_info, inode_cnt_sizes):
          """
          Apply hierarchical input sub-tiling when input tiles exceed memory limit.

          When output is fully tiled (each tile = 1 output row) but required input still
          exceeds memory, we sub-divide the input. For each input sub-tile iteration:
          - Output has size=0 for intermediate sub-tiles
          - Output has actual size only for the last sub-tile of each output tile

          Example:
            Original: output bases=[0,1,2], sizes=[1,1,1], input sizes=[8,0,0] (first needs 8 rows)
            If 8 rows too big, split into 4 sub-tiles of 2 rows each:
            Result: output bases=[0,0,0,0,1,2], sizes=[0,0,0,1,1,1]
                    input bases=[0,2,4,6,8,8], sizes=[2,2,2,2,0,0]

          Args:
              output_tile_specs: {out_idx: (bases, sizes)}
              trimmed_input_tiles: {var_name: (trimmed_bases, trimmed_sizes)} - already padding/halo removed
              input_tensor_info: List of input tensor info tuples
              output_tensor_info: List of output tensor info tuples
              inode_cnt_sizes: {inode_name: cnt_size} - counter sizes per inode

          Returns:
              (new_output_tile_specs, new_trimmed_input_tiles)
          """
          debug_print("  Applying input sub-tiling due to memory limit")
          debug_print(f"    inode_cnt_sizes: {inode_cnt_sizes}")

          # Build input var_name -> inode_name mapping
          input_var_to_inode = {}
          for edge, height, width, channels, elem_size, inode_name, mem_block, var_name in input_tensor_info:
            input_var_to_inode[var_name] = inode_name
          debug_print(f"    input_var_to_inode: {input_var_to_inode}")

          # Get memory per row for each input variable
          input_mem_per_row = {}
          for edge, height, width, channels, elem_size, inode_name, mem_block, var_name in input_tensor_info:
            if height > 0:
              input_mem_per_row[var_name] = mem_block.size / height
            else:
              input_mem_per_row[var_name] = mem_block.size
          debug_print(f"    input_mem_per_row: {input_mem_per_row}")

          # Debug output tensor info
          debug_print(f"    output_tensor_info:")
          for out_idx, (edge, height, width, channels, elem_size, inode_name, mem_block) in enumerate(output_tensor_info):
            debug_print(f"      out[{out_idx}]: inode={inode_name}, height={height}, mem_block.size={mem_block.size}")

          num_tiles = len(list(output_tile_specs.values())[0][0])
          debug_print(f"    num_tiles: {num_tiles}")

          # For each tile, calculate per-inode available memory and determine sub-tiling
          tile_sub_tile_info = []  # [(tile_idx, num_sub_tiles, max_var, sub_tile_size, total_input_size)]

          for tile_idx in range(num_tiles):
            # Calculate per-inode output memory for this tile
            inode_output_mem = defaultdict(float)
            for out_idx, (edge, height, width, channels, elem_size, inode_name, mem_block) in enumerate(output_tensor_info):
              out_bases, out_sizes = output_tile_specs[out_idx]
              if tile_idx < len(out_sizes):
                tile_height = out_sizes[tile_idx]
                if height > 0 and tile_height > 0:
                  tiled_size = math.ceil(mem_block.size * tile_height / height)
                  inode_output_mem[inode_name] += tiled_size

            # Calculate per-inode available memory for input = limit - cnt - output
            inode_available_memory = {}
            all_inodes = set(input_var_to_inode.values())
            for inode_name in all_inodes:
              cnt_size = inode_cnt_sizes.get(inode_name, 0)
              output_mem = inode_output_mem.get(inode_name, 0)
              inode_available_memory[inode_name] = ImcflowDeviceConfig.INODE_MAX_TILING_SIZE - cnt_size - output_mem

            # Debug: print per-tile inode memory info (only for first few tiles)
            if tile_idx < 3:
              debug_print(f"    --- Tile {tile_idx} ---")
              debug_print(f"      inode_output_mem: {dict(inode_output_mem)}")
              debug_print(f"      inode_available_memory: {inode_available_memory}")

            # Calculate per-inode input memory and track max input per inode
            inode_input_mem = defaultdict(float)
            inode_max_input_info = {}  # {inode_name: (max_var, max_size, mem_per_row)}

            for var_name, (trimmed_bases, trimmed_sizes) in trimmed_input_tiles.items():
              if tile_idx < len(trimmed_sizes):
                tile_input_size = trimmed_sizes[tile_idx]
                var_inode = input_var_to_inode[var_name]
                mem_per_row = input_mem_per_row[var_name]
                mem = tile_input_size * mem_per_row
                inode_input_mem[var_inode] += mem
                # Track the largest input in each inode
                if var_inode not in inode_max_input_info:
                  inode_max_input_info[var_inode] = (var_name, tile_input_size, mem_per_row)
                else:
                  curr_max_mem = inode_max_input_info[var_inode][1] * inode_max_input_info[var_inode][2]
                  if mem > curr_max_mem:
                    inode_max_input_info[var_inode] = (var_name, tile_input_size, mem_per_row)

            # Debug: print input memory info (only for first few tiles)
            if tile_idx < 3:
              debug_print(f"      inode_input_mem: {dict(inode_input_mem)}")
              debug_print(f"      inode_max_input_info: {inode_max_input_info}")

            # Find the most constrained inode (highest excess ratio)
            needs_subtiling = False
            subtiling_inode = None
            max_excess_ratio = 0

            for inode_name, input_mem in inode_input_mem.items():
              available = inode_available_memory.get(inode_name, float('inf'))
              if tile_idx < 3:
                debug_print(f"      inode {inode_name}: input_mem={input_mem}, available={available}, exceeds={input_mem > available}")
              if available > 0 and input_mem > available:
                excess_ratio = input_mem / available
                if excess_ratio > max_excess_ratio:
                  max_excess_ratio = excess_ratio
                  needs_subtiling = True
                  subtiling_inode = inode_name

            if tile_idx < 3:
              debug_print(f"      needs_subtiling={needs_subtiling}, subtiling_inode={subtiling_inode}, max_excess_ratio={max_excess_ratio}")

            # Calculate how many sub-tiles needed based on the most constrained inode
            if needs_subtiling: 
              max_var, max_input_size, mem_per_row = inode_max_input_info[subtiling_inode]
              available = inode_available_memory.get(subtiling_inode, 0)
              if mem_per_row > 0 and available > 0:
                max_rows_per_subtile = max(1, int(available / mem_per_row))
                num_sub_tiles = math.ceil(max_input_size / max_rows_per_subtile)
                sub_tile_size = math.ceil(max_input_size / num_sub_tiles)
                tile_sub_tile_info.append((tile_idx, num_sub_tiles, max_var, sub_tile_size, max_input_size))
                debug_print(f"    Tile {tile_idx}: inode {subtiling_inode} input {max_var} needs {num_sub_tiles} sub-tiles of ~{sub_tile_size} rows (total {max_input_size}, available={available}, mem_per_row={mem_per_row}, max_rows_per_subtile={max_rows_per_subtile})")
              else:
                raise ValueError("Cannot determine sub-tiling due to zero mem_per_row or available memory.")
            else:
              tile_sub_tile_info.append((tile_idx, 1, None, 0, 0))

          # Build new expanded tile specs
          new_output_tile_specs = {out_idx: ([], []) for out_idx in output_tile_specs.keys()}
          new_trimmed_input_tiles = {var_name: ([], []) for var_name in trimmed_input_tiles.keys()}

          debug_print("tile sub tile info:")
          debug_print(tile_sub_tile_info)
          for tile_idx, num_sub_tiles, max_var, sub_tile_size, total_input_size in tile_sub_tile_info:
            if num_sub_tiles == 1:
              # No sub-tiling needed, copy as-is
              for out_idx, (out_bases, out_sizes) in output_tile_specs.items():
                new_output_tile_specs[out_idx][0].append(out_bases[tile_idx])
                new_output_tile_specs[out_idx][1].append(out_sizes[tile_idx])

              for var_name, (in_bases, in_sizes) in trimmed_input_tiles.items():
                new_trimmed_input_tiles[var_name][0].append(in_bases[tile_idx])
                new_trimmed_input_tiles[var_name][1].append(in_sizes[tile_idx])
            else:
              # Sub-tiling needed
              # Get original output info for this tile
              out_bases_orig = {out_idx: output_tile_specs[out_idx][0][tile_idx] for out_idx in output_tile_specs}
              out_sizes_orig = {out_idx: output_tile_specs[out_idx][1][tile_idx] for out_idx in output_tile_specs}

              # Get original trimmed input info for this tile
              in_bases_orig = {var_name: trimmed_input_tiles[var_name][0][tile_idx] for var_name in trimmed_input_tiles}
              in_sizes_orig = {var_name: trimmed_input_tiles[var_name][1][tile_idx] for var_name in trimmed_input_tiles}

              # Get base for the max_var
              max_var_base = in_bases_orig[max_var]

              for sub_idx in range(num_sub_tiles):
                is_last_subtile = (sub_idx == num_sub_tiles - 1)

                # Output: size=0 for intermediate, actual size for last
                for out_idx in output_tile_specs.keys():
                  new_output_tile_specs[out_idx][0].append(out_bases_orig[out_idx])
                  if is_last_subtile:
                    new_output_tile_specs[out_idx][1].append(out_sizes_orig[out_idx])
                  else:
                    new_output_tile_specs[out_idx][1].append(0)

                # Input: divide into sub-tiles
                for var_name in trimmed_input_tiles.keys():
                  if var_name == max_var:
                    # This is the variable being sub-tiled
                    sub_base = max_var_base + sub_idx * sub_tile_size
                    sub_size = min(sub_tile_size, max_var_base + total_input_size - sub_base)
                    new_trimmed_input_tiles[var_name][0].append(sub_base)
                    new_trimmed_input_tiles[var_name][1].append(max(0, sub_size))
                  else:
                    # Other variables: proportional split
                    orig_base = in_bases_orig[var_name]
                    orig_size = in_sizes_orig[var_name]
                    if total_input_size > 0 and orig_size > 0:
                      ratio = sub_tile_size / total_input_size
                      other_sub_size = int(orig_size * ratio)
                      other_sub_base = orig_base + sub_idx * other_sub_size
                      new_trimmed_input_tiles[var_name][0].append(other_sub_base)
                      new_trimmed_input_tiles[var_name][1].append(other_sub_size)
                    else:
                      # Just repeat with size=0 for intermediate, orig for last
                      new_trimmed_input_tiles[var_name][0].append(orig_base)
                      if is_last_subtile:
                        new_trimmed_input_tiles[var_name][1].append(orig_size)
                      else:
                        new_trimmed_input_tiles[var_name][1].append(0)

          new_num_iterations = len(new_output_tile_specs[0][0]) if new_output_tile_specs else 0
          debug_print(f"  [{self.func_name}] Input sub-tiling: {num_tiles} tiles -> {new_num_iterations} iterations")

          return new_output_tile_specs, new_trimmed_input_tiles

        def remove_padding_and_halo(self, input_bases, input_sizes, input_height):
          """
          Remove padding and halo regions from input tile specifications.

          The raw input tile computed from backward calculation includes:
          1. Padding regions (negative indices or indices beyond input height)
          2. Halo regions (overlap with previous tiles that were already processed)

          This function trims these regions to get the actual new input data needed.

          Args:
              input_bases: List of input tile start positions (may include padding/halo)
              input_sizes: List of input tile sizes (may include padding/halo)
              input_height: Original input tensor height

          Returns:
              (trimmed_bases, trimmed_sizes) - adjusted to valid input ranges without overlap
          """
          trimmed_bases = []
          trimmed_sizes = []

          prev_end = 0  # Track where previous tile ended (for halo removal)

          for i, (in_base, in_size) in enumerate(zip(input_bases, input_sizes)):
            in_end = in_base + in_size

            # Clamp to valid input range [0, input_height)
            valid_start = max(0, in_base)
            valid_end = min(input_height, in_end)

            # Remove halo: don't include data that was already processed by previous tile
            if i > 0:
              actual_start = max(valid_start, prev_end)
            else:
              actual_start = valid_start

            actual_size = valid_end - actual_start

            trimmed_bases.append(actual_start)
            trimmed_sizes.append(max(0, actual_size))

            # Update prev_end for next tile's halo calculation
            prev_end = valid_end

          return trimmed_bases, trimmed_sizes

        def calculate_all_input_tiles_from_output(self, target_func, output_height_bases, output_height_sizes):
          """
          Calculate required input tile height coordinates and sizes for ALL input variables.

          For graphs with multiple inputs (e.g., ResNet skip connections, multi-input addition),
          this traces all paths from output to each input and computes the required tiles.

          Args:
              target_func: relay.Function to analyze
              output_height_bases: List[int] - start positions of each output tile
              output_height_sizes: List[int] - sizes of each output tile

          Returns:
              dict: {var_name: (input_bases, input_sizes)} for each input variable
          """
          body = target_func.body

          # Handle Tuple output - for now, just use the first output
          if isinstance(body, relay.Tuple):
            output_expr = body.fields[0]
          else:
            output_expr = body

          # Trace all paths to inputs
          paths = self._trace_all_paths_to_inputs(output_expr)

          # Calculate input tiles for each input variable
          results = {}
          for var_name, conv_params in paths.items():
            input_bases = []
            input_sizes = []
            for out_base, out_size in zip(output_height_bases, output_height_sizes):
              in_base, in_size = self._compute_input_tile_from_output(
                out_base, out_size, conv_params
              )
              input_bases.append(in_base)
              input_sizes.append(in_size)

            results[var_name] = (input_bases, input_sizes)

          return results

        def calculate_all_output_tiles_from_input(self, target_func, input_tiles_dict):
          """
          Calculate output tile specifications for ALL outputs based on given input tiles.

          This is the forward direction: given input tiles, determine what output tiles
          will be produced for each output in a multi-output function. This is crucial
          for multi-output functions where different outputs have different transformations
          (e.g., one goes through conv, another goes through simple multiply).

          IMPORTANT: The input tiles are already trimmed (padding/halo removed), so they
          are in the actual tensor coordinate system. When there are no conv operations
          (conv_params is empty), the output tiles should match the input tiles exactly.

          Args:
              target_func: relay.Function to analyze
              input_tiles_dict: dict {var_name: (input_bases, input_sizes)} for each input variable

          Returns:
              dict: {output_idx: (output_bases, output_sizes)} for each output in the function
          """
          body = target_func.body

          # Handle both single output and Tuple output
          if isinstance(body, relay.Tuple):
            output_exprs = body.fields
          else:
            output_exprs = [body]

          results = {}

          for output_idx, output_expr in enumerate(output_exprs):
            # Trace paths from this specific output to all inputs
            paths = self._trace_all_paths_to_inputs(output_expr)

            debug_print(f"  [{self.func_name}] Output[{output_idx}] paths: {list(paths.keys())}")
            for var_name, conv_params in paths.items():
              debug_print(f"  [{self.func_name}]   {var_name} -> conv_params: {conv_params}")

            if not paths:
              # No paths found - this shouldn't happen but handle gracefully
              # Use first available input tile spec as fallback
              if input_tiles_dict:
                first_var = list(input_tiles_dict.keys())[0]
                input_bases, input_sizes = input_tiles_dict[first_var]
                results[output_idx] = (list(input_bases), list(input_sizes))
              continue

            # For each input variable that contributes to this output, calculate output tiles
            # If multiple inputs contribute, we need to merge the results
            output_candidates = []

            for var_name, conv_params in paths.items():
              if var_name not in input_tiles_dict:
                # This input is not in our input tiles dict - skip
                continue

              input_bases, input_sizes = input_tiles_dict[var_name]

              # If there are no conv operations, output tiles = input tiles
              # This is the case for element-wise operations like multiply, add, etc.
              if not conv_params:
                debug_print(f"  [{self.func_name}]   No conv params for {var_name}, output tiles = input tiles")
                output_candidates.append((list(input_bases), list(input_sizes)))
              else:
                # If there are conv operations, we cannot reliably recalculate output tiles
                # from trimmed input tiles because:
                # 1. Trimmed input tiles have padding/halo removed
                # 2. They are in post-trim coordinate system
                # 3. The original output tiles were calculated before trimming
                #
                # Therefore, for conv paths, we should keep the original output tiles.
                # We signal this by returning None, which the caller will interpret as
                # "keep original tiles unchanged"
                debug_print(f"  [{self.func_name}]   Conv params exist for {var_name}, cannot recalculate from trimmed input")
                output_candidates.append(None)

            if output_candidates:
              # Filter out None values (conv paths that cannot be recalculated)
              valid_candidates = [c for c in output_candidates if c is not None]

              if not valid_candidates:
                # All paths have conv operations, cannot recalculate
                # Return None to signal "keep original tiles"
                results[output_idx] = None
              elif len(valid_candidates) == 1:
                results[output_idx] = valid_candidates[0]
              else:
                # Multiple valid candidates, merge them
                merged_bases, merged_sizes = self.merge_input_tile_boundaries(valid_candidates)
                results[output_idx] = (merged_bases, merged_sizes)

          return results

        def _compute_output_tile_from_input(self, in_base, in_size, conv_params):
          """
          Compute output tile range from input tile range (forward calculation).

          This is the inverse of _compute_input_tile_from_output.

          The backward formula is:
            input_base = output_base * stride - padding_top
            input_size = (output_size - 1) * stride + kernel_size

          So the forward formula (solving for output from input) is:
            output_base = (input_base + padding_top) / stride
            output_size = (input_size - kernel_size) / stride + 1

          Args:
              in_base: input tile start position
              in_size: input tile size
              conv_params: List of (kernel_size, stride, padding_top, padding_bottom)
                          in output→input order (we'll process in reverse)

          Returns:
              (output_base, output_size) tuple
          """
          curr_base = in_base
          curr_size = in_size

          # Process conv params in reverse order (input → output direction)
          for k, s, p_top, p_bottom in reversed(conv_params):
            # Forward calculation: input → output
            # From backward: input_base = output_base * stride - pad_top
            # Solving: output_base = (input_base + pad_top) / stride
            new_base = (curr_base + p_top) // s

            # From backward: input_size = (output_size - 1) * stride + kernel_size
            # Solving: (output_size - 1) * stride = input_size - kernel_size
            #          output_size - 1 = (input_size - kernel_size) / stride
            #          output_size = (input_size - kernel_size) / stride + 1
            new_size = (curr_size - k) // s + 1
            new_size = max(0, new_size)

            curr_base = new_base
            curr_size = new_size

          return curr_base, curr_size

        def merge_input_tile_boundaries(self, candidates):
          """
          Merge multiple input tile boundary candidates by taking the maximum range for each tile.

          When multiple outputs require different input tile boundaries for the same input variable,
          we need to select the tile boundaries that satisfy ALL outputs. This is done by taking
          the minimum base (earliest start) and maximum end (latest end) for each tile.

          Args:
              candidates: List of (input_bases, input_sizes) tuples, where each tuple represents
                         one candidate's tile specifications. Each input_bases is a list of
                         start positions, and each input_sizes is a list of sizes.

          Returns:
              (merged_bases, merged_sizes): The merged tile specification that covers all candidates.
          """
          if not candidates:
            return [], []

          # All candidates should have the same number of tiles
          num_tiles = len(candidates[0][0])
          for bases, sizes in candidates:
            if len(bases) != num_tiles or len(sizes) != num_tiles:
              raise ValueError("All candidates must have the same number of tiles")

          merged_bases = []
          merged_sizes = []

          for tile_idx in range(num_tiles):
            # For each tile, find the minimum start and maximum end across all candidates
            min_base = float('inf')
            max_end = float('-inf')

            for bases, sizes in candidates:
              base = bases[tile_idx]
              end = base + sizes[tile_idx]
              min_base = min(min_base, base)
              max_end = max(max_end, end)

            merged_bases.append(min_base)
            merged_sizes.append(max_end - min_base)

          return merged_bases, merged_sizes

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

            is_inode, inode_tensorid = self.is_inode_in_edge(edge)
            if not is_inode:
               debug_print(f"  [{self.func_name}] non-inode edge: {edge}")
               raise ValueError("Edge does not involve an inode.")
            hw_node_id = ImcflowDeviceConfig().get_hw_node(inode_tensorid.graph_node_id, edge.split_idx)
            # hw_node_id = hw_node_id if not isinstance(hw_node_id, tuple) else hw_node_id[edge.split_idx]

            inode_name = hw_node_id.name  # ex) inode_3_0

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
            elif tensor_type == "data" or tensor_type == "odata" or ("func_out" in tensor_type) or tensor_type == "var":
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
          # New approach:
          # 1. Collect output/input tensor info
          # 2. For each tiling factor, calculate tile specs for each output
          # 3. For each input, merge tile boundaries across all outputs
          # 4. Calculate memory using ceil formula
          # 5. If full output tiling fails, apply hierarchical input sub-tiling
          layout_map = ImcflowDeviceConfig().LayoutMap

          # Collect output tensor info: [(edge, height, width, channels, elem_size, inode_name, mem_block)]
          output_tensor_info = []
          for inode_name, tensors in inode_tensors.items():
            for edge, mem_block, inode_tensorid in tensors['output']:
              src_node = transform_utils.getNodeFromTensorID(edge.src_id)
              ttype = transform_utils.get_type(mod, src_node)
              shape,dtype = ttype.shape, ttype.dtype
              channels = int(shape[1])
              height = int(shape[2])
              width = int(shape[3])
              elem_size = np.dtype(dtype).itemsize
              output_tensor_info.append((edge, height, width, channels, elem_size, inode_name, mem_block))

          # Collect input tensor info: [(edge, height, width, channels, elem_size, inode_name, mem_block, var_name)]
          input_tensor_info = []
          for inode_name, tensors in inode_tensors.items():
            for edge, mem_block, inode_tensorid in tensors['input']:
              src_node = transform_utils.getNodeFromTensorID(edge.src_id)
              shape = src_node.type_annotation.shape
              dtype = src_node.type_annotation.dtype
              channels = int(shape[1])
              height = int(shape[2])
              width = int(shape[3])
              elem_size = np.dtype(dtype).itemsize
              var_name = src_node.name_hint if isinstance(src_node, relay.Var) else "data"
              input_tensor_info.append((edge, height, width, channels, elem_size, inode_name, mem_block, var_name))

          # max_tiling_factor: single unified max tiling factor = min of all output heights
          # All outputs share the same tiling factor
          max_tiling_factor = min(height for (_, height, _, _, _, _, _) in output_tensor_info)
          tiling_factor = 1

          # Find tiling factor by incrementing until memory fits
          debug_print(f"\n  [{self.func_name}] ===== TILING FACTOR SEARCH =====")
          debug_print(f"  [{self.func_name}] INODE_MAX_TILING_SIZE = {ImcflowDeviceConfig.INODE_MAX_TILING_SIZE} bytes")
          debug_print(f"  [{self.func_name}] max_tiling_factor (unified, max output height) = {max_tiling_factor}")
          debug_print(f"  [{self.func_name}] Output tensors:")
          for out_idx, (edge, height, width, channels, elem_size, inode_name, mem_block) in enumerate(output_tensor_info):
            actual_size = height * width * channels * elem_size
            debug_print(f"    Output[{out_idx}]: H={height}, W={width}, C={channels}, elem_size={elem_size}, actual_size={actual_size}, mem_block.size={mem_block.size}, inode={inode_name}")
          debug_print(f"  [{self.func_name}] Input tensors:")
          for edge, height, width, channels, elem_size, inode_name, mem_block, var_name in input_tensor_info:
            actual_size = height * width * channels * elem_size
            debug_print(f"    Input[{var_name}]: H={height}, W={width}, C={channels}, elem_size={elem_size}, actual_size={actual_size}, mem_block.size={mem_block.size}, inode={inode_name}")

          need_input_sub_tiling = False

          # Helper to check if tiling factor has reached max
          def at_max_tiling():
            return tiling_factor >= max_tiling_factor

          # Helper to get the number of tiles
          def get_num_tiles():
            return tiling_factor

          while True:
            # Step 1: Calculate output tile specs for each output tensor (unified tiling factor)
            output_tile_specs = {}  # {output_idx: (bases, sizes)}
            num_tiles = get_num_tiles()  # unified tile count
            for out_idx, (edge, height, width, channels, elem_size, inode_name, mem_block) in enumerate(output_tensor_info):
              # Use unified tiling_factor for all outputs
              base_tile_size = math.ceil(height / tiling_factor)
              bases = [i * base_tile_size for i in range(tiling_factor)]
              sizes = [min(base_tile_size, height - base) for base in bases]
              output_tile_specs[out_idx] = (bases, sizes)

            # Step 2: For each output, calculate input tile boundaries for all inputs
            # Then merge boundaries per input variable across all outputs
            input_tile_candidates = {}  # {var_name: [(bases, sizes), ...]}

            for out_idx, (out_bases, out_sizes) in output_tile_specs.items():
              debug_print(f"  [{self.func_name}] Output[{out_idx}] tiles for tiling_factor {tiling_factor}: bases={out_bases}, sizes={out_sizes}")
              # Calculate input tiles from this output
              all_input_tiles = self.calculate_all_input_tiles_from_output(
                func, out_bases, out_sizes
              )

              for var_name, (in_bases, in_sizes) in all_input_tiles.items():
                if var_name not in input_tile_candidates:
                  input_tile_candidates[var_name] = []
                input_tile_candidates[var_name].append((in_bases, in_sizes))
                debug_print(f"    Input[{var_name}] candidate tiles from Output[{out_idx}]: bases={in_bases}, sizes={in_sizes}")

            # Step 3: Merge input tile boundaries and compute trimmed tiles for each input variable
            # Build input_heights lookup for trimming
            input_heights = {var_name: height for (_, height, _, _, _, _, _, var_name) in input_tensor_info}

            merged_input_tiles = {}   # {var_name: (merged_bases, merged_sizes)}
            trimmed_input_tiles = {}  # {var_name: (trimmed_bases, trimmed_sizes)}
            for var_name, candidates in input_tile_candidates.items():
              merged_bases, merged_sizes = self.merge_input_tile_boundaries(candidates)
              merged_input_tiles[var_name] = (merged_bases, merged_sizes)

              # Compute trimmed (padding/halo removed) tiles
              input_height = input_heights.get(var_name, 0)
              trimmed_bases, trimmed_sizes = self.remove_padding_and_halo(merged_bases, merged_sizes, input_height)
              trimmed_input_tiles[var_name] = (trimmed_bases, trimmed_sizes)

              debug_print(f"  [{self.func_name}] Merged Input[{var_name}] tiles: bases={merged_bases}, sizes={merged_sizes}")
              debug_print(f"  [{self.func_name}] Trimmed Input[{var_name}] tiles: bases={trimmed_bases}, sizes={trimmed_sizes}")

            # Step 4: Calculate memory for each tile PER INODE using ceil formula
            # Memory = ceil(original_block_size / (total_height / tile_height))
            # Track max tile memory per inode separately
            inode_max_tile_memory = {}  # {inode_name: max_tile_memory}
            debug_tile_memories = []

            # Initialize inode memory tracking
            for inode_name in inode_tensors.keys():
              inode_max_tile_memory[inode_name] = 0

            for tile_idx in range(num_tiles):
              # Track tile memory per inode for this tile
              inode_tile_memory = {inode_name: 0 for inode_name in inode_tensors.keys()}
              tile_detail = {"tile_idx": tile_idx, "outputs": [], "inputs": [], "per_inode": {}}

              # Output tensor memory for this tile (per inode)
              for out_idx, (edge, height, width, channels, elem_size, inode_name, mem_block) in enumerate(output_tensor_info):
                out_bases, out_sizes = output_tile_specs[out_idx]
                if tile_idx < len(out_sizes):
                  tile_height = out_sizes[tile_idx]
                  original_size = mem_block.size
                  # ceil(original_size * tile_height / height)
                  if tile_height > 0 and height > 0:
                    tiled_size = math.ceil(original_size * tile_height / height)
                  else:
                    tiled_size = 0
                  inode_tile_memory[inode_name] += tiled_size
                  tile_detail["outputs"].append((out_idx, inode_name, tile_height, tiled_size))

              # Input tensor memory for this tile (using trimmed boundaries, per inode)
              # Track processed var_names to avoid counting multicast inputs multiple times
              processed_input_vars = set()
              for edge, height, width, channels, elem_size, inode_name, mem_block, var_name in input_tensor_info:
                if var_name in processed_input_vars:
                  continue  # Skip duplicate input variables (multicast case)
                processed_input_vars.add(var_name)
                if var_name in trimmed_input_tiles:
                  trimmed_bases, trimmed_sizes = trimmed_input_tiles[var_name]
                  if tile_idx < len(trimmed_sizes):
                    tile_height = trimmed_sizes[tile_idx]
                    original_size = mem_block.size
                    # ceil(original_size * tile_height / height)
                    if height > 0:
                      tiled_size = math.ceil(original_size * tile_height / height)
                    else:
                      tiled_size = original_size
                    inode_tile_memory[inode_name] += tiled_size
                    tile_detail["inputs"].append((var_name, inode_name, tile_height, tiled_size))
                else:
                  raise ValueError(f"Input variable '{var_name}' not found in trimmed input tiles.")

              # Update max tile memory per inode
              for inode_name, tile_mem in inode_tile_memory.items():
                inode_max_tile_memory[inode_name] = max(inode_max_tile_memory[inode_name], tile_mem)

              tile_detail["per_inode"] = dict(inode_tile_memory)
              debug_tile_memories.append((tile_idx, tile_detail))

            # Print detailed debug info
            debug_print(f"  [{self.func_name}] --- Tiling factor {tiling_factor} detail ---")
            for tile_idx, detail in debug_tile_memories[:3]:  # Show first 3 tiles
              debug_print(f"    Tile[{tile_idx}]: per_inode={detail['per_inode']}")

            # Add counter base address size per inode
            inode_cnt_sizes = {}  # {inode_name: cnt_size}
            for inode_name, tensors_for_inode in inode_tensors.items():
              cnt_size = len(tensors_for_inode['input'] + tensors_for_inode['output']) * 32
              inode_cnt_sizes[inode_name] = cnt_size
              inode_max_tile_memory[inode_name] += cnt_size

            # Check if this tiling configuration works FOR ALL INODES
            all_inodes_fit = True
            for inode_name, max_mem in inode_max_tile_memory.items():
              if max_mem > ImcflowDeviceConfig.INODE_MAX_TILING_SIZE:
                all_inodes_fit = False
                debug_print(f"  [{self.func_name}] Tiling factor {tiling_factor}: inode {inode_name} max tile memory = {max_mem} bytes (too large, limit={ImcflowDeviceConfig.INODE_MAX_TILING_SIZE})")
              else:
                debug_print(f"  [{self.func_name}] Tiling factor {tiling_factor}: inode {inode_name} max tile memory = {max_mem} bytes (fits)")

            if all_inodes_fit:
              debug_print(f"  [{self.func_name}] Tiling factor {tiling_factor}: all inodes fit")
              break
            else:
              # Increase unified tiling factor
              if tiling_factor < max_tiling_factor:
                tiling_factor += 1
                debug_print(f"  [{self.func_name}] Increasing tiling factor to {tiling_factor}")
              else:
                # Reached max tiling factor, need input sub-tiling
                debug_print(f"  [{self.func_name}] Reached max tiling factor, need input sub-tiling")
                break

          # Check if any inode exceeds the limit after reaching max tiling factor
          any_inode_exceeds = any(
            mem > ImcflowDeviceConfig.INODE_MAX_TILING_SIZE
            for mem in inode_max_tile_memory.values()
          )
          if at_max_tiling() and any_inode_exceeds:
            need_input_sub_tiling = True
            debug_print(f"  [{self.func_name}] Full output tiling reached, applying input sub-tiling")

          debug_print(f"  [{self.func_name}] Final tiling factor = {tiling_factor}")

          # Use the output_tile_specs and trimmed_input_tiles from the while loop
          # (already calculated with final tiling_factor)
          final_output_tile_specs = output_tile_specs
          final_trimmed_input_tiles = trimmed_input_tiles

          # Apply input sub-tiling if needed
          # Pass inode_cnt_sizes so the function can calculate per-inode available memory
          if need_input_sub_tiling:
            final_output_tile_specs, final_trimmed_input_tiles = self.apply_input_sub_tiling(
              final_output_tile_specs,
              final_trimmed_input_tiles,
              input_tensor_info,
              output_tensor_info,
              inode_cnt_sizes
            )
          
          # check all input and output tensor tiling factor is same (tile loop count)
          num_iterations_set = set()
          for out_idx, (bases, sizes) in final_output_tile_specs.items():
            num_iterations_set.add(len(bases))
          for var_name, (bases, sizes) in final_trimmed_input_tiles.items():
            num_iterations_set.add(len(bases))
          if len(num_iterations_set) != 1:
            raise ValueError(f"Tiling factor mismatch among input/output tensors: {num_iterations_set}")

          # Debug print final tile specs
          for out_idx, (bases, sizes) in final_output_tile_specs.items():
            debug_print(f"  [{self.func_name}] Final Output[{out_idx}] tiles: bases={bases}, sizes={sizes}")
          for var_name, (bases, sizes) in final_trimmed_input_tiles.items():
            debug_print(f"  [{self.func_name}] Final Input[{var_name}] tiles: bases={bases}, sizes={sizes}")

          # ================================================
          # Phase 2.5: Output Tile Verification and Recalculation
          # ================================================
          # After input tiles are finalized, recalculate output tiles to detect mismatches.
          # This is crucial for multi-output functions where different outputs have different
          # graph paths (e.g., one through conv, another through multiply).
          debug_print(f"  [{self.func_name}] ===== OUTPUT TILE VERIFICATION =====")

          if final_trimmed_input_tiles and len(final_output_tile_specs) > 0:
            # Recalculate output tiles from the finalized input tiles
            recalculated_output_tile_specs = self.calculate_all_output_tiles_from_input(
              func, final_trimmed_input_tiles
            )

            # Compare and warn/update if mismatches found
            for out_idx in sorted(final_output_tile_specs.keys()):
              original_bases, original_sizes = final_output_tile_specs[out_idx]

              if out_idx in recalculated_output_tile_specs:
                recalc_result = recalculated_output_tile_specs[out_idx]

                if recalc_result is None:
                  # Conv path - cannot recalculate from trimmed input, keep original
                  debug_print(f"  [{self.func_name}] Output[{out_idx}] has conv operations, keeping original tiles: bases={original_bases}, sizes={original_sizes}")
                else:
                  recalc_bases, recalc_sizes = recalc_result

                  # Check for mismatch
                  if (list(recalc_bases) != list(original_bases) or
                      list(recalc_sizes) != list(original_sizes)):
                    debug_print(f"  [{self.func_name}] MISMATCH DETECTED for Output[{out_idx}]:")
                    debug_print(f"    Original: bases={original_bases}, sizes={original_sizes}")
                    debug_print(f"    Recalculated: bases={recalc_bases}, sizes={recalc_sizes}")
                    debug_print(f"    Using recalculated values.")

                    # Update with recalculated values
                    final_output_tile_specs[out_idx] = (recalc_bases, recalc_sizes)
                  else:
                    debug_print(f"  [{self.func_name}] Output[{out_idx}] tiles verified: bases={recalc_bases}, sizes={recalc_sizes}")
              else:
                debug_print(f"  [{self.func_name}] Warning: Could not recalculate tiles for Output[{out_idx}]")

          debug_print(f"  [{self.func_name}] ===== END VERIFICATION =====")

          # ================================================
          # Phase 3: Perform actual allocation with tiling
          # ================================================
          ImcflowDeviceConfig().ImcflowFuncMap[func_name].tiling_factor = num_iterations_set.pop()
          for inode_name, tensors in inode_tensors.items():
            # Allocate weight tensors (no tiling, allow overlap)
            for edge, mem_block, inode_tensorid in tensors['weight']:
              ImcflowDeviceConfig().add_tensor_edge_info(edge, TensorEdgeInfo(data_block=mem_block))
              ImcflowDeviceConfig().MemLayout[self.func_name][f"{inode_name}_data"].allocate(mem_block, phase="init")

            # Allocate input tensors (with tiling if needed)
            for edge, mem_block, inode_tensorid in tensors['input']:
              src_node = transform_utils.getNodeFromTensorID(edge.src_id)
              v_tensor_shape = src_node.type_annotation.shape
              channels = int(v_tensor_shape[1])
              height = int(v_tensor_shape[2])
              width = int(v_tensor_shape[3])
              dtype = src_node.type_annotation.dtype
              elem_size = np.dtype(dtype).itemsize
              original_size = mem_block.size

              var_name = src_node.name_hint if isinstance(src_node, relay.Var) else "data"

              # Get number of iterations from final tile specs
              num_iterations = len(list(final_output_tile_specs.values())[0][0]) if final_output_tile_specs else 1
              has_tiling = num_iterations > 1

              # Get trimmed input tile specs for this variable (already has padding/halo removed)
              if var_name in final_trimmed_input_tiles and has_tiling:
                input_height_bases, input_height_sizes = final_trimmed_input_tiles[var_name]
                debug_print(f"    Input tensor {var_name} tiles: bases={input_height_bases}, sizes={input_height_sizes}")
              else:
                input_height_bases = [0]
                input_height_sizes = [height]
                debug_print(f"    Input tensor {var_name} no tiling applied.")

              if has_tiling:
                # Calculate tiled size using ceil formula
                max_tile_height = max(input_height_sizes)
                # tiled_size = ceil(original_size * max_tile_height / height)
                if height > 0:
                  tiled_size = math.ceil(original_size * max_tile_height / height)
                else:
                  tiled_size = original_size
                mem_block.set_size(tiled_size)
                debug_print(f"    Input tensor {var_name}: {original_size} -> {tiled_size} bytes (max tile height={max_tile_height}/{height})")

              # Set tiling info
              # Calculate CPU tile base addresses and sizes based on height boundaries
              # height_offset = prod(dims after height) * elem_size (bytes per height row)
              r_ttype = transform_utils.getRTTypeForEdge(mod, edge)
              r_height_index = imcflow_layout.get_height_dim_index(layout_map[transform_utils.getNodeFromTensorID(edge.src_id)])
              assert math.prod(r_ttype.shape[0:r_height_index]) == 1, "Upper of height dimension should be 1 in imcflow."
              height_offset_elem_num = math.prod(r_ttype.shape[r_height_index + 1:]) if r_height_index + 1 < len(r_ttype.shape) else 1
              height_offset = height_offset_elem_num * np.dtype(r_ttype.dtype).itemsize
              assert height_offset % 32 == 0, "Height offset should be multiple of 32 bytes."
              pkt_cnt_per_height = height_offset//32
              pkt_cnts = [pkt_cnt_per_height * h for h in input_height_sizes]

              # Calculate CPU variable offsets (byte offset) and sizes (int32 count)
              # base address = origin_base + height_base * height_offset
              # cnt = height_size * height_offset / sizeof(int)
              c_var_offsets = [base * height_offset for base in input_height_bases]
              c_var_sizes = [h_size * height_offset // 4 for h_size in input_height_sizes]  # div by sizeof(int)=4

              block_tiling_info = BlockTileInfo()
              block_tiling_info.set_info(
                height_base_coords=input_height_bases,
                height_sizes=input_height_sizes,
                pkt_cnts=pkt_cnts,
                c_var_offsets=c_var_offsets,
                c_var_sizes=c_var_sizes
              )
              mem_block.tiling_info = block_tiling_info
              debug_print(f"    Input tensor tiles: bases={input_height_bases}, sizes={input_height_sizes}, pkt_cnts={pkt_cnts}")

              # add edge info and allocate
              ImcflowDeviceConfig().add_tensor_edge_info(edge, TensorEdgeInfo(data_block=mem_block, block_tiling_info=block_tiling_info))
              ImcflowDeviceConfig().MemLayout[self.func_name][f"{inode_name}_data"].allocate(mem_block, phase="exec")

            # Allocate output tensors (with tiling if needed)
            for out_local_idx, (edge, mem_block, inode_tensorid) in enumerate(tensors['output']):
              src_node = transform_utils.getNodeFromTensorID(edge.src_id)
              v_tensor_shape = src_node.checked_type.shape
              channels = int(v_tensor_shape[1])
              height = int(v_tensor_shape[2])
              width = int(v_tensor_shape[3])
              dtype = src_node.checked_type.dtype
              elem_size = np.dtype(dtype).itemsize
              original_size = mem_block.size

              # Find this output's tile spec
              # Match by finding the output in output_tensor_info
              output_tile_bases = [0]
              output_tile_sizes = [height]
              for out_idx, (o_edge, o_h, o_w, o_c, o_e, o_inode, o_mem) in enumerate(output_tensor_info):
                if o_edge == edge:
                  output_tile_bases, output_tile_sizes = final_output_tile_specs[out_idx]
                  break

              # Get number of iterations from final tile specs
              num_iterations = len(list(final_output_tile_specs.values())[0][0]) if final_output_tile_specs else 1
              has_tiling = num_iterations > 1

              if has_tiling:
                # Calculate tiled size using ceil formula
                max_tile_height = max(output_tile_sizes)
                if height > 0:
                  tiled_size = math.ceil(original_size * max_tile_height / height)
                else:
                  tiled_size = original_size
                mem_block.set_size(tiled_size)
                debug_print(f"    Output tensor: {original_size} -> {tiled_size} bytes (max tile height={max_tile_height}/{height})")

              # Set tiling info
              # Calculate CPU tile base addresses and sizes based on height boundaries
              r_ttype = transform_utils.getRTTypeForEdge(mod, edge)
              r_height_index = imcflow_layout.get_height_dim_index(layout_map[transform_utils.getNodeFromTensorID(edge.src_id)])
              assert math.prod(r_ttype.shape[0:r_height_index]) == 1, "Upper of height dimension should be 1 in imcflow."
              height_offset_elem_num = math.prod(r_ttype.shape[r_height_index + 1:]) if r_height_index + 1 < len(r_ttype.shape) else 1
              height_offset = height_offset_elem_num * np.dtype(r_ttype.dtype).itemsize
              assert height_offset % 32 == 0, "Height offset should be multiple of 32 bytes."
              pkt_cnt_per_height = height_offset//32
              # total_pkt_cnt = getInodePktCntForEdge(mod, edge)
              pkt_cnts = [pkt_cnt_per_height * h for h in output_tile_sizes]

              # Calculate CPU variable offsets (byte offset) and sizes (int32 count)
              c_output_var_offsets = [base * height_offset for base in output_tile_bases]
              c_output_var_sizes = [h_size * height_offset // 4 for h_size in output_tile_sizes]  # div by sizeof(int)=4

              block_tiling_info = BlockTileInfo()
              block_tiling_info.set_info(
                height_base_coords=output_tile_bases,
                height_sizes=output_tile_sizes,
                pkt_cnts=pkt_cnts,
                c_var_offsets=c_output_var_offsets,
                c_var_sizes=c_output_var_sizes
              )
              debug_print(f"    Output tensor tiles: bases={output_tile_bases}, sizes={output_tile_sizes}, pkt_cnts={pkt_cnts}")
              mem_block.tiling_info = block_tiling_info

              # add edge info and allocate
              ImcflowDeviceConfig().add_tensor_edge_info(edge, TensorEdgeInfo(data_block=mem_block, block_tiling_info=block_tiling_info))
              ImcflowDeviceConfig().MemLayout[self.func_name][f"{inode_name}_data"].allocate(mem_block, phase="exec")

            # Allocate other tensors (no tiling)
            for edge, mem_block, inode_tensorid in tensors['other']:
              ImcflowDeviceConfig().add_tensor_edge_info(edge, TensorEdgeInfo(data_block=mem_block))
              ImcflowDeviceConfig().MemLayout[self.func_name][f"{inode_name}_data"].allocate(mem_block, phase="init")

            # Allocate counter base address block (32 bytes)
            for edge, mem_block, inode_tensorid in (tensors['input'] + tensors['output']):
              block_name = f"{edge.simple_name()}_cnt_base_addr"
              block_size = 32
              mem_block = DataBlock(block_name, block_size)
              ImcflowDeviceConfig().MemLayout[self.func_name][f"{inode_name}_data"].allocate(mem_block, phase="exec")

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
              src_node = transform_utils.getNodeFromTensorID(edge.src_id)
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
          self.run_(mod, func_info.func_node, gv.name_hint, ttype_map[gv.name_hint])

# Policy table generation is now handled by joint_pnr_ilp.py
# The old routing_pipeline.py, mcf_router.py, xy_router.py have been removed.


class TensorPathVisualizer:
    """
    Visualizes tensor routing paths in the 2D mesh NoC topology.

    For each imcflow function, generates an image showing:
    - 2D mesh grid with inodes and imces as labeled squares
    - Tensor paths as colored lines between nodes
    - Each tensor gets a unique color

    Paths are classified into three groups:
    - inst: Instruction paths (edges that are NodeID, not TensorEdge)
    - const: Constant tensor paths (weight, config, min, max, fused_scale, fused_bias, bias, scale, threshold)
    - data: Runtime data tensor paths (odata, data, lhs, rhs, func_out*, var)
    """

    # Tensor type classification constants (aligned with mcf_router.py)
    CONST_TENSOR_TYPES = frozenset([
        'weight', 'config', 'min', 'max', 'fused_scale', 'fused_bias',
        'bias', 'scale', 'threshold'
    ])
    DATA_TENSOR_TYPES = frozenset([
        'odata', 'data', 'lhs', 'rhs', 'var',
        *[f"func_out{i}" for i in range(30)]
    ])

    # Group display configuration
    GROUP_CONFIG = {
        'inst': {'name': 'Instructions', 'color_scheme': 'Purples', 'base_color': 'purple'},
        'const': {'name': 'Constants', 'color_scheme': 'Blues', 'base_color': 'blue'},
        'data': {'name': 'Data', 'color_scheme': 'Oranges', 'base_color': 'orange'},
    }

    # Visualization style configuration (easy to customize)
    # Marker styles: 'o'=circle, 's'=square, '^'=triangle up, 'v'=triangle down,
    #                '>'=triangle right, 'D'=diamond, '*'=star, 'p'=pentagon
    START_MARKER = 's'        # Start point marker (square)
    START_MARKER_SIZE = 10    # Start point marker size
    MID_MARKER = 'o'          # Intermediate point marker (circle)
    MID_MARKER_SIZE = 6       # Intermediate point marker size
    END_MARKER = '*'          # End point marker (triangle down)
    END_MARKER_SIZE = 15      # End point marker size

    # Plotly marker styles: circle, square, diamond, cross, x, triangle-up, triangle-down, etc.
    START_MARKER_PLOTLY = 'square'
    START_MARKER_SIZE_PLOTLY = 12
    MID_MARKER_PLOTLY = 'circle'
    MID_MARKER_SIZE_PLOTLY = 8
    END_MARKER_PLOTLY = 'star'
    END_MARKER_SIZE_PLOTLY = 12

    LINE_WIDTH = 2.5          # Width of path lines
    ARROW_HEAD_WIDTH = 0.08   # Width of arrow head
    ARROW_HEAD_LENGTH = 0.06  # Length of arrow head
    ARROW_LINE_WIDTH = 1.0    # Width of arrow line
    OFFSET_UNIT = 0.06        # Base offset unit for parallel paths

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

        # Check for plotly (optional, for interactive visualization)
        self.plotly_available = False
        try:
            import plotly.graph_objects as go
            self.go = go
            self.plotly_available = True
        except ImportError:
            pass  # Plotly is optional

    def _classify_edge(self, edge):
        """
        Classify an edge into one of the three groups: inst, const, data.

        Parameters
        ----------
        edge : TensorEdge or NodeID
            The edge to classify

        Returns
        -------
        str
            One of 'inst', 'const', or 'data'
        """
        # Instruction edges are NodeID, not TensorEdge
        if not isinstance(edge, TensorEdge):
            return 'inst'

        # Get tensor type from TensorEdge
        tensor_type = edge.src_id.tensor_type if hasattr(edge.src_id, 'tensor_type') else None

        if tensor_type in self.CONST_TENSOR_TYPES:
            return 'const'
        elif tensor_type in self.DATA_TENSOR_TYPES:
            return 'data'
        else:
            # Unknown type - default to data (more conservative)
            debug_print(f"[TensorPathVisualizer] Unknown tensor_type '{tensor_type}', defaulting to 'data' group")
            return 'data'

    def _group_paths_by_category(self, noc_paths):
        """
        Group NoC paths by their category (inst, const, data).

        Parameters
        ----------
        noc_paths : dict
            Dictionary mapping edges to mapping_info

        Returns
        -------
        dict
            Dictionary mapping group name to dict of {edge: mapping_info}
        """
        groups = {'inst': {}, 'const': {}, 'data': {}}
        for edge, mapping_info in noc_paths.items():
            group = self._classify_edge(edge)
            groups[group][edge] = mapping_info
        return groups

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
        Creates separate images for:
        - Each tensor type (odata, weight, bias, etc.)
        - Each group category (inst, const, data)
        - Overview of all paths
        - Interactive HTML visualization (if plotly available)

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

        # Group NoC paths by tensor type (for individual tensor type images)
        paths_by_type = {}
        for edge, mapping_info in noc_paths.items():
            if isinstance(edge, TensorEdge):
                tensor_type = edge.src_id.tensor_type
                if tensor_type not in paths_by_type:
                    paths_by_type[tensor_type] = {}
                paths_by_type[tensor_type][edge] = mapping_info

        # Group NoC paths by category (inst, const, data)
        paths_by_group = self._group_paths_by_category(noc_paths)

        # Log group statistics
        debug_print(f"  Path groups: inst={len(paths_by_group['inst'])}, "
                   f"const={len(paths_by_group['const'])}, data={len(paths_by_group['data'])}")

        # ============================================================
        # 1. Create visualizations for each GROUP (inst, const, data)
        # ============================================================
        debug_print(f"  Creating group-based visualizations...")
        for group_name, group_paths in paths_by_group.items():
            if not group_paths:
                continue

            group_config = self.GROUP_CONFIG[group_name]
            debug_print(f"    Creating {group_name} ({group_config['name']}): {len(group_paths)} paths")

            fig, ax = self._create_mesh_grid(
                title=f"{func_name} - {group_config['name']} Paths ({len(group_paths)} paths)"
            )

            # Draw paths for this group
            self._draw_tensor_paths(ax, group_paths, tensor_edge_list)

            # Save the figure
            output_path = os.path.join(func_output_dir, f"group_{group_name}.png")
            self.plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.plt.close(fig)
            debug_print(f"      Saved: {output_path}")

        # ============================================================
        # 2. Create visualizations for each TENSOR TYPE
        # ============================================================
        if not paths_by_type:
            debug_print(f"No tensor edges found for function {func_name}")
        else:
            # Create individual visualizations for each tensor type
            for tensor_type, type_paths in sorted(paths_by_type.items()):
                debug_print(f"  Creating visualization for {tensor_type}: {len(type_paths)} paths")

                fig, ax = self._create_mesh_grid(title=f"{func_name} - {tensor_type} Paths")
                self._draw_tensor_paths(ax, type_paths, tensor_edge_list)

                output_path = os.path.join(func_output_dir, f"{tensor_type}.png")
                self.plt.savefig(output_path, dpi=300, bbox_inches='tight')
                self.plt.close(fig)
                debug_print(f"    Saved: {output_path}")

        # ============================================================
        # 3. Create overview image with ALL paths
        # ============================================================
        debug_print(f"  Creating overview with all paths")
        fig, ax = self._create_mesh_grid(title=f"{func_name} - All Paths (Overview)")

        self._draw_tensor_paths(ax, noc_paths, tensor_edge_list)
        
        overview_path = os.path.join(func_output_dir, "00_overview_all_paths.png")
        self.plt.savefig(overview_path, dpi=300, bbox_inches='tight')
        self.plt.close(fig)
        debug_print(f"    Saved: {overview_path}")

        # ============================================================
        # 4. Create INTERACTIVE visualization (HTML with Plotly)
        # ============================================================
        if self.plotly_available:
            debug_print(f"  Creating interactive HTML visualization...")
            self._create_interactive_visualization(func_name, noc_paths, paths_by_group, func_output_dir)
        else:
            debug_print(f"  Skipping interactive visualization (plotly not installed)")

        debug_print(f"Completed visualization for {func_name}: "
                   f"groups=(inst={len(paths_by_group['inst'])}, const={len(paths_by_group['const'])}, "
                   f"data={len(paths_by_group['data'])}), types={len(paths_by_type)}")
    
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
    
    def _normalize_segment(self, p1, p2):
        """
        Normalize a segment so that both directions map to the same key.
        Returns tuple of (smaller_point, larger_point) based on lexicographic order.
        """
        if (p1[0], p1[1]) <= (p2[0], p2[1]):
            return (p1, p2)
        return (p2, p1)

    def _get_path_coords(self, edge, mapping_info, node_size, spacing, rows):
        """
        Extract path coordinates for an edge.

        Returns:
            List of (x, y) coordinates, or None if no valid path
        """
        source_node = mapping_info[0]
        dest_node = mapping_info[1]

        # Try to get full path from router entries
        if isinstance(edge, TensorEdge) and edge in ImcflowDeviceConfig().TensorEdgetoInfo:
            edge_info = ImcflowDeviceConfig().TensorEdgetoInfo[edge]
            if edge_info.policy_info:
                path_nodes = [entry.router_id for entry in edge_info.policy_info]
                path_coords = []
                for node_id in path_nodes:
                    coord = NodeID.to_coord(node_id)
                    row, col = coord
                    x = col * (node_size + spacing) + spacing + node_size / 2
                    y = (rows - 1 - row) * (node_size + spacing) + spacing + node_size / 2
                    path_coords.append((x, y))
                return path_coords

        # Fallback: direct line from source to dest
        src_coord = NodeID.to_coord(source_node)
        dst_coord = NodeID.to_coord(dest_node)

        src_x = src_coord[1] * (node_size + spacing) + spacing + node_size / 2
        src_y = (rows - 1 - src_coord[0]) * (node_size + spacing) + spacing + node_size / 2
        dst_x = dst_coord[1] * (node_size + spacing) + spacing + node_size / 2
        dst_y = (rows - 1 - dst_coord[0]) * (node_size + spacing) + spacing + node_size / 2

        return [(src_x, src_y), (dst_x, dst_y)]

    def _draw_tensor_paths(self, ax, noc_paths, tensor_edge_list):
        """
        Draw tensor paths on the mesh grid using 2-pass algorithm for parallel offset.

        The algorithm ensures that overlapping paths are drawn as parallel lines
        (strictly horizontal/vertical) instead of tilted lines.

        Parameters
        ----------
        ax : matplotlib axis
            The axis to draw on
        noc_paths : dict
            Dictionary mapping edges to (source_node, dest_node, dest_index) tuples
        tensor_edge_list : list
            List of TensorEdge objects for this function
        """
        # Constants
        node_size = 1.0
        spacing = 0.5
        rows = ImcflowDeviceConfig.INODE_NUM

        # Get colors
        num_edges = len(noc_paths)
        colors = self._generate_colors(num_edges)

        # ============================================================
        # PASS 1: Extract all paths and collect segment usage
        # ============================================================
        path_data = []  # List of (edge, mapping_info, path_coords, color)
        segment_paths = {}  # normalized_segment -> [path_index, ...]

        edge_idx = 0
        for edge, mapping_info in noc_paths.items():
            path_coords = self._get_path_coords(edge, mapping_info, node_size, spacing, rows)

            if path_coords and len(path_coords) >= 2:
                color = colors[edge_idx % len(colors)]
                path_index = len(path_data)
                path_data.append((edge, mapping_info, path_coords, color))

                # Collect segments for this path
                for i in range(len(path_coords) - 1):
                    p1 = path_coords[i]
                    p2 = path_coords[i + 1]
                    seg = self._normalize_segment(p1, p2)

                    if seg not in segment_paths:
                        segment_paths[seg] = []
                    segment_paths[seg].append(path_index)

            edge_idx += 1

        # ============================================================
        # PASS 2 & 3: Greedy path-by-path drawing with consistent offset
        # Each path gets ONE consistent offset slot across all its segments
        # ============================================================
        legend_entries = []

        # Track which slots are claimed on each segment
        segment_claimed_slots = {seg: set() for seg in segment_paths.keys()}

        # ============================================================
        # Track markers per node to avoid star/square overlap
        # ============================================================
        # Collect all markers (start=square, end=star) per node
        node_start_markers = {}  # node_id -> list of path_idx
        node_end_markers = {}    # node_id -> list of path_idx

        for path_idx, (edge, mapping_info, path_coords, color) in enumerate(path_data):
            source_node = mapping_info[0]
            dest_node = mapping_info[1]

            # Track start marker at source node
            src_key = source_node.noc_placement if hasattr(source_node, 'noc_placement') else id(source_node)
            if src_key not in node_start_markers:
                node_start_markers[src_key] = []
            node_start_markers[src_key].append(path_idx)

            # Track end marker at dest node
            dst_key = dest_node.noc_placement if hasattr(dest_node, 'noc_placement') else id(dest_node)
            if dst_key not in node_end_markers:
                node_end_markers[dst_key] = []
            node_end_markers[dst_key].append(path_idx)

        # Assign marker slots per node (combining start and end markers)
        # This ensures stars and squares at the same node don't overlap
        node_marker_slots = {}  # (node_key, marker_type, path_idx) -> slot_index
        MARKER_OFFSET_UNIT = 0.08  # Offset unit for separating markers

        for node_key in set(node_start_markers.keys()) | set(node_end_markers.keys()):
            # Collect all markers at this node
            markers_at_node = []
            for path_idx in node_start_markers.get(node_key, []):
                markers_at_node.append(('start', path_idx))
            for path_idx in node_end_markers.get(node_key, []):
                markers_at_node.append(('end', path_idx))

            # Assign slots in a grid pattern around the node center
            for slot_idx, (marker_type, path_idx) in enumerate(markers_at_node):
                node_marker_slots[(node_key, marker_type, path_idx)] = slot_idx

        def get_marker_offset(node_key, marker_type, path_idx, base_offset):
            """Calculate additional marker offset to avoid overlap."""
            key = (node_key, marker_type, path_idx)
            if key not in node_marker_slots:
                return (0, 0)

            slot_idx = node_marker_slots[key]
            if slot_idx == 0:
                return (0, 0)

            # Spread markers in a grid pattern
            # slot 1: (+x, 0), slot 2: (-x, 0), slot 3: (0, +y), slot 4: (0, -y)
            # slot 5: (+x, +y), slot 6: (-x, +y), slot 7: (+x, -y), slot 8: (-x, -y)
            patterns = [
                (0, 0),
                (MARKER_OFFSET_UNIT, 0),
                (-MARKER_OFFSET_UNIT, 0),
                (0, MARKER_OFFSET_UNIT),
                (0, -MARKER_OFFSET_UNIT),
                (MARKER_OFFSET_UNIT, MARKER_OFFSET_UNIT),
                (-MARKER_OFFSET_UNIT, MARKER_OFFSET_UNIT),
                (MARKER_OFFSET_UNIT, -MARKER_OFFSET_UNIT),
                (-MARKER_OFFSET_UNIT, -MARKER_OFFSET_UNIT),
            ]

            if slot_idx < len(patterns):
                return patterns[slot_idx]
            else:
                # For more markers, use a circular pattern
                import math
                angle = (slot_idx - len(patterns)) * (2 * math.pi / 8)
                radius = MARKER_OFFSET_UNIT * (1 + (slot_idx - len(patterns)) // 8)
                return (radius * math.cos(angle), radius * math.sin(angle))

        # Get the path's segments helper
        def get_path_segments(path_coords):
            segments = []
            for i in range(len(path_coords) - 1):
                seg = self._normalize_segment(path_coords[i], path_coords[i + 1])
                segments.append(seg)
            return segments

        for path_idx, (edge, mapping_info, path_coords, color) in enumerate(path_data):
            source_node = mapping_info[0]
            dest_node = mapping_info[1]

            # Get all segments for this path
            path_segments = get_path_segments(path_coords)

            # Find a slot that's available on ALL segments of this path
            # Try slot 0, 1, 2, ... until we find one that works
            chosen_slot = 0
            max_possible_slot = max(len(segment_paths.get(seg, [])) for seg in path_segments) + len(path_data)

            for candidate_slot in range(max_possible_slot):
                # Check if this slot is available on all segments of this path
                available = True
                for seg in path_segments:
                    if candidate_slot in segment_claimed_slots.get(seg, set()):
                        available = False
                        break
                if available:
                    chosen_slot = candidate_slot
                    break

            # Claim this slot on all segments of this path
            for seg in path_segments:
                if seg not in segment_claimed_slots:
                    segment_claimed_slots[seg] = set()
                segment_claimed_slots[seg].add(chosen_slot)

            # Calculate the offset for this path (consistent across all segments)
            # We use a simple linear offset based on chosen_slot
            # Centered around 0: slot 0 -> 0, slot 1 -> +offset, slot 2 -> -offset, etc.
            if chosen_slot == 0:
                path_offset = 0
            elif chosen_slot % 2 == 1:
                path_offset = ((chosen_slot + 1) // 2) * self.OFFSET_UNIT
            else:
                path_offset = -(chosen_slot // 2) * self.OFFSET_UNIT

            # Build offset path with CONSISTENT offset for entire path
            # At corner points, apply BOTH X and Y offsets so lines meet orthogonally
            all_segments_coords = []

            # First, determine direction (horizontal/vertical) for each segment
            segment_directions = []  # True = horizontal, False = vertical
            for i in range(len(path_coords) - 1):
                p1 = path_coords[i]
                p2 = path_coords[i + 1]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                is_horizontal = abs(dx) > abs(dy)
                segment_directions.append(is_horizontal)

            # Build offset coordinates for each segment
            for i in range(len(path_coords) - 1):
                p1 = path_coords[i]
                p2 = path_coords[i + 1]
                is_horizontal = segment_directions[i]

                # Determine if there's a direction change at p1 (start of this segment)
                # and at p2 (end of this segment)
                prev_is_horizontal = segment_directions[i - 1] if i > 0 else is_horizontal
                next_is_horizontal = segment_directions[i + 1] if i < len(segment_directions) - 1 else is_horizontal

                # Apply offset at p1
                if i == 0:
                    # First point: only apply this segment's perpendicular offset
                    if is_horizontal:
                        offset_p1 = (p1[0], p1[1] + path_offset)
                    else:
                        offset_p1 = (p1[0] + path_offset, p1[1])
                elif prev_is_horizontal != is_horizontal:
                    # Corner at p1: apply BOTH X and Y offsets for orthogonal meeting
                    offset_p1 = (p1[0] + path_offset, p1[1] + path_offset)
                else:
                    # Same direction continues: apply only this segment's offset
                    if is_horizontal:
                        offset_p1 = (p1[0], p1[1] + path_offset)
                    else:
                        offset_p1 = (p1[0] + path_offset, p1[1])

                # Apply offset at p2
                if i == len(segment_directions) - 1:
                    # Last point: only apply this segment's perpendicular offset
                    if is_horizontal:
                        offset_p2 = (p2[0], p2[1] + path_offset)
                    else:
                        offset_p2 = (p2[0] + path_offset, p2[1])
                elif is_horizontal != next_is_horizontal:
                    # Corner at p2: apply BOTH X and Y offsets for orthogonal meeting
                    offset_p2 = (p2[0] + path_offset, p2[1] + path_offset)
                else:
                    # Same direction continues: apply only this segment's offset
                    if is_horizontal:
                        offset_p2 = (p2[0], p2[1] + path_offset)
                    else:
                        offset_p2 = (p2[0] + path_offset, p2[1])

                all_segments_coords.append([offset_p1, offset_p2, is_horizontal])

            # Draw segments (no connectors needed - corners meet at single point)
            first_line = None
            corner_points = []  # Points where direction changes (for intermediate markers)

            for seg_idx, seg_data in enumerate(all_segments_coords):
                offset_p1, offset_p2, is_horizontal = seg_data
                xs = [offset_p1[0], offset_p2[0]]
                ys = [offset_p1[1], offset_p2[1]]

                line = ax.plot(xs, ys, color=color, linewidth=self.LINE_WIDTH, alpha=0.8,
                              zorder=10)

                if first_line is None:
                    first_line = line[0]

                # Record corner point for intermediate marker if direction changes
                if seg_idx < len(all_segments_coords) - 1:
                    next_seg = all_segments_coords[seg_idx + 1]
                    next_is_horizontal = next_seg[2]

                    # If direction changes, the corner point is where segments meet
                    if is_horizontal != next_is_horizontal:
                        # offset_p2 and next_seg[0] should be the same point now
                        corner_points.append(offset_p2)

            # Draw markers with slot-based offsets to avoid overlap
            if all_segments_coords:
                # Get node keys for marker slot lookup
                src_key = source_node.noc_placement if hasattr(source_node, 'noc_placement') else id(source_node)
                dst_key = dest_node.noc_placement if hasattr(dest_node, 'noc_placement') else id(dest_node)

                # Start marker (first point of first segment) with slot offset
                start_pt = all_segments_coords[0][0]
                start_marker_offset = get_marker_offset(src_key, 'start', path_idx, path_offset)
                start_x = start_pt[0] + start_marker_offset[0]
                start_y = start_pt[1] + start_marker_offset[1]
                ax.plot(start_x, start_y, color=color,
                       marker=self.START_MARKER, markersize=self.START_MARKER_SIZE,
                       markeredgecolor='white', markeredgewidth=0.5, zorder=12)

                # Intermediate markers at corner points
                for pt in corner_points:
                    ax.plot(pt[0], pt[1], color=color,
                           marker=self.MID_MARKER, markersize=self.MID_MARKER_SIZE,
                           markeredgecolor='white', markeredgewidth=0.5, zorder=11)

                # End marker (last point of last segment) with slot offset
                end_pt = all_segments_coords[-1][1]
                end_marker_offset = get_marker_offset(dst_key, 'end', path_idx, path_offset)
                end_x = end_pt[0] + end_marker_offset[0]
                end_y = end_pt[1] + end_marker_offset[1]
                ax.plot(end_x, end_y, color=color,
                       marker=self.END_MARKER, markersize=self.END_MARKER_SIZE,
                       markeredgecolor='white', markeredgewidth=0.5, zorder=12)

            # Add arrow at the last segment
            if all_segments_coords:
                last_seg = all_segments_coords[-1]
                p1, p2 = last_seg[0], last_seg[1]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length = (dx ** 2 + dy ** 2) ** 0.5

                if length > 0:
                    # Draw arrow from 50% to 90% of the segment (not reaching end marker)
                    arrow_start_x = p1[0] + dx * 0.5
                    arrow_start_y = p1[1] + dy * 0.5
                    ax.arrow(arrow_start_x, arrow_start_y, dx * 0.3, dy * 0.3,
                            head_width=self.ARROW_HEAD_WIDTH, head_length=self.ARROW_HEAD_LENGTH,
                            fc=color, ec=color, alpha=0.9, linewidth=self.ARROW_LINE_WIDTH, zorder=11)

            # Create legend entry
            if first_line is not None:
                if isinstance(edge, TensorEdge):
                    src_node_name = source_node.name
                    dst_node_name = dest_node.name
                    tensor_type = edge.src_id.tensor_type

                    src_custom_id = edge.src_id.graph_node_id
                    dst_custom_id = edge.dst_id.graph_node_id

                    src_id_str = f"{src_custom_id[1]}" if isinstance(src_custom_id, tuple) else f"{src_custom_id}"
                    dst_id_str = f"{dst_custom_id[1]}" if isinstance(dst_custom_id, tuple) else f"{dst_custom_id}"

                    tensor_label = f"{src_node_name} → {dst_node_name} | ID:{src_id_str}→{dst_id_str} ({tensor_type})"
                    if edge.split_idx is not None:
                        tensor_label += f"[{edge.split_idx}]"
                else:
                    # Instruction edge
                    tensor_label = f"{source_node.name} → {dest_node.name} (inst)"

                legend_entries.append((first_line, tensor_label))

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

    def _create_interactive_visualization(self, func_name, noc_paths, paths_by_group, output_dir):
        """
        Create an interactive HTML visualization using Plotly.

        Features:
        - Click on a path to highlight it and see detailed information
        - Hover tooltips showing edge info
        - Filter by group (inst, const, data)
        - Pan and zoom

        Parameters
        ----------
        func_name : str
            Name of the function being visualized
        noc_paths : dict
            Dictionary mapping edges to mapping_info
        paths_by_group : dict
            Dictionary mapping group names to their paths
        output_dir : str
            Output directory for the HTML file
        """
        if not self.plotly_available:
            return

        import plotly.graph_objects as go

        # Grid dimensions
        rows = ImcflowDeviceConfig.INODE_NUM
        cols = ImcflowDeviceConfig.NODE_COL_NUM
        node_size = 1.0
        spacing = 0.5

        fig = go.Figure()

        # ============================================================
        # Draw mesh nodes
        # ============================================================
        node_x = []
        node_y = []
        node_text = []
        node_colors = []

        for node_id in NodeID:
            coord = NodeID.to_coord(node_id)
            row, col = coord
            x = col * (node_size + spacing) + spacing + node_size / 2
            y = (rows - 1 - row) * (node_size + spacing) + spacing + node_size / 2
            node_x.append(x)
            node_y.append(y)
            node_text.append(node_id.name)
            node_colors.append('lightblue' if node_id.is_inode() else 'lightgreen')

        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(size=40, color=node_colors, line=dict(width=2, color='darkgray')),
            text=node_text,
            textposition='middle center',
            textfont=dict(size=10, color='black'),
            hoverinfo='text',
            hovertext=[f"Node: {t}" for t in node_text],
            name='Nodes',
            showlegend=False
        ))

        # ============================================================
        # Draw paths using 2-pass algorithm for parallel offsets
        # ============================================================
        group_colors = {
            'inst': 'purple',
            'const': 'blue',
            'data': 'orange'
        }

        # PASS 1: Collect all paths and their segments
        path_data = []  # List of (edge, mapping_info, path_coords, group_name, edge_info)
        segment_paths = {}  # normalized_segment -> [path_index, ...]

        for group_name, group_paths in paths_by_group.items():
            if not group_paths:
                continue

            group_config = self.GROUP_CONFIG[group_name]

            for edge, mapping_info in group_paths.items():
                source_node = mapping_info[0]
                dest_node = mapping_info[1]

                # Build edge info string
                if isinstance(edge, TensorEdge):
                    tensor_type = edge.src_id.tensor_type
                    src_custom_id = edge.src_id.graph_node_id
                    dst_custom_id = edge.dst_id.graph_node_id
                    src_id_str = f"{src_custom_id[1]}" if isinstance(src_custom_id, tuple) else f"{src_custom_id}"
                    dst_id_str = f"{dst_custom_id[1]}" if isinstance(dst_custom_id, tuple) else f"{dst_custom_id}"
                    edge_info = (f"Group: {group_config['name']}<br>"
                                f"Type: {tensor_type}<br>"
                                f"HW: {source_node.name} → {dest_node.name}<br>"
                                f"Graph ID: {src_id_str} → {dst_id_str}")
                    if edge.split_idx is not None:
                        edge_info += f"<br>Split: [{edge.split_idx}]"
                else:
                    edge_info = (f"Group: {group_config['name']}<br>"
                                f"Type: Instruction<br>"
                                f"HW: {source_node.name} → {dest_node.name}")

                # Get path coordinates
                path_coords = self._get_path_coords(edge, mapping_info, node_size, spacing, rows)

                if path_coords and len(path_coords) >= 2:
                    path_index = len(path_data)
                    path_data.append((edge, mapping_info, path_coords, group_name, edge_info))

                    # Collect segments
                    for i in range(len(path_coords) - 1):
                        p1 = path_coords[i]
                        p2 = path_coords[i + 1]
                        seg = self._normalize_segment(p1, p2)
                        if seg not in segment_paths:
                            segment_paths[seg] = []
                        segment_paths[seg].append(path_index)

        # PASS 2 & 3: Greedy path-by-path drawing with consistent offset
        all_path_data = []
        group_first_drawn = {'inst': False, 'const': False, 'data': False}

        # Track which slots are claimed on each segment
        segment_claimed_slots = {seg: set() for seg in segment_paths.keys()}

        # ============================================================
        # Track markers per node to avoid star/square overlap (Plotly)
        # ============================================================
        node_start_markers_plotly = {}  # node_id -> list of path_idx
        node_end_markers_plotly = {}    # node_id -> list of path_idx

        for path_idx, (edge, mapping_info, path_coords, group_name, edge_info) in enumerate(path_data):
            source_node = mapping_info[0]
            dest_node = mapping_info[1]

            src_key = source_node.noc_placement if hasattr(source_node, 'noc_placement') else id(source_node)
            if src_key not in node_start_markers_plotly:
                node_start_markers_plotly[src_key] = []
            node_start_markers_plotly[src_key].append(path_idx)

            dst_key = dest_node.noc_placement if hasattr(dest_node, 'noc_placement') else id(dest_node)
            if dst_key not in node_end_markers_plotly:
                node_end_markers_plotly[dst_key] = []
            node_end_markers_plotly[dst_key].append(path_idx)

        node_marker_slots_plotly = {}
        MARKER_OFFSET_UNIT_PLOTLY = 0.08

        for node_key in set(node_start_markers_plotly.keys()) | set(node_end_markers_plotly.keys()):
            markers_at_node = []
            for path_idx in node_start_markers_plotly.get(node_key, []):
                markers_at_node.append(('start', path_idx))
            for path_idx in node_end_markers_plotly.get(node_key, []):
                markers_at_node.append(('end', path_idx))

            for slot_idx, (marker_type, path_idx) in enumerate(markers_at_node):
                node_marker_slots_plotly[(node_key, marker_type, path_idx)] = slot_idx

        def get_marker_offset_plotly(node_key, marker_type, path_idx):
            key = (node_key, marker_type, path_idx)
            if key not in node_marker_slots_plotly:
                return (0, 0)

            slot_idx = node_marker_slots_plotly[key]
            if slot_idx == 0:
                return (0, 0)

            patterns = [
                (0, 0),
                (MARKER_OFFSET_UNIT_PLOTLY, 0),
                (-MARKER_OFFSET_UNIT_PLOTLY, 0),
                (0, MARKER_OFFSET_UNIT_PLOTLY),
                (0, -MARKER_OFFSET_UNIT_PLOTLY),
                (MARKER_OFFSET_UNIT_PLOTLY, MARKER_OFFSET_UNIT_PLOTLY),
                (-MARKER_OFFSET_UNIT_PLOTLY, MARKER_OFFSET_UNIT_PLOTLY),
                (MARKER_OFFSET_UNIT_PLOTLY, -MARKER_OFFSET_UNIT_PLOTLY),
                (-MARKER_OFFSET_UNIT_PLOTLY, -MARKER_OFFSET_UNIT_PLOTLY),
            ]

            if slot_idx < len(patterns):
                return patterns[slot_idx]
            else:
                import math
                angle = (slot_idx - len(patterns)) * (2 * math.pi / 8)
                radius = MARKER_OFFSET_UNIT_PLOTLY * (1 + (slot_idx - len(patterns)) // 8)
                return (radius * math.cos(angle), radius * math.sin(angle))

        def get_path_segments(path_coords):
            segments = []
            for i in range(len(path_coords) - 1):
                seg = self._normalize_segment(path_coords[i], path_coords[i + 1])
                segments.append(seg)
            return segments

        for path_idx, (edge, mapping_info, path_coords, group_name, edge_info) in enumerate(path_data):
            source_node = mapping_info[0]
            dest_node = mapping_info[1]
            base_color = group_colors.get(group_name, 'gray')
            group_config = self.GROUP_CONFIG[group_name]

            # Get all segments for this path
            path_segments = get_path_segments(path_coords)

            # Find a slot that's available on ALL segments of this path
            chosen_slot = 0
            max_possible_slot = max(len(segment_paths.get(seg, [])) for seg in path_segments) + len(path_data)

            for candidate_slot in range(max_possible_slot):
                available = True
                for seg in path_segments:
                    if candidate_slot in segment_claimed_slots.get(seg, set()):
                        available = False
                        break
                if available:
                    chosen_slot = candidate_slot
                    break

            # Claim this slot on all segments
            for seg in path_segments:
                if seg not in segment_claimed_slots:
                    segment_claimed_slots[seg] = set()
                segment_claimed_slots[seg].add(chosen_slot)

            # Calculate consistent offset for this path
            if chosen_slot == 0:
                path_offset = 0
            elif chosen_slot % 2 == 1:
                path_offset = ((chosen_slot + 1) // 2) * self.OFFSET_UNIT
            else:
                path_offset = -(chosen_slot // 2) * self.OFFSET_UNIT

            # Build offset coordinates with CONSISTENT offset and corner handling
            all_segments = []  # List of (offset_p1, offset_p2, is_horizontal)

            for i in range(len(path_coords) - 1):
                p1 = path_coords[i]
                p2 = path_coords[i + 1]

                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]

                # Apply perpendicular offset (same for entire path)
                if abs(dx) > abs(dy):
                    offset_p1 = (p1[0], p1[1] + path_offset)
                    offset_p2 = (p2[0], p2[1] + path_offset)
                    is_horizontal = True
                else:
                    offset_p1 = (p1[0] + path_offset, p1[1])
                    offset_p2 = (p2[0] + path_offset, p2[1])
                    is_horizontal = False

                all_segments.append((offset_p1, offset_p2, is_horizontal))

            # Build continuous path with corner connectors
            offset_xs = []
            offset_ys = []
            corner_points = []

            for seg_idx, (offset_p1, offset_p2, is_horizontal) in enumerate(all_segments):
                if seg_idx == 0:
                    offset_xs.append(offset_p1[0])
                    offset_ys.append(offset_p1[1])

                offset_xs.append(offset_p2[0])
                offset_ys.append(offset_p2[1])

                # Add corner connector if direction changes
                if seg_idx < len(all_segments) - 1:
                    next_seg = all_segments[seg_idx + 1]
                    next_is_horizontal = next_seg[2]

                    if is_horizontal != next_is_horizontal:
                        # Add corner point (start of next segment)
                        offset_xs.append(next_seg[0][0])
                        offset_ys.append(next_seg[0][1])
                        # Track corner midpoint for marker
                        corner_pt = ((offset_p2[0] + next_seg[0][0]) / 2,
                                    (offset_p2[1] + next_seg[0][1]) / 2)
                        corner_points.append(corner_pt)

            # Determine if this is the first path in this group
            is_first = not group_first_drawn[group_name]
            if is_first:
                group_first_drawn[group_name] = True

            # Draw line (no markers)
            fig.add_trace(go.Scatter(
                x=offset_xs,
                y=offset_ys,
                mode='lines',
                line=dict(color=base_color, width=2),
                opacity=0.6,
                hoverinfo='text',
                hovertext=edge_info,
                name=f"{group_config['name']}: {source_node.name}→{dest_node.name}",
                legendgroup=group_name,
                showlegend=is_first,
                legendgrouptitle_text=group_config['name'] if is_first else None
            ))

            # Draw start marker with slot-based offset
            if offset_xs:
                src_key = source_node.noc_placement if hasattr(source_node, 'noc_placement') else id(source_node)
                start_marker_off = get_marker_offset_plotly(src_key, 'start', path_idx)
                start_x = offset_xs[0] + start_marker_off[0]
                start_y = offset_ys[0] + start_marker_off[1]
                fig.add_trace(go.Scatter(
                    x=[start_x],
                    y=[start_y],
                    mode='markers',
                    marker=dict(symbol=self.START_MARKER_PLOTLY, size=self.START_MARKER_SIZE_PLOTLY,
                               color=base_color, line=dict(width=1, color='white')),
                    opacity=0.8,
                    hoverinfo='text',
                    hovertext=f"START<br>{edge_info}",
                    legendgroup=group_name,
                    showlegend=False
                ))

            # Draw intermediate markers at corner points
            if corner_points:
                corner_xs = [pt[0] for pt in corner_points]
                corner_ys = [pt[1] for pt in corner_points]
                fig.add_trace(go.Scatter(
                    x=corner_xs,
                    y=corner_ys,
                    mode='markers',
                    marker=dict(symbol=self.MID_MARKER_PLOTLY, size=self.MID_MARKER_SIZE_PLOTLY,
                               color=base_color, line=dict(width=1, color='white')),
                    opacity=0.6,
                    hoverinfo='text',
                    hovertext=edge_info,
                    legendgroup=group_name,
                    showlegend=False
                ))

            # Draw end marker with slot-based offset
            if len(offset_xs) > 1:
                dst_key = dest_node.noc_placement if hasattr(dest_node, 'noc_placement') else id(dest_node)
                end_marker_off = get_marker_offset_plotly(dst_key, 'end', path_idx)
                end_x = offset_xs[-1] + end_marker_off[0]
                end_y = offset_ys[-1] + end_marker_off[1]
                fig.add_trace(go.Scatter(
                    x=[end_x],
                    y=[end_y],
                    mode='markers',
                    marker=dict(symbol=self.END_MARKER_PLOTLY, size=self.END_MARKER_SIZE_PLOTLY,
                               color=base_color, line=dict(width=1, color='white')),
                    opacity=0.8,
                    hoverinfo='text',
                    hovertext=f"END<br>{edge_info}",
                    legendgroup=group_name,
                    showlegend=False
                ))

            all_path_data.append({
                'group': group_name,
                'edge': str(edge),
                'src': source_node.name,
                'dst': dest_node.name,
                'info': edge_info
            })

        # ============================================================
        # Layout configuration
        # ============================================================
        fig_width = cols * (node_size + spacing) + spacing
        fig_height = rows * (node_size + spacing) + spacing

        fig.update_layout(
            title=dict(
                text=f"{func_name} - Interactive NoC Path Visualization<br>"
                     f"<sup>Groups: inst={len(paths_by_group['inst'])}, "
                     f"const={len(paths_by_group['const'])}, data={len(paths_by_group['data'])}</sup>",
                x=0.5,
                font=dict(size=16)
            ),
            xaxis=dict(
                range=[0, fig_width + 0.5],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor='y',
                scaleratio=1
            ),
            yaxis=dict(
                range=[0, fig_height + 0.5],
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            hovermode='closest',
            showlegend=True,
            legend=dict(
                title="Path Groups (click to toggle)",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="gray",
                borderwidth=1
            ),
            width=900,
            height=700,
            margin=dict(r=200)
        )

        # Add instructions annotation
        fig.add_annotation(
            x=0.5, y=-0.1,
            xref='paper', yref='paper',
            text="Hover over paths for details. Click legend items to filter groups.",
            showarrow=False,
            font=dict(size=11, color='gray')
        )

        # Save as HTML
        output_path = os.path.join(output_dir, "interactive.html")
        fig.write_html(output_path, include_plotlyjs=True, full_html=True)
        debug_print(f"    Saved interactive visualization: {output_path}")


def generateNoCVisualizations(mod, output_dir="noc_visualizations"):
    """
    Generate NoC path visualizations for all imcflow functions in the module.

    This function should be called after PolicyTableGenerator has run and
    populated ImcflowDeviceConfig with NoC paths and tensor edge information.

    For each imcflow function, creates:
    - A subdirectory named after the function
    - Group-based images (group_inst.png, group_const.png, group_data.png)
    - Tensor type images (odata.png, weight.png, bias.png, etc.)
    - An overview image showing all paths (00_overview_all_paths.png)
    - Interactive HTML visualization (interactive.html) - requires plotly

    Groups:
    - inst: Instruction paths (non-TensorEdge)
    - const: Constant tensors (weight, config, min, max, fused_scale, fused_bias, bias, scale, threshold)
    - data: Runtime data tensors (odata, data, lhs, rhs, func_out*, var)

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
            00_overview_all_paths.png    # All paths overview
            group_inst.png               # Instruction paths only
            group_const.png              # Constant tensor paths only
            group_data.png               # Data tensor paths only
            interactive.html             # Interactive visualization (plotly)
            odata.png                    # Individual tensor type
            weight.png
            bias.png
            ...
        function_name_2/
            ...
    
    Example
    -------
    >>> # After running PolicyTableGenerator
    >>> generateNoCVisualizations(mod, "my_visualizations")
    """
    visualizer = TensorPathVisualizer(output_dir=output_dir)
    visualizer.visualize_all_functions(mod)
    debug_print(f"NoC visualizations saved to: {output_dir}")

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
      self.layout_map = ImcflowDeviceConfig().LayoutMap

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
      new_call = _expr.CallWithFields(new_call, new_call.op, new_call.args, new_attrs, new_call.type_args, new_call.span)
      if call in self.layout_map:
        self.layout_map[new_call] = self.layout_map[call]
      return new_call

    def visit_function(self, fn):
      new_fn = super().visit_function(fn)
      self.cnt = self.cnt + 1
      origin_attrs = new_fn.attrs
      new_attrs = self.update_attrs(origin_attrs, {"custom_id": self.cnt})
      new_fn = FunctionWithFields(new_fn, list(new_fn.params), new_fn.body, new_fn.ret_type, new_fn.type_params, new_attrs)
      if fn in self.layout_map:
        self.layout_map[new_fn] = self.layout_map[fn]
      return new_fn

  visitor = _Visitor()
  # Sort functions by name to ensure deterministic order
  for gv, func in sorted(mod.functions.items(), key=lambda x: x[0].name_hint):
    mod[gv] = visitor.visit(func)

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
  # Sort functions by name to ensure deterministic order
  for gv, func in sorted(mod.functions.items(), key=lambda x: x[0].name_hint):
    vis.visit(func)

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

def constructSplitInfo(mod):
  """Construct SplitInfo by analyzing split nodes and their downstream consumers.

  This function should be called after constructUsefulMappings() and constructCustomIDInFunc().
  It finds split nodes in the graph and determines:
  - is_multi_cast: True for normal conv (same data to multiple IC slices), False for DW conv
  - channels: Number of input channels being split
  - num_splits: Number of splits

  The is_multi_cast is determined by looking at the downstream conv2d/qdwconv nodes.
  """
  config = ImcflowDeviceConfig()
  id_in_func = CustomIDInFunc()
  id_to_node = CustomIDToNode()

  # Clear existing SplitInfo
  config.SplitInfo = {}

  class SplitInfoVisitor(tvm.relay.ExprVisitor):
    def __init__(self, func_name):
      super().__init__()
      self.func_name = func_name
      self.split_consumers = {}  # split_custom_id -> list of consumer nodes

    def visit_call(self, call):
      super().visit_call(call)

      # Check if this is a split node
      if isinstance(call.op, tvm.ir.Op) and call.op.name == "split":
        split_id = int(call.attrs.custom_id)
        if split_id not in self.split_consumers:
          self.split_consumers[split_id] = []

    def visit_tuple_getitem(self, tgi):
      super().visit_tuple_getitem(tgi)

      # Check if this TupleGetItem's tuple is a split
      if isinstance(tgi.tuple_value, relay.Call):
        if isinstance(tgi.tuple_value.op, tvm.ir.Op) and tgi.tuple_value.op.name == "split":
          split_id = int(tgi.tuple_value.attrs.custom_id)
          if split_id not in self.split_consumers:
            self.split_consumers[split_id] = []
          # We'll find consumers in a second pass

  class ConsumerFinder(tvm.relay.ExprVisitor):
    """Find what consumes each TupleGetItem from a split."""
    def __init__(self, split_ids):
      super().__init__()
      self.split_ids = split_ids
      self.tgi_to_split = {}  # id(tgi) -> split_id
      self.split_to_consumers = {sid: [] for sid in split_ids}

    def visit_tuple_getitem(self, tgi):
      super().visit_tuple_getitem(tgi)
      if isinstance(tgi.tuple_value, relay.Call):
        if isinstance(tgi.tuple_value.op, tvm.ir.Op) and tgi.tuple_value.op.name == "split":
          split_id = int(tgi.tuple_value.attrs.custom_id)
          if split_id in self.split_ids:
            self.tgi_to_split[hash(tgi)] = split_id

    def visit_call(self, call):
      super().visit_call(call)
      # Check if any argument is a TupleGetItem from our splits
      for arg in call.args:
        if isinstance(arg, relay.TupleGetItem) and hash(arg) in self.tgi_to_split:
          split_id = self.tgi_to_split[hash(arg)]
          self.split_to_consumers[split_id].append(call)

  for func_name in mod.functions:
    if "imcflow" not in func_name.name_hint:
      continue

    func = mod[func_name.name_hint]

    # First pass: find all split nodes
    visitor = SplitInfoVisitor(func_name.name_hint)
    visitor.visit(func)

    if not visitor.split_consumers:
      continue

    # Second pass: find consumers of each split
    consumer_finder = ConsumerFinder(visitor.split_consumers.keys())
    consumer_finder.visit(func)

    # Analyze each split
    for split_id, _ in visitor.split_consumers.items():
      split_node = id_to_node.get(split_id)
      if split_node is None:
        continue

      consumers = consumer_finder.split_to_consumers.get(split_id, [])

      # Determine if depthwise by checking consumer conv nodes
      is_depthwise = False
      for consumer in consumers:
        if isinstance(consumer.op, tvm.ir.Op):
          op_name = consumer.op.name
          if "qdwconv" in op_name or "dwconv" in op_name:
            is_depthwise = True
            break
          elif "qconv" in op_name or "conv2d" in op_name:
            is_depthwise = False
            break
        elif isinstance(consumer.op, relay.Function):
          # Check Composite attribute for composite functions
          if hasattr(consumer.op.attrs, "Composite"):
            composite_name = str(consumer.op.attrs["Composite"])
            if "qdwconv" in composite_name or "dwconv" in composite_name:
              is_depthwise = True
              break
            elif "qconv" in composite_name and "dwconv" not in composite_name:
              is_depthwise = False
              break

      # Get split info from the split node
      num_splits = len(split_node.attrs.indices_or_sections) + 1 if hasattr(split_node.attrs, 'indices_or_sections') else 2

      # Get input channels from split input shape
      channels = 0
      input_type = _get_type(mod, split_node.args[0])
      channels = int(input_type.shape[1])

      # Store split info
      if func_name.name_hint not in config.SplitInfo:
        config.SplitInfo[func_name.name_hint] = {}

      config.SplitInfo[func_name.name_hint][split_id] = {
        'is_multi_cast': not is_depthwise,
        'channels': channels,
        'num_splits': num_splits
      }
      debug_print(f"[constructSplitInfo] {func_name.name_hint}:{split_id} - is_multi_cast={not is_depthwise}, channels={channels}, num_splits={num_splits}")

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

def constructDataBlockDict(mod, update_compiled_blocks_only=False):
  imcflow_func_map = ImcflowDeviceConfig().ImcflowFuncMap
  for func_name_var, func in mod.functions.items():
    if func_name_var.name_hint == "main": continue
    elif isinstance(func, relay.Function) and hasattr(func.attrs, "Compiler") and func.attrs["Compiler"]=="imcflow" :
      if update_compiled_blocks_only:
        ImcflowDeviceConfig().update_compiled_blocks(func_name_var.name_hint)
      else:
        target_func = imcflow_func_map[func_name_var.name_hint]
        input_node_ids = [getNodeID(n) for n in getInputNodesOfFunc(target_func.func_node)]
        output_node_id = getNodeID(target_func.func_node)
        const_node_ids = [getNodeID(n) for n in getConstNodesOfFunc(target_func.func_node)]
        ImcflowDeviceConfig().update_compiled_blocks(func_name_var.name_hint)
        ImcflowDeviceConfig().update_data_blocks(func_name_var.name_hint, input_node_ids, output_node_id, const_node_ids)

class FIFOConflictMonitor:
    """
    Monitor and detect FIFO ID conflicts where multiple tensor edges
    are assigned to the same FIFO ID for a destination node.
    
    This should be executed after PolicyTableGenerator has assigned
    FIFO IDs to all tensor edges.
    
    The monitor builds a conflict table that records:
    - Destination tensor custom ID
    - Destination HW node ID
    - FIFO ID that has conflicts
    - List of conflicting tensor edges with their source information
    """
    
    def __init__(self):
        self.conflict_table = {}  # {func_name: [conflict_entries]}
    
    def run(self, mod):
        """
        Analyze all imcflow functions and detect FIFO ID conflicts.
        
        Parameters
        ----------
        mod : tvm.IRModule
            The module containing imcflow functions
            
        Returns
        -------
        dict
            Conflict table with structure:
            {
              func_name: [
                {
                  'dst_custom_id': int,
                  'dst_hw_node': NodeID,
                  'fifo_id': int,
                  'conflicting_edges': [
                    {
                      'src_custom_id': int,
                      'src_hw_node': NodeID,
                      'tensor_type': str,
                      'edge': TensorEdge
                    },
                    ...
                  ]
                },
                ...
              ],
              ...
            }
        """
        imcflow_func_map = ImcflowDeviceConfig().ImcflowFuncMap
        tensor_edge_to_info = ImcflowDeviceConfig().TensorEdgetoInfo
        hw_node_map = ImcflowDeviceConfig().HWNodeMap
        custom_id_to_name = CustomIDToName()
        
        for gv, func in mod.functions.items():
            if not (isinstance(func, relay.Function) and 
                   hasattr(func.attrs, "Compiler") and 
                   func.attrs["Compiler"] == "imcflow"):
                continue
            
            func_name = gv.name_hint
            self.conflict_table[func_name] = []
            
            # Get tensor edges for this function
            if func_name not in ImcflowDeviceConfig().TensorEdgeListDict:
                continue
            
            tensor_edge_list = ImcflowDeviceConfig().TensorEdgeListDict[func_name]
            
            # Group tensor edges by destination node and FIFO ID
            # Structure: {dst_tensor_id: {fifo_id: [edge_info_list]}}
            # Note: We use the actual tensor ID (second element if tuple) as the key
            dst_fifo_map = {}
            
            for edge in tensor_edge_list:
                if not isinstance(edge, TensorEdge):
                    continue
                
                # Get edge info to access FIFO ID
                edge_info = tensor_edge_to_info.get(edge, None)
                if edge_info is None:
                    debug_print(f"Warning: No edge info found for edge {edge}")
                    continue
                
                fifo_id = edge_info.fifo_id
                if fifo_id == -1:
                    # FIFO ID not assigned, skip
                    continue
                
                # Extract the actual destination tensor ID
                # If dst_id.graph_node_id is a tuple (composite_id, tensor_id), use tensor_id
                # Otherwise, use the graph_node_id directly
                dst_graph_node_id = edge.dst_id.graph_node_id
                if isinstance(dst_graph_node_id, tuple):
                    # Use the second element (actual tensor ID) as the grouping key
                    dst_tensor_key = dst_graph_node_id[1]
                else:
                    dst_tensor_key = dst_graph_node_id
                
                # Initialize nested dict if needed
                if dst_tensor_key not in dst_fifo_map:
                    dst_fifo_map[dst_tensor_key] = {
                        'dst_tensor_id': edge.dst_id,  # Store the full TensorID for later use
                        'fifo_map': {}
                    }
                
                if fifo_id not in dst_fifo_map[dst_tensor_key]['fifo_map']:
                    dst_fifo_map[dst_tensor_key]['fifo_map'][fifo_id] = []
                
                # Store edge information
                src_custom_id = getInnerNodeID(edge.src_id.graph_node_id)
                dst_custom_id = getInnerNodeID(edge.dst_id.graph_node_id)
                
                # Get HW node IDs
                src_hw_node = hw_node_map.get(src_custom_id, None)
                dst_hw_node = hw_node_map.get(dst_custom_id, None)
                
                edge_data = {
                    'src_custom_id': src_custom_id,
                    'src_hw_node': src_hw_node,
                    'src_name': custom_id_to_name.get(src_custom_id, "unknown"),
                    'tensor_type': edge.src_id.tensor_type,
                    'edge': edge
                }
                
                dst_fifo_map[dst_tensor_key]['fifo_map'][fifo_id].append(edge_data)
            
            # Detect conflicts: FIFO IDs with multiple edges
            for dst_tensor_key, dst_info in dst_fifo_map.items():
                dst_tensor_id = dst_info['dst_tensor_id']
                fifo_dict = dst_info['fifo_map']
                dst_custom_id = getInnerNodeID(dst_tensor_id.graph_node_id)
                dst_hw_node = hw_node_map.get(dst_custom_id, None)
                dst_name = custom_id_to_name.get(dst_custom_id, "unknown")
                
                for fifo_id, edge_list in fifo_dict.items():
                    if len(edge_list) > 1:
                        # Conflict detected!
                        conflict_entry = {
                            'dst_custom_id': dst_custom_id,
                            'dst_name': dst_name,
                            'dst_hw_node': dst_hw_node,
                            'dst_tensor_id': dst_tensor_id,
                            'fifo_id': fifo_id,
                            'num_conflicts': len(edge_list),
                            'conflicting_edges': edge_list
                        }
                        
                        self.conflict_table[func_name].append(conflict_entry)
                        
                        debug_print(f"[FIFO Conflict] Function: {func_name}")
                        debug_print(f"  Destination: {dst_name} (CustomID: {dst_custom_id}, HW: {dst_hw_node})")
                        debug_print(f"  FIFO ID: {fifo_id} has {len(edge_list)} edges:")
                        for edge_data in edge_list:
                            debug_print(f"    - Source: {edge_data['src_name']} "
                                      f"(CustomID: {edge_data['src_custom_id']}, "
                                      f"HW: {edge_data['src_hw_node']}, "
                                      f"Type: {edge_data['tensor_type']})")
        
        # Store in device config for later access
        ImcflowDeviceConfig().FIFOConflictTable = self.conflict_table
        
        return self.conflict_table
    
    def print_conflict_summary(self):
        """
        Print a summary of all detected FIFO conflicts.
        """
        total_conflicts = sum(len(conflicts) for conflicts in self.conflict_table.values())
        
        if total_conflicts == 0:
            print("\n" + "="*60)
            print("FIFO Conflict Monitor: No conflicts detected!")
            print("="*60)
            return
        
        print("\n" + "="*60)
        print(f"FIFO Conflict Monitor: {total_conflicts} conflict(s) detected")
        print("="*60)
        
        for func_name, conflicts in self.conflict_table.items():
            if not conflicts:
                continue
            
            print(f"\nFunction: {func_name}")
            print(f"  Number of conflicts: {len(conflicts)}")
            
            for i, conflict in enumerate(conflicts, 1):
                print(f"\n  Conflict #{i}:")
                print(f"    Destination Node: {conflict['dst_name']} (CustomID: {conflict['dst_custom_id']})")
                print(f"    HW Node: {conflict['dst_hw_node']}")
                print(f"    FIFO ID: {conflict['fifo_id']}")
                print(f"    Number of overlapping edges: {conflict['num_conflicts']}")
                print(f"    Conflicting sources:")
                
                for j, edge_data in enumerate(conflict['conflicting_edges'], 1):
                    print(f"      {j}. {edge_data['src_name']} "
                          f"(CustomID: {edge_data['src_custom_id']}, "
                          f"HW: {edge_data['src_hw_node']}, "
                          f"Type: {edge_data['tensor_type']})")
        
        print("\n" + "="*60)
    
    def export_conflict_table(self, output_path):
        """
        Export the conflict table to a text file.
        
        Parameters
        ----------
        output_path : str
            Path to the output file
        """
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("FIFO Conflict Monitor Report\n")
            f.write("="*80 + "\n\n")
            
            total_conflicts = sum(len(conflicts) for conflicts in self.conflict_table.values())
            f.write(f"Total conflicts detected: {total_conflicts}\n\n")
            
            for func_name, conflicts in self.conflict_table.items():
                if not conflicts:
                    f.write(f"Function: {func_name}\n")
                    f.write("  No conflicts\n\n")
                    continue
                
                f.write(f"Function: {func_name}\n")
                f.write(f"  Number of conflicts: {len(conflicts)}\n")
                
                for i, conflict in enumerate(conflicts, 1):
                    f.write(f"\n  Conflict #{i}:\n")
                    f.write(f"    Destination Node: {conflict['dst_name']} (CustomID: {conflict['dst_custom_id']})\n")
                    f.write(f"    HW Node: {conflict['dst_hw_node']}\n")
                    f.write(f"    Destination Tensor ID: {conflict['dst_tensor_id']}\n")
                    f.write(f"    FIFO ID: {conflict['fifo_id']}\n")
                    f.write(f"    Number of overlapping edges: {conflict['num_conflicts']}\n")
                    f.write(f"    Conflicting sources:\n")
                    
                    for j, edge_data in enumerate(conflict['conflicting_edges'], 1):
                        f.write(f"      {j}. {edge_data['src_name']}\n")
                        f.write(f"         CustomID: {edge_data['src_custom_id']}\n")
                        f.write(f"         HW Node: {edge_data['src_hw_node']}\n")
                        f.write(f"         Tensor Type: {edge_data['tensor_type']}\n")
                        f.write(f"         Edge: {edge_data['edge']}\n")
                
                f.write("\n" + "-"*80 + "\n\n")
        
        debug_print(f"FIFO conflict table exported to: {output_path}")


class NoCDeadlockDetector:
    """
    Detect potential deadlocks in NoC (Network on Chip) transfers by identifying
    circular dependencies in the transfer graph.
    
    A deadlock can occur when there is a cycle in the dependency graph formed by
    tensor transfers. This detector considers only the source and destination endpoints
    of each transfer, ignoring intermediate routing nodes.
    
    For example, if:
    - Tensor i is transferred from IMCE2 to IMCE4 (source->dest)
    - Tensor j is transferred from IMCE4 to IMCE2 (source->dest)
    
    This forms a cycle: IMCE2 -> IMCE4 -> IMCE2, which could lead to deadlock.
    
    The detector analyzes NoCPaths dictionary which maps tensor edges to their
    source and destination hardware nodes, and identifies all cycles in the
    resulting directed graph.
    """
    
    def __init__(self):
        self.deadlock_table = {}  # {func_name: [deadlock_cycles]}
        self.transfer_graph = {}  # {func_name: {src_node: [dst_nodes]}}
    
    def run(self, mod):
        """
        Analyze all imcflow functions and detect potential deadlock cycles.
        
        Parameters
        ----------
        mod : tvm.IRModule
            The module containing imcflow functions
            
        Returns
        -------
        dict
            Deadlock table with structure:
            {
              func_name: [
                {
                  'cycle': [NodeID, ...],  # List of nodes forming the cycle
                  'transfers': [           # Transfers involved in the cycle
                    {
                      'src_node': NodeID,
                      'dst_node': NodeID,
                      'edges': [TensorEdge, ...]  # Edges involved in this transfer
                    },
                    ...
                  ]
                },
                ...
              ],
              ...
            }
        """
        noc_paths = ImcflowDeviceConfig().NoCPaths
        custom_id_to_name = CustomIDToName()
        hw_node_map = ImcflowDeviceConfig().HWNodeMap
        
        for gv, func in mod.functions.items():
            if not (isinstance(func, relay.Function) and 
                   hasattr(func.attrs, "Compiler") and 
                   func.attrs["Compiler"] == "imcflow"):
                continue
            
            func_name = gv.name_hint
            self.deadlock_table[func_name] = []
            self.transfer_graph[func_name] = {}
            
            # Get NoC paths for this function
            if func_name not in noc_paths:
                continue
            
            func_noc_paths = noc_paths[func_name]
            
            # Build transfer graph: {src_node: {dst_node: [edges]}}
            transfer_map = {}  # {(src, dst): [edges]}
            
            for key, path_info in func_noc_paths.items():
                # Skip instruction paths (which have NodeID as key instead of TensorEdge)
                if isinstance(key, NodeID):
                    continue
                
                # path_info is a tuple: (src_hw_node, dst_hw_node, split_idx)
                if not isinstance(path_info, tuple) or len(path_info) < 2:
                    continue
                
                src_node = path_info[0]
                dst_node = path_info[1]
                
                # Only consider transfers between different nodes
                if src_node == dst_node:
                    continue
                
                # Store the edge for this transfer
                transfer_key = (src_node, dst_node)
                if transfer_key not in transfer_map:
                    transfer_map[transfer_key] = []
                transfer_map[transfer_key].append(key)  # key is the TensorEdge
            
            # Build adjacency list for cycle detection
            adjacency = {}
            for (src, dst), edges in transfer_map.items():
                if src not in adjacency:
                    adjacency[src] = []
                if dst not in adjacency[src]:
                    adjacency[src].append(dst)
            
            # Detect cycles using DFS
            cycles = self._detect_cycles(adjacency)
            
            # Build detailed cycle information
            for cycle_nodes in cycles:
                cycle_info = {
                    'cycle': cycle_nodes,
                    'transfers': []
                }
                
                # For each edge in the cycle, find the corresponding transfers
                for i in range(len(cycle_nodes)):
                    src = cycle_nodes[i]
                    dst = cycle_nodes[(i + 1) % len(cycle_nodes)]
                    
                    transfer_key = (src, dst)
                    if transfer_key in transfer_map:
                        transfer_edges = transfer_map[transfer_key]
                        
                        cycle_info['transfers'].append({
                            'src_node': src,
                            'dst_node': dst,
                            'edges': transfer_edges
                        })
                
                self.deadlock_table[func_name].append(cycle_info)
                
                # Debug output
                debug_print(f"[NoC Deadlock] Function: {func_name}")
                debug_print(f"  Cycle detected: {' -> '.join(str(n) for n in cycle_nodes)} -> {cycle_nodes[0]}")
                debug_print(f"  Transfers in cycle:")
                for transfer in cycle_info['transfers']:
                    debug_print(f"    {transfer['src_node']} -> {transfer['dst_node']}: "
                              f"{len(transfer['edges'])} edge(s)")
        
        # Store in device config for later access
        ImcflowDeviceConfig().NoCDeadlockTable = self.deadlock_table
        
        return self.deadlock_table
    
    def _detect_cycles(self, adjacency):
        """
        Detect all cycles in a directed graph using DFS.
        
        Parameters
        ----------
        adjacency : dict
            Adjacency list representation: {node: [neighbor_nodes]}
            
        Returns
        -------
        list
            List of cycles, where each cycle is a list of nodes
        """
        cycles = []
        visited = set()
        rec_stack = set()  # Recursion stack to track current path
        path = []  # Current path being explored
        
        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            if node in adjacency:
                for neighbor in adjacency[node]:
                    if neighbor not in visited:
                        dfs(neighbor)
                    elif neighbor in rec_stack:
                        # Cycle detected! Extract the cycle from path
                        cycle_start_idx = path.index(neighbor)
                        cycle = path[cycle_start_idx:]
                        
                        # Normalize cycle (start from smallest node to avoid duplicates)
                        min_idx = cycle.index(min(cycle, key=lambda x: x.value if isinstance(x, NodeID) else x))
                        normalized_cycle = cycle[min_idx:] + cycle[:min_idx]
                        
                        # Check if this cycle is already found
                        if normalized_cycle not in cycles:
                            cycles.append(normalized_cycle)
            
            path.pop()
            rec_stack.remove(node)
        
        # Explore from all nodes
        all_nodes = set(adjacency.keys())
        for node in adjacency.values():
            all_nodes.update(node)
        
        for node in all_nodes:
            if node not in visited:
                dfs(node)
        
        return cycles
    
    def print_deadlock_summary(self):
        """
        Print a summary of all detected potential deadlocks.
        """
        total_deadlocks = sum(len(cycles) for cycles in self.deadlock_table.values())
        
        if total_deadlocks == 0:
            print("\n" + "="*60)
            print("NoC Deadlock Detector: No potential deadlocks detected!")
            print("="*60)
            return
        
        print("\n" + "="*60)
        print(f"NoC Deadlock Detector: {total_deadlocks} potential deadlock(s) detected")
        print("="*60)
        print("\nWARNING: Circular dependencies detected in NoC transfers!")
        print("These cycles could potentially lead to deadlock situations.")
        
        for func_name, cycles in self.deadlock_table.items():
            if not cycles:
                continue
            
            print(f"\nFunction: {func_name}")
            print(f"  Number of cycles: {len(cycles)}")
            
            for i, cycle_info in enumerate(cycles, 1):
                cycle_nodes = cycle_info['cycle']
                transfers = cycle_info['transfers']
                
                print(f"\n  Cycle #{i}:")
                cycle_str = ' -> '.join(str(n) for n in cycle_nodes)
                print(f"    Path: {cycle_str} -> {cycle_nodes[0]}")
                print(f"    Transfers involved:")
                
                for j, transfer in enumerate(transfers, 1):
                    print(f"      {j}. {transfer['src_node']} -> {transfer['dst_node']}")
                    print(f"         Number of tensor edges: {len(transfer['edges'])}")
                    for edge in transfer['edges'][:3]:  # Show first 3 edges
                        print(f"           {edge}")
                    if len(transfer['edges']) > 3:
                        print(f"           ... and {len(transfer['edges']) - 3} more edge(s)")
        
        print("\n" + "="*60)
    
    def export_deadlock_table(self, output_path):
        """
        Export the deadlock detection results to a text file.
        
        Parameters
        ----------
        output_path : str
            Path to the output file
        """
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("NoC Deadlock Detection Report\n")
            f.write("="*80 + "\n\n")
            
            total_deadlocks = sum(len(cycles) for cycles in self.deadlock_table.values())
            f.write(f"Total potential deadlocks detected: {total_deadlocks}\n")
            
            if total_deadlocks > 0:
                f.write("\nWARNING: Circular dependencies detected in NoC transfers!\n")
                f.write("These cycles could potentially lead to deadlock situations.\n")
            
            f.write("\n")
            
            for func_name, cycles in self.deadlock_table.items():
                if not cycles:
                    f.write(f"Function: {func_name}\n")
                    f.write("  No deadlocks detected\n\n")
                    continue
                
                f.write(f"Function: {func_name}\n")
                f.write(f"  Number of cycles: {len(cycles)}\n")
                
                for i, cycle_info in enumerate(cycles, 1):
                    cycle_nodes = cycle_info['cycle']
                    transfers = cycle_info['transfers']
                    
                    f.write(f"\n  Cycle #{i}:\n")
                    cycle_str = ' -> '.join(str(n) for n in cycle_nodes)
                    f.write(f"    Path: {cycle_str} -> {cycle_nodes[0]}\n")
                    f.write(f"    Number of nodes in cycle: {len(cycle_nodes)}\n")
                    f.write(f"    Transfers involved:\n")
                    
                    for j, transfer in enumerate(transfers, 1):
                        f.write(f"\n      Transfer #{j}:\n")
                        f.write(f"        Source Node: {transfer['src_node']}\n")
                        f.write(f"        Destination Node: {transfer['dst_node']}\n")
                        f.write(f"        Number of tensor edges: {len(transfer['edges'])}\n")
                        f.write(f"        Edges:\n")
                        
                        for edge in transfer['edges']:
                            f.write(f"          {edge}\n")
                
                f.write("\n" + "-"*80 + "\n\n")

        debug_print(f"NoC deadlock table exported to: {output_path}")


def extract_outputs_by_custom_ids(mod, target_custom_ids):
    """
    Extract nodes with specific custom_ids and create a new module
    with those nodes as outputs.

    This function traverses the graph, finds nodes with target custom_ids,
    and modifies function bodies to make those nodes the outputs.
    Global functions and local (Composite) functions are preserved.
    Consumer nodes after the target are removed via DCE.

    Args:
        mod: relay.Module - The input module (should have custom_id annotations)
        target_custom_ids: list of int - The custom_ids of nodes to extract as outputs

    Returns:
        tuple: (new_mod, found_ids)
            - new_mod: New relay.Module with extracted outputs
            - found_ids: List of custom_ids that were actually found

    Example:
        >>> mod = annotateCustomId(mod)
        >>> extracted_mod, found_ids = extract_outputs_by_custom_ids(mod, [63])
    """
    target_ids_set = set(target_custom_ids)

    class CustomIdCollector(ExprVisitor):
        """
        Collect target nodes and their function context (path from main to target).
        func_path is a list of (type, key, func, call_node) tuples.
        """
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.visited_funcs = set()
            self.collected_nodes = {}  # custom_id -> (node, func_path)
            self.current_func_path = []

        def visit_call(self, call):
            if isinstance(call.op, GlobalVar):
                func_name = call.op.name_hint
                if func_name not in self.visited_funcs:
                    self.visited_funcs.add(func_name)
                    func = self.module[func_name]
                    self.current_func_path.append(("global", func_name, func, call))
                    self.visit(func)
                    self.current_func_path.pop()
            elif isinstance(call.op, relay.Function):
                self.current_func_path.append(("local", id(call.op), call.op, call))
                self.visit(call.op)
                self.current_func_path.pop()

            super().visit_call(call)

            if call.attrs and hasattr(call.attrs, 'custom_id'):
                try:
                    custom_id = int(call.attrs.custom_id)
                    if custom_id in target_ids_set:
                        self.collected_nodes[custom_id] = (call, list(self.current_func_path))
                except (TypeError, ValueError):
                    pass

    # Step 1: Collect target nodes
    collector = CustomIdCollector(mod)
    collector.visit(mod["main"].body)

    found_ids = []
    for cid in sorted(target_custom_ids):
        if cid in collector.collected_nodes:
            found_ids.append(cid)
        else:
            print(f"Warning: custom_id {cid} not found in graph")

    if not found_ids:
        raise ValueError("No target nodes found! Check if custom_ids exist in the graph.")

    if len(found_ids) > 1:
        print(f"Warning: Multiple targets found. Only using first target: {found_ids[0]}")

    target_cid = found_ids[0]
    target_node, func_path = collector.collected_nodes[target_cid]

    print("----- Extracted node with custom_id(s):", found_ids)

    # Step 2: Find containing global function and local function chain
    containing_global_name = None
    containing_global_func = None
    containing_global_call = None
    local_func_chain = []  # List of (local_func, call_node) from outermost to innermost

    for func_type, func_key, func, call_node in func_path:
        if func_type == "global":
            containing_global_name = func_key
            containing_global_func = func
            containing_global_call = call_node
        elif func_type == "local":
            local_func_chain.append((func, call_node))

    # Step 3: Build new function body from innermost to outermost
    # Start with target_node as the output
    current_body = target_node

    # Process local functions from innermost to outermost
    for i in range(len(local_func_chain) - 1, -1, -1):
        local_func, call_node = local_func_chain[i]
        # Create new local function with current_body as its body
        new_local_func = relay.Function(
            local_func.params,
            current_body,
            None,
            local_func.type_params,
            local_func.attrs
        )
        # Create a call to this new local function with same args and attrs
        current_body = relay.Call(
            new_local_func,
            call_node.args,
            call_node.attrs,
            call_node.type_args,
            call_node.span
        )

    # Step 4: Find used parameters in current_body for the global function
    # We need to keep only the parameters that are actually used
    class UsedVarCollector(ExprVisitor):
        def __init__(self):
            super().__init__()
            self.used_vars = set()

        def visit_var(self, var):
            self.used_vars.add(var)
            super().visit_var(var)

    used_params = []
    used_args = []
    param_to_arg_map = {}  # Map old param to new arg for updating body

    if containing_global_func is not None:
        # Collect used variables in current_body
        var_collector = UsedVarCollector()
        var_collector.visit(current_body)

        # Find which parameters are used
        for i, param in enumerate(containing_global_func.params):
            if param in var_collector.used_vars:
                used_params.append(param)
                if containing_global_call is not None and i < len(containing_global_call.args):
                    used_args.append(containing_global_call.args[i])

    # Step 5: Create new module
    new_mod = tvm.IRModule()

    # Process each global function
    for gv, func in mod.functions.items():
        func_name = gv.name_hint
        if func_name == "main":
            continue  # Handle main separately

        if func_name == containing_global_name:
            # This global function contains the target
            # Replace its body with current_body, keeping only used params
            new_func = relay.Function(
                used_params,
                current_body,
                None,
                containing_global_func.type_params,
                containing_global_func.attrs
            )
            new_mod[gv] = new_func
        else:
            # Keep original function
            new_mod[gv] = func

    # Step 6: Handle main function
    # We need to modify main's body to call the modified global function
    # and make its result the output of main
    main_func = mod["main"]

    if not func_path:
        # Target is directly in main function (no global or local functions)
        # Replace main body with just the target node
        new_main_body = target_node
        new_main_func = relay.Function(
            main_func.params,
            new_main_body,
            None,
            main_func.type_params,
            main_func.attrs
        )
    elif func_path[0][0] == "local":
        # Target is inside a local function directly in main (no global function wrapper)
        # current_body already contains the local function call chain
        new_main_body = current_body
        new_main_func = relay.Function(
            main_func.params,
            new_main_body,
            None,
            main_func.type_params,
            main_func.attrs
        )
    else:
        # Target is inside a global function
        # We need to replace the call to containing_global_func in main's body
        # with a call that becomes the final output

        class MainBodyCutter(ExprVisitor):
            """Find the call to the containing global function and return it."""
            def __init__(self, target_gv_name):
                super().__init__()
                self.target_gv_name = target_gv_name
                self.found_call = None

            def visit_call(self, call):
                # Check if this is the call to the containing global function
                if isinstance(call.op, GlobalVar) and call.op.name_hint == self.target_gv_name:
                    self.found_call = call
                # Continue visiting to find nested calls
                super().visit_call(call)

        # First, find the original call to the containing global function
        cutter = MainBodyCutter(containing_global_name)
        cutter.visit(main_func.body)

        if cutter.found_call is None:
            raise ValueError(f"Could not find call to {containing_global_name} in main function")

        # Get the GlobalVar for the containing function (it might be new or same)
        target_gv = None
        for gv in new_mod.functions:
            if gv.name_hint == containing_global_name:
                target_gv = gv
                break

        if target_gv is None:
            # Should not happen
            target_gv = containing_global_call.op

        # Create new main body: everything up to and including the modified global function call
        # The new main body is simply the call to the modified global function
        # We need to rebuild the computation graph up to this call

        class MainBodyBuilder(ExprMutator):
            """Rebuild main's body, stopping at the target global function call."""
            def __init__(self, target_gv_name, new_gv, new_args, orig_call):
                super().__init__()
                self.target_gv_name = target_gv_name
                self.new_gv = new_gv
                self.new_args = new_args
                self.orig_call = orig_call
                self.result = None

            def visit_call(self, call):
                # Check if this is the call to the containing global function
                if isinstance(call.op, GlobalVar) and call.op.name_hint == self.target_gv_name:
                    # Return the modified call - this is the final output
                    self.result = relay.Call(
                        self.new_gv,
                        self.new_args,
                        call.attrs,
                        call.type_args,
                        call.span
                    )
                    return self.result
                return super().visit_call(call)

        builder = MainBodyBuilder(containing_global_name, target_gv, used_args, cutter.found_call)
        builder.visit(main_func.body)

        if builder.result is not None:
            new_main_body = builder.result
        else:
            # Fallback: just use the found call with modified args
            new_main_body = relay.Call(
                target_gv,
                used_args,
                cutter.found_call.attrs,
                cutter.found_call.type_args,
                cutter.found_call.span
            )

        new_main_func = relay.Function(
            main_func.params,
            new_main_body,
            None,
            main_func.type_params,
            main_func.attrs
        )

    new_mod["main"] = new_main_func

    # Step 7: Clear all checked_type to allow InferType to work correctly
    # The module has a mix of typed (from original) and untyped (newly created) nodes
    @tvm.ir.transform.module_pass(opt_level=0)
    def clear_checked_types(mod, ctx):
        """Clear all checked_type annotations from the module."""
        class TypeClearer(ExprMutator):
            def __init__(self):
                super().__init__()
                # Don't use memo for this pass - we want to rebuild everything
                self.memo_map = {}

            def visit(self, expr):
                # Clear checked_type by rebuilding the expression
                result = super().visit(expr)
                return result

            def visit_function(self, fn):
                # Create new params (to clear any type annotations on them)
                new_params = []
                for p in fn.params:
                    # Create new Var with same name and type_annotation
                    new_param = relay.Var(p.name_hint, p.type_annotation)
                    self.memo_map[p] = new_param
                    new_params.append(new_param)

                new_body = self.visit(fn.body)
                # Create new function without ret_type (will be inferred)
                return relay.Function(
                    new_params,
                    new_body,
                    None,  # Clear return type
                    fn.type_params,
                    fn.attrs
                )

            def visit_call(self, call):
                new_args = [self.visit(arg) for arg in call.args]

                if isinstance(call.op, relay.Function):
                    new_op = self.visit(call.op)
                elif isinstance(call.op, GlobalVar):
                    new_op = call.op
                else:
                    new_op = self.visit(call.op)

                # Create new call without type_args (will be inferred)
                return relay.Call(new_op, new_args, call.attrs, [], call.span)

            def visit_var(self, var):
                # Return mapped var if exists (for function params)
                if var in self.memo_map:
                    return self.memo_map[var]
                # Otherwise create new var with same type_annotation
                return relay.Var(var.name_hint, var.type_annotation)

            def visit_tuple(self, tup):
                return relay.Tuple([self.visit(f) for f in tup.fields])

            def visit_tuple_getitem(self, tgi):
                return relay.TupleGetItem(self.visit(tgi.tuple_value), tgi.index)

            def visit_constant(self, const):
                # Create new constant without checked_type
                return relay.Constant(const.data)

        # Don't clear types - keep original types
        # Just return the module as-is
        return mod

    new_mod = clear_checked_types(new_mod)

    # DEBUG: Print module
    print("=" * 60)
    print("DEBUG: Extracted module")
    print("=" * 60)
    print(new_mod)
    print("=" * 60)

    # Skip InferType and DCE - return the module as-is
    # The module already has partial type information from the original module
    # DCE is not needed since we've already removed the consumer nodes by
    # making the target global function return the target node directly

    return new_mod, found_ids
