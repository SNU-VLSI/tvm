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

from tvm.relay.backend.contrib.imcflow.layout import ImcflowLayoutLegalizer

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
                            # This is only for multicast case, allow reusing existing path
                            # For single path case, this won't be triggered as the explored_router_list is None
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
                  src_node_data = f"instruction_{edge.name}"
                else:
                  src_node_data = edge.src_id.graph_node_id

                source_coord = NodeID.to_coord(source_node)
                dest_coord = NodeID.to_coord(dest_node)
                entry_addr = len(policy_tables[source_node])

                if router_entry_list is None: # initial handling
                    router_entry_list= []
                    if source_coord == dest_coord: # if same node, return
                        return
                    # check if there's previous path with same source and same source tensor id, which means multicast(i.e. split operation)
                    elif any(k[0] == source_node and k[2] == src_node_data for k in self.start_addr_dict.keys()):
                        handle_multicast(edge, mapping_info)
                        return
                    else:
                        self.start_addr_dict[(source_node, dest_node, src_node_data)] = entry_addr # each source can have several tensor type

                # Try X-Y routing first
                path_coords = get_path_coords(source_coord, dest_coord, True)
                if (source_node, dest_node, src_node_data) not in self.explored_router_list:
                    self.explored_router_list[(source_node, dest_node, src_node_data)] = path_coords
                else:
                    self.explored_router_list[(source_node, dest_node, src_node_data)].extend(path_coords)

                current_coord = source_coord
                current_node = source_node
                # Apply the successful path to tables
                for next_coord in path_coords:
                    direction = get_direction(current_coord, next_coord)
                    next_node = NodeID.from_coord(next_coord[0], next_coord[1])

                    #append entry to router's policy table
                    entry = {"Local": {"enable": False, "chunk_index": 0, "addr": 0}, \
                      "North": {"enable": False, "addr": 0}, \
                      "East": {"enable": False, "addr": 0},  \
                      "South": {"enable": False, "addr": 0}, \
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
                  "East": {"enable": False, "addr": 0},  \
                  "South": {"enable": False, "addr": 0}, \
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
                  src_node_data = f"instruction_{edge.name}"
                else:
                  src_node_data = edge.src_id.graph_node_id
                router_entry_list= []

                if source_node == dest_node: # if same node, return
                    return

                # Follow existing path and modify at divergence point
                previous_path_key = None
                for k in self.start_addr_dict.keys():
                  if k[0] == source_node and k[2] == src_node_data:
                    previous_path_key = k
                    break
                if previous_path_key is None:
                  raise ValueError("No previous path found for multicast handling.")

                entry_addr = self.start_addr_dict[previous_path_key]
                current_node = source_node
                current_coord = NodeID.to_coord(current_node)
                dest_coord = NodeID.to_coord(dest_node)
                next_coord = None

                while current_coord != dest_coord:
                    entry = policy_tables[current_node][entry_addr] # current policy table entry

                    # Find which direction to go next.
                    path_coords = get_path_coords(current_coord, dest_coord, self.explored_router_list[previous_path_key])
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

                      elif edge.src_id.tensor_type in ["weight", "bias", "fused_scale", "fused_bias", "min", "max", "threshold", "scale", "config"]:
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



