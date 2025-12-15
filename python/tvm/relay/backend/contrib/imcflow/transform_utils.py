from tvm.contrib.relay_viz import RelayVisualizer, DotPlotter, DotVizParser
from tvm.relay import pretty_print
import tvm
from tvm import relay
from tvm.relay import transform, op
from tvm.relay.op.contrib.imcflow import HashToCustomID, CustomIDToName, CustomIDInFunc, CustomIDToNode
from tvm.relay.ty import TupleType, TensorType
from tvm.relay.expr import (Call, GlobalVar, TupleGetItem, const, Let, Var, If, Tuple, Constant)
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.ir import Op
from tvm.relay.op.contrib import imcflow
from tvm.relay.function import Function, FunctionWithFields

from tvm.contrib.imcflow import ImcflowDeviceConfig

import os
import collections
import re

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


def printModel(result_dir, mod, param_dict, mod_name):
  RelayVisualizer(
      relay_mod=mod,
      relay_param=param_dict,
      plotter=DotPlotter(),
      parser=DotVizParser(),
  ).render(f"{result_dir}/{mod_name}")

  with open(f"{result_dir}/{mod_name}.txt", "w") as f:
    f.write(pretty_print(mod))

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
    self.sub_parsers = {} # if function has local functions, store their use-def parsers here
                          # {local func_node : UseDefChainParser}
    self.param_to_args = {} # {local func param : arg}
  
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

    if isinstance(call.op, relay.Function):
      if call.op not in self.sub_parsers:
        sub_parser = UseDefChainParser()
        sub_parser.visit(call.op.body)
        self.sub_parsers[call.op] = sub_parser
      
      for param, arg in zip(call.op.params, call.args):
        self.param_to_args[param] = arg

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

  def get_users(self, expr, recursive=False, depth=-1, skip_tuple=False):
    """
    Get all users (consumers) of an expression
    Args:
        expr: The expression to query
        recursive: If True, recursively find users through sub-functions
        depth: Maximum recursion depth (-1 for unlimited)
        skip_tuple: If True, skip through Tuple and TupleGetItem nodes to find Call users
    """
    if not recursive and not skip_tuple:
      return self.users.get(expr, [])

    final_users = []

    def _traverse(current_expr, current_parser, current_depth):
      direct_users = current_parser.users.get(current_expr, [])
      for user, tag in direct_users:
        # Skip through Tuple and TupleGetItem nodes if skip_tuple is enabled
        if skip_tuple and isinstance(user, (relay.Tuple, relay.TupleGetItem)):
          _traverse(user, current_parser, current_depth)
          continue

        if isinstance(user, relay.Call) and isinstance(user.op, relay.Function):
          if current_depth == 0:
            final_users.append((user, tag))
          else:
            sub_func = user.op
            if sub_func in current_parser.sub_parsers:
              sub_parser = current_parser.sub_parsers[sub_func]
              if tag < len(sub_func.params):
                param = sub_func.params[tag]
                _traverse(param, sub_parser, current_depth - 1 if current_depth > 0 else -1)
              else:
                final_users.append((user, tag))
            else:
              final_users.append((user, tag))
        else:
          final_users.append((user, tag))

    _traverse(expr, self, depth)
    return final_users

  def get_uses(self, expr, recursive=False, depth=-1, skip_tuple=False):
    """
    Get operands (dependencies) of an expression
    Args:
        expr: The expression to query
        recursive: If True, recursively find uses through sub-functions
        depth: Maximum recursion depth (-1 for unlimited)
        skip_tuple: If True, skip through Tuple and TupleGetItem nodes to find Call operands
    """
    if expr not in self.uses and expr not in self.users:
      for sub_parser in self.sub_parsers.values():
        if expr in sub_parser.uses or expr in sub_parser.users:
          return sub_parser.get_uses(expr, recursive, depth, skip_tuple)

    if not recursive and not skip_tuple:
      return self.uses.get(expr, [])

    final_uses = []

    def _traverse(current_expr, current_parser, current_depth):
      if isinstance(current_expr, relay.Function):
        direct_uses = [current_expr.body]
      else:
        direct_uses = current_parser.uses.get(current_expr, [])
      for operand in direct_uses:
        # Skip through Tuple and TupleGetItem nodes if skip_tuple is enabled
        if skip_tuple and isinstance(operand, (relay.Tuple, relay.TupleGetItem)):
          _traverse(operand, current_parser, current_depth)
          continue

        if isinstance(operand, relay.Call) and isinstance(operand.op, relay.Function):
          if current_depth == 0:
            final_uses.append(operand)
          else:
            sub_func = operand.op
            sub_parser = current_parser.sub_parsers[sub_func]
            _traverse(sub_func, sub_parser, current_depth - 1 if current_depth > 0 else -1)
        else:
          final_uses.append(operand)

    _traverse(expr, self, depth)
    return final_uses

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

class UseDefChainBuilder:
  """
  use def chain builder for module.
  """
  def __init__(self, mod):
    self.mod = mod
    self.use_def_chain_parsers = {}  # {global_var_name: UseDefChainParser}

    for global_var, func in mod.functions.items():
      if isinstance(func, relay.Function):
        parser = UseDefChainParser()
        parser.visit(func.body)
        self.use_def_chain_parsers[global_var.name_hint] = parser

  def get_parser_for_func(self, global_var_name):
    return self.use_def_chain_parsers[global_var_name]

def get_type(parent_mod, node):
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
        tuple_type = get_type(parent_mod, node.tuple_value)
        if isinstance(tuple_type, relay.TupleType):
          out_type = tuple_type.fields[node.index]
        else:
          raise RuntimeError(f"TupleGetItem node has non-tuple parent type: {tuple_type}")
      elif isinstance(node, relay.Tuple):
        # For Tuple, infer the type of each field and construct a TupleType
        field_types = [get_type(parent_mod, field) for field in node.fields]
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

def get_shape(mod, node):
    """A method to infer the type of a relay expression."""
    out_type = get_type(mod, node)
    # mod = tvm.IRModule.from_expr(node)
    # mod = relay.transform.InferType()(mod)
    # entry = mod["main"]

    # infer_out = entry if isinstance(node, relay.Function) else entry.body
    # out_type = infer_out._checked_type_

    if isinstance(out_type, TensorType):
        # Single tensor, get the shape directly
        shapes = [int(dim) for dim in out_type.shape]
    elif isinstance(out_type, TupleType):
        # Tuple of tensors, get the shape of each tensor in the tuple
        shapes = [int(field) for field in out_type.fields]
    else:
        raise RuntimeError(f"Unsupported output type {type(out_type)} in operator {node.op.name}")

    return shapes

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


def getConstNodesOfFunc(func):
  InNodes = []

  class _Visitor(tvm.relay.ExprVisitor):
    def visit_constant(self, const):
      InNodes.append(const)
      super().visit_constant(const)

  _Visitor().visit(func)
  return InNodes

def getInputNodesOfFunc(func):
  InNodes = []

  class _Visitor(tvm.relay.ExprVisitor):
    def visit_function(self, func):
      for param in func.params:
        InNodes.append(param)

  _Visitor().visit(func)
  return InNodes

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
      return relay.expr.CallWithFields(new_call, new_call.op, new_call.args, new_attrs, new_call.type_args, new_call.span)

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
