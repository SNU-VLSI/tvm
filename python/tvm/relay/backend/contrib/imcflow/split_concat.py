#TODO:
# 1. memory allocation is not optimal. we can split tensor to multiple inode. current implementation assign only one inode for entire tensor.

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

from tvm.relay.op.nn import imcflow_batch_norm, imcflow_qconv2d, imcflow_qdwconv2d
from tvm.relay.qnn.op.qnn import imcflow_min_max_quantize, imcflow_nu_quantize
from tvm.relay.op.transform import imcflow_packing, imcflow_unpacking, imcflow_4d_to_qconv_input, imcflow_mmquant_out_to_4d
import numpy as np
from tvm.relay.op.contrib import imcflow
from tvm.relay.backend.contrib.imcflow.acim_util import *

from tvm.relay.backend.contrib.imcflow.transform_utils import *
from tvm.relay.backend.contrib.imcflow.layout import *
from tvm.relay.backend.contrib.imcflow.conv_spliter import split_conv_to_atomic_impl
from tvm.relay.backend.contrib.imcflow.round_partition import partitionRound_impl
from tvm.relay.backend.contrib.imcflow.node_mapper import NodeMapper

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

def makeSplitConcatDepsRegions_impl(mod):
  for global_var, func in mod.functions.items():
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
