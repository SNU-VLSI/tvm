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
