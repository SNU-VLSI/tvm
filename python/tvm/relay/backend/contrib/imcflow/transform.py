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
from tvm.relay.backend.contrib.imcflow.split_concat import makeSplitConcatDepsRegions_impl, ConcatDistributor

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

def partitionImcflowSubGraph(mod):
  mod = relay.transform.InferType()(mod)
  region_list = get_imcflow_supported_regions(mod)
  mod = imcflow.ImcflowAnnotationPass(region_list)(mod)
  mod = transform.MergeCompilerRegions()(mod)
  mod = imcflow.ImcflowCleanRegionTag()(mod)
  mod = transform.PartitionGraph()(mod)
  return mod

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

def split_conv_to_atomic(mod, OldParamDict):
  return split_conv_to_atomic_impl(mod, OldParamDict)

def makeSplitConcatDepsRegions(mod):
  return makeSplitConcatDepsRegions_impl(mod) 

def run_concat_distributor(eval_mod):
  eval_mod = ConcatDistributor(max_inputs=4).run(eval_mod)
  return eval_mod

def partitionRound(mod):
  return partitionRound_impl(mod)

def flattenImcflowTopFuncs(mod):
  return imcflow.flattenImcflowTopFuncs(mod)

def prune_imcflow_subgraphs(mod):
  return imcflow.prune_imcflow_subgraphs(mod)

def map_nodes_to_hw_nodes(mod):
  NodeMapper().run(mod)

def legalize_imcflow_layout(mod):
  layout_legalizer = ImcflowLayoutLegalizer()
  eval_mod, ttype_map = layout_legalizer.transform_mod(mod)
  return eval_mod, ttype_map