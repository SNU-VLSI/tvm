import tvm
from tvm import relay
from tvm.relay import transform, op
from tvm.relay.ty import TupleType, TensorType
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.function import Function, FunctionWithFields
from tvm.relay.expr import (Call, GlobalVar, TupleGetItem, const, Let, Var, If, Tuple, Constant)
from tvm.relay.expr import RefCreate, RefRead, RefWrite
from tvm.relay.adt import Constructor, Match, Clause
from tvm.contrib.imcflow import ImcflowDeviceConfig, TensorEdge, TensorID, NodeID, TensorEdgeInfo, InstEdgeInfo, RouterEntry, DataBlock, MemoryLayout, MemoryRegion
from tvm.ir import Op
from tvm.relay.op.contrib.imcflow import HashToCustomID, CustomIDToName, CustomIDInFunc, CustomIDToNode
from tvm.relay.op.nn import imcflow_batch_norm, imcflow_qconv2d
from tvm.relay.qnn.op.qnn import imcflow_min_max_quantize, imcflow_nu_quantize
from tvm.relay.op.transform import imcflow_packing, imcflow_unpacking, imcflow_4d_to_qconv_input, imcflow_mmquant_out_to_4d
import numpy as np
from tvm.relay.op.contrib import imcflow

import math
from copy import deepcopy
import collections
import re
from dataclasses import dataclass
from enum import Enum
import json


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

def getNodeID(node) -> int:
  id_dict = HashToCustomID()
  if int(hash(node)) in id_dict:
    return id_dict[int(hash(node))]
  else:
    return -1

def getNodeDebugID(node):
  if hasattr(node.op, "attrs"):
    indicator = str(node.op.attrs["Composite"])
  else:
    indicator = str(node.op)
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
    print(f"node: {node}")

    # mod = tvm.IRModule.from_expr(node)
    # mod = relay.transform.InferType()(mod)
    # entry = mod["main"]

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
      out_type = node.checked_type
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
    else:
      raise RuntimeError(f"can't infer type for node {node}")

    # infer_out = entry if isinstance(node, relay.Function) else entry.body
    # out_type = infer_out._checked_type_

    # if isinstance(out_type, TensorType):
    #     # Single tensor, get the shape directly
    #     shapes = [int(dim) for dim in out_type.shape]
    # elif isinstance(out_type, TupleType):
    #     # Tuple of tensors, get the shape of each tensor in the tuple
    #     shapes = [int(field) for field in out_type.fields]
    # else:
    #     raise RuntimeError(f"Unsupported output type {type(out_type)} in operator {node.op.name}")

    print(f"out_type: {out_type}")
    return out_type

def getInputNodesOfFunc(func):
  InNodes = []

  class _Visitor(tvm.relay.ExprVisitor):
    def visit_var(self, var):
      InNodes.append(var)
      super().visit_var(var)

    def visit_constant(self, const):
      InNodes.append(const)
      super().visit_constant(const)

  _Visitor().visit(func)
  return InNodes

def getOutputNodesOfFunc(func):
  IsComposite = isinstance(func.body.op, relay.Function) and "Composite" in func.body.op.attrs and re.match(r"imcflow\..*", func.body.op.attrs["Composite"])
  if IsComposite:
    output_node = func.body.op.body
  else:
    output_node = func.body

  return output_node

def getInputTensorIDs(func):
  pass

def getOutputTensorIDs(func):
  pass

# def makeToQuantizedForm_old(mod):
#   """
#   List of transformations:
#     1. convert Conv to ImcflowQConv2D
#     2. data type conversion to int form
#       conv2d input  : packed 1D int8
#       conv2d weight : packed 1D int8
#       bias, relu, etc -> int16 data type
#   """
#   param_map = {}

#   class _OpConverter(tvm.relay.ExprMutator):
#     def __init__(self):
#       super().__init__()

#     def visit_call(self, call):
#       if call.op == op.get("nn.conv2d"):
#         new_op = op.get("nn.imcflow_qconv")
#         args = [self.visit(arg) for arg in call.args]
#         type_args = []
#         type_args.append(relay.TensorType(call.type_args[0].shape, "int8"))
#         type_args.append(relay.TensorType(call.type_args[1].shape, "int8"))
#         return imcflow_qconv2d(args[0], args[1], strides=(1, 1), padding=(1, 1))
#         # return Call(new_op, args, call.attrs, type_args, call.span)
#       elif call.op == op.get("qnn.imcflow_min_max_quantize"):
#         args = [self.visit(arg) for arg in call.args]
#         return imcflow_min_max_quantize(args[0], args[1], args[2], 1, "int8")
#       else:
#         return super().visit_call(call)

#     def visit_var(self, var):
#       new_var = relay.Var(var.name_hint, relay.TensorType(var.type_annotation.shape, "int8"))
#       param_map[var.name_hint] = new_var
#       return new_var

#     def visit_constant(self, const):
#       Data = const.data.numpy().astype(np.int8)
#       return relay.const(Data, "int8")

#     def visit_function(self, func):
#       # params = [relay.Var(func.params[0].name_hint, relay.TensorType(func.params[0].type_annotation.shape, "int8"))]
#       # func.params[0].type_annotation = relay.TensorType(func.params[0].type_annotation.shape, "int8")
#       # func.params[0] = relay.Var(func.params[0].name_hint, func.params[0].type_annotation)
#       # func.ret_type = relay.TensorType(func.ret_type.shape, "int8")
#       new_body = self.visit(func.body)
#       new_params = [param_map.get(p.name_hint, p) for p in func.params]
#       new_ret_type = relay.TensorType(func.ret_type.shape, "int8")
#       return relay.Function(new_params, new_body, new_ret_type)

#   mod['main'] = _OpConverter().visit(mod['main'])
#   return mod

@relay.transform.function_pass(opt_level=0)
class makeToQuantizedForm:
    """
    List of transformations:
      1. convert Conv to ImcflowQConv2D
      2. data type conversion to int form
        conv2d input  : packed 1D int8
        conv2d weight : packed 1D int8
        bias, relu, etc -> int16 data type
    """       
    def transform_function(self, func, mod, ctx):
      param_map = {}
      class _OpConverter(tvm.relay.ExprMutator):
        def __init__(self):
          super().__init__()
          self.NewParamDict = {}

        def visit_call(self, call):
          if call.op == op.get("nn.conv2d"):
            args = [self.visit(arg) for arg in call.args]

            # TODO: add in_channels and channels to attribute of nn.conv2d and fix this
            # TODO: Sanity check needed!!!!
            in_channels = 0 # FIXME
            channels = 0 # FIXME

            # input layout [ceil(IC/256), N, H, W, IB, 8] int32
            input_shape_orig = args[0].type_annotation.shape # N ic H W
            input_shape_new = [math.ceil(in_channels//256), input_shape_orig[0], input_shape_orig[2], input_shape_orig[3], 4, 8] # IB = 4
            input_new = relay.Var(args[0].name_hint, relay.TensorType(input_shape_new, "int32"))

            # weight layout [ceil(OC/64), ceil(IC/ic), 256, 8] int32
            weight_shape_orig = args[1].data.shape # OC/64, ic, KH, KW
            weight_numpy_array = np.zeros((weight_shape_orig[0], math.ceil(in_channels//weight_shape_orig[1]), 256, 8), dtype=np.int32) #TODO: this should be replaced with real weight tensor!!!!
            weight_new = relay.Constant(tvm.nd.array(weight_numpy_array))

            # append new params to param_map
            param_map[args[0].name_hint] = input_new

            return imcflow_qconv2d(input_new, weight_new, in_channels=in_channels, channels=channels, strides=(1, 1), padding=(1, 1), out_dtype="int16")

          # TODO: Add batchnorm transform and insert quantize op      
          # elif call.op == op.get("qnn.imcflow_min_max_quantize"):
          #   args = [self.visit(arg) for arg in call.args]
          #   return imcflow_min_max_quantize(args[0], args[1], args[2], 1, "int8")

          else:
            return super().visit_call(call)

        # def visit_var(self, var):
        #   # new_var = relay.Var(var.name_hint, relay.TensorType(var.type_annotation.shape, "int8"))
        #   # param_map[var.name_hint] = new_var
        #   # return new_var
        #   return var

        # def visit_constant(self, const):
        #   Data = const.data.numpy().astype(np.int8)
        #   return relay.const(Data, "int8")

        def visit_function(self, func):
          # params = [relay.Var(func.params[0].name_hint, relay.TensorType(func.params[0].type_annotation.shape, "int8"))]
          # func.params[0].type_annotation = relay.TensorType(func.params[0].type_annotation.shape, "int8")
          # func.params[0] = relay.Var(func.params[0].name_hint, func.params[0].type_annotation)
          # func.ret_type = relay.TensorType(func.ret_type.shape, "int8")
          new_body = self.visit(func.body)
          new_params = [param_map.get(p.name_hint, p) for p in func.params]
          return relay.Function(new_params, new_body)
        
          # new_ret_type = relay.TensorType(func.ret_type.shape, "int8")
          # return relay.Function(new_params, new_body, new_ret_type)

      # Returns list of (GlobalVar, Function) pairs sorted alphabetically by function name
      items = mod.functions_items()
      function_names = [item[0].name_hint for item in items]

      num_func = len(function_names)
      for i in range(num_func):
        if function_names[i]=="main": 
          continue
        elif "Compiler" in mod[function_names[i]].attrs and mod[function_names[i]].attrs["Compiler"]=="imcflow":
          print(f"Transforming imcflow function: {function_names[i]}")
          mod[function_names[i]] = _OpConverter().visit(mod[function_names[i]])

      return func #TODO: returning func is right???

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
  _SUPPORTED_OPS = {
    "nn.imcflow_qconv",
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
                    conv_nodes[(oc_id, ic_id)] = imcflow_qconv2d(
                      split_inputs[ic_id] if (not IsDepthWise) else split_inputs[oc_id],
                      split_conv_weights[oc_id][ic_id],
                      in_channels=ic_size if not IsDepthWise else 1,
                      channels=oc_size,
                      kernel_size=(KH, KW),
                      padding=padding,
                      strides=strides,
                      groups=1 if not IsDepthWise else oc_size,
                      out_dtype="int16"
                    )
                    # conv_nodes[(oc_id, ic_id)] = relay.nn.conv2d(
                    #     split_inputs[ic_id] if (not IsDepthWise) else split_inputs[oc_id],
                    #     split_conv_weights[oc_id][ic_id],
                    #     channels=oc_size,
                    #     in_channels=IC, # in_channels should be the same as original conv2d for layout transform pass
                    #     kernel_size=(KH, KW),
                    #     strides=expr.attrs.strides,
                    #     padding=expr.attrs.padding,
                    #     data_layout=expr.attrs.data_layout,
                    #     kernel_layout=expr.attrs.kernel_layout,
                    #     groups=1 if not IsDepthWise else oc_size
                    # )

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
                      post_nodes[oc_id] = imcflow_batch_norm(post_nodes[oc_id], *NewParams)[0]

                concat_node = relay.op.concatenate([post_nodes[oc_id] for oc_id in range(oc_split_num)], axis=1)
            else:
                concat_node = add_nodes[0]

            return concat_node

          def visit_call(self, call):
            if call.op == op.get("nn.imcflow_qconv"):
              PostProcess = self.PostProcess[:]
              self.PostProcess = []
              NewCall = super().visit_call(call)
              NewCall = self.split_and_optimize_conv2d(NewCall, mod, PostProcess)
              return NewCall
            elif call.op in [op.get("nn.bias_add"), op.get("nn.relu"), op.get("nn.batch_norm"), op.get("imcflow.fused_batch_norm")]:
              self.PostProcess.append(call)
              NewCall = super().visit_call(call)
              if hasattr(call, "ShouldDelete"):
                if call.op in [op.get("nn.batch_norm"), op.get("imcflow.fused_batch_norm")]:
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
  Traverse the graph and find post dominate nodes ended with call for all split nodes
  """

  Results = {}
  OutNodes = []
  InputNodes = []
  class _SplitVisitor(tvm.relay.ExprVisitor):

    def visit(self, expr):
        """Apply the visitor to an expression."""
        if isinstance(expr, Function):
            res = self.visit_function(expr)
        elif isinstance(expr, Call):
            res = self.visit_call(expr)
        elif isinstance(expr, Let):
            res = self.visit_let(expr)
        elif isinstance(expr, Var):
            res = self.visit_var(expr)
        elif isinstance(expr, GlobalVar):
            res = self.visit_global_var(expr)
        elif isinstance(expr, If):
            res = self.visit_if(expr)
        elif isinstance(expr, Tuple):
            res = self.visit_tuple(expr)
        elif isinstance(expr, TupleGetItem):
            res = self.visit_tuple_getitem(expr)
        elif isinstance(expr, Constant):
            res = self.visit_constant(expr)
        elif isinstance(expr, Op):
            res = self.visit_op(expr)
        elif isinstance(expr, RefCreate):
            res = self.visit_ref_create(expr)
        elif isinstance(expr, RefRead):
            res = self.visit_ref_read(expr)
        elif isinstance(expr, RefWrite):
            res = self.visit_ref_write(expr)
        elif isinstance(expr, Constructor):
            res = self.visit_constructor(expr)
        elif isinstance(expr, Match):
            res = self.visit_match(expr)
        else:
            raise Exception(f"warning unhandled case: {type(expr)}")

        return res

    def visit_call(self, call):
      if call.op == op.get("split"):
        # make dict entry if not exists
        if call not in Results:
          Results[call] = []

        # append OutNodes and flush
        if len(OutNodes) > 0:
          Results[call].append(OutNodes[:])
          OutNodes.clear()

        for a in call.args:
            self.visit(a)
      else:
        # only track most recent call node
        for a in call.args:
          OutNodes.clear()
          OutNodes.append(call)
          self.visit(a)

    def visit_tuple(self, tup):
      OutNodes.append(tup)
      super().visit_tuple(tup)

    def visit_tuple_getitem(self, t):
      OutNodes.append(t)
      super().visit_tuple_getitem(t)

  class _ConcatVisitor(tvm.relay.ExprVisitor):

    def visit(self, expr):
        """Apply the visitor to an expression."""
        if isinstance(expr, Function):
            res = self.visit_function(expr)
        elif isinstance(expr, Call):
            res = self.visit_call(expr)
        elif isinstance(expr, Let):
            res = self.visit_let(expr)
        elif isinstance(expr, Var):
            res = self.visit_var(expr)
        elif isinstance(expr, GlobalVar):
            res = self.visit_global_var(expr)
        elif isinstance(expr, If):
            res = self.visit_if(expr)
        elif isinstance(expr, Tuple):
            res = self.visit_tuple(expr)
        elif isinstance(expr, TupleGetItem):
            res = self.visit_tuple_getitem(expr)
        elif isinstance(expr, Constant):
            res = self.visit_constant(expr)
        elif isinstance(expr, Op):
            res = self.visit_op(expr)
        elif isinstance(expr, RefCreate):
            res = self.visit_ref_create(expr)
        elif isinstance(expr, RefRead):
            res = self.visit_ref_read(expr)
        elif isinstance(expr, RefWrite):
            res = self.visit_ref_write(expr)
        elif isinstance(expr, Constructor):
            res = self.visit_constructor(expr)
        elif isinstance(expr, Match):
            res = self.visit_match(expr)
        else:
            raise Exception(f"warning unhandled case: {type(expr)}")

        return res

    def visit_call(self, call):
      if call.op == op.get("concatenate"):
        # make dict entry if not exists
        if call not in Results:
          Results[call] = []

        for a in call.args:
          self.visit(a)
          # append InputNodes and flush
          if len(InputNodes) > 0:
            Results[call].append(InputNodes[:])
            InputNodes.clear()
      else:
        # only track most recent call node
        for a in call.args:
          self.visit(a)
        InputNodes.clear()
        InputNodes.append(call)

    def visit_tuple(self, tup):
      Nodes = []
      for x in tup.fields:
        self.visit(x)
        Nodes.extend(InputNodes)
      InputNodes.clear()
      InputNodes.extend(Nodes)
      InputNodes.append(tup)

    def visit_tuple_getitem(self, t):
      super().visit_tuple_getitem(t)
      InputNodes.append(t)

  _SplitVisitor().visit(func)
  _ConcatVisitor().visit(func)
  Regions = []
  for key, value in Results.items():
    Region = [key]
    for path in value:
      for v in path:
        if v not in Region:
          Region.append(v)
    Regions.append(Region)

  # merge region if intersection is not empty
  Changed=True
  while Changed:
    Changed = False
    for i in range(len(Regions)):
      for j in range(i+1, len(Regions)):
        if len(set(Regions[i]) & set(Regions[j])) > 0:
          Regions[i] = list(set(Regions[i]) | set(Regions[j]))
          Regions.pop(j)
          Changed = True
          break
      if Changed:
        break

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
          
          print(f"Warning: Unsupported node found in cost calculation: {call}")
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

                # Capacity check
                deletes = []
                for cand in candidate_regions:
                  if self.getRegionSize(cand) + self.getCost(node) > ImcflowDeviceConfig.IMCE_NUM:
                    deletes.append(cand)
                for d in deletes:
                  if d in candidate_regions:
                    candidate_regions.remove(d)

                # Selection policy: if multiple distinct input regions, create a new region
                uniq = list({id(r): r for r in candidate_regions}.values())
                if len(uniq) == 1:
                  Region = uniq[0]
                  self.addToRegion(Region, node)
                elif len(uniq) > 1:
                  Region = self.createRegion()
                  self.addToRegion(Region, node)
                else:
                  # No input region (inputs likely Var/Const). Prefer previous node's region if available.
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

def quantizeWeight(weight, scale):
  """
  do 4bit quantization. but numpy's smallest data type is int8.
  so sign extend to int8 and use only 16 values from -8 to 7.
  """
  data = weight.data.numpy()
  quantized = np.round(data / scale).astype(np.int8)
  quantized = np.clip(quantized, -8, 7)
  quantized = tvm.nd.array(quantized)
  return quantized

def gatherQuantScaleFactors(min_max_quant_node):
  """
  gather conv2d weight scale and nn.bn's scale and bias.
  traverse nodes until meet min_max_quant node.
  From gathered scale and bias, calculate min and max values.
  """
  return None

def calMinMaxParam(node):
  """
  calculate min and max values from scale and bias for this node
  """
  return None

def insertMinMaxQuant(input):
  """
  quantize input activation to 4bit.
  input relay expr is not constant, so just change dtype to int4.
  no need to construct numpy array
  """

def quantizeConv2d(conv, weight_scale):
  """
  convert nn.conv2d to imcflow.qconv2d with quantized weight
  1. quantize weight. int4 type but int8 sign extension
  2. insert min_max_quant before conv2d input if already int4 type.
  3. change conv2d to qconv2d
  """
  pass

def quantizeBatchNorm(bn, scale, bias):
  """
  convert nn.batch_norm to imcflow.qbatch_norm with quantized scale and bias
  1. quantize scale and bias. int16 type. 
  2. change batch_norm to qbatch_norm
  """
  pass

def quantizeImcflowFuncs(mod, scale_factor_dict):
  pass

@relay.transform.function_pass(opt_level=0)
class NodeMapper:
    def __init__(self):
      # self.MappingDict_2D = {}
      self.MappingDict = {}

    def run_(self, func):
      class _Nodemapper(tvm.relay.ExprVisitor):
        """
          Target Operators:
            conv2d, bias_add, batch_norm, relu, add and fused versions
            split, concat
        """
        def __init__(self):
            super().__init__()
            self.MappingDict ={}
            self.imce_index = ImcflowDeviceConfig.IMCE_NUM - 1
            self.inode_index = ImcflowDeviceConfig.INODE_NUM - 1

            self.undetermined_callnode_exists = False
            self.undetermined_callnode = None

        def traverse_func(self, func):
            self.visit(func)
            return self.MappingDict

        def visit_call(self, call):
          # post DFS search
          # traverse child node
          for a in call.args:
              self.visit(a)

          from_host = True if len(self.MappingDict) == 0 else False

          #for debugging
          indicator = getNodeDebugID(call)
          # if hasattr(call.op, "attrs"):
          #   indicator = call.op.attrs["Composite"]
          # else:
          #   indicator = call.op

          # check if this node is
          IsConcat = isinstance(call.op, tvm.ir.Op) and call.op.name in ["concatenate"]
          IsSplit = isinstance(call.op, tvm.ir.Op) and call.op.name in ["split"]
          # IsPacking = isinstance(call.op, tvm.ir.Op) and call.op.name in ["imcflow_packing"]
          # IsUnpacking = isinstance(call.op, tvm.ir.Op) and call.op.name in ["imcflow_unpacking"]
          if IsConcat:
              if from_host is True:
                  raise ValueError("concatenate should have at least 1 child node")
              self.MappingDict[getNodeID(call)] = self.MappingDict[getNodeID(call.args[-1].fields[-1])]
              # self.MappingDict[getNodeID(call)] = (last_child_mapping, indicator)
          elif IsSplit:
              if from_host is True:
                  # if this call receives tensor directly from host, map to arbitrary inode
                  self.MappingDict[getNodeID(call)] = NodeID.from_inode_coord(self.inode_index)
                  # self.MappingDict[int(hash(call))] = (f"inode_{self.inode_index}", indicator)
                  self.inode_index -= 1
              else:
                  # self.MappingDict[int(hash(call))] = (last_child_mapping, indicator)
                  self.MappingDict[getNodeID(call)] = self.MappingDict[getNodeID(call.args[-1])]
          # elif IsPacking:
          #   # map to child
          #   self.MappingDict[getNodeID(call)] = self.MappingDict[getNodeID(call.args[-1])]
          #   # # map to nearest inode of argument's hw node
          #   # SrcHWNodeID = NodeID.to_coord(self.MappingDict[getNodeID(call.args[-1])])[0]
          #   # InodeID = NodeID.from_inode_coord(SrcHWNodeID)
          #   # self.MappingDict[getNodeID(call)] = InodeID

          # elif IsUnpacking:
          #     # keep unpacking node and determine its NodeID in parent node
          #     self.undetermined_callnode_exists = True
          #     self.undetermined_callnode = getNodeID(call)
          else:
              self.MappingDict[getNodeID(call)] = NodeID.from_imce_coord(self.imce_index)
              self.imce_index -= 1
              # self.MappingDict[int(hash(call))] = (f"imce_{self.imce_index}", indicator)

          # # handle undetermined unpacking node
          # if IsUnpacking is False and self.undetermined_callnode_exists is True:
          #     DstHWNodeID = NodeID.to_coord(self.MappingDict[getNodeID(call)])[0]
          #     InodeID = NodeID.from_inode_coord(DstHWNodeID)
          #     self.MappingDict[self.undetermined_callnode] = InodeID # assign inode in a same row with parent node
          #     self.undetermined_callnode_exists = False

        def visit_tuple_getitem(self, op):
          super().visit_tuple_getitem(op)

        def visit_tuple(self, op):
          super().visit_tuple(op)

      # self.MappingDict.update(_Nodemapper().traverse_func(func))

      return _Nodemapper().traverse_func(func)
    
    def run(self, mod):
      for global_var, func in mod.functions.items():
        if isinstance(func, relay.Function) and "Compiler" in func.attrs and re.match(r"imcflow.*", func.attrs["Compiler"]):
          mapping_dict = self.run_(func)
          ImcflowDeviceConfig().HWNodeMap.update(mapping_dict)
      return mod

def constructTensorEdgeList(mod):
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
        InputGraphNodeID = self.getCustomID(fn.body)
        DstGraphNodeID = self.getCustomID(fn)
        SrcTag = "odata"
        DstTag = "odata"
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
          print(call)
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
          elif call.op == op.get("nn.imcflow_qconv"):
            _processInputNode(call.args[0], "odata", DstGraphNodeID, "data", self.getInputGraphNodeSplitIndex(call.args[0]))
            _processInputNode(call.args[1], "weight", DstGraphNodeID, "weight", None)
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

  for func_name_var, func in mod.functions.items():
    if func_name_var.name_hint == "main": continue
    elif func.attrs["Compiler"]=="imcflow":
      ImcflowDeviceConfig().TensorEdgeListDict[func_name_var.name_hint] = _Visitor().getTensorEdgeList(func_name_var, func)
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
  HwMapping = ImcflowDeviceConfig().HWNodeMap
  NocPaths = ImcflowDeviceConfig().NoCPaths
  IMCECOL = ImcflowDeviceConfig.IMCE_W_NUM
  for func_name_var, func in mod.functions.items():
    if func_name_var.name_hint == "main": continue
    elif func.attrs["Compiler"]=="imcflow":
      NocPaths[func_name_var.name_hint] = {}
      # tensor edge to path entry
      # if src graph is Var, hw_node_id is left-most inode of dst graph node hw_node_id
      # we will add instruction path to each IMCE node
      TensorEdgeList_ = ImcflowDeviceConfig().TensorEdgeListDict[func_name_var.name_hint]
      for tensor_edge in TensorEdgeList_:
        SrcTensorID = tensor_edge.src_id
        DstTensorID = tensor_edge.dst_id
        SplitIdx = tensor_edge.split_idx
        SrcGraphNode = CustomIDToNode()[getInnerNodeID(SrcTensorID.graph_node_id)]
        DstGraphNode = CustomIDToNode()[getInnerNodeID(DstTensorID.graph_node_id)]
        if isinstance(SrcGraphNode, (Var, Constant)):
          # else, map src node into inode
          DstHwNodeID = HwMapping[getOuterNodeID(DstTensorID.graph_node_id)]
          # if "inode" not in DstHwNodeID:
          if not DstHwNodeID.is_inode():
            InodeID = NodeID.from_inode_coord(NodeID.to_coord(DstHwNodeID)[0])
            NocPaths[func_name_var.name_hint][tensor_edge] = (
              (InodeID, DstHwNodeID, SplitIdx)
            )
            HwMapping[SrcTensorID.graph_node_id] = InodeID
        elif hasattr(DstGraphNode, "attrs") and hasattr(DstGraphNode.attrs, "Compiler") and DstGraphNode.attrs["Compiler"] == "imcflow" :
          # if this tensoredge is the final edge directly connected to host (= if destination is function)
          SrcHwNodeID = HwMapping[getOuterNodeID(SrcTensorID.graph_node_id)]
          InodeID = NodeID.from_inode_coord(NodeID.to_coord(SrcHwNodeID)[0])
          NocPaths[func_name_var.name_hint][tensor_edge] = (
            (HwMapping[getOuterNodeID(SrcTensorID.graph_node_id)], InodeID, SplitIdx)
          )
          HwMapping[DstTensorID.graph_node_id] = InodeID
        else:
          NocPaths[func_name_var.name_hint][tensor_edge] = (
            (HwMapping[getOuterNodeID(SrcTensorID.graph_node_id)], HwMapping[getOuterNodeID(DstTensorID.graph_node_id)], SplitIdx)
          )

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
    def run_(self, func):
      class _MemoryAllocator(tvm.relay.ExprVisitor):
        """
          Target Operators:
            conv2d, bias_add, batch_norm, relu, add and fused versions
            split, concat
        """
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

        def traverse_func(self, func):
            self.visit(func)
            self.allocate()
            return self.DataBlockDict

        def is_inode_in_edge(self, edge):
          dst_hw_node_id = None
          src_hw_node_id = None
          is_inode = False
          inode_tensorid = None

          #dst id
          if edge.dst_id.graph_node_id in self.hwnodemap:
            dst_hw_node_id = self.hwnodemap[edge.dst_id.graph_node_id]
            if dst_hw_node_id.name.startswith("inode"):
              # determine whether inode is included in the edge and which id it is.
              is_inode = True
              inode_tensorid = edge.dst_id

          #src id
          if edge.src_id.graph_node_id in self.hwnodemap:
            src_hw_node_id = self.hwnodemap[edge.src_id.graph_node_id]
            if src_hw_node_id.name.startswith("inode"):
              # determine whether inode is included in the edge and which id it is.
              is_inode = True
              inode_tensorid = edge.src_id

          return is_inode, inode_tensorid

        def find_edge_from_list(self, call):
          # find edges that call node belongs, and find valid edge which has inode
          tensor_edge_list = self.TensorEdgeList
          graph_node_id = getNodeID(call)

          def matches_node_id(node_id):
            if isinstance(node_id, int):
              return node_id == graph_node_id
            elif isinstance(node_id, tuple):
              return graph_node_id in node_id
            return False

          edges = []
          for edge in tensor_edge_list:
            if matches_node_id(edge.dst_id.graph_node_id) and self.is_inode_in_edge(edge)[0]:
              edges.append(edge)

          return edges

        def allocate(self):
          for edge, mem_block in self.DataBlockDict.items():
            if mem_block.size is None:
              raise ValueError("Memory size cannot be none.")

            # add tensor edge info to ImcflowDeviceConfig, but as a placeholder for now.
            # the policy info and fifo id will be set later in PolicyTableGenerator.
            ImcflowDeviceConfig().add_tensor_edge_info(edge, TensorEdgeInfo(data_block=mem_block))

            _, inode_tensorid = self.is_inode_in_edge(edge)
            hw_node_id = self.hwnodemap[inode_tensorid.graph_node_id]
            inode_name = hw_node_id.name # ex) inode_3

            if inode_tensorid.tensor_type == "weight":
              ImcflowDeviceConfig().MemLayout[f"{inode_name}_data"].allocate_allow_overlap(mem_block)
            else:
              ImcflowDeviceConfig().MemLayout[f"{inode_name}_data"].allocate(mem_block)

          return

        def visit_function(self, fn):
          def get_size(edge, call):
            size = None
            op_found = False
            arg_node = call.body
            # arg_node_shape = call.body.type_args[0].shape
            if isinstance(arg_node, Tuple):
              # find field of current edge
              for i, field in enumerate(arg_node.fields):
                if isinstance(edge.src_id.graph_node_id, tuple):
                  if getNodeID(field) in edge.src_id.graph_node_id:
                    arg_node = field
                    func_ret_shape = call.ret_type.fields[i].shape
                else:
                  if getNodeID(field) == edge.src_id.graph_node_id:
                    arg_node = field
                    func_ret_shape = call.ret_type.fields[i].shape
            else:                                 
              func_ret_shape = call.ret_type.shape
            # traverse until we find the final operation
            while op_found is False:
              if isinstance(arg_node, Call):
                arg_op = arg_node.op
                if isinstance(arg_op, Function):
                  # if arg is Composite node
                  arg_node = arg_node.op.body
                elif arg_op == op.get("qnn.imcflow_min_max_quantize") or arg_op == op.get("nn.imcflow_qconv"):
                  size = func_ret_shape[2] * func_ret_shape[3] * 4 #TODO: check again!
                  op_found = True
                elif arg_op == op.get("concatenate"):
                  arg_node = arg_node.args[0]
                elif arg_op in [op.get("multiply"), op.get("add"), op.get("nn.bias_add"), op.get("nn.relu"), op.get("imcflow.fused_batch_norm")]:
                  size = func_ret_shape[2] * func_ret_shape[3] * math.ceil(int(func_ret_shape[1])/16) #TODO: check again!
                  op_found = True
                else:
                  raise ValueError("Undefined operation!")
              elif isinstance(arg_node, Tuple):
                arg_node = arg_node.fields[0]
              elif isinstance(arg_node, Function):
                arg_node = arg_node.body
              else:
                raise ValueError("Undefined operation!")

            if size is not None:
              # imcflow word width = 256 bit
              size = int(size) * 256 / 8 #unit: bytes
            return size

          super().visit_function(fn)

          if hasattr(fn.attrs, "Compiler") and fn.attrs["Compiler"]=="imcflow":
            edges = self.find_edge_from_list(fn)
            for edge in edges:
              size = get_size(edge, fn)
              if size is not None:
                inode_tensorid = self.is_inode_in_edge(edge) # find which one is inode
                datablock = DataBlock(inode_tensorid[1], None)
                datablock.set_size(size)
                self.DataBlockDict[edge] = datablock
              else:
                raise ValueError("There should be at least one edge connected to function node.")

        def visit_call(self, call):
          def get_size(edge, call):
            size = None

            def get_op_from_id(node_id):
                if isinstance(node_id, int):
                    return self.name_dict[node_id]
                elif isinstance(node_id, tuple):
                    return self.name_dict[node_id[1]]
                else:
                  raise ValueError("CustomIDToName does not have this node id.")

            def get_arg_idx(edge, call):
              # find arg index from call by comparing edge's tensorid
              idx = None
              shape = None
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
                else:
                  if src_id == edge.src_id.graph_node_id:
                    idx = i
                    shape = call.type_args[idx].shape

                # Check if `dst_id` matches the source node in `edge`
                # this is only for the case where src node is Var node, because customID of Var node in subfunction is not the same one in tensoredge.
                if isinstance(edge.dst_id.graph_node_id, tuple):
                  if dst_id in edge.dst_id.graph_node_id and isinstance(arg, Var):
                    idx = i
                    shape = call.type_args[idx].shape

              return idx, shape

            src_op = get_op_from_id(edge.src_id.graph_node_id)
            dst_op = call.op

            #find which argument index this edge correspond to find corresponding shape by type_args.shape
            arg_idx, arg_shape = get_arg_idx(edge, call)

            # calculate size for inode memory allocation
            _, inode_id = self.is_inode_in_edge(edge)
            IsSrcInode = True if edge.src_id == inode_id else False
            IsDstInode = True if edge.dst_id == inode_id else False

            if arg_idx is not None:
                if src_op == "Op(split)":
                  # when first node of subgraph is split, memoryblock is already allocated by (src: var -> dst: split) case.
                  size = -1
                elif dst_op == op.get("split"):
                  # if split, same as conv2d
                  if isinstance(call.args[arg_idx], Var):
                    size = arg_shape[2] * arg_shape[3] * 4 #TODO: check again!
                elif dst_op in [op.get("multiply"), op.get("add"), op.get("nn.bias_add"), op.get("nn.relu"), op.get("imcflow.fused_batch_norm")]:
                  if isinstance(call.args[arg_idx], Constant):
                    size = math.ceil(int(arg_shape[0]) / 16)
                  elif isinstance(call.args[arg_idx], Var):
                    size = arg_shape[2] * arg_shape[3] * math.ceil(int(arg_shape[1])/16)
                elif dst_op == op.get("nn.imcflow_qconv"):
                  if isinstance(call.args[arg_idx], Var): # input var
                    size = arg_shape[2] * arg_shape[3] * 4 #TODO: check again!
                  elif isinstance(call.args[arg_idx], Constant):# const(weight)
                    size = 256 #TODO: check again!
                elif dst_op == op.get("divide"):
                  if isinstance(call.args[arg_idx], Constant):
                    size = 1 # TODO: check again!
                elif dst_op == op.get("qnn.imcflow_min_max_quantize"):
                  if isinstance(call.args[arg_idx], Constant):
                    size = 1
                  elif isinstance(call.args[arg_idx], Var):
                    size = arg_shape[2] * arg_shape[3] * math.ceil(int(arg_shape[1])/16) #TODO: check again!
                else:
                  raise ValueError("Undefined oeration!")

            if size is not None:
              # imcflow word width = 256 bit
              size = int(size) * 256 / 8 #unit: bytes
            else:
              raise ValueError("Size calculation error!")

            return size

          super().visit_call(call)

          IsSupportedOp = isinstance(call.op, tvm.ir.Op) and call.op.name in ImcflowDeviceConfig.SUPPORTED_OPS

          if IsSupportedOp:
            edges = self.find_edge_from_list(call)
            for edge in edges:
              size = get_size(edge, call)
              if size > 0:
                inode_tensorid = self.is_inode_in_edge(edge) # find which one is inode
                datablock = DataBlock(inode_tensorid[1], None)
                datablock.set_size(size)
                self.DataBlockDict[edge] = datablock

        def visit_tuple_getitem(self, op):
          super().visit_tuple_getitem(op)

        def visit_tuple(self, op):
          super().visit_tuple(op)

      _MemoryAllocator().traverse_func(func)
      return func
    
    def run(self, mod):
      for _, func in mod.functions.items():
        if isinstance(func, relay.Function) and hasattr(func.attrs, "Compiler") and func.attrs["Compiler"]=="imcflow":
          self.run_(func)

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
                      edgeinfo = ImcflowDeviceConfig().get_tensor_edge_info(edge)
                      edgeinfo.set_policy_info(router_entry_list)

                      if edge.src_id.tensor_type == "odata":
                        # get src node name from CustomIDToName
                        if isinstance(edge.dst_id.graph_node_id, tuple):
                          dst_node_name = ID_dict[edge.dst_id.graph_node_id[1]]
                        else:
                          dst_node_name = ID_dict[edge.dst_id.graph_node_id]

                        if dst_node_name == "Op(nn.imcflow_qconv)":
                          # if src is input of qconv, FIFO ID = 0
                          edgeinfo.set_fifo_id(0)
                        else:
                          # if not, FIFO ID = 2~6
                          edgeinfo.set_fifo_id(fifo_id_cnt[dest_node])

                          fifo_id_cnt[dest_node] = fifo_id_cnt[dest_node] + 1
                          if fifo_id_cnt[dest_node] >= 8:
                            raise ValueError("FIFO ID cannot be over 7!")

                      elif edge.src_id.tensor_type in ["odata", "weight", "bias", "fused_scale", "fused_bias", "min", "max", "threshold", "scale"]:
                        # if const, FIFO ID = 1
                        edgeinfo = TensorEdgeInfo(router_entry_list, None, 1)
                        ImcflowDeviceConfig().add_tensor_edge_info(edge, edgeinfo)
                      else:
                        raise ValueError("Wrong tensor type!")

                  else: # Instruction edge
                      # meminfo = get_meminfo(edge) # decided to erase MemoryBlock in EdgeInfo
                      edgeinfo = InstEdgeInfo(router_entry_list, None)
                      ImcflowDeviceConfig().add_inst_edge_info(edge, edgeinfo)

        def allocate(self):
          # Allocate memory for policy tables
          for node_id, policy_table in self.Policytable.items():
            if len(policy_table) == 0:
                continue
            mem_size = len(policy_table) * 32
            mem_block = DataBlock(f"{node_id.name}_policy", mem_size)
            inode_id = node_id.master() if node_id.is_imce() else node_id
            ImcflowDeviceConfig().MemLayout[f"{inode_id.name}_data"].allocate(mem_block)

        def traverse_func(self, func):
            # traverse input function by visit() to make PathDict and generate policy table for it
            self.generate_policy_table()
            self.add_EdgeInfo()
            self.allocate()
            return self.Policytable

      # Returns list of (GlobalVar, Function) pairs sorted alphabetically by function name
      items = mod.functions_items()
      function_names = [item[0].name_hint for item in items]

      num_func = len(function_names)
      for i in range(num_func):
        if function_names[i]=="main": continue
        elif mod[function_names[i]].attrs["Compiler"]=="imcflow":
          self.PolicyTable_2D[function_names[i]] = _PolicyTableGenerator(self.NoCPaths[function_names[i]]).traverse_func(mod[function_names[i]])
          for x in self.PolicyTable_2D[function_names[i]]:
            print(x)

      return func

@relay.transform.function_pass(opt_level=0)
class PackingInserter:
    def __init__(self):
      pass

    def transform_function(self, func, mod, ctx):
      class _PackingInserter(tvm.relay.ExprMutator):
        def __init__(self):
            super().__init__()
            self.func_param_map = {}
            self.InSubFunc = False
            self.VarOriginShape = {}

        def visit_var(self, var):
          if self.InSubFunc:
            # check var shape is 1D. If so, unpack it.
            input_node = self.func_param_map[var.name_hint]
            input_node_type = _get_type(mod, input_node)
            if len(input_node_type.shape) == 1:
              new_var = relay.Var(var.name_hint, relay.TensorType(input_node_type.shape, input_node_type.dtype))
              self.VarOriginShape[var.name_hint] = var.type_annotation.shape
              return new_var
              # return imcflow_unpacking(new_var, var.type_annotation.shape[0], "float32")
            else:
              return super().visit_var(var)
          else:
            return super().visit_var(var)

        def visit_function(self, func):
          if self.InSubFunc:
            new_params = [self.visit(param) for param in func.params]
            new_body = self.visit(func.body)

            # check last node is related to quantization like qnn.imcflow_min_max_quantize
            # check also concat of quantized data. If so, pack it.
            def _addPacking(node, parentNode=None):
              parentNode = node if parentNode is None else parentNode
              if isinstance(node, relay.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name in ImcflowDeviceConfig.QAUNT_OPS:
                Shape1D = 1
                # for shape in node.type_args[0].shape:
                for shape in node.checked_type.shape:
                  Shape1D = Shape1D * shape
                node = imcflow_packing(parentNode, [Shape1D], "int8")
              elif isinstance(node, relay.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "concatenate":
                # check if all input nodes are quantized. If so, pack it.
                AllQuant = True
                for arg in node.args:
                  if isinstance(arg, relay.Call) and isinstance(arg.op, tvm.ir.Op) and arg.op.name not in ImcflowDeviceConfig.QAUNT_OPS:
                    AllQuant = False
                if AllQuant:
                  Shape1D = 1
                  for shape in _get_type(mod, node).shape:
                    Shape1D = Shape1D * shape
                  node = imcflow_packing(parentNode, [Shape1D], "int8")
              elif isinstance(node, relay.Call) and isinstance(node.op, relay.Function):
                node = _addPacking(node.op.body, parentNode)

              return node

            new_body = _addPacking(new_body)
            new_func_ret_type = _get_type(mod, new_body)
            new_type_params = func.type_params
            return relay.Function(new_params, new_body, new_func_ret_type, new_type_params, func.attrs)
          else:
            new_params = [self.visit(param) for param in func.params]
            new_body = self.visit(func.body)
            new_ret_type = _get_type(mod, new_body)
            return relay.Function(new_params, new_body, new_ret_type, func.type_params, func.attrs)

        def visit_global_var(self, gvar):
          return relay.GlobalVar(gvar.name_hint)

        def visit_call(self, call):
          #post DFS
          new_args = [self.visit(arg) for arg in call.args]

          if self.InSubFunc:
            for idx, narg in enumerate(new_args):
              if isinstance(narg, relay.Var) and len(narg.type_annotation.shape) == 1:
                VarShape = self.VarOriginShape[narg.name_hint]
                new_args[idx] = imcflow_unpacking(narg, VarShape, "float32")
              else:
                new_args[idx] = narg

          if isinstance(call.op, relay.Function):
            InSubFunc = self.InSubFunc
            self.InSubFunc = False
            new_func = self.visit_function(call.op)
            self.InSubFunc = InSubFunc
            return Call(new_func, new_args, call.attrs, call.type_args, call.span)

          if isinstance(call.op, relay.GlobalVar):
             #make var to node mapping
             args = new_args
             params = mod[call.op.name_hint].params
             self.func_param_map = {param.name_hint: arg for param, arg in zip(params, args)}
             self.InSubFunc = True
            #  new_op = self.visit(call.op)
             mod[call.op.name_hint] = self.visit(mod[call.op.name_hint])
             self.InSubFunc = False
             return Call(call.op, new_args, call.attrs, call.type_args, call.span)

          if call.op == op.get("nn.imcflow_qconv"):
            # OriginWeight = call.args[1]
            OriginWeight = new_args[1]
            Weight1D = relay.Constant(tvm.nd.array(OriginWeight.data.asnumpy().flatten().astype("int8")))
            NewWeight = imcflow_unpacking(Weight1D, OriginWeight.checked_type.shape, "float32")
            # NewInput = super().visit(call.args[0])
            NewInput = new_args[0]

            return imcflow_qconv2d(NewInput, NewWeight,
                                   **call.attrs)

          if call.op == op.get("imcflow.fused_batch_norm"):
            # convert dtype to int16
            # NewScale = relay.Constant(tvm.nd.array(call.args[1].data.asnumpy().astype("int16")))
            # NewBias = relay.Constant(tvm.nd.array(call.args[2].data.asnumpy().astype("int16")))
            NewScale = relay.Constant(tvm.nd.array(new_args[1].data.asnumpy().astype("int16")))
            NewBias = relay.Constant(tvm.nd.array(new_args[2].data.asnumpy().astype("int16")))
            return imcflow_batch_norm(new_args[0], NewScale, NewBias,1).astuple()

          if call.op == op.get("qnn.imcflow_min_max_quantize"):
            # convert dtype to int16
            # NewMin = relay.Constant(tvm.nd.array(call.args[1].data.asnumpy().astype("int16")))
            # NewMax = relay.Constant(tvm.nd.array(call.args[2].data.asnumpy().astype("int16")))
            NewMin = relay.Constant(tvm.nd.array(new_args[1].data.asnumpy().astype("int16")))
            NewMax = relay.Constant(tvm.nd.array(new_args[2].data.asnumpy().astype("int16")))
            return imcflow_min_max_quantize(new_args[0], NewMin, NewMax,1, "float32", "int16")

          if call.op == op.get("qnn.imcflow_nu_quantize"):
            # convert dtype to int16
            NewThreshold = relay.Constant(tvm.nd.array(new_args[1].data.asnumpy().astype("int16")))
            return imcflow_nu_quantize(new_args[0], NewThreshold,1, "float32", "int16")

          if call.op == op.get("qnn.simulated_quantize"):
            # convert dtype to int16
            Shape = _get_type(mod, new_args[0]).shape
            Shape1D = 1
            for shape in Shape:
              Shape1D = Shape1D * shape
            NewQuantCall = relay.qnn.simulated_quantize(new_args[0], new_args[1], new_args[2], **call.attrs)
            return imcflow_packing(NewQuantCall, [Shape1D], "int8")

          new_op = self.visit(call.op)
          return Call(new_op, new_args, call.attrs, call.type_args, call.span)

      new_func = _PackingInserter().visit(func)
      return new_func

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

# @relay.transform.function_pass(opt_level=0)
# class IDAssigner:
#     def transform_function(self, func, mod, ctx):
#       class _Visitor(tvm.relay.ExprMutator):
#         def __init__(self):
#           super().__init__()
#           self.Cnt = 0

#         def visit_call(self, call):
#           NewAttr = {}
#           if isinstance(call.op, Function):
#             # new_fn.attrs["CustomID"] = self.Cnt
#             for key in call.attrs.keys():
#               NewAttr[key] = call.attrs.get_str(key)
#             NewAttr["CustomID"] = self.Cnt
#             dattr = tvm.ir.make_node("DictAttrs", **NewAttr)
#             self.Cnt = self.Cnt + 1
#             new_fn = FunctionWithFields(call.op, list(call.op.params), call.op.body, call.op.ret_type, call.op.ty_params, dattr)
#             new_call_attrs = call.attrs
#           elif isinstance(call.op, Op):
#             # call.attrs["CustomID"] = self.Cnt
#             if call.attrs is not None:
#               for key in call.attrs.keys():
#                 NewAttr[key] = call.attrs.get_str(key)
#             NewAttr["CustomID"] = self.Cnt
#             dattr = tvm.ir.make_node("DictAttrs", **NewAttr)
#             new_fn = call.op
#             new_call_attrs = dattr
#             self.Cnt = self.Cnt + 1

#           new_args = [self.visit(arg) for arg in call.args]

#           return Call(new_fn, new_args, new_call_attrs, call.type_args, call.span)

#       print("-----------------------func--------------------")
#       print(func)
#       # _Visitor().visit(func)

#       return _Visitor().visit(func)

# def assignID(mod):
#   class _Visitor(tvm.relay.ExprMutator):
#     def __init__(self):
#       super().__init__()
#       self.Cnt = 0

#     def visit_call(self, call):
#       setattr(call, "CustomID", self.Cnt)
#       return call

#   vis = _Visitor()
#   for func_name in mod.functions:
#     mod[func_name] = vis.visit(mod[func_name])
#   return mod

# def printID(mod):
#   class _Visitor(tvm.relay.ExprVisitor):
#     def visit_call(self, call):
#       print(call.CustomID)
#       super().visit_call(call)

#   vis = _Visitor()
#   for func_name in mod.functions:
#     vis.visit(mod[func_name])

def constructUsefulMappings(mod):
  id_dict = HashToCustomID()
  name_dict = CustomIDToName()
  data = CustomIDToNode()
  class _Visitor(tvm.relay.ExprVisitor):
    def __init__(self):
      super().__init__()
      self.Cnt = 0

    def visit_call(self, call):
      id_dict[int(hash(call))] = self.Cnt
      name_dict[self.Cnt] = getNodeDebugID(call)
      data[id_dict[int(hash(call))]] = call
      self.Cnt = self.Cnt + 1
      super().visit_call(call)

    def visit_function(self, call):
      id_dict[int(hash(call))] = self.Cnt
      name_dict[self.Cnt] = "Function"
      data[id_dict[int(hash(call))]] = call
      self.Cnt = self.Cnt + 1
      super().visit_function(call)

    def visit_var(self, var):
      id_dict[int(hash(var))] = self.Cnt
      name_dict[self.Cnt] = var.name_hint
      data[id_dict[int(hash(var))]] = var
      self.Cnt = self.Cnt + 1
      super().visit_var(var)

    def visit_constant(self, const):
      id_dict[int(hash(const))] = self.Cnt
      name_dict[self.Cnt] = "Const"
      data[id_dict[int(hash(const))]] = const
      self.Cnt = self.Cnt + 1
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

  output_node = getOutputNodesOfFunc(func)
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



# @relay.transform.function_pass(opt_level=0)
class ImcflowBoundaryNodeMarker:
  """
  A pass that identifies boundary nodes between IMCFLOW and CPU execution domains.
  
  This pass traverses the graph to find Function nodes with "Compiler" attribute set to "imcflow",
  then:
  1. Marks the first Call node inside the function as in_node=True
  2. Marks the last Call node inside the function as out_node=True  
  3. Inserts packing nodes before calls to imcflow functions
  4. Inserts unpacking nodes after calls to imcflow functions
  """

  #TODO: constant layout transform
  
  def __init__(self):
    self.input_call_dict = {}
    self.output_call_dict = {}
    
  def transform_function(self, mod):
    """
    Transform the function to mark boundary nodes and insert packing/unpacking.
    
    This iterates through all functions in the module and processes IMCFLOW functions.
    """
    
    # Get all function items from the module
    items = mod.functions_items()
    function_names = [item[0].name_hint for item in items]
    
    # Process each IMCFLOW function to mark internal boundaries
    num_func = len(function_names)
    for i in range(num_func):
      if function_names[i] == "main":
        continue
      elif ("Compiler" in mod[function_names[i]].attrs and 
            mod[function_names[i]].attrs["Compiler"] == "imcflow"):
        print(f"Marking boundary nodes for IMCFLOW function: {function_names[i]}")
        mod[function_names[i]] = self._mark_imcflow_function_boundaries(mod[function_names[i]])
        mod[function_names[i]] = self._mark_and_transform_imcflow_qconv(mod[function_names[i]])
    
    # Transform the main function to insert packing/unpacking around imcflow calls
    mod = self._insert_packing_unpacking(mod)
    
    # return transformed_func
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
        Packed = np.zeros((out_blocks, in_blocks, 256, 8), dtype=np.int32)
        for i in range(8):
          # Shift each int4 value to its position (4 bits per value)
          # Mask to 4 bits (0xF) to ensure int4 range
          Packed += ((ToPack[:, :, :, :, i].astype(np.int32) & 0xF) << (i * 4))
        
        NewWeight = relay.Constant(tvm.nd.array(Packed))
        new_args = [call.args[0], NewWeight]
        new_type_args = [call.type_args[0], relay.TensorType(NewWeight.data.shape, "int32")]

        return Call(call.op, new_args, call.attrs, new_type_args, call.span)

      return call

    class _BoundaryMarker(relay.ExprMutator):
      def visit_call(self, call):
        new_call = super().visit_call(call)
        
        # Mark imcflow_qconv calls as both input and output nodes
        if isinstance(call.op, tvm.ir.Op) and call.op == op.get("nn.imcflow_qconv"):
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

  def _insert_packing_unpacking(self, mod):
    """
    Insert packing nodes before imcflow function calls and unpacking nodes after.
    
    This transforms the main function to insert packing/unpacking around calls to 
    functions with "Compiler" attribute set to "imcflow".
    """

    #TODO: insert bitpack/bitunpack, layout transform. later cancel
    
    # Run type inference on the module to ensure all nodes have checked_type
    # mod = relay.transform.InferType()(mod)
    
    class _LayoutTransformer(relay.ExprMutator):
      def __init__(self, module):
        super().__init__()
        self.module = module

      # Helper function to find the child node (consumer) of a parameter in the function body
      def find_child_node_of_param(self, func_body, param):
        """Find the node that directly uses the given parameter as an argument"""
        child_nodes = []
        
        class _ChildNodeFinder(relay.ExprVisitor):
          def visit_call(self, call):
            # Check if this call uses the parameter as one of its arguments
            for arg in call.args:
              if arg == param:
                child_nodes.append(call)
                break
            # Continue visiting children
            super().visit_call(call)
        
        _ChildNodeFinder().visit(func_body)
        return child_nodes[0] if child_nodes else None

      # Helper function to find the parent node (producer) that generates the function's output
      def find_parent_node_of_output(self, func_body):
        """Find the node that directly produces the function's output (right before return)"""
        # Handle different body types
        if isinstance(func_body, relay.Call):
          # If body is directly a Call, that's our parent node
          return func_body
        elif isinstance(func_body, relay.TupleGetItem):
          # If body is TupleGetItem, find the call that produces the tuple
          return self.find_parent_node_of_output(func_body.tuple_value)
        elif isinstance(func_body, relay.Tuple):
          # If body is a Tuple, find the calls that produce each field
          parent_nodes = []
          for field in func_body.fields:
            parent_node = self.find_parent_node_of_output(field)
            parent_nodes.append(parent_node)
          return parent_nodes
        else:
          # For other types, return None
          return None        

      def visit_call(self, call):
        # First transform arguments recursively
        new_args = [self.visit(arg) for arg in call.args]
        
        # Check if this is a call to an imcflow function
        if isinstance(call.op, relay.GlobalVar):
          # Check if the target function has "Compiler" attribute set to "imcflow"
          target_func = self.module[call.op.name_hint]
          if ("Compiler" in target_func.attrs and target_func.attrs["Compiler"] == "imcflow"):
            print(f"Inserting packing/unpacking around imcflow function call: {call.op.name_hint}")
            
            # step1: Insert node before the call
            packed_args = []
            for i, arg in enumerate(new_args):
              # Get the receiver (child node) of this argument
              # The receiver is the parameter in the target imcflow function that receives this argument
              receiver_param = target_func.params[i]
              receiver_type = receiver_param.checked_type
              
              # Get the shape and dtype from the receiver
              if isinstance(arg, TupleGetItem):
                # NOTE: This is a hack. In resnet8, if arg is a TupleGetItem, then imcflow function precede right before it.
                packed_arg = arg
              elif isinstance(receiver_type, TensorType):
                # Find the child node inside target_func that uses this receiver_param
                child_node = self.find_child_node_of_param(target_func.body, receiver_param)
                if child_node.op == op.get("nn.imcflow_qconv"):
                  packed_arg = imcflow_4d_to_qconv_input(arg)
                elif child_node.op == op.get("qnn.imcflow_min_max_quantize"):
                  packed_arg = relay.layout_transform(arg, "NCHW", "NCHWc16")
                else: # Vector operations
                  packed_arg = relay.layout_transform(arg, "NCHW", "NCHWc16")
              else:
                raise ValueError("Unsupported receiver type for packing")
              
              # Insert packing node
              packed_args.append(packed_arg)

            # Create the call with packed arguments
            imcflow_call = relay.Call(call.op, packed_args, call.attrs, call.type_args, call.span)

            # step2: Insert node after the call
            # Get the return type of the imcflow function to determine unpacking shape
            return_type = target_func.ret_type                        
            if isinstance(return_type, relay.TupleType):
              # NOTE: This is a hack. In resnet8, if return_type is a tuple, then imcflow function follows right after it.
              unpacked_result = imcflow_call
            elif isinstance(return_type, TensorType):
              # Find the parent node inside target_func that produces the output
              parent_node = self.find_parent_node_of_output(target_func.body)
              if parent_node.op == op.get("qnn.imcflow_min_max_quantize"):
                unpacked_result = imcflow_mmquant_out_to_4d(imcflow_call, return_type.shape, str(return_type.dtype))
              elif parent_node.op == op.get("nn.imcflow_qconv"):
                raise ValueError("imcflow_qconv cannot be the last node in an imcflow function")
              else: # Vector operations
                unpacked_result = relay.layout_transform(imcflow_call, "NCHWc16", "NCHW")
            else:
              raise ValueError("Unsupported return type for unpacking")
            
            return unpacked_result
        
        # For non-imcflow function calls, proceed normally
        new_op = self.visit(call.op)
        return relay.Call(new_op, new_args, call.attrs, call.type_args, call.span)
    
    inserter = _LayoutTransformer(mod)
    new_main = inserter.visit(mod["main"])
    mod.update_func(mod.get_global_var("main"), new_main)
    return mod
def constructDataBlockDict(mod):
  for func_name_var, func in mod.functions.items():
    if func_name_var.name_hint == "main": continue
    elif func.attrs["Compiler"]=="imcflow":
      ImcflowDeviceConfig().get_data_block_dict(func)
