import tvm
from tvm import relay
from tvm.relay import transform, op
from tvm.relay.ty import TupleType, TensorType
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.function import Function, FunctionWithFields
from tvm.relay.expr import (Call, GlobalVar, TupleGetItem, const, Let, Var, If, Tuple, Constant)
from tvm.relay.expr import RefCreate, RefRead, RefWrite
from tvm.relay.adt import Constructor, Match, Clause
from tvm.contrib.imcflow import ImcflowDeviceConfig, TensorEdge, TensorID, NodeID, TensorEdgeInfo, InstEdgeInfo, RouterEntry
from tvm.ir import Op
from tvm.relay.op.contrib.imcflow import HashToCustomID, CustomIDToName, CustomIDInFunc, CustomIDToNode

import math
from copy import deepcopy
import re
from dataclasses import dataclass
from enum import Enum

def getNodeID(node):
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

def getFlagNodeID(node):
  if isinstance(node, tuple):
    return node[0]
  else:
    return node

@relay.transform.function_pass(opt_level=0)
class ConvSplitToAtom:
    def __init__(self, OldParamDict):
      self.OldParamDict = OldParamDict
      self.NewParamDict = {}

    def transform_function(self, func, mod, ctx):
      RemoveTargets = []
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

        def removeParamVar(self, Var):
          self.DeleteArgs.append(Var)
          self.NewParamDict.pop(Var.name_hint)

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

          if not ImcflowDeviceConfig.is_supported_kernel(KH, KW):
            return expr

          for PostNode in PostProcess:
            assert PostNode.op in [op.get("nn.bias_add"), op.get("nn.relu"), op.get("nn.batch_norm")], "Unsupported post process node"

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

          # Nested weight splits for each out channel slice
          # split_conv_weights = []
          # split_weights = relay.op.transform.split(expr.args[1], indices_or_sections=oc_sections, axis=0) if IsOCSplited else [expr.args[1]]
          # for oc_id in range(oc_split_num):
          #     weight_slice = relay.op.transform.split(split_weights[oc_id], indices_or_sections=ic_sections, axis=1) if (IsICSplited and (not IsDepthWise)) else [split_weights[oc_id]]
          #     split_conv_weights.append(weight_slice)

          # split weight and make New params
          split_conv_weights = [[None for _ in range(ic_split_num if (not IsDepthWise) else 1)] for _ in range(oc_split_num)]
          # self.DeleteArgs.append(expr.args[1])
          # self.NewParamDict.pop(expr.args[1].name_hint)
          self.removeParamVar(expr.args[1])
          for oc_id in range(oc_split_num):
            oc_size = out_ch_limit if (oc_id * out_ch_limit) + out_ch_limit - 1 < OC else OC % out_ch_limit
            for ic_id in range(ic_split_num if not IsDepthWise else 1):
              if IsDepthWise:
                ic_size = 1
              else:
                ic_size = in_ch_limit if (ic_id * in_ch_limit) + in_ch_limit - 1 < IC else IC % in_ch_limit
              SplitParam = relay.Var(f"{expr.args[1].name_hint}_oc{oc_id}_ic{ic_id}", relay.TensorType([oc_size, ic_size, KH, KW], dtype=expr.args[1].type_annotation.dtype))
              split_conv_weights[oc_id][ic_id] = SplitParam

              OldParam = self.OldParamDict[expr.args[1].name_hint]
              NewData = OldParam.numpy()[oc_id*out_ch_limit:(oc_id*out_ch_limit)+oc_size, ic_id*in_ch_limit:(ic_id*in_ch_limit)+ic_size, :, :]
              self.addParamVar(SplitParam, tvm.nd.array(NewData, device=OldParam.device))
              # self.NewParamDict[SplitParam.name_hint] = tvm.nd.array(NewData, device=OldParam.device)
              # self.AddArgs.append(SplitParam)

          # Create conv2d calls for each input-output channel slice
          conv_nodes = {}
          for oc_id in range(oc_split_num):
              oc_size = out_ch_limit if (oc_id * out_ch_limit) + out_ch_limit - 1 < OC else OC % out_ch_limit
              for ic_id in range(ic_split_num if not IsDepthWise else 1):
                  ic_size = in_ch_limit if (ic_id * in_ch_limit) + in_ch_limit - 1 < IC else IC % in_ch_limit
                  conv_nodes[(oc_id, ic_id)] = relay.nn.conv2d(
                      split_inputs[ic_id] if (not IsDepthWise) else split_inputs[oc_id],
                      split_conv_weights[oc_id][ic_id],
                      channels=oc_size,
                      kernel_size=(KH, KW),
                      strides=expr.attrs.strides,
                      padding=expr.attrs.padding,
                      data_layout=expr.attrs.data_layout,
                      kernel_layout=expr.attrs.kernel_layout,
                      groups=1 if not IsDepthWise else oc_size
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
                if PostNode.op == op.get("nn.bias_add"):
                  self.removeParamVar(PostNode.args[1])
                elif PostNode.op == op.get("nn.batch_norm"):
                  for i in range(1, 5):
                    self.removeParamVar(PostNode.args[i])

                for oc_id in range(oc_split_num):
                  oc_size = out_ch_limit if (oc_id * out_ch_limit) + out_ch_limit - 1 < OC else OC % out_ch_limit
                  if PostNode.op == op.get("nn.bias_add"):
                    ParamOldName = PostNode.args[1].name_hint
                    ParamNewName = f"{ParamOldName}_oc{oc_id}"
                    ParamNewType = relay.TensorType([oc_size], dtype=PostNode.args[1].type_annotation.dtype)
                    SplitParam = relay.Var(ParamNewName, ParamNewType)
                    OldParam = self.OldParamDict[ParamOldName]
                    NewData = OldParam.numpy()[oc_id*out_ch_limit:(oc_id*out_ch_limit)+oc_size]
                    self.addParamVar(SplitParam, tvm.nd.array(NewData, device=OldParam.device))
                    post_nodes[oc_id] = relay.nn.bias_add(post_nodes[oc_id], SplitParam, PostNode.attrs.axis)
                  elif PostNode.op == op.get("nn.relu"):
                    post_nodes[oc_id] = relay.nn.relu(post_nodes[oc_id])
                  elif PostNode.op == op.get("nn.batch_norm"):
                    NewParams = []
                    for i in range(1, 5):
                      ParamOldName = PostNode.args[i].name_hint
                      ParamNewName = f"{ParamOldName}_oc{oc_id}"
                      ParamNewType = relay.TensorType([oc_size], dtype=PostNode.args[i].type_annotation.dtype)
                      SplitParam = relay.Var(ParamNewName, ParamNewType)
                      OldParam = self.OldParamDict[ParamOldName]
                      NewData = OldParam.numpy()[oc_id*out_ch_limit:(oc_id*out_ch_limit)+oc_size]
                      self.addParamVar(SplitParam, tvm.nd.array(NewData, device=OldParam.device))
                      NewParams.append(SplitParam)
                    # post_nodes[oc_id] = relay.TupleGetItem(relay.nn.batch_norm(post_nodes[oc_id], *NewParams), 0)
                    post_nodes[oc_id] = relay.nn.batch_norm(post_nodes[oc_id], *NewParams)[0]

              concat_node = relay.op.concatenate([post_nodes[oc_id] for oc_id in range(oc_split_num)], axis=1)
          else:
              concat_node = add_nodes[0]
              # self.IsSplitedPostNode.extend([True for _ in range(len(PostProcess))])

          return concat_node


        def visit_call(self, call):
          if call.op == op.get("nn.conv2d"):
            PostProcess = self.PostProcess[:]
            self.PostProcess = []
            NewCall = super().visit_call(call)
            NewCall = self.split_and_optimize_conv2d(NewCall, mod, PostProcess)
            return NewCall
          elif call.op in [op.get("nn.bias_add"), op.get("nn.relu"), op.get("nn.batch_norm")]:
            self.PostProcess.append(call)
            NewCall = super().visit_call(call)
            if hasattr(call, "ShouldDelete"):
              return relay.Tuple([NewCall.args[0]]) if call.op == op.get("nn.batch_norm") else NewCall.args[0]
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

      NewFunc = relay.Function(NewArgs, NewFunc.body)
      NewFunc = _RedundantTupleRemover().visit(NewFunc)

      return NewFunc

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

def getSplitConcatDepsRegions(func):
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

@relay.transform.function_pass(opt_level=0)
class AnnotGenerator:
    def __init__(self):
      self.RegionList = []

    def transform_function(self, func, mod, ctx):
      RegionList = []

      class _Annotator(tvm.relay.ExprVisitor):
        """
          Target Operators:
            conv2d, bias_add, batch_norm, relu, add and fused versions
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

        def getCost(self, call):
          if not isinstance(call, Call):
             return 0

          IsComposite = isinstance(call.op, relay.Function) and "Composite" in call.op.attrs and re.match(r"imcflow\..*", call.op.attrs["Composite"])
          IsSupportedOp = isinstance(call.op, tvm.ir.Op) and call.op.name in ["nn.conv2d", "nn.bias_add", "nn.batch_norm", "nn.relu", "add", "split", "concatenate"]
          IsSuperNode = isinstance(call.op, relay.GlobalVar) and re.match(r"imcflow_.*", mod[call.op].attrs["Compiler"])
          IsNoCostCall = isinstance(call.op, tvm.ir.Op) and call.op.name in ["split", "concatenate"]

          class _CostVisitor(tvm.relay.ExprVisitor):
            def __init__(self, getCostFunc):
              super().__init__()
              self.Cost = 0
              self.getCost = getCostFunc

            def visit(self, expr):
              self.Cost = self.Cost + self.getCost(expr)
              super().visit(expr)

            def visit_call(self, call):
              if isinstance(call.op, relay.GlobalVar) and re.match(r"imcflow_.*", mod[call.op].attrs["Compiler"]):
                self.visit(call.op)
              for a in call.args:
                self.visit(a)

          if IsNoCostCall:
            return 0

          if IsComposite or IsSupportedOp:
            return 1

          if IsSuperNode:
            obj = _CostVisitor(self.getCost)
            obj.visit(mod[call.op].body)
            return obj.Cost

        def visit_call(self, call):
          # post DFS search
          for a in call.args:
              self.visit(a)

          # check this node is for imcflow
          IsComposite = isinstance(call.op, relay.Function) and "Composite" in call.op.attrs and re.match(r"imcflow\..*", call.op.attrs["Composite"])
          IsSupportedOp = isinstance(call.op, tvm.ir.Op) and call.op.name in ["nn.conv2d", "nn.bias_add", "nn.batch_norm", "nn.relu", "add", "split", "concatenate"]
          IsSuperNode = isinstance(call.op, relay.GlobalVar) and re.match(r"imcflow_.*", mod[call.op].attrs["Compiler"])

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
                    CandidateRegions.pop(InputRegion)
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
      if mod["main"] == func:
        _Annotator().visit(func)

      self.RegionList = RegionList

      return func

@relay.transform.function_pass(opt_level=0)
class NodeMapper:
    def __init__(self):
      # self.MappingDict_2D = {}
      self.MappingDict = {}

    def transform_function(self, func, mod, ctx):
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

        def traverse_func(self, func):
            self.visit(func)
            return self.MappingDict

        def visit_call(self, call):
          # post DFS search
          # traverse child node
          for a in call.args:
              self.visit(a)

          # check constraint and map imcflow node
          if self.MappingDict:
              # last_child_mapping, _ = list(self.MappingDict.items())[-1][1]
              last_child_mapping = list(self.MappingDict.items())[-1][1]
          else:
              last_child_mapping = None

          #for debugging
          indicator = getNodeDebugID(call)
          # if hasattr(call.op, "attrs"):
          #   indicator = call.op.attrs["Composite"]
          # else:
          #   indicator = call.op

          # check if this node is
          IsConcat = isinstance(call.op, tvm.ir.Op) and call.op.name in ["concatenate"]
          IsSplit = isinstance(call.op, tvm.ir.Op) and call.op.name in ["split"]
          if IsConcat:
              if last_child_mapping is None:
                  raise ValueError("split or concatenate should have at least 1 child node")
              self.MappingDict[getNodeID(call)] = last_child_mapping
              # self.MappingDict[getNodeID(call)] = (last_child_mapping, indicator)
          elif IsSplit:
              if last_child_mapping is None:
                  self.MappingDict[getNodeID(call)] = NodeID.from_inode_coord(self.inode_index)
                  # self.MappingDict[getNodeID(call)] = f"inode_{self.inode_index}"
                  # self.MappingDict[int(hash(call))] = (f"inode_{self.inode_index}", indicator)
                  self.inode_index -= 1
              else:
                  # self.MappingDict[int(hash(call))] = (last_child_mapping, indicator)
                  self.MappingDict[getNodeID(call)] = last_child_mapping
          else:
              self.MappingDict[getNodeID(call)] = NodeID.from_imce_coord(self.imce_index)
              # self.MappingDict[getNodeID(call)] = f"imce_{self.imce_index}"
              self.imce_index -= 1
              # self.MappingDict[int(hash(call))] = (f"imce_{self.imce_index}", indicator)

        def visit_tuple_getitem(self, op):
          super().visit_tuple_getitem(op)

        def visit_tuple(self, op):
          super().visit_tuple(op)

      # Returns list of (GlobalVar, Function) pairs sorted alphabetically by function name
      items = mod.functions_items()
      function_names = [item[0].name_hint for item in items]

      num_func = len(function_names)
      for i in range(num_func):
        if function_names[i]=="main": continue
          # _Nodemapper().visit(mod["main"])
        elif mod[function_names[i]].attrs["Compiler"]=="imcflow":
          self.MappingDict.update(_Nodemapper().traverse_func(mod[function_names[i]]))

      ImcflowDeviceConfig().HWNodeMap = self.MappingDict

      # # find all regions
      return func

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
        # self.SubFunctionMapping = None
        self.SubFunctionNodeID = None
        self.VarProperties = {}

    def getInputGraphNodeID(self, node):
      if isinstance(node, Call):
        if isinstance(node.op, relay.Function) and "Composite" in node.op.attrs and re.match(r"imcflow\..*", node.op.attrs["Composite"]):
          return (getNodeID(node), getNodeID(node.op.body))
        else:
          return getNodeID(node)
      elif isinstance(node, Tuple):
        result = []
        for b in node.fields:
          result.append(self.getInputGraphNodeID(b))
        return result
      elif isinstance(node, TupleGetItem):
          return self.getInputGraphNodeID(node.tuple_value)
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
          TensorEdge(TensorID.get(SrcGraphNodeID, SrcTag),
                     TensorID.get(DstGraphNodeID, DstTag),
                     SplitIdx)
        )
      else:
        raise ValueError("Invalid input tensor id pair")

    def visit_function(self, fn):
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
        IsSupportedOp = isinstance(call.op, tvm.ir.Op) and call.op.name in ["nn.conv2d", "nn.bias_add", "nn.batch_norm", "nn.relu", "add", "split", "concatenate"]

        if not IsComposite and not IsSupportedOp:
          raise ValueError("Unsupported operator detected. please check.")

        # visit composite function
        # we will collect Var Nodes usage and its properties
        def _processInputNode(SrcGraphNode, SrcTag, DstGraphNodeID, DstTag, SplitIdx):
          if not self.InSubFunction:
            InputGraphNodeID = self.getInputGraphNodeID(SrcGraphNode)
            self.appendToTensorEdgeList(InputGraphNodeID, DstGraphNodeID, SrcTag, DstTag, SplitIdx)
            return True
          else:
              if isinstance(SrcGraphNode, Var):
                self.VarProperties[SrcGraphNode]["src_tag"] = SrcTag
                self.VarProperties[SrcGraphNode]["dst_tag"] = DstTag
                self.VarProperties[SrcGraphNode]["dst_graph_node_id"] = DstGraphNodeID

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
            _processInputNode(call.args[0], "odata", DstGraphNodeID, "idata", self.getInputGraphNodeSplitIndex(call.args[0]))
          if call.op == op.get("concatenate"):
            _processInputNode(call.args[0], "odata", DstGraphNodeID, "idata", self.getInputGraphNodeSplitIndex(call.args[0]))
            # _processInputNode(call.args[0], lambda x: len(x) > 1, "data")
          if call.op == op.get("nn.conv2d"):
            _processInputNode(call.args[0], "odata", DstGraphNodeID, "idata", self.getInputGraphNodeSplitIndex(call.args[0]))
            _processInputNode(call.args[1], "weight", DstGraphNodeID, "weight", None)
            # _processInputNode(call.args[0], lambda x: len(x) == 1, "data")
            # InputNodeProperties = self.getInputTensorIDPair(call.args[1])
            # assert len(InputNodeProperties) == 1, "Conv2d should have only one weight node"
            # self.appendToPathList(InputNodeProperties, DstNodeProperty, "weight")
          if call.op == op.get("nn.bias_add"):
            _processInputNode(call.args[0], "odata", DstGraphNodeID, "idata", self.getInputGraphNodeSplitIndex(call.args[0]))
            _processInputNode(call.args[1], "bias", DstGraphNodeID, "bias", None)
            # _processInputNode(call.args[0], lambda x: len(x) == 1, "data")
            # InputNodeProperties = self.getInputTensorIDPair(call.args[1])
            # assert len(InputNodeProperties) == 1, "Bias_add should have only one bias node"
            # self.appendToPathList(InputNodeProperties, DstNodeProperty, "bias")
          if call.op == op.get("nn.batch_norm"):
            _processInputNode(call.args[0], "odata", DstGraphNodeID, "idata", self.getInputGraphNodeSplitIndex(call.args[0]))
            _processInputNode(call.args[1], "scale", DstGraphNodeID, "scale", None)
            _processInputNode(call.args[2], "bias", DstGraphNodeID, "bias", None)
            # _processInputNode(call.args[0], lambda x: len(x) == 1, "data")
            # self.appendToPathList(self.getInodePlaceHolderInputConstant(), DstNodeProperty, "scale")
            # self.appendToPathList(self.getInodePlaceHolderInputConstant(), DstNodeProperty, "bias")
          if call.op == op.get("nn.relu"):
            _processInputNode(call.args[0], "odata", DstGraphNodeID, "idata", self.getInputGraphNodeSplitIndex(call.args[0]))
            # _processInputNode(call.args[0], lambda x: len(x) == 1, "data")
          if call.op == op.get("add"):
            _processInputNode(call.args[0], "odata", DstGraphNodeID, "idata0", self.getInputGraphNodeSplitIndex(call.args[0]))
            _processInputNode(call.args[1], "odata", DstGraphNodeID, "idata1", self.getInputGraphNodeSplitIndex(call.args[1]))
            # _processInputNode(call.args[0], lambda x: len(x) == 1, "data0")
            # _processInputNode(call.args[1], lambda x: len(x) == 1, "data1")

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
      # ImcflowDeviceConfig().NoCPaths[func_name_var.name_hint] = []
      # tensor edge to path entry
      # if src graph is Var, hw_node_id is left-most inode of dst graph node hw_node_id
      # we will add instruction path to each IMCE node
      TensorEdgeList_ = ImcflowDeviceConfig().TensorEdgeListDict[func_name_var.name_hint]
      for tensor_edge in TensorEdgeList_:
        SrcTensorID = tensor_edge.src_id
        DstTensorID = tensor_edge.dst_id
        SplitIdx = tensor_edge.split_idx
        SrcGraphNode = CustomIDToNode()[getFlagNodeID(SrcTensorID.graph_node_id)]
        if isinstance(SrcGraphNode, (Var, Constant)):
          DstHwNodeID = HwMapping[getFlagNodeID(DstTensorID.graph_node_id)]
          # if "inode" not in DstHwNodeID:
          if not DstHwNodeID.is_inode():
            # DstIMCEIdx = int(re.match(r"imce_(\d+)", DstHwNodeID).group(1))
            InodeID = NodeID.from_inode_coord(NodeID.to_coord(DstHwNodeID)[0])
            # InodeID = f"inode_{DstIMCEIdx//IMCECOL}"
            # NocPaths[func_name_var.name_hint].append(
            #   (InodeID, DstHwNodeID, SplitIdx)
            # )
            NocPaths[func_name_var.name_hint][tensor_edge] = (
              (InodeID, DstHwNodeID, SplitIdx)
            )
            HwMapping[getFlagNodeID(SrcTensorID.graph_node_id)] = InodeID
        else:
          NocPaths[func_name_var.name_hint][tensor_edge] = (
            (HwMapping[getFlagNodeID(SrcTensorID.graph_node_id)], HwMapping[getFlagNodeID(DstTensorID.graph_node_id)], SplitIdx)
          )
          # NocPaths[func_name_var.name_hint].append(
          #   (HwMapping[getFlagNodeID(SrcTensorID.graph_node_id)], HwMapping[getFlagNodeID(DstTensorID.graph_node_id)], SplitIdx)
          # )

      for ActiveIMCE in ImcflowDeviceConfig().ActiveIMCEPerFunc[func_name_var.name_hint]:
        DstHwNodeID = ActiveIMCE
        # DstIMCEIdx = int(re.match(r"imce_(\d+)", DstHwNodeID).group(1))
        InodeID = NodeID.from_inode_coord(NodeID.to_coord(DstHwNodeID)[0])
        # InodeID = f"inode_{DstIMCEIdx//IMCECOL}"
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
            self.TensorEdgetoInfo_temp = {}
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

            def handle_single_dest(edge, mapping_info, init_addr_save=True, router_entry_list=None):
                """Append new entries to policy tables for a single destination"""
                source_node = mapping_info[0]
                dest_node = mapping_info[1]
                dest_index = mapping_info[2]                
                if isinstance(edge, NodeID):
                  source_node_data_type ="instruction"
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
                self.TensorEdgetoInfo_temp[edge] = router_entry_list

            def handle_multicast(edge, mapping_info):
                """Handle multiple destinations with potential path sharing"""
                source_node = mapping_info[0]
                dest_node = mapping_info[1]
                # dest_index = mapping_info[2]
                if isinstance(edge, NodeID):
                  source_node_data_type ="instruction"
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
                        
                        #create RouterEntry and append to router_entry_list
                        router_entry_list.append((current_node, entry_addr))
                        
                        # diverge into new path
                        new_mapping = (next_node, mapping_info[1], mapping_info[2])
                        handle_single_dest(edge, new_mapping, init_addr_save=False, router_entry_list=router_entry_list)                        
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
                            self.TensorEdgetoInfo_temp[edge] = router_entry_list
                            break

            # Main logic
            for edge, mapping_info in self.NoCPaths.items():
                handle_single_dest(edge, mapping_info)

            self.Policytable = policy_tables

        def add_EdgeInfo(self):
            # after policy table entry generation finished, add to TensorEdgeToInfo
            fifo_id_cnt = {node_id: 0 for node_id in NodeID}
            for edge, mapping_info in self.NoCPaths.items():
              # if tensoredge, save to TensorEdgetoInfo
              dest_node = mapping_info[1]
              router_entry_list=[]
              if edge in self.TensorEdgetoInfo_temp:
                  for entry_tuple in self.TensorEdgetoInfo_temp[edge]:
                      router_entry_list.append(RouterEntry(entry_tuple[0], entry_tuple[1], self.Policytable[entry_tuple[0]][entry_tuple[1]]))

                  if isinstance(edge, TensorEdge): # TensorEdge
                      edgeinfo = TensorEdgeInfo(router_entry_list, None, fifo_id_cnt[dest_node])
                      ImcflowDeviceConfig().add_tensor_edge_info(edge, edgeinfo)
                      fifo_id_cnt[dest_node] = fifo_id_cnt[dest_node] + 1
                  else: # Instruction edge
                      edgeinfo = InstEdgeInfo(router_entry_list, None)
                      ImcflowDeviceConfig().add_inst_edge_info(edge, edgeinfo)
                      
        def traverse_func(self, func):
            # traverse input function by visit() to make PathDict and generate policy table for it
            self.generate_policy_table()
            self.add_EdgeInfo()
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