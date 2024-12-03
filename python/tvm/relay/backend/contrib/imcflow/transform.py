import tvm
from tvm import relay
from tvm.relay import transform, op
from tvm.relay.ty import TupleType, TensorType
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.function import Function, FunctionWithFields
from tvm.relay.expr import (Call, GlobalVar, TupleGetItem, const, Let, Var, If, Tuple, Constant)
from tvm.relay.expr import RefCreate, RefRead, RefWrite
from tvm.relay.adt import Constructor, Match, Clause
from tvm.ir import Op

import math
from copy import deepcopy
import re

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

          if not ((KH == 1 and KW == 1) or (KH == 3 and KW == 3) or (KH == 5 and KW == 5) or (KH == 7 and KW == 7)):
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
      if int(hash(expr)) != int(hash(call)):
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
        if int(hash(expr)) == int(hash(arg)):
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
      IMCE_NUM = 16

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
            if self.getCost(call) > IMCE_NUM:
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
              if self.getRegionSize(CandidateRegion) + self.getCost(call) > IMCE_NUM:
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
    def __init__(self, IMCE_NUM=4, INODE_NUM=4):
      self.MappingDict_2D = {}
      self.IMCE_NUM = IMCE_NUM
      self.INODE_NUM = INODE_NUM

    def transform_function(self, func, mod, ctx):
      IMCE_NUM = self.IMCE_NUM
      INODE_NUM = self.INODE_NUM
      
      class _Nodemapper(tvm.relay.ExprVisitor):
        """
          Target Operators:
            conv2d, bias_add, batch_norm, relu, add and fused versions
            split, concat
        """
        def __init__(self):
            super().__init__()
            self.MappingDict ={}
            self.imce_index = IMCE_NUM
            self.inode_index = INODE_NUM
        
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
              _, last_child_mapping = list(self.MappingDict.items())[-1]
          else:
              last_child_mapping = None
                 
          # check this node is for imcflow
          IsConcat = isinstance(call.op, tvm.ir.Op) and call.op.name in ["concatenate"]
          IsSplit = isinstance(call.op, tvm.ir.Op) and call.op.name in ["split"]
          if IsConcat:
              if last_child_mapping is None:
                  raise ValueError("split or concatenate should have at least 1 child node")
              self.MappingDict[int(hash(call))] = last_child_mapping
          elif IsSplit:
              if last_child_mapping is None:
                  self.MappingDict[int(hash(call))] = f"inode_{self.inode_index}"
                  self.inode_index -= 1
              else:
                  self.MappingDict[int(hash(call))] = last_child_mapping
          else:
              self.MappingDict[int(hash(call))] = f"imce_{self.imce_index}"
              self.imce_index -= 1

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
          self.MappingDict_2D[function_names[i]] = _Nodemapper().traverse_func(mod[function_names[i]])

      # # find all regions
      return func