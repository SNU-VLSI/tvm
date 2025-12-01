import tvm
from tvm import relay
from tvm.relay import transform, op

from tvm.relay.backend.contrib.imcflow.transform_utils import get_shape
from tvm.contrib.imcflow import (
  ImcflowDeviceConfig
)

from tvm.relay.op.nn import imcflow_batch_norm, imcflow_qconv2d, imcflow_qdwconv2d

import math

def split_conv_to_atomic_impl(mod, OldParamDict):
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
            # Extract input and kernel shapes
            _, IC, IH, IW = get_shape(mod, expr.args[0])  # Input shape
            OC, _, KH, KW = get_shape(mod, expr.args[1])  # Kernel shape
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
                    N, IC_slice, IH_slice, IW_slice = get_shape(mod, input_node)

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