import tvm
from tvm import relay
from tvm.relay import transform, op
from tvm.relay.ty import TupleType, TensorType

import math

@relay.transform.function_pass(opt_level=0)
class CustomPipeline:
    """Simple test function to replace one argument to another."""

    def __init__(self, prefix):
        self.prefix = prefix

    # This function can define a pass.
    def transform_function(self, func, mod, ctx):
      print(func)
      return func

@relay.transform.function_pass(opt_level=0)
class ConvSplitToAtom:
    def __init__(self):
      pass

    def transform_function(self, func, mod, ctx):
      class Spliter(tvm.relay.ExprMutator):
        """Split large conv2d into smaller conv2d, split, concat, add, etc"""

        def split_and_optimize_conv2d(self, expr, mod):

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
          split_conv_weights = []
          split_weights = relay.op.transform.split(expr.args[1], indices_or_sections=oc_sections, axis=0) if IsOCSplited else [expr.args[1]]
          for oc_id in range(oc_split_num):
              weight_slice = relay.op.transform.split(split_weights[oc_id], indices_or_sections=ic_sections, axis=1) if (IsICSplited and (not IsDepthWise)) else [split_weights[oc_id]]
              split_conv_weights.append(weight_slice)

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

          # If output channels were split, concatenate along the output axis
          if IsOCSplited:
              concat_node = relay.op.concatenate([add_nodes[oc_id] for oc_id in range(oc_split_num)], axis=1)
          else:
              concat_node = add_nodes[0]

          return concat_node


        def visit_call(self, call):
          if call.op == op.get("nn.conv2d"):
            NewCall = super().visit_call(call)
            NewCall = self.split_and_optimize_conv2d(NewCall, mod)
            return NewCall
          else:
            return super().visit_call(call)

      return Spliter().visit(func)

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