import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.function import FunctionWithFields
from tvm.relay.expr import (Call, Constant)

import numpy as np
import re

def clearPrimitiveTag(mod):
  class _Visitor(tvm.relay.ExprMutator):
    def visit_function(self, fn):
      fn = super().visit_function(fn)

      NewAttrs = {}
      if fn.attrs:
        for key in fn.attrs.keys():
          NewAttrs[key] = fn.attrs[key]
      
      # for attr in ["Primitive", "Composite", "global_symbol", "Compiler"]:
      for attr in ["Primitive", "Composite", "Compiler"]:
        if attr in NewAttrs:
          del NewAttrs[attr]

      return FunctionWithFields(fn, list(fn.params), fn.body, fn.ret_type, fn.type_params, tvm.ir.make_node("DictAttrs", **NewAttrs))

    def visit_call(self, call):
      # Inline local functions with Composite attribute
      if isinstance(call.op, relay.Function) and call.op.attrs and "Composite" in call.op.attrs:
        # We want to inline this function.
        # 1. Visit arguments
        new_args = [self.visit(arg) for arg in call.args]
        # 2. Visit body of the function
        new_body = self.visit(call.op.body)
        # 3. Bind parameters to arguments
        bind_map = dict(zip(call.op.params, new_args))
        return relay.bind(new_body, bind_map)
      
      return super().visit_call(call)

  for func_name in mod.functions:
    mod[func_name] = _Visitor().visit(mod[func_name])

  return mod

def clearCompilerAttr(mod):
  class _Visitor(tvm.relay.ExprMutator):
    def visit_function(self, fn):
      fn = super().visit_function(fn)

      NewAttrs = {}
      if fn.attrs:
        for key in fn.attrs.keys():
          NewAttrs[key] = fn.attrs[key]
      if "Compiler" in NewAttrs:
        del NewAttrs["Compiler"]

      return FunctionWithFields(fn, list(fn.params), fn.body, fn.ret_type, fn.type_params, tvm.ir.make_node("DictAttrs", **NewAttrs))

  for func_name in mod.functions:
    mod[func_name] = _Visitor().visit(mod[func_name])

  return mod


def make_cpu_runnable(mod):
  """
  Transform the given Relay module to be runnable on CPU.
  1. clear compiler and primitive attributes from all functions
  2. revert qconv2d weight shape back to standard layout NCHW.
     we assume the weight was transformed to imcflow layout if
     data type is uint32.
  """
  mod = clearCompilerAttr(mod)
  mod = clearPrimitiveTag(mod)
  mod = relay.transform.Inline()(mod)

  class WeightReverter(ExprMutator):
    def visit_call(self, call):
      new_call = super().visit_call(call)
      
      if isinstance(new_call.op, tvm.ir.Op):
        if new_call.op.name == "nn.imcflow_qconv":
          weight = new_call.args[1]
          if isinstance(weight, Constant) and weight.data.dtype == "uint32":
            # Revert qconv weight
            packed_data = weight.data.numpy()
            
            OC = new_call.attrs.channels.value
            IC = new_call.attrs.in_channels.value
            KH, KW = new_call.attrs.kernel_size
            KH = KH.value
            KW = KW.value
            
            ic = 256 // (KH * KW)
            spatial_elements = ic * KH * KW
            
            out_blocks = packed_data.shape[0]
            in_blocks = packed_data.shape[1]
            
            # Unpack 8 int4s from uint32
            # ToPack shape: (out_blocks, in_blocks, 256, 8, 8)
            unpacked = np.zeros((out_blocks, in_blocks, 256, 8, 8), dtype=np.int8)
            for i in range(8):
                vals = (packed_data >> (i * 4)) & 0xF
                # Handle sign for int4
                vals = vals.astype(np.int8)
                mask = vals >= 8
                vals[mask] -= 16
                unpacked[..., i] = vals
                
            # Reshape to Padded: (out_blocks, in_blocks, 256, 64)
            padded = unpacked.reshape(out_blocks, in_blocks, 256, 64)
            
            # Slice spatial elements
            if spatial_elements < 256:
                flattened = padded[:, :, :spatial_elements, :]
            else:
                flattened = padded
                
            # Reshape to Transposed: (out_blocks, in_blocks, ic, kh, kw, 64)
            transposed = flattened.reshape(out_blocks, in_blocks, ic, KH, KW, 64)
            
            # Transpose back to Reshaped: (out_blocks, 64, in_blocks, ic, kh, kw)
            reshaped = transposed.transpose(0, 5, 1, 2, 3, 4)
            
            # Reshape to PaddedWeight: (padded_OC, padded_IC, KH, KW)
            padded_weight = reshaped.reshape(out_blocks * 64, in_blocks * ic, KH, KW)
            
            # Crop
            restored_weight_data = padded_weight[:OC, :IC, :, :]
            restored_weight = relay.Constant(tvm.nd.array(restored_weight_data))
            
            # Keep qconv2d op
            new_args = [new_call.args[0], restored_weight, new_call.args[2]]
            
            new_type_args = list(new_call.type_args)
            if len(new_type_args) > 1:
                new_type_args[1] = relay.TensorType(restored_weight.data.shape, restored_weight.data.dtype)

            new_attrs = {}
            for key in new_call.attrs.keys():
                new_attrs[str(key)] = new_call.attrs[key]
            new_attrs["const_packed_node"] = False
            new_attr_node = tvm.ir.make_node("relay.attrs.ImcflowQConv2DAttrs", **new_attrs)
            return Call(new_call.op, new_args, new_attr_node, new_type_args, new_call.span)
            
        elif new_call.op.name == "nn.imcflow_qdwconv":
          weight = new_call.args[1]
          if isinstance(weight, Constant) and weight.data.dtype == "uint32":
            # Revert qdwconv weight
            packed_data = weight.data.numpy()
            
            OC = new_call.attrs.channels
            KH, KW = new_call.attrs.kernel_size
            # For depthwise, in_channels is 1 per group. groups=OC.
            
            restored_weight_data = np.zeros((OC, 1, KH, KW), dtype=np.int8)
            
            for c in range(OC):
                for kh_ in range(KH):
                    for kw_ in range(KW):
                        oc_block = c // 16
                        oc_offset = c % 16
                        khkw_index = kh_ * KW + kw_
                        word_idx = (oc_offset * 16 + khkw_index) // 32
                        word_offset = (oc_offset * 16 + khkw_index) % 32
                        
                        val = 0
                        for wb in range(8):
                            bit = (packed_data[oc_block, wb, word_idx] >> word_offset) & 1
                            val |= (bit << wb)
                        
                        # Handle sign (int8)
                        if val >= 128:
                            val -= 256
                        restored_weight_data[c, 0, kh_, kw_] = val
            
            restored_weight = relay.Constant(tvm.nd.array(restored_weight_data))
            
            new_args = [new_call.args[0], restored_weight, new_call.args[2]]
            
            new_type_args = list(new_call.type_args)
            if len(new_type_args) > 1:
                new_type_args[1] = relay.TensorType(restored_weight.data.shape, restored_weight.data.dtype)

            new_attrs = {}
            for key in new_call.attrs.keys():
                new_attrs[key] = new_call.attrs[key]
            new_attrs["const_packed_node"] = False
            new_attr_node = tvm.ir.make_node("relay.attrs.ImcflowQDwConv2DAttrs", **new_attrs)
            return Call(new_call.op, new_args, new_attr_node, new_type_args, new_call.span)

      return new_call

  for gv in mod.functions:
    if isinstance(mod[gv], relay.Function):
      mod[gv] = WeightReverter().visit(mod[gv])
      
  return mod