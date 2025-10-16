import tvm
import numpy as np
import pathlib
from tvm.micro import export_model_library_format
from tvm.micro.testing import get_target
from tvm.contrib.utils import tempdir
import tvm.testing
from tvm.relay import pretty_print
from tvm.contrib.relay_viz import RelayVisualizer, DotPlotter, DotVizParser
from tvm.relay.build_module import bind_params_by_name
from tvm.relay import transform
from tvm.relay.backend.contrib.imcflow import transform as imcflow_transform
from tvm.relay.backend.contrib.imcflow import codegen as imcflow_codegen
from tvm.relay.op.contrib import imcflow
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from tvm.relay.backend import Executor, Runtime
from tvm.contrib.imcflow import DataBlock
import os
from tvm.relay.op.transform import imcflow_4d_to_qconv_input, imcflow_mmquant_out_to_4d
from tvm.relay.qnn.op.qnn import imcflow_min_max_quantize, imcflow_nu_quantize
from tvm.relay.op.nn import imcflow_batch_norm, imcflow_qconv2d
import tvm.relay as relay
from tvm.relay.expr import (Call, GlobalVar, TupleGetItem, const, Let, Var, If, Tuple, Constant)

from models import real_model, real_model2, test_models
from models import small_model
from models import resnet8_cifar

def printModel(result_dir, mod, param_dict, mod_name):
  RelayVisualizer(
      relay_mod=mod,
      relay_param=param_dict,
      plotter=DotPlotter(),
      parser=DotVizParser(),
  ).render(f"{result_dir}/{mod_name}")

  with open(f"{result_dir}/{mod_name}.txt", "w") as f:
    f.write(pretty_print(mod))

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

def test_min_max_quant():
  # input_shape = (1, 1, 32, 32, 16)
  input_shape = (1, 16, 32, 32)

  x = relay.var("data", shape=input_shape, dtype="int16")
  x2 = relay.var("quant_min_1", shape=(), dtype="int16")
  x3 = relay.var("quant_max_1", shape=(), dtype="int16")
  y = imcflow_min_max_quantize(x, x2, x3, axis=1, out_dtype="int4", channel=16)

  y = modify_call_node_attrs(y, in_node=True, out_node=True)

  func = relay.Function([x, x2, x3], y)

  mod = tvm.IRModule.from_expr(func)
  print(mod)
  mod = transform.InferType()(mod)
  print(mod)

if __name__ == "__main__":
  tvm.testing.main()
