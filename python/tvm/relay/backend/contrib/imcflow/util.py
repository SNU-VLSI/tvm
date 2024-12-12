"""
Helper utility Enums and Functions used through out code generation.

The rest of the utility functions are misc.
Refer to the description inside such functions
"""

import tvm
from tvm import relay


def is_imcflow_func(func: relay.Function) -> bool:
  """Check if the given function is an IMCFLOW function."""
  return "Compiler" in func.attrs and func.attrs["Compiler"] == "imcflow"


def create_imcflow_function_pass(opt_level: int, name: str = ""):
  """
  A utility decorator that wraps a given class as an imcflow function pass. That is,
  a pass that behaves like a function pass and only traverses imcflow external
  functions. How each imcflow function is mutated is defined by the
  `transform_function(global_variable, relay_function)` function which should
  be created in the class that is to be decorated. See the example below.

  Example
  -------
  This small example demonstrates a pass over imcflow functions that performs no
  mutation.

  @create_imcflow_function_pass(opt_level=1)
  class MyPass:
      def transform_function(self, global_var, func):
          return func

  mod = tvm.IRModule()
  mod = MyPass()(mod)

  Parameters
  ----------
  opt_level: int
      Optimization level for the module pass.
  name: str, optional
      Name for the module pass.

  Returns
  -------
  decorator
      The imcflow_pass decorator.
  """

  def decorator(imcflow_pass_class):
    @tvm.ir.transform.module_pass(name=name, opt_level=opt_level)
    class ModulePassWrapper:
      """The wrapper for the imcflow pass."""

      def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

      def transform_module(self, mod: tvm.ir.IRModule, _) -> tvm.ir.IRModule:
        imcflow_functions = filter(
            lambda x: is_imcflow_func(x[1]), mod.functions.items())
        for global_var, func in imcflow_functions:
          import pdb; pdb.set_trace()
          imcflow_pass = imcflow_pass_class(*self.args, **self.kwargs)
          func = imcflow_pass.transform_function(global_var, func)
          mod.update_func(global_var, func)
        return mod

    return ModulePassWrapper

  return decorator
