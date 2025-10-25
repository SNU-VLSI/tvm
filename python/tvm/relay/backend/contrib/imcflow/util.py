"""
Helper utility Enums and Functions used through out code generation.

The rest of the utility functions are misc.
Refer to the description inside such functions
"""

import tvm
from tvm import relay


def is_imcflow_func(func: relay.Function) -> bool:
  """
  Check if the given function is an IMCFLOW function required for CodeGenSuite.
  It is updated to bypass the wrapping (layout transform and bitpack) functions in the IMCFLOW function tagged with Compiler:imcflow.
  """
  return isinstance(func, relay.Function) and "Compiler" in func.attrs and func.attrs["Compiler"] == "imcflow"


def extract_composite_function(func: relay.Function) -> relay.Function:
  """
  Extract the inner composite function from an imcflow function.

  The imcflow function body contains a nested function with a Composite attribute.
  This function traverses the body to find and return that composite function.

  Parameters
  ----------
  func: relay.Function
      The imcflow function with Compiler="imcflow" attribute

  Returns
  -------
  relay.Function
      The inner composite function
  """
  class CompositeFinder(relay.ExprVisitor):
    def __init__(self):
      super().__init__()
      self.composite_func = None

    def visit_function(self, fn):
      # Check if this function has a Composite attribute
      if isinstance(fn, relay.Function) and "Composite" in fn.attrs:
        # Only set if we haven't found one yet (take the first/top-level one)
        if self.composite_func is None:
          self.composite_func = fn
      # Continue visiting to ensure we explore the entire tree
      super().visit_function(fn)

  finder = CompositeFinder()
  finder.visit(func.body)

  if finder.composite_func is None:
    raise ValueError(f"No composite function found in imcflow function")

  return finder.composite_func


def create_imcflow_function_pass(opt_level: int, name: str = ""):
  """
  A utility decorator that wraps a given class as an imcflow function pass. That is,
  a pass that behaves like a function pass and only traverses imcflow external
  functions. It extracts the inner composite function from each imcflow function
  and passes that to transform_function. How each composite function is processed
  is defined by the `transform_function(global_variable, relay_function)` function
  which should be created in the class that is to be decorated. See the example below.

  Example
  -------
  This small example demonstrates a pass over imcflow composite functions that performs no
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
          # Extract the inner composite function from the imcflow function
          composite_func = extract_composite_function(func)
          # Pass the composite function to transform_function
          imcflow_pass = imcflow_pass_class(*self.args, **self.kwargs)
          # Note: We extract but don't update the module since transform_function
          # is used for analysis/code generation, not transformation
          imcflow_pass.transform_function(global_var, composite_func)
        return mod

    return ModulePassWrapper

  return decorator
