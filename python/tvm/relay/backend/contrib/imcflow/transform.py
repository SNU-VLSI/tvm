import tvm
import tvm.relay as relay

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

          def visit_call(self, call):
            if call.op.name == "nn.conv2d":
              print(call.op)
            
            return super().visit_call(call)

      return Spliter().visit(func)