
import tvm
from tvm import relay
from tvm.relay.backend.contrib.imcflow.transform_utils import UseDefChainParser

def test_local_function_param_mapping():
    # Create a graph:
    # def local_func(x, y):
    #     return x + y
    # a = var('a')
    # b = var('b')
    # res = local_func(a, b)
    
    x = relay.var("x")
    y = relay.var("y")
    # Body of local function
    add_op = x + y
    local_func = relay.Function([x, y], add_op)
    
    a = relay.var("a")
    b = relay.var("b")
    call = local_func(a, b)
    
    # Wrap in a main function to be a valid module (optional but good practice)
    main_func = relay.Function([a, b], call)
    mod = tvm.IRModule.from_expr(main_func)
    
    parser = UseDefChainParser()
    parser.visit(main_func)
    
    # 1. Check param_to_args
    # Note: parser visits main_func. main_func body is 'call'.
    # 'call' invokes 'local_func'.
    # So parser should have recorded mapping for local_func params.
    
    print("Checking param_to_args...")
    assert hasattr(parser, "param_to_args"), "parser should have param_to_args attribute"
    
    # x maps to a, y maps to b
    assert x in parser.param_to_args, "x should be in param_to_args"
    assert parser.param_to_args[x] == a
    assert y in parser.param_to_args, "y should be in param_to_args"
    assert parser.param_to_args[y] == b
    print("param_to_args check passed.")
    
    # 2. Check get_uses for node inside local function
    print("Checking get_uses for inner node...")
    
    # add_op is inside local_func. It uses x and y.
    # parser.get_uses(add_op) should return [x, y]
    # Since add_op is not in the top-level function's direct uses (it's in sub-parser),
    # the parser needs to delegate to sub-parser.
    
    uses = parser.get_uses(add_op)
    assert len(uses) == 2
    assert x in uses
    assert y in uses
    print("get_uses for inner node passed.")

if __name__ == "__main__":
    test_local_function_param_mapping()
