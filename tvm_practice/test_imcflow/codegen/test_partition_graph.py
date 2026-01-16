"""
Test PartitionGraph behavior with branching structures.

This test explores how PartitionGraph handles nodes in the same region
when there are branches (diverge-converge patterns).

Problem:
  A -> {B, C} -> D structure where A, B, C are in the same region
  After PartitionGraph, we get 3 separate functions instead of 2.
  Expected: {A, B, C} in one function, {D} in another.
"""

import tvm
from tvm import relay
from tvm.relay import transform
import numpy as np

# Enable debug output
import os
os.environ['IMCFLOW_DEBUG'] = '1'


def create_branch_graph():
    """
    Create a simple graph with branch structure:

        input (A)
          |
        conv1 (diverge point)
        /    \
      relu   multiply (B and C - two branches)
        \    /
         add (D - converge point)
          |
        output

    All nodes except 'add' should be in region1.
    'add' should be in region2.
    """
    data = relay.var("data", shape=(1, 32, 8, 8), dtype="float32")
    weight = relay.const(np.random.randn(32, 32, 3, 3).astype("float32"))
    scale = relay.const(np.random.randn(32, 1, 1).astype("float32"))

    # A: conv (diverge point)
    conv = relay.nn.conv2d(data, weight, in_channels=32, channels=32,
                           kernel_size=(3, 3), padding=(1, 1))

    # B: relu branch
    branch_b = relay.nn.relu(conv)

    # C: multiply branch (skip-like)
    branch_c = relay.multiply(conv, scale)

    # D: add (converge point)
    add_out = relay.add(branch_b, branch_c)

    func = relay.Function([data], add_out)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)

    return mod


def annotate_with_regions(mod, region1_ops, region2_ops, compiler_prefix="test"):
    """
    Manually annotate nodes with compiler_begin/end.

    Args:
        mod: The module to annotate
        region1_ops: List of op names to put in region1
        region2_ops: List of op names to put in region2
        compiler_prefix: Prefix for compiler name
    """
    region1_compiler = f"{compiler_prefix}_region1"
    region2_compiler = f"{compiler_prefix}_region2"

    class Annotator(relay.ExprMutator):
        def __init__(self):
            super().__init__()
            self.visited = {}

        def visit_call(self, call):
            if call in self.visited:
                return self.visited[call]

            # Visit args first
            new_args = [self.visit(arg) for arg in call.args]

            if isinstance(call.op, tvm.ir.Op):
                op_name = call.op.name

                if op_name in region1_ops:
                    # Wrap inputs with compiler_begin
                    wrapped_args = []
                    for arg in new_args:
                        wrapped = relay.annotation.compiler_begin(arg, region1_compiler)
                        wrapped_args.append(wrapped)

                    # Create the call
                    new_call = relay.Call(call.op, wrapped_args, call.attrs)

                    # Wrap output with compiler_end
                    result = relay.annotation.compiler_end(new_call, region1_compiler)
                    self.visited[call] = result
                    return result

                elif op_name in region2_ops:
                    # Wrap inputs with compiler_begin
                    wrapped_args = []
                    for arg in new_args:
                        wrapped = relay.annotation.compiler_begin(arg, region2_compiler)
                        wrapped_args.append(wrapped)

                    # Create the call
                    new_call = relay.Call(call.op, wrapped_args, call.attrs)

                    # Wrap output with compiler_end
                    result = relay.annotation.compiler_end(new_call, region2_compiler)
                    self.visited[call] = result
                    return result

            # Default: just update args
            new_call = relay.Call(call.op, new_args, call.attrs, call.type_args, call.span)
            self.visited[call] = new_call
            return new_call

    annotator = Annotator()
    new_body = annotator.visit(mod["main"].body)
    new_func = relay.Function(mod["main"].params, new_body)
    new_mod = tvm.IRModule.from_expr(new_func)
    new_mod = relay.transform.InferType()(new_mod)

    return new_mod


def test_partition_graph_basic():
    """Test basic PartitionGraph behavior without branches."""
    print("\n" + "="*60)
    print("TEST: PartitionGraph Basic (No Branches)")
    print("="*60)

    # Simple linear graph: data -> conv -> relu -> output
    data = relay.var("data", shape=(1, 32, 8, 8), dtype="float32")
    weight = relay.const(np.random.randn(32, 32, 3, 3).astype("float32"))

    conv = relay.nn.conv2d(data, weight, in_channels=32, channels=32,
                           kernel_size=(3, 3), padding=(1, 1))
    relu = relay.nn.relu(conv)

    func = relay.Function([data], relu)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)

    print("\n--- Original ---")
    print(relay.pretty_print(mod))

    # Annotate: conv, relu in region1
    mod = annotate_with_regions(mod, ["nn.conv2d", "nn.relu"], [], "test")
    print("\n--- After Annotation ---")
    print(relay.pretty_print(mod))

    # Merge regions
    mod = transform.MergeCompilerRegions()(mod)
    print("\n--- After MergeCompilerRegions ---")
    print(relay.pretty_print(mod))

    # Partition graph
    mod = transform.PartitionGraph()(mod)
    print("\n--- After PartitionGraph ---")
    print(relay.pretty_print(mod))

    # Count functions
    func_count = len(mod.functions)
    print(f"\nFunction count: {func_count}")

    print("\n✓ Basic test completed")


def test_partition_graph_with_branch():
    """Test PartitionGraph with branch structure (the problem case)."""
    print("\n" + "="*60)
    print("TEST: PartitionGraph With Branch (Problem Case)")
    print("="*60)

    mod = create_branch_graph()

    print("\n--- Original Graph ---")
    print(relay.pretty_print(mod))

    # Annotate: conv, relu, multiply in region1; add in region2
    mod = annotate_with_regions(
        mod,
        region1_ops=["nn.conv2d", "nn.relu", "multiply"],
        region2_ops=["add"],
        compiler_prefix="test"
    )
    print("\n--- After Annotation ---")
    print(relay.pretty_print(mod))

    # Merge regions
    mod = transform.MergeCompilerRegions()(mod)
    print("\n--- After MergeCompilerRegions ---")
    print(relay.pretty_print(mod))

    # Partition graph
    mod = transform.PartitionGraph()(mod)
    print("\n--- After PartitionGraph ---")
    print(relay.pretty_print(mod))

    # Count functions
    func_count = len(mod.functions)
    print(f"\nFunction count: {func_count}")
    print(f"Expected: 3 (main + region1 + region2)")
    print(f"Actual:   {func_count}")

    # List all functions
    print("\nFunctions in module:")
    for name, func in mod.functions.items():
        print(f"  - {name}")

    print("\n✓ Branch test completed")


def test_partition_graph_branch_same_region():
    """
    Test if branches in the same region stay together after PartitionGraph.

    Structure:
        input
          |
        conv (A)
        /    \
      relu   multiply (B, C - both in region1)
        \    /
         add (D - region2)

    We want A, B, C in ONE function, D in another.
    But PartitionGraph may split them.
    """
    print("\n" + "="*60)
    print("TEST: Branch Same Region - Does It Stay Together?")
    print("="*60)

    data = relay.var("data", shape=(1, 32, 8, 8), dtype="float32")
    weight = relay.const(np.random.randn(32, 32, 3, 3).astype("float32"))
    scale = relay.const(np.random.randn(32, 1, 1).astype("float32"))

    # A: conv
    conv = relay.nn.conv2d(data, weight, in_channels=32, channels=32,
                           kernel_size=(3, 3), padding=(1, 1))

    # B: relu
    relu = relay.nn.relu(conv)

    # C: multiply
    mul = relay.multiply(conv, scale)

    # D: add
    add_out = relay.add(relu, mul)

    func = relay.Function([data], add_out)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)

    print("\n--- Original ---")
    print(relay.pretty_print(mod))

    # Manually annotate with same region for A, B, C
    region1 = "test_region1"
    region2 = "test_region2"

    # Build annotated expression manually
    # A, B, C: region1
    # D: region2

    data_in = relay.var("data", shape=(1, 32, 8, 8), dtype="float32")

    # A: conv in region1
    conv_in = relay.annotation.compiler_begin(data_in, region1)
    weight_in = relay.annotation.compiler_begin(weight, region1)
    conv_out = relay.nn.conv2d(conv_in, weight_in, in_channels=32, channels=32,
                               kernel_size=(3, 3), padding=(1, 1))
    conv_end = relay.annotation.compiler_end(conv_out, region1)

    # B: relu in region1 (input comes from conv which ended)
    relu_in = relay.annotation.compiler_begin(conv_end, region1)
    relu_out = relay.nn.relu(relu_in)
    relu_end = relay.annotation.compiler_end(relu_out, region1)

    # C: multiply in region1 (input comes from conv)
    mul_in1 = relay.annotation.compiler_begin(conv_end, region1)
    mul_in2 = relay.annotation.compiler_begin(scale, region1)
    mul_out = relay.multiply(mul_in1, mul_in2)
    mul_end = relay.annotation.compiler_end(mul_out, region1)

    # D: add in region2
    add_in1 = relay.annotation.compiler_begin(relu_end, region2)
    add_in2 = relay.annotation.compiler_begin(mul_end, region2)
    add_out = relay.add(add_in1, add_in2)
    add_end = relay.annotation.compiler_end(add_out, region2)

    func = relay.Function([data_in], add_end)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)

    print("\n--- After Manual Annotation ---")
    print(relay.pretty_print(mod))

    # Merge regions
    mod = transform.MergeCompilerRegions()(mod)
    print("\n--- After MergeCompilerRegions ---")
    print(relay.pretty_print(mod))

    # Check how many compiler_begin/end pairs exist
    class AnnotationCounter(relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.begin_count = {}
            self.end_count = {}

        def visit_call(self, call):
            if isinstance(call.op, tvm.ir.Op):
                if call.op.name == "annotation.compiler_begin":
                    compiler = call.attrs.compiler
                    self.begin_count[compiler] = self.begin_count.get(compiler, 0) + 1
                elif call.op.name == "annotation.compiler_end":
                    compiler = call.attrs.compiler
                    self.end_count[compiler] = self.end_count.get(compiler, 0) + 1
            super().visit_call(call)

    counter = AnnotationCounter()
    counter.visit(mod["main"])
    print(f"\nAfter merge - compiler_begin counts: {counter.begin_count}")
    print(f"After merge - compiler_end counts: {counter.end_count}")

    # Partition graph
    mod = transform.PartitionGraph()(mod)
    print("\n--- After PartitionGraph ---")
    print(relay.pretty_print(mod))

    # Count functions
    func_count = len(mod.functions)
    print(f"\n*** Function count: {func_count} ***")

    # List all functions
    print("\nFunctions in module:")
    for name, func in mod.functions.items():
        attrs = dict(func.attrs) if func.attrs else {}
        compiler = attrs.get("Compiler", "N/A")
        print(f"  - {name}: Compiler={compiler}")

    # Analysis
    region1_funcs = [name for name, f in mod.functions.items()
                     if f.attrs and "Compiler" in f.attrs and "region1" in str(f.attrs["Compiler"])]
    region2_funcs = [name for name, f in mod.functions.items()
                     if f.attrs and "Compiler" in f.attrs and "region2" in str(f.attrs["Compiler"])]

    print(f"\nRegion1 functions: {len(region1_funcs)}")
    print(f"Region2 functions: {len(region2_funcs)}")

    if len(region1_funcs) > 1:
        print("\n⚠️  WARNING: Region1 was split into multiple functions!")
        print("   This is the problem - branches cause region splitting.")
    else:
        print("\n✓ Region1 stayed as one function.")

    print("\n✓ Test completed")


def test_alternative_annotation():
    """
    Test alternative annotation strategy:
    Don't put compiler_end between connected ops in the same region.
    """
    print("\n" + "="*60)
    print("TEST: Alternative Annotation (No intermediate end)")
    print("="*60)

    data = relay.var("data", shape=(1, 32, 8, 8), dtype="float32")
    weight = relay.const(np.random.randn(32, 32, 3, 3).astype("float32"))
    scale = relay.const(np.random.randn(32, 1, 1).astype("float32"))

    region1 = "test_region1"
    region2 = "test_region2"

    # Strategy: Only put compiler_begin at inputs, compiler_end at outputs to other regions

    # Region1 inputs
    data_in = relay.annotation.compiler_begin(data, region1)
    weight_in = relay.annotation.compiler_begin(weight, region1)
    scale_in = relay.annotation.compiler_begin(scale, region1)

    # Region1 ops (no intermediate annotations)
    conv = relay.nn.conv2d(data_in, weight_in, in_channels=32, channels=32,
                           kernel_size=(3, 3), padding=(1, 1))
    relu = relay.nn.relu(conv)
    mul = relay.multiply(conv, scale_in)

    # Region1 outputs (going to region2)
    relu_end = relay.annotation.compiler_end(relu, region1)
    mul_end = relay.annotation.compiler_end(mul, region1)

    # Region2
    add_in1 = relay.annotation.compiler_begin(relu_end, region2)
    add_in2 = relay.annotation.compiler_begin(mul_end, region2)
    add_out = relay.add(add_in1, add_in2)
    add_end = relay.annotation.compiler_end(add_out, region2)

    func = relay.Function([data], add_end)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)

    print("\n--- Annotated (No intermediate end) ---")
    print(relay.pretty_print(mod))

    # Skip MergeCompilerRegions - it's already optimally annotated
    # Partition graph
    mod = transform.PartitionGraph()(mod)
    print("\n--- After PartitionGraph ---")
    print(relay.pretty_print(mod))

    # Count functions
    func_count = len(mod.functions)
    print(f"\n*** Function count: {func_count} ***")

    # List all functions
    print("\nFunctions in module:")
    for name, func in mod.functions.items():
        attrs = dict(func.attrs) if func.attrs else {}
        compiler = attrs.get("Compiler", "N/A")
        print(f"  - {name}: Compiler={compiler}")

    print("\n✓ Test completed")


def test_imcflow_annotation_pass_with_branch():
    """
    Test ImcflowAnnotationPass with branch structure.

    This tests the fix where compiler_begin nodes are cached
    to ensure branches from the same input stay connected.
    """
    print("\n" + "="*60)
    print("TEST: ImcflowAnnotationPass With Branch (Fixed)")
    print("="*60)

    from tvm.relay.op.contrib import imcflow

    data = relay.var("data", shape=(1, 32, 8, 8), dtype="float32")
    weight = relay.const(np.random.randn(32, 32, 3, 3).astype("float32"))
    scale = relay.const(np.random.randn(32, 1, 1).astype("float32"))

    # A: conv (diverge point)
    conv = relay.nn.conv2d(data, weight, in_channels=32, channels=32,
                           kernel_size=(3, 3), padding=(1, 1))

    # B: relu branch
    relu = relay.nn.relu(conv)

    # C: multiply branch (skip-like, using input directly)
    mul = relay.multiply(data, scale)  # data is used again!

    # D: add (converge point)
    add_out = relay.add(relu, mul)

    func = relay.Function([data], add_out)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)

    print("\n--- Original ---")
    print(relay.pretty_print(mod))

    # Create region list: conv, relu, multiply in region1; add in region2
    # We need to get the actual Relay expression objects
    # For this test, let's use a simpler approach

    # Build RegionList by collecting expressions
    region1_ops = []
    region2_ops = []

    class ExprCollector(relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.exprs_by_op = {}

        def visit_call(self, call):
            if isinstance(call.op, tvm.ir.Op):
                op_name = call.op.name
                if op_name not in self.exprs_by_op:
                    self.exprs_by_op[op_name] = []
                self.exprs_by_op[op_name].append(call)
            super().visit_call(call)

    collector = ExprCollector()
    collector.visit(mod["main"])

    # Region1: conv, relu, multiply
    # Region2: add
    region1 = []
    region2 = []
    for op_name, exprs in collector.exprs_by_op.items():
        for expr in exprs:
            if op_name in ["nn.conv2d", "nn.relu", "multiply"]:
                region1.append(expr)
            elif op_name == "add":
                region2.append(expr)

    print(f"\nRegion1 ops: {[e.op.name for e in region1 if isinstance(e.op, tvm.ir.Op)]}")
    print(f"Region2 ops: {[e.op.name for e in region2 if isinstance(e.op, tvm.ir.Op)]}")

    # Apply ImcflowAnnotationPass
    RegionList = [region1, region2]
    mod = imcflow.ImcflowAnnotationPass(RegionList, "test_")(mod)
    print("\n--- After ImcflowAnnotationPass ---")
    print(relay.pretty_print(mod))

    # Count compiler_begin nodes for input
    class CompilerBeginCounter(relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.begin_count = {}

        def visit_call(self, call):
            if isinstance(call.op, tvm.ir.Op) and call.op.name == "annotation.compiler_begin":
                # Check if the argument is the input variable
                if isinstance(call.args[0], relay.Var):
                    compiler = call.attrs.compiler
                    key = (call.args[0].name_hint, compiler)
                    self.begin_count[key] = self.begin_count.get(key, 0) + 1
            super().visit_call(call)

    counter = CompilerBeginCounter()
    counter.visit(mod["main"])
    print(f"\ncompiler_begin counts for Vars: {counter.begin_count}")

    # The fix should ensure only ONE compiler_begin for the input going to region1
    for (var_name, compiler), count in counter.begin_count.items():
        if count > 1:
            print(f"\n⚠️  WARNING: {var_name} has {count} compiler_begin nodes for {compiler}")
            print("   This will cause PartitionGraph to create separate functions!")
        else:
            print(f"\n✓ {var_name} has exactly 1 compiler_begin for {compiler}")

    # Apply MergeCompilerRegions
    mod = transform.MergeCompilerRegions()(mod)
    print("\n--- After MergeCompilerRegions ---")
    print(relay.pretty_print(mod))

    # Apply PartitionGraph
    mod = transform.PartitionGraph()(mod)
    print("\n--- After PartitionGraph ---")
    print(relay.pretty_print(mod))

    # Count functions per region
    region1_funcs = [name for name, f in mod.functions.items()
                     if f.attrs and "Compiler" in f.attrs and "region1" in str(f.attrs["Compiler"])]
    region2_funcs = [name for name, f in mod.functions.items()
                     if f.attrs and "Compiler" in f.attrs and "region2" in str(f.attrs["Compiler"])]

    print(f"\n*** Results ***")
    print(f"Total functions: {len(mod.functions)}")
    print(f"Region1 functions: {len(region1_funcs)}")
    print(f"Region2 functions: {len(region2_funcs)}")

    if len(region1_funcs) == 1:
        print("\n✓ SUCCESS: Region1 stayed as ONE function!")
    else:
        print(f"\n✗ FAILURE: Region1 was split into {len(region1_funcs)} functions!")

    print("\n✓ Test completed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# PartitionGraph Branch Behavior Tests")
    print("#"*60)

    try:
        test_partition_graph_basic()
    except Exception as e:
        print(f"\n✗ Basic test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_partition_graph_with_branch()
    except Exception as e:
        print(f"\n✗ Branch test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_partition_graph_branch_same_region()
    except Exception as e:
        print(f"\n✗ Branch same region test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_alternative_annotation()
    except Exception as e:
        print(f"\n✗ Alternative annotation test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_imcflow_annotation_pass_with_branch()
    except Exception as e:
        print(f"\n✗ ImcflowAnnotationPass test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "#"*60)
    print("# Tests Complete")
    print("#"*60)


if __name__ == "__main__":
    run_all_tests()
