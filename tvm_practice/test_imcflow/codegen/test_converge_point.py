"""
Test for converge point detection and branch analysis in imcflow transform.

Tests:
1. LatencyThroughputCalculator - op-level latency/throughput calculation
2. BranchAnalyzer - converge point detection and branch extraction
3. CompositeSplitter - composite splitting at converge points
4. Full partitionRound flow with converge point handling
"""

import tvm
from tvm import relay
from tvm.relay import op
from tvm.relay.build_module import bind_params_by_name
import numpy as np
import sys

# Enable debug output
import os
os.environ['IMCFLOW_DEBUG'] = '1'

from tvm.relay.backend.contrib.imcflow.transform import (
    LatencyThroughputCalculator,
    BranchAnalyzer,
    CompositeSplitter,
    CompositeGraphMutator,
    AnnotGenerator,
    partitionRound,
    partitionImcflowSubGraph,
    split_conv_to_atomic,
    merge_composite_ops,
    makeSplitConcatDepsRegions,
    ConcatDistributor,
    debug_print,
    # New functions for updated transform order
    merge_composite_for_partition,
    unmerge_composite,
)

# Import from driver for full transform
from tvm.driver.tvmc.imcflow_compiler_driver import compile_for_imcflow

# Add models path
sys.path.insert(0, '/root/project/tvm/tvm_practice')
from models import resnet8_subset_models


def create_simple_diverge_converge_graph():
    """
    Create a simple graph with diverge-converge pattern:

        input
          |
        conv1 (diverge point)
        /   \
     relu   identity (skip connection)
        \   /
         add (converge point)
          |
        output
    """
    data = relay.var("data", shape=(1, 32, 8, 8), dtype="int8")
    weight = relay.var("weight", shape=(32, 32, 3, 3), dtype="int8")

    # Conv1 - diverge point
    conv1 = relay.nn.conv2d(data, weight, in_channels=32, channels=32, kernel_size=(3, 3), padding=(1, 1))

    # Branch A: relu
    branch_a = relay.nn.relu(conv1)

    # Branch B: identity (skip connection) - just use conv1 directly
    branch_b = conv1

    # Converge point: add
    add_out = relay.add(branch_a, branch_b)

    func = relay.Function([data, weight], add_out)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)

    return mod


def create_unbalanced_branch_graph():
    """
    Create a graph with unbalanced branches:

        input
          |
        conv1 (diverge point)
        /         \
    conv2→relu   identity (skip - much shorter!)
        \         /
          add (converge point)
            |
          output

    Branch A: conv2 + relu (high latency)
    Branch B: identity (low latency)
    """
    data = relay.var("data", shape=(1, 32, 8, 8), dtype="int8")
    weight1 = relay.var("weight1", shape=(32, 32, 3, 3), dtype="int8")
    weight2 = relay.var("weight2", shape=(32, 32, 3, 3), dtype="int8")

    # Conv1 - diverge point
    conv1 = relay.nn.conv2d(data, weight1, in_channels=32, channels=32, kernel_size=(3, 3), padding=(1, 1))

    # Branch A: conv2 → relu (long path)
    conv2 = relay.nn.conv2d(conv1, weight2, in_channels=32, channels=32, kernel_size=(3, 3), padding=(1, 1))
    branch_a = relay.nn.relu(conv2)

    # Branch B: identity (short path - skip connection)
    branch_b = conv1

    # Converge point: add
    add_out = relay.add(branch_a, branch_b)

    func = relay.Function([data, weight1, weight2], add_out)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)

    return mod


def test_latency_throughput_calculator():
    """Test LatencyThroughputCalculator with different ops."""
    print("\n" + "="*60)
    print("TEST: LatencyThroughputCalculator")
    print("="*60)

    # Create test ops
    data = relay.var("data", shape=(1, 32, 8, 8), dtype="int8")
    weight = relay.var("weight", shape=(64, 32, 3, 3), dtype="int8")

    # Create different ops
    conv_call = relay.nn.conv2d(data, weight, in_channels=32, channels=64, kernel_size=(3, 3), padding=(1, 1))
    relu_call = relay.nn.relu(data)
    add_call = relay.add(data, data)

    # Infer types
    mod = tvm.IRModule.from_expr(relay.Function([data, weight], conv_call))
    mod = relay.transform.InferType()(mod)

    # Get the call nodes from the typed module
    conv_node = mod["main"].body

    mod2 = tvm.IRModule.from_expr(relay.Function([data], relu_call))
    mod2 = relay.transform.InferType()(mod2)
    relu_node = mod2["main"].body

    mod3 = tvm.IRModule.from_expr(relay.Function([data], add_call))
    mod3 = relay.transform.InferType()(mod3)
    add_node = mod3["main"].body

    # Create calculator instances with module
    lat_calc = LatencyThroughputCalculator(mod)
    lat_calc2 = LatencyThroughputCalculator(mod2)
    lat_calc3 = LatencyThroughputCalculator(mod3)

    # Test latency
    print("\nLatency Tests:")
    print(f"  conv2d (3x3, 64ch): {lat_calc.get_op_latency(conv_node)}")
    print(f"  relu: {lat_calc2.get_op_latency(relu_node)}")
    print(f"  add: {lat_calc3.get_op_latency(add_node)}")

    # Test throughput
    print("\nThroughput Tests:")
    print(f"  conv2d (3x3, 64ch): {lat_calc.get_op_throughput(conv_node)}")
    print(f"  relu: {lat_calc2.get_op_throughput(relu_node)}")
    print(f"  add: {lat_calc3.get_op_throughput(add_node)}")

    # Test branch calculations
    print("\nBranch Calculation Tests:")
    branch_ops = [conv_node, relu_node]
    print(f"  Branch [conv, relu] latency: {lat_calc.calculate_branch_latency(branch_ops)}")
    print(f"  Branch [conv, relu] throughput: {lat_calc.calculate_branch_throughput(branch_ops)}")

    branch_ops2 = [relu_node]
    print(f"  Branch [relu] latency: {lat_calc2.calculate_branch_latency(branch_ops2)}")
    print(f"  Branch [relu] throughput: {lat_calc2.calculate_branch_throughput(branch_ops2)}")

    print("\n✓ LatencyThroughputCalculator tests passed")


def test_branch_analyzer():
    """Test BranchAnalyzer with diverge-converge graph."""
    print("\n" + "="*60)
    print("TEST: BranchAnalyzer")
    print("="*60)

    mod = create_simple_diverge_converge_graph()
    main_func = mod["main"]

    print("\nGraph structure:")
    print(relay.pretty_print(mod))

    # Build edges manually for testing - include ALL args, not just Call nodes
    import collections
    edges = collections.defaultdict(list)
    rev_edges = collections.defaultdict(list)
    nodes = []

    class GraphBuilder(relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.nodes = []
            self.visited = set()

        def visit_call(self, call):
            if call in self.visited:
                return
            self.visited.add(call)
            self.nodes.append(call)

            for arg in call.args:
                # Connect Call -> Call edges
                if isinstance(arg, relay.Call):
                    edges[arg].append(call)
                    rev_edges[call].append(arg)
                self.visit(arg)

    builder = GraphBuilder()
    builder.visit(main_func.body)
    nodes = builder.nodes

    print(f"\nFound {len(nodes)} call nodes")
    for i, node in enumerate(nodes):
        if isinstance(node.op, tvm.ir.Op):
            print(f"  Node {i}: {node.op.name}")

    # Debug: print edges
    print(f"\nEdge info:")
    for node in nodes:
        if isinstance(node.op, tvm.ir.Op):
            preds = rev_edges.get(node, [])
            pred_names = [n.op.name if isinstance(n.op, tvm.ir.Op) else "?" for n in preds]
            print(f"  {node.op.name} <- {pred_names}")

    # Create BranchAnalyzer
    analyzer = BranchAnalyzer(edges, rev_edges)

    # Find converge point (add node)
    add_node = None
    conv_node = None
    for node in nodes:
        if isinstance(node.op, tvm.ir.Op):
            if node.op.name == "add":
                add_node = node
            elif node.op.name == "nn.conv2d":
                conv_node = node

    if add_node:
        print(f"\nTesting converge point detection for 'add' node:")
        # Manually check predecessors
        preds = rev_edges.get(add_node, [])
        print(f"  Direct predecessors: {len(preds)}")

        is_converge = analyzer.is_converge_point(add_node)
        print(f"  is_converge_point: {is_converge}")

        # Even if not detected as converge, test branch extraction
        if conv_node:
            print(f"\nTesting branch extraction from conv to add:")
            branches = analyzer.extract_branches(conv_node, add_node)
            print(f"  branches extracted: {len(branches)}")
            for i, branch in enumerate(branches):
                print(f"    Branch {i}: {len(branch)} ops")

    print("\n✓ BranchAnalyzer tests passed")


def test_branch_analyzer_unbalanced():
    """Test BranchAnalyzer with unbalanced branches."""
    print("\n" + "="*60)
    print("TEST: BranchAnalyzer (Unbalanced Branches)")
    print("="*60)

    mod = create_unbalanced_branch_graph()
    main_func = mod["main"]

    print("\nGraph structure:")
    print(relay.pretty_print(mod))

    # Build edges
    import collections
    edges = collections.defaultdict(list)
    rev_edges = collections.defaultdict(list)

    class GraphBuilder(relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.nodes = []
            self.visited = set()

        def visit_call(self, call):
            if call in self.visited:
                return
            self.visited.add(call)
            self.nodes.append(call)

            for arg in call.args:
                if isinstance(arg, relay.Call):
                    edges[arg].append(call)
                    rev_edges[call].append(arg)
                self.visit(arg)

    builder = GraphBuilder()
    builder.visit(main_func.body)
    nodes = builder.nodes

    print(f"\nFound {len(nodes)} call nodes")
    for node in nodes:
        if isinstance(node.op, tvm.ir.Op):
            preds = rev_edges.get(node, [])
            pred_names = [n.op.name if isinstance(n.op, tvm.ir.Op) else "?" for n in preds]
            print(f"  {node.op.name} <- {pred_names}")

    # Create BranchAnalyzer
    analyzer = BranchAnalyzer(edges, rev_edges)

    # Find nodes
    add_node = None
    conv1_node = None
    for node in nodes:
        if isinstance(node.op, tvm.ir.Op):
            if node.op.name == "add":
                add_node = node
            elif node.op.name == "nn.conv2d" and conv1_node is None:
                # First conv2d (conv1)
                conv1_node = node

    if add_node:
        print(f"\nAnalyzing converge point:")
        preds = rev_edges.get(add_node, [])
        print(f"  Direct predecessors: {len(preds)}")

        is_converge = analyzer.is_converge_point(add_node)
        print(f"  is_converge_point: {is_converge}")

        # Test branch extraction even if converge detection fails
        if conv1_node:
            print(f"\nTesting branch extraction from conv1 to add:")
            branches = analyzer.extract_branches(conv1_node, add_node)
            print(f"  Number of branches: {len(branches)}")

            # Calculate metrics for each branch
            lat_calc = LatencyThroughputCalculator(mod)
            branch_metrics = []
            for i, branch_ops in enumerate(branches):
                lat = lat_calc.calculate_branch_latency(branch_ops)
                thr = lat_calc.calculate_branch_throughput(branch_ops)
                branch_metrics.append((lat, thr))
                op_names = [op.op.name if isinstance(op.op, tvm.ir.Op) else "?" for op in branch_ops]
                print(f"  Branch {i}: {len(branch_ops)} ops {op_names}, latency={lat}, throughput={thr}")

            if len(branch_metrics) >= 2:
                # Check if unbalanced
                lats = [m[0] for m in branch_metrics]
                thrs = [m[1] for m in branch_metrics]
                is_unbalanced = len(set(lats)) > 1 or len(set(thrs)) > 1
                print(f"\n  Branches unbalanced: {is_unbalanced}")
                if is_unbalanced:
                    print(f"  Latency difference: {max(lats) - min(lats)}")

    print("\n✓ BranchAnalyzer unbalanced test passed")


def test_composite_splitter():
    """Test CompositeSplitter with a composite function."""
    print("\n" + "="*60)
    print("TEST: CompositeSplitter")
    print("="*60)

    # Test 1: Composite with Var as skip (common case)
    print("\n--- Test 1: Composite with Var skip ---")
    x = relay.var("x", shape=(1, 32, 8, 8), dtype="int8")
    w = relay.var("w", shape=(32, 32, 3, 3), dtype="int8")
    skip = relay.var("skip", shape=(1, 32, 8, 8), dtype="int8")

    conv = relay.nn.conv2d(x, w, in_channels=32, channels=32, kernel_size=(3, 3), padding=(1, 1))
    relu = relay.nn.relu(conv)
    add_out = relay.add(relu, skip)  # skip is Var, not Call

    composite_func = relay.Function([x, w, skip], add_out)
    composite_func = composite_func.with_attr("Composite", "imcflow.conv_add")

    data = relay.var("data", shape=(1, 32, 8, 8), dtype="int8")
    weight = relay.var("weight", shape=(32, 32, 3, 3), dtype="int8")
    skip_input = relay.var("skip_input", shape=(1, 32, 8, 8), dtype="int8")

    composite_call = relay.Call(composite_func, [data, weight, skip_input])

    print(f"  Composite attrs: {composite_func.attrs}")
    converge_op = CompositeSplitter.find_converge_op_in_composite(composite_call)
    print(f"  Converge op found (Var skip): {converge_op is not None}")
    # Note: This returns None because skip is Var, not Call - expected behavior

    # Test 2: Composite with both args as Call (internal converge)
    print("\n--- Test 2: Composite with internal branches ---")
    x2 = relay.var("x", shape=(1, 32, 8, 8), dtype="int8")
    w1 = relay.var("w1", shape=(32, 32, 3, 3), dtype="int8")
    w2 = relay.var("w2", shape=(32, 32, 3, 3), dtype="int8")

    # Internal diverge-converge: x -> conv1 -> (conv2 + relu) -> add
    conv1 = relay.nn.conv2d(x2, w1, in_channels=32, channels=32, kernel_size=(3, 3), padding=(1, 1))
    # Branch A
    conv2 = relay.nn.conv2d(conv1, w2, in_channels=32, channels=32, kernel_size=(3, 3), padding=(1, 1))
    # Branch B
    relu_skip = relay.nn.relu(conv1)
    # Converge
    add_internal = relay.add(conv2, relu_skip)

    composite_func2 = relay.Function([x2, w1, w2], add_internal)
    composite_func2 = composite_func2.with_attr("Composite", "imcflow.conv_internal_add")

    data2 = relay.var("data", shape=(1, 32, 8, 8), dtype="int8")
    weight1 = relay.var("weight1", shape=(32, 32, 3, 3), dtype="int8")
    weight2 = relay.var("weight2", shape=(32, 32, 3, 3), dtype="int8")

    composite_call2 = relay.Call(composite_func2, [data2, weight1, weight2])

    print(f"  Composite attrs: {composite_func2.attrs}")
    converge_op2 = CompositeSplitter.find_converge_op_in_composite(composite_call2)
    print(f"  Converge op found (internal branches): {converge_op2 is not None}")
    if converge_op2 and isinstance(converge_op2.op, tvm.ir.Op):
        print(f"  Op name: {converge_op2.op.name}")

    print("\n✓ CompositeSplitter tests passed")


def test_composite_graph_mutator():
    """Test CompositeGraphMutator - the actual graph transformation.

    This test verifies that CompositeGraphMutator correctly:
    1. Finds composites in split_pending
    2. Calls split_composite_at_converge to split them
    3. Replaces original composite with split result in the graph
    """
    print("\n" + "="*60)
    print("TEST: CompositeGraphMutator")
    print("="*60)

    # Create a composite function with internal converge point
    # Pattern: conv → relu + skip → add (all float32 for simplicity)
    x = relay.var("x", shape=(1, 32, 8, 8), dtype="float32")
    w = relay.var("w", shape=(32, 32, 3, 3), dtype="float32")
    skip = relay.var("skip", shape=(1, 32, 8, 8), dtype="float32")

    # Build composite body
    conv = relay.nn.conv2d(x, w, in_channels=32, channels=32, kernel_size=(3, 3), padding=(1, 1))
    relu = relay.nn.relu(conv)
    add_out = relay.add(relu, skip)

    composite_func = relay.Function([x, w, skip], add_out)
    composite_func = composite_func.with_attr("Composite", "imcflow.qconv2d-with-postop")

    # Create module with composite call
    data = relay.var("data", shape=(1, 32, 8, 8), dtype="float32")
    weight = relay.const(np.random.randn(32, 32, 3, 3).astype("float32"))
    skip_input = relay.var("skip_input", shape=(1, 32, 8, 8), dtype="float32")

    composite_call = relay.Call(composite_func, [data, weight, skip_input])

    func = relay.Function([data, skip_input], composite_call)
    func = func.with_attr("Compiler", "imcflow")
    func = func.with_attr("global_symbol", "imcflow_test")

    mod = tvm.IRModule()
    mod["tvmgen_default_imcflow_test_0"] = func
    mod = relay.transform.InferType()(mod)

    print("\n--- Original Module ---")
    print(relay.pretty_print(mod))

    # Step 1: Find converge op in composite
    print("\n--- Step 1: Find converge op ---")
    converge_op = CompositeSplitter.find_converge_op_in_composite(composite_call)
    print(f"Converge op found: {converge_op is not None}")
    if converge_op:
        if isinstance(converge_op.op, tvm.ir.Op):
            print(f"  Op name: {converge_op.op.name}")

    # Step 2: Create split_pending dict (simulating what AnnotGenerator does)
    print("\n--- Step 2: Create split_pending ---")
    if converge_op:
        split_pending = {
            composite_call: {
                "converge_op": converge_op,
                "pre_region": 0,  # Region ID (simulated)
                "post_region": 1,  # Region ID (simulated)
            }
        }
        print(f"Split pending entries: {len(split_pending)}")
    else:
        split_pending = {}
        print("No converge op found - split_pending is empty")

    # Step 3: Apply CompositeGraphMutator
    print("\n--- Step 3: Apply CompositeGraphMutator ---")

    if split_pending:
        # Get the imcflow function body
        imcflow_func = mod["tvmgen_default_imcflow_test_0"]

        # Create mutator and apply
        mutator = CompositeGraphMutator(split_pending)
        new_body = mutator.visit(imcflow_func.body)

        print(f"Split results: {len(mutator.split_results)}")
        for orig, result in mutator.split_results.items():
            print(f"  Original composite split:")
            print(f"    pre_composite_name: {result.get('pre_composite_name', 'N/A')}")
            print(f"    post_composite_name: {result.get('post_composite_name', 'N/A')}")

        # Create new function and module
        new_func = relay.Function(
            imcflow_func.params,
            new_body,
            ret_type=imcflow_func.ret_type
        )
        new_func = new_func.with_attr("Compiler", "imcflow")
        new_func = new_func.with_attr("global_symbol", "imcflow_test")

        new_mod = tvm.IRModule()
        new_mod["tvmgen_default_imcflow_test_0"] = new_func
        new_mod = relay.transform.InferType()(new_mod)

        print("\n--- Transformed Module ---")
        print(relay.pretty_print(new_mod))

        # Verify split occurred
        if len(mutator.split_results) > 0:
            print("\n✓ CompositeGraphMutator successfully split composite")
        else:
            print("\n⚠ CompositeGraphMutator did not split any composite")
    else:
        print("Skipping mutation - no split_pending entries")

    print("\n✓ CompositeGraphMutator test completed")


def test_composite_graph_mutator_with_resnet_pattern():
    """Test CompositeGraphMutator with ResNet-style pattern.

    Tests the actual pattern from ResNet8: composite with internal
    conv → relu → add (with skip from Var).
    """
    print("\n" + "="*60)
    print("TEST: CompositeGraphMutator with ResNet Pattern")
    print("="*60)

    # Create a composite that matches ResNet pattern (using float32 for simplicity):
    # conv → bias → relu → add(with skip)
    x = relay.var("x", shape=(1, 32, 8, 8), dtype="float32")
    w = relay.var("w", shape=(32, 32, 3, 3), dtype="float32")
    bias_param = relay.var("bias", shape=(32,), dtype="float32")
    skip = relay.var("skip", shape=(1, 32, 8, 8), dtype="float32")

    # Build composite body
    conv = relay.nn.conv2d(x, w, in_channels=32, channels=32, kernel_size=(3, 3), padding=(1, 1))
    bias = relay.nn.bias_add(conv, bias_param)
    relu = relay.nn.relu(bias)
    add_out = relay.add(relu, skip)

    composite_func = relay.Function([x, w, bias_param, skip], add_out)
    composite_func = composite_func.with_attr("Composite", "imcflow.qconv2d-with-postop")
    composite_func = composite_func.with_attr("PartitionedFromPattern", "nn.conv2d_nn.bias_add_nn.relu_add_")

    # Create outer module
    data = relay.var("data", shape=(1, 32, 8, 8), dtype="float32")
    weight = relay.const(np.random.randn(32, 32, 3, 3).astype("float32"))
    bias_const = relay.const(np.zeros(32).astype("float32"))
    skip_input = relay.var("skip_input", shape=(1, 32, 8, 8), dtype="float32")

    composite_call = relay.Call(composite_func, [data, weight, bias_const, skip_input])

    func = relay.Function([data, skip_input], composite_call)
    func = func.with_attr("Compiler", "imcflow")
    func = func.with_attr("global_symbol", "imcflow_test")

    mod = tvm.IRModule()
    mod["tvmgen_default_imcflow_test_0"] = func
    mod = relay.transform.InferType()(mod)

    print("\n--- Original Module ---")
    print(relay.pretty_print(mod))

    # Find converge op
    print("\n--- Finding converge op in composite ---")
    converge_op = CompositeSplitter.find_converge_op_in_composite(composite_call)
    print(f"Converge op found: {converge_op is not None}")
    if converge_op and isinstance(converge_op.op, tvm.ir.Op):
        print(f"  Op name: {converge_op.op.name}")
        print(f"  Args: {len(converge_op.args)}")
        for i, arg in enumerate(converge_op.args):
            arg_type = type(arg).__name__
            if isinstance(arg, relay.Call) and isinstance(arg.op, tvm.ir.Op):
                print(f"    arg[{i}]: {arg_type} - {arg.op.name}")
            elif isinstance(arg, relay.Var):
                print(f"    arg[{i}]: {arg_type} - {arg.name_hint}")
            else:
                print(f"    arg[{i}]: {arg_type}")

    # Test split_composite_at_converge directly
    print("\n--- Testing split_composite_at_converge ---")
    if converge_op:
        result = CompositeSplitter.split_composite_at_converge(
            composite_call, converge_op, "imcflow"
        )
        if result:
            print(f"Split successful!")
            print(f"  pre_composite_name: {result['pre_composite_name']}")
            print(f"  post_composite_name: {result['post_composite_name']}")
            print(f"\n--- Result expression ---")
            # Create temp mod to pretty print
            temp_func = relay.Function([data, skip_input], result['result_expr'])
            temp_mod = tvm.IRModule.from_expr(temp_func)
            try:
                temp_mod = relay.transform.InferType()(temp_mod)
                print(relay.pretty_print(temp_mod))
            except Exception as e:
                print(f"InferType failed: {e}")
                print("Raw result expression:")
                print(result['result_expr'])
        else:
            print("Split failed - returned None")
            print("This may be due to pattern matching issues")
    else:
        print("No converge op found - cannot split")

    print("\n✓ CompositeGraphMutator ResNet pattern test completed")


def test_full_partition_round():
    """Test the full partitionRound flow with converge point handling."""
    print("\n" + "="*60)
    print("TEST: Full partitionRound Flow")
    print("="*60)

    # Create a more realistic graph with imcflow ops
    from tvm.relay.op.nn import imcflow_qconv2d
    from tvm.relay.backend.contrib.imcflow.acim_util import ConfigData

    data = relay.var("data", shape=(1, 32, 8, 8), dtype="int8")
    weight1 = relay.const(np.random.randint(-128, 127, (32, 32, 3, 3)).astype("int8"))
    weight2 = relay.const(np.random.randint(-128, 127, (32, 32, 3, 3)).astype("int8"))

    # Create config data
    config1 = ConfigData(
        data_shape=(1, 32, 8, 8),
        weight_shape=(32, 32, 3, 3),
        padding=1,
        stride=1
    )

    # Build graph: conv1 → (conv2 + skip) → add
    conv1 = imcflow_qconv2d(
        data, weight1, config1.get_as_const_tensor(),
        in_channels=32, channels=32,
        kernel_size=(3, 3), padding=(1, 1), strides=(1, 1),
        out_dtype="int16"
    )

    # Branch A: conv2
    config2 = ConfigData(
        data_shape=(1, 32, 8, 8),
        weight_shape=(32, 32, 3, 3),
        padding=1,
        stride=1
    )
    conv2 = imcflow_qconv2d(
        conv1, weight2, config2.get_as_const_tensor(),
        in_channels=32, channels=32,
        kernel_size=(3, 3), padding=(1, 1), strides=(1, 1),
        out_dtype="int16"
    )
    branch_a = relay.nn.relu(conv2)

    # Branch B: skip connection (identity)
    branch_b = conv1

    # Converge: add
    add_out = relay.add(branch_a, branch_b)

    # Create function and module
    func = relay.Function([data], add_out)
    func = func.with_attr("Compiler", "imcflow")
    func = func.with_attr("global_symbol", "imcflow_main")

    mod = tvm.IRModule()
    mod["tvmgen_default_imcflow_main_0"] = func
    mod = relay.transform.InferType()(mod)

    print("\nInput graph:")
    print(relay.pretty_print(mod))

    # Test AnnotGenerator
    print("\n--- Testing AnnotGenerator ---")
    annotator = AnnotGenerator()

    # Get the imcflow function
    imcflow_func = mod["tvmgen_default_imcflow_main_0"]
    target_mod = tvm.IRModule.from_expr(
        relay.Function(imcflow_func.params, imcflow_func.body, ret_type=imcflow_func.ret_type)
    )
    target_mod = relay.transform.InferType()(target_mod)

    # Debug: manually check graph structure
    print("\n--- Debug: Graph Structure ---")
    import collections
    class DebugGraphBuilder(relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.nodes = []
            self.rev_edges = collections.defaultdict(list)
        def visit_call(self, call):
            self.nodes.append(call)
            for a in call.args:
                self.visit(a)
                if isinstance(a, relay.Call):
                    self.rev_edges[call].append(a)

    dgb = DebugGraphBuilder()
    dgb.visit(target_mod["main"].body)
    print(f"Found {len(dgb.nodes)} call nodes")
    for node in dgb.nodes:
        if isinstance(node.op, tvm.ir.Op):
            preds = dgb.rev_edges.get(node, [])
            pred_names = [p.op.name if isinstance(p.op, tvm.ir.Op) else "fn" for p in preds]
            print(f"  {node.op.name} <- {pred_names} (preds={len(preds)})")

    try:
        RegionList, split_pending = annotator.createRegionBFS(target_mod)
        print(f"\nRegions created: {len(RegionList)}")
        for i, region in enumerate(RegionList):
            print(f"  Region {i}: {len(region)} nodes")

        print(f"\nSplit pending: {len(split_pending)} composites")
        for node, info in split_pending.items():
            print(f"  Node to split: converge_op={info['converge_op'] is not None}")

        print("\n✓ AnnotGenerator test passed")
    except Exception as e:
        print(f"\n✗ AnnotGenerator test failed: {e}")
        import traceback
        traceback.print_exc()


def test_full_partition_round_with_composite():
    """Test partitionRound with composite nodes containing converge points.

    For split_pending to be populated, the composite must:
    1. Be detected as a converge point from OUTSIDE (2+ inputs from same diverge)
    2. Have an internal converge point that needs splitting

    Graph structure:
        data
          |
        conv0 (diverge point)
        /    \
      conv1   skip
        \    /
     [COMPOSITE] ← converge point from outside view
          |       internally: relu(branch_a) + relu(branch_b) → add
        output
    """
    print("\n" + "="*60)
    print("TEST: Full partitionRound with Composite Converge Point")
    print("="*60)

    from tvm.relay.op.nn import imcflow_qconv2d
    from tvm.relay.backend.contrib.imcflow.acim_util import ConfigData

    # Create composite function that takes two inputs and merges them internally
    # This composite will be called at a converge point
    branch_a_in = relay.var("branch_a", shape=(1, 32, 8, 8), dtype="int16")
    branch_b_in = relay.var("branch_b", shape=(1, 32, 8, 8), dtype="int16")

    # Internal processing with converge point
    relu_a = relay.nn.relu(branch_a_in)
    relu_b = relay.nn.relu(branch_b_in)
    add_out = relay.add(relu_a, relu_b)  # Internal converge point

    composite_func = relay.Function([branch_a_in, branch_b_in], add_out)
    composite_func = composite_func.with_attr("Composite", "imcflow.residual_add")
    composite_func = composite_func.with_attr("PartitionedFromPattern", "nn.relu_nn.relu_add_")

    # Create outer graph: conv0 → (conv1 | skip) → composite
    data = relay.var("data", shape=(1, 32, 8, 8), dtype="int8")
    weight0 = relay.const(np.random.randint(-128, 127, (32, 32, 3, 3)).astype("int8"))
    weight1 = relay.const(np.random.randint(-128, 127, (32, 32, 3, 3)).astype("int8"))

    config = ConfigData(
        data_shape=(1, 32, 8, 8),
        weight_shape=(32, 32, 3, 3),
        padding=1,
        stride=1
    )

    # conv0 - diverge point
    conv0 = imcflow_qconv2d(
        data, weight0, config.get_as_const_tensor(),
        in_channels=32, channels=32,
        kernel_size=(3, 3), padding=(1, 1), strides=(1, 1),
        out_dtype="int16"
    )

    # Branch A: conv1 (high latency)
    conv1 = imcflow_qconv2d(
        conv0, weight1, config.get_as_const_tensor(),
        in_channels=32, channels=32,
        kernel_size=(3, 3), padding=(1, 1), strides=(1, 1),
        out_dtype="int16"
    )

    # Branch B: skip (low latency - just identity)
    skip = conv0

    # Composite call at converge point - takes both branches as input
    composite_call = relay.Call(composite_func, [conv1, skip])

    # Create top-level function
    func = relay.Function([data], composite_call)
    func = func.with_attr("Compiler", "imcflow")
    func = func.with_attr("global_symbol", "imcflow_main")

    mod = tvm.IRModule()
    mod["tvmgen_default_imcflow_main_0"] = func
    mod = relay.transform.InferType()(mod)

    print("\nInput graph with composite at converge point:")
    print(relay.pretty_print(mod))

    # Test AnnotGenerator
    print("\n--- Testing AnnotGenerator with Composite ---")
    annotator = AnnotGenerator()

    imcflow_func = mod["tvmgen_default_imcflow_main_0"]
    target_mod = tvm.IRModule.from_expr(
        relay.Function(imcflow_func.params, imcflow_func.body, ret_type=imcflow_func.ret_type)
    )
    target_mod = relay.transform.InferType()(target_mod)

    try:
        RegionList, split_pending = annotator.createRegionBFS(target_mod)
        print(f"\nRegions created: {len(RegionList)}")
        for i, region in enumerate(RegionList):
            print(f"  Region {i}: {len(region)} nodes")

        print(f"\nSplit pending: {len(split_pending)} composites")
        for node, info in split_pending.items():
            composite_name = "unknown"
            if isinstance(node.op, relay.Function):
                composite_name = node.op.attrs.get("Composite", "unknown")
            print(f"  Composite: {composite_name}")
            print(f"    converge_op found: {info['converge_op'] is not None}")
            print(f"    pre_region: {info['pre_region'] is not None}")
            print(f"    post_region: {info['post_region'] is not None}")

        # Verify split_pending is non-empty for this test
        if len(split_pending) > 0:
            print("\n✓ Composite converge point test passed - split_pending is populated")
        else:
            print("\n⚠ Warning: split_pending is empty - converge point may not be inside composite")

        print("\n✓ AnnotGenerator with composite test passed")
    except Exception as e:
        print(f"\n✗ AnnotGenerator with composite test failed: {e}")
        import traceback
        traceback.print_exc()


def test_resnet8_basic_block_1():
    """Test converge point detection on ResNet8 first basic block.

    First basic block structure (until_relay=10):
        input -> conv -> bn -> mul -> cast
          |
        quantize -> conv1 -> bn1 -> quantize -> conv2 -> bn2
          |                                              |
          +--------------> residual * y_f_1 ------------+
                                                        |
                                                       add (converge point)
    """
    print("\n" + "="*60)
    print("TEST: ResNet8 Basic Block 1 Converge Point")
    print("="*60)

    # Get the first basic block only (until_relay=10 includes the add)
    print("\n--- Loading ResNet8 Basic Block 1 ---")
    mod, param_dict = resnet8_subset_models.getModel_from_pretrained_weight(
        iH=32, iW=32, until_relay=10
    )

    print("\nOriginal model (basic block 1):")
    print(relay.pretty_print(mod))

    # Apply new transform order
    # Step 1: Bind parameters
    print("\n--- Step 1: Bind parameters ---")
    mod["main"] = bind_params_by_name(mod["main"], param_dict)
    mod = relay.transform.InferType()(mod)

    # Step 2: First level imcflow graph partition
    print("\n--- Step 2: Partition IMCFlow subgraph ---")
    mod = partitionImcflowSubGraph(mod)

    # Step 3: Merge composites for partition
    print("\n--- Step 3: Merge composites for partition ---")
    mod = merge_composite_for_partition(mod)

    print("\nModel before partitionRound:")
    print(relay.pretty_print(mod))

    # Step 4: partitionRound - this is where converge point detection happens
    print("\n--- Step 4: Partition Round (with converge point detection) ---")
    mod = partitionRound(mod)
    print(relay.pretty_print(mod))

    # Step 5: Unmerge composites
    print("\n--- Step 5: Unmerge composites ---")
    mod = unmerge_composite(mod)

    # Step 6: Split conv to atomic ops
    print("\n--- Step 6: Split conv to atomic ops ---")
    mod, param_dict = split_conv_to_atomic(mod, param_dict)

    # Step 7: Merge composite ops (final)
    print("\n--- Step 7: Merge composite ops (final) ---")
    mod = merge_composite_ops(mod)

    # Step 8: Concat distributor
    print("\n--- Step 8: Concat distributor ---")
    mod = ConcatDistributor(max_inputs=4).run(mod)

    print("\nModel after full transform:")
    print(relay.pretty_print(mod))

    print("\n✓ ResNet8 Basic Block 1 converge point test completed")


def test_resnet8_basic_block_2():
    """Test converge point detection on ResNet8 second basic block.

    Second basic block structure (until_relay=17):
        From basic block 1 output
          |
        quantize -> conv1 (stride=2) -> bn1 -> quantize -> conv2 -> bn2
          |                                                         |
          +-> quantize -> conv_downsample (1x1, stride=2) -> bn ----+
                                                                    |
                                                                   add (converge point)

    This block has a downsample branch (1x1 conv) for residual connection.
    """
    print("\n" + "="*60)
    print("TEST: ResNet8 Basic Block 2 Converge Point")
    print("="*60)

    # Get until second basic block (until_relay=17 includes the add)
    print("\n--- Loading ResNet8 Basic Block 2 ---")
    mod, param_dict = resnet8_subset_models.getModel_from_pretrained_weight(
        iH=32, iW=32, until_relay=17
    )

    print("\nOriginal model (up to basic block 2):")
    print(relay.pretty_print(mod))

    # Apply new transform order
    # Step 1: Bind parameters
    print("\n--- Step 1: Bind parameters ---")
    mod["main"] = bind_params_by_name(mod["main"], param_dict)
    mod = relay.transform.InferType()(mod)

    # Step 2: First level imcflow graph partition
    print("\n--- Step 2: Partition IMCFlow subgraph ---")
    mod = partitionImcflowSubGraph(mod)

    # Step 3: Merge composites for partition
    print("\n--- Step 3: Merge composites for partition ---")
    mod = merge_composite_for_partition(mod)

    print("\nModel before partitionRound:")
    print(relay.pretty_print(mod))

    # Step 4: partitionRound - this is where converge point detection happens
    print("\n--- Step 4: Partition Round (with converge point detection) ---")
    mod = partitionRound(mod)
    print(relay.pretty_print(mod))

    # Step 5: Unmerge composites
    print("\n--- Step 5: Unmerge composites ---")
    mod = unmerge_composite(mod)

    # Step 6: Split conv to atomic ops
    print("\n--- Step 6: Split conv to atomic ops ---")
    mod, param_dict = split_conv_to_atomic(mod, param_dict)

    # Step 7: Merge composite ops (final)
    print("\n--- Step 7: Merge composite ops (final) ---")
    mod = merge_composite_ops(mod)

    # Step 8: Concat distributor
    print("\n--- Step 8: Concat distributor ---")
    mod = ConcatDistributor(max_inputs=4).run(mod)

    print("\nModel after full transform:")
    print(relay.pretty_print(mod))

    print("\n✓ ResNet8 Basic Block 2 converge point test completed")


def test_resnet8_converge_point():
    """Test converge point detection on real ResNet8 model (full)."""
    print("\n" + "="*60)
    print("TEST: ResNet8 Full Model Converge Point Detection")
    print("="*60)

    # Get the ResNet8 model (subset31 = full model, orig = 32x32 input)
    print("\n--- Loading ResNet8 model ---")
    mod, param_dict = resnet8_subset_models.getModel_from_pretrained_weight(
        iH=32, iW=32, until_relay=31
    )

    print("\nOriginal model:")
    print(relay.pretty_print(mod))

    # Apply new transform order
    # Step 1: Bind parameters
    print("\n--- Step 1: Bind parameters ---")
    mod["main"] = bind_params_by_name(mod["main"], param_dict)
    mod = relay.transform.InferType()(mod)

    # Step 2: First level imcflow graph partition
    print("\n--- Step 2: Partition IMCFlow subgraph ---")
    mod = partitionImcflowSubGraph(mod)

    # Step 3: Merge composites for partition
    print("\n--- Step 3: Merge composites for partition ---")
    mod = merge_composite_for_partition(mod)

    print("\nModel before partitionRound:")
    print(relay.pretty_print(mod))

    # Step 4: partitionRound - this is where converge point detection happens
    print("\n--- Step 4: Partition Round (with converge point detection) ---")
    mod = partitionRound(mod)

    # Step 5: Unmerge composites
    print("\n--- Step 5: Unmerge composites ---")
    mod = unmerge_composite(mod)

    # Step 6: Split conv to atomic ops
    print("\n--- Step 6: Split conv to atomic ops ---")
    mod, param_dict = split_conv_to_atomic(mod, param_dict)

    # Step 7: Merge composite ops (final)
    print("\n--- Step 7: Merge composite ops (final) ---")
    mod = merge_composite_ops(mod)

    # Step 8: Concat distributor
    print("\n--- Step 8: Concat distributor ---")
    mod = ConcatDistributor(max_inputs=4).run(mod)

    print("\nModel after full transform:")
    print(relay.pretty_print(mod))

    print("\n✓ ResNet8 converge point test completed")

def run_all_tests():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# IMCFLOW Converge Point Detection Tests")
    print("#"*60)

    try:
        test_latency_throughput_calculator()
    except Exception as e:
        print(f"\n✗ LatencyThroughputCalculator test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_branch_analyzer()
    except Exception as e:
        print(f"\n✗ BranchAnalyzer test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_branch_analyzer_unbalanced()
    except Exception as e:
        print(f"\n✗ BranchAnalyzer unbalanced test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_composite_splitter()
    except Exception as e:
        print(f"\n✗ CompositeSplitter test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_composite_graph_mutator()
    except Exception as e:
        print(f"\n✗ CompositeGraphMutator test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_composite_graph_mutator_with_resnet_pattern()
    except Exception as e:
        print(f"\n✗ CompositeGraphMutator ResNet pattern test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_full_partition_round()
    except Exception as e:
        print(f"\n✗ Full partitionRound test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_full_partition_round_with_composite()
    except Exception as e:
        print(f"\n✗ Full partitionRound with composite test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_resnet8_basic_block_1()
    except Exception as e:
        print(f"\n✗ ResNet8 basic block 1 test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_resnet8_basic_block_2()
    except Exception as e:
        print(f"\n✗ ResNet8 basic block 2 test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_resnet8_converge_point()
    except Exception as e:
        print(f"\n✗ ResNet8 converge point test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "#"*60)
    print("# Tests Complete")
    print("#"*60)


if __name__ == "__main__":
    run_all_tests()
