"""
Test to check if TupleGetItem has checked_type after InferType
"""
import tvm
from tvm import relay
import numpy as np

def test_tuple_get_item_checked_type():
    """Test if TupleGetItem has checked_type"""
    print("=" * 80)
    print("Testing TupleGetItem checked_type")
    print("=" * 80)
    
    # Create a simple graph with Tuple and TupleGetItem
    # Input
    x = relay.var("x", shape=(1, 3, 224, 224), dtype="float32")
    
    # Create two outputs
    out1 = relay.nn.relu(x)
    out2 = relay.nn.relu(x)
    
    # Create a tuple
    tuple_out = relay.Tuple([out1, out2])
    
    # Get items from tuple
    tgi0 = relay.TupleGetItem(tuple_out, 0)
    tgi1 = relay.TupleGetItem(tuple_out, 1)
    
    # Use the tuple items
    result = relay.add(tgi0, tgi1)
    
    # Create function
    func = relay.Function([x], result)
    
    print("\n1. Before InferType:")
    print("-" * 80)
    print(f"Function:\n{func}")
    print(f"\nFunction body type: {type(func.body)}")
    print(f"Function body: {func.body}")
    
    # Check TupleGetItem before InferType
    # We need to traverse the function to find TupleGetItem nodes
    class TGIFinder(relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.tgi_nodes = []
        
        def visit_tuple_getitem(self, tgi):
            self.tgi_nodes.append(tgi)
            super().visit_tuple_getitem(tgi)
    
    finder = TGIFinder()
    finder.visit(func)
    
    print(f"\nFound {len(finder.tgi_nodes)} TupleGetItem nodes before InferType")
    for i, tgi in enumerate(finder.tgi_nodes):
        print(f"  TGI[{i}]: index={tgi.index}")
        try:
            print(f"    checked_type: {tgi.checked_type}")
        except Exception as e:
            print(f"    checked_type: NOT AVAILABLE ({e})")
    
    # Create module and run InferType
    mod = tvm.IRModule.from_expr(func)
    print("\n2. Running InferType...")
    mod = relay.transform.InferType()(mod)
    
    # Get the main function after InferType
    main_func = mod["main"]
    
    print("\n3. After InferType:")
    print("-" * 80)
    print(f"Function:\n{main_func}")
    
    # Find TupleGetItem nodes after InferType
    finder2 = TGIFinder()
    finder2.visit(main_func)
    
    print(f"\nFound {len(finder2.tgi_nodes)} TupleGetItem nodes after InferType")
    for i, tgi in enumerate(finder2.tgi_nodes):
        print(f"  TGI[{i}]: index={tgi.index}")
        try:
            checked_type = tgi.checked_type
            print(f"    checked_type: {checked_type}")
            print(f"    checked_type type: {type(checked_type)}")
            if hasattr(checked_type, 'shape'):
                print(f"    shape: {checked_type.shape}")
            if hasattr(checked_type, 'dtype'):
                print(f"    dtype: {checked_type.dtype}")
        except Exception as e:
            print(f"    checked_type: ERROR - {e}")
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)

def test_tuple_get_item_from_function_call():
    """Test TupleGetItem when it comes from a function call"""
    print("\n" + "=" * 80)
    print("Testing TupleGetItem from Function Call")
    print("=" * 80)
    
    # Create a function that returns a tuple
    x = relay.var("x", shape=(1, 3, 224, 224), dtype="float32")
    out1 = relay.nn.relu(x)
    out2 = relay.nn.relu(x) 
    tuple_out = relay.Tuple([out1, out2])
    
    # Create the function
    inner_func = relay.Function([x], tuple_out)
    
    # Create main function that calls inner function
    y = relay.var("y", shape=(1, 3, 224, 224), dtype="float32")
    call_result = relay.Call(inner_func, [y])
    
    # Get items from the call result
    tgi0 = relay.TupleGetItem(call_result, 0)
    tgi1 = relay.TupleGetItem(call_result, 1)
    
    result = relay.add(tgi0, tgi1)
    
    main_func = relay.Function([y], result)
    
    print("\n1. Before InferType:")
    print("-" * 80)
    print(f"Main function:\n{main_func}")
    
    # Create module and run InferType
    mod = tvm.IRModule.from_expr(main_func)
    print("\n2. Running InferType...")
    mod = relay.transform.InferType()(mod)
    
    # Get the main function after InferType
    main_func_typed = mod["main"]
    
    print("\n3. After InferType:")
    print("-" * 80)
    print(f"Main function:\n{main_func_typed}")
    
    # Find TupleGetItem nodes
    class TGIFinder(relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.tgi_nodes = []
        
        def visit_tuple_getitem(self, tgi):
            self.tgi_nodes.append(tgi)
            super().visit_tuple_getitem(tgi)
    
    finder = TGIFinder()
    finder.visit(main_func_typed)
    
    print(f"\nFound {len(finder.tgi_nodes)} TupleGetItem nodes after InferType")
    for i, tgi in enumerate(finder.tgi_nodes):
        print(f"  TGI[{i}]: index={tgi.index}")
        print(f"    tuple_value type: {type(tgi.tuple_value)}")
        try:
            checked_type = tgi.checked_type
            print(f"    checked_type: {checked_type}")
            if hasattr(checked_type, 'shape'):
                print(f"    shape: {checked_type.shape}")
            if hasattr(checked_type, 'dtype'):
                print(f"    dtype: {checked_type.dtype}")
        except Exception as e:
            print(f"    checked_type: ERROR - {e}")
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)

def test_tuple_get_item_from_split():
    """Test TupleGetItem when it comes from split operation"""
    print("\n" + "=" * 80)
    print("Testing TupleGetItem from Split Operation")
    print("=" * 80)
    
    # Create a simple graph with split
    x = relay.var("x", shape=(1, 64, 56, 56), dtype="float32")
    
    # Split along channel dimension into 2 parts
    # split returns a Tuple of tensors
    split_result = relay.split(x, indices_or_sections=2, axis=1)
    
    print("\n1. Split result type:")
    print(f"  Type: {type(split_result)}")
    print(f"  Split result: {split_result}")
    
    # TupleWrapper provides convenient indexing
    # We can access elements directly or use the underlying tuple
    split_0 = split_result[0]  # Using TupleWrapper indexing
    split_1 = split_result[1]
    
    print(f"\n  split_0 type: {type(split_0)}")
    print(f"  split_1 type: {type(split_1)}")
    
    # Use the split outputs
    result = relay.add(split_0, split_1)
    
    # Create function
    func = relay.Function([x], result)
    
    print("\n2. Before InferType:")
    print("-" * 80)
    print(f"Function:\n{func}")
    
    # Check TupleGetItem before InferType
    class TGIFinder(relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.tgi_nodes = []
            self.split_nodes = []
        
        def visit_tuple_getitem(self, tgi):
            self.tgi_nodes.append(tgi)
            super().visit_tuple_getitem(tgi)
        
        def visit_call(self, call):
            if hasattr(call.op, 'name') and call.op.name == 'split':
                self.split_nodes.append(call)
            super().visit_call(call)
    
    finder = TGIFinder()
    finder.visit(func)
    
    print(f"\nFound {len(finder.split_nodes)} split nodes")
    print(f"Found {len(finder.tgi_nodes)} TupleGetItem nodes before InferType")
    for i, tgi in enumerate(finder.tgi_nodes):
        print(f"  TGI[{i}]: index={tgi.index}")
        print(f"    tuple_value type: {type(tgi.tuple_value)}")
        try:
            print(f"    checked_type: {tgi.checked_type}")
        except Exception as e:
            print(f"    checked_type: NOT AVAILABLE ({e})")
    
    # Create module and run InferType
    mod = tvm.IRModule.from_expr(func)
    print("\n3. Running InferType...")
    mod = relay.transform.InferType()(mod)
    
    # Get the main function after InferType
    main_func = mod["main"]
    
    print("\n4. After InferType:")
    print("-" * 80)
    print(f"Function:\n{main_func}")
    
    # Find nodes after InferType
    finder2 = TGIFinder()
    finder2.visit(main_func)
    
    print(f"\nFound {len(finder2.split_nodes)} split nodes after InferType")
    for i, split_node in enumerate(finder2.split_nodes):
        print(f"  Split[{i}]:")
        try:
            checked_type = split_node.checked_type
            print(f"    checked_type: {checked_type}")
            print(f"    checked_type type: {type(checked_type)}")
            if hasattr(checked_type, 'fields'):
                print(f"    number of fields: {len(checked_type.fields)}")
                for j, field in enumerate(checked_type.fields):
                    print(f"    field[{j}]: {field}")
        except Exception as e:
            print(f"    checked_type: ERROR - {e}")
    
    print(f"\nFound {len(finder2.tgi_nodes)} TupleGetItem nodes after InferType")
    for i, tgi in enumerate(finder2.tgi_nodes):
        print(f"  TGI[{i}]: index={tgi.index}")
        print(f"    tuple_value type: {type(tgi.tuple_value)}")
        try:
            # Check if tuple_value is a Call to split
            if isinstance(tgi.tuple_value, relay.Call):
                if hasattr(tgi.tuple_value.op, 'name'):
                    print(f"    tuple_value is Call to: {tgi.tuple_value.op.name}")
            
            checked_type = tgi.checked_type
            print(f"    checked_type: {checked_type}")
            print(f"    checked_type type: {type(checked_type)}")
            if hasattr(checked_type, 'shape'):
                print(f"    shape: {checked_type.shape}")
            if hasattr(checked_type, 'dtype'):
                print(f"    dtype: {checked_type.dtype}")
        except Exception as e:
            print(f"    checked_type: ERROR - {e}")
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)

def test_tuple_get_item_from_concatenate():
    """Test operations that might use tuple outputs"""
    print("\n" + "=" * 80)
    print("Testing TupleGetItem patterns with concatenate")
    print("=" * 80)
    
    # Create inputs
    x1 = relay.var("x1", shape=(1, 32, 56, 56), dtype="float32")
    x2 = relay.var("x2", shape=(1, 32, 56, 56), dtype="float32")
    
    # Split x1 - TupleWrapper provides convenient indexing
    split_result = relay.split(x1, indices_or_sections=2, axis=1)
    s1_0 = split_result[0]
    s1_1 = split_result[1]
    
    # Split x2
    split_result2 = relay.split(x2, indices_or_sections=2, axis=1)
    s2_0 = split_result2[0]
    s2_1 = split_result2[1]
    
    # Concatenate splits
    concat = relay.concatenate([s1_0, s2_0, s1_1, s2_1], axis=1)
    
    # Create function
    func = relay.Function([x1, x2], concat)
    
    print("\n1. Before InferType:")
    print("-" * 80)
    print(f"Function:\n{func}")
    
    # Create module and run InferType
    mod = tvm.IRModule.from_expr(func)
    print("\n2. Running InferType...")
    mod = relay.transform.InferType()(mod)
    
    # Get the main function after InferType
    main_func = mod["main"]
    
    print("\n3. After InferType:")
    print("-" * 80)
    print(f"Function:\n{main_func}")
    
    # Find TupleGetItem nodes
    class TGIFinder(relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.tgi_nodes = []
        
        def visit_tuple_getitem(self, tgi):
            self.tgi_nodes.append(tgi)
            super().visit_tuple_getitem(tgi)
    
    finder = TGIFinder()
    finder.visit(main_func)
    
    print(f"\nFound {len(finder.tgi_nodes)} TupleGetItem nodes after InferType")
    for i, tgi in enumerate(finder.tgi_nodes):
        print(f"  TGI[{i}]: index={tgi.index}")
        try:
            checked_type = tgi.checked_type
            print(f"    checked_type: {checked_type}")
            if hasattr(checked_type, 'shape'):
                print(f"    shape: {checked_type.shape}")
        except Exception as e:
            print(f"    checked_type: ERROR - {e}")
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)

if __name__ == "__main__":
    test_tuple_get_item_checked_type()
    test_tuple_get_item_from_function_call()
    test_tuple_get_item_from_split()
    test_tuple_get_item_from_concatenate()
