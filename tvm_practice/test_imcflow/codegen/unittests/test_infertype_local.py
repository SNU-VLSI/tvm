"""
Test InferTypeLocal functionality
"""
import tvm
from tvm import relay
from tvm.relay import transform

def test_infer_type_local_basic():
  """Test basic InferTypeLocal usage"""
  print("=" * 80)
  print("Test 1: Basic InferTypeLocal Usage")
  print("=" * 80)
  
  # Create a simple function
  x = relay.var("x", shape=(1, 3, 224, 224), dtype="float32")
  y = relay.nn.relu(x)
  z = relay.nn.relu(y)
  
  func = relay.Function([x], z)
  
  print("\n1. Before InferTypeLocal:")
  print("-" * 80)
  print(func)
  
  try:
    # Try InferTypeLocal directly on function
    print("\n2. Attempting InferTypeLocal on function...")
    inferred_func = relay.transform.InferTypeLocal(func)
    print("SUCCESS!")
    print("-" * 80)
    print(inferred_func)
    
    # Check if types are populated
    print("\n3. Checking types:")
    print(f"  Function return type: {inferred_func.ret_type}")
    print(f"  Body checked_type: {inferred_func.body.checked_type}")
  except Exception as e:
    print(f"FAILED: {e}")
    print(f"Error type: {type(e)}")

def test_infer_type_local_vs_infer_type():
  """Compare InferTypeLocal vs InferType"""
  print("\n" + "=" * 80)
  print("Test 2: InferTypeLocal vs InferType Comparison")
  print("=" * 80)
  
  # Create a function with updated parameters (simulating param update scenario)
  x = relay.var("x", shape=(1, 64, 56, 56), dtype="float32")
  y = relay.nn.relu(x)
  
  func = relay.Function([x], y, ret_type=None)
  
  print("\n1. Original function (no return type):")
  print("-" * 80)
  print(func)
  
  # Method 1: InferTypeLocal
  print("\n2. Method 1: Using InferTypeLocal")
  print("-" * 80)
  try:
    func_local = relay.transform.InferTypeLocal(func)
    print("InferTypeLocal SUCCESS!")
    print(f"Return type: {func_local.ret_type}")
    print(f"Body checked_type: {func_local.body.checked_type}")
  except Exception as e:
    print(f"InferTypeLocal FAILED: {e}")
  
  # Method 2: InferType with IRModule
  print("\n3. Method 2: Using InferType with IRModule")
  print("-" * 80)
  try:
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    func_module = mod["main"]
    print("InferType SUCCESS!")
    print(f"Return type: {func_module.ret_type}")
    print(f"Body checked_type: {func_module.body.checked_type}")
  except Exception as e:
    print(f"InferType FAILED: {e}")

def test_infer_type_local_with_param_update():
  """Test InferTypeLocal after parameter update (real use case)"""
  print("\n" + "=" * 80)
  print("Test 3: InferTypeLocal After Parameter Type Update")
  print("=" * 80)
  
  # Original function
  x = relay.var("x", shape=(1, 64, 56, 56), dtype="float32")
  y = relay.nn.relu(x)
  z = relay.add(y, y)
  
  original_func = relay.Function([x], z)
  
  print("\n1. Original function:")
  print("-" * 80)
  print(original_func)
  
  # Simulate parameter update (e.g., NCHW -> NCHW16c)
  new_x = relay.var("x", shape=(1, 4, 56, 56, 16), dtype="float32")
  
  # Update function body with new parameter
  new_body = relay.bind(original_func.body, {x: new_x})
  
  updated_func = relay.Function([new_x], new_body, ret_type=None)
  
  print("\n2. Updated function (new param shape, no return type):")
  print("-" * 80)
  print(updated_func)
  
  # Try InferTypeLocal
  print("\n3. Attempting InferTypeLocal...")
  print("-" * 80)
  try:
    inferred_func = relay.transform.InferTypeLocal(updated_func)
    print("SUCCESS!")
    print(f"Return type inferred: {inferred_func.ret_type}")
    print(f"Body checked_type: {inferred_func.body.checked_type}")
    print("\nFull function:")
    print(inferred_func)
  except Exception as e:
    print(f"FAILED: {e}")
    print(f"Error type: {type(e)}")
    
  # Compare with IRModule approach
  print("\n4. Comparing with IRModule approach...")
  print("-" * 80)
  try:
    mod = tvm.IRModule.from_expr(updated_func)
    mod = relay.transform.InferType()(mod)
    func_module = mod["main"]
    print("IRModule approach SUCCESS!")
    print(f"Return type: {func_module.ret_type}")
    print(f"Body checked_type: {func_module.body.checked_type}")
  except Exception as e:
    print(f"IRModule approach FAILED: {e}")

def test_infer_type_local_with_global_var():
  """Test InferTypeLocal with GlobalVar (function calls)"""
  print("\n" + "=" * 80)
  print("Test 4: InferTypeLocal With GlobalVar/Function Calls")
  print("=" * 80)
  
  # Create inner function
  x = relay.var("x", shape=(1, 64, 56, 56), dtype="float32")
  inner_body = relay.nn.relu(x)
  inner_func = relay.Function([x], inner_body)
  
  # Create main function that calls inner function
  y = relay.var("y", shape=(1, 64, 56, 56), dtype="float32")
  # Note: This creates a direct function call, not GlobalVar
  call_result = relay.Call(inner_func, [y])
  main_func = relay.Function([y], call_result, ret_type=None)
  
  print("\n1. Main function with inner function call:")
  print("-" * 80)
  print(main_func)
  
  print("\n2. Attempting InferTypeLocal on main function...")
  print("-" * 80)
  try:
    inferred = relay.transform.InferTypeLocal(main_func)
    print("SUCCESS!")
    print(f"Return type: {inferred.ret_type}")
    print(f"Body checked_type: {inferred.body.checked_type}")
  except Exception as e:
    print(f"FAILED: {e}")
    
  # Now test with actual GlobalVar
  print("\n3. Testing with GlobalVar (requires IRModule)...")
  print("-" * 80)
  mod = tvm.IRModule()
  inner_gv = relay.GlobalVar("inner_func")
  mod[inner_gv] = inner_func
  
  # Create main that calls GlobalVar
  call_result2 = relay.Call(inner_gv, [y])
  main_func2 = relay.Function([y], call_result2, ret_type=None)
  
  print("Main function with GlobalVar call:")
  print(main_func2)
  
  print("\n4. Attempting InferTypeLocal on function with GlobalVar...")
  try:
    inferred2 = relay.transform.InferTypeLocal(main_func2)
    print("SUCCESS!")
    print(f"Return type: {inferred2.ret_type}")
  except Exception as e:
    print(f"FAILED: {e}")
    print("Note: InferTypeLocal likely cannot handle GlobalVar without module context")

def test_infer_type_local_limitations():
  """Test limitations and edge cases of InferTypeLocal"""
  print("\n" + "=" * 80)
  print("Test 5: InferTypeLocal Limitations and Edge Cases")
  print("=" * 80)
  
  # Case 1: Function with Tuple return
  print("\n1. Case: Function with Tuple return")
  print("-" * 80)
  x = relay.var("x", shape=(1, 64, 56, 56), dtype="float32")
  out1 = relay.nn.relu(x)
  out2 = relay.nn.relu(x)
  tuple_out = relay.Tuple([out1, out2])
  
  func_tuple = relay.Function([x], tuple_out, ret_type=None)
  print(f"Function: {func_tuple}")
  
  try:
    inferred = relay.transform.InferTypeLocal(func_tuple)
    print("SUCCESS!")
    print(f"Return type: {inferred.ret_type}")
  except Exception as e:
    print(f"FAILED: {e}")
  
  # Case 2: Function with TupleGetItem
  print("\n2. Case: Function with TupleGetItem")
  print("-" * 80)
  tgi = relay.TupleGetItem(tuple_out, 0)
  func_tgi = relay.Function([x], tgi, ret_type=None)
  print(f"Function: {func_tgi}")
  
  try:
    inferred = relay.transform.InferTypeLocal(func_tgi)
    print("SUCCESS!")
    print(f"Return type: {inferred.ret_type}")
    print(f"Body checked_type: {inferred.body.checked_type}")
  except Exception as e:
    print(f"FAILED: {e}")
  
  # Case 3: Function with Let binding
  print("\n3. Case: Function with Let binding")
  print("-" * 80)
  var = relay.var("tmp")
  value = relay.nn.relu(x)
  body = relay.add(var, var)
  let_expr = relay.Let(var, value, body)
  func_let = relay.Function([x], let_expr, ret_type=None)
  
  print(f"Function: {func_let}")
  try:
    inferred = relay.transform.InferTypeLocal(func_let)
    print("SUCCESS!")
    print(f"Return type: {inferred.ret_type}")
  except Exception as e:
    print(f"FAILED: {e}")

def test_infer_type_local_best_practices():
  """Demonstrate best practices for using InferTypeLocal"""
  print("\n" + "=" * 80)
  print("Test 6: InferTypeLocal Best Practices")
  print("=" * 80)
  
  print("\n1. When to use InferTypeLocal:")
  print("-" * 80)
  print("  ✓ Standalone functions without GlobalVar dependencies")
  print("  ✓ After parameter updates within a single function")
  print("  ✓ Quick type inference for simple expressions")
  print("  ✓ When you don't need module-level context")
  
  print("\n2. When to use InferType (with IRModule):")
  print("-" * 80)
  print("  ✓ Functions with GlobalVar calls")
  print("  ✓ Multiple functions in a module")
  print("  ✓ Cross-function type dependencies")
  print("  ✓ When you need full module context")
  
  print("\n3. Example: Safe usage pattern")
  print("-" * 80)
  
  # Create a simple function
  x = relay.var("x", shape=(1, 64, 56, 56), dtype="float32")
  y = relay.nn.relu(x)
  func = relay.Function([x], y, ret_type=None)
  
  print("Original function:")
  print(func)
  
  # Safe pattern: try InferTypeLocal, fallback to IRModule
  print("\nTrying InferTypeLocal...")
  try:
    inferred_func = relay.transform.InferTypeLocal(func)
    print("✓ InferTypeLocal succeeded!")
    print(f"  Return type: {inferred_func.ret_type}")
  except Exception as e:
    print(f"✗ InferTypeLocal failed: {e}")
    print("\nFalling back to IRModule approach...")
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    inferred_func = mod["main"]
    print("✓ IRModule approach succeeded!")
    print(f"  Return type: {inferred_func.ret_type}")

if __name__ == "__main__":
  # Run all InferTypeLocal tests
  test_infer_type_local_basic()
  test_infer_type_local_vs_infer_type()
  test_infer_type_local_with_param_update()
  test_infer_type_local_with_global_var()
  test_infer_type_local_limitations()
  test_infer_type_local_best_practices()
  
  print("\n" + "=" * 80)
  print("All InferTypeLocal tests completed!")
  print("=" * 80)
