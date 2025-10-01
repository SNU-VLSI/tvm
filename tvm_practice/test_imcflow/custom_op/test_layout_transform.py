"""
Test script for TVM's layout_transform operator
Tests various layout transformations like NCHW->NHWC, NHWC->NCHW, and packed layouts
"""

import numpy as np
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.contrib import graph_executor
import tvm.testing

np.random.seed(42)


def test_layout_transform_nchw_to_nhwc():
    """Test NCHW to NHWC layout transformation"""
    print("\n" + "="*60)
    print("Test: NCHW -> NHWC")
    print("="*60)
    
    # Define input shape in NCHW format
    batch, channel, height, width = 1, 3, 8, 8
    shape_nchw = (batch, channel, height, width)
    
    # Create relay function
    x = relay.var("x", shape=shape_nchw, dtype="float32")
    y = relay.layout_transform(x, "NCHW", "NHWC")
    func = relay.Function([x], y)
    
    # Create test data
    x_data = np.random.uniform(size=shape_nchw).astype("float32")
    
    # Build and run
    target = "llvm"
    dev = tvm.cpu(0)
    
    mod = tvm.IRModule.from_expr(func)
    mod = transform.InferType()(mod)
    print("Input IR:")
    print(mod)
    
    graph, lib, params = relay.build(mod, target=target)
    m = graph_executor.create(graph, lib, device=dev)
    m.set_input("x", x_data)
    m.run()
    
    # Get output
    output = m.get_output(0).asnumpy()
    
    # Verify transformation: NCHW (1,3,8,8) -> NHWC (1,8,8,3)
    expected_shape = (batch, height, width, channel)
    print(f"Input shape (NCHW): {x_data.shape}")
    print(f"Output shape (NHWC): {output.shape}")
    print(f"Expected shape: {expected_shape}")
    
    assert output.shape == expected_shape, f"Shape mismatch: {output.shape} != {expected_shape}"
    
    # Verify data correctness by comparing with numpy transpose
    expected_output = np.transpose(x_data, (0, 2, 3, 1))  # NCHW -> NHWC
    tvm.testing.assert_allclose(output, expected_output, rtol=1e-5, atol=1e-5)
    
    print("✓ Test passed!")
    return True


def test_layout_transform_nhwc_to_nchw():
    """Test NHWC to NCHW layout transformation"""
    print("\n" + "="*60)
    print("Test: NHWC -> NCHW")
    print("="*60)
    
    # Define input shape in NHWC format
    batch, height, width, channel = 1, 8, 8, 3
    shape_nhwc = (batch, height, width, channel)
    
    # Create relay function
    x = relay.var("x", shape=shape_nhwc, dtype="float32")
    y = relay.layout_transform(x, "NHWC", "NCHW")
    func = relay.Function([x], y)
    
    # Create test data
    x_data = np.random.uniform(size=shape_nhwc).astype("float32")
    
    # Build and run
    target = "llvm"
    dev = tvm.cpu(0)
    
    mod = tvm.IRModule.from_expr(func)
    mod = transform.InferType()(mod)
    print("Input IR:")
    print(mod)
    
    graph, lib, params = relay.build(mod, target=target)
    m = graph_executor.create(graph, lib, device=dev)
    m.set_input("x", x_data)
    m.run()
    
    # Get output
    output = m.get_output(0).asnumpy()
    
    # Verify transformation: NHWC (1,8,8,3) -> NCHW (1,3,8,8)
    expected_shape = (batch, channel, height, width)
    print(f"Input shape (NHWC): {x_data.shape}")
    print(f"Output shape (NCHW): {output.shape}")
    print(f"Expected shape: {expected_shape}")
    
    assert output.shape == expected_shape, f"Shape mismatch: {output.shape} != {expected_shape}"
    
    # Verify data correctness by comparing with numpy transpose
    expected_output = np.transpose(x_data, (0, 3, 1, 2))  # NHWC -> NCHW
    tvm.testing.assert_allclose(output, expected_output, rtol=1e-5, atol=1e-5)
    
    print("✓ Test passed!")
    return True


def test_layout_transform_nchw_to_nchw4c():
    """Test NCHW to NCHW4c packed layout transformation"""
    print("\n" + "="*60)
    print("Test: NCHW -> NCHW4c (Packed Layout)")
    print("="*60)
    
    # Define input shape in NCHW format
    # Channel must be divisible by 4 for NCHW4c
    batch, channel, height, width = 1, 8, 16, 16
    shape_nchw = (batch, channel, height, width)
    
    # Create relay function
    x = relay.var("x", shape=shape_nchw, dtype="float32")
    y = relay.layout_transform(x, "NCHW", "NCHW4c")
    func = relay.Function([x], y)
    
    # Create test data
    x_data = np.random.uniform(size=shape_nchw).astype("float32")
    
    # Build and run
    target = "llvm"
    dev = tvm.cpu(0)
    
    mod = tvm.IRModule.from_expr(func)
    mod = transform.InferType()(mod)
    print("Input IR:")
    print(mod)
    
    graph, lib, params = relay.build(mod, target=target)
    m = graph_executor.create(graph, lib, device=dev)
    m.set_input("x", x_data)
    m.run()
    
    # Get output
    output = m.get_output(0).asnumpy()
    
    # Verify transformation: NCHW (1,8,16,16) -> NCHW4c (1,2,16,16,4)
    # C=8 is split into 2 blocks of 4 channels
    expected_shape = (batch, channel // 4, height, width, 4)
    print(f"Input shape (NCHW): {x_data.shape}")
    print(f"Output shape (NCHW4c): {output.shape}")
    print(f"Expected shape: {expected_shape}")
    
    assert output.shape == expected_shape, f"Shape mismatch: {output.shape} != {expected_shape}"
    
    # Verify data correctness
    # Reshape input to match the packed layout
    expected_output = x_data.reshape(batch, channel // 4, 4, height, width)
    expected_output = np.transpose(expected_output, (0, 1, 3, 4, 2))
    tvm.testing.assert_allclose(output, expected_output, rtol=1e-5, atol=1e-5)
    
    print("✓ Test passed!")
    return True


def test_layout_transform_nchw4c_to_nchw():
    """Test NCHW4c to NCHW unpacking transformation"""
    print("\n" + "="*60)
    print("Test: NCHW4c -> NCHW (Unpacking)")
    print("="*60)
    
    # Define input shape in NCHW4c format
    batch, channel_blocks, height, width, pack_size = 1, 2, 16, 16, 4
    shape_nchw4c = (batch, channel_blocks, height, width, pack_size)
    
    # Create relay function
    x = relay.var("x", shape=shape_nchw4c, dtype="float32")
    y = relay.layout_transform(x, "NCHW4c", "NCHW")
    func = relay.Function([x], y)
    
    # Create test data
    x_data = np.random.uniform(size=shape_nchw4c).astype("float32")
    
    # Build and run
    target = "llvm"
    dev = tvm.cpu(0)
    
    mod = tvm.IRModule.from_expr(func)
    mod = transform.InferType()(mod)
    print("Input IR:")
    print(mod)
    
    graph, lib, params = relay.build(mod, target=target)
    m = graph_executor.create(graph, lib, device=dev)
    m.set_input("x", x_data)
    m.run()
    
    # Get output
    output = m.get_output(0).asnumpy()
    
    # Verify transformation: NCHW4c (1,2,16,16,4) -> NCHW (1,8,16,16)
    expected_shape = (batch, channel_blocks * pack_size, height, width)
    print(f"Input shape (NCHW4c): {x_data.shape}")
    print(f"Output shape (NCHW): {output.shape}")
    print(f"Expected shape: {expected_shape}")
    
    assert output.shape == expected_shape, f"Shape mismatch: {output.shape} != {expected_shape}"
    
    # Verify data correctness
    expected_output = np.transpose(x_data, (0, 1, 4, 2, 3))
    expected_output = expected_output.reshape(batch, channel_blocks * pack_size, height, width)
    tvm.testing.assert_allclose(output, expected_output, rtol=1e-5, atol=1e-5)
    
    print("✓ Test passed!")
    return True


def test_layout_transform_chain():
    """Test chained layout transformations (NCHW -> NHWC -> NCHW)"""
    print("\n" + "="*60)
    print("Test: Chained Layout Transform (NCHW -> NHWC -> NCHW)")
    print("="*60)
    
    # Define input shape
    batch, channel, height, width = 1, 16, 32, 32
    shape_nchw = (batch, channel, height, width)
    
    # Create relay function with chained transformations
    x = relay.var("x", shape=shape_nchw, dtype="float32")
    y = relay.layout_transform(x, "NCHW", "NHWC")
    y = relay.nn.relu(y)  # Add an operation in between
    y = relay.layout_transform(y, "NHWC", "NCHW")
    func = relay.Function([x], y)
    
    # Create test data
    x_data = np.random.uniform(size=shape_nchw).astype("float32")
    
    # Build and run
    target = "llvm"
    dev = tvm.cpu(0)
    
    mod = tvm.IRModule.from_expr(func)
    mod = transform.InferType()(mod)
    print("Input IR:")
    print(mod)
    
    graph, lib, params = relay.build(mod, target=target)
    m = graph_executor.create(graph, lib, device=dev)
    m.set_input("x", x_data)
    m.run()
    
    # Get output
    output = m.get_output(0).asnumpy()
    
    # After round-trip transformation, shape should be the same
    print(f"Input shape: {x_data.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == x_data.shape, f"Shape mismatch: {output.shape} != {x_data.shape}"
    
    # Output should equal relu(input) after round-trip
    expected_output = np.maximum(x_data, 0)
    tvm.testing.assert_allclose(output, expected_output, rtol=1e-5, atol=1e-5)
    
    print("✓ Test passed!")
    return True


def test_layout_transform_with_conv():
    """Test layout transform combined with conv2d"""
    print("\n" + "="*60)
    print("Test: Layout Transform with Conv2D")
    print("="*60)
    
    # Input in NHWC format
    batch, height, width, in_channels = 1, 32, 32, 3
    out_channels = 16
    kernel_size = 3
    
    # Create relay function
    x = relay.var("x", shape=(batch, height, width, in_channels), dtype="float32")
    weight = relay.var("weight", shape=(kernel_size, kernel_size, in_channels, out_channels), dtype="float32")
    
    # Transform NHWC -> NCHW for conv2d
    x_nchw = relay.layout_transform(x, "NHWC", "NCHW")
    weight_oihw = relay.layout_transform(weight, "HWIO", "OIHW")
    
    # Perform conv2d in NCHW layout
    y = relay.nn.conv2d(
        x_nchw,
        weight_oihw,
        channels=out_channels,
        kernel_size=(kernel_size, kernel_size),
        padding=(1, 1),
        data_layout="NCHW",
        kernel_layout="OIHW"
    )
    
    # Transform back to NHWC
    y_nhwc = relay.layout_transform(y, "NCHW", "NHWC")
    
    func = relay.Function([x, weight], y_nhwc)
    
    # Create test data
    x_data = np.random.uniform(size=(batch, height, width, in_channels)).astype("float32")
    weight_data = np.random.uniform(size=(kernel_size, kernel_size, in_channels, out_channels)).astype("float32")
    
    # Build and run
    target = "llvm"
    dev = tvm.cpu(0)
    
    mod = tvm.IRModule.from_expr(func)
    mod = transform.InferType()(mod)
    print("Input IR:")
    print(mod)
    
    params = {"weight": weight_data}
    graph, lib, params = relay.build(mod, target=target, params=params)
    m = graph_executor.create(graph, lib, device=dev)
    m.set_input(**params)
    m.set_input("x", x_data)
    m.run()
    
    # Get output
    output = m.get_output(0).asnumpy()
    
    # Output should be in NHWC format
    expected_shape = (batch, height, width, out_channels)
    print(f"Input shape (NHWC): {x_data.shape}")
    print(f"Output shape (NHWC): {output.shape}")
    print(f"Expected shape: {expected_shape}")
    
    assert output.shape == expected_shape, f"Shape mismatch: {output.shape} != {expected_shape}"
    
    print("✓ Test passed!")
    return True


def test_layout_transform_different_dtypes():
    """Test layout transform with different data types"""
    print("\n" + "="*60)
    print("Test: Layout Transform with Different Dtypes")
    print("="*60)
    
    shape_nchw = (1, 4, 8, 8)
    
    for dtype in ["float32", "float16", "int8", "int32"]:
        print(f"\nTesting dtype: {dtype}")
        
        x = relay.var("x", shape=shape_nchw, dtype=dtype)
        y = relay.layout_transform(x, "NCHW", "NHWC")
        func = relay.Function([x], y)
        
        # Create test data
        if "float" in dtype:
            x_data = np.random.uniform(size=shape_nchw).astype(dtype)
        else:
            x_data = np.random.randint(-10, 10, size=shape_nchw).astype(dtype)
        
        # Build and run
        target = "llvm"
        dev = tvm.cpu(0)
        
        mod = tvm.IRModule.from_expr(func)
        mod = transform.InferType()(mod)
        
        graph, lib, params = relay.build(mod, target=target)
        m = graph_executor.create(graph, lib, device=dev)
        m.set_input("x", x_data)
        m.run()
        
        output = m.get_output(0).asnumpy()
        
        expected_output = np.transpose(x_data, (0, 2, 3, 1))
        tvm.testing.assert_allclose(output, expected_output, rtol=1e-5, atol=1e-5)
        
        print(f"  ✓ {dtype} passed!")
    
    print("\n✓ All dtype tests passed!")
    return True


def run_all_tests():
    """Run all layout transform tests"""
    print("\n" + "="*70)
    print(" TVM Layout Transform Operator Test Suite")
    print("="*70)
    
    tests = [
        ("NCHW to NHWC", test_layout_transform_nchw_to_nhwc),
        ("NHWC to NCHW", test_layout_transform_nhwc_to_nchw),
        ("NCHW to NCHW4c", test_layout_transform_nchw_to_nchw4c),
        ("NCHW4c to NCHW", test_layout_transform_nchw4c_to_nchw),
        ("Chained Transform", test_layout_transform_chain),
        ("With Conv2D", test_layout_transform_with_conv),
        ("Different Dtypes", test_layout_transform_different_dtypes),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ Test '{test_name}' failed with error:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print(f" Test Results: {passed} passed, {failed} failed")
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
