#!/usr/bin/env python3
"""Basic bitunpack functionality test"""

import numpy as np
import tvm
from tvm import relay, te
from tvm.contrib import graph_executor

def test_bitunpack_uint32_bits1():
    """Test basic bitunpack with uint32 and bits=1"""
    print("\n" + "="*70)
    print("Test 1: Basic bitunpack (uint32, bits=1)")
    print("="*70)
    
    # Create original data
    original_np = np.array([[[[0, 1, 0, 1, 0, 1, 0, 1,  # 8 elements
                               1, 0, 1, 0, 1, 0, 1, 0,  # 8 elements
                               0, 0, 1, 1, 0, 0, 1, 1,  # 8 elements
                               1, 1, 0, 0, 1, 1, 0, 0]]]], dtype=np.uint8)  # 8 elements = 32 total
    
    print(f"Original shape: {original_np.shape}")
    print(f"Original data: {original_np[0, 0, 0, :8]}")
    
    # Pack
    original = relay.var("data", shape=(1, 1, 1, 32), dtype="uint8")
    packed = relay.nn.bitpack(original, bits=1, pack_axis=3, bit_axis=3,
                              pack_type="uint32", msb_first=False)
    
    # Build pack function
    pack_func = relay.Function([original], packed)
    target = "llvm"
    dev = tvm.device(target, 0)
    
    with tvm.transform.PassContext(opt_level=3):
        pack_lib = relay.build(pack_func, target=target)
    
    pack_module = graph_executor.GraphModule(pack_lib["default"](dev))
    pack_module.set_input("data", original_np)
    pack_module.run()
    packed_result = pack_module.get_output(0).numpy()
    
    print(f"Packed shape: {packed_result.shape}")
    print(f"Expected packed shape: (1, 1, 1, 1)")
    
    # Now unpack
    packed_var = relay.var("packed", shape=packed_result.shape, dtype="uint32")
    unpacked = relay.nn.bitunpack(packed_var, bits=1, pack_axis=3, bit_axis=3,
                                   pack_type="uint32", out_size=32,
                                   out_dtype="uint8", msb_first=False)
    
    # Build unpack function
    unpack_func = relay.Function([packed_var], unpacked)
    
    with tvm.transform.PassContext(opt_level=3):
        unpack_lib = relay.build(unpack_func, target=target)
    
    unpack_module = graph_executor.GraphModule(unpack_lib["default"](dev))
    unpack_module.set_input("packed", packed_result)
    unpack_module.run()
    unpacked_result = unpack_module.get_output(0).numpy()
    
    print(f"Unpacked shape: {unpacked_result.shape}")
    print(f"Unpacked data: {unpacked_result[0, 0, 0, :8]}")
    print(f"Expected:      {original_np[0, 0, 0, :8]}")
    
    # Verify
    if np.array_equal(original_np, unpacked_result):
        print("✓ PASS: Round-trip successful!")
        return True
    else:
        print("✗ FAIL: Data mismatch!")
        print(f"Differences: {np.sum(original_np != unpacked_result)} elements")
        return False


def test_bitunpack_uint256_bits4():
    """Test bitunpack with uint256 and bits=4 (with padding)"""
    print("\n" + "="*70)
    print("Test 2: bitunpack with uint256, bits=4, and padding")
    print("="*70)
    
    # Original: 100 channels (will be padded to 256)
    original_np = np.random.randint(0, 16, size=(1, 100, 5, 5), dtype=np.uint8)
    
    print(f"Original shape: {original_np.shape}")
    print(f"Original sample: {original_np[0, :5, 0, 0]}")
    
    # Pack
    original = relay.var("data", shape=(1, 100, 5, 5), dtype="uint8")
    packed = relay.nn.bitpack(original, bits=4, pack_axis=1, bit_axis=4,
                              pack_type="uint256", msb_first=False)
    
    # Build pack function
    pack_func = relay.Function([original], packed)
    target = "llvm"
    dev = tvm.device(target, 0)
    
    with tvm.transform.PassContext(opt_level=3):
        pack_lib = relay.build(pack_func, target=target)
    
    pack_module = graph_executor.GraphModule(pack_lib["default"](dev))
    pack_module.set_input("data", original_np)
    pack_module.run()
    packed_result = pack_module.get_output(0).numpy()
    
    print(f"Packed shape: {packed_result.shape}")
    print(f"Expected packed shape: (1, 1, 5, 5, 4, 8)")
    
    # Unpack with original size to remove padding
    packed_var = relay.var("packed", shape=packed_result.shape, dtype="uint32")
    unpacked = relay.nn.bitunpack(packed_var, bits=4, pack_axis=1, bit_axis=4,
                                   pack_type="uint256", out_size=100,  # Original size!
                                   out_dtype="uint8", msb_first=False)
    
    # Build unpack function
    unpack_func = relay.Function([packed_var], unpacked)
    
    with tvm.transform.PassContext(opt_level=3):
        unpack_lib = relay.build(unpack_func, target=target)
    
    unpack_module = graph_executor.GraphModule(unpack_lib["default"](dev))
    unpack_module.set_input("packed", packed_result)
    unpack_module.run()
    unpacked_result = unpack_module.get_output(0).numpy()
    
    print(f"Unpacked shape: {unpacked_result.shape}")
    print(f"Unpacked sample: {unpacked_result[0, :5, 0, 0]}")
    print(f"Expected shape:  (1, 100, 5, 5)")
    
    # Verify
    if unpacked_result.shape == (1, 100, 5, 5):
        print("✓ Shape correct!")
    else:
        print(f"✗ Shape mismatch: {unpacked_result.shape}")
        return False
    
    if np.array_equal(original_np, unpacked_result):
        print("✓ PASS: Round-trip with padding successful!")
        return True
    else:
        print("✗ FAIL: Data mismatch!")
        mismatches = np.sum(original_np != unpacked_result)
        print(f"Differences: {mismatches} elements out of {original_np.size}")
        
        # Show first few mismatches
        if mismatches > 0:
            mismatch_indices = np.where(original_np != unpacked_result)
            for i in range(min(5, mismatches)):
                idx = tuple(m[i] for m in mismatch_indices)
                print(f"  Position {idx}: expected {original_np[idx]}, got {unpacked_result[idx]}")
        
        return False


def test_bitunpack_various_shapes():
    """Test bitunpack with various shapes and configurations"""
    print("\n" + "="*70)
    print("Test 3: bitunpack with various shapes")
    print("="*70)
    
    test_configs = [
        # (shape, bits, pack_axis, bit_axis, pack_type, description)
        ((64,), 2, 0, 0, "uint32", "1D: 64 elements, bits=2"),
        ((128,), 4, 0, 0, "uint64", "1D: 128 elements, uint64"),
        ((1, 50), 8, 1, 1, "uint256", "2D: 50 elements with padding"),
        ((1, 200), 4, 1, 1, "uint256", "2D: 200 elements, no padding"),
        ((1, 300, 7), 2, 1, 2, "uint128", "3D: 300 elements"),
        ((2, 80, 5, 5), 4, 1, 3, "uint256", "4D: 80 channels"),
        ((1, 150, 3, 3), 4, 1, 3, "uint256", "4D: 150 channels with padding"),
        ((4, 32, 16, 16), 1, 1, 1, "uint32", "4D: 32 channels, bits=1"),
    ]
    
    target = "llvm"
    dev = tvm.device(target, 0)
    all_passed = True
    
    for shape, bits, pack_axis, bit_axis, pack_type, description in test_configs:
        print(f"\n  Testing: {description}")
        print(f"    Shape: {shape}, bits={bits}, pack_type={pack_type}")
        
        try:
            # Generate random data
            max_val = (1 << bits) - 1
            original_np = np.random.randint(0, max_val + 1, size=shape, dtype=np.uint8)
            
            # Pack
            original = relay.var("data", shape=shape, dtype="uint8")
            packed = relay.nn.bitpack(original, bits=bits, pack_axis=pack_axis, 
                                     bit_axis=bit_axis, pack_type=pack_type, 
                                     msb_first=False)
            
            pack_func = relay.Function([original], packed)
            with tvm.transform.PassContext(opt_level=3):
                pack_lib = relay.build(pack_func, target=target)
            
            pack_module = graph_executor.GraphModule(pack_lib["default"](dev))
            pack_module.set_input("data", original_np)
            pack_module.run()
            packed_result = pack_module.get_output(0).numpy()
            
            print(f"    Packed shape: {packed_result.shape}")
            
            # Unpack
            packed_var = relay.var("packed", shape=packed_result.shape, dtype="uint32")
            unpacked = relay.nn.bitunpack(packed_var, bits=bits, pack_axis=pack_axis,
                                         bit_axis=bit_axis, pack_type=pack_type,
                                         out_size=shape[pack_axis], out_dtype="uint8",
                                         msb_first=False)
            
            unpack_func = relay.Function([packed_var], unpacked)
            with tvm.transform.PassContext(opt_level=3):
                unpack_lib = relay.build(unpack_func, target=target)
            
            unpack_module = graph_executor.GraphModule(unpack_lib["default"](dev))
            unpack_module.set_input("packed", packed_result)
            unpack_module.run()
            unpacked_result = unpack_module.get_output(0).numpy()
            
            # Verify
            if np.array_equal(original_np, unpacked_result):
                print(f"    ✓ PASS")
            else:
                print(f"    ✗ FAIL: Data mismatch")
                mismatches = np.sum(original_np != unpacked_result)
                print(f"      Differences: {mismatches}/{original_np.size} elements")
                all_passed = False
                
        except Exception as e:
            print(f"    ✗ FAIL: Exception - {e}")
            all_passed = False
    
    return all_passed


def test_bitunpack_edge_cases():
    """Test bitunpack edge cases"""
    print("\n" + "="*70)
    print("Test 4: bitunpack edge cases")
    print("="*70)
    
    target = "llvm"
    dev = tvm.device(target, 0)
    all_passed = True
    
    # Test 1: Exact multiple (no padding)
    print("\n  Case 1: Exact multiple - 256 elements, uint256, bits=4")
    try:
        original_np = np.random.randint(0, 16, size=(1, 256, 3, 3), dtype=np.uint8)
        original = relay.var("data", shape=(1, 256, 3, 3), dtype="uint8")
        packed = relay.nn.bitpack(original, bits=4, pack_axis=1, bit_axis=3,
                                  pack_type="uint256", msb_first=False)
        
        pack_func = relay.Function([original], packed)
        with tvm.transform.PassContext(opt_level=3):
            pack_lib = relay.build(pack_func, target=target)
        
        pack_module = graph_executor.GraphModule(pack_lib["default"](dev))
        pack_module.set_input("data", original_np)
        pack_module.run()
        packed_result = pack_module.get_output(0).numpy()
        
        print(f"    Packed shape: {packed_result.shape}")
        
        packed_var = relay.var("packed", shape=packed_result.shape, dtype="uint32")
        unpacked = relay.nn.bitunpack(packed_var, bits=4, pack_axis=1, bit_axis=3,
                                     pack_type="uint256", out_size=256,
                                     out_dtype="uint8", msb_first=False)
        
        unpack_func = relay.Function([packed_var], unpacked)
        with tvm.transform.PassContext(opt_level=3):
            unpack_lib = relay.build(unpack_func, target=target)
        
        unpack_module = graph_executor.GraphModule(unpack_lib["default"](dev))
        unpack_module.set_input("packed", packed_result)
        unpack_module.run()
        unpacked_result = unpack_module.get_output(0).numpy()
        
        if np.array_equal(original_np, unpacked_result):
            print(f"    ✓ PASS")
        else:
            print(f"    ✗ FAIL")
            all_passed = False
    except Exception as e:
        print(f"    ✗ FAIL: {e}")
        all_passed = False
    
    # Test 2: Small size (< pack_width)
    print("\n  Case 2: Small size - 10 elements, uint256, bits=4")
    try:
        original_np = np.random.randint(0, 16, size=(1, 10), dtype=np.uint8)
        original = relay.var("data", shape=(1, 10), dtype="uint8")
        packed = relay.nn.bitpack(original, bits=4, pack_axis=1, bit_axis=1,
                                  pack_type="uint256", msb_first=False)
        
        pack_func = relay.Function([original], packed)
        with tvm.transform.PassContext(opt_level=3):
            pack_lib = relay.build(pack_func, target=target)
        
        pack_module = graph_executor.GraphModule(pack_lib["default"](dev))
        pack_module.set_input("data", original_np)
        pack_module.run()
        packed_result = pack_module.get_output(0).numpy()
        
        print(f"    Packed shape: {packed_result.shape}")
        
        packed_var = relay.var("packed", shape=packed_result.shape, dtype="uint32")
        unpacked = relay.nn.bitunpack(packed_var, bits=4, pack_axis=1, bit_axis=1,
                                     pack_type="uint256", out_size=10,
                                     out_dtype="uint8", msb_first=False)
        
        unpack_func = relay.Function([packed_var], unpacked)
        with tvm.transform.PassContext(opt_level=3):
            unpack_lib = relay.build(unpack_func, target=target)
        
        unpack_module = graph_executor.GraphModule(unpack_lib["default"](dev))
        unpack_module.set_input("packed", packed_result)
        unpack_module.run()
        unpacked_result = unpack_module.get_output(0).numpy()
        
        if np.array_equal(original_np, unpacked_result):
            print(f"    ✓ PASS")
        else:
            print(f"    ✗ FAIL")
            all_passed = False
    except Exception as e:
        print(f"    ✗ FAIL: {e}")
        all_passed = False
    
    # Test 3: Different bit widths
    print("\n  Case 3: Various bit widths (1, 2, 4, 8 bits)")
    for bits in [1, 2, 4, 8]:
        try:
            max_val = (1 << bits) - 1
            original_np = np.random.randint(0, max_val + 1, size=(1, 100), dtype=np.uint8)
            original = relay.var("data", shape=(1, 100), dtype="uint8")
            packed = relay.nn.bitpack(original, bits=bits, pack_axis=1, bit_axis=1,
                                      pack_type="uint128", msb_first=False)
            
            pack_func = relay.Function([original], packed)
            with tvm.transform.PassContext(opt_level=3):
                pack_lib = relay.build(pack_func, target=target)
            
            pack_module = graph_executor.GraphModule(pack_lib["default"](dev))
            pack_module.set_input("data", original_np)
            pack_module.run()
            packed_result = pack_module.get_output(0).numpy()
            
            packed_var = relay.var("packed", shape=packed_result.shape, dtype="uint32")
            unpacked = relay.nn.bitunpack(packed_var, bits=bits, pack_axis=1, bit_axis=1,
                                         pack_type="uint128", out_size=100,
                                         out_dtype="uint8", msb_first=False)
            
            unpack_func = relay.Function([packed_var], unpacked)
            with tvm.transform.PassContext(opt_level=3):
                unpack_lib = relay.build(unpack_func, target=target)
            
            unpack_module = graph_executor.GraphModule(unpack_lib["default"](dev))
            unpack_module.set_input("packed", packed_result)
            unpack_module.run()
            unpacked_result = unpack_module.get_output(0).numpy()
            
            if np.array_equal(original_np, unpacked_result):
                print(f"    bits={bits}: ✓ PASS")
            else:
                print(f"    bits={bits}: ✗ FAIL")
                all_passed = False
        except Exception as e:
            print(f"    bits={bits}: ✗ FAIL: {e}")
            all_passed = False
    
    return all_passed


if __name__ == "__main__":
    print("\n" + "="*70)
    print("BITUNPACK COMPREHENSIVE FUNCTIONALITY TESTS")
    print("="*70)
    
    results = []
    
    try:
        results.append(("uint32 bits=1", test_bitunpack_uint32_bits1()))
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("uint32 bits=1", False))
    
    try:
        results.append(("uint256 bits=4 with padding", test_bitunpack_uint256_bits4()))
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("uint256 bits=4 with padding", False))
    
    try:
        results.append(("Various shapes", test_bitunpack_various_shapes()))
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Various shapes", False))
    
    try:
        results.append(("Edge cases", test_bitunpack_edge_cases()))
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Edge cases", False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED!")
    else:
        print(f"\n✗ {total - passed} TEST(S) FAILED")
        exit(1)
