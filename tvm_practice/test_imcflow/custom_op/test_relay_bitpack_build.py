"""
Test Relay-level bitpack operator with relay.build()
Tests:
1. MSB-first vs LSB-first packing
2. uint64 chunking (2x uint32)
3. uint128 chunking (4x uint32)
4. Build and execute the graph
"""

import numpy as np
import tvm
from tvm import relay
from tvm.relay import testing
import tvm.testing

def test_relay_bitpack_msb_lsb():
    """Test MSB-first vs LSB-first at Relay level with relay.build()"""
    print("\n" + "="*70)
    print("Test 1: Relay MSB-first vs LSB-first with relay.build()")
    print("="*70)
    
    # Input: [0, 1, 2, 3, 4, 5, 6, 7] -> LSBs are [0,1,0,1,0,1,0,1]
    data_np = np.array([[[0, 1, 2, 3, 4, 5, 6, 7]]], dtype=np.uint8)
    print(f"Input data: {data_np[0, 0, :]}")
    print(f"Input LSBs: {[int(x & 1) for x in data_np[0, 0, :]]}")
    
    # Test MSB-first
    print("\n--- MSB-first (default) ---")
    data_var = relay.var("data", shape=(1, 1, 8), dtype="uint8")
    out_msb = relay.nn.bitpack(data_var, bits=1, pack_axis=2, bit_axis=2, 
                                 pack_type="uint8", msb_first=True)
    func_msb = relay.Function([data_var], out_msb)
    print(f"Relay graph created: {func_msb}")
    
    # Build and execute
    target = "llvm"
    dev = tvm.device(target, 0)
    
    with tvm.transform.PassContext(opt_level=3):
        lib_msb = relay.build(func_msb, target=target)
    
    module_msb = tvm.contrib.graph_executor.GraphModule(lib_msb["default"](dev))
    module_msb.set_input("data", data_np)
    module_msb.run()
    result_msb = module_msb.get_output(0).numpy()
    
    print(f"Output shape: {result_msb.shape}")
    # Handle both (1,1,1,1) and (1,1,1) shapes
    msb_val = result_msb.flatten()[0]
    print(f"Output value: 0x{int(msb_val):02x} = {format(int(msb_val), '08b')}b")
    
    # Test LSB-first
    print("\n--- LSB-first ---")
    out_lsb = relay.nn.bitpack(data_var, bits=1, pack_axis=2, bit_axis=2, 
                                 pack_type="uint8", msb_first=False)
    func_lsb = relay.Function([data_var], out_lsb)
    print(f"Relay graph created: {func_lsb}")
    
    with tvm.transform.PassContext(opt_level=3):
        lib_lsb = relay.build(func_lsb, target=target)
    
    module_lsb = tvm.contrib.graph_executor.GraphModule(lib_lsb["default"](dev))
    module_lsb.set_input("data", data_np)
    module_lsb.run()
    result_lsb = module_lsb.get_output(0).numpy()
    
    print(f"Output shape: {result_lsb.shape}")
    # Handle both (1,1,1,1) and (1,1,1) shapes
    lsb_val = result_lsb.flatten()[0]
    print(f"Output value: 0x{int(lsb_val):02x} = {format(int(lsb_val), '08b')}b")
    
    # Verify results
    print("\n--- Verification ---")
    expected_msb = 0x55  # 01010101b
    expected_lsb = 0xaa  # 10101010b
    
    msb_pass = int(msb_val) == expected_msb
    lsb_pass = int(lsb_val) == expected_lsb
    
    print(f"Expected MSB-first: 0x{expected_msb:02x}")
    print(f"Expected LSB-first: 0x{expected_lsb:02x}")
    print(f"✓ MSB-first {'PASS' if msb_pass else 'FAIL'}")
    print(f"✓ LSB-first {'PASS' if lsb_pass else 'FAIL'}")
    
    assert msb_pass, f"MSB-first failed: got 0x{int(msb_val):02x}, expected 0x{expected_msb:02x}"
    assert lsb_pass, f"LSB-first failed: got 0x{int(lsb_val):02x}, expected 0x{expected_lsb:02x}"
    
    return True


def test_relay_bitpack_uint64_chunking():
    """Test uint64 chunking (2x uint32) at Relay level"""
    print("\n" + "="*70)
    print("Test 2: Relay uint64 → 2x uint32 chunks with relay.build()")
    print("="*70)
    
    # Input shape: [1, 64, 2, 2] - 64 bits along pack_axis
    data_np = np.random.randint(0, 256, size=(1, 64, 2, 2), dtype=np.uint8)
    print(f"Input shape:  {data_np.shape}")
    
    # Create Relay graph
    data_var = relay.var("data", shape=(1, 64, 2, 2), dtype="uint8")
    out = relay.nn.bitpack(data_var, bits=1, pack_axis=1, bit_axis=1, 
                            pack_type="uint64", msb_first=True)
    func = relay.Function([data_var], out)
    
    # Build and execute
    target = "llvm"
    dev = tvm.device(target, 0)
    
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target=target)
    
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    module.set_input("data", data_np)
    module.run()
    result = module.get_output(0).numpy()
    
    print(f"Output shape: {result.shape}")
    print(f"Expected:     (1, 1, 1, 2, 2, 2)  <- (1,1,1,2,2) + chunk dim=2")
    
    # Verify shape: should be [1, 1, 1, 2, 2, 2]
    # Original: [1, 64, 2, 2]
    # After bitpack: 64 bits packed into uint64 = 64/64 = 1 element
    # uint64 needs 2 chunks of uint32, so add dimension of size 2
    # Final: [1, 1, 1, 2, 2, 2]
    expected_shape = (1, 1, 1, 2, 2, 2)
    
    shape_match = result.shape == expected_shape
    print(f"{'✓' if shape_match else '✗'} Shape {'matches' if shape_match else 'does not match'}")
    
    assert shape_match, f"Shape mismatch: got {result.shape}, expected {expected_shape}"
    
    # Verify chunk dimension is last
    assert result.shape[-1] == 2, f"Chunk dimension should be 2, got {result.shape[-1]}"
    
    print("✓ uint64 chunking works correctly at Relay level")
    return True


def test_relay_bitpack_uint128_chunking():
    """Test uint128 chunking (4x uint32) at Relay level"""
    print("\n" + "="*70)
    print("Test 3: Relay uint128 → 4x uint32 chunks with relay.build()")
    print("="*70)
    
    # Input shape: [1, 128, 2, 2] - 128 bits along pack_axis
    data_np = np.random.randint(0, 256, size=(1, 128, 2, 2), dtype=np.uint8)
    print(f"Input shape:  {data_np.shape}")
    
    # Create Relay graph
    data_var = relay.var("data", shape=(1, 128, 2, 2), dtype="uint8")
    out = relay.nn.bitpack(data_var, bits=1, pack_axis=1, bit_axis=1, 
                            pack_type="uint128", msb_first=True)
    func = relay.Function([data_var], out)
    
    # Build and execute
    target = "llvm"
    dev = tvm.device(target, 0)
    
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target=target)
    
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    module.set_input("data", data_np)
    module.run()
    result = module.get_output(0).numpy()
    
    print(f"Output shape: {result.shape}")
    print(f"Expected:     (1, 1, 1, 2, 2, 4)  <- (1,1,1,2,2) + chunk dim=4")
    
    # Verify shape: should be [1, 1, 1, 2, 2, 4]
    # Original: [1, 128, 2, 2]
    # After bitpack: 128 bits packed into uint128 = 128/128 = 1 element
    # uint128 needs 4 chunks of uint32, so add dimension of size 4
    # Final: [1, 1, 1, 2, 2, 4]
    expected_shape = (1, 1, 1, 2, 2, 4)
    
    shape_match = result.shape == expected_shape
    print(f"{'✓' if shape_match else '✗'} Shape {'matches' if shape_match else 'does not match'}")
    
    assert shape_match, f"Shape mismatch: got {result.shape}, expected {expected_shape}"
    
    # Verify chunk dimension is last
    assert result.shape[-1] == 4, f"Chunk dimension should be 4, got {result.shape[-1]}"
    
    print("✓ uint128 chunking works correctly at Relay level")
    return True


def test_relay_bitpack_uint32_no_chunking():
    """Test uint32 has NO chunking (backward compatibility)"""
    print("\n" + "="*70)
    print("Test 4: Relay uint32 backward compatibility (no chunking)")
    print("="*70)
    
    # Input shape: [1, 32, 2, 2] - 32 bits along pack_axis
    data_np = np.random.randint(0, 256, size=(1, 32, 2, 2), dtype=np.uint8)
    print(f"Input shape:  {data_np.shape}")
    
    # Create Relay graph
    data_var = relay.var("data", shape=(1, 32, 2, 2), dtype="uint8")
    out = relay.nn.bitpack(data_var, bits=1, pack_axis=1, bit_axis=1, 
                            pack_type="uint32", msb_first=True)
    func = relay.Function([data_var], out)
    
    # Build and execute
    target = "llvm"
    dev = tvm.device(target, 0)
    
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target=target)
    
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    module.set_input("data", data_np)
    module.run()
    result = module.get_output(0).numpy()
    
    print(f"Output shape: {result.shape}")
    print(f"Expected:     (1, 1, 1, 2, 2)  <- NO extra chunk dimension")
    
    # Verify shape: should be [1, 1, 1, 2, 2] (NO chunk dimension)
    # Original: [1, 32, 2, 2]
    # After bitpack: 32 bits packed into uint32 = 32/32 = 1 element
    # uint32 needs NO chunking, so NO extra dimension
    # Final: [1, 1, 1, 2, 2]
    expected_shape = (1, 1, 1, 2, 2)
    
    shape_match = result.shape == expected_shape
    print(f"{'✓' if shape_match else '✗'} Shape {'matches' if shape_match else 'does not match'}")
    
    assert shape_match, f"Shape mismatch: got {result.shape}, expected {expected_shape}"
    
    print("✓ uint32 backward compatibility confirmed (no chunking)")
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("RELAY BITPACK OPERATOR TEST WITH relay.build()")
    print("="*70)
    
    try:
        # Test 1: MSB vs LSB
        test_relay_bitpack_msb_lsb()
        
        # Test 2: uint64 chunking
        test_relay_bitpack_uint64_chunking()
        
        # Test 3: uint128 chunking
        test_relay_bitpack_uint128_chunking()
        
        # Test 4: uint32 no chunking
        test_relay_bitpack_uint32_no_chunking()
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print("MSB vs LSB                ✓ PASS")
        print("uint64 chunking           ✓ PASS")
        print("uint128 chunking          ✓ PASS")
        print("uint32 backward compat    ✓ PASS")
        print("="*70)
        print("✓ ALL RELAY TESTS PASSED!")
        print("="*70)
        
    except Exception as e:
        print("\n" + "="*70)
        print("✗ TEST FAILED")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise
