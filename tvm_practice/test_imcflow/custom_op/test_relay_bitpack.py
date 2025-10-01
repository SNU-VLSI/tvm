#!/usr/bin/env python3
"""Test Relay bitpack operator with msb_first parameter"""
import numpy as np
import tvm
from tvm import relay
from tvm.relay import testing
import sys

def test_relay_bitpack_msb_lsb():
    """Test relay.nn.bitpack with MSB vs LSB ordering"""
    print("="*70)
    print("Test: Relay bitpack with MSB-first vs LSB-first")
    print("="*70)
    
    # Create simple test data: [0, 1, 2, 3, 4, 5, 6, 7]
    # LSBs: [0, 1, 0, 1, 0, 1, 0, 1]
    N, C, H, W = 1, 8, 1, 1
    data_np = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32).reshape(N, C, H, W)
    
    # Create Relay function with MSB-first
    data = relay.var("data", relay.TensorType((N, C, H, W), "int32"))
    out_msb = relay.nn.bitpack(data, bits=8, pack_axis=1, bit_axis=1, 
                                pack_type="uint8", name="bitpack_msb", msb_first=True)
    func_msb = relay.Function([data], out_msb)
    
    # Create Relay function with LSB-first
    out_lsb = relay.nn.bitpack(data, bits=8, pack_axis=1, bit_axis=1, 
                                pack_type="uint8", name="bitpack_lsb", msb_first=False)
    func_lsb = relay.Function([data], out_lsb)
    
    print(f"Input shape: {data_np.shape}")
    print(f"Input data: {data_np.flatten()}")
    print(f"Input LSBs: {[x & 1 for x in data_np.flatten()]}")
    
    # Build and run MSB-first
    target = "llvm"
    dev = tvm.cpu()
    
    with tvm.transform.PassContext(opt_level=3):
        lib_msb = relay.build(func_msb, target=target)
        module_msb = tvm.contrib.graph_executor.GraphModule(lib_msb["default"](dev))
    
    module_msb.set_input("data", data_np)
    module_msb.run()
    output_msb = module_msb.get_output(0).numpy()
    
    # Build and run LSB-first
    with tvm.transform.PassContext(opt_level=3):
        lib_lsb = relay.build(func_lsb, target=target)
        module_lsb = tvm.contrib.graph_executor.GraphModule(lib_lsb["default"](dev))
    
    module_lsb.set_input("data", data_np)
    module_lsb.run()
    output_lsb = module_lsb.get_output(0).numpy()
    
    print(f"\nOutput shape: {output_msb.shape}")
    print(f"MSB-first output[0, 0, 0, 0, 0]: {output_msb[0, 0, 0, 0, 0]:#04x} = {output_msb[0, 0, 0, 0, 0]:08b}b")
    print(f"LSB-first output[0, 0, 0, 0, 0]: {output_lsb[0, 0, 0, 0, 0]:#04x} = {output_lsb[0, 0, 0, 0, 0]:08b}b")
    
    # Expected values:
    # MSB-first: data[7]→bit7, ..., data[0]→bit0 = 0b01010101 = 0x55
    # LSB-first: data[0]→bit0, ..., data[7]→bit7 = 0b10101010 = 0xaa
    expected_msb = 0x55
    expected_lsb = 0xaa
    
    print(f"\nExpected MSB-first: {expected_msb:#04x}")
    print(f"Expected LSB-first: {expected_lsb:#04x}")
    
    if output_msb[0, 0, 0, 0, 0] == expected_msb:
        print("✓ MSB-first PASS")
    else:
        print(f"✗ MSB-first FAIL (got {output_msb[0, 0, 0, 0, 0]:#04x})")
        return False
    
    if output_lsb[0, 0, 0, 0, 0] == expected_lsb:
        print("✓ LSB-first PASS")
    else:
        print(f"✗ LSB-first FAIL (got {output_lsb[0, 0, 0, 0, 0]:#04x})")
        return False
    
    print("\n✓ ALL TESTS PASSED!")
    return True

def test_relay_bitpack_uint64():
    """Test relay.nn.bitpack with uint64 (multi-chunk)"""
    print("\n" + "="*70)
    print("Test: Relay bitpack with uint64 (2x uint32 chunks)")
    print("="*70)
    
    N, C, H, W = 1, 64, 1, 1
    data_np = np.arange(C, dtype=np.int32).reshape(N, C, H, W)
    
    # Create Relay function
    data = relay.var("data", relay.TensorType((N, C, H, W), "int32"))
    out = relay.nn.bitpack(data, bits=8, pack_axis=1, bit_axis=1, 
                           pack_type="uint64", name="bitpack_uint64", msb_first=True)
    func = relay.Function([data], out)
    
    # Build and run
    target = "llvm"
    dev = tvm.cpu()
    
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target=target)
        module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    
    module.set_input("data", data_np)
    module.run()
    output = module.get_output(0).numpy()
    
    print(f"Input shape: {data_np.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: [1, 1, 8, 1, 1, 2] (2 chunks for uint64)")
    
    # Check shape includes chunk dimension
    if output.shape[-1] == 2:
        print("✓ Chunk dimension added correctly")
        print("\n✓ TEST PASSED!")
        return True
    else:
        print(f"✗ Expected chunk dimension=2, got shape {output.shape}")
        return False

if __name__ == "__main__":
    print("Testing Relay bitpack operator with msb_first parameter\n")
    
    results = []
    
    try:
        results.append(("MSB vs LSB", test_relay_bitpack_msb_lsb()))
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("MSB vs LSB", False))
    
    try:
        results.append(("uint64 chunks", test_relay_bitpack_uint64()))
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("uint64 chunks", False))
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:25s} {status}")
    
    all_passed = all(r[1] for r in results)
    print("="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*70)
