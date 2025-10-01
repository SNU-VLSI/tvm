#!/usr/bin/env python3
"""Simple test for TE-level bitpack with msb_first parameter"""
import numpy as np
import tvm
from tvm import te
import sys
sys.path.insert(0, '/root/project/tvm/python')
from tvm.topi.nn.bitserial_util import bitpack

def test_msb_lsb_comparison():
    """Test MSB-first vs LSB-first bit packing"""
    print("="*70)
    print("Test: MSB-first vs LSB-first bit packing (TE level)")
    print("="*70)
    
    # Create simple test data: [0, 1, 2, 3, 4, 5, 6, 7]
    # LSBs: [0, 1, 0, 1, 0, 1, 0, 1]
    N, C, H, W = 1, 8, 1, 1
    data_np = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32).reshape(N, C, H, W)
    
    # Test MSB-first
    data_msb = te.placeholder((N, C, H, W), name='data', dtype='int32')
    output_msb = bitpack(data_msb, bits=8, pack_axis=1, bit_axis=1, 
                         pack_type='uint8', name='bitpack_msb', msb_first=True)
    
    s_msb = te.create_schedule(output_msb.op)
    func_msb = tvm.build(s_msb, [data_msb, output_msb], target='llvm')
    
    dev = tvm.cpu()
    data_tvm = tvm.nd.array(data_np, dev)
    output_shape_msb = tuple(int(x) for x in output_msb.shape)
    output_msb_tvm = tvm.nd.array(np.zeros(output_shape_msb, dtype=np.uint8), dev)
    
    func_msb(data_tvm, output_msb_tvm)
    result_msb = output_msb_tvm.numpy()
    
    # Test LSB-first
    data_lsb = te.placeholder((N, C, H, W), name='data', dtype='int32')
    output_lsb = bitpack(data_lsb, bits=8, pack_axis=1, bit_axis=1, 
                         pack_type='uint8', name='bitpack_lsb', msb_first=False)
    
    s_lsb = te.create_schedule(output_lsb.op)
    func_lsb = tvm.build(s_lsb, [data_lsb, output_lsb], target='llvm')
    
    output_shape_lsb = tuple(int(x) for x in output_lsb.shape)
    output_lsb_tvm = tvm.nd.array(np.zeros(output_shape_lsb, dtype=np.uint8), dev)
    
    func_lsb(data_tvm, output_lsb_tvm)
    result_lsb = output_lsb_tvm.numpy()
    
    print(f"Input data: {data_np.flatten()}")
    print(f"Input LSBs: {[x & 1 for x in data_np.flatten()]}")
    print(f"\nMSB-first output[0, 0, 0, 0, 0]: {result_msb[0, 0, 0, 0, 0]:#04x} = {result_msb[0, 0, 0, 0, 0]:08b}b")
    print(f"LSB-first output[0, 0, 0, 0, 0]: {result_lsb[0, 0, 0, 0, 0]:#04x} = {result_lsb[0, 0, 0, 0, 0]:08b}b")
    
    # Expected values:
    # MSB-first: data[7]→bit7, ..., data[0]→bit0 = 0b01010101 = 0x55
    # LSB-first: data[0]→bit0, ..., data[7]→bit7 = 0b10101010 = 0xaa
    expected_msb = 0x55
    expected_lsb = 0xaa
    
    print(f"\nExpected MSB-first: {expected_msb:#04x}")
    print(f"Expected LSB-first: {expected_lsb:#04x}")
    
    success = True
    if result_msb[0, 0, 0, 0, 0] == expected_msb:
        print("✓ MSB-first PASS")
    else:
        print(f"✗ MSB-first FAIL (got {result_msb[0, 0, 0, 0, 0]:#04x})")
        success = False
    
    if result_lsb[0, 0, 0, 0, 0] == expected_lsb:
        print("✓ LSB-first PASS")
    else:
        print(f"✗ LSB-first FAIL (got {result_lsb[0, 0, 0, 0, 0]:#04x})")
        success = False
    
    return success

def test_uint32_backward_compat():
    """Test uint32 backward compatibility"""
    print("\n" + "="*70)
    print("Test: uint32 backward compatibility")
    print("="*70)
    
    N, C, H, W = 1, 32, 2, 2
    data = te.placeholder((N, C, H, W), name='data', dtype='int32')
    
    output = bitpack(data, bits=8, pack_axis=1, bit_axis=1, pack_type='uint32')
    
    print(f"Input shape:  {data.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected:     [1, 1, 8, 2, 2] (no chunk dimension)")
    
    # Shape should NOT have extra dimension
    if len(output.shape) == 5:
        print("✓ No chunk dimension (correct)")
        return True
    else:
        print(f"✗ Expected 5D, got {len(output.shape)}D")
        return False

def test_uint64_chunking():
    """Test uint64 chunking"""
    print("\n" + "="*70)
    print("Test: uint64 → 2x uint32 chunks")
    print("="*70)
    
    N, C, H, W = 1, 64, 2, 2
    data = te.placeholder((N, C, H, W), name='data', dtype='int32')
    
    output = bitpack(data, bits=8, pack_axis=1, bit_axis=1, pack_type='uint64')
    
    print(f"Input shape:  {data.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected:     [1, 1, 8, 2, 2, 2] (last dim=2 for uint32 chunks)")
    
    # Shape should have chunk dimension
    if len(output.shape) == 6 and int(output.shape[-1]) == 2:
        print("✓ Chunk dimension added correctly")
        return True
    else:
        print(f"✗ Expected shape[...., 2], got {output.shape}")
        return False

if __name__ == "__main__":
    print("Testing TE-level bitpack with msb_first parameter\n")
    
    results = []
    
    try:
        results.append(("MSB vs LSB", test_msb_lsb_comparison()))
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("MSB vs LSB", False))
    
    try:
        results.append(("uint32 backward compat", test_uint32_backward_compat()))
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("uint32 backward compat", False))
    
    try:
        results.append(("uint64 chunking", test_uint64_chunking()))
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("uint64 chunking", False))
    
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
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        sys.exit(1)
    print("="*70)
