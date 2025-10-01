#!/usr/bin/env python3
"""Test MSB-first vs LSB-first bit packing"""
import numpy as np
import tvm
from tvm import te
import sys
sys.path.insert(0, '/root/project/tvm/python')
from tvm.topi.nn.bitserial_util import bitpack

def test_msb_vs_lsb():
    """Compare MSB-first and LSB-first packing"""
    print("="*70)
    print("Test: MSB-first vs LSB-first bit packing")
    print("="*70)
    
    N, C, H, W = 1, 8, 1, 1
    data = te.placeholder((N, C, H, W), name='data', dtype='int32')
    
    # Create both versions
    output_msb = bitpack(data, bits=8, pack_axis=1, bit_axis=1, pack_type='uint8', name='MSB', msb_first=True)
    output_lsb = bitpack(data, bits=8, pack_axis=1, bit_axis=1, pack_type='uint8', name='LSB', msb_first=False)
    
    print(f"Input shape: {data.shape}")
    print(f"MSB output shape: {output_msb.shape}")
    print(f"LSB output shape: {output_lsb.shape}")
    
    # Build
    s_msb = te.create_schedule(output_msb.op)
    func_msb = tvm.build(s_msb, [data, output_msb], target='llvm', name='bitpack_msb')
    
    s_lsb = te.create_schedule(output_lsb.op)
    func_lsb = tvm.build(s_lsb, [data, output_lsb], target='llvm', name='bitpack_lsb')
    
    # Test with sequential data: [0, 1, 2, 3, 4, 5, 6, 7]
    data_np = np.arange(8, dtype=np.int32).reshape(N, C, H, W)
    
    print(f"\nInput data: {data_np[0, :, 0, 0]}")
    print(f"Binary representations:")
    for i in range(8):
        print(f"  data[{i}] = {data_np[0, i, 0, 0]:3d} = {data_np[0, i, 0, 0]:08b}")
    
    dev = tvm.cpu()
    data_tvm = tvm.nd.array(data_np, dev)
    
    # Test MSB first
    output_msb_shape = tuple(int(x) for x in output_msb.shape)
    output_msb_tvm = tvm.nd.array(np.zeros(output_msb_shape, dtype=np.uint8), dev)
    func_msb(data_tvm, output_msb_tvm)
    output_msb_np = output_msb_tvm.numpy()
    
    # Test LSB first
    output_lsb_shape = tuple(int(x) for x in output_lsb.shape)
    output_lsb_tvm = tvm.nd.array(np.zeros(output_lsb_shape, dtype=np.uint8), dev)
    func_lsb(data_tvm, output_lsb_tvm)
    output_lsb_np = output_lsb_tvm.numpy()
    
    print("\n" + "="*70)
    print("MSB-first packing (original):")
    print("="*70)
    for bit in range(8):
        val = output_msb_np[0, 0, bit, 0, 0]
        print(f"Bit plane {bit}: {val:3d} = {val:08b}")
    
    print("\n" + "="*70)
    print("LSB-first packing (new):")
    print("="*70)
    for bit in range(8):
        val = output_lsb_np[0, 0, bit, 0, 0]
        print(f"Bit plane {bit}: {val:3d} = {val:08b}")
    
    # Verify LSB first
    print("\n" + "="*70)
    print("Verification:")
    print("="*70)
    
    # For LSB first with data [0,1,2,3,4,5,6,7]:
    # Bit 0: should have pattern [0,1,0,1,0,1,0,1] = 0b01010101 = 0x55
    # Bit 1: should have pattern [0,0,1,1,0,0,1,1] = 0b00110011 = 0x33
    # Bit 2: should have pattern [0,0,0,0,1,1,1,1] = 0b00001111 = 0x0F
    
    expected_lsb = [0x55, 0x33, 0x0F, 0x00, 0x00, 0x00, 0x00, 0x00]
    
    print("\nLSB-first expected vs actual:")
    all_match = True
    for bit in range(8):
        actual = output_lsb_np[0, 0, bit, 0, 0]
        expected = expected_lsb[bit]
        match = (actual == expected)
        all_match = all_match and match
        status = "✓" if match else "✗"
        print(f"  Bit {bit}: expected {expected:#04x} ({expected:08b}), actual {actual:#04x} ({actual:08b}) {status}")
    
    # For MSB first:
    # Bit packing goes from right to left (MSB first)
    # data[0]=0 at position 7, data[1]=1 at position 6, etc.
    expected_msb = [0xAA, 0xCC, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00]
    
    print("\nMSB-first expected vs actual:")
    for bit in range(8):
        actual = output_msb_np[0, 0, bit, 0, 0]
        expected = expected_msb[bit]
        match = (actual == expected)
        status = "✓" if match else "✗"
        print(f"  Bit {bit}: expected {expected:#04x} ({expected:08b}), actual {actual:#04x} ({actual:08b}) {status}")
    
    print("\n" + "="*70)
    if all_match:
        print("✓ LSB-first packing works correctly!")
    else:
        print("✗ LSB-first packing has issues")
    print("="*70)
    
    return all_match

def test_uint64_lsb():
    """Test LSB-first with uint64"""
    print("\n" + "="*70)
    print("Test: uint64 with LSB-first")
    print("="*70)
    
    N, C, H, W = 1, 64, 1, 1
    data = te.placeholder((N, C, H, W), name='data', dtype='int32')
    
    output = bitpack(data, bits=8, pack_axis=1, bit_axis=1, pack_type='uint64', msb_first=False)
    
    s = te.create_schedule(output.op)
    func = tvm.build(s, [data, output], target='llvm')
    
    # Simple test: first 8 elements are [0,1,2,3,4,5,6,7], rest are 0
    data_np = np.zeros((N, C, H, W), dtype=np.int32)
    data_np[0, :8, 0, 0] = np.arange(8)
    
    dev = tvm.cpu()
    data_tvm = tvm.nd.array(data_np, dev)
    output_shape = tuple(int(x) for x in output.shape)
    output_tvm = tvm.nd.array(np.zeros(output_shape, dtype=np.uint32), dev)
    
    func(data_tvm, output_tvm)
    output_np = output_tvm.numpy()
    
    print(f"Output shape: {output_np.shape}")
    print(f"\nFirst 8 input values: {data_np[0, :8, 0, 0]}")
    print(f"\nBit plane 0, chunk 0 (first 32 bits):")
    val = output_np[0, 0, 0, 0, 0, 0]
    print(f"  Value: {val:#010x}")
    print(f"  Binary (first 8 bits): {val & 0xFF:#010b}")
    
    # Expected: bit 0 of [0,1,2,3,4,5,6,7] = [0,1,0,1,0,1,0,1] = 0x55
    expected = 0x55
    if (val & 0xFF) == expected:
        print(f"  ✓ Matches expected: {expected:#04x}")
        return True
    else:
        print(f"  ✗ Expected {expected:#04x}, got {val & 0xFF:#04x}")
        return False

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Testing MSB-first vs LSB-first bit packing")
    print("="*70 + "\n")
    
    results = []
    
    try:
        results.append(("MSB vs LSB", test_msb_vs_lsb()))
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("MSB vs LSB", False))
    
    try:
        results.append(("uint64 LSB", test_uint64_lsb()))
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("uint64 LSB", False))
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:25s} {status}")
    print("="*70)
    
    if all(r[1] for r in results):
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*70)
