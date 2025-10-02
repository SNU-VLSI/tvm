"""
Comprehensive Bitpack Operator Test Suite
Tests both     s = te.create_schedule(out_msb.op)
    f_msb = tvm.build(s, [data, out_msb], "llvm", name="bitpack_msb")
    
    out_shape_msb = tuple(int(s) for s in out_msb.shape)
    out_tvm_msb = tvm.nd.array(np.zeros(out_shape_msb, dtype="uint8"), ctx)
    f_msb(data_tvm, out_tvm_msb)vel and Relay-level implementations with various configurations.

Test Coverage:
1. TE-level tests:
   - MSB-first vs LSB-first (bits=1)
   - uint32/uint64/uint128 chunking (bits=1)
   - bits=4 with various pack types
   - Different bit_axis positions
   - Padding scenarios
   
2. Relay-level tests:
   - MSB-first vs LSB-first with relay.build()
   - uint32/uint64/uint128 chunking
   - bits=4 cases
   - Different bit_axis positions
"""

import numpy as np
import tvm
from tvm import te, relay
from tvm.topi.nn import bitpack
import tvm.testing


# ============================================================================
# TE-Level Tests
# ============================================================================

def test_te_msb_lsb_bits1():
    """Test MSB-first vs LSB-first bit packing with bits=1"""
    print("\n" + "="*70)
    print("TE Test 1: MSB-first vs LSB-first (bits=1)")
    print("="*70)
    
    # Input: [0, 1, 2, 3, 4, 5, 6, 7] -> LSBs are [0,1,0,1,0,1,0,1]
    data_np = np.array([[[[[0, 1, 2, 3, 4, 5, 6, 7]]]]], dtype=np.uint8)
    print(f"Input data: {data_np[0, 0, 0, 0, :]}")
    print(f"Input LSBs: {[int(x & 1) for x in data_np[0, 0, 0, 0, :]]}")
    
    # MSB-first
    data = te.placeholder((1, 1, 1, 1, 8), dtype="uint8", name="data")
    out_msb = bitpack(data, bits=1, pack_axis=4, bit_axis=4, 
                      pack_type="uint8", msb_first=True)
    s = te.create_schedule(out_msb.op)
    f_msb = tvm.build(s, [data, out_msb], "llvm", name="bitpack_msb")
    
    ctx = tvm.cpu()
    data_tvm = tvm.nd.array(data_np, ctx)
    out_shape_msb = tuple(int(s) for s in out_msb.shape)
    out_tvm_msb = tvm.nd.array(np.zeros(out_shape_msb, dtype="uint8"), ctx)
    f_msb(data_tvm, out_tvm_msb)
    result_msb = out_tvm_msb.numpy()
    
    # LSB-first
    out_lsb = bitpack(data, bits=1, pack_axis=4, bit_axis=4, 
                      pack_type="uint8", msb_first=False)
    s = te.create_schedule(out_lsb.op)
    f_lsb = tvm.build(s, [data, out_lsb], "llvm", name="bitpack_lsb")
    
    out_shape_lsb = tuple(int(s) for s in out_lsb.shape)
    out_tvm_lsb = tvm.nd.array(np.zeros(out_shape_lsb, dtype="uint8"), ctx)
    f_lsb(data_tvm, out_tvm_lsb)
    result_lsb = out_tvm_lsb.numpy()
    
    msb_val = result_msb[0, 0, 0, 0, 0]
    lsb_val = result_lsb[0, 0, 0, 0, 0]

    print(f"\nMSB-first output: 0x{msb_val.item():02x} = {format(msb_val.item(), '08b')}b")
    print(f"LSB-first output: 0x{lsb_val.item():02x} = {format(lsb_val.item(), '08b')}b")
    print(f"Expected MSB: 0x55 (01010101b)")
    print(f"Expected LSB: 0xaa (10101010b)")
    
    assert msb_val == 0x55, f"MSB-first failed: got 0x{msb_val.item():02x}"
    assert lsb_val == 0xaa, f"LSB-first failed: got 0x{lsb_val.item():02x}"
    print("✓ PASS")
    return True


def test_te_uint64_chunking_bits1():
    """Test uint64 → 2x uint32 chunks with bits=1"""
    print("\n" + "="*70)
    print("TE Test 2: uint64 chunking (bits=1)")
    print("="*70)
    
    data_np = np.random.randint(0, 256, size=(1, 64, 2, 2), dtype=np.uint8)
    print(f"Input shape:  {data_np.shape}")
    
    data = te.placeholder((1, 64, 2, 2), dtype="uint8", name="data")
    out = bitpack(data, bits=1, pack_axis=1, bit_axis=1, 
                  pack_type="uint64", msb_first=True)
    
    print(f"Output shape: {out.shape}")
    print(f"Expected:     (1, 1, 1, 2, 2, 2) - last dim=2 for chunks")
    
    assert len(out.shape) == 6, f"Should have 6 dimensions, got {len(out.shape)}"
    assert out.shape[-1] == 2, f"Last dimension should be 2, got {out.shape[-1]}"
    print("✓ PASS")
    return True


def test_te_uint128_chunking_bits1():
    """Test uint128 → 4x uint32 chunks with bits=1"""
    print("\n" + "="*70)
    print("TE Test 3: uint128 chunking (bits=1)")
    print("="*70)
    
    data_np = np.random.randint(0, 256, size=(1, 128, 2, 2), dtype=np.uint8)
    print(f"Input shape:  {data_np.shape}")
    
    data = te.placeholder((1, 128, 2, 2), dtype="uint8", name="data")
    out = bitpack(data, bits=1, pack_axis=1, bit_axis=1, 
                  pack_type="uint128", msb_first=True)
    
    print(f"Output shape: {out.shape}")
    print(f"Expected:     (1, 1, 1, 2, 2, 4) - last dim=4 for chunks")
    
    assert len(out.shape) == 6, f"Should have 6 dimensions, got {len(out.shape)}"
    assert out.shape[-1] == 4, f"Last dimension should be 4, got {out.shape[-1]}"
    print("✓ PASS")
    return True


def test_te_uint32_no_chunking_bits1():
    """Test uint32 has NO chunking (backward compatibility, bits=1)"""
    print("\n" + "="*70)
    print("TE Test 4: uint32 backward compatibility (bits=1)")
    print("="*70)
    
    data_np = np.random.randint(0, 256, size=(1, 32, 2, 2), dtype=np.uint8)
    print(f"Input shape:  {data_np.shape}")
    
    data = te.placeholder((1, 32, 2, 2), dtype="uint8", name="data")
    out = bitpack(data, bits=1, pack_axis=1, bit_axis=1, 
                  pack_type="uint32", msb_first=True)
    
    print(f"Output shape: {out.shape}")
    print(f"Expected:     (1, 1, 1, 2, 2) - NO chunk dimension")
    
    assert len(out.shape) == 5, f"Should have 5 dimensions, got {len(out.shape)}"
    print("✓ PASS - No chunk dimension added")
    return True


def test_te_bits4_uint8():
    """Test bits=4 with uint8 pack_type"""
    print("\n" + "="*70)
    print("TE Test 5: bits=4 with uint8")
    print("="*70)
    
    # Input: 16 elements, extract 4 bits each
    # 16 * 4 bits = 64 bits = 8 bytes
    data_np = np.array([[[[0xA5, 0x3C, 0xF0, 0x1E, 
                           0x7B, 0x94, 0x62, 0xD8,
                           0x29, 0xE1, 0x56, 0xC3,
                           0x8F, 0x4A, 0xB7, 0x0D]]]], dtype=np.uint8)
    print(f"Input shape: {data_np.shape}")
    print(f"Input data (first 4): {[hex(x) for x in data_np[0, 0, 0, :4]]}")
    
    data = te.placeholder((1, 1, 1, 16), dtype="uint8", name="data")
    out = bitpack(data, bits=4, pack_axis=3, bit_axis=3, 
                  pack_type="uint8", msb_first=True)
    
    print(f"Output shape: {out.shape}")
    print(f"Expected:     (1, 1, 1, 4, 2) - 4 bit planes, 16/8=2 packed uint8s per plane")
    
    s = te.create_schedule(out.op)
    f = tvm.build(s, [data, out], "llvm", name="bitpack_bits4")
    
    ctx = tvm.cpu()
    data_tvm = tvm.nd.array(data_np, ctx)
    out_shape = tuple(int(s) for s in out.shape)
    out_tvm = tvm.nd.array(np.zeros(out_shape, dtype="uint8"), ctx)
    f(data_tvm, out_tvm)
    result = out_tvm.numpy()
    
    print(f"Result shape: {result.shape}")
    print(f"Result sample [0,0,0,0,:]: {[hex(x) for x in result[0, 0, 0, 0, :]]}")
    
    assert result.shape == (1, 1, 1, 4, 2), f"Shape mismatch: {result.shape}"
    print("✓ PASS")
    return True


def test_te_bits4_uint64():
    """Test bits=4 with uint64 (should create 2x uint32 chunks)"""
    print("\n" + "="*70)
    print("TE Test 6: bits=4 with uint64 chunking")
    print("="*70)
    
    # Input: 128 elements, extract 4 bits each
    # 128 * 4 bits = 512 bits = 8 uint64 = 16 uint32
    data_np = np.random.randint(0, 256, size=(1, 128, 2, 2), dtype=np.uint8)
    print(f"Input shape: {data_np.shape}")
    
    data = te.placeholder((1, 128, 2, 2), dtype="uint8", name="data")
    out = bitpack(data, bits=4, pack_axis=1, bit_axis=1, 
                  pack_type="uint64", msb_first=True)
    
    print(f"Output shape: {out.shape}")
    print(f"Expected:     (1, 4, 2, 2, 2, 2)")
    print(f"  - 128 elements / 64 bits_per_uint64 = 2 packed per plane")
    print(f"  - Each uint64 → 2 uint32 chunks (last dim)")
    print(f"  - 4 bit planes (bit_axis dimension)")
    
    expected_shape = (1, 4, 2, 2, 2, 2)
    assert tuple(out.shape) == expected_shape, f"Shape mismatch: got {out.shape}, expected {expected_shape}"
    print("✓ PASS")
    return True


def test_te_bit_axis_second_lowest():
    """Test with bit_axis at second lowest dimension (bits=1)"""
    print("\n" + "="*70)
    print("TE Test 7: bit_axis at second lowest dimension (bits=1)")
    print("="*70)
    
    # Input shape: (1, 2, 32, 4)
    # bit_axis=2 (second from end)
    # pack_axis=2
    data_np = np.random.randint(0, 256, size=(1, 2, 32, 4), dtype=np.uint8)
    print(f"Input shape: {data_np.shape}")
    print(f"bit_axis=2, pack_axis=2 (second lowest dim)")
    
    data = te.placeholder((1, 2, 32, 4), dtype="uint8", name="data")
    out = bitpack(data, bits=1, pack_axis=2, bit_axis=2, 
                  pack_type="uint32", msb_first=True)
    
    print(f"Output shape: {out.shape}")
    print(f"Expected:     (1, 2, 1, 1, 4)")
    print(f"  - bit_axis=2 inserts 'bits' dimension")
    print(f"  - pack_axis=2 compresses 32 → 1 (32bits/32bits)")
    
    expected_shape = (1, 2, 1, 1, 4)
    assert tuple(out.shape) == expected_shape, f"Shape mismatch: got {out.shape}, expected {expected_shape}"
    print("✓ PASS")
    return True


def test_te_bit_axis_second_lowest_bits4():
    """Test with bit_axis at second lowest dimension (bits=4)"""
    print("\n" + "="*70)
    print("TE Test 8: bit_axis at second lowest dim with bits=4")
    print("="*70)
    
    # Input shape: (1, 3, 64, 5)
    # bit_axis=2 (second from end)
    # pack_axis=2
    # bits=4 means 4 bit planes
    data_np = np.random.randint(0, 256, size=(1, 3, 64, 5), dtype=np.uint8)
    print(f"Input shape: {data_np.shape}")
    print(f"bit_axis=2, pack_axis=2, bits=4")
    
    data = te.placeholder((1, 3, 64, 5), dtype="uint8", name="data")
    out = bitpack(data, bits=4, pack_axis=2, bit_axis=2, 
                  pack_type="uint32", msb_first=True)
    
    print(f"Output shape: {out.shape}")
    print(f"Expected:     (1, 3, 4, 2, 5)")
    print(f"  - bit_axis=2 inserts 'bits=4' dimension")
    print(f"  - pack_axis=2: 64 elements / 32 bits = 2 packed per plane")
    
    expected_shape = (1, 3, 4, 2, 5)
    assert tuple(out.shape) == expected_shape, f"Shape mismatch: got {out.shape}, expected {expected_shape}"
    print("✓ PASS")
    return True


def test_te_padding_bits4():
    """Test automatic padding with bits=4"""
    print("\n" + "="*70)
    print("TE Test 9: Padding with bits=4")
    print("="*70)
    
    # 11 elements, 4 bits each = 44 bits
    # Need to pad to 48 bits (6 bytes) for uint8
    data_np = np.arange(11, dtype=np.uint8).reshape(1, 1, 11)
    print(f"Input shape: {data_np.shape}")
    print(f"Input data: {data_np[0, 0, :]}")
    print(f"11 elements * 4 bits = 44 bits → pad to 48 bits (6 bytes)")
    
    data = te.placeholder((1, 1, 11), dtype="uint8", name="data")
    out = bitpack(data, bits=4, pack_axis=2, bit_axis=2, 
                  pack_type="uint8", msb_first=True)
    
    s = te.create_schedule(out.op)
    f = tvm.build(s, [data, out], "llvm", name="bitpack_padding")
    
    ctx = tvm.cpu()
    data_tvm = tvm.nd.array(data_np, ctx)
    out_shape = tuple(int(s) for s in out.shape)
    out_tvm = tvm.nd.array(np.zeros(out_shape, dtype="uint8"), ctx)
    f(data_tvm, out_tvm)
    result = out_tvm.numpy()
    
    print(f"Output shape: {result.shape}")
    print(f"Expected:     (1, 1, 4, 2) - 4 bit planes, 11/8=2 packed uint8s per plane (with padding)")
    
    expected_shape = (1, 1, 4, 2)
    assert result.shape == expected_shape, f"Shape mismatch: {result.shape}"
    print("✓ PASS")
    return True


# ============================================================================
# Relay-Level Tests
# ============================================================================

def test_relay_msb_lsb_bits1():
    """Test MSB-first vs LSB-first at Relay level with bits=1"""
    print("\n" + "="*70)
    print("Relay Test 1: MSB vs LSB with relay.build() (bits=1)")
    print("="*70)
    
    data_np = np.array([[[0, 1, 2, 3, 4, 5, 6, 7]]], dtype=np.uint8)
    print(f"Input data: {data_np[0, 0, :]}")
    print(f"Input LSBs: {[int(x & 1) for x in data_np[0, 0, :]]}")
    
    target = "llvm"
    dev = tvm.device(target, 0)
    
    # MSB-first
    data_var = relay.var("data", shape=(1, 1, 8), dtype="uint8")
    out_msb = relay.nn.bitpack(data_var, bits=1, pack_axis=2, bit_axis=2, 
                                pack_type="uint8", msb_first=True)
    func_msb = relay.Function([data_var], out_msb)
    
    with tvm.transform.PassContext(opt_level=3):
        lib_msb = relay.build(func_msb, target=target)
    
    module_msb = tvm.contrib.graph_executor.GraphModule(lib_msb["default"](dev))
    module_msb.set_input("data", data_np)
    module_msb.run()
    result_msb = module_msb.get_output(0).numpy()
    
    # LSB-first
    out_lsb = relay.nn.bitpack(data_var, bits=1, pack_axis=2, bit_axis=2, 
                                pack_type="uint8", msb_first=False)
    func_lsb = relay.Function([data_var], out_lsb)
    
    with tvm.transform.PassContext(opt_level=3):
        lib_lsb = relay.build(func_lsb, target=target)
    
    module_lsb = tvm.contrib.graph_executor.GraphModule(lib_lsb["default"](dev))
    module_lsb.set_input("data", data_np)
    module_lsb.run()
    result_lsb = module_lsb.get_output(0).numpy()
    
    msb_val = int(result_msb.flatten()[0])
    lsb_val = int(result_lsb.flatten()[0])
    
    print(f"\nMSB-first: 0x{msb_val:02x} = {format(msb_val, '08b')}b")
    print(f"LSB-first: 0x{lsb_val:02x} = {format(lsb_val, '08b')}b")
    print(f"Expected MSB: 0x55, Expected LSB: 0xaa")
    
    assert msb_val == 0x55, f"MSB failed: got 0x{msb_val:02x}"
    assert lsb_val == 0xaa, f"LSB failed: got 0x{lsb_val:02x}"
    print("✓ PASS")
    return True


def test_relay_uint64_chunking_bits1():
    """Test uint64 chunking at Relay level with bits=1"""
    print("\n" + "="*70)
    print("Relay Test 2: uint64 → 2x uint32 chunks (bits=1)")
    print("="*70)
    
    data_np = np.random.randint(0, 256, size=(1, 64, 2, 2), dtype=np.uint8)
    print(f"Input shape: {data_np.shape}")
    
    data_var = relay.var("data", shape=(1, 64, 2, 2), dtype="uint8")
    out = relay.nn.bitpack(data_var, bits=1, pack_axis=1, bit_axis=1, 
                           pack_type="uint64", msb_first=True)
    func = relay.Function([data_var], out)
    
    target = "llvm"
    dev = tvm.device(target, 0)
    
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target=target)
    
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    module.set_input("data", data_np)
    module.run()
    result = module.get_output(0).numpy()
    
    print(f"Output shape: {result.shape}")
    print(f"Expected:     (1, 1, 1, 2, 2, 2)")
    
    expected_shape = (1, 1, 1, 2, 2, 2)
    assert result.shape == expected_shape, f"Shape mismatch: {result.shape}"
    print("✓ PASS")
    return True


def test_relay_uint128_chunking_bits1():
    """Test uint128 chunking at Relay level with bits=1"""
    print("\n" + "="*70)
    print("Relay Test 3: uint128 → 4x uint32 chunks (bits=1)")
    print("="*70)
    
    data_np = np.random.randint(0, 256, size=(1, 128, 2, 2), dtype=np.uint8)
    print(f"Input shape: {data_np.shape}")
    
    data_var = relay.var("data", shape=(1, 128, 2, 2), dtype="uint8")
    out = relay.nn.bitpack(data_var, bits=1, pack_axis=1, bit_axis=1, 
                           pack_type="uint128", msb_first=True)
    func = relay.Function([data_var], out)
    
    target = "llvm"
    dev = tvm.device(target, 0)
    
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target=target)
    
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    module.set_input("data", data_np)
    module.run()
    result = module.get_output(0).numpy()
    
    print(f"Output shape: {result.shape}")
    print(f"Expected:     (1, 1, 1, 2, 2, 4)")
    
    expected_shape = (1, 1, 1, 2, 2, 4)
    assert result.shape == expected_shape, f"Shape mismatch: {result.shape}"
    print("✓ PASS")
    return True


def test_relay_bits4_uint8():
    """Test bits=4 with uint8 at Relay level"""
    print("\n" + "="*70)
    print("Relay Test 4: bits=4 with uint8")
    print("="*70)
    
    data_np = np.random.randint(0, 256, size=(1, 1, 16), dtype=np.uint8)
    print(f"Input shape: {data_np.shape}")
    
    data_var = relay.var("data", shape=(1, 1, 16), dtype="uint8")
    out = relay.nn.bitpack(data_var, bits=4, pack_axis=2, bit_axis=2, 
                           pack_type="uint8", msb_first=True)
    func = relay.Function([data_var], out)
    
    target = "llvm"
    dev = tvm.device(target, 0)
    
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target=target)
    
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    module.set_input("data", data_np)
    module.run()
    result = module.get_output(0).numpy()
    
    print(f"Output shape: {result.shape}")
    print(f"Expected:     (1, 1, 4, 2) - 4 bit planes, 16/8=2 packed uint8s per plane")
    
    expected_shape = (1, 1, 4, 2)
    assert result.shape == expected_shape, f"Shape mismatch: {result.shape}"
    print("✓ PASS")
    return True


def test_relay_bits4_uint64():
    """Test bits=4 with uint64 chunking at Relay level"""
    print("\n" + "="*70)
    print("Relay Test 5: bits=4 with uint64 chunking")
    print("="*70)
    
    data_np = np.random.randint(0, 256, size=(1, 128, 2, 2), dtype=np.uint8)
    print(f"Input shape: {data_np.shape}")
    
    data_var = relay.var("data", shape=(1, 128, 2, 2), dtype="uint8")
    out = relay.nn.bitpack(data_var, bits=4, pack_axis=1, bit_axis=1, 
                           pack_type="uint64", msb_first=True)
    func = relay.Function([data_var], out)
    
    target = "llvm"
    dev = tvm.device(target, 0)
    
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target=target)
    
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    module.set_input("data", data_np)
    module.run()
    result = module.get_output(0).numpy()
    
    print(f"Output shape: {result.shape}")
    print(f"Expected:     (1, 4, 2, 2, 2, 2) - 4 bit planes, 128/64=2 packed per plane, 2 chunks")
    
    expected_shape = (1, 4, 2, 2, 2, 2)
    assert result.shape == expected_shape, f"Shape mismatch: {result.shape}"
    print("✓ PASS")
    return True


def test_relay_bit_axis_second_lowest():
    """Test bit_axis at second lowest dimension at Relay level"""
    print("\n" + "="*70)
    print("Relay Test 6: bit_axis at second lowest dim (bits=1)")
    print("="*70)
    
    data_np = np.random.randint(0, 256, size=(1, 2, 32, 4), dtype=np.uint8)
    print(f"Input shape: {data_np.shape}")
    print(f"bit_axis=2, pack_axis=2")
    
    data_var = relay.var("data", shape=(1, 2, 32, 4), dtype="uint8")
    out = relay.nn.bitpack(data_var, bits=1, pack_axis=2, bit_axis=2, 
                           pack_type="uint32", msb_first=True)
    func = relay.Function([data_var], out)
    
    target = "llvm"
    dev = tvm.device(target, 0)
    
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target=target)
    
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    module.set_input("data", data_np)
    module.run()
    result = module.get_output(0).numpy()
    
    print(f"Output shape: {result.shape}")
    print(f"Expected:     (1, 2, 1, 1, 4)")
    
    expected_shape = (1, 2, 1, 1, 4)
    assert result.shape == expected_shape, f"Shape mismatch: {result.shape}"
    print("✓ PASS")
    return True


def test_relay_bit_axis_second_lowest_bits4():
    """Test bit_axis at second lowest dimension with bits=4 at Relay level"""
    print("\n" + "="*70)
    print("Relay Test 7: bit_axis at second lowest dim with bits=4")
    print("="*70)
    
    data_np = np.random.randint(0, 256, size=(1, 3, 64, 5), dtype=np.uint8)
    print(f"Input shape: {data_np.shape}")
    print(f"bit_axis=2, pack_axis=2, bits=4")
    
    data_var = relay.var("data", shape=(1, 3, 64, 5), dtype="uint8")
    out = relay.nn.bitpack(data_var, bits=4, pack_axis=2, bit_axis=2, 
                           pack_type="uint32", msb_first=True)
    func = relay.Function([data_var], out)
    
    target = "llvm"
    dev = tvm.device(target, 0)
    
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target=target)
    
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    module.set_input("data", data_np)
    module.run()
    result = module.get_output(0).numpy()
    
    print(f"Output shape: {result.shape}")
    print(f"Expected:     (1, 3, 4, 2, 5) - 4 bit planes, 64/32=2 packed per plane")
    
    expected_shape = (1, 3, 4, 2, 5)
    assert result.shape == expected_shape, f"Shape mismatch: {result.shape}"
    print("✓ PASS")
    return True


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("COMPREHENSIVE BITPACK OPERATOR TEST SUITE")
    print("="*70)
    
    te_tests = [
        ("TE: MSB vs LSB (bits=1)", test_te_msb_lsb_bits1),
        ("TE: uint64 chunking (bits=1)", test_te_uint64_chunking_bits1),
        ("TE: uint128 chunking (bits=1)", test_te_uint128_chunking_bits1),
        ("TE: uint32 no chunking (bits=1)", test_te_uint32_no_chunking_bits1),
        ("TE: bits=4 with uint8", test_te_bits4_uint8),
        ("TE: bits=4 with uint64", test_te_bits4_uint64),
        ("TE: bit_axis 2nd lowest (bits=1)", test_te_bit_axis_second_lowest),
        ("TE: bit_axis 2nd lowest (bits=4)", test_te_bit_axis_second_lowest_bits4),
        ("TE: padding with bits=4", test_te_padding_bits4),
    ]
    
    relay_tests = [
        ("Relay: MSB vs LSB (bits=1)", test_relay_msb_lsb_bits1),
        ("Relay: uint64 chunking (bits=1)", test_relay_uint64_chunking_bits1),
        ("Relay: uint128 chunking (bits=1)", test_relay_uint128_chunking_bits1),
        ("Relay: bits=4 with uint8", test_relay_bits4_uint8),
        ("Relay: bits=4 with uint64", test_relay_bits4_uint64),
        ("Relay: bit_axis 2nd lowest (bits=1)", test_relay_bit_axis_second_lowest),
        ("Relay: bit_axis 2nd lowest (bits=4)", test_relay_bit_axis_second_lowest_bits4),
    ]
    
    failed_tests = []
    
    try:
        # Run TE tests
        print("\n" + "="*70)
        print("RUNNING TE-LEVEL TESTS")
        print("="*70)
        for name, test_func in te_tests:
            try:
                test_func()
            except Exception as e:
                print(f"✗ FAILED: {name}")
                print(f"  Error: {e}")
                failed_tests.append((name, e))
        
        # Run Relay tests
        print("\n" + "="*70)
        print("RUNNING RELAY-LEVEL TESTS")
        print("="*70)
        for name, test_func in relay_tests:
            try:
                test_func()
            except Exception as e:
                print(f"✗ FAILED: {name}")
                print(f"  Error: {e}")
                failed_tests.append((name, e))
        
        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        total_tests = len(te_tests) + len(relay_tests)
        passed_tests = total_tests - len(failed_tests)
        
        print(f"\nTotal tests:  {total_tests}")
        print(f"Passed:       {passed_tests}")
        print(f"Failed:       {len(failed_tests)}")
        
        if failed_tests:
            print("\nFailed tests:")
            for name, error in failed_tests:
                print(f"  - {name}")
                print(f"    {error}")
            print("\n✗ SOME TESTS FAILED")
        else:
            print("\n✓ ALL TESTS PASSED!")
        
        print("="*70)
        
    except Exception as e:
        print("\n" + "="*70)
        print("✗ TEST SUITE FAILED")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise
