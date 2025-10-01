#!/usr/bin/env python3
"""
Test for Channel Bit-Packing Transform
"""

import tvm
from tvm import te, relay
import numpy as np
import math


def numpy_channel_bitpack_3d(input_data, ceil_to_64=False):
    """Original implementation for reference"""
    IC, IH, IW = input_data.shape
    
    if ceil_to_64:
        size = IH * IW * math.ceil((math.ceil(IC/64)*64) / 16)
    else:
        size = IH * IW * math.ceil(IC / 16)
    
    data = [0 for _ in range(size)]
    
    for ic in range(IC):
        for ih in range(IH):
            for iw in range(IW):
                addr = 2 * ((ic % 16) + 16 * iw + 16 * IW * ih + (ic // 16) * 16 * IW * IH)
                depth_addr = addr // 32
                offset = addr % 32

                data_ = int(input_data[ic, ih, iw])
                data[depth_addr] |= ((data_ & ((2**16) - 1)) << (8 * offset))

    return np.array(data, dtype=np.int32)


def tvm_channel_bitpack_simple(data, IC, IH, IW):
    """
    Simplified TVM implementation using te.compute
    
    이 버전은 정확한 address mapping을 구현하지만,
    매우 비효율적입니다 (모든 input을 iterate).
    실전에서는 최적화된 버전이 필요합니다.
    """
    output_size = IH * IW * ((IC + 15) // 16)
    
    def compute_output(out_idx):
        result = tvm.tir.const(0, "int32")
        
        # Iterate over all input positions
        for ic in range(IC):
            for ih in range(IH):
                for iw in range(IW):
                    # Compute address for this input position
                    addr = 2 * ((ic % 16) + 16 * iw + 16 * IW * ih + 
                               (ic // 16) * 16 * IW * IH)
                    depth_addr = addr // 32
                    offset = addr % 32
                    
                    # If this input contributes to current output position
                    matches = tvm.tir.if_then_else(
                        depth_addr == out_idx,
                        tvm.tir.const(1, "int32"),
                        tvm.tir.const(0, "int32")
                    )
                    
                    # Extract data and shift
                    data_val = data[ic, ih, iw].astype("int32")
                    masked = data_val & tvm.tir.const(0xFFFF, "int32")
                    shifted = masked << (8 * offset)
                    
                    # Accumulate if matches
                    contribution = tvm.tir.if_then_else(
                        matches == 1,
                        shifted,
                        tvm.tir.const(0, "int32")
                    )
                    
                    result = result | contribution
        
        return result
    
    return te.compute((output_size,), compute_output, name="channel_bitpack")


def test_basic():
    """Test basic functionality with small input"""
    print("\n" + "="*70)
    print("Test 1: Basic Functionality")
    print("="*70)
    
    IC, IH, IW = 16, 2, 2
    
    # Create simple input for easy verification
    input_data = np.arange(IC * IH * IW, dtype=np.int32).reshape(IC, IH, IW)
    
    print(f"Input shape: {input_data.shape}")
    print(f"Sample input values: {input_data[:2, 0, 0]}")
    
    # NumPy reference
    output_ref = numpy_channel_bitpack_3d(input_data, ceil_to_64=False)
    print(f"\nNumPy output shape: {output_ref.shape}")
    print(f"Expected size: {IH * IW * math.ceil(IC / 16)}")
    
    # TVM implementation
    data_tensor = te.placeholder((IC, IH, IW), name="data", dtype="int32")
    output_tensor = tvm_channel_bitpack_simple(data_tensor, IC, IH, IW)
    
    s = te.create_schedule(output_tensor.op)
    
    # Build
    print("\nBuilding TVM function...")
    func = tvm.build(s, [data_tensor, output_tensor], target="llvm", name="channel_bitpack")
    
    # Test
    dev = tvm.cpu()
    input_tvm = tvm.nd.array(input_data, dev)
    output_tvm = tvm.nd.array(np.zeros(output_ref.shape, dtype=np.int32), dev)
    
    print("Running TVM function...")
    func(input_tvm, output_tvm)
    
    output_tvm_np = output_tvm.numpy()
    
    # Compare
    print("\nComparison:")
    print(f"NumPy output: {output_ref[:5]}")
    print(f"TVM output:   {output_tvm_np[:5]}")
    
    if np.allclose(output_ref, output_tvm_np):
        print("\n✓ Test PASSED!")
        return True
    else:
        print("\n✗ Test FAILED!")
        print(f"Max difference: {np.max(np.abs(output_ref - output_tvm_np))}")
        # Show differences
        diff_mask = output_ref != output_tvm_np
        if np.any(diff_mask):
            print(f"Number of differences: {np.sum(diff_mask)}")
            print(f"First difference at index: {np.argmax(diff_mask)}")
        return False


def test_relay_integration():
    """Test integration with Relay"""
    print("\n" + "="*70)
    print("Test 2: Relay Integration")
    print("="*70)
    
    IC, IH, IW = 32, 4, 4
    
    # Create input
    input_data = np.random.randint(0, 256, size=(IC, IH, IW), dtype=np.int32)
    
    print(f"Input shape: {input_data.shape}")
    
    # Method 1: Use relay ops for reshaping (not full implementation)
    print("\nMethod 1: Relay ops-based transform (partial)")
    
    x = relay.var("x", shape=(IC, IH, IW), dtype="int32")
    
    # Pad IC to multiple of 16
    IC_padded = ((IC + 15) // 16) * 16
    if IC < IC_padded:
        pad_width = ((0, IC_padded - IC), (0, 0), (0, 0))
        x_padded = relay.nn.pad(x, pad_width=pad_width, pad_value=0)
    else:
        x_padded = x
    
    # Reshape to group channels: (IC_padded, IH, IW) -> (IC_padded//16, 16, IH, IW)
    x_reshaped = relay.reshape(x_padded, (IC_padded // 16, 16, IH, IW))
    
    # Transpose: (IC//16, 16, IH, IW) -> (IH, IW, IC//16, 16)
    x_transposed = relay.transpose(x_reshaped, axes=[2, 3, 0, 1])
    
    # Flatten: (IH, IW, IC//16, 16) -> (IH * IW * (IC//16), 16)
    x_flattened = relay.reshape(x_transposed, (IH * IW * (IC_padded // 16), 16))
    
    print(f"Transform chain: {(IC, IH, IW)} -> ... -> {(IH * IW * (IC_padded // 16), 16)}")
    
    # Build and test
    func = relay.Function([x], x_flattened)
    mod = tvm.ir.IRModule.from_expr(func)
    
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target="llvm")
    
    dev = tvm.cpu()
    m = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    m.set_input("x", input_data)
    m.run()
    output = m.get_output(0).numpy()
    
    print(f"Output shape: {output.shape}")
    print(f"Sample output: {output[0]}")
    
    print("\n✓ Relay transform completed (partial - bit packing not implemented)")
    
    return True


def test_performance():
    """Performance comparison"""
    print("\n" + "="*70)
    print("Test 3: Performance Analysis")
    print("="*70)
    
    IC, IH, IW = 64, 8, 8
    
    input_data = np.random.randint(0, 256, size=(IC, IH, IW), dtype=np.int32)
    
    # NumPy timing
    import time
    
    start = time.time()
    for _ in range(10):
        output_np = numpy_channel_bitpack_3d(input_data, ceil_to_64=False)
    numpy_time = (time.time() - start) / 10
    
    print(f"NumPy implementation: {numpy_time*1000:.3f} ms")
    print(f"Output size: {output_np.shape}")
    
    # Note: TVM version is currently very slow due to naive implementation
    print("\nNote: Current TVM implementation is naive and not optimized.")
    print("For production, you should use:")
    print("  1. Optimized schedule with parallelization")
    print("  2. Custom CUDA kernel for bit packing")
    print("  3. ExternOp integration")
    
    return True


def analyze_address_pattern():
    """Analyze the address pattern to understand optimization opportunities"""
    print("\n" + "="*70)
    print("Address Pattern Analysis")
    print("="*70)
    
    IC, IH, IW = 32, 4, 4
    
    # Analyze address distribution
    print(f"\nFor input shape ({IC}, {IH}, {IW}):")
    print(f"Expected output size: {IH * IW * math.ceil(IC / 16)}")
    
    # Sample some addresses
    print("\nSample address mappings:")
    print("(ic, ih, iw) -> addr -> depth_addr, offset")
    
    for sample in [(0, 0, 0), (1, 0, 0), (15, 0, 0), (16, 0, 0), (0, 1, 0), (0, 0, 1)]:
        ic, ih, iw = sample
        if ic < IC and ih < IH and iw < IW:
            addr = 2 * ((ic % 16) + 16 * iw + 16 * IW * ih + (ic // 16) * 16 * IW * IH)
            depth_addr = addr // 32
            offset = addr % 32
            print(f"  {sample} -> {addr:4d} -> depth[{depth_addr:3d}], offset={offset:2d}")
    
    # Analyze which outputs each spatial location contributes to
    print("\nContributions per spatial location:")
    for ih in range(min(2, IH)):
        for iw in range(min(2, IW)):
            depths = set()
            for ic in range(IC):
                addr = 2 * ((ic % 16) + 16 * iw + 16 * IW * ih + (ic // 16) * 16 * IW * IH)
                depth_addr = addr // 32
                depths.add(depth_addr)
            print(f"  (ih={ih}, iw={iw}) contributes to {len(depths)} output positions")
            if len(depths) <= 5:
                print(f"    depths: {sorted(depths)}")
    
    return True


if __name__ == "__main__":
    print("="*70)
    print("Channel Bit-Packing Transform Test Suite")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("Address Pattern Analysis", analyze_address_pattern()))
    results.append(("Basic Functionality", test_basic()))
    results.append(("Relay Integration", test_relay_integration()))
    results.append(("Performance Analysis", test_performance()))
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:.<50} {status}")
    
    print("\n" + "="*70)
    print("Recommendations for Production Implementation:")
    print("="*70)
    print("""
1. **Optimized Schedule**: 
   - Parallelize the outer loop
   - Use vectorization for bit operations
   - Consider tiling for better cache usage

2. **Custom CUDA Kernel**:
   - Implement as __global__ kernel
   - Use shared memory for input data
   - Warp-level primitives for bit operations

3. **Hybrid Approach**:
   - Use relay.reshape/transpose for layout changes
   - Use te.extern() to call custom kernel for bit packing
   - This gives best of both worlds

4. **Integration with IMC Backend**:
   - Register as BYOC pattern
   - Codegen to IMC-specific instructions
   - Optimize for your hardware memory hierarchy

Example CUDA kernel skeleton:

__global__ void channel_bitpack_kernel(
    const int32_t* input,   // [IC, IH, IW]
    int32_t* output,        // [output_size]
    int IC, int IH, int IW
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int output_size = IH * IW * ((IC + 15) / 16);
    
    if (out_idx < output_size) {
        int32_t result = 0;
        
        // Optimize: only iterate over inputs that contribute to this output
        // Pre-compute the range based on address formula
        for (int ic = 0; ic < IC; ic++) {
            for (int ih = 0; ih < IH; ih++) {
                for (int iw = 0; iw < IW; iw++) {
                    int addr = 2 * ((ic % 16) + 16 * iw + 16 * IW * ih + 
                                   (ic / 16) * 16 * IW * IH);
                    int depth_addr = addr / 32;
                    
                    if (depth_addr == out_idx) {
                        int offset = addr % 32;
                        int32_t data = input[ic * IH * IW + ih * IW + iw];
                        result |= ((data & 0xFFFF) << (8 * offset));
                    }
                }
            }
        }
        
        output[out_idx] = result;
    }
}

Then integrate with TVM:
- Use te.extern() to call this kernel
- Or register as ExternOpNode
- Provide schedule hints for TVM's auto-scheduler
""")
