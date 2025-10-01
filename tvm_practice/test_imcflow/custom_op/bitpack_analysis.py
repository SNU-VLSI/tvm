"""
TVM bitpack 코드 상세 분석 및 커스텀 활용법

=============================================================================
1. te.compute에서 tuple이 반환되는 이유
=============================================================================

TVM의 te.compute()는 **multiple outputs**을 지원합니다.
이것이 tuple을 반환하는 핵심 이유입니다.

예제:
```python
def my_compute(*indices):
    # 여러 개의 값을 동시에 계산
    output1 = data[indices] + 1
    output2 = data[indices] * 2
    return (output1, output2)  # tuple 반환!

# 결과는 tuple of tensors
result = te.compute(shape, my_compute)
# result[0] = first output tensor
# result[1] = second output tensor
```

bitpack에서는:
- bits=8이면 8개의 bit plane을 각각 별도 tensor로 반환
- 각 bit plane은 해당 bit 위치의 값들만 모음
- 나중에 concatenate로 하나의 tensor로 합침

=============================================================================
2. bitpack 코드 라인별 상세 설명
=============================================================================
"""

import numpy as np
import tvm
from tvm import te
from tvm.topi.transform import concatenate

def bitpack_explained(data, bits, pack_axis, bit_axis, pack_type):
    """
    bitpack의 각 단계를 상세히 설명하는 버전
    
    목적: data의 각 원소를 bit 단위로 분해해서 packing
    
    예제:
    입력: [0b10110101, 0b11001100] (2개 원소, 각 8-bit)
    pack_type = "uint8" (8개씩 묶음)
    bits = 8 (8개 bit plane 생성)
    
    출력: 8개의 bit plane
    - plane[0]: LSB (최하위 비트들)
    - plane[7]: MSB (최상위 비트들)
    """
    
    ishape = data.shape
    n = len(ishape)
    
    # Step 1: data_width 결정
    # "uint64"이면 64개 원소를 하나로 packing
    if pack_type == "uint8":
        data_width = 8
    elif pack_type == "uint16":
        data_width = 16
    elif pack_type == "uint32":
        data_width = 32
    elif pack_type == "uint64":
        data_width = 64
    
    print(f"[Step 1] data_width = {data_width}")
    print(f"         {data_width}개의 원소를 하나의 {pack_type}로 packing")
    
    # Step 2: Output shape 계산
    # 예: (N, 256, H, W) -> pack_axis=1, bit_axis=4
    #     -> (N, 256/64, H, W, 1) if uint64
    #     -> (N, 4, H, W, 1)
    shape_vec = list(ishape)
    shape_vec[pack_axis] = shape_vec[pack_axis] // data_width
    shape_vec.insert(bit_axis, 1)  # bit axis를 위한 placeholder
    bitserial_oshape = tuple(shape_vec)
    
    print(f"\n[Step 2] Output shape 계산")
    print(f"         Input shape:  {ishape}")
    print(f"         pack_axis={pack_axis}을 {data_width}로 나눔")
    print(f"         bit_axis={bit_axis}에 새 dimension 삽입")
    print(f"         Output shape: {bitserial_oshape}")
    
    # Step 3: Bit masks
    # LSB부터 MSB까지의 mask
    masks = np.array([0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80])
    print(f"\n[Step 3] Bit masks: {[bin(m) for m in masks[:bits]]}")
    
    # Step 4: pack_axis adjustment
    # bit_axis가 pack_axis보다 앞에 있으면 pack_axis가 1 증가
    if bit_axis <= pack_axis:
        pack_axis += 1
    
    print(f"\n[Step 4] Adjusted pack_axis = {pack_axis}")
    
    # Step 5: Compute function (핵심!)
    def _bitpack(*indices):
        """
        각 output position에 대해 호출됨
        
        indices: output tensor의 좌표
        예: (n, c_packed, h, w, bit_plane) for shape (N, C/64, H, W, bits)
        
        이 함수는:
        1. data_width개(64개)의 input 원소를 읽음
        2. 각 원소에서 bit를 추출
        3. bits개의 separate value로 반환 (tuple!)
        """
        
        # bits개의 출력 값 초기화 (각 bit plane당 하나)
        packed_data = [tvm.tir.const(0, pack_type)] * bits
        
        print(f"\n[Step 5] _bitpack 함수 설명:")
        print(f"         - {bits}개의 bit plane 생성")
        print(f"         - 각 plane은 {pack_type} 타입")
        
        # data_width개의 원소를 순회 (예: 64개)
        for k in range(data_width):
            print(f"\n  [k={k}] {k}번째 원소 처리:")
            
            # Step 5-1: Output index를 input index로 역변환
            # 예: output[n, 2, h, w, bit] -> input[n, 2*64+k, h, w]
            idx = [0] * n
            j = 0
            for i in range(n + 1):
                if i == bit_axis:
                    continue  # bit_axis는 input에 없음
                if i == pack_axis:
                    # pack_axis는 확장: indices[i] * data_width + k
                    idx[j] = indices[i] * data_width + k
                    print(f"    pack_axis: idx[{j}] = {indices[i]} * {data_width} + {k}")
                else:
                    idx[j] = indices[i]
                j += 1
            
            print(f"    Reconstructed input index: {idx}")
            
            # Step 5-2: Input 원소 읽기
            element = data(*idx)
            print(f"    Read element: {element}")
            
            # Step 5-3: 각 bit 추출하고 해당 bit plane에 추가
            for b in range(bits):
                # b번째 bit 추출
                extracted_bit = ((element & tvm.tir.const(masks[b], "int32")) >> b).astype(pack_type)
                
                # 이전 값과 OR
                packed_data[b] = packed_data[b] | extracted_bit
                
                # 다음 원소를 위해 left shift (마지막 원소 제외)
                if k < data_width - 1:
                    packed_data[b] = packed_data[b] << 1
                
                if k == 0:  # 첫 번째 원소일 때만 출력
                    print(f"    Bit {b}: extract & pack")
        
        print(f"\n  최종: {bits}개의 값을 tuple로 반환")
        
        # 중요! tuple 반환
        return tuple(packed_data)
    
    # Step 6: te.compute 호출
    # _bitpack은 tuple을 반환하므로 output_tuple도 tuple of tensors
    output_tuple = te.compute(bitserial_oshape, _bitpack, name="bitpack", tag="bitpack")
    
    print(f"\n[Step 6] te.compute 결과:")
    print(f"         output_tuple은 {bits}개의 tensor를 담은 tuple")
    print(f"         각 tensor shape: {bitserial_oshape}")
    
    # Step 7: Concatenate (optional)
    if bits > 1:
        # bits개의 tensor를 bit_axis 방향으로 합침
        # (N, C/64, H, W, 1) x 8 -> (N, C/64, H, W, 8)
        result = concatenate(output_tuple, axis=bit_axis)
        print(f"\n[Step 7] Concatenate along bit_axis={bit_axis}")
        print(f"         Final shape: {result.shape}")
        return result
    
    return output_tuple


"""
=============================================================================
3. 당신의 Use Case: [N, C, H, W] -> [N, C/256, H, W, IB, 4] uint64
=============================================================================

목표:
- 256 channels을 하나로 packing
- IB (Input Bits) dimension 추가
- 4개의 uint64 values (256 bits = 4 x 64 bits)

이는 다음을 의미:
- 256개 channel의 동일 위치 (h, w) 값들을 모음
- 각 값에서 특정 bit 추출 (IB dimension)
- 64개씩 묶어서 4개의 uint64로 packing

예제:
C[0][h][w] = 0b10110101
C[1][h][w] = 0b11001100
...
C[255][h][w] = 0b00110011

-> 각 bit position마다:
   bit 0: C[0]~C[63]의 bit 0 -> uint64_0
          C[64]~C[127]의 bit 0 -> uint64_1
          C[128]~C[191]의 bit 0 -> uint64_2
          C[192]~C[255]의 bit 0 -> uint64_3
"""

def custom_channel_bitpack(data, input_bits=8):
    """
    [N, C, H, W] -> [N, C/256, H, W, IB, 4] uint64
    
    Parameters
    ----------
    data : te.Tensor
        Input with shape (N, C, H, W)
        C must be multiple of 256
    input_bits : int
        Number of bits per input element (IB dimension size)
    
    Returns
    -------
    output : te.Tensor
        Packed tensor with shape (N, C//256, H, W, input_bits, 4)
        dtype: uint64
    """
    N, C, H, W = data.shape
    
    assert C % 256 == 0, "C must be multiple of 256"
    
    C_packed = C // 256  # Number of 256-channel groups
    
    # Output shape: [N, C/256, H, W, IB, 4]
    output_shape = (N, C_packed, H, W, input_bits, 4)
    
    # Bit masks for extraction
    masks = np.array([1 << i for i in range(input_bits)])
    
    def _compute_packed(*indices):
        """
        indices: (n, c_group, h, w, ib, pack_idx)
        
        n: batch index
        c_group: which 256-channel group (0 ~ C/256-1)
        h, w: spatial position
        ib: which input bit (0 ~ input_bits-1)
        pack_idx: which uint64 (0~3 for 256 channels)
        """
        n, c_group, h, w, ib, pack_idx = indices
        
        # This uint64 packs channels [c_start : c_start + 64]
        c_start = c_group * 256 + pack_idx * 64
        
        result = tvm.tir.const(0, "uint64")
        
        # Pack 64 channels into one uint64
        for i in range(64):
            c = c_start + i
            
            # Read input element
            element = data[n, c, h, w].astype("int32")
            
            # Extract bit at position 'ib'
            bit_val = (element & masks[ib]) >> ib
            
            # Pack into uint64: LSB is channel 0, MSB is channel 63
            result = result | (bit_val.astype("uint64") << i)
        
        return result
    
    return te.compute(output_shape, _compute_packed, name="channel_bitpack", tag="channel_bitpack")


"""
=============================================================================
4. 더 효율적인 버전: bitpack 스타일 활용
=============================================================================

위의 custom 버전은 이해하기 쉽지만 비효율적입니다.
bitpack 스타일로 개선하면 더 효율적입니다.
"""

def custom_channel_bitpack_v2(data, input_bits=8):
    """
    bitpack 스타일을 활용한 효율적인 버전
    
    Strategy:
    1. 먼저 channel axis를 (C/256, 256)으로 reshape (conceptually)
    2. 256 channels을 (4, 64)로 다시 reshape
    3. Bitpack 로직 적용
    """
    N, C, H, W = data.shape
    
    assert C % 256 == 0, "C must be multiple of 256"
    
    C_packed = C // 256
    output_shape = (N, C_packed, H, W, input_bits, 4)
    
    masks = np.array([1 << i for i in range(input_bits)])
    
    def _bitpack_compute(*indices):
        """
        각 output position마다 64개 channel을 packing
        
        이 함수는 bitpack의 _bitpack과 유사하지만:
        - tuple 대신 단일 값 반환 (bit plane은 별도 dimension으로)
        - 64개 channel을 순회하며 bit 추출 & packing
        """
        n, c_group, h, w, ib, pack_idx = indices
        
        c_base = c_group * 256 + pack_idx * 64
        
        packed = tvm.tir.const(0, "uint64")
        
        # 64 channels을 iterate
        for k in range(64):
            c = c_base + k
            
            element = data[n, c, h, w].astype("int32")
            
            # Extract bit
            bit = ((element & tvm.tir.const(masks[ib], "int32")) >> ib).astype("uint64")
            
            # Pack: k번째 channel의 bit를 k번째 bit position에
            packed = packed | (bit << k)
        
        return packed
    
    return te.compute(output_shape, _bitpack_compute, name="channel_bitpack_v2")


"""
=============================================================================
5. tuple 반환 버전: 4개의 uint64를 tuple로
=============================================================================

만약 마지막 dimension (4)를 tuple로 반환하고 싶다면:
"""

def custom_channel_bitpack_tuple(data, input_bits=8):
    """
    Tuple을 반환하는 버전: 4개의 separate tensors
    
    Returns
    -------
    output : tuple of 4 te.Tensors
        Each tensor has shape (N, C//256, H, W, input_bits)
    """
    N, C, H, W = data.shape
    
    assert C % 256 == 0, "C must be multiple of 256"
    
    C_packed = C // 256
    
    # Output shape WITHOUT the last dimension (4)
    # Because 4 will be represented as tuple
    output_shape = (N, C_packed, H, W, input_bits)
    
    masks = np.array([1 << i for i in range(input_bits)])
    
    def _bitpack_tuple(*indices):
        """
        Returns tuple of 4 uint64 values
        """
        n, c_group, h, w, ib = indices
        
        results = [tvm.tir.const(0, "uint64")] * 4
        
        # Pack 4 separate uint64 values
        for pack_idx in range(4):
            c_base = c_group * 256 + pack_idx * 64
            
            packed = tvm.tir.const(0, "uint64")
            
            for k in range(64):
                c = c_base + k
                element = data[n, c, h, w].astype("int32")
                bit = ((element & tvm.tir.const(masks[ib], "int32")) >> ib).astype("uint64")
                packed = packed | (bit << k)
            
            results[pack_idx] = packed
        
        # Return tuple!
        return tuple(results)
    
    # output_tuple is tuple of 4 tensors
    output_tuple = te.compute(output_shape, _bitpack_tuple, name="channel_bitpack_tuple")
    
    # Optional: concatenate along new axis to get single tensor
    # return topi.transform.concatenate(output_tuple, axis=-1)
    
    return output_tuple


"""
=============================================================================
6. 사용 예제
=============================================================================
"""

def test_custom_bitpack():
    import tvm
    from tvm import te
    
    print("="*70)
    print("Custom Channel Bitpack Test")
    print("="*70)
    
    # Input
    N, C, H, W = 2, 256, 4, 4
    input_bits = 8
    
    # Create input tensor
    data = te.placeholder((N, C, H, W), name="data", dtype="int32")
    
    # Version 1: Single tensor output
    print("\n[Version 1] Single tensor output")
    output_v1 = custom_channel_bitpack(data, input_bits)
    print(f"Output shape: {output_v1.shape}")
    print(f"Output dtype: {output_v1.dtype}")
    
    # Version 2: More efficient
    print("\n[Version 2] Efficient version")
    output_v2 = custom_channel_bitpack_v2(data, input_bits)
    print(f"Output shape: {output_v2.shape}")
    
    # Version 3: Tuple output
    print("\n[Version 3] Tuple output")
    output_tuple = custom_channel_bitpack_tuple(data, input_bits)
    print(f"Output: tuple of {len(output_tuple)} tensors")
    print(f"Each tensor shape: {output_tuple[0].shape}")
    
    # Build and test
    print("\n[Build] Creating schedule...")
    s = te.create_schedule(output_v2.op)
    
    # Optimize schedule
    n, c_group, h, w, ib, pack_idx = output_v2.op.axis
    s[output_v2].parallel(n)
    s[output_v2].parallel(c_group)
    
    print("Building function...")
    func = tvm.build(s, [data, output_v2], target="llvm", name="channel_bitpack")
    
    print("✓ Build successful!")
    
    # Run test
    print("\n[Test] Running with random data...")
    import numpy as np
    
    np.random.seed(42)
    data_np = np.random.randint(0, 256, size=(N, C, H, W), dtype=np.int32)
    
    dev = tvm.cpu()
    data_tvm = tvm.nd.array(data_np, dev)
    output_tvm = tvm.nd.array(
        np.zeros((N, C//256, H, W, input_bits, 4), dtype=np.uint64),
        dev
    )
    
    func(data_tvm, output_tvm)
    
    output_np = output_tvm.numpy()
    
    print(f"Output shape: {output_np.shape}")
    print(f"Sample output[0, 0, 0, 0, 0, :]: {output_np[0, 0, 0, 0, 0, :]}")
    
    # Verify correctness
    print("\n[Verify] Checking correctness...")
    
    # Check a few samples
    n, c_group, h, w, ib, pack_idx = 0, 0, 0, 0, 0, 0
    
    # Expected: pack channels [0:64] at bit position 0
    expected = 0
    for k in range(64):
        c = c_group * 256 + pack_idx * 64 + k
        element = data_np[n, c, h, w]
        bit = (element >> ib) & 1
        expected |= (bit << k)
    
    actual = output_np[n, c_group, h, w, ib, pack_idx]
    
    print(f"Expected: {expected:#018x}")
    print(f"Actual:   {actual:#018x}")
    
    if expected == actual:
        print("✓ Verification PASSED!")
    else:
        print("✗ Verification FAILED!")


"""
=============================================================================
7. 핵심 정리
=============================================================================

Q1: te.compute에서 tuple이 나올 수 있는 이유?
A1: TVM은 **multiple outputs**을 지원합니다.
    compute 함수가 tuple을 반환하면, 각 원소가 별도의 tensor가 됩니다.
    이는 여러 관련된 출력을 동시에 계산할 때 유용합니다.

Q2: [N,C,H,W]를 [N,C/256,H,W,IB,4] uint64로 만들려면?
A2: 위의 custom_channel_bitpack_v2() 함수를 사용하세요.
    - 256 channels을 64x4로 나눔
    - 각 IB (input bit)마다 bit 추출
    - 64개씩 packing하여 uint64 생성
    - 결과: (N, C/256, H, W, IB, 4) shape

Q3: Tuple vs Single Tensor?
A3: 둘 다 가능:
    - Single tensor: 마지막 dimension을 explicit하게
    - Tuple: 4개의 separate tensors, 나중에 concatenate 가능
    
    대부분의 경우 single tensor가 더 편리합니다.

Q4: 성능 최적화?
A4: Schedule을 통해:
    - s[output].parallel(): 병렬화
    - s[output].vectorize(): SIMD 활용
    - Auto-scheduler: 자동 튜닝
"""

if __name__ == "__main__":
    print(__doc__)
    
    print("\n" + "="*70)
    print("Running test...")
    print("="*70)
    
    test_custom_bitpack()
