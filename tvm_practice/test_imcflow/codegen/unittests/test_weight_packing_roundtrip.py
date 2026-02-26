"""
Round-trip test for qconv weight packing/unpacking.

Verifies that:
  pack (qconv_weight_transform) → unpack (WeightReverter)
produces identical weights for int4-range values [-8, 7].

Usage:
  cd /root/project/tvm/tvm_practice/test_imcflow/codegen
  python -m pytest unittests/test_weight_packing_roundtrip.py -v -s
"""

import numpy as np
import pytest


# ============================================================================
# Standalone pack / unpack extracted from layout.py and cpu_run.py
# (no Relay dependency — pure numpy for isolated testing)
# ============================================================================

def pack_qconv_weight(weight: np.ndarray) -> np.ndarray:
    """
    Pack int8 weight to uint32 packed format.
    Mirrors qconv_weight_transform() in layout.py.

    Args:
        weight: int8 array, shape (OC, IC, KH, KW)

    Returns:
        uint32 array, shape (out_blocks, in_blocks, 256, 8)
    """
    out_channels, in_channels, kh, kw = weight.shape
    ic = 256 // (kh * kw)
    spatial_elements = ic * kh * kw

    out_blocks = (out_channels + 63) // 64
    in_blocks = (in_channels + ic - 1) // ic

    padded_oc = out_blocks * 64
    padded_ic = in_blocks * ic

    # Step 1: Pad
    padded = np.zeros((padded_oc, padded_ic, kh, kw), dtype=np.int8)
    padded[:out_channels, :in_channels, :, :] = weight

    # Step 2: Reshape to (out_blocks, 64, in_blocks, ic, kh, kw)
    reshaped = padded.reshape(out_blocks, 64, in_blocks, ic, kh, kw)

    # Step 3: Transpose to (out_blocks, in_blocks, ic, kh, kw, 64)
    transposed = reshaped.transpose(0, 2, 3, 4, 5, 1)

    # Step 4: Flatten spatial → (out_blocks, in_blocks, spatial_elements, 64)
    flattened = transposed.reshape(out_blocks, in_blocks, spatial_elements, 64)

    # Step 5: Pad spatial dim to 256
    if spatial_elements < 256:
        padding = 256 - spatial_elements
        flattened = np.pad(flattened, ((0, 0), (0, 0), (0, padding), (0, 0)),
                          mode='constant', constant_values=0)

    # Step 6: Reshape for int4 packing → (out_blocks, in_blocks, 256, 8, 8)
    to_pack = flattened.reshape(out_blocks, in_blocks, 256, 8, 8)

    # Step 7: Pack 8 int4 values into uint32
    packed = np.zeros((out_blocks, in_blocks, 256, 8), dtype=np.uint32)
    for i in range(8):
        packed += (to_pack[:, :, :, :, i].astype(np.uint32) & 0xF) << (i * 4)

    return packed


def unpack_qconv_weight(packed: np.ndarray, OC: int, IC: int, KH: int, KW: int) -> np.ndarray:
    """
    Unpack uint32 packed weight back to int8.
    Mirrors WeightReverter in cpu_run.py.

    Args:
        packed: uint32 array, shape (out_blocks, in_blocks, 256, 8)
        OC, IC, KH, KW: original weight dimensions

    Returns:
        int8 array, shape (OC, IC, KH, KW)
    """
    out_blocks, in_blocks = packed.shape[0], packed.shape[1]
    ic = 256 // (KH * KW)
    spatial_elements = ic * KH * KW

    # Step 1: Unpack 8 int4 values from each uint32
    unpacked = np.zeros((out_blocks, in_blocks, 256, 8, 8), dtype=np.int8)
    for i in range(8):
        vals = (packed >> (i * 4)) & 0xF
        vals = vals.astype(np.int8)
        mask = vals >= 8
        vals[mask] -= 16
        unpacked[..., i] = vals

    # Step 2: Reshape → (out_blocks, in_blocks, 256, 64)
    padded = unpacked.reshape(out_blocks, in_blocks, 256, 64)

    # Step 3: Slice spatial
    if spatial_elements < 256:
        flattened = padded[:, :, :spatial_elements, :]
    else:
        flattened = padded

    # Step 4: Reshape → (out_blocks, in_blocks, ic, KH, KW, 64)
    transposed = flattened.reshape(out_blocks, in_blocks, ic, KH, KW, 64)

    # Step 5: Transpose → (out_blocks, 64, in_blocks, ic, KH, KW)
    reshaped = transposed.transpose(0, 5, 1, 2, 3, 4)

    # Step 6: Reshape → (padded_OC, padded_IC, KH, KW)
    padded_weight = reshaped.reshape(out_blocks * 64, in_blocks * ic, KH, KW)

    # Step 7: Crop
    return padded_weight[:OC, :IC, :, :]


# ============================================================================
# Tests
# ============================================================================

# Convolution configurations from resnet8_subset31
CONV_CONFIGS = [
    # (OC, IC, KH, KW, description)
    (16, 3, 3, 3, "first conv: 3->16, 3x3"),
    (16, 16, 3, 3, "basic block conv: 16->16, 3x3"),
    (32, 16, 3, 3, "downsample conv: 16->32, 3x3"),
    (32, 32, 3, 3, "basic block conv: 32->32, 3x3"),
    (64, 32, 3, 3, "downsample conv: 32->64, 3x3"),
    (64, 64, 3, 3, "basic block conv: 64->64, 3x3"),
    (10, 64, 1, 1, "final fc as conv: 64->10, 1x1"),
    # Edge cases
    (1, 1, 3, 3, "minimal: 1->1, 3x3"),
    (64, 64, 1, 1, "square block: 64->64, 1x1"),
    (65, 29, 3, 3, "non-aligned: 65->29, 3x3"),
]


@pytest.mark.parametrize("OC,IC,KH,KW,desc", CONV_CONFIGS)
def test_roundtrip_int4_range(OC, IC, KH, KW, desc):
    """Round-trip with values in int4 range [-8, 7] — must be exact."""
    np.random.seed(42)
    weight = np.random.randint(-8, 8, size=(OC, IC, KH, KW), dtype=np.int8)

    packed = pack_qconv_weight(weight)
    restored = unpack_qconv_weight(packed, OC, IC, KH, KW)

    np.testing.assert_array_equal(
        restored, weight,
        err_msg=f"Round-trip failed for {desc} ({OC},{IC},{KH},{KW})"
    )


@pytest.mark.parametrize("OC,IC,KH,KW,desc", CONV_CONFIGS)
def test_roundtrip_full_int8_clamped(OC, IC, KH, KW, desc):
    """
    Round-trip with full int8 range [-128, 127].
    Values outside [-8, 7] get truncated to 4 bits — verify truncation is consistent.
    """
    np.random.seed(42)
    weight = np.random.randint(-128, 128, size=(OC, IC, KH, KW), dtype=np.int8)

    packed = pack_qconv_weight(weight)
    restored = unpack_qconv_weight(packed, OC, IC, KH, KW)

    # The round-trip should match: original & 0xF → sign-extended back
    expected = (weight.astype(np.int32) & 0xF).astype(np.int8)
    mask = expected >= 8
    expected[mask] -= 16

    np.testing.assert_array_equal(
        restored, expected,
        err_msg=f"Truncation mismatch for {desc} ({OC},{IC},{KH},{KW})"
    )


@pytest.mark.parametrize("OC,IC,KH,KW,desc", CONV_CONFIGS)
def test_roundtrip_zeros(OC, IC, KH, KW, desc):
    """All-zeros weight — should be trivially exact."""
    weight = np.zeros((OC, IC, KH, KW), dtype=np.int8)

    packed = pack_qconv_weight(weight)
    restored = unpack_qconv_weight(packed, OC, IC, KH, KW)

    np.testing.assert_array_equal(restored, weight)


@pytest.mark.parametrize("OC,IC,KH,KW,desc", CONV_CONFIGS)
def test_roundtrip_identity_pattern(OC, IC, KH, KW, desc):
    """Linear index pattern mod 16, shifted to signed int4 range."""
    total = OC * IC * KH * KW
    weight = ((np.arange(total, dtype=np.int32) % 16) - 8).astype(np.int8)
    weight = weight.reshape(OC, IC, KH, KW)

    packed = pack_qconv_weight(weight)
    restored = unpack_qconv_weight(packed, OC, IC, KH, KW)

    np.testing.assert_array_equal(
        restored, weight,
        err_msg=f"Identity pattern failed for {desc}"
    )


def test_roundtrip_pretrained_weights():
    """
    Round-trip with actual pretrained weights from the checkpoint.
    Skipped if checkpoint is not available.
    """
    import os

    checkpoint_path = '/root/project/CIM/trained_models/image_classification/NAT/prange_full_psum_duplication_1/2025-Nov-20-18-05-24/imcflow/2026-Feb-13-10-40-47/checkpoint.pth.tar'
    if not os.path.exists(checkpoint_path):
        pytest.skip("Pretrained checkpoint not available")

    import torch
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']

    # Test each conv weight in the checkpoint
    failures = []
    for key, tensor in state_dict.items():
        if tensor.ndim != 4:
            continue
        if tensor.shape[2] not in (1, 3) or tensor.shape[3] not in (1, 3):
            continue

        weight_np = tensor.numpy()
        OC, IC, KH, KW = weight_np.shape

        # Clamp to int4 range (as the real pipeline does after quantization)
        weight_int4 = np.clip(weight_np, -8, 7).astype(np.int8)

        packed = pack_qconv_weight(weight_int4)
        restored = unpack_qconv_weight(packed, OC, IC, KH, KW)

        if not np.array_equal(restored, weight_int4):
            diff_count = np.sum(restored != weight_int4)
            failures.append(f"{key}: {diff_count} mismatches out of {weight_int4.size}")

    assert len(failures) == 0, "Round-trip failures:\n" + "\n".join(failures)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
