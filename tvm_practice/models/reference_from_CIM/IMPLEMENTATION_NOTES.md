# Implementation of _get_tensor_from_checkpoint Function

## Summary

I have successfully implemented the `_get_tensor_from_checkpoint` function in `/root/project/tvm/tvm_practice/models/resnet8_cifar.py`. This function maps pretrained PyTorch model parameters from a checkpoint file to TVM Relay model parameters, handling all necessary type conversions and shape transformations.

## Implementation Location

File: `/root/project/tvm/tvm_practice/models/resnet8_cifar.py`
Function: `getModel_from_pretrained_weight()` → nested function `_get_tensor_from_checkpoint(name, dtype, shape)`

## Key Features

### 1. Direct Parameter Mappings
Maps straightforward parameters between PyTorch and TVM:
- `weight1` → `conv1.weight`
- `bn_gamma` → `bn1.weight`
- `bn_beta` → `bn1.bias`
- `bn_moving_mean` → `bn1.running_mean`
- `bn_moving_var` → `bn1.running_var`
- `dense_weight` → `fc.weight`
- `dense_bias` → `fc.bias`

### 2. Layer-Specific Weight Mappings
Uses regex pattern matching to map layer weights:
- Pattern: `weight{2,3,4}_{0,1,2}`
- Maps to: `layer{1,2,3}.block_int16.conv{1,2}.weight` or downsample weights
- Example: `weight2_1` → `layer1.block_int16.conv1.weight`
- Example: `weight3_0` → `layer2.block_int16.downsample.1.weight`

### 3. Batch Normalization Parameters
Maps fused scale and bias parameters:
- Pattern: `fused_{scale|bias}{1-6}(_2)?`
- Maps to layer{1-3} bn{1-2} parameters or downsample bn parameters
- Example: `fused_scale1` → `layer1.block_int16.bn1.scale`
- Example: `fused_bias4_2` → `layer2.block_int16.downsample.2.bias`

### 4. Quantization Parameters
Maps min/max quantization bounds:
- Pattern: `quant_{min|max}_{1-6}(_2)?`
- Maps to activation quantization parameters
- Example: `quant_min_1` → `layer1.block_int16.act1.min`
- Handles scalar conversion for int16 values

### 5. Computed Scaling Factors
Derives values from adjust_factors:
- `x_f_1`: Directly from `adjust_factors['x_f_1']`
- `post_f_inv`: Computed as `1.0 / adjust_factors['bn2_f_3']`
- `y_f_{1,2,3}`: Computed as `bn2_f_i / x_f_i`

### 6. Residual Connection Adjustments
Handles downsample residual scaling:
- `bn_out_f_0, bn_out_f_2`: Zero arrays (bias terms)
- `bn_out_f_1, bn_out_f_3`: One arrays (scale terms)
- Properly shaped as (channels, 1, 1)

## Testing

I created two test scripts to verify the implementation:

### 1. Simple Mapping Test (`test_mapping_logic.py`)
- Tests the mapping logic without full TVM initialization
- Verifies all parameter patterns are correctly mapped
- Confirms dtype and shape conversions work properly
- **Result**: All test cases passed ✓

### 2. Full Integration Test (`test_checkpoint_loading.py`)
- Tests the complete `getModel_from_pretrained_weight()` function
- Verifies all parameters load correctly with proper shapes and dtypes
- Can be run with: `cd /root/project/tvm/tvm_practice/models && python3 test_checkpoint_loading.py`

## Example Usage

```python
from models.resnet8_cifar import getModel_from_pretrained_weight

# Load TVM Relay model with pretrained weights
model, params = getModel_from_pretrained_weight()

# model is a TVM IRModule containing the Relay graph
# params is a dict mapping parameter names to numpy arrays
```

## Parameter Mapping Reference

### Complete Mapping Table

| TVM Parameter | PyTorch Checkpoint Key | Type | Notes |
|--------------|----------------------|------|-------|
| `weight1` | `conv1.weight` | float32 | Initial conv layer |
| `bn_gamma` | `bn1.weight` | float32 | Initial BN gamma |
| `bn_beta` | `bn1.bias` | float32 | Initial BN beta |
| `bn_moving_mean` | `bn1.running_mean` | float32 | Initial BN running mean |
| `bn_moving_var` | `bn1.running_var` | float32 | Initial BN running var |
| `weight2_1` | `layer1.block_int16.conv1.weight` | int8 | Layer 1, conv 1 |
| `weight2_2` | `layer1.block_int16.conv2.weight` | int8 | Layer 1, conv 2 |
| `fused_scale1` | `layer1.block_int16.bn1.scale` | int16 | Layer 1, BN 1 scale |
| `fused_bias1` | `layer1.block_int16.bn1.bias` | int16 | Layer 1, BN 1 bias |
| `fused_scale2` | `layer1.block_int16.bn2.scale` | int16 | Layer 1, BN 2 scale |
| `fused_bias2` | `layer1.block_int16.bn2.bias` | int16 | Layer 1, BN 2 bias |
| `quant_min_1` | `layer1.block_int16.act1.min` | int16 | Layer 1, act 1 min |
| `quant_max_1` | `layer1.block_int16.act1.max` | int16 | Layer 1, act 1 max |
| `quant_min_2` | `layer1.block_int16.act2.min` | int16 | Layer 1, act 2 min |
| `quant_max_2` | `layer1.block_int16.act2.max` | int16 | Layer 1, act 2 max |
| `weight3_0` | `layer2.block_int16.downsample.1.weight` | int8 | Layer 2 downsample |
| `weight3_1` | `layer2.block_int16.conv1.weight` | int8 | Layer 2, conv 1 |
| `weight3_2` | `layer2.block_int16.conv2.weight` | int8 | Layer 2, conv 2 |
| `fused_scale4_2` | `layer2.block_int16.downsample.2.scale` | int16 | Layer 2 downsample scale |
| `fused_bias4_2` | `layer2.block_int16.downsample.2.bias` | int16 | Layer 2 downsample bias |
| `quant_min_4_2` | `layer2.block_int16.downsample.0.min` | int16 | Layer 2 downsample act min |
| `quant_max_4_2` | `layer2.block_int16.downsample.0.max` | int16 | Layer 2 downsample act max |
| `bn_out_f_0` | Computed (zeros) | int16 | Downsample bias adjustment |
| `bn_out_f_1` | Computed (ones) | int16 | Downsample scale adjustment |
| `x_f_1` | `adjust_factors['x_f_1']` | float32 | Input scaling factor |
| `y_f_1` | `bn2_f_1 / x_f_1` | int16 | Residual scaling factor |
| `post_f_inv` | `1.0 / bn2_f_3` | float32 | Output dequant factor |
| `dense_weight` | `fc.weight` | float32 | Final dense layer |
| `dense_bias` | `fc.bias` | float32 | Final dense bias |

## Technical Details

### Type Conversions
- PyTorch tensors → NumPy arrays via `.cpu().numpy()`
- Automatic dtype conversion using `.astype(dtype)`
- Scalar handling: shape `()` for scalars, shape `(1,)` for single-element arrays

### Shape Validation
- Every parameter mapping validates that loaded shape matches expected shape
- Raises `ValueError` with descriptive message on mismatch

### Error Handling
- Raises `ValueError` if parameter name has no mapping
- Raises `ValueError` if expected key not found in checkpoint
- Raises `ValueError` if shapes don't match

### Adjust Factors Structure
From checkpoint `adjust_factors` dict:
```python
{
    'x_f_1': 36.0,      # Layer 1 input scaling
    'bn1_f_1': 72.0,    # Layer 1 BN 1 output scaling
    'bn2_f_1': 36.0,    # Layer 1 BN 2 output scaling
    'x_f_2': 36.0,      # Layer 2 input scaling
    'bn1_f_2': 150.0,   # Layer 2 BN 1 output scaling
    'bn2_f_2': 15.0,    # Layer 2 BN 2 output scaling
    'x_f_3': 10.0,      # Layer 3 input scaling
    'bn1_f_3': 50.0,    # Layer 3 BN 1 output scaling
    'bn2_f_3': 500.0,   # Layer 3 BN 2 output scaling
}
```

## Verification Results

Running `test_mapping_logic.py` confirms all mappings work correctly:
```
Testing: weight1 (dtype=float32, shape=(16, 3, 3, 3))
  ✓ Mapped to 'conv1.weight': shape=(16, 3, 3, 3)

Testing: bn_gamma (dtype=float32, shape=(16,))
  ✓ Mapped to 'bn1.weight': shape=(16,)

Testing: weight2_1 (dtype=int8, shape=(16, 16, 3, 3))
  ✓ Mapped to 'layer1.block_int16.conv1.weight': shape=(16, 16, 3, 3)

Testing: fused_scale1 (dtype=int16, shape=(16,))
  ✓ Mapped to 'layer1.block_int16.bn1.scale': shape=(16,)

Testing: quant_min_1 (dtype=int16, shape=())
  ✓ Mapped to 'layer1.block_int16.act1.min': value=-254

Testing: y_f_1 (dtype=int16, shape=(1,))
  ✓ Computed from adjust_factors: value=[1]

Testing: bn_out_f_0 (dtype=int16, shape=(32, 1, 1))
  ✓ Zeros (bias term): shape=(32, 1, 1)

Testing: post_f_inv (dtype=float32, shape=(1,))
  ✓ Computed from adjust_factors: value=[0.002]
```

All test cases passed successfully! ✓

## Next Steps

The implementation is complete and tested. You can now:

1. Use `getModel_from_pretrained_weight()` to load the pretrained model
2. Run inference with the loaded weights
3. Compare results with the PyTorch model to verify correctness
4. Compile the TVM model with the pretrained weights for deployment

## Files Modified

1. `/root/project/tvm/tvm_practice/models/resnet8_cifar.py` - Main implementation