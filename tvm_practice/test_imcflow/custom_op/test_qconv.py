import numpy as np
import sys
import pytest
import tvm
import tvm.testing
from tvm import relay
from tvm.relay import transform
from tvm.contrib import graph_executor
from tvm.relay.op.nn.nn import imcflow_qconv2d
from tvm.relay.backend.contrib.imcflow.acim_util import ConfigData, ADCMode, VMode, MultMode, AccMask

# Add path to IMCU simulator
sys.path.insert(0, '/root/project/imcflow/pmap/ISA_sim/multi_core')
from imcflow_sim.imcflow.imce import IMCU

np.random.seed(0)

def transform_weight_for_imcu(weight_data, kH, kW, IC, OC):
  """
  Transform weight from (OC, IC, kH, kW) to (kH*kW*IC, OC, BW) layout for IMCU.

  Args:
    weight_data: numpy array of shape (OC, IC, kH, kW) with dtype int8
    kH, kW: kernel height and width
    IC, OC: input and output channels

  Returns:
    transformed_weight: numpy array of shape (kH*kW*IC, OC, 4) with dtype uint8
  """
  # Transpose from (OC, IC, kH, kW) to (kH, kW, IC, OC)
  weight_transposed = np.transpose(weight_data, (2, 3, 1, 0))

  # Reshape to (kH*kW*IC, OC)
  weight_reshaped = weight_transposed.reshape(kH * kW * IC, OC)

  # Convert to unsigned representation (4-bit values: -8 to 7 -> 0 to 15)
  # In 4-bit two's complement: -8=0b1000, -1=0b1111, 0=0b0000, 7=0b0111
  weight_unsigned = weight_reshaped.astype(np.int16) & 0xF

  # Extract bit planes (BW=4) in the same order as IMCU.write_to_imcu
  # Order is [b3, b2, b1, b0] where b3 is MSB
  bit_planes = np.zeros((kH * kW * IC, OC, 4), dtype=np.uint8)

  # Extract from MSB to LSB
  b3 = np.where(weight_unsigned >= 8, 1, 0).astype(np.uint8)
  weight_unsigned = weight_unsigned - b3 * 8
  b2 = np.where(weight_unsigned >= 4, 1, 0).astype(np.uint8)
  weight_unsigned = weight_unsigned - b2 * 4
  b1 = np.where(weight_unsigned >= 2, 1, 0).astype(np.uint8)
  weight_unsigned = weight_unsigned - b1 * 2
  b0 = np.where(weight_unsigned >= 1, 1, 0).astype(np.uint8)

  bit_planes[:, :, 0] = b3
  bit_planes[:, :, 1] = b2
  bit_planes[:, :, 2] = b1
  bit_planes[:, :, 3] = b0

  return bit_planes

def pad_and_extract_windows(input_data, N, IC, H, W, kH, kW, padding, stride):
  """
  Pad input and extract sliding windows for convolution.

  Args:
    input_data: numpy array of shape (N, IC, H, W) with dtype uint8
    N, IC, H, W: batch, input channels, height, width
    kH, kW: kernel height and width
    padding: padding size (assuming same padding for both H and W)
    stride: stride size (assuming same stride for both H and W)

  Returns:
    padded_input: numpy array of shape (N, IC, H+2*padding, W+2*padding)
    windows: list of flattened windows, each of shape (kH*kW*IC,)
    output_positions: list of (n, oh, ow) tuples indicating output positions
  """
  # Pad the input
  padded_input = np.pad(
    input_data,
    ((0, 0), (0, 0), (padding, padding), (padding, padding)),
    mode='constant',
    constant_values=0
  )

  # Calculate output dimensions
  OH = (H + 2 * padding - kH) // stride + 1
  OW = (W + 2 * padding - kW) // stride + 1

  windows = []
  output_positions = []

  for n in range(N):
    for oh in range(OH):
      for ow in range(OW):
        h_start = oh * stride
        w_start = ow * stride

        # Extract window: (IC, kH, kW)
        window = padded_input[n, :, h_start:h_start+kH, w_start:w_start+kW]

        # Transpose to (kH, kW, IC) and flatten to (kH*kW*IC,)
        window_transposed = np.transpose(window, (1, 2, 0))
        window_flat = window_transposed.reshape(-1)

        windows.append(window_flat)
        output_positions.append((n, oh, ow))

  return padded_input, windows, output_positions

def imcu_reference_conv2d(input_data, weight_data, N, IC, OC, H, W, kH, kW, padding, stride, adcmode=0, vmode=0, acc_mask=6):
  """
  Reference convolution implementation using IMCU.compute.

  Args:
    input_data: numpy array of shape (N, IC, H, W) with dtype uint8
    weight_data: numpy array of shape (OC, IC, kH, kW) with dtype int8
    N, IC, OC, H, W: batch, input/output channels, input height/width
    kH, kW: kernel height and width
    padding, stride: convolution parameters
    adcmode, vmode, acc_mask: IMCU compute parameters

  Returns:
    output: numpy array of shape (N, OC, OH, OW) with dtype int16
  """
  # Transform weight to IMCU format
  weight_imcu = transform_weight_for_imcu(weight_data, kH, kW, IC, OC)

  # Pad 256 - (kH*kW*IC) rows with zeros if needed
  rows_needed = 256
  current_rows = kH * kW * IC
  if current_rows < rows_needed:
    padding_rows = rows_needed - current_rows
    weight_imcu = np.pad(weight_imcu, ((0, padding_rows), (0, 0), (0, 0)), mode='constant', constant_values=0)
  
  # Pad 64 - OC columns with zeros if needed
  cols_needed = 64
  current_cols = OC
  if current_cols < cols_needed:
    padding_cols = cols_needed - current_cols
    weight_imcu = np.pad(weight_imcu, ((0, 0), (0, padding_cols), (0, 0)), mode='constant', constant_values=0)

  # Create IMCU instance and load weights
  imcu = IMCU()
  imcu.mem = weight_imcu

  # Extract input windows
  padded_input, windows, output_positions = pad_and_extract_windows(
    input_data, N, IC, H, W, kH, kW, padding, stride
  )

  # Calculate output dimensions
  OH = (H + 2 * padding - kH) // stride + 1
  OW = (W + 2 * padding - kW) // stride + 1

  # Initialize output tensor
  output = np.zeros((N, OC, OH, OW), dtype=np.int16)

  # Compute for each output position
  for window, (n, oh, ow) in zip(windows, output_positions):
    # Pad window to 256 elements if needed
    if len(window) < 256:
      window_padded = np.pad(window, (0, 256 - len(window)), mode='constant', constant_values=0)
    else:
      window_padded = window

    # Call IMCU.compute
    print(f"window_padded: {window_padded}, shape: {window_padded.shape}")
    result = imcu.compute(window_padded, adcmode, vmode, acc_mask)

    # Store result
    print(f"result: {result}, shape: {result.shape}")
    output[n, :, oh, ow] = result[:OC]  # Trim to original OC size

  return output

@pytest.mark.parametrize("acc_mask", [AccMask.BM_0000, AccMask.BM_1111])
def test_imcflow_qconv2d(acc_mask):
  N, H, W = 1, 1, 1
  IC, OC = 16, 16
  kH, kW = 3, 3
  padding, stride = 1, 1

  input_ = relay.var("input", shape=(N, IC, H, W), dtype="uint8")
  weight_ = relay.var("weight", shape=(OC, IC, kH, kW), dtype="int8")

  # Create ConfigData object
  config_data = ConfigData((N, IC, H, W), (OC, IC, kH, kW),
                           padding=padding, stride=stride,
                           acc_mask=acc_mask)

  # Extract parameters from config_data for Relay op attributes
  config_ = config_data.get_as_const_tensor()

  y = imcflow_qconv2d(
    input_,
    weight_,
    config_,
    channels=OC,
    in_channels=IC,
    kernel_size=(kH, kW),
    padding=(padding, padding),
    out_dtype="int16"
  )

  func = relay.Function([input_, weight_], y)
  mod = tvm.IRModule.from_expr(func)
  mod = transform.InferType()(mod)

  print(mod)

  target = "llvm"
  ctx = tvm.cpu(0)

  with tvm.transform.PassContext(opt_level=0):
    graph, lib, params = relay.build(mod, target=target)
  mod = graph_executor.create(graph, lib, device=ctx)
  # Load constant parameters (config)
  if params:
    mod.load_params(tvm.runtime.save_param_dict(params))


  # input_data = np.random.randint(0, 16, size=(1, 28, 1, 1)).astype("uint8")
  # weight_data = np.random.randint(-8, 7, size=(64, 28, 3, 3)).astype("int8")
  input_data = np.zeros((1, 16, 1, 1)).astype("uint8")
  # non_zero_indices = [0,1,2,4,5,6, 7,10,11,13,14,15,16]
  non_zero_indices = [0,1,2,4,5,6, 7,10,11,13,14,15]
  non_zero_values = [5,7,13,3,6,7,10, 3, 4, 7, 4, 4]
  for idx, val in zip(non_zero_indices, non_zero_values):
    input_data[0, idx, 0, 0] = val
  # [ 0  0  0  0  5  0  0  0  0  0  0  0  0  7  0  0  0  0  0  0  0  0 13  0
  # 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  3  0  0  0  0  0  0  0
  # 0  6  0  0  0  0  0  0  0  0  7  0  0  0  0  0  0  0  0 10  0  0  0  0
  # 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  3  0
  # 0  0  0  0  0  0  0  4  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
  # 0  7  0  0  0  0  0  0  0  0  4  0  0  0  0  0  0  0  0  4  0  0  0  0
  # 0  0  0  0  7  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
  # 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
  # 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
  # 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
  # 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
  weight_data = np.ones((16, 16, 3, 3)).astype("int8")

  mod.set_input(input=input_data)
  mod.set_input(weight=weight_data)
  mod.run()

  res = mod.get_output(0).asnumpy()

  # Use IMCU reference implementation
  ref_res = imcu_reference_conv2d(
    input_data, weight_data,
    N=input_data.shape[0], IC=IC, OC=OC,
    H=input_data.shape[2], W=input_data.shape[3],
    kH=kH, kW=kW,
    padding=padding, stride=stride,
    adcmode=0, vmode=0, acc_mask=acc_mask.value
  )

  print("TVM output:", res)
  print("IMCU reference output:", ref_res)

  # Simple assertion instead of tvm.testing.assert_allclose
  np.testing.assert_allclose(res, ref_res, atol=1e-5, rtol=1e-5)
  print("Test passed!")

if __name__ == "__main__":
  tvm.testing.main()