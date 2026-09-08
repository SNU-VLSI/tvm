#include <assert.h>
#include "dae_dataset_metrics.h"

int main(void) {
  float input[640], output[641];
  for (int i = 0; i < 640; i++) { input[i] = (float)i / 10; output[i + 1] = input[i] + 2; }
  output[0] = 999;
  int64_t shape[] = {1, 640};
  NpyDataset images = {0};
  images.ndim = 2;
  images.sample_numel = 640;
  images.sample_size = 640 * sizeof(float);
  images.dtype = (DLDataType){kDLFloat, 32, 1};
  DLTensor tensor = {0};
  tensor.data = output;
  tensor.dtype = images.dtype;
  tensor.ndim = 2;
  tensor.shape = shape;
  tensor.byte_offset = sizeof(float);
  double mse;
  assert(dae_window_mse(&images, input, &tensor, &mse) == 0 && fabs(mse - 4) < 1e-5);
  assert(dae_write_sample(stdout, 7, 1, &tensor, mse, 1) == 0);
  assert(dae_write_sample(stdout, 7, -1, &tensor, mse, 0) != 0);
  output[1] = NAN;
  assert(dae_window_mse(&images, input, &tensor, &mse) != 0);
  output[1] = INFINITY;
  assert(dae_window_mse(&images, input, &tensor, &mse) != 0);
  output[1] = 2;
  images.dtype.code = kDLInt;
  assert(dae_window_mse(&images, input, &tensor, &mse) != 0);
  images.dtype.code = kDLFloat;
  shape[1] = 639;
  assert(dae_window_mse(&images, input, &tensor, &mse) != 0);
  shape[1] = 640;
  int64_t strides[] = {1280, 2};
  tensor.strides = strides;
  assert(dae_window_mse(&images, input, &tensor, &mse) != 0);
  return 0;
}
