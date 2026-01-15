// Helper functions for loading test inputs from binary files
// This allows sharing test inputs between Python and C code

#ifndef TEST_INPUT_LOADER_H
#define TEST_INPUT_LOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlpack/dlpack.h>

/**
 * Load tensor data from a binary file.
 * 
 * @param path Path to the .bin file
 * @param tensor DLTensor with pre-allocated data buffer
 * @return 0 on success, -1 on failure
 */
static int load_tensor_from_bin(const char* path, DLTensor* tensor) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "Failed to open input file: %s\n", path);
    return -1;
  }
  
  // Calculate expected size
  size_t numel = 1;
  for (int i = 0; i < tensor->ndim; ++i) {
    numel *= (size_t)tensor->shape[i];
  }
  
  size_t elem_size = 0;
  switch (tensor->dtype.code) {
    case kDLInt:
      elem_size = (tensor->dtype.bits + 7) / 8;
      break;
    case kDLUInt:
      elem_size = (tensor->dtype.bits + 7) / 8;
      break;
    case kDLFloat:
      elem_size = (tensor->dtype.bits + 7) / 8;
      break;
    default:
      fprintf(stderr, "Unsupported dtype code: %d\n", tensor->dtype.code);
      fclose(f);
      return -1;
  }
  
  size_t expected_bytes = numel * elem_size;
  
  // Read data
  size_t bytes_read = fread(tensor->data, 1, expected_bytes, f);
  fclose(f);
  
  if (bytes_read != expected_bytes) {
    fprintf(stderr, "Expected %zu bytes, but read %zu bytes from %s\n",
            expected_bytes, bytes_read, path);
    return -1;
  }
  
  fprintf(stderr, "Loaded %zu bytes from %s\n", bytes_read, path);
  return 0;
}

/**
 * Parse metadata from metadata file.
 *
 * @param meta_path Path to .meta.txt file
 * @param shape Output array for shape (must be pre-allocated)
 * @param max_ndim Maximum number of dimensions
 * @param out_ndim Output: actual number of dimensions
 * @param out_dtype Output: DLDataType parsed from dtype field
 * @return 0 on success, -1 on failure
 */
static int parse_meta(const char* meta_path, int64_t* shape,
                      int max_ndim, int* out_ndim, DLDataType* out_dtype) {
  FILE* f = fopen(meta_path, "r");
  if (!f) {
    fprintf(stderr, "Failed to open metadata file: %s\n", meta_path);
    return -1;
  }

  char line[256];
  int found_shape = 0;
  int found_dtype = 0;

  while (fgets(line, sizeof(line), f)) {
    if (strncmp(line, "shape: ", 7) == 0) {
      // Parse comma-separated shape values
      char* shape_str = line + 7;
      int ndim = 0;
      char* token = strtok(shape_str, ",");
      while (token && ndim < max_ndim) {
        shape[ndim++] = atoll(token);
        token = strtok(NULL, ",");
      }
      *out_ndim = ndim;
      found_shape = 1;
    } else if (strncmp(line, "dtype: ", 7) == 0) {
      // Parse dtype string (e.g., "int8", "uint8", "float32")
      char dtype_str[32];
      sscanf(line + 7, "%31s", dtype_str);

      // Parse dtype code and bits
      if (strncmp(dtype_str, "int", 3) == 0) {
        out_dtype->code = kDLInt;
        out_dtype->bits = atoi(dtype_str + 3);
      } else if (strncmp(dtype_str, "uint", 4) == 0) {
        out_dtype->code = kDLUInt;
        out_dtype->bits = atoi(dtype_str + 4);
      } else if (strncmp(dtype_str, "float", 5) == 0) {
        out_dtype->code = kDLFloat;
        out_dtype->bits = atoi(dtype_str + 5);
      } else {
        fprintf(stderr, "Unsupported dtype: %s\n", dtype_str);
        fclose(f);
        return -1;
      }
      out_dtype->lanes = 1;
      found_dtype = 1;
    }
  }

  fclose(f);

  if (!found_shape) {
    fprintf(stderr, "Shape not found in metadata file: %s\n", meta_path);
    return -1;
  }

  if (!found_dtype) {
    fprintf(stderr, "Dtype not found in metadata file: %s\n", meta_path);
    return -1;
  }

  return 0;
}

/**
 * Load tensor from directory (reads both .bin and .meta.txt).
 * 
 * @param input_dir Directory containing input files
 * @param input_name Name of input (without extension)
 * @param device DLDevice for memory allocation
 * @param out_tensor Output tensor (will be allocated and filled)
 * @return 0 on success, -1 on failure
 */
static int load_tensor_from_dir(const char* input_dir, const char* input_name,
                                 DLDevice device, DLTensor* out_tensor) {
  // Construct paths
  char bin_path[512];
  char meta_path[512];
  snprintf(bin_path, sizeof(bin_path), "%s/%s.bin", input_dir, input_name);
  snprintf(meta_path, sizeof(meta_path), "%s/%s.meta.txt", input_dir, input_name);
  
  // Parse metadata
  int64_t shape[8];  // Support up to 8 dimensions
  int ndim = 0;
  DLDataType dtype;
  if (parse_meta(meta_path, shape, 8, &ndim, &dtype) != 0) {
    return -1;
  }

  fprintf(stderr, "Parsed shape: [");
  for (int i = 0; i < ndim; ++i) {
    fprintf(stderr, "%lld%s", (long long)shape[i], i < ndim - 1 ? ", " : "");
  }
  fprintf(stderr, "]\n");

  const char* dtype_name = "unknown";
  if (dtype.code == kDLInt) dtype_name = "int";
  else if (dtype.code == kDLUInt) dtype_name = "uint";
  else if (dtype.code == kDLFloat) dtype_name = "float";
  fprintf(stderr, "Parsed dtype: %s%d\n", dtype_name, dtype.bits);

  // Allocate tensor
  out_tensor->device = device;
  out_tensor->ndim = ndim;

  // Allocate shape array
  out_tensor->shape = (int64_t*)malloc(ndim * sizeof(int64_t));
  memcpy(out_tensor->shape, shape, ndim * sizeof(int64_t));

  out_tensor->dtype = dtype;

  // Calculate size and allocate data
  size_t numel = 1;
  for (int i = 0; i < ndim; ++i) {
    numel *= (size_t)shape[i];
  }
  size_t elem_bytes = (dtype.bits + 7) / 8;
  size_t nbytes = numel * elem_bytes;
  
  if (TVMPlatformMemoryAllocate(nbytes, device, &out_tensor->data) != 0) {
    fprintf(stderr, "Failed to allocate %zu bytes\n", nbytes);
    free(out_tensor->shape);
    return -1;
  }
  
  // Load data from binary file
  if (load_tensor_from_bin(bin_path, out_tensor) != 0) {
    TVMPlatformMemoryFree(out_tensor->data, device);
    free(out_tensor->shape);
    return -1;
  }
  
  return 0;
}

#endif  // TEST_INPUT_LOADER_H
