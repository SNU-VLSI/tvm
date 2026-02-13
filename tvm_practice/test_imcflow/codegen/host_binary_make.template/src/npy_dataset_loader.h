/**
 * NPY Dataset Loader
 *
 * Helper functions for loading NumPy .npy dataset files (e.g., CIFAR-10).
 * Supports loading multi-dimensional arrays and extracting individual samples.
 *
 * Pure C implementation - no C++ dependencies.
 */

#ifndef NPY_DATASET_LOADER_H
#define NPY_DATASET_LOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <dlpack/dlpack.h>

/**
 * NPY Dataset structure
 *
 * Holds the entire dataset loaded from a .npy file.
 */
typedef struct {
  void* data;           // Entire data buffer
  int64_t* shape;       // Shape array (e.g., [10000, 3, 32, 32])
  int ndim;             // Number of dimensions
  DLDataType dtype;     // Data type
  size_t num_samples;   // First dimension (number of samples)
  size_t sample_size;   // Size of one sample in bytes
  size_t sample_numel;  // Number of elements in one sample
} NpyDataset;

/**
 * Parse NumPy dtype string to DLDataType
 *
 * @param dtype_str NumPy dtype string (e.g., "<f4", "<i8", ">u1")
 * @param out_dtype Output DLDataType
 * @return 0 on success, -1 on failure
 */
static int parse_numpy_dtype(const char* dtype_str, DLDataType* out_dtype) {
  if (strlen(dtype_str) < 2) {
    fprintf(stderr, "Invalid dtype string: %s\n", dtype_str);
    return -1;
  }

  // Skip endianness marker ('|', '<', '>')
  const char* p = dtype_str;
  if (*p == '<' || *p == '>' || *p == '|') {
    p++;
  }

  // Parse type character
  char type_char = *p++;
  switch (type_char) {
    case 'f':  // float
      out_dtype->code = kDLFloat;
      break;
    case 'i':  // signed int
      out_dtype->code = kDLInt;
      break;
    case 'u':  // unsigned int
      out_dtype->code = kDLUInt;
      break;
    default:
      fprintf(stderr, "Unsupported numpy dtype: %s\n", dtype_str);
      return -1;
  }

  // Parse byte size
  int bytes = atoi(p);
  out_dtype->bits = bytes * 8;
  out_dtype->lanes = 1;

  return 0;
}

/**
 * Parse shape tuple from NumPy header string
 *
 * @param header Full header string containing 'shape': (...)
 * @param shape Output shape array (must be pre-allocated)
 * @param max_ndim Maximum dimensions to parse
 * @param out_ndim Output: actual number of dimensions
 * @return 0 on success, -1 on failure
 */
static int parse_numpy_shape(const char* header, int64_t* shape,
                              int max_ndim, int* out_ndim) {
  // Find 'shape': (...)
  const char* shape_key = strstr(header, "'shape':");
  if (!shape_key) {
    shape_key = strstr(header, "\"shape\":");
  }
  if (!shape_key) {
    fprintf(stderr, "Failed to find 'shape' in header\n");
    return -1;
  }

  // Find opening parenthesis
  const char* p = shape_key + 8;  // Skip "'shape':"
  while (*p && *p != '(') p++;
  if (*p != '(') {
    fprintf(stderr, "Failed to find shape tuple\n");
    return -1;
  }
  p++;  // Skip '('

  // Parse shape values
  int ndim = 0;
  while (*p && *p != ')' && ndim < max_ndim) {
    // Skip whitespace
    while (*p && isspace(*p)) p++;

    // Parse number
    if (isdigit(*p)) {
      shape[ndim++] = atoll(p);
      // Skip past number
      while (*p && (isdigit(*p))) p++;
    }

    // Skip comma and whitespace
    while (*p && (*p == ',' || isspace(*p))) p++;
  }

  *out_ndim = ndim;
  return 0;
}

/**
 * Parse dtype from NumPy header string
 *
 * @param header Full header string containing 'descr': '...'
 * @param out_dtype Output DLDataType
 * @return 0 on success, -1 on failure
 */
static int parse_numpy_descr(const char* header, DLDataType* out_dtype) {
  // Find 'descr': '...'
  const char* descr_key = strstr(header, "'descr':");
  if (!descr_key) {
    descr_key = strstr(header, "\"descr\":");
  }
  if (!descr_key) {
    fprintf(stderr, "Failed to find 'descr' in header\n");
    return -1;
  }

  // Find opening quote
  const char* p = descr_key + 8;  // Skip "'descr':"
  while (*p && *p != '\'' && *p != '"') p++;
  if (!*p) {
    fprintf(stderr, "Failed to find dtype string\n");
    return -1;
  }
  char quote_char = *p++;

  // Extract dtype string
  char dtype_str[16];
  int i = 0;
  while (*p && *p != quote_char && i < 15) {
    dtype_str[i++] = *p++;
  }
  dtype_str[i] = '\0';

  return parse_numpy_dtype(dtype_str, out_dtype);
}

/**
 * Load NPY dataset from file
 *
 * @param path Path to .npy file
 * @param dataset Output NpyDataset structure (will be allocated)
 * @return 0 on success, -1 on failure
 */
static int load_npy_dataset(const char* path, NpyDataset* dataset) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "Failed to open NPY file: %s\n", path);
    return -1;
  }

  // Read and verify magic number
  unsigned char magic[6];
  if (fread(magic, 1, 6, f) != 6) {
    fprintf(stderr, "Failed to read magic number\n");
    fclose(f);
    return -1;
  }

  if (magic[0] != 0x93 || magic[1] != 'N' || magic[2] != 'U' ||
      magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') {
    fprintf(stderr, "Invalid NPY magic number\n");
    fclose(f);
    return -1;
  }

  // Read version
  unsigned char version[2];
  if (fread(version, 1, 2, f) != 2) {
    fprintf(stderr, "Failed to read version\n");
    fclose(f);
    return -1;
  }

  // Read header length
  uint32_t header_len = 0;
  if (version[0] == 1) {
    // Version 1.0: 2-byte header length
    uint16_t hlen16;
    if (fread(&hlen16, sizeof(uint16_t), 1, f) != 1) {
      fprintf(stderr, "Failed to read header length\n");
      fclose(f);
      return -1;
    }
    header_len = hlen16;
  } else if (version[0] == 2 || version[0] == 3) {
    // Version 2.0/3.0: 4-byte header length
    if (fread(&header_len, sizeof(uint32_t), 1, f) != 1) {
      fprintf(stderr, "Failed to read header length\n");
      fclose(f);
      return -1;
    }
  } else {
    fprintf(stderr, "Unsupported NPY version: %d.%d\n", version[0], version[1]);
    fclose(f);
    return -1;
  }

  // Read header
  char* header = (char*)malloc(header_len + 1);
  if (!header) {
    fprintf(stderr, "Failed to allocate header buffer\n");
    fclose(f);
    return -1;
  }

  if (fread(header, 1, header_len, f) != header_len) {
    fprintf(stderr, "Failed to read header\n");
    free(header);
    fclose(f);
    return -1;
  }
  header[header_len] = '\0';

  // Parse shape
  int64_t shape[8];
  int ndim = 0;
  if (parse_numpy_shape(header, shape, 8, &ndim) != 0) {
    free(header);
    fclose(f);
    return -1;
  }

  // Parse dtype
  DLDataType dtype;
  if (parse_numpy_descr(header, &dtype) != 0) {
    free(header);
    fclose(f);
    return -1;
  }

  free(header);

  // Calculate total size
  size_t numel = 1;
  for (int i = 0; i < ndim; i++) {
    numel *= (size_t)shape[i];
  }
  size_t elem_size = (dtype.bits + 7) / 8;
  size_t total_bytes = numel * elem_size;

  // Calculate sample size (everything except first dimension)
  size_t sample_numel = 1;
  for (int i = 1; i < ndim; i++) {
    sample_numel *= (size_t)shape[i];
  }
  size_t sample_size = sample_numel * elem_size;

  // Allocate data buffer
  void* data = malloc(total_bytes);
  if (!data) {
    fprintf(stderr, "Failed to allocate %zu bytes for data\n", total_bytes);
    fclose(f);
    return -1;
  }

  // Read data
  size_t bytes_read = fread(data, 1, total_bytes, f);
  fclose(f);

  if (bytes_read != total_bytes) {
    fprintf(stderr, "Expected %zu bytes, read %zu bytes\n", total_bytes, bytes_read);
    free(data);
    return -1;
  }

  // Fill dataset structure
  dataset->data = data;
  dataset->ndim = ndim;
  dataset->dtype = dtype;
  dataset->num_samples = (size_t)shape[0];
  dataset->sample_size = sample_size;
  dataset->sample_numel = sample_numel;

  // Copy shape
  dataset->shape = (int64_t*)malloc(ndim * sizeof(int64_t));
  memcpy(dataset->shape, shape, ndim * sizeof(int64_t));

  fprintf(stderr, "Loaded NPY dataset: %s\n", path);
  fprintf(stderr, "  Shape: [");
  for (int i = 0; i < ndim; i++) {
    fprintf(stderr, "%lld%s", (long long)shape[i], i < ndim - 1 ? ", " : "");
  }
  fprintf(stderr, "]\n");

  const char* dtype_name = "unknown";
  if (dtype.code == kDLFloat) dtype_name = "float";
  else if (dtype.code == kDLInt) dtype_name = "int";
  else if (dtype.code == kDLUInt) dtype_name = "uint";
  fprintf(stderr, "  Dtype: %s%d\n", dtype_name, dtype.bits);
  fprintf(stderr, "  Samples: %zu, Sample size: %zu bytes\n",
          dataset->num_samples, dataset->sample_size);

  return 0;
}

/**
 * Get pointer to a specific sample in the dataset
 *
 * @param dataset NpyDataset structure
 * @param index Sample index (0-based)
 * @return Pointer to sample data, or NULL if index out of range
 */
static void* get_sample(const NpyDataset* dataset, size_t index) {
  if (index >= dataset->num_samples) {
    fprintf(stderr, "Sample index %zu out of range (max: %zu)\n",
            index, dataset->num_samples - 1);
    return NULL;
  }
  return (char*)dataset->data + index * dataset->sample_size;
}

/**
 * Get label value from labels dataset
 *
 * @param dataset Labels NpyDataset (typically 1D array)
 * @param index Sample index (0-based)
 * @return Label value as int64_t
 */
static int64_t get_label(const NpyDataset* dataset, size_t index) {
  if (index >= dataset->num_samples) {
    fprintf(stderr, "Label index %zu out of range (max: %zu)\n",
            index, dataset->num_samples - 1);
    return -1;
  }

  // Handle different int types
  if (dataset->dtype.code == kDLInt) {
    switch (dataset->dtype.bits) {
      case 8:
        return (int64_t)((int8_t*)dataset->data)[index];
      case 16:
        return (int64_t)((int16_t*)dataset->data)[index];
      case 32:
        return (int64_t)((int32_t*)dataset->data)[index];
      case 64:
        return ((int64_t*)dataset->data)[index];
    }
  } else if (dataset->dtype.code == kDLUInt) {
    switch (dataset->dtype.bits) {
      case 8:
        return (int64_t)((uint8_t*)dataset->data)[index];
      case 16:
        return (int64_t)((uint16_t*)dataset->data)[index];
      case 32:
        return (int64_t)((uint32_t*)dataset->data)[index];
      case 64:
        return (int64_t)((uint64_t*)dataset->data)[index];
    }
  }

  fprintf(stderr, "Unsupported label dtype\n");
  return -1;
}

/**
 * Free NpyDataset resources
 *
 * @param dataset NpyDataset to free
 */
static void free_npy_dataset(NpyDataset* dataset) {
  if (dataset->data) {
    free(dataset->data);
    dataset->data = NULL;
  }
  if (dataset->shape) {
    free(dataset->shape);
    dataset->shape = NULL;
  }
}

/**
 * Compute argmax for float32 array
 *
 * @param data Float32 array
 * @param size Number of elements
 * @return Index of maximum value
 */
static int argmax_float32(const float* data, int size) {
  int max_idx = 0;
  float max_val = data[0];
  for (int i = 1; i < size; i++) {
    if (data[i] > max_val) {
      max_val = data[i];
      max_idx = i;
    }
  }
  return max_idx;
}

/**
 * Compute argmax for int32 array
 *
 * @param data Int32 array
 * @param size Number of elements
 * @return Index of maximum value
 */
static int argmax_int32(const int32_t* data, int size) {
  int max_idx = 0;
  int32_t max_val = data[0];
  for (int i = 1; i < size; i++) {
    if (data[i] > max_val) {
      max_val = data[i];
      max_idx = i;
    }
  }
  return max_idx;
}

/**
 * Compute argmax for int8 array
 *
 * @param data Int8 array
 * @param size Number of elements
 * @return Index of maximum value
 */
static int argmax_int8(const int8_t* data, int size) {
  int max_idx = 0;
  int8_t max_val = data[0];
  for (int i = 1; i < size; i++) {
    if (data[i] > max_val) {
      max_val = data[i];
      max_idx = i;
    }
  }
  return max_idx;
}

#endif  // NPY_DATASET_LOADER_H
