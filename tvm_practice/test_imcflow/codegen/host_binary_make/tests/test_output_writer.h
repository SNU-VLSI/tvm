// Helper functions for saving test outputs to NumPy .npy format
// This allows sharing test outputs between C code and Python validation
// Pure C implementation - no C++ dependencies

#ifndef TEST_OUTPUT_WRITER_H
#define TEST_OUTPUT_WRITER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlpack/dlpack.h>

/**
 * Get NumPy dtype string for DLDataType
 *
 * @param dtype DLDataType to convert
 * @param out_str Output buffer for dtype string (must be at least 4 bytes)
 * @return 0 on success, -1 on unsupported type
 */
static int get_numpy_dtype_string(DLDataType dtype, char* out_str) {
  // NumPy dtype format: '<' (little-endian) + type_char + bytes
  out_str[0] = '<';  // little-endian

  switch (dtype.code) {
    case kDLInt:
      out_str[1] = 'i';
      break;
    case kDLUInt:
      out_str[1] = 'u';
      break;
    case kDLFloat:
      out_str[1] = 'f';
      break;
    default:
      fprintf(stderr, "Unsupported dtype code for numpy: %d\n", dtype.code);
      return -1;
  }

  // Add byte size
  int bytes = (dtype.bits + 7) / 8;
  out_str[2] = '0' + bytes;
  out_str[3] = '\0';

  return 0;
}

/**
 * Save tensor to NumPy .npy file
 *
 * @param path Output file path (should end in .npy)
 * @param tensor DLTensor to save
 * @return 0 on success, -1 on failure
 */
static int save_tensor_to_npy(const char* path, const DLTensor* tensor) {
  FILE* f = fopen(path, "wb");
  if (!f) {
    fprintf(stderr, "Failed to open %s for writing\n", path);
    return -1;
  }

  // Write NumPy magic number and version
  const unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
  fwrite(magic, 1, 6, f);

  // Version 1.0
  fputc(1, f);  // major version
  fputc(0, f);  // minor version

  // Build header dictionary
  char dtype_str[4];
  if (get_numpy_dtype_string(tensor->dtype, dtype_str) != 0) {
    fclose(f);
    return -1;
  }

  // Build shape string
  char shape_str[256] = "";
  char temp[32];
  for (int i = 0; i < tensor->ndim; i++) {
    snprintf(temp, sizeof(temp), "%lld%s",
             (long long)tensor->shape[i],
             i < tensor->ndim - 1 ? ", " : "");
    strcat(shape_str, temp);
  }

  // Build complete header dict
  char header[512];
  int header_len = snprintf(header, sizeof(header),
    "{'descr': '%s', 'fortran_order': False, 'shape': (%s%s), }",
    dtype_str,
    shape_str,
    tensor->ndim == 1 ? "," : "");  // Add trailing comma for 1D arrays

  // Pad header to 16-byte boundary (including 10-byte prefix)
  // Prefix: 6 (magic) + 2 (version) + 2 (header_len) = 10 bytes
  int total_prefix = 10;
  int padded_len = ((total_prefix + header_len + 15) / 16) * 16 - total_prefix;

  // Write header length (little-endian uint16)
  uint16_t hlen = (uint16_t)padded_len;
  fwrite(&hlen, sizeof(uint16_t), 1, f);

  // Write header with padding
  fwrite(header, 1, header_len, f);
  for (int i = header_len; i < padded_len - 1; i++) {
    fputc(' ', f);
  }
  fputc('\n', f);  // Header must end with newline

  // Write data
  size_t numel = 1;
  for (int i = 0; i < tensor->ndim; i++) {
    numel *= (size_t)tensor->shape[i];
  }

  size_t elem_size = (tensor->dtype.bits + 7) / 8;
  size_t total_bytes = numel * elem_size;

  size_t written = fwrite(tensor->data, 1, total_bytes, f);
  fclose(f);

  if (written != total_bytes) {
    fprintf(stderr, "Failed to write all data: wrote %zu / %zu bytes\n",
            written, total_bytes);
    return -1;
  }

  fprintf(stderr, "✅ Saved output to %s (%zu elements, %zu bytes)\n",
          path, numel, total_bytes);
  return 0;
}

/**
 * Save tensor to directory with both .npy and .meta.txt
 * Creates files: {output_dir}/{output_name}.npy and {output_dir}/{output_name}.meta.txt
 *
 * @param output_dir Directory to save files in
 * @param output_name Base name for output files (without extension)
 * @param tensor DLTensor to save
 * @return 0 on success, -1 on failure
 */
static int save_tensor_to_dir(const char* output_dir, const char* output_name,
                               const DLTensor* tensor) {
  char npy_path[512];
  char meta_path[512];

  snprintf(npy_path, sizeof(npy_path), "%s/%s.npy", output_dir, output_name);
  snprintf(meta_path, sizeof(meta_path), "%s/%s.meta.txt", output_dir, output_name);

  // Save .npy file
  if (save_tensor_to_npy(npy_path, tensor) != 0) {
    return -1;
  }

  // Save .meta.txt file
  FILE* meta = fopen(meta_path, "w");
  if (!meta) {
    fprintf(stderr, "Failed to create metadata file: %s\n", meta_path);
    return -1;
  }

  // Write metadata
  fprintf(meta, "shape: ");
  for (int i = 0; i < tensor->ndim; i++) {
    fprintf(meta, "%lld%s", (long long)tensor->shape[i],
            i < tensor->ndim - 1 ? "," : "");
  }
  fprintf(meta, "\n");

  // Write dtype info
  const char* dtype_name = "unknown";
  switch (tensor->dtype.code) {
    case kDLInt: dtype_name = "int"; break;
    case kDLUInt: dtype_name = "uint"; break;
    case kDLFloat: dtype_name = "float"; break;
  }
  fprintf(meta, "dtype: %s%d\n", dtype_name, tensor->dtype.bits);

  // Calculate and write total elements
  size_t numel = 1;
  for (int i = 0; i < tensor->ndim; i++) {
    numel *= (size_t)tensor->shape[i];
  }
  fprintf(meta, "numel: %zu\n", numel);

  fclose(meta);
  fprintf(stderr, "✅ Saved metadata to %s\n", meta_path);

  return 0;
}

#endif  // TEST_OUTPUT_WRITER_H
