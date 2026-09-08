/* DAE results use staged window IDs; the master restores original IDs through
 * sample_map.json. No anomaly decision or AUC is inferred from one window. */
#ifndef DAE_DATASET_METRICS_H
#define DAE_DATASET_METRICS_H
#include <math.h>
#include "npy_dataset_loader.h"

static int dae_dataset_mode(void) {
  const char* task = getenv("IMCFLOW_DATASET_TASK");
  return task && strcmp(task, "anomaly_detection") == 0;
}

static int dae_write_run_header(FILE* stream) {
  const char* run_id = getenv("IMCFLOW_DAE_RUN_ID");
  if (!stream || !run_id || strlen(run_id) != 32 || strspn(run_id, "0123456789abcdef") != 32) return -1;
  fprintf(stream, "DAE_RUN %s\n", run_id);
  return ferror(stream) || fflush(stream) != 0 ? -1 : 0;
}

static int dae_window_mse(const NpyDataset* images, const void* sample,
                          const DLTensor* output, double* mse) {
  if (!images || !sample || !output || !output->data || !mse ||
      images->ndim != 2 || images->sample_numel != 640 || images->sample_size != 640 * sizeof(float) ||
      images->dtype.code != kDLFloat || images->dtype.bits != 32 || images->dtype.lanes != 1 ||
      output->dtype.code != kDLFloat || output->dtype.bits != 32 || output->dtype.lanes != 1 ||
      output->ndim != 2 || !output->shape || output->shape[0] != 1 || output->shape[1] != 640 ||
      (output->strides && (output->strides[0] != 640 || output->strides[1] != 1))) {
    return -1;
  }
  const float* x = (const float*)sample;
  const float* y = (const float*)((const char*)output->data + output->byte_offset);
  double sum = 0;
  for (size_t i = 0; i < 640; i++) {
    if (!isfinite(x[i]) || !isfinite(y[i])) return -1;
    double diff = (double)y[i] - (double)x[i];
    sum += diff * diff;
  }
  *mse = sum / 640.0;
  return isfinite(*mse) ? 0 : -1;
}

static int dae_write_sample(FILE* stream, size_t sample_id, int64_t label,
                            const DLTensor* output, double mse, int save_reconstruction) {
  if (!stream || !isfinite(mse) || mse < 0 || (label != 0 && label != 1)) return -1;
  fprintf(stream, "DAE_SAMPLE {\"sample_id\":%zu,\"label\":%lld,\"mse\":%.17g",
          sample_id, (long long)label, mse);
  if (save_reconstruction) {
    const float* values = (const float*)((const char*)output->data + output->byte_offset);
    fprintf(stream, ",\"reconstruction\":[");
    for (size_t i = 0; i < 640; i++) fprintf(stream, "%s%.9g", i ? "," : "", values[i]);
    fprintf(stream, "]");
  }
  fprintf(stream, "}\n");
  return ferror(stream) || fflush(stream) != 0 ? -1 : 0;
}

static int dae_record_sample(FILE* result_file, size_t sample_id, int64_t label,
                             const NpyDataset* images, const void* sample,
                             const DLTensor* output, double* mse) {
  if (dae_window_mse(images, sample, output, mse) != 0) return -1;
  const char* save = getenv("IMCFLOW_DAE_SAVE_RECONSTRUCTION");
  int preserve = save && strcmp(save, "1") == 0;
  if (dae_write_sample(result_file, sample_id, label, output, *mse, preserve) != 0) return -1;
  return dae_write_sample(stdout, sample_id, label, output, *mse, 0);
}

static void dae_progress(size_t current, size_t total, int evaluated, int failed) {
  printf("Progress: %zu/%zu, Evaluated: %d, Failed: %d, Metric: window_mse\n",
         current, total, evaluated, failed);
  fflush(stdout);
}
#endif
