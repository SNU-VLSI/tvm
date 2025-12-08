#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/platform.h>
#include <tvm/runtime/crt/graph_executor.h>
#include <tvm/runtime/crt/module.h>
#include <dlpack/dlpack.h>
// Internal CRT executor structures for debugging
#include <tvm/runtime/crt/internal/graph_executor/graph_executor.h>

// Direct call prototypes for generated kernels
#include <tvm/runtime/c_backend_api.h>

static char* read_entire_file(const char* path, size_t* out_size) {
  FILE* f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char* buf = (char*)malloc(sz + 1);
  if (!buf) { fclose(f); return NULL; }
  size_t n = fread(buf, 1, sz, f);
  fclose(f);
  buf[n] = '\0';
  if (out_size) *out_size = (size_t)n;
  return buf;
}

static int load_params_file(const char* path, char** out_buf, size_t* out_size) {
  size_t size = 0;
  char* buf = read_entire_file(path, &size);
  if (!buf) return -1;
  *out_buf = buf;
  *out_size = size;
  return 0;
}

__attribute__((weak)) void TVMLogf(const char* format, ...) {
  va_list args; va_start(args, format); vfprintf(stderr, format, args); va_end(args);
}

static void fill_tensor(DLTensor* t, float value) {
  float* data = (float*)t->data;
  size_t numel = 1;
  for (int i = 0; i < t->ndim; ++i) numel *= (size_t)t->shape[i];
  for (size_t i = 0; i < numel; ++i) data[i] = value;
}

static void fill_tensor_constant(DLTensor* t, float value) {
  float* data = (float*)t->data;
  size_t numel = 1;
  for (int i = 0; i < t->ndim; ++i) numel *= (size_t)t->shape[i];
  for (size_t i = 0; i < numel; ++i) data[i] = value;
}

static void fill_tensor_ramp(DLTensor* t, float start, float step) {
  float* data = (float*)t->data;
  size_t numel = 1;
  for (int i = 0; i < t->ndim; ++i) numel *= (size_t)t->shape[i];
  for (size_t i = 0; i < numel; ++i) data[i] = start + (float)i * step;
}

static void fill_tensor_linear_chw_index(DLTensor* t) {
  int8_t* data = (int8_t*)t->data;
  // expects shape [N,C,H,W] with N=1
  if (t->ndim != 4) return;
  int64_t C = t->shape[1];
  int64_t H = t->shape[2];
  int64_t W = t->shape[3];
  for (int64_t c = 0; c < C; ++c) {
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t w = 0; w < W; ++w) {
        int64_t idx = c * H * W + h * W + w;
        // data[idx] = (int16_t)idx;
        data[idx] = 1;
      }
    }
  }
}

static uint32_t checksum_u32(const float* data, size_t n) {
  uint32_t c = 0;
  for (size_t i = 0; i < n; ++i) {
    union { float f; uint32_t u; } u;
    u.f = data[i];
    c = (c * 16777619u) ^ u.u;
  }
  return c;
}

static uint32_t checksum_fnv1a_u32(const float* data, size_t n) {
  uint32_t c = 2166136261u; // FNV-1a seed
  for (size_t i = 0; i < n; ++i) {
    union { float f; uint32_t u; } u;
    u.f = data[i];
    c ^= u.u;
    c *= 16777619u;
  }
  return c;
}

static size_t num_elements_from_shape(const DLTensor* t) {
  size_t n = 1;
  for (int i = 0; i < t->ndim; ++i) n *= (size_t)t->shape[i];
  return n;
}

static size_t nnz_tensor(const DLTensor* t) {
  size_t n = num_elements_from_shape(t);
  const float* f = (const float*)t->data;
  size_t nnz = 0;
  for (size_t i = 0; i < n; ++i) if (f[i] != 0.0f) ++nnz;
  return nnz;
}

int main(int argc, char** argv) {
  const char* graph_path = argc > 1 ? argv[1] : "mlf/executor-config/graph/default.graph";
  const char* params_path = argc > 2 ? argv[2] : "mlf/parameters/default.params";

  if (TVMPlatformInitialize() != kTvmErrorNoError) {
    fprintf(stderr, "Platform init failed\n");
    return 2;
  }
  fprintf(stderr, "dbg: platform inited\n");

  size_t graph_size = 0;
  char* graph_json = read_entire_file(graph_path, &graph_size);
  if (!graph_json) {
    fprintf(stderr, "Failed to read graph: %s\n", graph_path);
    return 2;
  }
  fprintf(stderr, "dbg: graph loaded (%zu bytes)\n", graph_size);

  const TVMModule* syslib = TVMSystemLibEntryPoint();
  TVMModuleHandle mod;
  if (TVMModCreateFromCModule(syslib, &mod) != 0) {
    fprintf(stderr, "Failed to create module handle\n");
    return -1;
  }
  fprintf(stderr, "dbg: got system lib handle %p\n", (void*)mod);

  DLDevice dev = {kDLCPU, 0};
  TVMGraphExecutor* exec = NULL;
  fprintf(stderr, "dbg: creating executor...\n");
  int rc = TVMGraphExecutor_Create(graph_json, mod, &dev, &exec);
  fprintf(stderr, "dbg: create rc=%d, exec=%p\n", rc, (void*)exec);
  if (rc != 0) {
    fprintf(stderr, "TVMGraphExecutor_Create failed: %d\n", rc);
    return 2;
  }

  char* params_buf = NULL; size_t params_size = 0;
  if (load_params_file(params_path, &params_buf, &params_size) != 0) {
    fprintf(stderr, "Failed to read params: %s\n", params_path);
    return 2;
  }
  fprintf(stderr, "dbg: params loaded (%zu bytes)\n", params_size);
  rc = TVMGraphExecutor_LoadParams(exec, params_buf, (uint32_t)params_size);
  fprintf(stderr, "dbg: load params rc=%d\n", rc);
  if (rc != 0) {
    fprintf(stderr, "LoadParams failed: %d\n", rc);
    return 2;
  }

  int N, C, H, W, OC;
  N = 1; C = 3; H = 8; W = 8;
  OC=10;

  int64_t shape4[4] = {N, C, H, W};
  DLTensor x1 = {0};
  x1.device = dev;
  x1.ndim = 4;
  x1.shape = shape4;
  x1.dtype = (DLDataType){kDLInt, 8, 1};
  size_t numel = (size_t)shape4[0]*shape4[1]*shape4[2]*shape4[3];
  size_t nbytes = numel * sizeof(int8_t);
  void* x1_data = NULL;
  TVMPlatformMemoryAllocate(nbytes, dev, &x1_data);
  x1.data = x1_data;
  // Initialize like cpp_graph_deploy.cpp
  fill_tensor_linear_chw_index(&x1);
  printf("dbg: inputs prepared (%zu bytes each)\n", nbytes);
  for(size_t i=0; i<numel; ++i) {
    printf("%d ", ((int8_t*)x1_data)[i]);
    if((i+1)%16 == 0) printf("\n");
  }

#if 0
  // Optional input dumps for debugging
  size_t to_print = (numel < 16) ? numel : 16;
  printf("x1:");
  for (size_t i = 0; i < to_print; ++i) printf(" %g", ((float*)x1_data)[i]);
  printf("\n");
#endif

  TVMGraphExecutor_SetInput(exec, "model_input", &x1);
  TVMGraphExecutor_Run(exec);

#if 0
  // Deep instrumentation (kept for reference)
  uint32_t x1_eid = TVMGraphExecutor_GetEntryId((TVMGraphExecutor*)exec, 0, 0);
  uint32_t x2_eid = TVMGraphExecutor_GetEntryId((TVMGraphExecutor*)exec, 1, 0);
  DLTensor* x1_exec = &(((TVMGraphExecutor*)exec)->data_entry[x1_eid].dl_tensor);
  DLTensor* x2_exec = &(((TVMGraphExecutor*)exec)->data_entry[x2_eid].dl_tensor);
  fprintf(stderr, "dbg: x1 ptr match: %s, x2 ptr match: %s\n",
          (x1_exec->data == x1_data ? "yes" : "no"),
          (x2_exec->data == x2_data ? "yes" : "no"));
#endif

  // // Execute using direct kernel calls (avoids CRT PackedFunc marshaling issues)
  // rc = run_graph_direct((TVMGraphExecutor*)exec);
  // if (rc != 0) {
  //   fprintf(stderr, "Kernel execution failed: %d\n", rc);
  //   return 2;
  // }

#if 0
  // Legacy packed-executor path
  TVMGraphExecutor_Run(exec);
#endif

  // First, check what shape the executor actually has
  uint32_t output_eid = TVMGraphExecutor_GetEntryId(exec, exec->outputs[0].node_id, exec->outputs[0].index);
  DLTensor* internal_output = &(exec->data_entry[output_eid].dl_tensor);
  fprintf(stderr, "dbg: internal output ndim=%d, shape=[", internal_output->ndim);
  for (int i = 0; i < internal_output->ndim; ++i) {
    fprintf(stderr, "%lld%s", (long long)internal_output->shape[i],
            i < internal_output->ndim - 1 ? ", " : "");
  }
  fprintf(stderr, "]\n");

  DLTensor out = {0};
  out.device = dev;
  int64_t oshape[4] = {1, OC, 1, 1};  // Try matching internal shape
  out.ndim = internal_output->ndim;  // Use actual ndim from executor
  out.shape = oshape;
  out.dtype = (DLDataType){kDLFloat, 32, 1};
  size_t onumel = 1;
  for (int i = 0; i < out.ndim; ++i) {
    oshape[i] = internal_output->shape[i];  // Copy actual shape
    onumel *= (size_t)internal_output->shape[i];
  }
  size_t onbytes = onumel * sizeof(float);
  void* out_data = NULL; TVMPlatformMemoryAllocate(onbytes, dev, &out_data);
  out.data = out_data;
  rc = TVMGraphExecutor_GetOutput(exec, 0, &out);
  if (rc != 0) {
    fprintf(stderr, "GetOutput failed: %d\n", rc);
    return 2;
  }

  // Print first 16 values and checksum (concise verification)
  float* out_f = (float*)out.data;
  printf("Output tensor (%zu elements):\n", onumel);
  for(int i=0; i<onumel; ++i) {
    printf("%f ", out_f[i]);
  }

  // prepare reference
  //TODO: calculate reference
  // int16_t ref[onumel];
  // for (size_t i = 0; i < onumel; ++i) ref[i] = (((int16_t*)x1_data)[i] > 0) ? ((int16_t*)x1_data)[i] : 0;

  // Verify against reference
  // size_t nerr = 0;
  // for (size_t i = 0; i < onumel; ++i) {
  //   if (out_f[i] != ref[i]) {
  //     if (nerr < 16) {
  //       printf("mismatch at %zu: got %d, expected %d\n", i, out_f[i], ref[i]);
  //     }
  //     ++nerr;
  //   }
  // }
  // if(nerr == 0) {
  //   printf("PASS: all %zu values match reference\n", onumel);
  // } else {
  //   printf("FAIL: %zu mismatches found\n", nerr);
  // }

  TVMPlatformMemoryFree(x1_data, dev);
  TVMPlatformMemoryFree(out_data, dev);
  free(graph_json);
  free(params_buf);
  TVMGraphExecutor_Release(&exec);
  return 0;
}