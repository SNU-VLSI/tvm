#include <stddef.h>
#include <stdint.h>
#include <stdarg.h>
#ifdef TVM_ARM_LINUX
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#endif

#include <dlpack/dlpack.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/platform.h>
#include <tvm/runtime/crt/graph_executor.h>
#include <tvm/runtime/crt/internal/graph_executor/graph_executor.h>
#include <tvm/runtime/c_backend_api.h>

// System lib entry provided by the generated C
extern const TVMModule* TVMSystemLibEntryPoint(void);

// ---- User-provided (embed) graph JSON and params ----
__attribute__((weak)) const char tvm_graph_json[] = "";
__attribute__((weak)) const unsigned char tvm_params[] = {0};
__attribute__((weak)) const unsigned int tvm_params_len = 0;

// ---- Minimal platform hooks (page allocator) ----
#include <tvm/runtime/crt/page_allocator.h>
#ifndef TVM_WORKSPACE_SIZE_BYTES
#define TVM_WORKSPACE_SIZE_BYTES (512 * 1024)
#endif
static uint8_t g_workspace[TVM_WORKSPACE_SIZE_BYTES];
static MemoryManagerInterface* g_memory_manager = NULL;

size_t TVMPlatformFormatMessage(char* out_buf, size_t out_buf_size_bytes, const char* fmt, va_list args) {
  (void)out_buf; (void)out_buf_size_bytes; (void)fmt; (void)args; return 0;
}

void __attribute__((noreturn)) TVMPlatformAbort(tvm_crt_error_t code) {
#ifdef TVM_ARM_LINUX
  fprintf(stderr, "TVMPlatformAbort: code=%d\n", (int)code);
#endif
  (void)code; while (1) {}
}

// CRT logging stub
void TVMLogf(const char* fmt, ...) { (void)fmt; }

tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev, void** out_ptr) {
  return g_memory_manager->Allocate(g_memory_manager, num_bytes, dev, out_ptr);
}

tvm_crt_error_t TVMPlatformMemoryFree(void* ptr, DLDevice dev) {
  return g_memory_manager->Free(g_memory_manager, ptr, dev);
}

tvm_crt_error_t TVMPlatformTimerStart() { return kTvmErrorNoError; }

tvm_crt_error_t TVMPlatformTimerStop(double* elapsed_time_seconds) { (void)elapsed_time_seconds; return kTvmErrorNoError; }

tvm_crt_error_t TVMPlatformBeforeMeasurement() { return kTvmErrorNoError; }

tvm_crt_error_t TVMPlatformAfterMeasurement() { return kTvmErrorNoError; }

tvm_crt_error_t TVMPlatformInitialize() {
  int status = PageMemoryManagerCreate(&g_memory_manager, g_workspace, sizeof(g_workspace), 8);
  if (status != 0) return kTvmErrorPlatformMemoryManagerInitialized;
  return kTvmErrorNoError;
}

#ifdef TVM_ARM_LINUX
// ---- File I/O helpers (Linux) ----
static char* read_entire_text(const char* path, size_t* out_size) {
  FILE* f = fopen(path, "rb"); if (!f) return NULL;
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  char* buf = (char*)malloc(sz + 1); if (!buf) { fclose(f); return NULL; }
  size_t n = fread(buf, 1, sz, f); fclose(f); buf[n] = '\0';
  if (out_size) *out_size = n; return buf;
}
static unsigned char* read_entire_bin(const char* path, size_t* out_size) {
  FILE* f = fopen(path, "rb"); if (!f) return NULL;
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  unsigned char* buf = (unsigned char*)malloc(sz); if (!buf) { fclose(f); return NULL; }
  size_t n = fread(buf, 1, sz, f); fclose(f);
  if (out_size) *out_size = n; return buf;
}
#endif

// ---- Utilities ----
static void fill_tensor_linear_chw_index(DLTensor* t) {
  float* data = (float*)t->data; if (t->ndim != 4) return;
  int64_t C = t->shape[1], H = t->shape[2], W = t->shape[3];
  for (int64_t c = 0; c < C; ++c)
    for (int64_t h = 0; h < H; ++h)
      for (int64_t w = 0; w < W; ++w)
        data[c * H * W + h * W + w] = (float)(c * H * W + h * W + w);
}

static uint32_t checksum_fnv1a_u32(const float* data, size_t n) {
  uint32_t c = 2166136261u; for (size_t i = 0; i < n; ++i) { union { float f; uint32_t u; } u; u.f = data[i]; c ^= u.u; c *= 16777619u; } return c;
}

static size_t bytes_per_element(DLDataType dtype) {
  size_t bytes = (dtype.bits + 7) / 8; if (dtype.lanes > 1) bytes *= dtype.lanes; return bytes;
}

static size_t num_elements(const DLTensor* t) {
  size_t n = 1; for (int i = 0; i < t->ndim; ++i) n *= (size_t)t->shape[i]; return n;
}

static int run_graph_direct(TVMGraphExecutor* exec) {
#ifdef TVM_ARM_LINUX
  fprintf(stderr, "runner: calling GraphExecutor_Run\n");
#endif
  TVMGraphExecutor_Run(exec);
#ifdef TVM_ARM_LINUX
  fprintf(stderr, "runner: GraphExecutor_Run finished\n");
#endif
  return 0;
}

volatile uint32_t tvm_last_checksum = 0;

int tvm_run_once(void) {
#ifdef TVM_ARM_LINUX
  fprintf(stderr, "runner: init platform\n");
#endif
  if (TVMPlatformInitialize() != kTvmErrorNoError) return -1;
  const TVMModule* syslib = TVMSystemLibEntryPoint(); TVMModuleHandle mod = (TVMModuleHandle)syslib; DLDevice dev = {kDLCPU, 0};

  // Load graph JSON
  char* graph_json_buf = (char*)tvm_graph_json; size_t graph_len = 0;
#ifdef TVM_ARM_LINUX
  const char* graph_env = getenv("TVM_GRAPH_JSON"); const char* params_env = getenv("TVM_PARAMS_BIN");
  const char* graph_path = graph_env ? graph_env : "mlf/executor-config/graph/default.graph";
  const char* params_path = params_env ? params_env : "mlf/parameters/default.params";
  if (tvm_graph_json[0] == '\0') {
    graph_json_buf = read_entire_text(graph_path, &graph_len);
    if (!graph_json_buf) { fprintf(stderr, "failed to read graph: %s\n", graph_path); return -10; }
  }
  fprintf(stderr, "runner: graph loaded (%zu bytes)\n", graph_len);
#endif

  TVMGraphExecutor* exec = NULL;
  if (TVMGraphExecutor_Create(graph_json_buf, mod, &dev, &exec) != 0) return -2;
#ifdef TVM_ARM_LINUX
  fprintf(stderr, "runner: executor created\n");
#endif

  // Load params
#ifdef TVM_ARM_LINUX
  if (tvm_params_len == 0) {
    size_t params_len = 0; unsigned char* params_buf = read_entire_bin(params_path, &params_len);
    if (!params_buf || params_len == 0) { fprintf(stderr, "failed to read params: %s\n", params_path); return -11; }
    if (TVMGraphExecutor_LoadParams(exec, (const char*)params_buf, (uint32_t)params_len) != 0) return -3; free(params_buf);
    fprintf(stderr, "runner: params loaded (%zu bytes)\n", params_len);
  } else
#endif
  {
    if (tvm_params_len == 0) return -4;
    if (TVMGraphExecutor_LoadParams(exec, (const char*)tvm_params, (uint32_t)tvm_params_len) != 0) return -3;
  }

  // Prepare input using the executor's first entry (node 0, entry 0)
  uint32_t in_eid = TVMGraphExecutor_GetEntryId(exec, 0, 0);
  DLTensor* in_exec = &(exec->data_entry[in_eid].dl_tensor);
  size_t nbytes = num_elements(in_exec) * bytes_per_element(in_exec->dtype);
  DLTensor in = (DLTensor){0}; in.device = dev; in.ndim = in_exec->ndim; in.shape = in_exec->shape; in.dtype = in_exec->dtype;
  void* in_data = NULL; TVMPlatformMemoryAllocate(nbytes, dev, &in_data); in.data = in_data;
  fill_tensor_linear_chw_index(&in);
#ifdef TVM_ARM_LINUX
  fprintf(stderr, "runner: input prepared (%zu bytes)\n", nbytes);
#endif

  // Try common input names; default graph uses "input_1"
  TVMGraphExecutor_SetInput(exec, "input_1", &in);
#ifdef TVM_ARM_LINUX
  fprintf(stderr, "runner: input set\n");
#endif

  if (run_graph_direct(exec) != 0) return -5;

  // Read output 0 using executor-provided shape
  uint32_t out_eid = TVMGraphExecutor_GetEntryId(exec, 15, 0);
  DLTensor* out_exec = &(exec->data_entry[out_eid].dl_tensor);
  size_t out_bytes = num_elements(out_exec) * bytes_per_element(out_exec->dtype);
  DLTensor out = (DLTensor){0}; out.device = dev; out.ndim = out_exec->ndim; out.shape = out_exec->shape; out.dtype = out_exec->dtype; void* out_data = NULL; TVMPlatformMemoryAllocate(out_bytes, dev, &out_data); out.data = out_data;
  if (TVMGraphExecutor_GetOutput(exec, 0, &out) != 0) return -6; tvm_last_checksum = checksum_fnv1a_u32((const float*)out.data, num_elements(&out));
#ifdef TVM_ARM_LINUX
  fprintf(stderr, "runner: output captured\n");
#endif

#ifdef TVM_ARM_LINUX
  if (graph_json_buf && graph_json_buf != tvm_graph_json) free(graph_json_buf);
#endif
  TVMPlatformMemoryFree(in_data, dev); TVMPlatformMemoryFree(out_data, dev); TVMGraphExecutor_Release(&exec);
  return 0;
}

#ifdef TVM_ARM_LINUX
int main(void) {
  int rc = tvm_run_once(); if (rc != 0) { fprintf(stderr, "runner failed rc=%d\n", rc); return 1; }
  printf("checksum=0x%08x\n", tvm_last_checksum); return 0;
}
#else
// Bare-metal stubs for newlib
#include <sys/stat.h>
int _write(int fd, const void* buf, size_t cnt) { (void)fd; (void)buf; return (int)cnt; }
int _read(int fd, void* buf, size_t cnt) { (void)fd; (void)buf; (void)cnt; return 0; }
int _close(int fd) { (void)fd; return 0; }
int _fstat(int fd, struct stat* st) { (void)fd; if (st) { st->st_mode = S_IFCHR; } return 0; }
int _isatty(int fd) { (void)fd; return 1; }
int _lseek(int fd, int ptr, int dir) { (void)fd; (void)ptr; (void)dir; return 0; }
int _getpid(void) { return 1; }
int _kill(int pid, int sig) { (void)pid; (void)sig; return -1; }
void _exit(int status) { (void)status; while (1) {} }
void* _sbrk(ptrdiff_t incr) { (void)incr; return (void*)-1; }
int main(void) { (void)tvm_run_once(); while (1) {} }
#endif