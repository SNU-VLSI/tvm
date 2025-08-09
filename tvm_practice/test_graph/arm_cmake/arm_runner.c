#include <stddef.h>
#include <stdint.h>
#include <stdarg.h>

#include <dlpack/dlpack.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/platform.h>
#include <tvm/runtime/crt/graph_executor.h>
#include <tvm/runtime/crt/internal/graph_executor/graph_executor.h>
#include <tvm/runtime/c_backend_api.h>

// Prototypes for generated kernels (from MLF C sources)
extern int32_t tvmgen_default_fused_layout_transform(TVMValue* args, int32_t* arg_type_ids, int32_t num_args, TVMValue* out_ret_value, int32_t* out_ret_tcode, void* resource_handle);
extern int32_t tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu(TVMValue* args, int32_t* arg_type_ids, int32_t num_args, TVMValue* out_ret_value, int32_t* out_ret_tcode, void* resource_handle);
extern int32_t tvmgen_default_fused_nn_global_max_pool2d(TVMValue* args, int32_t* arg_type_ids, int32_t num_args, TVMValue* out_ret_value, int32_t* out_ret_tcode, void* resource_handle);
extern int32_t tvmgen_default_fused_layout_transform_add_layout_transform(TVMValue* args, int32_t* arg_type_ids, int32_t num_args, TVMValue* out_ret_value, int32_t* out_ret_tcode, void* resource_handle);

// System lib entry provided by the generated C
extern const TVMModule* TVMSystemLibEntryPoint(void);

// ---- User-provided (embed) graph JSON and params ----
// Provide these symbols in your firmware by converting files to arrays (e.g., xxd -i):
//   default.graph -> const char tvm_graph_json[];
//   default.params -> const unsigned char tvm_params[]; const unsigned int tvm_params_len;
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
  // Implement to route messages to your UART/logger if desired.
  // For minimal footprint, just return 0 (no output).
  (void)out_buf; (void)out_buf_size_bytes; (void)fmt; (void)args; return 0;
}

void __attribute__((noreturn)) TVMPlatformAbort(tvm_crt_error_t code) {
  (void)code;
  while (1) { /* hang */ }
}

// CRT logging stub (avoid pulling host RPC server)
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

// ---- Utilities ----
static void fill_tensor_linear_chw_index(DLTensor* t) {
  float* data = (float*)t->data;
  if (t->ndim != 4) return;
  int64_t C = t->shape[1], H = t->shape[2], W = t->shape[3];
  for (int64_t c = 0; c < C; ++c)
    for (int64_t h = 0; h < H; ++h)
      for (int64_t w = 0; w < W; ++w)
        data[c * H * W + h * W + w] = (float)(c * H * W + h * W + w);
}

static uint32_t checksum_fnv1a_u32(const float* data, size_t n) {
  uint32_t c = 2166136261u; for (size_t i = 0; i < n; ++i) { union { float f; uint32_t u; } u; u.f = data[i]; c ^= u.u; c *= 16777619u; } return c;
}

static int run_graph_direct(TVMGraphExecutor* exec) {
  // node 0: x1, 1: x2
  uint32_t x1_eid = TVMGraphExecutor_GetEntryId(exec, 0, 0);
  uint32_t x2_eid = TVMGraphExecutor_GetEntryId(exec, 1, 0);
  DLTensor* x1 = &(exec->data_entry[x1_eid].dl_tensor);
  DLTensor* x2 = &(exec->data_entry[x2_eid].dl_tensor);

  // node 2
  uint32_t out2_eid = TVMGraphExecutor_GetEntryId(exec, 2, 0);
  DLTensor* out2 = &(exec->data_entry[out2_eid].dl_tensor);
  TVMValue a2[2]; int32_t tc2[2] = {kTVMDLTensorHandle, kTVMDLTensorHandle};
  a2[0].v_handle = (void*)x1; a2[1].v_handle = (void*)out2;
  int r = tvmgen_default_fused_layout_transform(a2, tc2, 2, NULL, NULL, NULL); if (r) return r;

  // nodes 3/4 (params), node 5
  uint32_t w_eid = TVMGraphExecutor_GetEntryId(exec, 3, 0);
  uint32_t b_eid = TVMGraphExecutor_GetEntryId(exec, 4, 0);
  DLTensor* w = &(exec->data_entry[w_eid].dl_tensor);
  DLTensor* b = &(exec->data_entry[b_eid].dl_tensor);
  uint32_t out5_eid = TVMGraphExecutor_GetEntryId(exec, 5, 0);
  DLTensor* out5 = &(exec->data_entry[out5_eid].dl_tensor);
  TVMValue a5[4]; int32_t tc5[4] = {kTVMDLTensorHandle, kTVMDLTensorHandle, kTVMDLTensorHandle, kTVMDLTensorHandle};
  a5[0].v_handle = (void*)out2; a5[1].v_handle = (void*)w; a5[2].v_handle = (void*)b; a5[3].v_handle = (void*)out5;
  r = tvmgen_default_fused_nn_contrib_conv2d_NCHWc_add_nn_relu(a5, tc5, 4, NULL, NULL, NULL); if (r) return r;

  // node 6
  uint32_t out6_eid = TVMGraphExecutor_GetEntryId(exec, 6, 0);
  DLTensor* out6 = &(exec->data_entry[out6_eid].dl_tensor);
  TVMValue a6[2]; int32_t tc6[2] = {kTVMDLTensorHandle, kTVMDLTensorHandle};
  a6[0].v_handle = (void*)out5; a6[1].v_handle = (void*)out6;
  r = tvmgen_default_fused_nn_global_max_pool2d(a6, tc6, 2, NULL, NULL, NULL); if (r) return r;

  // node 7
  uint32_t out7_eid = TVMGraphExecutor_GetEntryId(exec, 7, 0);
  DLTensor* out7 = &(exec->data_entry[out7_eid].dl_tensor);
  TVMValue a7[3]; int32_t tc7[3] = {kTVMDLTensorHandle, kTVMDLTensorHandle, kTVMDLTensorHandle};
  a7[0].v_handle = (void*)x2; a7[1].v_handle = (void*)out6; a7[2].v_handle = (void*)out7;
  r = tvmgen_default_fused_layout_transform_add_layout_transform(a7, tc7, 3, NULL, NULL, NULL); if (r) return r;

  return 0;
}

volatile uint32_t tvm_last_checksum = 0;

int tvm_m3_run_once(void) {
  if (TVMPlatformInitialize() != kTvmErrorNoError) return -1;

  const TVMModule* syslib = TVMSystemLibEntryPoint();
  TVMModuleHandle mod = (TVMModuleHandle)syslib;
  DLDevice dev = {kDLCPU, 0};

  TVMGraphExecutor* exec = NULL;
  if (TVMGraphExecutor_Create(tvm_graph_json, mod, &dev, &exec) != 0) return -2;

  if (tvm_params_len > 0) {
    if (TVMGraphExecutor_LoadParams(exec, (const char*)tvm_params, (uint32_t)tvm_params_len) != 0) return -3;
  } else {
    return -4; // params not provided
  }

  // Prepare inputs
  int64_t shape4[4] = {1, 32, 56, 56};
  size_t numel = (size_t)shape4[0]*shape4[1]*shape4[2]*shape4[3];
  size_t nbytes = numel * sizeof(float);
  DLTensor x1 = {0}, x2 = {0};
  x1.device = dev; x2.device = dev; x1.ndim = 4; x2.ndim = 4; x1.shape = shape4; x2.shape = shape4;
  x1.dtype = (DLDataType){kDLFloat, 32, 1}; x2.dtype = (DLDataType){kDLFloat, 32, 1};
  void* x1_data = NULL; void* x2_data = NULL;
  TVMPlatformMemoryAllocate(nbytes, dev, &x1_data);
  TVMPlatformMemoryAllocate(nbytes, dev, &x2_data);
  x1.data = x1_data; x2.data = x2_data;
  fill_tensor_linear_chw_index(&x1);
  fill_tensor_linear_chw_index(&x2);

  TVMGraphExecutor_SetInput(exec, "x1", &x1);
  TVMGraphExecutor_SetInput(exec, "x2", &x2);

  // Execute graph via direct calls
  if (run_graph_direct(exec) != 0) return -5;

  // Get output and compute checksum
  int64_t oshape[4] = {1, 32, 56, 56};
  DLTensor out = {0}; out.device = dev; out.ndim = 4; out.shape = oshape; out.dtype = (DLDataType){kDLFloat, 32, 1};
  void* out_data = NULL; TVMPlatformMemoryAllocate(nbytes, dev, &out_data); out.data = out_data;
  if (TVMGraphExecutor_GetOutput(exec, 0, &out) != 0) return -6;
  tvm_last_checksum = checksum_fnv1a_u32((const float*)out.data, numel);

  TVMPlatformMemoryFree(x1_data, dev);
  TVMPlatformMemoryFree(x2_data, dev);
  TVMPlatformMemoryFree(out_data, dev);
  TVMGraphExecutor_Release(&exec);
  return 0;
}

// Optional bare-metal entry for quick bring-up; replace with your firmware entry
int main(void) {
  int rc = tvm_m3_run_once();
  (void)rc; // you can set a breakpoint here and inspect tvm_last_checksum
  for (;;) {}
}

// ---- Minimal syscalls stubs for newlib (bare metal) ----
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