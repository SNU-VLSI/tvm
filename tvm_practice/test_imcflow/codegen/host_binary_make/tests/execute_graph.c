/**
 * Unified TVM Graph Executor
 *
 * This executable can run any TVM compiled model by:
 * - Auto-detecting input/output names from graph JSON
 * - Loading inputs from standardized .bin/.meta.txt files
 * - Saving outputs to NumPy .npy format
 * - Optional validation against CPU reference outputs
 *
 * Usage:
 *   execute_graph <graph.json> <params.params> <input_dir> <output_dir>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/crt/platform.h>
#include <tvm/runtime/crt/graph_executor.h>
#include <tvm/runtime/crt/module.h>
#include <dlpack/dlpack.h>
// Internal CRT executor structures for accessing executor internals
#include <tvm/runtime/crt/internal/graph_executor/graph_executor.h>

// Shared test utilities
#include "test_input_loader.h"
#include "test_output_writer.h"

// ============================================================================
// Utility Functions
// ============================================================================

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
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
}

static size_t num_elements_from_shape(const DLTensor* t) {
  size_t n = 1;
  for (int i = 0; i < t->ndim; ++i) n *= (size_t)t->shape[i];
  return n;
}

/**
 * Simple JSON parser to extract input node names
 *
 * Strategy:
 * 1. Find "arg_nodes": [...]  - contains indices of input nodes
 * 2. Find "nodes": [...] - contains all node definitions
 * 3. For each index in arg_nodes, find corresponding node with "op": "null"
 * 4. Extract "name" field from those nodes
 */
static int parse_input_names_from_graph(const char* graph_json, char input_names[][64], int max_inputs) {
  // Find arg_nodes array
  const char* arg_nodes_start = strstr(graph_json, "\"arg_nodes\": [");
  if (!arg_nodes_start) {
    fprintf(stderr, "Failed to find 'arg_nodes' in graph JSON\n");
    return -1;
  }

  // Parse arg_nodes indices
  int arg_node_indices[16];  // Support up to 16 inputs
  int num_arg_nodes = 0;

  const char* p = arg_nodes_start + strlen("\"arg_nodes\": [");
  while (*p && *p != ']' && num_arg_nodes < 16) {
    // Skip whitespace
    while (*p && isspace(*p)) p++;

    // Parse number
    if (isdigit(*p)) {
      arg_node_indices[num_arg_nodes++] = atoi(p);
      // Skip past number
      while (*p && isdigit(*p)) p++;
    }

    // Skip comma
    if (*p == ',') p++;
  }

  if (num_arg_nodes == 0) {
    fprintf(stderr, "No arg_nodes found in graph JSON\n");
    return -1;
  }

  fprintf(stderr, "Found %d arg_nodes: ", num_arg_nodes);
  for (int i = 0; i < num_arg_nodes; i++) {
    fprintf(stderr, "%d%s", arg_node_indices[i], i < num_arg_nodes - 1 ? ", " : "");
  }
  fprintf(stderr, "\n");

  // Find nodes array
  const char* nodes_start = strstr(graph_json, "\"nodes\": [");
  if (!nodes_start) {
    fprintf(stderr, "Failed to find 'nodes' in graph JSON\n");
    return -1;
  }

  // Parse nodes and extract names for arg_node indices
  int inputs_found = 0;
  p = nodes_start + strlen("\"nodes\": [");
  int node_idx = 0;

  while (*p && *p != ']' && inputs_found < max_inputs && inputs_found < num_arg_nodes) {
    // Find next node object (starts with '{')
    while (*p && *p != '{') p++;
    if (*p != '{') break;

    const char* node_start = p;

    // Find matching closing brace (simple depth counting)
    int depth = 0;
    const char* node_end = p;
    while (*node_end) {
      if (*node_end == '{') depth++;
      else if (*node_end == '}') {
        depth--;
        if (depth == 0) break;
      }
      node_end++;
    }

    // Check if this node index is in arg_nodes
    int is_input = 0;
    for (int i = 0; i < num_arg_nodes; i++) {
      if (node_idx == arg_node_indices[i]) {
        is_input = 1;
        break;
      }
    }

    if (is_input) {
      // Extract name field from this node
      // Look for "name": "xxx"
      const char* name_key = strstr(node_start, "\"name\": \"");
      if (name_key && name_key < node_end) {
        const char* name_start = name_key + strlen("\"name\": \"");
        const char* name_end = strchr(name_start, '"');
        if (name_end && name_end < node_end) {
          int name_len = name_end - name_start;
          if (name_len < 64) {
            strncpy(input_names[inputs_found], name_start, name_len);
            input_names[inputs_found][name_len] = '\0';
            fprintf(stderr, "  Input %d: '%s' (node %d)\n", inputs_found, input_names[inputs_found], node_idx);
            inputs_found++;
          }
        }
      }
    }

    // Move to next node
    node_idx++;
    p = node_end + 1;
  }

  return inputs_found;
}

/**
 * Print tensor info for debugging
 */
static void print_tensor_info(const char* name, const DLTensor* tensor) {
  fprintf(stderr, "  %s:\n", name);
  fprintf(stderr, "    Shape: [");
  for (int i = 0; i < tensor->ndim; ++i) {
    fprintf(stderr, "%lld%s", (long long)tensor->shape[i],
            i < tensor->ndim - 1 ? ", " : "");
  }
  fprintf(stderr, "]\n");

  const char* dtype_name = "unknown";
  switch (tensor->dtype.code) {
    case kDLInt: dtype_name = "int"; break;
    case kDLUInt: dtype_name = "uint"; break;
    case kDLFloat: dtype_name = "float"; break;
  }
  fprintf(stderr, "    Dtype: %s%d\n", dtype_name, tensor->dtype.bits);
  fprintf(stderr, "    Elements: %zu\n", num_elements_from_shape(tensor));
}

// ============================================================================
// Main Execution Logic
// ============================================================================

int main(int argc, char** argv) {
  // Parse command line arguments
  const char* graph_path = argc > 1 ? argv[1] : "mlf/executor-config/graph/default.graph";
  const char* params_path = argc > 2 ? argv[2] : "mlf/parameters/default.params";
  const char* input_dir = argc > 3 ? argv[3] : "./test_inputs";
  const char* output_dir = argc > 4 ? argv[4] : "./test_outputs";

  fprintf(stderr, "\n");
  fprintf(stderr, "========================================\n");
  fprintf(stderr, "  TVM Unified Graph Executor\n");
  fprintf(stderr, "========================================\n");
  fprintf(stderr, "Graph:   %s\n", graph_path);
  fprintf(stderr, "Params:  %s\n", params_path);
  fprintf(stderr, "Inputs:  %s/\n", input_dir);
  fprintf(stderr, "Outputs: %s/\n", output_dir);
  fprintf(stderr, "========================================\n\n");

  // Initialize TVM platform
  if (TVMPlatformInitialize() != kTvmErrorNoError) {
    fprintf(stderr, "❌ Platform init failed\n");
    return 2;
  }
  fprintf(stderr, "✅ Platform initialized\n");

  // Load graph JSON
  size_t graph_size = 0;
  char* graph_json = read_entire_file(graph_path, &graph_size);
  if (!graph_json) {
    fprintf(stderr, "❌ Failed to read graph: %s\n", graph_path);
    return 2;
  }
  fprintf(stderr, "✅ Graph loaded (%zu bytes)\n", graph_size);

  // Parse input names from graph
  char input_names[16][64];
  int num_inputs = parse_input_names_from_graph(graph_json, input_names, 16);
  if (num_inputs <= 0) {
    fprintf(stderr, "❌ Failed to parse input names from graph\n");
    return 2;
  }
  fprintf(stderr, "✅ Parsed %d input name(s) from graph\n", num_inputs);

  // Get system library
  const TVMModule* syslib = TVMSystemLibEntryPoint();
  TVMModuleHandle mod;
  if (TVMModCreateFromCModule(syslib, &mod) != 0) {
    fprintf(stderr, "❌ Failed to create module handle\n");
    return -1;
  }
  fprintf(stderr, "✅ System library loaded\n");

  // Create graph executor
  DLDevice dev = {kDLCPU, 0};
  TVMGraphExecutor* exec = NULL;
  int rc = TVMGraphExecutor_Create(graph_json, mod, &dev, &exec);
  if (rc != 0) {
    fprintf(stderr, "❌ TVMGraphExecutor_Create failed: %d\n", rc);
    return 2;
  }
  fprintf(stderr, "✅ Graph executor created\n");

  // Load parameters
  char* params_buf = NULL;
  size_t params_size = 0;
  if (load_params_file(params_path, &params_buf, &params_size) != 0) {
    fprintf(stderr, "❌ Failed to read params: %s\n", params_path);
    return 2;
  }
  rc = TVMGraphExecutor_LoadParams(exec, params_buf, (uint32_t)params_size);
  if (rc != 0) {
    fprintf(stderr, "❌ LoadParams failed: %d\n", rc);
    return 2;
  }
  fprintf(stderr, "✅ Parameters loaded (%zu bytes)\n", params_size);

  // ============================================================================
  // Input Loading
  // ============================================================================
  fprintf(stderr, "\n--- Loading Inputs ---\n");

  int inputs_loaded = 0;
  DLTensor* loaded_inputs[16] = {NULL};  // Support up to 16 inputs

  for (int i = 0; i < num_inputs; i++) {
    DLTensor* input_tensor = (DLTensor*)malloc(sizeof(DLTensor));
    memset(input_tensor, 0, sizeof(DLTensor));

    if (load_tensor_from_dir(input_dir, input_names[i], dev, input_tensor) == 0) {
      fprintf(stderr, "✅ Loaded input '%s'\n", input_names[i]);
      print_tensor_info(input_names[i], input_tensor);

      // Set input in executor (returns void)
      TVMGraphExecutor_SetInput(exec, input_names[i], input_tensor);
      loaded_inputs[inputs_loaded] = input_tensor;
      inputs_loaded++;
    } else {
      fprintf(stderr, "⚠️  Failed to load input '%s' from %s/%s.bin\n",
              input_names[i], input_dir, input_names[i]);
      free(input_tensor);
    }
  }

  if (inputs_loaded == 0) {
    fprintf(stderr, "❌ No inputs loaded! Check input directory and file names\n");
    fprintf(stderr, "   Expected files:\n");
    for (int i = 0; i < num_inputs; i++) {
      fprintf(stderr, "     %s/%s.bin\n", input_dir, input_names[i]);
      fprintf(stderr, "     %s/%s.meta.txt\n", input_dir, input_names[i]);
    }
    return 2;
  }

  fprintf(stderr, "✅ Total inputs loaded: %d/%d\n", inputs_loaded, num_inputs);

  // ============================================================================
  // Execute Graph
  // ============================================================================
  fprintf(stderr, "\n--- Executing Graph ---\n");
  TVMGraphExecutor_Run(exec);
  fprintf(stderr, "✅ Graph execution completed\n");

  // ============================================================================
  // Output Retrieval and Saving
  // ============================================================================
  fprintf(stderr, "\n--- Saving Outputs ---\n");

  // Get number of outputs
  int num_outputs = TVMGraphExecutor_GetNumOutputs(exec);
  fprintf(stderr, "Number of outputs: %d\n", num_outputs);

  for (int output_idx = 0; output_idx < num_outputs; output_idx++) {
    // Get output tensor info from executor internals
    uint32_t output_eid = TVMGraphExecutor_GetEntryId((TVMGraphExecutor*)exec,
                                                       ((TVMGraphExecutor*)exec)->outputs[output_idx].node_id,
                                                       ((TVMGraphExecutor*)exec)->outputs[output_idx].index);
    DLTensor* internal_output = &(((TVMGraphExecutor*)exec)->data_entry[output_eid].dl_tensor);

    fprintf(stderr, "\nOutput %d:\n", output_idx);
    print_tensor_info("output", internal_output);

    // Allocate output buffer with correct shape and dtype
    DLTensor out = {0};
    out.device = dev;
    out.ndim = internal_output->ndim;

    // Allocate and copy shape
    int64_t* oshape = (int64_t*)malloc(out.ndim * sizeof(int64_t));
    size_t onumel = 1;
    for (int i = 0; i < out.ndim; i++) {
      oshape[i] = internal_output->shape[i];
      onumel *= (size_t)internal_output->shape[i];
    }
    out.shape = oshape;
    out.dtype = internal_output->dtype;

    // Allocate data buffer
    size_t elem_size = (out.dtype.bits + 7) / 8;
    size_t onbytes = onumel * elem_size;
    void* out_data = NULL;
    TVMPlatformMemoryAllocate(onbytes, dev, &out_data);
    out.data = out_data;

    // Get output from executor
    rc = TVMGraphExecutor_GetOutput(exec, output_idx, &out);
    if (rc != 0) {
      fprintf(stderr, "❌ GetOutput failed for output %d: %d\n", output_idx, rc);
      continue;
    }

    // Print first few values for debugging
    printf("\nOutput %d values (first 16):\n  ", output_idx);
    for (size_t i = 0; i < onumel && i < 16; i++) {
      switch (out.dtype.code) {
        case kDLInt:
          if (out.dtype.bits == 8) printf("%d ", ((int8_t*)out.data)[i]);
          else if (out.dtype.bits == 16) printf("%d ", ((int16_t*)out.data)[i]);
          else if (out.dtype.bits == 32) printf("%d ", ((int32_t*)out.data)[i]);
          break;
        case kDLUInt:
          if (out.dtype.bits == 8) printf("%u ", ((uint8_t*)out.data)[i]);
          else if (out.dtype.bits == 16) printf("%u ", ((uint16_t*)out.data)[i]);
          else if (out.dtype.bits == 32) printf("%u ", ((uint32_t*)out.data)[i]);
          break;
        case kDLFloat:
          if (out.dtype.bits == 32) printf("%.6f ", ((float*)out.data)[i]);
          else if (out.dtype.bits == 64) printf("%.6f ", ((double*)out.data)[i]);
          break;
      }
    }
    printf("\n");

    // Save output
    char output_name[64];
    if (num_outputs == 1) {
      strcpy(output_name, "output");
    } else {
      snprintf(output_name, sizeof(output_name), "output_%d", output_idx);
    }

    if (save_tensor_to_dir(output_dir, output_name, &out) == 0) {
      printf("✅ Saved to %s/%s.npy\n", output_dir, output_name);
    } else {
      fprintf(stderr, "⚠️  Failed to save output %d\n", output_idx);
    }

    // Cleanup output buffer
    free(oshape);
    TVMPlatformMemoryFree(out_data, dev);
  }

  // ============================================================================
  // Cleanup
  // ============================================================================
  fprintf(stderr, "\n--- Cleaning Up ---\n");

  // Free loaded inputs
  for (int i = 0; i < inputs_loaded; i++) {
    if (loaded_inputs[i]) {
      if (loaded_inputs[i]->data) TVMPlatformMemoryFree(loaded_inputs[i]->data, dev);
      if (loaded_inputs[i]->shape) free(loaded_inputs[i]->shape);
      free(loaded_inputs[i]);
    }
  }

  free(graph_json);
  free(params_buf);
  TVMGraphExecutor_Release(&exec);

  fprintf(stderr, "✅ Cleanup completed\n");
  printf("\n========================================\n");
  printf("✅ Execution completed successfully\n");
  printf("========================================\n\n");

  return 0;
}
