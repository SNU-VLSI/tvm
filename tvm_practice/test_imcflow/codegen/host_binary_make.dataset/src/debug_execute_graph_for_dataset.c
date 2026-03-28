/**
 * TVM Graph Executor for Dataset Evaluation
 *
 * This executable runs a TVM compiled model on a dataset (e.g., CIFAR-10)
 * and computes classification accuracy.
 *
 * Features:
 * - Loads images and labels from NumPy .npy files
 * - Runs inference on multiple samples
 * - Computes and reports accuracy
 *
 * Usage:
 *   execute_graph_for_dataset <graph.json> <params.params> <images.npy> <labels.npy> [num_samples_or_indices] [output_path]
 *
 * Arguments:
 *   graph.json    - Path to TVM graph JSON file
 *   params.params - Path to TVM parameters file
 *   images.npy    - Path to images dataset (shape: [N, C, H, W], dtype: float32)
 *   labels.npy    - Path to labels dataset (shape: [N], dtype: int64 or similar)
 *   num_samples_or_indices - Optional: number of samples (e.g. 100) or
 *                            comma-separated indices (e.g. 1,2,3,4). Default: all.
 *   output_path   - Optional: path to save results (default: /tmp/tvm_dataset_results.txt)
 *
 * Example:
 *   ./execute_graph_for_dataset \
 *       mlf/executor-config/graph/default.graph \
 *       mlf/parameters/default.params \
 *       /path/to/cifar10/images.npy \
 *       /path/to/cifar10/labels.npy \
 *       100
 *
 *   # Or with specific indices:
 *   ./execute_graph_for_dataset \
 *       mlf/executor-config/graph/default.graph \
 *       mlf/parameters/default.params \
 *       /path/to/cifar10/images.npy \
 *       /path/to/cifar10/labels.npy \
 *       0,5,10,15,20
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
#include <tvm/runtime/crt/internal/graph_executor/graph_executor.h>

// Dataset loader
#include "npy_dataset_loader.h"
// Output writer (for per-node debug dumping)
#include "test_output_writer.h"

// Result file handle (global for use by helper functions)
static FILE* g_result_file = NULL;

// Global failure flag: set by IMCFlow kernel on timeout, checked by main loop
volatile int g_imcflow_kernel_failed = 0;

#define DEFAULT_RESULT_PATH "/tmp/tvm_dataset_results.txt"
#define MAX_INDICES 10000

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

/**
 * Find the input name that is not a parameter (p0, p1, etc.)
 * Returns the first non-parameter input name found.
 */
static int find_model_input_name(const char* graph_json, char* out_name, int max_len) {
  // Find arg_nodes array
  const char* arg_nodes_start = strstr(graph_json, "\"arg_nodes\": [");
  if (!arg_nodes_start) return -1;

  // Parse arg_nodes indices
  int arg_node_indices[16];
  int num_arg_nodes = 0;

  const char* p = arg_nodes_start + strlen("\"arg_nodes\": [");
  while (*p && *p != ']' && num_arg_nodes < 16) {
    while (*p && isspace(*p)) p++;
    if (isdigit(*p)) {
      arg_node_indices[num_arg_nodes++] = atoi(p);
      while (*p && isdigit(*p)) p++;
    }
    if (*p == ',') p++;
  }

  // Find nodes array
  const char* nodes_start = strstr(graph_json, "\"nodes\": [");
  if (!nodes_start) return -1;

  // Parse nodes and find first non-parameter input
  p = nodes_start + strlen("\"nodes\": [");
  int node_idx = 0;

  while (*p && *p != ']') {
    while (*p && *p != '{') p++;
    if (*p != '{') break;

    const char* node_start = p;
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
      const char* name_key = strstr(node_start, "\"name\": \"");
      if (name_key && name_key < node_end) {
        const char* name_start = name_key + strlen("\"name\": \"");
        const char* name_end = strchr(name_start, '"');
        if (name_end && name_end < node_end) {
          int name_len = name_end - name_start;
          if (name_len < max_len) {
            strncpy(out_name, name_start, name_len);
            out_name[name_len] = '\0';

            // Skip parameter names (p0, p1, p2, ...)
            if (!(out_name[0] == 'p' && isdigit(out_name[1]) &&
                  (out_name[2] == '\0' || isdigit(out_name[2])))) {
              return 0;  // Found non-parameter input
            }
          }
        }
      }
    }

    node_idx++;
    p = node_end + 1;
  }

  return -1;  // No model input found
}

/**
 * Get argmax from output tensor based on dtype
 */
static int get_argmax(const DLTensor* tensor) {
  size_t numel = 1;
  for (int i = 0; i < tensor->ndim; i++) {
    numel *= (size_t)tensor->shape[i];
  }

  if (tensor->dtype.code == kDLFloat && tensor->dtype.bits == 32) {
    return argmax_float32((float*)tensor->data, (int)numel);
  } else if (tensor->dtype.code == kDLInt && tensor->dtype.bits == 32) {
    return argmax_int32((int32_t*)tensor->data, (int)numel);
  } else if (tensor->dtype.code == kDLInt && tensor->dtype.bits == 8) {
    return argmax_int8((int8_t*)tensor->data, (int)numel);
  }

  fprintf(stderr, "Unsupported output dtype for argmax\n");
  return -1;
}

/**
 * Print class scores for debugging
 * Shows all class scores, predicted class, label, and O/X result
 */
static void print_class_scores(const DLTensor* tensor, int num_classes,
                                int predicted, int64_t label, size_t sample_idx) {
  FILE* outs[] = { stdout, g_result_file };
  int num_outs = g_result_file ? 2 : 1;

  for (int o = 0; o < num_outs; o++) {
    FILE* f = outs[o];
    fprintf(f, "\n[Sample %zu] ", sample_idx);
    fprintf(f, "Scores: [");

    // Print all class scores based on dtype
    if (tensor->dtype.code == kDLFloat && tensor->dtype.bits == 32) {
      float* data = (float*)tensor->data;
      for (int i = 0; i < num_classes; i++) {
        fprintf(f, "%.4f%s", data[i], i < num_classes - 1 ? ", " : "");
      }
    } else if (tensor->dtype.code == kDLInt && tensor->dtype.bits == 32) {
      int32_t* data = (int32_t*)tensor->data;
      for (int i = 0; i < num_classes; i++) {
        fprintf(f, "%d%s", data[i], i < num_classes - 1 ? ", " : "");
      }
    } else if (tensor->dtype.code == kDLInt && tensor->dtype.bits == 8) {
      int8_t* data = (int8_t*)tensor->data;
      for (int i = 0; i < num_classes; i++) {
        fprintf(f, "%d%s", data[i], i < num_classes - 1 ? ", " : "");
      }
    }
    fprintf(f, "]\n");

    // Print predicted, label, and result
    int is_correct = (predicted == (int)label);
    fprintf(f, "         Predicted: %d, Label: %lld, Result: %s\n",
           predicted, (long long)label, is_correct ? "O" : "X");
  }
}

// ============================================================================
// Main Execution Logic
// ============================================================================

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "Usage: %s <graph.json> <params.params> <images.npy> <labels.npy> [num_samples_or_indices] [output_path]\n", argv[0]);
    fprintf(stderr, "\nExample:\n");
    fprintf(stderr, "  %s mlf/executor-config/graph/default.graph mlf/parameters/default.params \\\n", argv[0]);
    fprintf(stderr, "      /path/to/images.npy /path/to/labels.npy 100\n");
    fprintf(stderr, "  %s mlf/executor-config/graph/default.graph mlf/parameters/default.params \\\n", argv[0]);
    fprintf(stderr, "      /path/to/images.npy /path/to/labels.npy 0,5,10,15\n");
    return 1;
  }

  const char* graph_path = argv[1];
  const char* params_path = argv[2];
  const char* images_path = argv[3];
  const char* labels_path = argv[4];
  const char* result_path = argc > 6 ? argv[6] : DEFAULT_RESULT_PATH;

  // Parse 5th argument: comma-separated indices or num_samples
  int num_samples_arg = 0;
  int sample_indices[MAX_INDICES];
  int num_indices = 0;

  if (argc > 5) {
    // Check if argument contains a comma -> indices mode
    if (strchr(argv[5], ',') != NULL) {
      char* indices_str = strdup(argv[5]);
      char* token = strtok(indices_str, ",");
      while (token != NULL && num_indices < MAX_INDICES) {
        sample_indices[num_indices++] = atoi(token);
        token = strtok(NULL, ",");
      }
      free(indices_str);
      fprintf(stderr, "Index mode: %d specific indices\n", num_indices);
    } else {
      num_samples_arg = atoi(argv[5]);
    }
  }

  // Open result file
  g_result_file = fopen(result_path, "w");
  if (!g_result_file) {
    fprintf(stderr, "Warning: Could not open result file '%s', results will only go to stdout\n", result_path);
  } else {
    fprintf(stderr, "Results will be saved to: %s\n", result_path);
  }

  fprintf(stderr, "\n");
  fprintf(stderr, "========================================\n");
  fprintf(stderr, "  TVM Dataset Evaluation\n");
  fprintf(stderr, "========================================\n");
  fprintf(stderr, "Graph:   %s\n", graph_path);
  fprintf(stderr, "Params:  %s\n", params_path);
  fprintf(stderr, "Images:  %s\n", images_path);
  fprintf(stderr, "Labels:  %s\n", labels_path);
  fprintf(stderr, "========================================\n\n");

  // Write header to result file
  if (g_result_file) {
    fprintf(g_result_file, "========================================\n");
    fprintf(g_result_file, "  TVM Dataset Evaluation Results\n");
    fprintf(g_result_file, "========================================\n");
    fprintf(g_result_file, "Graph:   %s\n", graph_path);
    fprintf(g_result_file, "Params:  %s\n", params_path);
    fprintf(g_result_file, "Images:  %s\n", images_path);
    fprintf(g_result_file, "Labels:  %s\n", labels_path);
    fprintf(g_result_file, "========================================\n");
  }

  // ============================================================================
  // Load Datasets
  // ============================================================================
  fprintf(stderr, "--- Loading Datasets ---\n");

  NpyDataset images, labels;
  if (load_npy_dataset(images_path, &images) != 0) {
    fprintf(stderr, "Failed to load images dataset\n");
    return 2;
  }

  if (load_npy_dataset(labels_path, &labels) != 0) {
    fprintf(stderr, "Failed to load labels dataset\n");
    free_npy_dataset(&images);
    return 2;
  }

  // Verify dataset sizes match
  if (images.num_samples != labels.num_samples) {
    fprintf(stderr, "Mismatch: %zu images but %zu labels\n",
            images.num_samples, labels.num_samples);
    free_npy_dataset(&images);
    free_npy_dataset(&labels);
    return 2;
  }

  // Determine number of samples to evaluate
  size_t num_samples;
  if (num_indices > 0) {
    num_samples = (size_t)num_indices;
    // Validate indices
    for (int idx = 0; idx < num_indices; idx++) {
      if (sample_indices[idx] < 0 || (size_t)sample_indices[idx] >= images.num_samples) {
        fprintf(stderr, "Error: index %d out of range [0, %zu)\n",
                sample_indices[idx], images.num_samples);
        free_npy_dataset(&images);
        free_npy_dataset(&labels);
        return 2;
      }
    }
    fprintf(stderr, "\nWill evaluate %zu specific indices\n\n", num_samples);
  } else {
    num_samples = images.num_samples;
    if (num_samples_arg > 0 && (size_t)num_samples_arg < num_samples) {
      num_samples = (size_t)num_samples_arg;
    }
    fprintf(stderr, "\nWill evaluate %zu samples\n\n", num_samples);
  }

  // ============================================================================
  // Initialize TVM
  // ============================================================================
  fprintf(stderr, "--- Initializing TVM ---\n");

  if (TVMPlatformInitialize() != kTvmErrorNoError) {
    fprintf(stderr, "Platform init failed\n");
    free_npy_dataset(&images);
    free_npy_dataset(&labels);
    return 2;
  }
  fprintf(stderr, "Platform initialized\n");

  // Load graph JSON
  size_t graph_size = 0;
  char* graph_json = read_entire_file(graph_path, &graph_size);
  if (!graph_json) {
    fprintf(stderr, "Failed to read graph: %s\n", graph_path);
    free_npy_dataset(&images);
    free_npy_dataset(&labels);
    return 2;
  }
  fprintf(stderr, "Graph loaded (%zu bytes)\n", graph_size);

  // Find model input name
  char input_name[64];
  if (find_model_input_name(graph_json, input_name, sizeof(input_name)) != 0) {
    fprintf(stderr, "Failed to find model input name in graph\n");
    free(graph_json);
    free_npy_dataset(&images);
    free_npy_dataset(&labels);
    return 2;
  }
  fprintf(stderr, "Model input name: '%s'\n", input_name);

  // Get system library
  const TVMModule* syslib = TVMSystemLibEntryPoint();
  TVMModuleHandle mod;
  if (TVMModCreateFromCModule(syslib, &mod) != 0) {
    fprintf(stderr, "Failed to create module handle\n");
    free(graph_json);
    free_npy_dataset(&images);
    free_npy_dataset(&labels);
    return 2;
  }
  fprintf(stderr, "System library loaded\n");

  // Create graph executor
  DLDevice dev = {kDLCPU, 0};
  TVMGraphExecutor* exec = NULL;
  int rc = TVMGraphExecutor_Create(graph_json, mod, &dev, &exec);
  if (rc != 0) {
    fprintf(stderr, "TVMGraphExecutor_Create failed: %d\n", rc);
    free(graph_json);
    free_npy_dataset(&images);
    free_npy_dataset(&labels);
    return 2;
  }
  fprintf(stderr, "Graph executor created\n");

  // Load parameters
  char* params_buf = NULL;
  size_t params_size = 0;
  if (load_params_file(params_path, &params_buf, &params_size) != 0) {
    fprintf(stderr, "Failed to read params: %s\n", params_path);
    free(graph_json);
    free_npy_dataset(&images);
    free_npy_dataset(&labels);
    TVMGraphExecutor_Release(&exec);
    return 2;
  }
  rc = TVMGraphExecutor_LoadParams(exec, params_buf, (uint32_t)params_size);
  if (rc != 0) {
    fprintf(stderr, "LoadParams failed: %d\n", rc);
    free(graph_json);
    free(params_buf);
    free_npy_dataset(&images);
    free_npy_dataset(&labels);
    TVMGraphExecutor_Release(&exec);
    return 2;
  }
  fprintf(stderr, "Parameters loaded (%zu bytes)\n", params_size);

  // ============================================================================
  // Prepare Input Tensor (reusable)
  // ============================================================================
  fprintf(stderr, "\n--- Preparing Input Tensor ---\n");

  // Create input tensor with shape (1, C, H, W) from dataset shape (N, C, H, W)
  DLTensor input_tensor = {0};
  input_tensor.device = dev;
  input_tensor.ndim = images.ndim;  // Same ndim, but first dim is 1

  int64_t* input_shape = (int64_t*)malloc(images.ndim * sizeof(int64_t));
  input_shape[0] = 1;  // Batch size = 1
  for (int i = 1; i < images.ndim; i++) {
    input_shape[i] = images.shape[i];
  }
  input_tensor.shape = input_shape;
  input_tensor.dtype = images.dtype;

  // Allocate input data buffer
  void* input_data = NULL;
  if (TVMPlatformMemoryAllocate(images.sample_size, dev, &input_data) != 0) {
    fprintf(stderr, "Failed to allocate input buffer\n");
    free(input_shape);
    free(graph_json);
    free(params_buf);
    free_npy_dataset(&images);
    free_npy_dataset(&labels);
    TVMGraphExecutor_Release(&exec);
    return 2;
  }
  input_tensor.data = input_data;

  fprintf(stderr, "Input shape: [");
  for (int i = 0; i < input_tensor.ndim; i++) {
    fprintf(stderr, "%lld%s", (long long)input_tensor.shape[i],
            i < input_tensor.ndim - 1 ? ", " : "");
  }
  fprintf(stderr, "]\n");

  // ============================================================================
  // Prepare Output Tensor
  // ============================================================================
  // Get output info from executor
  uint32_t output_eid = TVMGraphExecutor_GetEntryId(exec,
                                                     exec->outputs[0].node_id,
                                                     exec->outputs[0].index);
  DLTensor* internal_output = &(exec->data_entry[output_eid].dl_tensor);

  DLTensor output_tensor = {0};
  output_tensor.device = dev;
  output_tensor.ndim = internal_output->ndim;
  output_tensor.dtype = internal_output->dtype;

  int64_t* output_shape = (int64_t*)malloc(internal_output->ndim * sizeof(int64_t));
  size_t output_numel = 1;
  for (int i = 0; i < internal_output->ndim; i++) {
    output_shape[i] = internal_output->shape[i];
    output_numel *= (size_t)internal_output->shape[i];
  }
  output_tensor.shape = output_shape;

  size_t output_elem_size = (output_tensor.dtype.bits + 7) / 8;
  size_t output_bytes = output_numel * output_elem_size;
  void* output_data = NULL;
  if (TVMPlatformMemoryAllocate(output_bytes, dev, &output_data) != 0) {
    fprintf(stderr, "Failed to allocate output buffer\n");
    free(input_shape);
    free(output_shape);
    TVMPlatformMemoryFree(input_data, dev);
    free(graph_json);
    free(params_buf);
    free_npy_dataset(&images);
    free_npy_dataset(&labels);
    TVMGraphExecutor_Release(&exec);
    return 2;
  }
  output_tensor.data = output_data;

  fprintf(stderr, "Output shape: [");
  for (int i = 0; i < output_tensor.ndim; i++) {
    fprintf(stderr, "%lld%s", (long long)output_tensor.shape[i],
            i < output_tensor.ndim - 1 ? ", " : "");
  }
  fprintf(stderr, "] (%zu elements)\n", output_numel);

  int num_classes = (int)output_numel;
  if (output_tensor.ndim >= 2) {
    // If output is (batch, classes), use last dimension
    num_classes = (int)output_tensor.shape[output_tensor.ndim - 1];
  }
  fprintf(stderr, "Number of classes: %d\n", num_classes);

  // ============================================================================
  // Main Evaluation Loop
  // ============================================================================
  fprintf(stderr, "\n--- Evaluating Dataset ---\n");
  printf("\nStarting evaluation of %zu samples...\n", num_samples);
  if (g_result_file) {
    fprintf(g_result_file, "\nStarting evaluation of %zu samples...\n", num_samples);
  }

  int correct = 0;
  int failed_count = 0;
  int failed_indices[MAX_INDICES];

  for (size_t iter = 0; iter < num_samples; iter++) {
    // Determine actual sample index
    size_t sample_idx = (num_indices > 0) ? (size_t)sample_indices[iter] : iter;

    // Reset failure flag before each inference
    g_imcflow_kernel_failed = 0;

    // Copy image data to input tensor
    void* sample_data = get_sample(&images, sample_idx);
    if (!sample_data) {
      fprintf(stderr, "Failed to get sample %zu\n", sample_idx);
      continue;
    }
    memcpy(input_tensor.data, sample_data, images.sample_size);

    // Set input
    TVMGraphExecutor_SetInput(exec, input_name, &input_tensor);

    // Run inference (node-by-node debug mode)
    // Create debug output directory for this sample
    char debug_dir[512];
    snprintf(debug_dir, sizeof(debug_dir), "/tmp/tvm_debug_nodes/sample_%zu", sample_idx);
    {
      char mkdir_cmd[600];
      snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", debug_dir);
      system(mkdir_cmd);
    }

    int sample_failed = 0;
    for (uint32_t nid = 0; nid < exec->op_execs_count; nid++) {
      if (exec->op_execs[nid].fexec) {
        exec->op_execs[nid].Call(&(exec->op_execs[nid]));

        if (g_imcflow_kernel_failed) {
          fprintf(stderr, "  ⚠️  [%u] IMCFlow kernel timeout on node '%s' (sample %zu)\n",
                  nid, exec->nodes[nid].name, sample_idx);
          sample_failed = 1;
          break;
        }

        // Save each output of this node
        for (uint32_t out_idx = 0; out_idx < exec->nodes[nid].param.num_outputs; out_idx++) {
          uint32_t eid = TVMGraphExecutor_GetEntryId(exec, nid, out_idx);
          DLTensor* tensor = &(exec->data_entry[eid].dl_tensor);

          char fname[256];
          if (exec->nodes[nid].param.num_outputs == 1) {
            snprintf(fname, sizeof(fname), "%03u_%s", nid, exec->nodes[nid].name);
          } else {
            snprintf(fname, sizeof(fname), "%03u_%s_%u", nid, exec->nodes[nid].name, out_idx);
          }
          save_tensor_to_dir(debug_dir, fname, tensor);
        }
      }
    }

    if (sample_failed) {
      fprintf(stderr, "[SKIP] Sample %zu failed (timeout)\n", sample_idx);
      if (g_result_file) {
        fprintf(g_result_file, "\n[Sample %zu] FAILED (timeout)\n", sample_idx);
      }
      failed_indices[failed_count++] = (int)sample_idx;
      g_imcflow_kernel_failed = 0;
      continue;
    }

    // Check if kernel failed (timeout)
    if (g_imcflow_kernel_failed) {
      fprintf(stderr, "[SKIP] Sample %zu failed (timeout)\n", sample_idx);
      if (g_result_file) {
        fprintf(g_result_file, "\n[Sample %zu] FAILED (timeout)\n", sample_idx);
      }
      failed_indices[failed_count++] = (int)sample_idx;
      g_imcflow_kernel_failed = 0;
      continue;
    }

    // Get output
    rc = TVMGraphExecutor_GetOutput(exec, 0, &output_tensor);
    if (rc != 0) {
      fprintf(stderr, "GetOutput failed for sample %zu: %d\n", sample_idx, rc);
      continue;
    }

    // Get prediction (argmax)
    int predicted = get_argmax(&output_tensor);

    // Get ground truth label
    int64_t label = get_label(&labels, sample_idx);

    // Check if correct
    if (predicted == (int)label) {
      correct++;
    }

    // Debug: Print class scores for each sample
    print_class_scores(&output_tensor, num_classes, predicted, label, sample_idx);

    // Print progress every 100 samples
    int evaluated = (int)(iter + 1) - failed_count;
    if ((iter + 1) % 100 == 0 || iter + 1 == num_samples) {
      double accuracy = evaluated > 0 ? 100.0 * correct / evaluated : 0.0;
      printf("Progress: %zu/%zu, Correct: %d/%d, Failed: %d, Accuracy: %.2f%%\n",
             iter + 1, num_samples, correct, evaluated, failed_count, accuracy);
      if (g_result_file) {
        fprintf(g_result_file, "Progress: %zu/%zu, Correct: %d/%d, Failed: %d, Accuracy: %.2f%%\n",
                iter + 1, num_samples, correct, evaluated, failed_count, accuracy);
        fflush(g_result_file);
      }
    }
  }

  // ============================================================================
  // Final Results
  // ============================================================================
  int evaluated = (int)num_samples - failed_count;
  double final_accuracy = evaluated > 0 ? 100.0 * correct / evaluated : 0.0;

  FILE* outs[] = { stdout, g_result_file };
  int num_outs = g_result_file ? 2 : 1;

  for (int o = 0; o < num_outs; o++) {
    FILE* f = outs[o];
    fprintf(f, "\n========================================\n");
    fprintf(f, "  FINAL RESULTS\n");
    fprintf(f, "========================================\n");
    fprintf(f, "Total Samples: %zu\n", num_samples);
    fprintf(f, "Evaluated:     %d\n", evaluated);
    fprintf(f, "Failed:        %d\n", failed_count);
    fprintf(f, "Correct:       %d\n", correct);
    fprintf(f, "Accuracy:      %.2f%% (%d/%d)\n", final_accuracy, correct, evaluated);
    if (failed_count > 0) {
      fprintf(f, "Failed indices:");
      for (int i = 0; i < failed_count; i++) {
        fprintf(f, " %d", failed_indices[i]);
      }
      fprintf(f, "\n");
    }
    fprintf(f, "========================================\n\n");
  }

  // ============================================================================
  // Cleanup
  // ============================================================================
  fprintf(stderr, "--- Cleaning Up ---\n");

  if (g_result_file) {
    fclose(g_result_file);
    g_result_file = NULL;
    fprintf(stderr, "Results saved to: %s\n", result_path);
  }

  free(input_shape);
  free(output_shape);
  TVMPlatformMemoryFree(input_data, dev);
  TVMPlatformMemoryFree(output_data, dev);
  free(graph_json);
  free(params_buf);
  free_npy_dataset(&images);
  free_npy_dataset(&labels);
  TVMGraphExecutor_Release(&exec);

  fprintf(stderr, "Cleanup completed\n");
  return 0;
}
