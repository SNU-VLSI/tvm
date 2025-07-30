#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdarg.h>
#include "tvmgen_default.h"

// Input data: 1792 float32 values (1, 28, 8, 8)
static float input_data[1792];

// Output data: 10 float32 values (1, 10)
static float output_data[10];

void initialize_test_data() {
    // Initialize with some test pattern
    for (int i = 0; i < 1792; i++) {
        input_data[i] = (float)(i % 100) / 100.0f;  // 0.0 to 0.99
    }
}

int main(void) {
    printf("TVM Model Test on Server (x86_64)\n");
    printf("===================================\n");

    // Initialize test data
    initialize_test_data();

    // Prepare input/output structures
    struct tvmgen_default_inputs inputs = {
        .input_1 = input_data,
    };

    struct tvmgen_default_outputs outputs = {
        .output = output_data,
    };

    // Measure inference time
    clock_t start = clock();

    // Run the model
    int ret = tvmgen_default_run(&inputs, &outputs);

    clock_t end = clock();
    double inference_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

    if (ret != 0) {
        printf("❌ Model execution failed with error: %d\n", ret);
        return -1;
    }

    // Display results
    printf("✅ Inference completed successfully!\n");
    printf("⏱️  Inference time: %.2f ms\n", inference_time);
    printf("\nOutput results:\n");
    for (int i = 0; i < 10; i++) {
        printf("  output[%d] = %.6f\n", i, output_data[i]);
    }

    // Find predicted class (for classification models)
    int max_idx = 0;
    float max_val = output_data[0];
    for (int i = 1; i < 10; i++) {
        if (output_data[i] > max_val) {
            max_val = output_data[i];
            max_idx = i;
        }
    }
    printf("\n🎯 Predicted class: %d (confidence: %.4f)\n", max_idx, max_val);

    printf("\nMemory usage summary:\n");
    printf("  Input size:     %d bytes\n", TVMGEN_DEFAULT_INPUT_1_SIZE);
    printf("  Output size:    %d bytes\n", TVMGEN_DEFAULT_OUTPUT_SIZE);
    printf("  Workspace size: %d bytes\n", TVMGEN_DEFAULT_WORKSPACE_SIZE);

    return 0;
}

// Platform implementations for x86_64
void TVMPlatformAbort(int error_code) {
    fprintf(stderr, "TVM Platform abort: %d\n", error_code);
    exit(error_code);
}

void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t size,
                               int dtype_code_hint, int dtype_bits_hint) {
    void* ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "Failed to allocate %lu bytes\n", size);
    }
    return ptr;
}

int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
    free(ptr);
    return 0;
}

void TVMLogf(const char* msg, ...) {
    va_list args;
    va_start(args, msg);
    vprintf(msg, args);
    va_end(args);
}