#ifndef IMCFLOW_POWER_MEASURE_RUNTIME_H_
#define IMCFLOW_POWER_MEASURE_RUNTIME_H_

#include <stddef.h>
#include <stdio.h>

#include "power_region.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  IMCFLOW_POWER_SCOPE_MODEL = 0,
  IMCFLOW_POWER_SCOPE_REGION = 1,
  IMCFLOW_POWER_SCOPE_TILE = 2
} imcflow_power_scope_t;

typedef struct {
  power_region_context_t region;
  int selected;
  int pass_count;
  int started;
  imcflow_power_scope_t scope;
  char name[129];
} power_measure_scope_context_t;

int power_measure_runtime_start(void);
int power_measure_runtime_finish(void);
int power_measure_runtime_is_enabled(void);
int power_measure_runtime_is_degraded(void);
int power_measure_runtime_scope_is(imcflow_power_scope_t scope);

int power_measure_scope_begin(power_measure_scope_context_t *ctx,
                              imcflow_power_scope_t scope,
                              const char *name);
int power_measure_scope_next(power_measure_scope_context_t *ctx);
int power_measure_scope_end(power_measure_scope_context_t *ctx);
int power_measure_runtime_model_start_after_first_warmup(void);

void power_measure_runtime_phase(const char *phase);
void power_measure_runtime_sample(size_t sample_index);
void power_measure_runtime_clear_sample(void);
void power_measure_runtime_event(const char *name);
int power_measure_runtime_print_build_info(FILE *stream);

#define TVM_POWER_REGION_BEGIN(SCOPE, NAME)                                  \
  do {                                                                        \
    power_measure_scope_context_t _tvm_power_scope_ctx;                       \
    (void)power_measure_scope_begin(                                          \
        &_tvm_power_scope_ctx, (SCOPE), (NAME));                              \
    while (power_measure_scope_next(&_tvm_power_scope_ctx)) {

#define TVM_POWER_REGION_END()                                                \
    }                                                                         \
    (void)power_measure_scope_end(&_tvm_power_scope_ctx);                     \
  } while (0)

#ifdef __cplusplus
}
#endif

#endif  /* IMCFLOW_POWER_MEASURE_RUNTIME_H_ */
