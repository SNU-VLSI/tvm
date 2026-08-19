#include "power_measure_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef IMCFLOW_BUILD_TVM_GIT_REV
#define IMCFLOW_BUILD_TVM_GIT_REV "unknown"
#endif
#ifndef IMCFLOW_BUILD_MEASUREMENT_UTILS_GIT_REV
#define IMCFLOW_BUILD_MEASUREMENT_UTILS_GIT_REV "unknown"
#endif
#ifndef IMCFLOW_BUILD_TREE_DIRTY
#define IMCFLOW_BUILD_TREE_DIRTY 1
#endif

static int g_power_enabled;
static int g_power_finished;
static int g_power_degraded;
static int g_power_atexit_installed;
static imcflow_power_scope_t g_power_scope = IMCFLOW_POWER_SCOPE_REGION;
static power_region_policy_t g_power_policy;
static power_measure_scope_context_t *g_pending_model_context;

static int parse_u64(const char *text, uint64_t *value)
{
  char *end = NULL;
  unsigned long long parsed;
  if (!text || !value || text[0] == '\0' || text[0] == '-')
    return -1;
  parsed = strtoull(text, &end, 10);
  if (!end || *end != '\0')
    return -1;
  *value = (uint64_t)parsed;
  return 0;
}

static int parse_double(const char *text, double *value)
{
  char *end = NULL;
  double parsed;
  if (!text || !value || text[0] == '\0')
    return -1;
  parsed = strtod(text, &end);
  if (!end || *end != '\0' || parsed < 0.0)
    return -1;
  *value = parsed;
  return 0;
}

static void runtime_atexit(void)
{
  if (power_measure_runtime_finish() != 0)
    fprintf(stderr, "[POWER] atexit finalize failed: %s\n",
            power_region_last_error());
}

int power_measure_runtime_start(void)
{
  const char *request = getenv("IMCFLOW_POWER_REQUEST");
  const char *scope = getenv("IMCFLOW_POWER_SCOPE");
  const char *loop = getenv("IMCFLOW_POWER_LOOP_ENABLE");
  const char *min_samples = getenv("IMCFLOW_POWER_MIN_SAMPLES");
  const char *min_seconds = getenv("IMCFLOW_POWER_MIN_SECONDS");
  uint64_t parsed_samples = 0;
  double parsed_seconds = 0.0;

  if (g_power_enabled && !g_power_finished)
    return 0;
  if (!request || request[0] == '\0')
    return 0;
  if (!scope || scope[0] == '\0' || strcmp(scope, "REGION") == 0)
    g_power_scope = IMCFLOW_POWER_SCOPE_REGION;
  else if (strcmp(scope, "MODEL") == 0)
    g_power_scope = IMCFLOW_POWER_SCOPE_MODEL;
  else if (strcmp(scope, "TILE") == 0)
    g_power_scope = IMCFLOW_POWER_SCOPE_TILE;
  else {
    fprintf(stderr, "[POWER] invalid IMCFLOW_POWER_SCOPE: %s\n", scope);
    return -1;
  }
  if (loop && strcmp(loop, "0") != 0 && strcmp(loop, "1") != 0) {
    fprintf(stderr, "[POWER] IMCFLOW_POWER_LOOP_ENABLE must be 0 or 1\n");
    return -1;
  }
  if ((min_samples && parse_u64(min_samples, &parsed_samples) != 0) ||
      (min_seconds && parse_double(min_seconds, &parsed_seconds) != 0)) {
    fprintf(stderr, "[POWER] invalid minimum loop policy\n");
    return -1;
  }
  g_power_policy.loop_enable = loop ? atoi(loop) : 0;
  g_power_policy.min_samples = parsed_samples;
  g_power_policy.min_seconds = parsed_seconds;
  if (g_power_scope != IMCFLOW_POWER_SCOPE_MODEL) {
    if (g_power_policy.loop_enable)
      fprintf(stderr,
              "[POWER] loop ignored: loop is supported only for MODEL scope\n");
    g_power_policy.loop_enable = 0;
    g_power_policy.min_samples = 0;
    g_power_policy.min_seconds = 0.0;
  }
  if (power_region_runtime_init(request) != 0) {
    fprintf(stderr, "[POWER] runtime init failed: %s\n",
            power_region_last_error());
    return -1;
  }
  g_power_enabled = 1;
  g_power_finished = 0;
  g_power_degraded = 0;
  if (!g_power_atexit_installed && atexit(runtime_atexit) != 0) {
    (void)power_region_runtime_shutdown();
    g_power_enabled = 0;
    return -1;
  }
  g_power_atexit_installed = 1;
  fprintf(stderr,
          "[POWER] scope=%s loop=%d min_samples=%llu min_seconds=%.9g "
          "tvm_revision=%s measurement_utils_revision=%s build_dirty=%d\n",
          scope ? scope : "REGION", g_power_policy.loop_enable,
          (unsigned long long)g_power_policy.min_samples,
          g_power_policy.min_seconds, IMCFLOW_BUILD_TVM_GIT_REV,
          IMCFLOW_BUILD_MEASUREMENT_UTILS_GIT_REV, IMCFLOW_BUILD_TREE_DIRTY);
  return 0;
}

int power_measure_runtime_scope_is(imcflow_power_scope_t scope)
{
  return g_power_enabled && !g_power_finished && g_power_scope == scope;
}

int power_measure_scope_begin(power_measure_scope_context_t *ctx,
                              imcflow_power_scope_t scope,
                              const char *name)
{
  if (!ctx)
    return -1;
  memset(ctx, 0, sizeof(*ctx));
  ctx->selected = power_measure_runtime_scope_is(scope);
  ctx->scope = scope;
  if (!ctx->selected)
    return 0;
  if (scope == IMCFLOW_POWER_SCOPE_MODEL) {
    if (!name || name[0] == '\0' || strlen(name) >= sizeof(ctx->name) ||
        g_pending_model_context || power_region_is_active()) {
      g_power_degraded = 1;
      return -1;
    }
    snprintf(ctx->name, sizeof(ctx->name), "%s", name);
    g_pending_model_context = ctx;
    return 0;
  }
  if (power_region_begin(&ctx->region, name, g_power_policy) != 0) {
    g_power_degraded = 1;
    fprintf(stderr, "[POWER] region begin failed: %s\n",
            power_region_last_error());
    return -1;
  }
  ctx->started = 1;
  return 0;
}

int power_measure_runtime_model_start_after_first_warmup(void)
{
  power_measure_scope_context_t *ctx = g_pending_model_context;
  if (!ctx)
    return 0;
  g_pending_model_context = NULL;
  if (power_region_begin(&ctx->region, ctx->name, g_power_policy) != 0) {
    ctx->started = ctx->region.active;
    g_power_degraded = 1;
    fprintf(stderr, "[POWER] MODEL begin failed: %s\n",
            power_region_last_error());
    return -1;
  }
  ctx->started = 1;
  if (!power_region_next(&ctx->region)) {
    g_power_degraded = 1;
    return -1;
  }
  return 0;
}

int power_measure_scope_next(power_measure_scope_context_t *ctx)
{
  if (!ctx)
    return 0;
  if (!ctx->selected)
    return ctx->pass_count++ == 0;
  if (ctx->scope == IMCFLOW_POWER_SCOPE_MODEL) {
    if (ctx->pass_count++ == 0)
      return 1;
    if (!ctx->started) {
      g_power_degraded = 1;
      return 0;
    }
  }
  return power_region_next(&ctx->region);
}

int power_measure_scope_end(power_measure_scope_context_t *ctx)
{
  int result;
  if (!ctx || !ctx->selected)
    return 0;
  if (ctx->scope == IMCFLOW_POWER_SCOPE_MODEL) {
    if (g_pending_model_context == ctx)
      g_pending_model_context = NULL;
    if (!ctx->started) {
      g_power_degraded = 1;
      return -1;
    }
  }
  result = power_region_end(&ctx->region);
  if (result != 0) {
    g_power_degraded = 1;
    fprintf(stderr, "[POWER] region end failed: %s\n",
            power_region_last_error());
  }
  return result;
}

static void set_tag(const char *key, const char *value)
{
  if (g_power_enabled && !g_power_finished && power_region_is_active() &&
      power_tag_set(key, value) != 0)
    g_power_degraded = 1;
}

void power_measure_runtime_phase(const char *phase)
{
  set_tag("phase", phase);
}

void power_measure_runtime_sample(size_t sample_index)
{
  char value[32];
  snprintf(value, sizeof(value), "%zu", sample_index);
  set_tag("sample", value);
}

void power_measure_runtime_clear_sample(void)
{
  if (g_power_enabled && !g_power_finished && power_region_is_active() &&
      power_tag_clear("sample") != 0)
    g_power_degraded = 1;
}

void power_measure_runtime_event(const char *name)
{
  if (g_power_enabled && !g_power_finished && power_region_is_active() &&
      power_tag_event(name) != 0)
    g_power_degraded = 1;
}

int power_measure_runtime_finish(void)
{
  int result;
  if (!g_power_enabled || g_power_finished)
    return 0;
  g_power_finished = 1;
  result = power_region_runtime_shutdown();
  if (result != 0)
    g_power_degraded = 1;
  return g_power_degraded ? -1 : 0;
}

int power_measure_runtime_is_enabled(void) { return g_power_enabled; }
int power_measure_runtime_is_degraded(void) { return g_power_degraded; }

int power_measure_runtime_print_build_info(FILE *stream)
{
  if (!stream)
    return -1;
  return fprintf(stream,
                 "IMCFLOW_POWER_BUILD_INFO tvm=%s measurement_utils=%s dirty=%d\n",
                 IMCFLOW_BUILD_TVM_GIT_REV,
                 IMCFLOW_BUILD_MEASUREMENT_UTILS_GIT_REV,
                 IMCFLOW_BUILD_TREE_DIRTY) < 0 ? -1 : 0;
}
