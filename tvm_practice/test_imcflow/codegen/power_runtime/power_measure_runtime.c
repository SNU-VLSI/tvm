#include "power_measure_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dmm_measure.h"

#ifndef IMCFLOW_BUILD_TVM_GIT_REV
#define IMCFLOW_BUILD_TVM_GIT_REV "unknown"
#endif

#ifndef IMCFLOW_BUILD_MEASUREMENT_UTILS_GIT_REV
#define IMCFLOW_BUILD_MEASUREMENT_UTILS_GIT_REV "unknown"
#endif

#ifndef IMCFLOW_BUILD_TREE_DIRTY
#define IMCFLOW_BUILD_TREE_DIRTY 1
#endif


static int g_power_enabled = 0;
static int g_power_finished = 0;
static int g_power_degraded = 0;
static int g_power_cleanup_started = 0;
static int g_power_region_scope = 0;
static int g_power_region_active = 0;
static int g_power_atexit_installed = 0;
static const char *g_power_request_path = NULL;


static void record_tag_result(int result, const char *operation)
{
  if (result == 0)
    return;
  g_power_degraded = 1;
  fprintf(stderr, "[POWER] %s failed: %s\n", operation, dmm_last_error());
}


static void power_measure_runtime_atexit(void)
{
  if (power_measure_runtime_finish() != 0) {
    fprintf(stderr, "[POWER] atexit finalize failed: %s\n", dmm_last_error());
  }
}


int power_measure_runtime_start(void)
{
  const char *request_path;
  const char *scope;

  if (g_power_enabled && !g_power_finished)
    return 0;

  request_path = getenv("IMCFLOW_POWER_REQUEST");
  if (request_path == NULL || request_path[0] == '\0')
    return 0;

  scope = getenv("IMCFLOW_POWER_SCOPE");
  if (scope == NULL || scope[0] == '\0' || strcmp(scope, "continuous") == 0) {
    g_power_region_scope = 0;
  } else if (strcmp(scope, "region") == 0) {
    g_power_region_scope = 1;
  } else {
    fprintf(stderr, "[POWER] invalid IMCFLOW_POWER_SCOPE: %s\n", scope);
    return -1;
  }

  g_power_request_path = request_path;
  if (!g_power_region_scope && dmm_session_start_file(request_path) != 0) {
    fprintf(stderr, "[POWER] session start failed: %s\n", dmm_last_error());
    return -1;
  }

  g_power_enabled = 1;
  g_power_finished = 0;
  g_power_degraded = 0;
  g_power_cleanup_started = 0;
  g_power_region_active = 0;
  if (!g_power_atexit_installed && atexit(power_measure_runtime_atexit) != 0) {
    fprintf(stderr, "[POWER] failed to install finalize handler\n");
    if (dmm_session_is_active())
      (void)dmm_session_abort("failed to install atexit handler");
    g_power_enabled = 0;
    g_power_finished = 1;
    return -1;
  }
  g_power_atexit_installed = 1;

  if (g_power_region_scope) {
    fprintf(stderr,
            "[POWER] scope=region tvm_revision=%s "
            "measurement_utils_revision=%s build_dirty=%d\n",
            IMCFLOW_BUILD_TVM_GIT_REV,
            IMCFLOW_BUILD_MEASUREMENT_UTILS_GIT_REV,
            IMCFLOW_BUILD_TREE_DIRTY);
  } else {
    fprintf(stderr,
            "[POWER] scope=continuous session=%s server_revision=%s "
            "tvm_revision=%s measurement_utils_revision=%s build_dirty=%d\n",
            dmm_session_id(), dmm_server_revision(), IMCFLOW_BUILD_TVM_GIT_REV,
            IMCFLOW_BUILD_MEASUREMENT_UTILS_GIT_REV, IMCFLOW_BUILD_TREE_DIRTY);
  }
  power_measure_runtime_phase("process_setup");
  return 0;
}


int power_measure_runtime_region_begin(const char *region_name)
{
  if (!g_power_enabled || g_power_finished || !g_power_region_scope)
    return 0;
  if (g_power_region_active || dmm_session_is_active()) {
    fprintf(stderr, "[POWER] nested power region is not allowed\n");
    g_power_degraded = 1;
    return -1;
  }
  if (region_name == NULL || region_name[0] == '\0') {
    fprintf(stderr, "[POWER] region name is empty\n");
    g_power_degraded = 1;
    return -1;
  }
  if (dmm_region_start_file(g_power_request_path, region_name) != 0) {
    fprintf(stderr, "[POWER] region start failed: %s\n", dmm_last_error());
    g_power_degraded = 1;
    return -1;
  }
  g_power_region_active = 1;
  record_tag_result(dmm_tag_set("region", region_name), "region tag");
  fprintf(stderr, "[POWER] region=%s session=%s started\n",
          region_name, dmm_session_id());
  return g_power_degraded ? -1 : 0;
}


int power_measure_runtime_region_end(void)
{
  int result;
  if (!g_power_enabled || g_power_finished || !g_power_region_scope)
    return 0;
  if (!g_power_region_active)
    return 0;
  record_tag_result(dmm_tag_event("region_end"), "region end event");
  result = dmm_session_stop();
  g_power_region_active = 0;
  if (result != 0) {
    g_power_degraded = 1;
    fprintf(stderr, "[POWER] region finalize failed: %s\n", dmm_last_error());
    return -1;
  }
  fprintf(stderr, "[POWER] region finalized%s\n",
          g_power_degraded ? " with tag errors" : "");
  return g_power_degraded ? -1 : 0;
}


void power_measure_runtime_phase(const char *phase)
{
  if (!g_power_enabled || g_power_finished)
    return;
  record_tag_result(dmm_tag_set("phase", phase), "phase tag");
  if (phase != NULL && strcmp(phase, "cleanup") == 0)
    g_power_cleanup_started = 1;
}


void power_measure_runtime_sample(size_t sample_index)
{
  char value[32];
  if (!g_power_enabled || g_power_finished)
    return;
  snprintf(value, sizeof(value), "%zu", sample_index);
  record_tag_result(dmm_tag_set("sample", value), "sample tag");
}


void power_measure_runtime_clear_sample(void)
{
  if (!g_power_enabled || g_power_finished)
    return;
  record_tag_result(dmm_tag_clear("sample"), "sample tag clear");
}


void power_measure_runtime_event(const char *name)
{
  if (!g_power_enabled || g_power_finished)
    return;
  record_tag_result(dmm_tag_event(name), "event tag");
}


int power_measure_runtime_finish(void)
{
  int result;
  if (!g_power_enabled || g_power_finished)
    return 0;

  if (!g_power_cleanup_started)
    power_measure_runtime_phase("cleanup");
  g_power_finished = 1;
  if (g_power_region_scope) {
    if (g_power_region_active || dmm_session_is_active()) {
      result = dmm_session_abort("process finalized with active power region");
      g_power_region_active = 0;
      if (result != 0) {
        g_power_degraded = 1;
        fprintf(stderr, "[POWER] active region abort failed: %s\n",
                dmm_last_error());
      }
    }
    fprintf(stderr, "[POWER] region scope finalized%s\n",
            g_power_degraded ? " with errors" : "");
    return g_power_degraded ? -1 : 0;
  }
  result = dmm_session_stop();
  if (result != 0) {
    g_power_degraded = 1;
    fprintf(stderr, "[POWER] session finalize failed: %s\n", dmm_last_error());
    return -1;
  }
  fprintf(stderr, "[POWER] session finalized%s\n",
          g_power_degraded ? " with tag errors" : "");
  return g_power_degraded ? -1 : 0;
}


int power_measure_runtime_is_enabled(void)
{
  return g_power_enabled;
}


int power_measure_runtime_is_degraded(void)
{
  return g_power_degraded;
}


int power_measure_runtime_print_build_info(FILE *stream)
{
  if (stream == NULL)
    return -1;
  if (fprintf(stream,
              "IMCFLOW_POWER_BUILD_INFO tvm=%s measurement_utils=%s dirty=%d\n",
              IMCFLOW_BUILD_TVM_GIT_REV,
              IMCFLOW_BUILD_MEASUREMENT_UTILS_GIT_REV,
              IMCFLOW_BUILD_TREE_DIRTY) < 0)
    return -1;
  return 0;
}
