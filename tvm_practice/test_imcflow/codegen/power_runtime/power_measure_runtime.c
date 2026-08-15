#include "power_measure_runtime.h"

#include <stdio.h>
#include <stdlib.h>

#include "dmm_measure.h"


static int g_power_enabled = 0;
static int g_power_finished = 0;
static int g_power_degraded = 0;


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

  if (g_power_enabled && !g_power_finished)
    return 0;

  request_path = getenv("IMCFLOW_POWER_REQUEST");
  if (request_path == NULL || request_path[0] == '\0')
    return 0;

  if (dmm_session_start_file(request_path) != 0) {
    fprintf(stderr, "[POWER] session start failed: %s\n", dmm_last_error());
    return -1;
  }

  g_power_enabled = 1;
  g_power_finished = 0;
  g_power_degraded = 0;
  if (atexit(power_measure_runtime_atexit) != 0) {
    fprintf(stderr, "[POWER] failed to install finalize handler\n");
    (void)dmm_session_abort("failed to install atexit handler");
    g_power_enabled = 0;
    g_power_finished = 1;
    return -1;
  }

  fprintf(stderr, "[POWER] session=%s server_revision=%s\n",
          dmm_session_id(), dmm_server_revision());
  power_measure_runtime_phase("process_setup");
  return 0;
}


void power_measure_runtime_phase(const char *phase)
{
  if (!g_power_enabled || g_power_finished)
    return;
  record_tag_result(dmm_tag_set("phase", phase), "phase tag");
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

  power_measure_runtime_phase("cleanup");
  g_power_finished = 1;
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
