#ifndef IMCFLOW_POWER_MEASURE_RUNTIME_H_
#define IMCFLOW_POWER_MEASURE_RUNTIME_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Start the optional whole-process measurement session selected by
 * IMCFLOW_POWER_REQUEST.  A missing/empty variable means power is disabled and
 * all helpers below are no-ops.  A configured session that cannot start is an
 * error and the workload must not run.
 */
int power_measure_runtime_start(void);

/* General active-map and event helpers used by host executables. */
void power_measure_runtime_phase(const char *phase);
void power_measure_runtime_sample(size_t sample_index);
void power_measure_runtime_clear_sample(void);
void power_measure_runtime_event(const char *name);

/*
 * Set phase=cleanup and synchronously finalize the session.  It is idempotent;
 * an atexit fallback invokes it for ordinary early returns.
 */
int power_measure_runtime_finish(void);

int power_measure_runtime_is_enabled(void);
int power_measure_runtime_is_degraded(void);

#ifdef __cplusplus
}
#endif

#endif  /* IMCFLOW_POWER_MEASURE_RUNTIME_H_ */
