#ifndef IMCFLOW_POWER_MEASURE_RUNTIME_H_
#define IMCFLOW_POWER_MEASURE_RUNTIME_H_

#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialize the optional measurement runtime selected by
 * IMCFLOW_POWER_REQUEST.  IMCFLOW_POWER_SCOPE selects ``continuous`` (one
 * whole-process session) or ``region`` (sessions opened by generated kernels).
 * A missing/empty request means power is disabled and all helpers are no-ops.
 */
int power_measure_runtime_start(void);

/*
 * Open/close one non-nestable region session.  In continuous scope these are
 * no-ops, so generated kernels work with both acquisition scopes.  Region
 * begin returns only after GET; region end freezes and finalizes the artifact.
 */
int power_measure_runtime_region_begin(const char *region_name);
int power_measure_runtime_region_end(void);

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

/* Print the revisions embedded by CMake for runner-side identity checks. */
int power_measure_runtime_print_build_info(FILE *stream);

#ifdef __cplusplus
}
#endif

#endif  /* IMCFLOW_POWER_MEASURE_RUNTIME_H_ */
