#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "power_measure_runtime.h"

static int begin_count;
static int end_count;
static int active;
static power_region_policy_t last_policy;

int power_region_runtime_init(const char *path)
{
    return path && path[0] ? 0 : -1;
}

int power_region_runtime_shutdown(void) { return active ? -1 : 0; }

int power_region_begin(power_region_context_t *ctx, const char *name,
                       power_region_policy_t policy)
{
    (void)name;
    memset(ctx, 0, sizeof(*ctx));
    ctx->abi_version = POWER_REGION_ABI_VERSION;
    ctx->policy = policy;
    last_policy = policy;
    ctx->active = 1;
    active = 1;
    ++begin_count;
    return 0;
}

int power_region_next(power_region_context_t *ctx)
{
    return ctx->iteration_count++ == 0;
}

int power_region_end(power_region_context_t *ctx)
{
    ctx->active = 0;
    active = 0;
    ++end_count;
    return 0;
}

int power_region_is_active(void) { return active; }
int power_tag_set(const char *key, const char *value)
{
    (void)key;
    (void)value;
    return 0;
}
int power_tag_clear(const char *key)
{
    (void)key;
    return 0;
}
int power_tag_event(const char *name)
{
    (void)name;
    return 0;
}
int power_region_last_status(void) { return 0; }
const char *power_region_last_error(void) { return ""; }

int main(void)
{
    power_measure_scope_context_t ctx;
    setenv("IMCFLOW_POWER_REQUEST", "/tmp/request.json", 1);
    setenv("IMCFLOW_POWER_SCOPE", "MODEL", 1);
    setenv("IMCFLOW_POWER_LOOP_ENABLE", "1", 1);
    setenv("IMCFLOW_POWER_MIN_SAMPLES", "30000", 1);

    if (power_measure_runtime_start() != 0)
        return 1;
    if (power_measure_scope_begin(
            &ctx, IMCFLOW_POWER_SCOPE_MODEL, "model") != 0)
        return 2;
    if (begin_count != 0 || !power_measure_scope_next(&ctx))
        return 3;
    if (power_measure_runtime_model_start_after_first_warmup() != 0)
        return 4;
    if (begin_count != 1 || !active || !last_policy.loop_enable ||
        last_policy.min_samples != 30000)
        return 5;
    if (power_measure_runtime_model_start_after_first_warmup() != 0 ||
        begin_count != 1)
        return 6;
    if (power_measure_scope_next(&ctx) != 0)
        return 7;
    if (power_measure_scope_end(&ctx) != 0 || end_count != 1 || active)
        return 8;
    if (power_measure_runtime_finish() != 0)
        return 9;

    setenv("IMCFLOW_POWER_SCOPE", "REGION", 1);
    if (power_measure_runtime_start() != 0)
        return 10;
    if (power_measure_scope_begin(
            &ctx, IMCFLOW_POWER_SCOPE_REGION, "region") != 0)
        return 11;
    if (!last_policy.loop_enable || last_policy.min_samples != 30000 ||
        last_policy.min_seconds != 0.0)
        return 12;
    if (!power_measure_scope_next(&ctx) || power_measure_scope_next(&ctx) != 0)
        return 13;
    if (power_measure_scope_end(&ctx) != 0)
        return 14;
    if (power_measure_runtime_finish() != 0)
        return 15;
    puts("MODEL/REGION loop policy: OK");
    return 0;
}
