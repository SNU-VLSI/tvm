#include "power_measure_runtime.h"

void generated_scope_shape(int fail)
{
    int retry_count = 0;
    do {
        int retry_requested = 0;
        TVM_POWER_REGION_BEGIN(IMCFLOW_POWER_SCOPE_REGION, "region");
        if (fail) {
            retry_requested = 1;
            break;
        }
        for (int tile = 0; tile < 2; ++tile) {
            TVM_POWER_REGION_BEGIN(IMCFLOW_POWER_SCOPE_TILE, "tile");
            if (fail) {
                retry_requested = 1;
                break;
            }
            TVM_POWER_REGION_END();
            if (retry_requested)
                break;
        }
        TVM_POWER_REGION_END();
        if (retry_requested) {
            ++retry_count;
            continue;
        }
        break;
    } while (retry_count < 3);
}
