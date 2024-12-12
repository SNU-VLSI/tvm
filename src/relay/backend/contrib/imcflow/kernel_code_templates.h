#ifndef KERNEL_CODE_TEMPLATES_H_
#define KERNEL_CODE_TEMPLATES_H_

#include <string>

const std::string header_init_ = R"(
#ifndef TVM_RUNTIME_CONTRIB_IMCFLOW_IMCFLOW_KERNEL_H_
#define TVM_RUNTIME_CONTRIB_IMCFLOW_IMCFLOW_KERNEL_H_

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>

#include <vector>

namespace tvm {
namespace runtime {
namespace contrib {

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_IMCFLOW_IMCFLOW_KERNEL_H_
)";

const std::string source_init_ = R"(
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "imcflow_kernel.h"

namespace tvm {
namespace runtime {
namespace contrib {

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
)";

#endif  // KERNEL_CODE_TEMPLATES_H_
