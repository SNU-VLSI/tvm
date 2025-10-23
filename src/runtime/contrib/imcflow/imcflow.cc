/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/contrib/imcflow/imcflow.cc
 * \brief TVM compatible wrappers for imcflow kernels.
 */

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include "imcflow_kernel.h"

#define INODE_0_S_AXI_BASEADDR 0x43C00000
#define INODE_1_S_AXI_BASEADDR 0x43C00000

namespace tvm {
namespace runtime {
namespace contrib {

extern "C" void imcflow_fused_kernel(float* data, float* out) {
  out = data;
  // 1. reorder input
  // 2. device code transfer
  // 3. data transfer (host -> device)
  // 4. trigger device
  // 5. wait for interrupt
  // 6. data transfer (device -> host)
  // 7. reorder output
}

/*
void imcflow_relu(float* data, float* out, std::vector<int64_t> shape) {
  int64_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
  for (int64_t i = 0; i < size; i++) {
    out[i] = std::max(data[i], 0.0f);
  }
}
*/

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
