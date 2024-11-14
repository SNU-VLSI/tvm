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

namespace tvm {
namespace runtime {
namespace contrib {

namespace imcflow {

typedef struct {
  void** data;
} ImcflowPackedArgs;

void imcflow_conv2d_common(float* data, float* weights, float* bias, float* out, int p_N_, int p_C_,
                        int p_H_, int p_W_, int p_O_, int p_G_, int p_Ph0_, int p_Pw0_, int p_Ph1_,
                        int p_Pw1_, int p_Kh_, int p_Kw_, int p_Sh_, int p_Sw_, std::string attr,
                        bool channel_last, bool pre_cast, bool post_cast) {
}

// IMCFLOW Conv2d single OP
TVM_REGISTER_GLOBAL("tvm.contrib.imcflow.conv2d").set_body([](TVMArgs args, TVMRetValue* ret) {
  DLTensor* input = args[0];
  DLTensor* weights = args[1];
  DLTensor* output = args[2];
  int p_Ph0_ = args[3], p_Pw0_ = args[4], p_Ph1_ = args[5], p_Pw1_ = args[6], p_Sh_ = args[7],
      p_Sw_ = args[8], p_G_ = args[9];
  bool channel_last = args[10];
  bool pre_cast = args[11];
  bool post_cast = args[12];

  int p_N_ = input->shape[0], p_C_ = input->shape[1], p_H_ = input->shape[2],
      p_W_ = input->shape[3], p_O_ = output->shape[1], p_Kh_ = weights->shape[2],
      p_Kw_ = weights->shape[3];

  if (channel_last) {
    p_N_ = input->shape[0];
    p_H_ = input->shape[1];
    p_W_ = input->shape[2];
    p_C_ = input->shape[3];
    p_O_ = output->shape[3];
    p_Kh_ = weights->shape[0];
    p_Kw_ = weights->shape[1];
  }

  std::vector<float> bias(p_O_, 0);
  std::string attr;
  return imcflow_conv2d_common(static_cast<float*>(input->data), static_cast<float*>(weights->data),
                            bias.data(), static_cast<float*>(output->data), p_N_, p_C_, p_H_, p_W_,
                            p_O_, p_G_, p_Ph0_, p_Pw0_, p_Ph1_, p_Pw1_, p_Kh_, p_Kw_, p_Sh_, p_Sw_,
                            attr, channel_last, pre_cast, post_cast);
});

}  // namespace imcflow
}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
