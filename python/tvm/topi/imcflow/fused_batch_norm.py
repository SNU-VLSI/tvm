# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Imcflow Batch normalization."""
import typing
from functools import reduce

from tvm import te
from tvm import topi


def fused_batch_norm(
    data: te.Tensor,
    fused_scale: te.Tensor,
    fused_bias: te.Tensor,
    axis: typing.Optional[int] = None,
) -> typing.List[te.Tensor]:
    """Imcflow Batch normalization layer.

    Normalizes the input at each batch, i.e. applies a transformation
    that maintains the mean activation close to 0 and the activation
    standard deviation close to 1.

    Parameters
    ----------
    data : tvm.te.Tensor
        Input to be batch-normalized.

    fused_scale : tvm.te.Tensor
        constant fused_scale factor to be used in batch_norm inference.

    fused_bias : tvm.te.Tensor
        constant fused_bias factor to be used in batch_norm inference.

    axis : int, optional, default=1
        Specify along which shape axis the normalization should occur.

    Returns
    -------
    output : list of tvm.te.Tensor
        Normalized data with same shape as input

    fused_scale : tvm.te.Tensor
        constant fused_scale factor to be used in batch_norm inference.

    fused_bias : tvm.te.Tensor
        constant fused_bias factor to be used in batch_norm inference.
    """
    if axis is None:
        axis = 1

    shape = [1] * len(data.shape)
    shape[axis] = data.shape[axis]

    fused_scale_rs = topi.reshape(fused_scale, shape)
    fused_bias_rs = topi.reshape(fused_bias, shape)
    out = data * fused_scale_rs + fused_bias_rs

    # placeholder reuse, we multiply by 1 and return them.
    return [out, fused_scale, fused_bias]
