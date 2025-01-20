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
"""Imcflow Packing"""
import typing
from functools import reduce

from tvm import te
from tvm import topi
from tvm import tir


def imcflow_packing_test(
    data: te.Tensor,
    newshape,
):
    """Imcflow Packing.

    Parameters
    ----------
    data : tvm.te.Tensor
        Input to be quantized and packed.
        this data is already quantize to int4
        but the data type is stored as fp32
    
    newshape : tuple of ints
        The new shape

    Returns
    -------
    output : tvm.te.Tensor
        int8 data type tensor which is packed two int4 tensor
        if the number of inout data is odd, pad 4 bit zeros

    """

    dims = data.shape
    data_int8 = topi.cast(data, "int8")
    num_elements = topi.utils.prod(dims)
    data_rs = topi.reshape(data_int8, (num_elements,))

    packed_size = (num_elements + 1) // 2

    if newshape is None:
        newshape = [packed_size,]

    
    sign = topi.math.right_shift(data_rs, 4)
    sign = topi.broadcast.bitwise_and(sign, tir.const(0x08, "int8"))
    value = topi.broadcast.bitwise_and(data_rs, tir.const(0x07, "int8"))
    int4_data = topi.bitwise_or(sign, value)

    out = te.compute(
        newshape,
        lambda i: int4_data[i * 2] | (te.if_then_else(i * 2 + 1 < num_elements, int4_data[i * 2 + 1], tir.const(0, "int8")) << 4),
        name = "imcflow_packing_test"
    )

    return out
