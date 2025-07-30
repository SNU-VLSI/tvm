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
"""Imcflow Unpacking"""
import typing
from functools import reduce

from tvm import te
from tvm import topi
from tvm import tir


def imcflow_unpacking(
    data: te.Tensor,
    newshape,
    out_dtype = "float32",
):
    """Imcflow Packing.

    Parameters
    ----------
    data : tvm.te.Tensor
        int8 data type tensor which is packed two int4 tensor
    
    newshape : tuple of ints
        The new shape

    Returns
    -------
    output : tvm.te.Tensor
        float32 data type tensor which is unpacked two int4 tensor

    """

    """Todo. use te instead of numpy"""
    """using topi.math / topi.broadcast functions
    topi.right_shift, topi.left_shift, topi.bitwise_and, topi.add, topi.subtract, topi.multiply"""

    dims = data.shape
    dims_unpacked = topi.utils.prod(newshape)
    data_copyed = te.compute(dims_unpacked, lambda i: te.if_then_else(i % 2 == 0, data[i // 2], (data[i // 2] >> 4) ), name = "data_copyed")

    sign = topi.broadcast.bitwise_and(data_copyed, tir.const(0x08, "int8"))
    value = topi.broadcast.bitwise_and(data_copyed, tir.const(0x07, "int8"))
    value = te.compute(dims_unpacked, lambda i: te.if_then_else(sign[i] == tir.const(0x08, "int8"), value[i] | tir.const(0x78, "int8"), value[i]), name = "value")
    sign = topi.math.left_shift(sign, 4)
    int8_data = topi.bitwise_or(sign, value)

    out = topi.reshape(int8_data, newshape)

    out = topi.cast(out, out_dtype)

    return out
