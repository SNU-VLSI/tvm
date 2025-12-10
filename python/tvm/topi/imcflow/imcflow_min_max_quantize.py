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
"""Imcflow Min Max Quantize"""
import typing
from functools import reduce
from typing import List

from tvm import te
from tvm import topi
from tvm import tir


# def imcflow_min_max_quantize(
#     data: te.Tensor,
#     min,
#     max,
#     axis: int,
#     out_dtype = "float32",
#     param_dtype = "int16",
# ):
#     """Imcflow Min Max Quantize.

#     Parameters
#     ----------
#     data : tvm.te.Tensor
#         Input data should be quantized
#         dtype of data is float32
    
#     min : Expr
#         The minimum value of the quantization range
#         dtype of min is int16

#     max : Expr
#         The maximum value of the quantization range
#         dtype of max is int16
    
#     axis : int
#         Specify along which shape axis the quantization should occur.
    
#     out_dtype : Datatype, default="float32"
#         The output data type of the quantized data

#     param_dtype : Datatype, default="int16"
#         The data type of the min and max values

#     Returns
#     -------
#     output : tvm.te.Tensor
#         Quantized data with same shape as input
#         dtype = out_dtype

#     """
#     if axis == None:
#         axis = 1

#     scale = topi.div(tir.const(15, dtype=param_dtype), (max - min))
#     quantized_data = topi.clip(topi.floor(topi.cast(data, dtype=param_dtype) - min) * scale, 0.0, 15.0)
#     output = topi.cast(quantized_data, dtype=out_dtype)

#     return output


def imcflow_min_max_quantize(
    data: te.Tensor,
    min,
    max,
    axis: int,
    out_dtype = "float32",
    param_dtype = "int16",
):
    """Hardware-defined Min-Max Quantization.

    Quantize the input data using HW-defined min-max quantization.
    Computes thresholds as: threshold[i] = min + floor((i+1) * (max - min) / 2^bits)
    for i in [0, 2^bits - 2], then quantizes data based on these thresholds.

    Parameters
    ----------
    data : tvm.te.Tensor
        Input data to be quantized

    min : Expr
        The minimum value of the quantization range

    max : Expr
        The maximum value of the quantization range

    axis : int
        Specify along which shape axis the quantization should occur.

    out_dtype : str, default="float32"
        The output data type of the quantized data

    param_dtype : str, default="int16"
        The data type of the min and max values

    Returns
    -------
    output : tvm.te.Tensor
        Quantized data with same shape as input, values in range [0, 2^bits - 1]
    """
    bits = 4  # Fixed to 4 bits as per hardware definition
    # Compute thresholds: min + floor((i+1) * (max - min) / 2^bits)
    num_thresholds = (1 << bits) - 1  # 2^bits - 1
    thresholds = []
    
    range_val = topi.cast(max, dtype="int32") - topi.cast(min, dtype="int32")
    divisor = tir.const(1 << bits, dtype=param_dtype)  # 2^bits
    
    for i in range(num_thresholds):
        offset = tir.const(i + 1, dtype=param_dtype) * range_val
        norm_offset = topi.floor(topi.cast(offset, dtype="float32") / topi.cast(divisor, dtype="float32"))
        threshold = min + topi.cast(norm_offset, dtype=param_dtype)
        thresholds.append(threshold)
    
    # Cast input data to param_dtype for comparison
    data_param = topi.cast(data, dtype=param_dtype)
    
    # Create output tensor with the same shape as input
    output = te.compute(
        data.shape,
        lambda *indices: topi.cast(
            reduce(
                lambda result, i: tir.if_then_else(
                    data_param[indices] <= thresholds[i],
                    tir.const(i, dtype=param_dtype),
                    result
                ),
                range(len(thresholds) - 1, -1, -1),
                tir.const(len(thresholds), dtype=param_dtype)
            ),
            dtype=out_dtype
        ),
        name="imcflow_min_max_quantize"
    )
    
    return output
