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

from tvm import te
from tvm import topi
from tvm import tir


def imcflow_min_max_quantize(
    data: te.Tensor,
    min,
    max,
    axis: int,
    out_dtype = "float32",
    param_dtype = "int16",
):
    """Imcflow Min Max Quantize.

    Parameters
    ----------
    data : tvm.te.Tensor
        Input data should be quantized
        dtype of data is float32
    
    min : Expr
        The minimum value of the quantization range
        dtype of min is int16

    max : Expr
        The maximum value of the quantization range
        dtype of max is int16
    
    axis : int
        Specify along which shape axis the quantization should occur.
    
    out_dtype : Datatype, default="float32"
        The output data type of the quantized data

    param_dtype : Datatype, default="int16"
        The data type of the min and max values

    Returns
    -------
    output : tvm.te.Tensor
        Quantized data with same shape as input
        dtype = out_dtype

    """
    if axis == None:
        axis = 1

    scale = topi.div(tir.const(15, dtype=param_dtype), (max - min))
    quantized_data = topi.clip(topi.floor(topi.cast(data, dtype=param_dtype) - min) * scale, 0.0, 15.0)
    output = topi.cast(quantized_data, dtype=out_dtype)

    return output
