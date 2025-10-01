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
# pylint: disable=invalid-name, too-many-locals, too-many-arguments
"""Utility functions for bitserial operators"""
import numpy as np
import tvm
from tvm import te
from tvm.topi.transform import concatenate
from ..utils import get_const_int


def bitpack(data, bits, pack_axis, bit_axis, pack_type, name="QuantizeInput", msb_first=True):
    """Packs data into format necessary for bitserial computation
    
    For 32-bit targets, uint64/uint128/uint256 are automatically split into
    multiple uint32 chunks in little-endian order.

    Parameters
    ----------
    pack_axis : int
       index of the axis to pack in data
    bit_axis : int
       index of axis to place bit axis in resulting packed data
    pack_type : str
       data type for packing (uint8, uint16, uint32, uint64, uint128, uint256)
    msb_first : bool, optional
       If True (default), pack bits MSB first (original behavior).
       If False, pack bits LSB first.
    """
    ishape = data.shape
    n = len(ishape)
    
    # Determine data width and chunking for 32-bit targets
    if pack_type == "uint8":
        data_width = 8
        num_chunks = 1
        chunk_type = "uint8"
    elif pack_type == "uint16":
        data_width = 16
        num_chunks = 1
        chunk_type = "uint16"
    elif pack_type == "uint32":
        data_width = 32
        num_chunks = 1
        chunk_type = "uint32"
    elif pack_type == "uint64":
        data_width = 64
        num_chunks = 2  # Split into 2x uint32 (32-bit target)
        chunk_type = "uint32"
    elif pack_type == "uint128":
        data_width = 128
        num_chunks = 4  # Split into 4x uint32 (32-bit target)
        chunk_type = "uint32"
    elif pack_type == "uint256":
        data_width = 256
        num_chunks = 8  # Split into 8x uint32 (32-bit target)
        chunk_type = "uint32"
    else:
        raise ValueError(f"Unsupported pack_type: {pack_type}")

    # Calculate packed size with padding support
    pack_axis_size = get_const_int(ishape[pack_axis])
    packed_size = (pack_axis_size + data_width - 1) // data_width

    shape_vec = list(ishape)
    shape_vec[pack_axis] = packed_size
    shape_vec.insert(bit_axis, 1)
    if num_chunks > 1:
        shape_vec.append(num_chunks)  # Add chunk dimension at the end
    bitserial_oshape = tuple(shape_vec)
    masks = np.array([0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80])

    # pack axis shifts if bit axis comes before
    if bit_axis <= pack_axis:
        pack_axis += 1

    def _bitpack(*indices):
        if num_chunks == 1:
            # Original behavior for uint8/uint16/uint32
            packed_data = [tvm.tir.const(0, chunk_type)] * bits
            for k in range(data_width):
                # Translate indices for packed data back to original
                idx = [0] * n
                j = 0
                pack_axis_idx = None
                for i in range(n + 1):
                    if i == bit_axis:
                        continue
                    if i == pack_axis:
                        idx[j] = indices[i] * data_width + k
                        pack_axis_idx = indices[i] * data_width + k
                    else:
                        idx[j] = indices[i]
                    j += 1

                # Check bounds for padding using tvm.tir.if_then_else
                element = tvm.tir.if_then_else(
                    pack_axis_idx < pack_axis_size,
                    data(*idx),
                    tvm.tir.const(0, data.dtype)
                )

                for b in range(bits):
                    extracted_bit = ((element & tvm.tir.const(masks[b], "int32")) >> b).astype(
                        chunk_type
                    )
                    if msb_first:
                        # MSB first: shift left to make room for next bit
                        packed_data[b] = packed_data[b] | extracted_bit
                        if k < data_width - 1:
                            packed_data[b] = packed_data[b] << 1
                    else:
                        # LSB first: shift the bit to its position
                        packed_data[b] = packed_data[b] | (extracted_bit << k)

                if k == data_width - 1:
                    return tuple(packed_data)
            return tuple(packed_data)
        else:
            # Multi-chunk behavior for uint64/uint128/uint256
            # Extract chunk index from indices
            chunk_idx = indices[-1]
            # Remove chunk index for processing
            base_indices = indices[:-1]
            
            # Each chunk handles 32 bits
            chunk_width = 32
            packed_data = [tvm.tir.const(0, chunk_type)] * bits
            
            for k in range(chunk_width):
                # Calculate position in original data (little-endian)
                bit_offset = chunk_idx * chunk_width + k
                
                # Translate indices for packed data back to original
                idx = [0] * n
                j = 0
                pack_axis_idx = None
                for i in range(n + 1):
                    if i == bit_axis:
                        continue
                    if i == pack_axis:
                        idx[j] = base_indices[i] * data_width + bit_offset
                        pack_axis_idx = base_indices[i] * data_width + bit_offset
                    else:
                        idx[j] = base_indices[i]
                    j += 1

                # Check bounds for padding using tvm.tir.if_then_else
                element = tvm.tir.if_then_else(
                    pack_axis_idx < pack_axis_size,
                    data(*idx),
                    tvm.tir.const(0, data.dtype)
                )

                for b in range(bits):
                    extracted_bit = ((element & tvm.tir.const(masks[b], "int32")) >> b).astype(
                        chunk_type
                    )
                    if msb_first:
                        # MSB first: shift left to make room for next bit
                        packed_data[b] = packed_data[b] | extracted_bit
                        if k < chunk_width - 1:
                            packed_data[b] = packed_data[b] << 1
                    else:
                        # LSB first: shift the bit to its position
                        packed_data[b] = packed_data[b] | (extracted_bit << k)

                if k == chunk_width - 1:
                    return tuple(packed_data)
            return tuple(packed_data)

    output_tuple = te.compute(bitserial_oshape, _bitpack, name=name, tag="bitpack")

    if bits > 1:
        return concatenate(output_tuple, axis=bit_axis)
    return output_tuple


def binary_op_multiplier(pack_dtype):
    """ "Returns number of bits packed into
    pack_dtype: string
        pack type for the operator (must be a uint)"""
    return int(pack_dtype[4:])
