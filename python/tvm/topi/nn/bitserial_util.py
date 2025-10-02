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
    if pack_type == "uint255": pack_type = "uint256"  # Handle special case for 256 bits
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


def bitunpack(data, bits, pack_axis, bit_axis, pack_type, out_size=None, 
              out_dtype="uint8", name="BitUnpack", msb_first=True):
    """Unpack bitpacked data back to original format.
    
    This function reverses the bitpack operation, extracting individual elements
    from their packed representation.
    
    Parameters
    ----------
    data : tvm.te.Tensor
        Packed input tensor with shape (..., packed_size, ..., bits, [chunks])
    bits : int
        Number of bits per element
    pack_axis : int
        Original axis that was packed (will be expanded)
    bit_axis : int
        Axis where bit planes are located (will be removed)
    pack_type : str or DataType
        Original pack type (uint8, uint16, uint32, uint64, uint128, uint256)
    out_size : int, optional
        Original size before padding. None means full unpack.
    out_dtype : str
        Output data type
    name : str
        Operation name
    msb_first : bool
        Whether bits were packed MSB first or LSB first
    
    Returns
    -------
    result : tvm.te.Tensor
        Unpacked tensor with shape (..., out_size or full_size, ...)
    """
    from tvm import tir
    
    # Parse pack_type
    if isinstance(pack_type, str):
        pack_type_str = pack_type
    else:
        # DataType object
        pack_type_str = str(pack_type)
    
    # Handle special case for 256 bits (same as bitpack)
    if pack_type_str == "uint255":
        pack_type_str = "uint256"
    
    # Determine data width and chunking
    if pack_type_str == "uint8":
        data_width = 8
        num_chunks = 1
        chunk_type = "uint8"
    elif pack_type_str == "uint16":
        data_width = 16
        num_chunks = 1
        chunk_type = "uint16"
    elif pack_type_str == "uint32":
        data_width = 32
        num_chunks = 1
        chunk_type = "uint32"
    elif pack_type_str == "uint64":
        data_width = 64
        num_chunks = 2
        chunk_type = "uint32"
    elif pack_type_str == "uint128":
        data_width = 128
        num_chunks = 4
        chunk_type = "uint32"
    elif pack_type_str == "uint256":
        data_width = 256
        num_chunks = 8
        chunk_type = "uint32"
    else:
        raise ValueError(f"Unsupported pack_type: {pack_type_str}")
    
    print(f"[bitunpack] pack_type: {pack_type_str}, data_width: {data_width}, num_chunks: {num_chunks}")
    
    ishape = data.shape
    ndim = len(ishape)
    
    # Handle negative indexing
    if pack_axis < 0:
        pack_axis = ndim + pack_axis
    if bit_axis < 0:
        bit_axis = ndim + bit_axis
    
    # Adjust pack_axis if bit_axis was inserted before it (same as bitpack)
    # This is because when bitpack creates the packed tensor, it inserts bit_axis first,
    # which shifts pack_axis by 1 if bit_axis <= pack_axis
    adjusted_pack_axis = pack_axis
    if bit_axis <= pack_axis:
        adjusted_pack_axis = pack_axis + 1
    
    print(f"[bitunpack] pack_axis: {pack_axis}, bit_axis: {bit_axis}, adjusted_pack_axis: {adjusted_pack_axis}")
    
    # Calculate sizes using adjusted_pack_axis
    packed_size = get_const_int(ishape[adjusted_pack_axis])
    # Each packed unit stores data_width elements (one element per bit position)
    elements_per_packed = data_width
    full_unpacked_size = packed_size * elements_per_packed
    
    if out_size is None:
        actual_out_size = full_unpacked_size
    else:
        actual_out_size = out_size
    
    print(f"[bitunpack] packed_size: {packed_size}, elements_per_packed: {elements_per_packed}")
    print(f"[bitunpack] full_unpacked_size: {full_unpacked_size}, actual_out_size: {actual_out_size}")
    
    # Build output shape (remove bit_axis and chunk dimension if present)
    oshape = []
    chunk_axis = -1
    if num_chunks > 1:
        chunk_axis = ndim - 1  # Chunks are at the end
    
    for i in range(ndim):
        if i == bit_axis:
            continue  # Skip bit dimension
        elif num_chunks > 1 and i == chunk_axis:
            continue  # Skip chunk dimension
        elif i == adjusted_pack_axis:
            oshape.append(actual_out_size)  # Expanded size
        else:
            oshape.append(ishape[i])
    
    print(f"[bitunpack] input shape: {ishape}, output shape: {oshape}")
    
    def unpack_compute(*output_indices):
        """Compute function to extract unpacked value"""
        # Get the unpacked element index
        unpack_idx = output_indices[pack_axis]
        
        # Determine which packed unit and offset within it
        packed_idx = unpack_idx // elements_per_packed
        element_offset = unpack_idx % elements_per_packed
        
        # For multi-chunk types, determine chunk and bit offset
        if num_chunks > 1:
            chunk_idx = element_offset // 32
            bit_offset = element_offset % 32
        else:
            chunk_idx = 0
            bit_offset = element_offset
        
        # Reconstruct value from bit planes
        value = tir.const(0, out_dtype)
        
        for b in range(bits):
            # Build input indices: translate output_indices to packed input space
            # Similar to bitpack but in reverse
            input_idx = []
            j = 0
            for i in range(len(oshape) + 1 + (1 if num_chunks > 1 else 0)):
                if i == bit_axis:
                    input_idx.append(b)  # Insert bit index
                elif i == adjusted_pack_axis:
                    input_idx.append(packed_idx)  # Insert packed index
                    j += 1
                elif num_chunks > 1 and i == len(oshape) + 1:
                    input_idx.append(chunk_idx)  # Insert chunk index at end
                else:
                    if j < len(output_indices):
                        input_idx.append(output_indices[j])
                        j += 1
            
            # Read packed value
            packed_val = data[tuple(input_idx)]
            
            # Extract bit at bit_offset position
            bit = (packed_val >> bit_offset) & tir.const(1, chunk_type)
            bit = tir.Cast(out_dtype, bit)
            
            # Combine based on MSB/LSB first
            if msb_first:
                # MSB first: bit b is at position (bits - 1 - b)
                value = value | (bit << (bits - 1 - b))
            else:
                # LSB first: bit b is at position b
                value = value | (bit << b)
        
        return value
    
    return te.compute(oshape, unpack_compute, name=name, tag="bitunpack")


def binary_op_multiplier(pack_dtype):
    """ "Returns number of bits packed into
    pack_dtype: string
        pack type for the operator (must be a uint)"""
    return int(pack_dtype[4:])
