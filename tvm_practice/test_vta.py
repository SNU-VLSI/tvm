from __future__ import absolute_import, print_function

import os
import tvm
from tvm import te
import vta
import numpy as np
from tvm import rpc
from tvm.contrib import utils
from vta.testing import simulator

if __name__ == "__main__":
  env = vta.get_env()
  print(env.cfg_dict)

  remote = rpc.LocalSession()

  # Output channel factor m - total 64 x 16 = 1024 output channels
  m = 64
  # Batch factor o - total 1 x 1 = 1
  o = 1
  # A placeholder tensor in tiled data format
  A = te.placeholder((o, m, env.BATCH, env.BLOCK_OUT), name="A", dtype=env.acc_dtype)
  # B placeholder tensor in tiled data format
  B = te.placeholder((o, m, env.BATCH, env.BLOCK_OUT), name="B", dtype=env.acc_dtype)

  # A copy buffer
  A_buf = te.compute((o, m, env.BATCH, env.BLOCK_OUT), lambda *i: A(*i), "A_buf")
  # B copy buffer
  B_buf = te.compute((o, m, env.BATCH, env.BLOCK_OUT), lambda *i: B(*i), "B_buf")

  # Describe the in-VTA vector addition
  C_buf = te.compute(
      (o, m, env.BATCH, env.BLOCK_OUT),
      lambda *i: A_buf(*i).astype(env.acc_dtype) + B_buf(*i).astype(env.acc_dtype),
      name="C_buf",
  )

  # Cast to output type, and send to main memory
  C = te.compute(
      (o, m, env.BATCH, env.BLOCK_OUT), lambda *i: C_buf(*i).astype(env.inp_dtype), name="C"
  )

  # Let's take a look at the generated schedule
  s = te.create_schedule(C.op)

  print(tvm.lower(s, [A, B, C], simple_mode=True))

  s[A_buf].set_scope(env.acc_scope)
  s[B_buf].set_scope(env.acc_scope)
  s[C_buf].set_scope(env.acc_scope)

  # Tag the buffer copies with the DMA pragma to map a copy loop to a
  # DMA transfer operation
  s[A_buf].pragma(s[A_buf].op.axis[0], env.dma_copy)
  s[B_buf].pragma(s[B_buf].op.axis[0], env.dma_copy)
  s[C].pragma(s[C].op.axis[0], env.dma_copy)

  # Tell TVM that the computation needs to be performed
  # on VTA's vector ALU
  s[C_buf].pragma(C_buf.op.axis[0], env.alu)

  # Let's take a look at the finalized schedule
  print(vta.lower(s, [A, B, C], simple_mode=True))

  my_vadd = vta.build(
      s, [A, B, C], tvm.target.Target("ext_dev", host=env.target_host), name="my_vadd"
  )
  print(my_vadd.get_source())

  # Write the compiled module into an object file.
  temp = utils.tempdir()
  my_vadd.save(temp.relpath("vadd.o"))

  # Send the executable over RPC
  remote.upload(temp.relpath("vadd.o"))

  # Send the executable over RPC
  f = remote.load_module("vadd.o")

  # Get the remote device context
  ctx = remote.ext_dev(0)

  # Initialize the A and B arrays randomly in the int range of (-128, 128]
  A_orig = np.random.randint(-128, 128, size=(o * env.BATCH, m * env.BLOCK_OUT)).astype(A.dtype)
  B_orig = np.random.randint(-128, 128, size=(o * env.BATCH, m * env.BLOCK_OUT)).astype(B.dtype)

  # Apply packing to the A and B arrays from a 2D to a 4D packed layout
  A_packed = A_orig.reshape(o, env.BATCH, m, env.BLOCK_OUT).transpose((0, 2, 1, 3))
  B_packed = B_orig.reshape(o, env.BATCH, m, env.BLOCK_OUT).transpose((0, 2, 1, 3))

  # Format the input/output arrays with tvm.nd.array to the DLPack standard
  A_nd = tvm.nd.array(A_packed, ctx)
  B_nd = tvm.nd.array(B_packed, ctx)
  C_nd = tvm.nd.array(np.zeros((o, m, env.BATCH, env.BLOCK_OUT)).astype(C.dtype), ctx)

  # Invoke the module to perform the computation
  f(A_nd, B_nd, C_nd)

  # Compute reference result with numpy
  C_ref = (A_orig.astype(env.acc_dtype) + B_orig.astype(env.acc_dtype)).astype(C.dtype)
  C_ref = C_ref.reshape(o, env.BATCH, m, env.BLOCK_OUT).transpose((0, 2, 1, 3))
  np.testing.assert_equal(C_ref, C_nd.numpy())
  print("Successful vector add test!")