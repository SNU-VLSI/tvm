import tvm
from tvm import relay
import numpy as np

def get_param_info_from_relay_func(y):
  # Collect parameter vars from the graph (exclude the input var)
  free_vars = relay.analysis.free_vars(y)
  var_info = {}
  for v in free_vars:
    if v.name_hint == "model_input":
      continue
    name = v.name_hint
    # Deduplicate by name in case of separately-constructed Vars with the same name
    if name in var_info:
      continue
    ttype = v.type_annotation
    if isinstance(ttype, relay.ty.TensorType):
      # Convert TVM shape (IntImm / PrimExpr) to Python ints when possible
      shape = []
      for dim in ttype.shape:
        try:
          shape.append(int(dim))
        except Exception:
          # Fallback if dynamic: leave as-is
          shape.append(dim)
      var_info[name] = {"shape": tuple(shape), "dtype": ttype.dtype}
    else:
      # If no TensorType annotation, skip or set defaults
      continue
  return var_info

def _rand_tensor(dtype: str, shape):
  # Handle common dtypes with appropriate ranges
  if dtype in ("float32", "float16", "float64"):
    return np.random.uniform(-1, 1, shape).astype(dtype)
  if dtype.startswith("int"):
    # Parse bit width if available (e.g., int4, int8, int16, int32)
    try:
      bits = int(dtype.replace("int", ""))
    except Exception:
      bits = 32
    if bits == 4:
      # No native int4 in numpy; store in int8 within valid int4 range
      return np.random.randint(-8, 8, size=shape, dtype=np.int8)
    if bits == 8:
      return np.random.randint(-128, 128, size=shape, dtype=np.int8)
    if bits == 16:
      return np.random.randint(-32768, 32768, size=shape, dtype=np.int16)
    if bits == 32:
      return np.random.randint(-2**31, 2**31, size=shape, dtype=np.int32)
    if bits == 64:
      return np.random.randint(-2**63, 2**63 - 1, size=shape, dtype=np.int64)
    # Fallback: use int32
    return np.random.randint(-2**31, 2**31, size=shape, dtype=np.int32)
  if dtype.startswith("uint"):
    try:
      bits = int(dtype.replace("uint", ""))
    except Exception:
      bits = 32
    if bits == 4:
      return np.random.randint(0, 16, size=shape, dtype=np.uint8)
    if bits == 8:
      return np.random.randint(0, 256, size=shape, dtype=np.uint8)
    if bits == 16:
      return np.random.randint(0, 2**16, size=shape, dtype=np.uint16)
    if bits == 32:
      return np.random.randint(0, 2**32, size=shape, dtype=np.uint32)
    if bits == 64:
      # numpy uint64 randint high is exclusive and must be <= 2**64-1
      return np.random.randint(0, np.iinfo(np.uint64).max, size=shape, dtype=np.uint64)
    return np.random.randint(0, 2**32, size=shape, dtype=np.uint32)
  # Default float32 if unrecognized
  return np.random.uniform(-1, 1, shape).astype("float32")