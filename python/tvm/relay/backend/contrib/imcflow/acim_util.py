import argparse
from enum import Enum
import numpy as np
from tvm import relay

class EnumAction(argparse.Action):
  """
  Argparse action for handling Enums
  """

  def __init__(self, **kwargs):
    # Pop off the type value
    enum_type = kwargs.pop("type", None)

    # Ensure an Enum subclass is provided
    if enum_type is None:
      raise ValueError("type must be assigned an Enum when using EnumAction")
    if not issubclass(enum_type, Enum):
      raise TypeError("type must be an Enum when using EnumAction")

    # Convert default string to Enum if needed
    default = kwargs.get("default")
    if isinstance(default, enum_type):
       pass
    elif isinstance(default, str):
      try:
        kwargs["default"] = cast_str_to_enum(default, enum_type)
      except KeyError:
        raise ValueError(f"Invalid default '{default}' for Enum '{enum_type.__name__}'")
    else:
       raise TypeError(f"Default value must be an Enum or a string, got {type(default)}")

    # Generate choices from the Enum
    kwargs.setdefault("choices", tuple(str(e) for e in enum_type))

    super(EnumAction, self).__init__(**kwargs)

    self._enum = enum_type

  def __call__(self, parser, namespace, values, option_string=None):
    # Convert value back into an Enum
    value = cast_str_to_enum(values, self._enum)
    setattr(namespace, self.dest, value)

class ADCMode(Enum):
  SIX = 0
  FIVE = 1
  FOUR = 2


class VMode(Enum):
  FULL = 0
  HALF = 1
  QRTR = 2


class MultMode(Enum):
  SINGLE = 0
  DOUBLE = 1
  TRIPLE = 2
  QUADRU = 3


class MultModeSet(str, Enum):
  S4 = "00000000"
  S2D1 = "00000101"
  D1S2 = "01010000"
  D2 = "01010101"
  S1T1 = "00101010"
  T1S1 = "10101000"
  Q1 = "11111111"

  @property
  def as_int(self) -> int:
    return int(self.value, 2)


class AccMask(Enum):
  BM_0000 = 0
  BM_0001 = 1
  BM_0010 = 2
  BM_0011 = 3
  BM_0100 = 4
  BM_0101 = 5
  BM_0110 = 6
  BM_0111 = 7
  BM_1000 = 8
  BM_1001 = 9
  BM_1010 = 10
  BM_1011 = 11
  BM_1100 = 12
  BM_1101 = 13
  BM_1110 = 14
  BM_1111 = 15

def cast_str_to_enum(s: str, expected_enum_type=None):
    """
    Cast a string like "ADCMode.SIX" or "SIX" into an Enum member.
    If expected_enum_type is given, prefix must match or be omitted.
    """
    if not isinstance(s, str):
        raise TypeError(f"Expected string, got {type(s)}")

    s = s.strip()
    if '.' in s:
        prefix, member = s.split('.', 1)
        if expected_enum_type and prefix != expected_enum_type.__name__:
            raise ValueError(f"Enum type mismatch: got '{prefix}', expected '{expected_enum_type.__name__}'")
        enum_type = globals().get(prefix)
        if enum_type is None or not issubclass(enum_type, Enum):
            raise ValueError(f"Unknown enum type: '{prefix}'")
    else:
        if expected_enum_type is None:
            raise ValueError("Enum type must be specified for unqualified name")
        enum_type = expected_enum_type
        member = s

    try:
        return enum_type[member]
    except KeyError:
        raise ValueError(f"Enum '{enum_type.__name__}' has no member '{member}'")

class ConfigData(dict):
  def __init__(self, data_shape, weight_shape, padding, stride,
               adcmode=ADCMode.SIX, vmode=VMode.FULL, multmode_set=MultModeSet.S4,
               acc_mask=AccMask.BM_0000, use_imcu=1):
    
    # Helper function to convert TVM types to Python int
    def to_int(val):
      """Convert TVM Array, PrimExpr, IntImm, or other types to Python int"""
      if isinstance(val, int):
        return val
      elif hasattr(val, '__int__'):
        return int(val)
      elif hasattr(val, 'value'):  # IntImm, Integer
        return int(val.value)
      else:
        try:
          return int(val)
        except:
          raise TypeError(f"Cannot convert {type(val)} to int: {val}")
    
    # Helper function to extract single value from tuple/list/Array
    def extract_single(val):
      """Extract single value from tuple/list/Array, or return as-is"""
      if isinstance(val, (list, tuple)):
        return to_int(val[0]) if len(val) > 0 else 0
      # TVM Array
      elif hasattr(val, '__getitem__') and hasattr(val, '__len__'):
        return to_int(val[0]) if len(val) > 0 else 0
      else:
        return to_int(val)
    
    # Convert all shape/padding/stride values to Python int
    ksel = to_int(weight_shape[2])
    pad = extract_single(padding)
    stride_val = extract_single(stride)
    W = to_int(data_shape[3])
    H = to_int(data_shape[2])
    
    super().__init__({
        "ksel": ksel,
        "pad": pad,
        "stride": stride_val,
        "W": W,
        "H": H,
        "adcmode": adcmode.value,
        "vmode": vmode.value,
        "multmode_set": multmode_set.as_int,
        "acc_mask": acc_mask.value,
        "use_imcu": use_imcu
    })

  def get_reg(self):
    reg_val = 0
    reg_val |= (self["ksel"] & 0x7) << 0
    reg_val |= (self["pad"] & 0x7) << 3
    reg_val |= (self["stride"] & 0x3) << 6
    reg_val |= (self["W"] & 0x7F) << 8
    reg_val |= (self["H"] & 0x7F) << 15
    reg_val |= (self["adcmode"] & 0x3) << 22
    reg_val |= (self["vmode"] & 0x3) << 24
    reg_val |= (self["multmode_set"] & 0xFF) << 26
    reg_val |= (self["acc_mask"] & 0xF) << 34
    reg_val |= (self["use_imcu"] & 0x1) << 38

    return reg_val
  
  def get_as_const_tensor(self):
    """
    return 1D tensor of length 8 and data type is uint32.
    slice reg_val 32bit and return as tensor. there are 2 slice and 6 zero padding slice.
    """
    print("ConfigData:", self)
    reg_val = self.get_reg()
    # Split 39-bit reg_val into two 32-bit slices
    slice0 = reg_val & 0xFFFFFFFF  # Lower 32 bits
    slice1 = (reg_val >> 32) & 0xFFFFFFFF  # Upper 7 bits (bits 32-38)
    # Create tensor with 2 data slices and 6 zero padding slices
    tensor_data = np.array([slice0, slice1, 0, 0, 0, 0, 0, 0], dtype=np.uint32)
    return relay.const(tensor_data, dtype="uint32")