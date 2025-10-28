import torch
import traceback
import inspect
import logging

logger = logging.getLogger(__name__)


def in_range(x: torch.Tensor, min, max):
  max_val = torch.max(x)
  min_val = torch.min(x)
  if (max_val > max) or (min_val < min):
    return False
  return True


def log_out_of_range(x, min, max, is_error=False):
  max_val = torch.max(x)
  min_val = torch.min(x)
  if not in_range(x, min, max):
    warning_msg = f"Value is out of range for {min}, {max}. Min: {min_val}, Max: {max_val}, Range: [{min}, {max}]"
    if is_error:
      logger.error(warning_msg)
    else:
      logger.warning(warning_msg)
      # Uncomment the next line if you want to see the full traceback
      # traceback.print_stack()


def to_int16(x, assert_range=False):
  if not isinstance(x, torch.Tensor):
    x = torch.tensor(x)

  # Log out of range
  log_out_of_range(x, -2**15, 2**15 - 1, is_error=assert_range)

  # Clamp values to int16 range and convert
  return torch.clamp(x, -2**15, 2**15 - 1).to(torch.int32)


def to_uint4(x, assert_range=False):
  if not isinstance(x, torch.Tensor):
    x = torch.tensor(x)

  # Log out of range
  log_out_of_range(x, 0, 2**4 - 1, is_error=assert_range)

  # Clamp values to uint4 range and convert
  return torch.clamp(x, 0, 2**4 - 1).to(torch.int32)


def to_int4(x, assert_range=False):
  if not isinstance(x, torch.Tensor):
    x = torch.tensor(x)

  # Log out of range
  log_out_of_range(x, -2**3, 2**3 - 1, is_error=assert_range)

  # Clamp values to int4 range and convert
  return torch.clamp(x, -2**3, 2**3 - 1).to(torch.int32)