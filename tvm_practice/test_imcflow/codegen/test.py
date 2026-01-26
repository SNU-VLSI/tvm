import pytest
import tvm
import numpy as np
import tvm.testing
from tvm.contrib.relay_viz import RelayVisualizer, DotPlotter, DotVizParser
from tvm.contrib import graph_executor
from tvm.relay.build_module import bind_params_by_name
from tvm.contrib.debugger import debug_executor
from tvm.relay import transform
from tvm.relay.backend.contrib.imcflow import transform as imcflow_transform
from tvm.relay.backend.contrib.imcflow import cpu_run as cpu_run
from tvm.relay.op.contrib import imcflow
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from tvm.relay.backend import Executor, Runtime
import os
import shutil
import subprocess
import copy
import pprint
import glob
import pickle
import sys
from contextlib import contextmanager

# Import IMCFlow compiler driver
from tvm.driver.tvmc.imcflow_compiler_driver import compile_for_imcflow, rebuild_imcflow_cpp_only

from models import real_model, real_model2, test_models
from models import resnet8_cifar, mobilenet_imcflow, deep_autoencoder_imcflow, ds_cnn_imcflow
from models import resnet8_subset_models
from models import models_for_test

# Import shared input generator
from input_generator import InputGenerator

# Import ImcFlow runner abstraction
from imcflow_runner import get_runner

np.random.seed(1234)

DEBUG_EXECUTOR=1
DEBUG_SUBSET=1

# Print environment configuration at startup
print(f"Environment: IMCFLOW_RUNNER={os.getenv('IMCFLOW_RUNNER', 'py')}, IMCFLOW_DEBUG={os.getenv('IMCFLOW_DEBUG', '0')}")

# ============================================================================
# Model Registry
# ============================================================================
# Maps test_name -> (model_getter_function, default_input_pattern)
# default_input_pattern: "random", "ones", "zeros", "linear"
# NOTE: it is recommended to not include the input pattern strings in the test_name
# NOTE: e.g. one_relu_random is discouraged, as it makes harder to collect pytest patterns
MODEL_REGISTRY = {
    # Simple test models
    "one_relu": (models_for_test.getOneReluModel, "linear"),
    "one_conv_small": (lambda: models_for_test.getOneConvModel(iH=4,iW=4), "random"),
    "one_conv_big": (lambda: models_for_test.getOneConvModel(iH=32,iW=32), "random"),
    "one_conv_wide": (lambda: models_for_test.getOneConvModel(iH=4,iW=4,IC=56), "random"),
    "conv_quant_conv_big": (lambda: models_for_test.getConvQuantConvModel(iH=32, iW=32), "random"),
    "s2_conv_quant_conv_med": (lambda: models_for_test.getS2ConvQuantConvModel(iH=16, iW=16), "random"),
    "s2_conv_quant_conv_big": (lambda: models_for_test.getS2ConvQuantConvModel(iH=32, iW=64), "random"),
    "one_mmquant": (models_for_test.getOneMMQuantModel, "linear"),
    "one_conv_quant": (models_for_test.getOneConvQuantModel, "ones"),
    "one_fused_bn" : (models_for_test.getOneFusedBNModel, "random"),
    "one_conv_bn": (models_for_test.getOneConvBnModel, "ones"),
    "big_conv": (lambda: models_for_test.getBigConvModel(False), "random"),
    "big_conv_rparam": (lambda: models_for_test.getBigConvModel(True), "random"),
    "super_big_conv_rev1"           : (lambda: models_for_test.getSuperBigConvModel([1, 28, 1, 1], 128, False), "random"),
    "super_big_conv_rparam_rev1"    : (lambda: models_for_test.getSuperBigConvModel([1, 28, 1, 1], 128, True), "random"),
    "super_big_conv_rev2"           : (lambda: models_for_test.getSuperBigConvModel([1, 56, 1, 1], 128, False), "random"),
    "super_big_conv_rparam_rev2"    : (lambda: models_for_test.getSuperBigConvModel([1, 56, 1, 1], 128, True), "random"),
    "super_big_conv_rev3"           : (lambda: models_for_test.getSuperBigConvModel([1, 56, 2, 2], 128, False), "random"),
    "super_big_conv_rparam_rev3"    : (lambda: models_for_test.getSuperBigConvModel([1, 56, 2, 2], 128, True), "random"),
    "super_big_conv_rev4"           : (lambda: models_for_test.getSuperBigConvModel([1, 56, 4, 4], 128, False), "random"),
    "super_big_conv_rparam_rev4"    : (lambda: models_for_test.getSuperBigConvModel([1, 56, 4, 4], 128, True), "random"),
    "super_big_conv_rev5"           : (lambda: models_for_test.getSuperBigConvModel([1, 64, 4, 4], 128, False), "random"),
    "super_big_conv_rparam_rev5"    : (lambda: models_for_test.getSuperBigConvModel([1, 64, 4, 4], 128, True), "random"),

    "super_big_conv_bn_quant_rev1"       : (lambda: models_for_test.getSuperBigConvBnQuantModel([1, 28, 1, 1], 128, False), "ones"),
    "super_big_conv_bn_quant_rparam_rev1": (lambda: models_for_test.getSuperBigConvBnQuantModel([1, 28, 1, 1], 128, True),  "ones"),
    "super_big_conv_bn_quant_rev2"       : (lambda: models_for_test.getSuperBigConvBnQuantModel([1, 28, 4, 4], 128, False), "ones"),
    "super_big_conv_bn_quant_rparam_rev2": (lambda: models_for_test.getSuperBigConvBnQuantModel([1, 28, 4, 4], 128, True),  "ones"),
    "super_big_conv_bn_quant_rev3"       : (lambda: models_for_test.getSuperBigConvBnQuantModel([1, 56, 1, 1], 128, False), "ones"),
    "super_big_conv_bn_quant_rparam_rev3": (lambda: models_for_test.getSuperBigConvBnQuantModel([1, 56, 1, 1], 128, True),  "ones"),
    "super_big_conv_bn_quant_rev4"       : (lambda: models_for_test.getSuperBigConvBnQuantModel([1, 56, 4, 4], 128, False), "ones"),
    "super_big_conv_bn_quant_rparam_rev4": (lambda: models_for_test.getSuperBigConvBnQuantModel([1, 56, 4, 4], 128, True),  "ones"),
    "super_big_conv_bn_quant_rev5"       : (lambda: models_for_test.getSuperBigConvBnQuantModel([1, 64, 1, 1], 128, False), "ones"),
    "super_big_conv_bn_quant_rparam_rev5": (lambda: models_for_test.getSuperBigConvBnQuantModel([1, 64, 1, 1], 128, True),  "ones"),
    "super_big_conv_bn_quant_rev6"       : (lambda: models_for_test.getSuperBigConvBnQuantModel([1, 64, 4, 4], 128, False), "ones"),
    "super_big_conv_bn_quant_rparam_rev6": (lambda: models_for_test.getSuperBigConvBnQuantModel([1, 64, 4, 4], 128, True),  "ones"),

    "conv_bn_quant": (models_for_test.getConvBNQuantModel, "ones"),
    "conv_bn_mult_add": (models_for_test.getConvBNMultAddModel, "ones"),
    "conv_quant_conv": (lambda: models_for_test.getConvQuantConvModel(iH=1,iW=1), "ones"),
    "s2_conv_quant_conv": (lambda: models_for_test.getS2ConvQuantConvModel(iH=1,iW=1), "random"),
    "big_conv_quant_conv": (models_for_test.getBigConvQuantConvModel, "ones"),
    "residual_model": (lambda: models_for_test.getResidualModel(False), "ones"),
    "residual_rnd_model": (lambda: models_for_test.getResidualModel(True), "ones"),

    # ResNet8 variants
    "resnet8_subset04_pretrained_super_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=2, iW=2, until_relay=4), "ones"),
    "resnet8_subset05_pretrained_super_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=2, iW=2, until_relay=5), "ones"),
    "resnet8_subset06_pretrained_super_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=2, iW=2, until_relay=6), "ones"),
    "resnet8_subset07_pretrained_super_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=2, iW=2, until_relay=7), "ones"),
    "resnet8_subset08_pretrained_super_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=2, iW=2, until_relay=8), "ones"),
    "resnet8_subset09_pretrained_super_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=2, iW=2, until_relay=9), "ones"),
    "resnet8_subset10_pretrained_super_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=2, iW=2, until_relay=10), "ones"),
    "resnet8_subset11_pretrained_super_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=2, iW=2, until_relay=11), "ones"),
    "resnet8_subset12_pretrained_super_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=2, iW=2, until_relay=12), "ones"),
    "resnet8_subset13_pretrained_super_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=2, iW=2, until_relay=13), "ones"),
    "resnet8_subset14_pretrained_super_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=2, iW=2, until_relay=14), "ones"),
    "resnet8_subset15_pretrained_super_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=2, iW=2, until_relay=15), "ones"),
    "resnet8_subset16_pretrained_super_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=2, iW=2, until_relay=16), "ones"),
    "resnet8_subset17_pretrained_super_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=2, iW=2, until_relay=17), "ones"),
    "resnet8_subset18_pretrained_super_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=2, iW=2, until_relay=18), "ones"),
    "resnet8_subset19_pretrained_super_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=2, iW=2, until_relay=19), "ones"),
    "resnet8_subset20_pretrained_super_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=2, iW=2, until_relay=20), "ones"),
    "resnet8_subset21_pretrained_super_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=2, iW=2, until_relay=21), "ones"),
    "resnet8_subset22_pretrained_super_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=2, iW=2, until_relay=22), "ones"),
    "resnet8_subset23_pretrained_super_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=2, iW=2, until_relay=23), "ones"),
    "resnet8_subset24_pretrained_super_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=2, iW=2, until_relay=24), "ones"),
    "resnet8_subset25_pretrained_super_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=2, iW=2, until_relay=25), "ones"),

    # ResNet8 variants
    "resnet8_subset04_pretrained_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=8, iW=8, until_relay=4), "ones"),
    "resnet8_subset05_pretrained_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=8, iW=8, until_relay=5), "ones"),
    "resnet8_subset06_pretrained_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=8, iW=8, until_relay=6), "ones"),
    "resnet8_subset07_pretrained_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=8, iW=8, until_relay=7), "ones"),
    "resnet8_subset08_pretrained_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=8, iW=8, until_relay=8), "ones"),
    "resnet8_subset09_pretrained_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=8, iW=8, until_relay=9), "ones"),
    "resnet8_subset10_pretrained_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=8, iW=8, until_relay=10), "ones"),
    "resnet8_subset11_pretrained_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=8, iW=8, until_relay=11), "ones"),
    "resnet8_subset12_pretrained_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=8, iW=8, until_relay=12), "ones"),
    "resnet8_subset13_pretrained_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=8, iW=8, until_relay=13), "ones"),
    "resnet8_subset14_pretrained_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=8, iW=8, until_relay=14), "ones"),
    "resnet8_subset15_pretrained_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=8, iW=8, until_relay=15), "ones"),
    "resnet8_subset16_pretrained_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=8, iW=8, until_relay=16), "ones"),
    "resnet8_subset17_pretrained_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=8, iW=8, until_relay=17), "ones"),
    "resnet8_subset18_pretrained_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=8, iW=8, until_relay=18), "ones"),
    "resnet8_subset19_pretrained_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=8, iW=8, until_relay=19), "ones"),
    "resnet8_subset20_pretrained_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=8, iW=8, until_relay=20), "ones"),
    "resnet8_subset21_pretrained_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=8, iW=8, until_relay=21), "ones"),
    "resnet8_subset22_pretrained_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=8, iW=8, until_relay=22), "ones"),
    "resnet8_subset23_pretrained_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=8, iW=8, until_relay=23), "ones"),
    "resnet8_subset24_pretrained_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=8, iW=8, until_relay=24), "ones"),
    "resnet8_subset25_pretrained_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=8, iW=8, until_relay=25), "ones"),
    "resnet8_subset31_pretrained_small": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=8, iW=8, until_relay=31), "ones"),

    "resnet8_subset01_pretrained_orig": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=32, iW=32, until_relay=1), "ones"),
    "resnet8_subset02_pretrained_orig": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=32, iW=32, until_relay=2), "ones"),
    "resnet8_subset03_pretrained_orig": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=32, iW=32, until_relay=3), "ones"),
    "resnet8_subset04_pretrained_orig": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=32, iW=32, until_relay=4), "ones"),
    "resnet8_subset05_pretrained_orig": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=32, iW=32, until_relay=5), "ones"),
    "resnet8_subset06_pretrained_orig": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=32, iW=32, until_relay=6), "ones"),
    "resnet8_subset07_pretrained_orig": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=32, iW=32, until_relay=7), "ones"),
    "resnet8_subset08_pretrained_orig": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=32, iW=32, until_relay=8), "ones"),
    "resnet8_subset09_pretrained_orig": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=32, iW=32, until_relay=9), "ones"),
    "resnet8_subset10_pretrained_orig": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=32, iW=32, until_relay=10), "ones"),
    "resnet8_subset11_pretrained_orig": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=32, iW=32, until_relay=11), "ones"),
    "resnet8_subset12_pretrained_orig": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=32, iW=32, until_relay=12), "ones"),
    "resnet8_subset13_pretrained_orig": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=32, iW=32, until_relay=13), "ones"),
    "resnet8_subset14_pretrained_orig": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=32, iW=32, until_relay=14), "ones"),
    "resnet8_subset15_pretrained_orig": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=32, iW=32, until_relay=15), "ones"),
    "resnet8_subset16_pretrained_orig": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=32, iW=32, until_relay=16), "ones"),
    "resnet8_subset17_pretrained_orig": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=32, iW=32, until_relay=17), "ones"),
    "resnet8_subset18_pretrained_orig": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=32, iW=32, until_relay=18), "ones"),
    "resnet8_subset19_pretrained_orig": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=32, iW=32, until_relay=19), "ones"),
    "resnet8_subset20_pretrained_orig": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=32, iW=32, until_relay=20), "ones"),
    "resnet8_subset21_pretrained_orig": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=32, iW=32, until_relay=21), "ones"),
    "resnet8_subset22_pretrained_orig": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=32, iW=32, until_relay=22), "ones"),
    "resnet8_subset23_pretrained_orig": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=32, iW=32, until_relay=23), "ones"),
    "resnet8_subset24_pretrained_orig": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=32, iW=32, until_relay=24), "ones"),
    "resnet8_subset25_pretrained_orig": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=32, iW=32, until_relay=25), "ones"),
    "resnet8_subset31_pretrained_orig": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=32, iW=32, until_relay=31), "ones"),

    "resnet8_subset31_pretrained_big": (lambda: resnet8_subset_models.getModel_from_pretrained_weight(iH=16, iW=64, until_relay=31), "ones"),
    "resnet8_last_bb" : (lambda: models_for_test.getResidualPathTest(small_debug=True, random_param=True), "ones"),

    # "resnet8_small": (lambda: resnet8_cifar.getModel(True), "ones"),
    "resnet8_small_pretrained": (lambda: resnet8_cifar.getModel_from_pretrained_weight(True), "ones"),
    # "resnet_cifar10_small": (lambda: models_for_test.getResnetCifar10Small(small_debug=True), "ones"),
    # "resnet_cifar10_small_pretrained": (lambda: models_for_test.getResnetCifar10SmallPretrained(small_debug=True), "ones"),
    # "resnet_cifar10_subset_small_manual_param": (lambda: models_for_test.getResnetCifar10SmallManualParam(small_debug=True), "ones"),

    # Other models
    # "mobilenet_imcflow": (lambda: mobilenet_imcflow.getModel(False), "random"),
    # "deep_autoencoder_imcflow": (lambda: deep_autoencoder_imcflow.getModel(False), "random"),
    # "ds_cnn_imcflow": (lambda: ds_cnn_imcflow.getModel(False), "random"),

    # Legacy models (for backward compatibility)
    # "big": (real_model.getModel, "random"),
    # "small": (real_model2.getModel, "random"),
    # "model_v2": (real_model2.getModelV2, "random"),
    # "model_1": (test_models.get_model1, "random"),
}

# Available input patterns for testing
INPUT_PATTERNS = ["random", "ones", "zeros", "linear"]

# ============================================================================
# Utility Functions
# ============================================================================
class TeeLogger:
  """A class that writes to both stdout and a file simultaneously"""
  def __init__(self, log_file):
    self.terminal = sys.stdout
    self.log = log_file

  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)
    self.log.flush()  # Ensure immediate write to file

  def flush(self):
    self.terminal.flush()
    self.log.flush()


@contextmanager
def tee_output_to_log(log_path):
  """Context manager that tees stdout/stderr to both console and log file

  Args:
    log_path: Path to the log file
  """
  # Save original stdout/stderr
  original_stdout = sys.stdout
  original_stderr = sys.stderr

  # Open log file
  log_file = open(log_path, 'w')

  try:
    # Create tee logger
    tee_logger = TeeLogger(log_file)

    # Redirect both stdout and stderr to tee logger
    sys.stdout = tee_logger
    sys.stderr = tee_logger

    yield
  finally:
    # Restore original stdout/stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr

    # Close log file
    log_file.close()


def save_transformed_model(mod, param_dict, eval_dir, pkl_name="transformed_model.pkl"):
  """Save transformed model and parameters to file for reuse

  Args:
    mod: Transformed TVM relay module
    param_dict: Transformed model parameters
    eval_dir: Directory to save the model
    pkl_name: Name of the pickle file to save
  """
  import pickle
  model_save_path = os.path.join(eval_dir, pkl_name)

  save_data = {
    "mod": mod,
    "param_dict": param_dict,
  }

  with open(model_save_path, "wb") as f:
    pickle.dump(save_data, f)

  print(f"💾 Saved transformed model to: {model_save_path}")


def load_transformed_model(eval_dir, pkl_name="transformed_model.pkl"):
  """Load previously transformed model and parameters from file

  Args:
    eval_dir: Directory containing the saved model
    pkl_name: Name of the pickle file to load

  Returns:
    tuple: (mod, param_dict)

  Raises:
    FileNotFoundError: If transformed model file doesn't exist
  """
  import pickle
  model_save_path = os.path.join(eval_dir, pkl_name)

  if not os.path.exists(model_save_path):
    raise FileNotFoundError(
      f"Transformed model not found at: {model_save_path}\n"
      f"Cannot use skip_setup=True without a previous run.\n"
      f"Run without --skip-setup first to compile and save the model."
    )

  with open(model_save_path, "rb") as f:
    save_data = pickle.load(f)

  print(f"📂 Loaded transformed model from: {model_save_path}")
  return save_data["mod"], save_data["param_dict"]


def setup_dir(test_name, suffix=""):
  def clean_dir_recursive(path):
    """Recursively clean all files but keep directory structure intact."""
    for item in os.listdir(path):
      item_path = os.path.join(path, item)
      if os.path.isfile(item_path) or os.path.islink(item_path):
        os.remove(item_path)
      elif os.path.isdir(item_path) and item != "logs":
        clean_dir_recursive(item_path)

  def clean_runner_logs(logs_path, runner_dirs):
    """Clean specific runner log directories."""
    for runner_dir in runner_dirs:
      runner_path = os.path.join(logs_path, runner_dir)
      if os.path.exists(runner_path):
        shutil.rmtree(runner_path)

  dir_name = f"{test_name}{suffix}"
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)
  else:
    clean_dir_recursive(dir_name)
    # Clean runner-specific logs based on IMCFLOW_RUNNER
    runner_env = os.getenv('IMCFLOW_RUNNER', 'py').lower()
    logs_path = os.path.join(dir_name, "logs")
    if runner_env == 'rtl':
      clean_runner_logs(logs_path, ['rtl_runner'])
    elif runner_env == 'both':
      clean_runner_logs(logs_path, ['py_runner', 'rtl_runner'])
    else:  # 'py' or default
      clean_runner_logs(logs_path, ['py_runner'])

  os.makedirs(os.path.join(dir_name, "logs"), exist_ok=True)
  os.makedirs(os.path.join(dir_name, "test_inputs"), exist_ok=True)
  os.makedirs(os.path.join(dir_name, "test_outputs"), exist_ok=True)
  os.makedirs(os.path.join(dir_name, "test_references"), exist_ok=True)

  return dir_name


def printModel(result_dir, mod, param_dict, mod_name):
  RelayVisualizer(
      relay_mod=mod,
      relay_param=param_dict,
      plotter=DotPlotter(),
      parser=DotVizParser(),
  ).render(f"{result_dir}/{mod_name}")

  with open(f"{result_dir}/{mod_name}.txt", "w") as f:
    # f.write(pretty_print(mod))
    f.write(mod.astext(show_meta_data=True))

def run_cpu_validation(mod, param_dict, input_data_dict, model_dir, skip_setup=False):
  """Run transformed model on CPU for validation

  Args:
    mod: The TVM relay module
    param_dict: Model parameters
    input_data_dict: Dictionary of input name -> numpy array
    model_dir: Directory to save CPU outputs
    skip_setup: If True, load from previously transformed CPU model

  Returns:
    output: The CPU execution output as numpy array
  """
  print("\n" + "="*40)
  print("RUNNING CPU VALIDATION")
  print("="*40)

  target = "llvm"
  ctx = tvm.cpu(0)

  if not skip_setup:
    # Transform model to be CPU runnable
    # cpu_mod = copy.deepcopy(mod)
    cpu_mod = copy.copy(mod)
    cpu_mod = cpu_run.make_cpu_runnable(cpu_mod)
    printModel(model_dir, cpu_mod, param_dict, "cpu_runnable_model")

    # Save the CPU runnable model
    save_transformed_model(cpu_mod, param_dict, model_dir, pkl_name="transformed_cpu_model.pkl")
  else:
    # Load previously transformed CPU model
    print("⏭️  Skipping CPU model transformation, loading from file...")
    cpu_mod, param_dict = load_transformed_model(model_dir, pkl_name="transformed_cpu_model.pkl")

  executor_ = Executor("graph")
  runtime_  = Runtime("crt", {"system-lib": True})
  with tvm.transform.PassContext(opt_level=0, config={"tir.disable_vectorize": True}):
    graph, lib, params = tvm.relay.build(cpu_mod, target=target, params=param_dict,
                                         executor=executor_, runtime=runtime_)

  if DEBUG_EXECUTOR:
    executor = debug_executor.create(graph, lib, device=ctx)
  else:
    executor = graph_executor.create(graph, lib, device=ctx)

  # Load constant parameters
  if params:
    executor.load_params(tvm.runtime.save_param_dict(params))

  # Set input data
  if input_data_dict:
    for name, data in input_data_dict.items():
      executor.set_input(name, data)

  # Run inference
  executor.run()

  if DEBUG_EXECUTOR:
    print("Debug executor output tensors:")
    tvm_dict = executor.debug_datum.get_output_tensors()
    print(tvm_dict)
    np_dict = {}
    for k, v in tvm_dict.items():
      np_dict[k] = v.asnumpy()
    pickle.dump(np_dict, open(f"{model_dir}/debug_executor_output_tensors.pkl", "wb"))

  # Get output
  output = executor.get_output(0).asnumpy()

  # Save output for reference
  output_dir = os.path.abspath(os.path.join(model_dir, "test_references"))
  np.save(f"{output_dir}/cpu_reference_output.npy", output)
  print(f"CPU output saved to: {output_dir}/cpu_reference_output.npy")
  print(f"CPU output shape: {output.shape}, dtype: {output.dtype}")

  return output


def run_simulation(eval_dir, HOST_ISA="x86"):
  """Run simulation by building and executing the graph with proper output streaming

  Args:
    eval_dir: Evaluation directory containing the model

  Returns:
    imcflow_output: The IMCFLOW simulation output as numpy array, or None if output file doesn't exist.
                    When running both runners, returns the py_runner output for backward compatibility.
  """
  log_dir = f"{eval_dir}/logs"

  print("\n" + "="*60)
  print("RUNNING SIMULATION")
  print("="*60)

  # Build the host binary
  print("\n--- Building Host Binary ---")

  # Copy host_binary_make template to test directory if it doesn't exist
  test_host_binary_dir = f"{eval_dir}/host_binary_make"
  print(f"Copying host_binary_make template to {test_host_binary_dir}")
  shutil.copytree("./host_binary_make.template", test_host_binary_dir, dirs_exist_ok=True)

  host_build_dir = f"{test_host_binary_dir}/build"
  os.makedirs(host_build_dir, exist_ok=True)

  # Build in the test-specific directory (use current directory "." since eval_dir is now relative)
  build_command = ["direnv", "exec", ".", "../build.sh", "execute_graph.c", ".", HOST_ISA]
  build_log_path = os.path.join(log_dir, "build.log")

  with open(build_log_path, "w") as log_file:
    process = subprocess.Popen(
      build_command,
      cwd=host_build_dir,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True
    )

    # Stream output line by line to both terminal and log file
    for line in process.stdout:
      print(line, end='')
      log_file.write(line)

    process.wait()
    if process.returncode != 0:
      raise subprocess.CalledProcessError(process.returncode, build_command)

  print(f"✅ Build completed, log saved to: {build_log_path}")

  # Run gem5 simulation
  if (HOST_ISA == "arm"):
    print("\n-- Skipping gem5 simulation for ARM architecture --")
    return None

  # Get the appropriate runner(s) based on IMCFLOW_RUNNER env var
  runners = get_runner()  # Returns single runner or list of runners

  # Normalize to list for uniform handling
  if not isinstance(runners, list):
    runners = [runners]

  # Dictionary to store outputs from each runner
  outputs = {}

  # Run each runner sequentially
  for runner in runners:
    if len(runners) > 1:
      print(f"\n{'='*60}")
      print(f"Running {runner.name}")
      print(f"{'='*60}")

    # Setup runner (VCS compilation for RTL if needed)
    runner.setup()

    # Create runner-specific log directory
    runner_log_dir = os.path.join(log_dir, runner.name)
    os.makedirs(runner_log_dir, exist_ok=True)

    # Run gem5 simulation
    interrupted = False
    simul_err = False
    try:
      runner.run(
        binary_name="tvm_host_runner",
        gdb_mode="no",
        test_name=eval_dir,
        eval_dir=eval_dir
      )
    except KeyboardInterrupt:
      interrupted = True
      print("❌ Simulation interrupted by user")
    except Exception as e:
      simul_err = True
      print(f"❌ Simulation failed for {runner.name}: {e}")

    # Logs are automatically written to runner_log_dir during run()
    # No collection needed

    # Re-raise KeyboardInterrupt
    if interrupted:
      raise KeyboardInterrupt("Simulation interrupted by user")

    # Load the output from this runner
    runner_output_path = runner.get_output_path(test_name=eval_dir)

    if not simul_err:
      if os.path.exists(runner_output_path):
        output_data = np.load(runner_output_path)
        print(f"✅ {runner.name} output found at: {runner_output_path}")
        outputs[runner.name] = output_data
      else:
        print(f"⚠️  {runner.name} output not found at: {runner_output_path}")
        outputs[runner.name] = None

  # If running both runners, compare outputs
  if len(runners) > 1:
    print(f"\n--- Comparing Runner Outputs ---")
    _compare_runner_outputs(outputs)

  # Return py_runner output for backward compatibility (or first runner's output)
  if "py_runner" in outputs:
    return outputs["py_runner"]
  elif outputs:
    return list(outputs.values())[0]
  else:
    return None


def _compare_runner_outputs(outputs):
  """Compare outputs from different runners

  Args:
    outputs: Dictionary mapping runner name to output array
  """
  runner_names = list(outputs.keys())
  if len(runner_names) < 2:
    return

  # Get reference output (first runner)
  ref_name = runner_names[0]
  ref_output = outputs[ref_name]

  if ref_output is None:
    print(f"❌ Cannot compare: {ref_name} output is None")
    return

  # Compare against other runners
  all_match = True
  for runner_name in runner_names[1:]:
    other_output = outputs[runner_name]

    if other_output is None:
      print(f"❌ {runner_name} output is None")
      all_match = False
      continue

    # Shape and dtype check
    if ref_output.shape != other_output.shape:
      print(f"❌ Shape mismatch: {ref_name}{ref_output.shape} vs {runner_name}{other_output.shape}")
      all_match = False
      continue

    # Value comparison
    if ref_output.dtype in [np.float32, np.float64]:
      if np.allclose(ref_output, other_output, rtol=1e-5, atol=1e-8):
        print(f"✅ {ref_name} == {runner_name}")
      else:
        diff = np.sum(~np.isclose(ref_output, other_output, rtol=1e-5, atol=1e-8))
        print(f"❌ {ref_name} != {runner_name} ({diff}/{ref_output.size} differ)")
        all_match = False
    else:
      if np.array_equal(ref_output, other_output):
        print(f"✅ {ref_name} == {runner_name}")
      else:
        diff = np.sum(ref_output != other_output)
        print(f"❌ {ref_name} != {runner_name} ({diff}/{ref_output.size} differ)")
        all_match = False

  if not all_match:
    print(f"⚠️  Runner outputs differ")


def compare_outputs(cpu_output, imcflow_output):
  """Compare CPU reference output with IMCFLOW simulation output

  Args:
    cpu_output: CPU reference output as numpy array
    imcflow_output: IMCFLOW simulation output as numpy array

  Raises:
    pytest.fail: If outputs don't match
  """
  print("\n" + "="*60)
  print("COMPARING OUTPUTS")
  print("="*60)

  print(f"\n--- Output Comparison ---")
  print(f"CPU output shape: {cpu_output.shape}, dtype: {cpu_output.dtype}")
  print(f"IMCFLOW output shape: {imcflow_output.shape}, dtype: {imcflow_output.dtype}")

  # Shape check
  if cpu_output.shape != imcflow_output.shape:
    pytest.fail(f"Output shape mismatch: CPU {cpu_output.shape} vs IMCFLOW {imcflow_output.shape}")

  # Dtype check
  if cpu_output.dtype != imcflow_output.dtype:
    pytest.fail(f"Output dtype mismatch: CPU {cpu_output.dtype} vs IMCFLOW {imcflow_output.dtype}")

  # Value comparison
  fail = False
  if cpu_output.dtype in [np.float32, np.float64]:
    if np.allclose(cpu_output, imcflow_output, rtol=1e-5, atol=1e-8):
      print("✅ IMCFLOW output matches CPU reference fp output (within tolerance)")
      print(f"IMCFLOW == CPU reference output: {cpu_output}")
    else:
      fail_indices = np.where(~np.isclose(cpu_output, imcflow_output, rtol=1e-5, atol=1e-8))
      fail = True
  elif np.array_equal(cpu_output, imcflow_output):
    print("✅ IMCFLOW output matches CPU reference output (exact match)")
    print(f"IMCFLOW == CPU reference output: {cpu_output}")
  else:
    fail_indices = np.where(cpu_output != imcflow_output)
    fail = True
  
  if fail:
    max_print_cnt = 10
    print(f"❌ Output values do not match at {len(fail_indices[0])} locations. Showing up to {max_print_cnt} mismatches:")
    for i in range(min(len(fail_indices[0]), max_print_cnt)):
      idx = tuple(index[i] for index in fail_indices)
      print(f"  Index {idx}: CPU={cpu_output[idx]}, IMCFLOW={imcflow_output[idx]}")
    # Flush stdout to ensure tee logger captures everything before pytest.fail raises exception
    sys.stdout.flush()
    pytest.fail(f"Output comparison failed.\nReference output: {cpu_output}\nIMCFLOW output: {imcflow_output}")
  else:
    print("\n✅ Test completed successfully")

def run_test(test_name, eval_dir, mod, param_dict, input_data_dict=None, skip_setup=False, rebuild_modified_cpp=False):
  """Generate IMCFLOW evaluation results with optional CPU validation

  Args:
    test_name: Name of the test
    eval_dir: evaluation directory name
    mod: The TVM relay module (only used if skip_setup=False)
    param_dict: Model parameters (only used if skip_setup=False)
    input_data_dict: Optional dict of input name -> numpy array for CPU validation
    skip_setup: If True, skip transformations, codegen, and graph generation.
                Loads previously transformed model from file.
    rebuild_modified_cpp: If True, rebuild modified C++ files before simulation.
  """
  print(f"\n{'='*60}")
  print(f"GENERATING EVALUATION RESULTS FOR: {test_name}")
  print(f"{'='*60}")

  if not skip_setup and not rebuild_modified_cpp:
    # Full IMCFlow compilation pipeline (transform, codegen, graph executor)
    mod, param_dict, _ = compile_for_imcflow(mod, param_dict, eval_dir)

    # Save transformed model for future reuse
    save_transformed_model(mod, param_dict, eval_dir)
  else:
    # Skip setup: load previously transformed model
    print("\n⏭️  Skipping model transformation, codegen, and graph generation (skip_setup=True)")
    print("   Loading previously transformed model from file...")
    mod, param_dict = load_transformed_model(eval_dir)

  if rebuild_modified_cpp:
    mod, param_dict, _ = rebuild_imcflow_cpp_only(mod, param_dict, eval_dir)

  # Run CPU validation if input data is provided
  if input_data_dict is not None:
    cpu_output = run_cpu_validation(mod, param_dict, input_data_dict, eval_dir, skip_setup)
    if cpu_output is not None:
      print("✅ CPU validation completed successfully")

  config = DevConfig()

  # Run simulation (build + gem5 execution)
  try:
    imcflow_output = run_simulation(eval_dir, config.HOST_ISA)
  except KeyboardInterrupt:
    print("\n⚠️  Simulation interrupted - skipping output comparison")
    raise  # Re-raise to let pytest handle the interruption

  if (config.HOST_ISA == "arm"):
    return None

  # Compare the reference CPU output with IMCFLOW simulated output
  if input_data_dict is not None:
    if imcflow_output is None:
      pytest.fail(f"IMCFLOW output file missing, cannot compare outputs")

    compare_outputs(cpu_output, imcflow_output)


# ============================================================================
# Test Pipeline
# ============================================================================
def run_test_pipeline(test_name, input_pattern="default", skip_setup=False, rebuild_modified_cpp=False):
  """
  Test pipeline that:
  1. Gets the model from registry
  2. Generates and saves test inputs with specified pattern
  3. Loads test inputs for CPU validation
  4. Runs the full evaluation pipeline

  Args:
    test_name: Name of the test (must exist in MODEL_REGISTRY)
    input_pattern: Input pattern to use ("random", "ones", "zeros", "linear").
                   If "default", uses the default pattern from MODEL_REGISTRY.
    skip_setup: If True, skip codegen, and graph generation steps.
                Useful for testing different inputs on an already-compiled model.
                NOTE: When True, assumes directory already exists from previous run.
    rebuild_modified_cpp: If True, rebuild modified C++ files before simulation.
  """
  if test_name not in MODEL_REGISTRY:
    raise ValueError(f"Unknown test: {test_name}. Available tests: {list(MODEL_REGISTRY.keys())}")

  # Get model and default input pattern from registry
  model_getter, default_input_pattern = MODEL_REGISTRY[test_name]

  # Use provided pattern or fall back to default
  if input_pattern == "default":
    input_pattern = default_input_pattern

  # Determine directory name
  dir_name = f"{test_name}_evl"

  # Setup directory: only clean/create if NOT skipping setup
  if not skip_setup and not rebuild_modified_cpp:
    # Full setup: clean and recreate directory
    setup_dir(dir_name)
  else:
    # Skip setup: directory must already exist, just ensure subdirs exist
    if not os.path.exists(dir_name):
      raise FileNotFoundError(
        f"Directory '{dir_name}' does not exist. "
        f"Cannot use skip_setup=True or rebuild_modified_cpp=True without a previous run. "
        f"Run without --skip-setup and --rebuild_modified_cpp first to compile the model."
      )
    # Ensure test_inputs directory exists for new input files
    os.makedirs(os.path.join(dir_name, "test_inputs"), exist_ok=True)
    print(f"⏭️  Reusing existing directory: {dir_name}")

  # Setup log file path in the test directory's logs folder
  log_dir = os.path.join(dir_name, "logs")
  os.makedirs(log_dir, exist_ok=True)
  log_file_path = os.path.join(log_dir, f"test_{input_pattern}.log")

  # Wrap the entire test execution in the tee logger
  with tee_output_to_log(log_file_path):
    print(f"Logging test output to: {log_file_path}")
    print(f"Test: {test_name}, Input pattern: {input_pattern}, Skip setup: {skip_setup}, Rebuild modified C++: {rebuild_modified_cpp}")

    # Get original model (needed for input generation)
    # This is lightweight compared to transformation/codegen
    mod, param_dict = model_getter()

    # Generate and save test inputs
    input_dir = f"./{dir_name}/test_inputs"
    print(f"Generating test inputs for {test_name} with pattern '{input_pattern}'...")
    known_keys = param_dict.keys() if param_dict is not None else []
    gen = InputGenerator(mod=mod, known_keys=known_keys, seed=42)
    inputs = gen.generate_input(pattern=input_pattern)
    gen.save_to_files(inputs, input_dir)
    gen.save_to_files(param_dict, input_dir) # also save params

    # Load test inputs for CPU validation
    input_name = list(gen.input_info.keys())[0]
    input_data = gen.load_from_files(input_dir, input_name)
    input_dict = {input_name: input_data}

    # Run with CPU validation enabled
    # Note: When skip_setup=True, run_test will load the transformed model from file
    run_test(test_name, dir_name, mod, param_dict, input_data_dict=input_dict, skip_setup=skip_setup, rebuild_modified_cpp=rebuild_modified_cpp)


# ============================================================================
# Pytest Fixtures for Setup Caching
# ============================================================================
# Cache to track which models have been set up (for pytest session reuse)
_setup_cache = {}

@pytest.fixture(scope="session")
def setup_cache():
  """Session-scoped fixture to cache model setups across tests"""
  return _setup_cache


# ============================================================================
# Parametrized Tests
# ============================================================================
def _generate_test_parameters():
  """Generate test parameters with default pattern marked.

  This creates test IDs like:
  - one_conv-random(default) - matches both pytest -k "random" and -k "default"
  - one_conv-ones
  - one_conv-zeros
  - one_conv-linear

  This ensures each pattern is tested only once per model, with the default
  pattern explicitly marked.
  """
  test_params = []
  for model_name, (_, default_pattern) in MODEL_REGISTRY.items():
    # Create a set of unique patterns for this model
    patterns_to_test = set(INPUT_PATTERNS)

    for pattern in patterns_to_test:
      # Mark the pattern as default if it matches the registry default
      is_default = (pattern == default_pattern)
      test_params.append((model_name, pattern, is_default))

  return test_params

@pytest.mark.parametrize("test_name,input_pattern,is_default",
  _generate_test_parameters(),
  ids=[f"{params[0]}-{params[1]}{'(default)' if params[2] else ''}"
       for params in _generate_test_parameters()]
)
def test_imcflow_model_with_pattern(test_name, input_pattern, is_default, setup_cache):
  """Parametrized test for IMCFLOW models with all input patterns

  Uses setup caching: first pattern does full setup, subsequent patterns skip setup.
  The default pattern for each model is marked with (default) in the test ID,
  allowing both 'pytest -k default' and 'pytest -k <pattern>' to work correctly.
  """
  # Check if this model has already been set up in this test session
  skip_setup = test_name in setup_cache

  if not skip_setup:
    print(f"\n🔧 First run for {test_name}: Running full setup")
    setup_cache[test_name] = True
  else:
    print(f"\n⚡ Reusing compiled model for {test_name}")

  run_test_pipeline(test_name, input_pattern, skip_setup=skip_setup)


if __name__ == "__main__":
  tvm.testing.main()
