import pytest
import tvm
import numpy as np
from tvm.micro import export_model_library_format
import tvm.testing
from tvm.contrib.relay_viz import RelayVisualizer, DotPlotter, DotVizParser
from tvm.contrib import graph_executor
from tvm.relay.build_module import bind_params_by_name
from tvm.relay import transform
from tvm.relay.backend.contrib.imcflow import transform as imcflow_transform
from tvm.relay.backend.contrib.imcflow import cpu_run as cpu_run
from tvm.relay.backend.contrib.imcflow import codegen as imcflow_codegen
from tvm.relay.op.contrib import imcflow
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from tvm.relay.backend import Executor, Runtime
import os
import subprocess
import copy
import pprint

from tvm.relay.op.contrib.imcflow import HashToCustomID
from models import real_model, real_model2, test_models
from models import resnet8_cifar, mobilenet_imcflow, deep_autoencoder_imcflow, ds_cnn_imcflow
from models import models_for_test

# Import shared input generator
from input_generator import InputGenerator

# ============================================================================
# Model Registry
# ============================================================================
# Maps test_name -> (model_getter_function, input_pattern)
# input_pattern: "random", "ones", "zeros", "linear"
# NOTE: models should have _evl suffix for its directory to be ignored by git.
MODEL_REGISTRY = {
    # Simple test models
    "one_relu": (models_for_test.getOneReluModel, "linear"),
    "one_conv": (models_for_test.getOneConvModel, "ones"),
    "residual_model": (models_for_test.getResidualModel, "ones"),
    "mini_imcflow": (models_for_test.getMiniImcflowModel, "ones"),

    # ResNet8 variants - all use small_debug=True
    "resnet8_small": (lambda: resnet8_cifar.getModel(True), "ones"),
    "resnet8_small_pretrained": (lambda: resnet8_cifar.getModel_from_pretrained_weight(True), "ones"),
    "resnet_cifar10_small": (lambda: models_for_test.getResnetCifar10Small(small_debug=True), "ones"),
    "resnet_cifar10_small_pretrained": (lambda: models_for_test.getResnetCifar10SmallPretrained(small_debug=True), "ones"),

    # Other models
    "mobilenet_imcflow": (lambda: mobilenet_imcflow.getModel(False), "random"),
    "deep_autoencoder_imcflow": (lambda: deep_autoencoder_imcflow.getModel(False), "random"),
    "ds_cnn_imcflow": (lambda: ds_cnn_imcflow.getModel(False), "random"),

    # Legacy models (for backward compatibility)
    "big": (real_model.getModel, "random"),
    "small": (real_model2.getModel, "random"),
    "one_conv_quant": (real_model2.getOneConvQuantModel, "ones"),
    "model_v2": (real_model2.getModelV2, "random"),
    "model_1": (test_models.get_model1, "random"),
}

# ============================================================================
# Utility Functions
# ============================================================================
def setup_dir(test_name, suffix="_evl"):
  def clean_dir_recursive(path):
    """Recursively clean all files but keep all directory inodes intact"""
    for item in os.listdir(path):
      item_path = os.path.join(path, item)
      if os.path.isfile(item_path) or os.path.islink(item_path):
        os.remove(item_path)
      elif os.path.isdir(item_path):
        # Recursively clean subdirectory but keep the directory itself
        clean_dir_recursive(item_path)

  dir_name = f"{test_name}{suffix}"
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)
  else:
    # clean up all files recursively but keep all directory structures intact
    clean_dir_recursive(dir_name)

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

def run_cpu_validation(mod, param_dict, input_data_dict, model_dir):
  """Run transformed model on CPU for validation

  Args:
    mod: The TVM relay module
    param_dict: Model parameters
    input_data_dict: Dictionary of input name -> numpy array
    model_dir: Directory to save CPU outputs

  Returns:
    output: The CPU execution output as numpy array
  """
  print("\n" + "="*40)
  print("RUNNING CPU VALIDATION")
  print("="*40)

  target = "llvm"
  ctx = tvm.cpu(0)

  cpu_mod = copy.deepcopy(mod)
  cpu_mod = cpu_run.make_cpu_runnable(cpu_mod)
  printModel(model_dir, cpu_mod, param_dict, "cpu_runnable_model")
  with tvm.transform.PassContext(opt_level=0, config={"tir.disable_vectorize": True}):
    graph, lib, params = tvm.relay.build(cpu_mod, target=target, params=param_dict)

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

  # Get output
  output = executor.get_output(0).asnumpy()

  # Save output for reference
  output_dir = os.path.abspath(os.path.join(model_dir, "test_references"))
  np.save(f"{output_dir}/cpu_reference_output.npy", output)
  print(f"CPU output saved to: {output_dir}/cpu_reference_output.npy")
  print(f"CPU output shape: {output.shape}, dtype: {output.dtype}")

  return output


def generate_graph_executor(mod, param_dict, dir_name):
  executor_cfg = Executor("graph")
  runtime_cfg = Runtime("crt", {"system-lib": True})
  print("\n" + "="*40)
  print("GENERATING GRAPH EXECUTOR")
  print("="*40)

  with tvm.transform.PassContext(opt_level=0, config={"tir.disable_vectorize": True}):
    module = tvm.relay.build(
      mod,
      target="c",
      params=param_dict,
      executor=executor_cfg,
      runtime=runtime_cfg,
    )

  script_dir = os.path.dirname(os.path.realpath(__file__))
  tar_name = f"lib_graph_system-lib.tar"
  tar_path = os.path.join(script_dir, dir_name, tar_name)
  export_model_library_format(module, tar_path)
  return module, tar_path

def transform_model_for_imcflow(mod, param_dict, dir):
  DevConfig().clear()

  # origin
  printModel(dir, mod, param_dict, "0_origin")

  # bind param
  mod["main"] = bind_params_by_name(mod["main"], param_dict)
  mod = transform.InferType()(mod)
  printModel(dir, mod, param_dict, "1_after_bind")

  # first level imcflow graph partition
  mod = imcflow_transform.partitionImcflowSubGraph(mod)
  printModel(dir, mod, param_dict, "2_after_L1_partition")

  # split imcflow function conv to atomic ops
  mod, param_dict = imcflow_transform.split_conv_to_atomic(mod, param_dict)
  printModel(dir, mod, param_dict, "2_after_atom_split")

  # merge composite OPs
  mod = imcflow_transform.merge_composite_ops(mod)
  printModel(dir, mod, param_dict, "3_after_merge")

  # make split and concat super node
  mod = imcflow_transform.makeSplitConcatDepsRegions(mod)
  printModel(dir, mod, param_dict, "4_after_split_concat_partition")

  mod = imcflow_transform.ConcatDistributor(max_inputs=4).run(mod)
  printModel(dir, mod, param_dict, "4.5_after_concat_distributor")

  mod = imcflow_transform.partitionRound(mod)
  printModel(dir, mod, param_dict, "5_after_annot")

  mod = imcflow.flattenImcflowTopFuncs(mod)
  printModel(dir, mod, param_dict, "6_after_flatten")

  mod = imcflow.prune_imcflow_subgraphs(mod)
  printModel(dir, mod, param_dict, "7_after_prune_model")

  mod = imcflow_transform.annotateCustomId(mod)
  printModel(dir, mod, param_dict, "7.5_after_annotate_custom_id")
  imcflow_transform.constructUsefulMappings(mod)
  print("-------------------- CustomID TO Name --------------------")
  with open(f"{dir}/custom_id_to_name.txt", "w") as f:
    pprint.pprint(imcflow.CustomIDToName(), stream=f)
  print("-------------------- Node TO CustomID --------------------")
  with open(f"{dir}/node_to_custom_id.txt", "w") as f:
    pprint.pprint(HashToCustomID(), stream=f)
  printModel(dir, mod, param_dict, "7.6_with_custom_id")

  mod, ttype_map = imcflow_transform.legalizeImcflowLayout(mod)
  printModel(dir, mod, param_dict, "7.7_after_mark_in_out")
  print("-------------------- Real Tensor Type Map --------------------")
  pprint.pprint(ttype_map)

  # -----------------------------------------------------------------
  # annotate custom ID for debugging
  # -----------------------------------------------------------------
  mod = imcflow_transform.annotateCustomId(mod)
  printModel(dir, mod, param_dict, "8.5_after_annotate_custom_id")

  imcflow_transform.constructUsefulMappings(mod)
  imcflow_transform.constructCustomIDInFunc(mod)
  imcflow_transform.constructImcflowFuncMap(mod)
  print("-------------------- CustomID TO Name --------------------")
  with open(f"{dir}/custom_id_to_name.txt", "w") as f:
    pprint.pprint(imcflow.CustomIDToName(), stream=f)
  print("-------------------- Node TO CustomID --------------------")
  with open(f"{dir}/node_to_custom_id.txt", "w") as f:
    pprint.pprint(HashToCustomID(), stream=f)
  print("-------------------- func map --------------------")
  with open(f"{dir}/func_map.txt", "w") as f:
    pprint.pprint(DevConfig().ImcflowFuncMap, stream=f)
  printModel(dir, mod, param_dict, "9_with_custom_id")

  imcflow_transform.NodeMapper().run(mod)
  print("------------------------------- HW MAP ----------------------------------")
  with open(f"{dir}/hw_node_map.txt", "w") as f:
    pprint.pprint(DevConfig().HWNodeMap, stream=f)

  imcflow_transform.constructTensorEdgeList(mod)
  print("------------------------------- Tensor Edge List --------------------------------------")
  with open(f"{dir}/tensor_edge_list.txt", "w") as f:
    for key, paths in DevConfig().TensorEdgeListDict.items():
      print(key, file=f)
      for path in paths:
        print(path, file=f)

  imcflow_transform.constructActiveIMCEDict(mod)
  print("------------------------------  Active IMCE list ---------------------- ")
  with open(f"{dir}/active_imce_list.txt", "w") as f:
    pprint.pprint(DevConfig().ActiveIMCEPerFunc, stream=f)

  imcflow_transform.constructTensorIDToTensorEdgeDict()
  print("Tensor ID to Tensor Edge")
  with open(f"{dir}/tensor_id_to_edge.txt", "w") as f:
    for key, paths in DevConfig().TensorIDtoEdge.items():
      print(f"{key} : {paths}", file=f)

  imcflow_transform.constructNoCPathDict(mod)
  print("NoC Paths")
  with open(f"{dir}/noc_paths.txt", "w") as f:
    for key, paths in DevConfig().NoCPaths.items():
      print(key, file=f)
      for k, v in paths.items():
        print(k, v, file=f)

  imcflow_transform.MemoryAllocator().run(mod, ttype_map)
  print("------------------------------- Memory Layout ----------------------------------")
  with open(f"{dir}/mem_layout.txt", "w") as f:
    pprint.pprint(DevConfig().MemLayout, stream=f)

  imcflow_transform.PolicyTableGenerator(DevConfig().NoCPaths).run(mod)
  with open(f"{dir}/policy_table.txt", "w") as f:
    f.write(DevConfig().format_policy_table())

  imcflow_transform.generateNoCVisualizations(mod, dir + "/noc_visualizations")

  fifo_monitor = imcflow_transform.FIFOConflictMonitor()
  fifo_monitor.run(mod)
  fifo_monitor.print_conflict_summary()
  fifo_monitor.export_conflict_table(f"{dir}/fifo_conflict_table.txt")

  deadlock_detector = imcflow_transform.NoCDeadlockDetector()
  deadlock_detector.run(mod)
  deadlock_detector.print_deadlock_summary()
  deadlock_detector.export_deadlock_table(f"{dir}/noc_deadlock_table.txt")

  # get the config
  config = DevConfig()

  def _dump(title, dict):
    with open(f"{dir}/final_imcflow_config_{title}.txt", "w") as f:
      print(f"----------------------- {title} ------------------------", file=f)
      for key, value in dict.items():
        pprint.pprint(f"{key} : {value}", stream=f)

  _dump("HWNodeMap", config.HWNodeMap)
  _dump("TensorEdgetoInfo", config.TensorEdgetoInfo)
  _dump("TensorIDtoEdge", config.TensorIDtoEdge)
  _dump("PolicyTableDict", config.PolicyTableDict)

  return mod, param_dict

def run_imcflow_codegen(mod, dir):
  """Run IMCFLOW codegen to generate hardware deployment code"""
  config = DevConfig()

  CodegenSuite = imcflow_codegen.CodegenSuite(dir, mod, host_isa=DevConfig().HOST_ISA)
  CodegenSuite(mod)
  print(f"mem_layout: {config.MemLayout}")

  imcflow_transform.constructDataBlockDict(mod)
  print(f"data_blocks: {config.DataBlocks}")


def run_simulation(eval_dir):
  """Run simulation by building and executing the graph with proper output streaming

  Args:
    eval_dir: Evaluation directory containing the model

  Returns:
    imcflow_output: The IMCFLOW simulation output as numpy array, or None if output file doesn't exist
  """
  log_dir = f"{eval_dir}/logs"

  print("\n" + "="*60)
  print("RUNNING SIMULATION")
  print("="*60)

  # Build the host binary
  print("\n--- Building Host Binary ---")
  host_build_dir = "./host_binary_make/build"
  build_command = ["direnv", "exec", ".", "../build.sh", "execute_graph.c", eval_dir, "x86"]
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
  print("\n--- Running gem5 Simulation ---")
  imcflow_gem5_dir = "/root/project/imcflow/pmap/ISA_sim/gem5/tests/imcflow"
  sim_command = ["direnv", "exec", ".", "./run.sh", "tvm_host_runner", "no", eval_dir]
  sim_log_path = os.path.join(log_dir, "gem5.log")

  with open(sim_log_path, "w") as log_file:
    process = subprocess.Popen(
      sim_command,
      cwd=imcflow_gem5_dir,
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
      raise subprocess.CalledProcessError(process.returncode, sim_command)

  print(f"✅ Simulation completed, log saved to: {sim_log_path}")

  # Load and return the simulation output
  imcflow_output_path = os.path.abspath(os.path.join(eval_dir, "test_outputs", "output.npy"))
  if os.path.exists(imcflow_output_path):
    imcflow_output = np.load(imcflow_output_path)
    print(f"✅ Loaded IMCFLOW output from: {imcflow_output_path}")
    return imcflow_output
  else:
    print(f"⚠️  IMCFLOW output not found at: {imcflow_output_path}")
    return None


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
  if cpu_output.dtype in [np.float32, np.float64]:
    if np.allclose(cpu_output, imcflow_output, rtol=1e-5, atol=1e-8):
      print("✅ IMCFLOW output matches CPU reference fp output (within tolerance)")
    else:
      pytest.fail(f"Reference output: {cpu_output}\n IMCFLOW output: {imcflow_output}")
  elif np.array_equal(cpu_output, imcflow_output):
    print("✅ IMCFLOW output matches CPU reference output (exact match)")
  else:
    pytest.fail(f"Reference output: {cpu_output}\n IMCFLOW output: {imcflow_output}")


def run_test(test_name, eval_dir, mod, param_dict, input_data_dict=None):
  """Generate IMCFLOW evaluation results with optional CPU validation

  Args:
    test_name: Name of the test
    eval_dir: evaluation directory name
    mod: The TVM relay module
    param_dict: Model parameters
    input_data_dict: Optional dict of input name -> numpy array for CPU validation
  """
  print(f"\n{'='*60}")
  print(f"GENERATING EVALUATION RESULTS FOR: {test_name}")
  print(f"{'='*60}")

  # Transform the model for IMCFLOW
  mod, param_dict = transform_model_for_imcflow(mod, param_dict, eval_dir)

  # Run CPU validation if input data is provided
  if input_data_dict is not None:
    cpu_output = run_cpu_validation(mod, param_dict, input_data_dict, eval_dir)
    if cpu_output is not None:
      print("✅ CPU validation completed successfully")

  # Run IMCFLOW codegen to generate hardware deployment code
  run_imcflow_codegen(mod, eval_dir)

  # Generate graph executor for hardware deployment
  generate_graph_executor(mod, param_dict, eval_dir)

  # Run simulation (build + gem5 execution)
  imcflow_output = run_simulation(eval_dir)

  # Compare the reference CPU output with IMCFLOW simulated output
  if input_data_dict is not None:
    if imcflow_output is None:
      pytest.fail(f"IMCFLOW output file missing, cannot compare outputs")

    compare_outputs(cpu_output, imcflow_output)


# ============================================================================
# Test Pipeline
# ============================================================================
def run_test_pipeline(test_name):
  """
  Test pipeline that:
  1. Gets the model from registry
  2. Generates and saves test inputs
  3. Loads test inputs for CPU validation
  4. Runs the full evaluation pipeline

  Args:
    test_name: Name of the test (must exist in MODEL_REGISTRY)
  """
  if test_name not in MODEL_REGISTRY:
    raise ValueError(f"Unknown test: {test_name}. Available tests: {list(MODEL_REGISTRY.keys())}")

  dir_name = setup_dir(test_name, "_evl")

  # Get model and input pattern from registry
  model_getter, input_pattern = MODEL_REGISTRY[test_name]
  mod, param_dict = model_getter()

  # Generate and save test inputs
  input_dir = f"./{dir_name}/test_inputs"
  print(f"Generating test inputs for {test_name}...")
  gen = InputGenerator(mod=mod, seed=42)
  inputs = gen.generate_input(pattern=input_pattern)
  gen.save_to_files(inputs, input_dir)

  # Load test inputs for CPU validation
  gen = InputGenerator(mod=mod)
  input_name = list(gen.input_info.keys())[0]
  input_data = InputGenerator.load_from_files(input_dir, input_name)
  input_dict = {input_name: input_data}

  # Run with CPU validation enabled
  run_test(test_name, dir_name, mod, param_dict, input_data_dict=input_dict)


# ============================================================================
# Parametrized Tests
# ============================================================================
@pytest.mark.parametrize("test_name", list(MODEL_REGISTRY.keys()))
def test_imcflow_model(test_name):
  """Parametrized test for IMCFLOW models"""
  run_test_pipeline(test_name)


if __name__ == "__main__":
  tvm.testing.main()
