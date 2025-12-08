import tvm
import numpy as np
import pathlib
from tvm.micro import export_model_library_format
from tvm.micro.testing import get_target
from tvm.contrib.utils import tempdir
import tvm.testing
from tvm.relay import pretty_print
from tvm.contrib.relay_viz import RelayVisualizer, DotPlotter, DotVizParser
from tvm.contrib import graph_executor
from tvm.relay.build_module import bind_params_by_name
from tvm.relay import transform
from tvm.relay.backend.contrib.imcflow import transform as imcflow_transform
from tvm.relay.backend.contrib.imcflow import codegen as imcflow_codegen
from tvm.relay.op.contrib import imcflow
from tvm.contrib.imcflow import ImcflowDeviceConfig as DevConfig
from tvm.relay.backend import Executor, Runtime
from tvm.contrib.imcflow import DataBlock
import os
import shutil
from tvm.relay.op.transform import imcflow_4d_to_qconv_input, imcflow_mmquant_out_to_4d
import tvm.relay as relay
import pprint

from tvm.relay.op.contrib.imcflow import HashToCustomID, CustomIDToName, CustomIDInFunc, CustomIDToNode
from models import real_model, real_model2, test_models
from models import small_model
from models import resnet8_cifar, mobilenet_imcflow, deep_autoencoder_imcflow, ds_cnn_imcflow
from models import models_for_test

def setup_dir(dir_name):
  def clean_dir_recursive(path):
    """Recursively clean all files but keep all directory inodes intact"""
    for item in os.listdir(path):
      item_path = os.path.join(path, item)
      if os.path.isfile(item_path) or os.path.islink(item_path):
        os.remove(item_path)
      elif os.path.isdir(item_path):
        # Recursively clean subdirectory but keep the directory itself
        clean_dir_recursive(item_path)

  if not os.path.exists(dir_name):
    os.makedirs(dir_name)
  else:
    # clean up all files recursively but keep all directory structures intact
    clean_dir_recursive(dir_name)


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

def run_cpu_reference(mod, param_dict, dir_name):
  with tvm.transform.PassContext(opt_level=0, config={"tir.disable_vectorize": True}):
    graph, lib, params = tvm.relay.build(mod, target="llvm")
  mod = graph_executor.create(graph, lib, tvm.cpu(0))
  mod.set_input(**params)
  mod.run()


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

def transform_model_for_imcflow(mod, param_dict, dir, test_name):
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

  CodegenSuite = imcflow_codegen.CodegenSuite(dir, mod, host_isa=DevConfig().HOST_ISA)
  CodegenSuite(mod)

  print(f"mem_layout: {config.MemLayout}")
  print(f"Evaluation generation completed for {test_name}")

  imcflow_transform.constructDataBlockDict(mod)
  print(f"data_blocks: {config.DataBlocks}")
  return mod, param_dict


def run_test_evl(test_name, mod, param_dict):
  """Generate IMCFLOW evaluation results (original function renamed)"""
  print(f"\n{'='*60}")
  print(f"GENERATING EVALUATION RESULTS FOR: {test_name}")
  print(f"{'='*60}")

  eval_dir = f"{test_name}_evl"
  setup_dir(eval_dir)

  # Transform the model for IMCFLOW
  mod, param_dict = transform_model_for_imcflow(mod, param_dict, eval_dir, test_name)

  # Generate graph executor for the transformed model
  generate_graph_executor(mod, param_dict, eval_dir)


def run_test_ref(test_name, mod, param_dict):
  """Generate reference TVM compilation results"""
  print(f"\n{'='*60}")
  print(f"GENERATING REFERENCE RESULTS FOR: {test_name}")
  print(f"{'='*60}")

  ref_dir = f"{test_name}_ref"
  setup_dir(ref_dir)

  # Transform the model for IMCFLOW
  mod, param_dict = transform_model_for_imcflow(mod, param_dict, ref_dir, test_name)

  run_cpu_reference(mod, param_dict, ref_dir)

  print(f"Reference generation completed for {test_name}")
  return mod


def test_big_ref():
  """Generate only reference for big model"""
  assert False, "Big model reference is not supported yet"


def test_small_ref():
  """Generate only reference for small model"""
  mod, param_dict, _ = small_model.getTestModel()
  run_test_ref("small", mod, param_dict)


def test_big_evl():
  """Generate only evaluation for big model"""
  mod, param_dict = real_model.getModel()
  run_test_evl("big", mod, param_dict)


def test_small_evl():
  """Generate only evaluation for small model"""
  mod, param_dict = real_model2.getModel()
  run_test_evl("small", mod, param_dict)

def test_one_conv_quant_evl():
  mod, param_dict = real_model2.getOneConvQuantModel()
  run_test_evl("one_conv_quant", mod, param_dict)

def test_one_relu_evl():
  """Generate evaluation for relu model"""
  mod, param_dict = models_for_test.getOneReluModel()
  run_test_evl("one_relu", mod, param_dict)

def test_one_conv_ref():
  """Generate reference for conv model"""
  mod, param_dict = models_for_test.getOneConvModel()
  run_test_ref("one_conv", mod, param_dict)

def test_one_conv_evl():
  """Generate evaluation for conv model"""
  mod, param_dict = models_for_test.getOneConvModel()
  run_test_evl("one_conv", mod, param_dict)

def test_model_v2():
  """Generate evaluation for relu model"""
  mod, param_dict = real_model2.getModelV2()
  run_test_evl("model_v2", mod, param_dict)

def test_model_1():
  """Generate evaluation for model 1"""
  mod, param_dict = test_models.get_model1()
  run_test_evl("model_1", mod, param_dict)

def test_resnet8():
  mod, param_dict = resnet8_cifar.getModel(True)
  run_test_evl("resnet8", mod, param_dict)

def test_resnet8_from_pretrained():
  mod, param_dict = resnet8_cifar.getModel_from_pretrained_weight(True)
  run_test_evl("resnet8", mod, param_dict)

def test_mobilenet_imcflow():
  mod, param_dict = mobilenet_imcflow.getModel(False)
  run_test_evl("mobilenet_imcflow", mod, param_dict)

def test_deep_autoencoder_imcflow():
  mod, param_dict = deep_autoencoder_imcflow.getModel(False)
  run_test_evl("deep_autoencoder_imcflow", mod, param_dict)

def test_ds_cnn_imcflow():
  mod, param_dict = ds_cnn_imcflow.getModel(False)
  run_test_evl("ds_cnn_imcflow", mod, param_dict)

def test_residual_model():
  """Generate evaluation for residual model"""
  mod, param_dict = models_for_test.getResidualModel()
  run_test_evl("residual_model", mod, param_dict)

def test_resnet_cifar10_small_pretrained():
  """Generate evaluation for resnet cifar10 small model"""
  mod, param_dict = models_for_test.getResnetCifar10SmallPretrained(True)
  run_test_evl("resnet_cifar10_small", mod, param_dict)

if __name__ == "__main__":
  tvm.testing.main()
  # test_resnet8()
