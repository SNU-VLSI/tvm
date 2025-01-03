import torch
import tvm
from typing import Sequence
from tvm.relay.qnn.op.qnn import imcflow_min_max_quantize, imcflow_nu_quantize
import numpy as np
import tvm.relay.op as _op
import tvm.relay.expr as _expr
from tvm.relay import pretty_print
from tvm.contrib.relay_viz import RelayVisualizer, DotPlotter, DotVizParser
from tvm.relay.build_module import bind_params_by_name
from tvm.relay import transform
from tvm.relay.backend.contrib.imcflow import transform as imcflow_transform
from tvm.relay.op.contrib import imcflow
from tvm.contrib.imcflow import ImcflowDeviceConfig

@torch.library.custom_op("imcflow::min_max_quant", mutates_args=())
def min_max_quant(pic: torch.Tensor, min:int, max:int) -> torch.Tensor:
    return pic.clone()

@torch.library.custom_op("imcflow::linear_quant", mutates_args=())
def linear_quant(x:torch.Tensor, scale:float, zero_point:int) -> torch.Tensor:
  return x.clone()

def make_min_max(input, input_types):
  MinNDArray = tvm.runtime.ndarray.array(np.array(input[1], dtype=np.float32))
  MaxNDArray = tvm.runtime.ndarray.array(np.array(input[2], dtype=np.float32))
  return imcflow_min_max_quantize(input[0], tvm.relay.Constant(MinNDArray), tvm.relay.Constant(MaxNDArray), 1, "float32")

def make_quantize(input, input_types):
  scale = tvm.relay.Constant(tvm.runtime.ndarray.array(np.array(input[1], dtype=np.float32)))
  bias = tvm.relay.Constant(tvm.runtime.ndarray.array(np.array(input[2], dtype=np.int32)))
  return tvm.relay.qnn.op.quantize(input[0], scale, bias, 1, "float32")

class TestNetwork(torch.nn.Module):
    def __init__(self):
      super(TestNetwork, self).__init__()
      self.conv = torch.nn.Conv2d(64, 64, 3, 1, 1)
      self.conv1_1 = torch.nn.Conv2d(28, 64, 3, 1, 1)
      self.conv1_2 = torch.nn.Conv2d(28, 64, 3, 1, 1)
      self.conv1_3 = torch.nn.Conv2d(8, 64, 3, 1, 1)
      self.conv1_4 = torch.nn.Conv2d(28, 64, 3, 1, 1)
      self.conv1_5 = torch.nn.Conv2d(28, 64, 3, 1, 1)
      self.conv1_6 = torch.nn.Conv2d(8, 64, 3, 1, 1)
      self.bn1 = torch.nn.BatchNorm2d(64)
      self.bn2 = torch.nn.BatchNorm2d(64)
      self.conv2_1 = torch.nn.Conv2d(28, 64, 3, 1, 1)
      self.conv2_2 = torch.nn.Conv2d(28, 64, 3, 1, 1)
      self.conv2_3 = torch.nn.Conv2d(28, 64, 3, 1, 1)
      self.conv2_4 = torch.nn.Conv2d(28, 64, 3, 1, 1)
      self.conv2_5 = torch.nn.Conv2d(16, 64, 3, 1, 1)

      self.conv2_6 = torch.nn.Conv2d(28, 64, 3, 1, 1)
      self.conv2_7 = torch.nn.Conv2d(28, 64, 3, 1, 1)
      self.conv2_8 = torch.nn.Conv2d(28, 64, 3, 1, 1)
      self.conv2_9 = torch.nn.Conv2d(28, 64, 3, 1, 1)
      self.conv2_10 = torch.nn.Conv2d(16, 64, 3, 1, 1)
      self.bn3 = torch.nn.BatchNorm2d(64)
      self.bn4 = torch.nn.BatchNorm2d(64)
      self.relu = torch.nn.ReLU()
      self.quant = min_max_quant

    def forward(self, x):
      x = self.conv(x)
      x = x/2.0
      x1, x2, x3 = torch.split(x, [28, 28, 8], dim=1)
      y1 = (self.conv1_1(x1) + self.conv1_2(x2))/2 + self.conv1_3(x3)/2
      y1 = self.bn1(y1)
      y1 = self.relu(y1)
      y1 = self.quant(y1, 0, 1)

      y2 = (self.conv1_4(x1) + self.conv1_5(x2))/2 + self.conv1_6(x3)/2
      y2 = self.bn2(y2)
      y2 = self.relu(y2)
      y2 = self.quant(y2, 0, 1)
      # y1 = self.conv1_1(x[:, 0:28, :, :]) + self.conv1_2(x[:, 28:56, :, :]) + self.conv1_3(x[:, 56:60, :, :])
      # y2 = self.conv1_4(x[:, 0:28, :, :]) + self.conv1_5(x[:, 28:56, :, :]) + self.conv1_6(x[:, 56:60, :, :])
      y = torch.concat([y1, y2], dim=1)

      y1, y2, y3, y4, y5 = torch.split(y, [28, 28, 28, 28, 16], dim=1)
      y_1 = (self.conv2_1(y1) + self.conv2_2(y2))/2 + (self.conv2_3(y3) + self.conv2_4(y4))/2 + self.conv2_5(y5)/2
      y_1 = self.bn3(y_1)
      y_1 = self.relu(y_1)
      y_1 = self.quant(y_1, 0, 1)

      y_2 = (self.conv2_6(y1) + self.conv2_7(y2))/2 + (self.conv2_8(y3) + self.conv2_9(y4))/2 + self.conv2_10(y5)/2
      y_2 = self.bn4(y_2)
      y_2 = self.relu(y_2)
      y_2 = self.quant(y_2, 0, 1)
      y = torch.concat([y_1, y_2], dim=1)
      return y

def printModel(mod, param_dict, mod_name):
  RelayVisualizer(
    relay_mod = mod,
    relay_param = param_dict,
    plotter = DotPlotter(),
    parser = DotVizParser(),
  ).render(f"results/{mod_name}")

  with open(f"results/{mod_name}.txt", "w") as f:
    f.write(pretty_print(mod))

if __name__ == "__main__":
  TestNetwork_ = TestNetwork()
  x = torch.rand((1, 64, 28, 28))
  traced_model = torch.jit.trace(TestNetwork_, x)
  mod, params = tvm.relay.frontend.from_pytorch(
    traced_model,
    [("input0", x.shape)],
    custom_convert_map={
      "imcflow::min_max_quant": make_min_max,
      "imcflow::linear_quant" : make_quantize})
  eval_mod, eval_param_dict = mod, params

  # origin
  printModel(eval_mod, eval_param_dict, "origin")

  # bind param
  eval_mod["main"] = bind_params_by_name(eval_mod["main"], eval_param_dict)
  eval_mod = transform.InferType()(eval_mod)
  printModel(eval_mod, eval_param_dict, "after_bind")

  # transform to QuantModel
  eval_mod = imcflow_transform.makeToQuantizedForm(eval_mod)
  printModel(eval_mod, eval_param_dict, "after_quant1")
  eval_mod = transform.InferType()(eval_mod)
  printModel(eval_mod, eval_param_dict, "after_quant2")

  # # byoc pass
  # eval_mod = transform.MergeComposite(imcflow.pattern_table())(eval_mod)
  # printModel(eval_mod, eval_param_dict, "after_merge")

  # # merge split and concat nodes
  # SplitConcatRegions = imcflow_transform.getSplitConcatDepsRegions(eval_mod["main"])
  # eval_mod = imcflow.ImcflowAnnotationPass(SplitConcatRegions)(eval_mod)
  # eval_mod = transform.MergeCompilerRegions()(eval_mod)
  # eval_mod = transform.PartitionGraph()(eval_mod)
  # printModel(eval_mod, eval_param_dict, "after_split_concat_partition")

  # # annotation
  # AnnotGenerator = imcflow_transform.AnnotGenerator()
  # AnnotGenerator(eval_mod)
  # # print(AnnotGenerator.RegionList)
  # eval_mod = imcflow.ImcflowAnnotationPass(AnnotGenerator.RegionList)(eval_mod)
  # printModel(eval_mod, eval_param_dict, "after_annot")

  # eval_mod = transform.MergeCompilerRegions()(eval_mod)
  # printModel(eval_mod, eval_param_dict, "after_merge_region")

  # eval_mod = imcflow.ImcflowCleanRegionTag()(eval_mod)
  # printModel(eval_mod, eval_param_dict, "after_clean_region")

  # eval_mod = transform.PartitionGraph()(eval_mod)
  # printModel(eval_mod, eval_param_dict, "after_partition_graph")

  # eval_mod = imcflow.flattenSubgraphs(eval_mod)
  # printModel(eval_mod, eval_param_dict, "after_flatten")

  # eval_mod = imcflow.prune_imcflow_subgraphs(eval_mod)
  # imcflow_transform.constructUsefulMappings(eval_mod)
  # imcflow_transform.constructCustomIDInFunc(eval_mod)
  # printModel(eval_mod, eval_param_dict, "after_prune_model")

  # imcflow_transform.NodeMapper()(eval_mod)
  # imcflow_transform.constructTensorEdgeList(eval_mod)
  # imcflow_transform.constructActiveIMCEDict(eval_mod)

  # print("Active IMCE list")
  # print(ImcflowDeviceConfig().ActiveIMCEPerFunc)

  # print("HW MAP")
  # print(ImcflowDeviceConfig().HWNodeMap)

  # print("CustomID TO Name")
  # print(imcflow.CustomIDToName())

  # print("Tensor Edge List")
  # for key, paths in ImcflowDeviceConfig().TensorEdgeListDict.items():
  #   print(key)
  #   for path in paths:
  #     print(path)
  
  # imcflow_transform.constructTensorIDToTensorEdgeDict()
  # print("Tensor ID to Tensor Edge")
  # for key, paths in ImcflowDeviceConfig().TensorIDtoEdge.items():
  #   print(f"{key} : {paths}")
  
  # imcflow_transform.constructNoCPathDict(eval_mod)
  # print("NoC Paths")
  # for key, paths in ImcflowDeviceConfig().NoCPaths.items():
  #   print(key)
  #   for k, v in paths.items():
  #     print(k, v)

  # MemoryCalculator = imcflow_transform.MemoryCalculator()(eval_mod)
  # PolicyTableGenerator = imcflow_transform.PolicyTableGenerator(ImcflowDeviceConfig().NoCPaths)(eval_mod)