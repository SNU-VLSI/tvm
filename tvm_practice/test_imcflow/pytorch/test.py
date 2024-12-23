import torch
import tvm
from typing import Sequence
from tvm.relay.qnn.op.qnn import imcflow_min_max_quantize, imcflow_nu_quantize
import numpy as np
import tvm.relay.op as _op
import tvm.relay.expr as _expr
from tvm.relay import pretty_print
from tvm.contrib.relay_viz import RelayVisualizer, DotPlotter, DotVizParser

class Quant(torch.nn.Module):
    def __init__(self):
      super(Quant, self).__init__()
    
    def forward(self, x):
        return x * 0.5

@torch.library.custom_op("imcflow::min_max_quant", mutates_args=())
def min_max_quant(pic: torch.Tensor, min:int, max:int) -> torch.Tensor:
    return pic.clone()

def linear(inputs, input_types):
    # https://pytorch.org/docs/stable/nn.functional.html#linear
    return _op.add(inputs[0], inputs[0])

def make_min_max(input, input_types):
  MinNDArray = tvm.runtime.ndarray.array(np.array(input[1], dtype=np.float32))
  MaxNDArray = tvm.runtime.ndarray.array(np.array(input[2], dtype=np.float32))
  return imcflow_min_max_quantize(input[0], tvm.relay.Constant(MinNDArray), tvm.relay.Constant(MaxNDArray), 1, "float32")

class TestNetwork(torch.nn.Module):
    def __init__(self):
      super(TestNetwork, self).__init__()
      self.conv1_1 = torch.nn.Conv2d(28, 64, 3, 1, 1)
      self.conv1_2 = torch.nn.Conv2d(28, 64, 3, 1, 1)
      self.conv1_3 = torch.nn.Conv2d(8, 64, 3, 1, 1)
      self.conv1_4 = torch.nn.Conv2d(28, 64, 3, 1, 1)
      self.conv1_5 = torch.nn.Conv2d(28, 64, 3, 1, 1)
      self.conv1_6 = torch.nn.Conv2d(8, 64, 3, 1, 1)
      self.bn1 = torch.nn.BatchNorm2d(64)
      self.bn2 = torch.nn.BatchNorm2d(64)
      self.relu = torch.nn.ReLU()
      self.quant = min_max_quant

    def forward(self, x):
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
      y = torch.concat([y1, y2], dim=0)
      return y

def printModel(mod, param_dict, mod_name, test_name):
  RelayVisualizer(
    relay_mod = mod,
    relay_param = param_dict,
    plotter = DotPlotter(),
    parser = DotVizParser(),
  ).render(f"{test_name}/{mod_name}")

  with open(f"{test_name}/{mod_name}.txt", "w") as f:
    f.write(pretty_print(mod))

if __name__ == "__main__":
  TestNetwork_ = TestNetwork()
  x = torch.rand((1, 64, 28, 28))
  traced_model = torch.jit.trace(TestNetwork_, x)
  print(traced_model.graph)

  mod, params = tvm.relay.frontend.from_pytorch(traced_model, [("input0", x.shape)], custom_convert_map={"imcflow::min_max_quant": make_min_max})
  print(mod)
  printModel(mod, params, "test", "test")