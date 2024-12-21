import torch
import tvm
from typing import Sequence
from tvm.relay.qnn.op.qnn import imcflow_min_max_quantize, imcflow_nu_quantize
import numpy as np
import tvm.relay.op as _op
import tvm.relay.expr as _expr

class Quant(torch.nn.Module):
    def __init__(self):
      super(Quant, self).__init__()
    
    def forward(self, x):
        return x * 0.5

@torch.library.custom_op("imcflow::min_max_quant", mutates_args=())
def min_max_quant(pic: torch.Tensor, min:int, max:int) -> torch.Tensor:
    return pic.clone()

class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)/4
        new_h = min_max_quant(new_h, 1, 2)
        return new_h

my_cell = MyCell()
x, h = torch.rand(3, 4), torch.rand(3, 4)
traced_cell = torch.jit.trace(my_cell, (x, h))
print(traced_cell)
# print(traced_cell(x, h))
print(traced_cell.graph)
for node in traced_cell.graph.nodes():
    print(node)

def linear(inputs, input_types):
    # https://pytorch.org/docs/stable/nn.functional.html#linear
    return _op.add(inputs[0], inputs[0])

def make_min_max(input, input_types):
  MinNDArray = tvm.runtime.ndarray.array(np.array(input[1], dtype=np.float32))
  MaxNDArray = tvm.runtime.ndarray.array(np.array(input[2], dtype=np.float32))
  # return imcflow_min_max_quantize(input[0], tvm.relay.Constant(MinNDArray), tvm.relay.Constant(MaxNDArray), 1, "int4")
  return 

mod = tvm.relay.frontend.from_pytorch(traced_cell, [("input0", x.shape), ("input1", h.shape)],

                                      custom_convert_map={"imcflow::min_max_quant": linear})

print(mod)