import itertools
import numpy as np
import sys
import subprocess
import math
import collections
import os
import argparse

from tvm.relay.backend import te_compiler
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.op.contrib import dnnl
import tvm.testing
from tvm.contrib import utils
from tvm import runtime as tvm_runtime
from tvm.contrib import graph_executor

from tvm.relay.backend import Executor, Runtime
from tvm.relay import pretty_print
from tvm.relay.backend.contrib.imcflow import transform as imcflow_transform

from models import *

np.random.seed(0)

def get_graph(IC, IH, IW, OC, KH, KW):
    x1 = relay.var("x1", shape=(1, IC, IH, IW))
    bias = relay.var("bias", shape=(OC,))
    weight = relay.var("weight", shape=(OC, IC, KH, KW))
    y = relay.nn.conv2d(
        x1,
        weight,
        channels=OC,
        kernel_size=(KH, KW),
        padding=(1, 1),
    )
    y = relay.nn.bias_add(y, bias)
    y = relay.nn.relu(y)
    y = relay.nn.global_max_pool2d(y)
    dic = {
        "x1": (1, IC, IH, IW),
        "weight": (OC, IC, KH, KW),
        "bias": (OC,),
    }
    param_lst = ["weight", "bias"]
    out = tvm.IRModule.from_expr(y)
    return out, dic, param_lst

def RunTestModel(name="test"):
  from tvm.contrib.relay_viz import RelayVisualizer, DotPlotter, DotVizParser

  irmod, dic, param_list = get_graph(32, 16, 16, 64, 3, 3)
  TestName=name

  with open(f"{TestName}.txt", "w") as f:
    f.write(pretty_print(irmod))

  dot_graph = RelayVisualizer(
    relay_mod = irmod,
    relay_param = None,
    plotter = DotPlotter(),
    parser = DotVizParser(),
  ).render(f"{TestName}")

if __name__ == "__main__":
  RunTestModel()