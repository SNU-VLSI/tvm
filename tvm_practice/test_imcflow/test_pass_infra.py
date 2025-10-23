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

np.random.seed(0)

def get_graph():
    x1 = relay.var("x1", shape=(1, 32, 56, 56))
    x2 = relay.var("x2", shape=(1, 32, 56, 56))
    bias = relay.var("bias", shape=(32,))
    weight = relay.var("weight", shape=(32, 32, 3, 3))
    y = relay.nn.conv2d(
        x1,
        weight,
        channels=32,
        kernel_size=(3, 3),
        padding=(1, 1),
    )
    y = relay.nn.bias_add(y, bias)
    y = relay.nn.relu(y)
    y = relay.nn.global_max_pool2d(y)
    y = relay.add(y, x2)
    dic = {
        "x1": (1, 32, 56, 56),
        "x2": (1, 32, 56, 56),
        "weight": (32, 32, 3, 3),
        "bias": (32,),
    }
    param_lst = ["weight", "bias"]
    out = tvm.IRModule.from_expr(y)
    return out, dic, param_lst

if __name__ == "__main__":

  irmod, dic, param_lst = get_graph()

  CustomPass = imcflow_transform.CustomPipeline("test")
  CustomPass(irmod)