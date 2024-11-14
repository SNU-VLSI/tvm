import pytest
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

def build(runtime="cpp", executor="graph", system_lib=False):
    dtype="float32"
    f, input_shapes, is_param = get_graph()
    params = {x: np.random.uniform(-1, 1, input_shapes[x]).astype(dtype) for x in is_param}
    input_dict = {
        k: np.random.uniform(-1, 1, v).astype(dtype)
        for k, v in input_shapes.items()
        if k not in is_param
    }

    dev = tvm.cpu()
    result_key = f"{runtime}_{executor}" + ("_system-lib" if system_lib else "")
    print(f"Running {result_key}...")

    Executor_ = Executor(executor)
    Runtime_  = Runtime(runtime, {"system-lib" : system_lib}) 

    # build
    # pretty_print(f)
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
      te_compiler.get().clear()
      # mod = relay.build(f, target="llvm", params=params, executor=Executor_, runtime=Runtime_)
      mod = relay.build(f, target="c", params=params, executor=Executor_, runtime=Runtime_)
    # pretty_print(mod)

    # export
    # test_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    # source_dir = os.path.join(test_dir, "..")
    # contrib_path = os.path.join(source_dir, "src", "runtime", "contrib")

    # kwargs = {}
    # kwargs["options"] = ["-O2", "-std=c++17", "-I" + contrib_path, "-I/root/anaconda3/envs/py3.10/include"]

    lib_name = "lib.so"
    # lib_name = "lib.tar"
    os.makedirs(result_key, exist_ok=True)
    lib_path = os.path.join(result_key, lib_name)
    # lib.export_library(lib_path, fcompile=False, **kwargs, workspace_dir=output_dir)
    mod.export_library(lib_path)

    return mod

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-r", type=str, default="cpp")
  parser.add_argument("-e", type=str, default="graph")
  parser.add_argument("-s", type=bool, default=False)
  args = parser.parse_args()

  mod = build(args.r, args.e, args.s)