# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import pytest
import itertools
import numpy as np
import sys
import subprocess
import math
import collections
import os

from tvm.relay.backend import te_compiler
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.op.contrib import imcflow
import tvm.testing
from tvm.contrib import utils, graph_executor
from tvm import runtime as tvm_runtime

has_imcflow_codegen = pytest.mark.skipif(
    not tvm.get_global_func("relay.ext.imcflow", True), reason="IMCFLOW codegen not available"
)

run_module = tvm.testing.parameter(
    pytest.param(False, marks=[has_imcflow_codegen, *tvm.testing.requires_llvm.marks()]),
    ids=["compile"],
)

def partition_for_imcflow(mod, params=None, alter_layout=True, prune_subgraphs=True):
    """Partition the graph greedily offloading supported operators to IMCFLOW.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.
    Returns
    -------
    mod : Module
        Annotated and partitioned module.
    """
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    with TempOpAttr("nn.conv2d", "FTVMLegalize", imcflow.legalize_group_conv):
        with TempOpAttr("nn.conv2d_transpose", "FTVMLegalize", imcflow.legalize_group_conv):
            seq = tvm.transform.Sequential(
                [
                    transform.CanonicalizeOps(),
                    transform.InferType(),
                    transform.SimplifyInference(),
                    transform.FoldConstant(),
                    transform.FoldScaleAxis(),
                    # fold consecutive add ops to simplify pattern `conv2d-bias_add-bn-relu`
                    transform.SimplifyExpr(),
                    transform.FoldConstant(),
                    # alter group conv /conv_transpose layout to `GOIHW` / `GIOHW`
                    transform.Legalize(),
                    transform.FoldConstant(),
                ]
            )
            with tvm.transform.PassContext(opt_level=3):
                mod = seq(mod)
    if alter_layout:
        with TempOpAttr("nn.conv1d", "FTVMAlterOpLayout", imcflow.alter_conv):
            with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", imcflow.alter_conv):
                with TempOpAttr("nn.conv3d", "FTVMAlterOpLayout", imcflow.alter_conv):
                    with TempOpAttr(
                        "nn.conv2d_transpose", "FTVMAlterOpLayout", imcflow.alter_conv_transpose
                    ):
                        with TempOpAttr(
                            "nn.conv3d_transpose", "FTVMAlterOpLayout", imcflow.alter_conv_transpose
                        ):
                            alter_layout_seq = tvm.transform.Sequential(
                                [
                                    transform.AlterOpLayout(),
                                    transform.FoldConstant(),
                                ]
                            )
                            with tvm.transform.PassContext(opt_level=3):
                                mod = alter_layout_seq(mod)

    mod = imcflow.rewrite_layer_norm(mod)
    mod = imcflow.rewrite_dense_bias_gelu_reshape_last(mod)
    mod = imcflow.legalize_qnn_for_imcflow(mod)

    byoc_seq = tvm.transform.Sequential(
        [
            transform.MergeComposite(imcflow.pattern_table()),
            transform.AnnotateTarget("imcflow"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )

    with tvm.transform.PassContext(opt_level=3):
        mod = byoc_seq(mod)
        if prune_subgraphs:
            mod = imcflow.prune_imcflow_subgraphs(mod)
    return mod


def vmobj_to_list(o):
    if isinstance(o, tvm.nd.NDArray):
        o_np = o.numpy()
        if o_np.dtype == np.uint16:
            o_np = np.left_shift(o_np.astype("uint32"), 16).view("<f4")
        return [o_np]
    elif isinstance(o, tvm.runtime.container.ADT) or isinstance(o, list):
        return [vmobj_to_list(f) for f in o]
    else:
        raise RuntimeError("Unknown object type: %s" % type(o))


def assert_result_dict_holds(result_dict):
    for k1, k2 in itertools.combinations(result_dict, 2):
        res1 = vmobj_to_list(result_dict[k1])
        res2 = vmobj_to_list(result_dict[k2])
        for r1, r2 in zip(res1, res2):
            # ignore the accuracy checking if only one bf16 result presents
            if ("bf16" in k1) == ("bf16" in k2):
                tvm.testing.assert_allclose(r1, r2, rtol=1e-3, atol=1e-3)


def check_imcflow_used(mod, subgraph_num=None):
    num_imcflow_subgraphs = sum([1 if "imcflow" in gv.name_hint else 0 for gv in mod.get_global_vars()])
    if subgraph_num:
        assert num_imcflow_subgraphs == subgraph_num
    else:
        assert num_imcflow_subgraphs >= 1


def update_lib(lib, output_dir='./output', lib_name='lib.so'):
    test_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    source_dir = os.path.join(test_dir, "..")
    contrib_path = os.path.join(source_dir, "src", "runtime", "contrib")

    # Use a wrapper around the standard C compiler to save the C source
    kwargs = {}
    kwargs["options"] = ["-O2", "-std=c++17", "-I" + contrib_path, "-I/root/anaconda3/envs/py3.10/include"]

    # setup output directory
    if output_dir is None:
        output_dir = utils.tempdir()
    os.makedirs(output_dir, exist_ok=True)
    lib_path = os.path.join(output_dir, lib_name)

    # Export library and intermediate C source code
    print(f"Exporting library to {lib_path} with intermediate source code")
    lib.export_library(lib_path, workspace_dir=output_dir, **kwargs)
    lib = tvm_runtime.load_module(lib_path)

    return lib


def run_and_verify(mod, input, params, target, run_module, subgraph_num=None, test_bf16=True):
    dev = tvm.cpu()
    result_dict = dict()
    for mode in ["graph"]:
        configs = [
            (True, False, False),
        ]

        for use_imcflow, alter_layout, use_bf16 in configs:
            result_key = (
                mode
                + ("_imcflow" if use_imcflow else "")
                + ("_layout" if alter_layout else "")
                + ("_bf16" if use_bf16 else "_fp32")
            )
            print(f"Running {result_key}...")
            processed_mod = mod
            if use_bf16:
                processed_mod = relay.transform.ToMixedPrecision("bfloat16")(processed_mod)
                if tvm.ir.structural_equal(processed_mod, mod):
                    print("can not convert to bfloat16, skipping...")
                    continue
            if use_imcflow:
                processed_mod = partition_for_imcflow(processed_mod, params, alter_layout)
                print(processed_mod)
                check_imcflow_used(processed_mod)
            with tvm.transform.PassContext(opt_level=3):
                te_compiler.get().clear()
                lib = relay.build(processed_mod, target=target, params=params)
                lib = update_lib(lib)
                gmod = graph_executor.GraphModule(lib["default"](dev))

            if run_module:
                if isinstance(input, dict):
                    gmod.set_input(**input)
                    gmod.set_input(**params)
                    gmod.run()
                    result_dict[result_key] = gmod.get_output(0)
                    print(result_dict[result_key])
                else:
                    raise RuntimeError("input should be a dict")

    if run_module:
        assert_result_dict_holds(result_dict)


def run_and_verify_func(
    config, run_module, subgraph_num=None, target="llvm", dtype="float32", test_bf16=True
):
    """Test a Relay func by compiling, running, and comparing TVM and IMCFLOW outputs.
    Parameters
    ----------
    config : Tuple[relay.Function, Dict[str, NDArray], List[str]]
        A tuple containing 1) The function to test, 2) A dictionary of var names to input shapes and
        3) A list of which vars should be considered params.
    run_module: bool
        If True, the built module will be run after being compiled.
    """
    f, input_shapes, is_param = config
    params = {x: np.random.uniform(-1, 1, input_shapes[x]).astype(dtype) for x in is_param}
    input_dict = {
        k: np.random.uniform(-1, 1, v).astype(dtype)
        for k, v in input_shapes.items()
        if k not in is_param
    }
    run_and_verify(
        f,
        input_dict,
        params,
        subgraph_num=subgraph_num,
        target=target,
        run_module=run_module,
        test_bf16=test_bf16,
    )


def add_activation(activation, out, dic, param_lst):
    if activation == "relu":
        return relay.nn.relu(out), dic, param_lst
    if activation is None:
        return out, dic, param_lst
    else:
        raise NotImplementedError(f"Activation {activation} is not supported.")


def get_activation(x_shape=(1, 32, 8, 8), activation=None, dtype="float32"):
    x = relay.var("x", shape=(x_shape), dtype=dtype)
    dic = {"x": x_shape}
    param_lst = []
    return add_activation(activation, x, dic, param_lst)


def get_conv2d(
    x_shape=(1, 32, 8, 8),
    k_shape=(16, 32, 3, 3),
    groups=1,
    padding=(0, 0),
    strides=(1, 1),
    dilation=(1, 1),
    activation=None,
    dtype="float32",
):
    x = relay.var("x", shape=(x_shape), dtype=dtype)
    kernel = relay.var("kernel", shape=(k_shape), dtype=dtype)
    out = relay.nn.conv2d(
        x,
        kernel,
        kernel_size=k_shape[2:4],
        groups=groups,
        padding=padding,
        strides=strides,
        dilation=dilation,
        channels=k_shape[0],
    )
    dic = {"x": x_shape, "kernel": k_shape}
    param_lst = ["kernel"]
    return add_activation(activation, out, dic, param_lst)

def test_my_relu(run_module, dtype="float32"):
    relu, dic, param_lst = get_activation(
        x_shape=(1, 32, 8, 8),
        activation="relu",
        dtype=dtype,
    )
    relu = tvm.IRModule.from_expr(relu)
    config = relu, dic, param_lst
    run_and_verify_func(config, run_module=run_module, dtype=dtype)

def test_my_conv2d(run_module, dtype="float32"):
    conv2d, dic, param_lst = get_conv2d(
        x_shape=(1, 32, 8, 8),
        k_shape=(32, 32, 3, 3),
        groups=1,
        padding=(2, 2),
        strides=(1, 1),
        dilation=(2, 2),
        dtype=dtype,
    )
    conv2d = tvm.IRModule.from_expr(conv2d)
    config = conv2d, dic, param_lst
    run_and_verify_func(config, run_module=run_module, dtype=dtype)

def test_my_conv2d_relu(run_module, dtype="float32"):
    conv2d, dic, param_lst = get_conv2d(
        x_shape=(1, 32, 8, 8),
        k_shape=(32, 32, 3, 3),
        groups=1,
        padding=(2, 2),
        strides=(1, 1),
        dilation=(2, 2),
        activation="relu",
        dtype=dtype,
    )
    conv2d = tvm.IRModule.from_expr(conv2d)
    config = conv2d, dic, param_lst
    run_and_verify_func(config, run_module=run_module, dtype=dtype)

def test_prune_imcflow_subgraph(run_module):
    """In this test, OP "add" should be offloaded from imcflow codegen."""

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

    run_and_verify_func(get_graph(), subgraph_num=1, run_module=run_module, test_bf16=False)


if __name__ == "__main__":
    tvm.testing.main()
