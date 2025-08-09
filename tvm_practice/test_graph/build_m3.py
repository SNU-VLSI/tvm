import numpy as np
import os
import argparse

import tvm
from tvm import relay
from tvm.relay.backend import te_compiler
from tvm.relay.backend import Executor, Runtime
from tvm import runtime as tvm_runtime

# microTVM export
from tvm.micro import export_model_library_format

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

    input_shapes = {
        "x1": (1, 32, 56, 56),
        "x2": (1, 32, 56, 56),
        "weight": (32, 32, 3, 3),
        "bias": (32,),
    }
    param_names = ["weight", "bias"]
    ir_module = tvm.IRModule.from_expr(y)
    return ir_module, input_shapes, param_names


def build_m3(executor_name: str = "aot", system_lib: bool = True):
    dtype = "float32"
    ir_module, input_shapes, param_names = get_graph()

    params = {
        name: np.random.uniform(-1, 1, input_shapes[name]).astype(dtype)
        for name in param_names
    }
    # Inputs (not used here but kept for symmetry / future tests)
    _inputs = {
        name: np.random.uniform(-1, 1, shape).astype(dtype)
        for name, shape in input_shapes.items()
        if name not in param_names
    }

    executor_cfg = Executor(executor_name)
    runtime_cfg = Runtime("crt", {"system-lib": system_lib})

    # Build for C target with CRT runtime (suitable for microTVM / bare metal)
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        te_compiler.get().clear()
        built = relay.build(
            ir_module,
            target="c",
            params=params,
            executor=executor_cfg,
            runtime=runtime_cfg,
        )

    # Export microTVM Model Library Format (MLF) tarball to the current directory
    script_dir = os.path.dirname(os.path.realpath(__file__))
    tar_name = f"lib_m3_{executor_name}" + ("_system-lib" if system_lib else "") + ".tar"
    tar_path = os.path.join(script_dir, tar_name)
    export_model_library_format(built, tar_path)
    print(f"Exported MLF to: {tar_path}")

    return built, tar_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--executor", type=str, default="aot", choices=["aot", "graph"])
    parser.add_argument("-s", "--system-lib", action="store_true", default=True)
    args = parser.parse_args()

    build_m3(args.executor, args.system_lib)