#!/usr/bin/env python3
"""
Column-disable verification script.

Usage:
    python test_column_disable.py --model resnet8_subset06_pretrained_orig --phase compile_only
    python test_column_disable.py --model resnet8_subset06_pretrained_orig --phase cpu_validate
    IMCFLOW_RUNNER=py python test_column_disable.py --model resnet8_subset06_pretrained_orig --phase full
"""

import argparse
import os
import sys
import numpy as np

import tvm
from tvm import relay

# ── model registry (reuse from test.py) ──────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from models import resnet8_subset_models  # noqa: E402

MODEL_REGISTRY = {
    f"resnet8_subset{n:02d}_pretrained_orig": (
        lambda n=n: resnet8_subset_models.getModel_from_pretrained_weight(
            iH=32, iW=32, until_relay=n
        ),
        "ones",
    )
    for n in list(range(1, 26)) + [31]
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "column_disable_config.json")


def load_model(model_name):
    """Load model and parameter dict from registry."""
    getter, input_pattern = MODEL_REGISTRY[model_name]
    mod, param_dict = getter()
    return mod, param_dict, input_pattern


def make_test_input(mod, pattern="ones"):
    """Create a simple test input tensor."""
    main_func = mod["main"]
    input_var = main_func.params[0]
    shape = [int(d) for d in input_var.type_annotation.shape]
    dtype = str(input_var.type_annotation.dtype)
    if pattern == "ones":
        data = np.ones(shape, dtype=dtype)
    elif pattern == "zeros":
        data = np.zeros(shape, dtype=dtype)
    elif pattern == "random":
        if "float" in dtype:
            data = np.random.randn(*shape).astype(dtype)
        else:
            info = np.iinfo(dtype)
            data = np.random.randint(info.min, info.max + 1, size=shape, dtype=dtype)
    else:
        data = np.ones(shape, dtype=dtype)
    return {input_var.name_hint: data}


# ── Phase A: Compile-only ─────────────────────────────────────────────

def run_compile_only(model_name, config_path, eval_dir, num_disable_columns=8,
                     noise_layout_json=None):
    """Run driver_v2 transform-only (skip codegen). Return transformed mod."""
    from tvm.driver.tvmc.imcflow_compiler_driver_v2 import compile_for_imcflow_v2

    mod, param_dict, _ = load_model(model_name)
    print(f"\n{'='*60}")
    print(f"Phase A: compile-only  model={model_name}")
    print(f"{'='*60}")

    mod, param_dict, _ = compile_for_imcflow_v2(
        mod, param_dict, eval_dir,
        column_disable_config_path=config_path,
        num_disable_columns=num_disable_columns,
        skip_codegen=True,
        save_intermediate=True,
        random_seed=42,
        noise_layout_json_path=noise_layout_json,
    )
    print(f"\nCompilation succeeded. Intermediates in {eval_dir}")
    return mod, param_dict


# ── Phase B: CPU validation ───────────────────────────────────────────

def run_cpu_reference(model_name):
    """Run original model (no column-disable) on CPU. Return output ndarray."""
    mod, param_dict, input_pattern = load_model(model_name)
    input_data = make_test_input(mod, input_pattern)

    # Bind params and build for CPU
    mod["main"] = relay.build_module.bind_params_by_name(mod["main"], param_dict)
    with tvm.transform.PassContext(opt_level=0):
        exe = relay.build(mod, target="llvm")
    from tvm.contrib.graph_executor import GraphModule
    dev = tvm.cpu()
    gmod = GraphModule(exe["default"](dev))
    for name, data in input_data.items():
        gmod.set_input(name, tvm.nd.array(data, dev))
    gmod.run()
    return gmod.get_output(0).numpy()


def run_cpu_validate(model_name, config_path, eval_dir, num_disable_columns=8,
                     noise_layout_json=None):
    """Run column-disable transform, then execute on CPU. Compare to reference."""
    from tvm.driver.tvmc.imcflow_compiler_driver_v2 import compile_for_imcflow_v2
    from tvm.relay.backend.contrib.imcflow.cpu_run import make_cpu_runnable

    mod, param_dict, input_pattern = load_model(model_name)
    input_data = make_test_input(mod, input_pattern)

    print(f"\n{'='*60}")
    print(f"Phase B: CPU validation  model={model_name}")
    print(f"{'='*60}")

    # Step 1: transform with column-disable (skip codegen)
    mod_cd, param_dict_cd, _ = compile_for_imcflow_v2(
        mod, param_dict, eval_dir,
        column_disable_config_path=config_path,
        num_disable_columns=num_disable_columns,
        skip_codegen=True,
        save_intermediate=True,
        random_seed=42,
        noise_layout_json_path=noise_layout_json,
    )

    # Step 2: make CPU-runnable
    mod_cpu = make_cpu_runnable(mod_cd, use_saturating_arithmetic=True)

    # Step 3: build & execute on CPU
    with tvm.transform.PassContext(opt_level=0):
        exe = relay.build(mod_cpu, target="llvm", params=param_dict_cd)
    from tvm.contrib.graph_executor import GraphModule
    dev = tvm.cpu()
    gmod = GraphModule(exe["default"](dev))
    for name, data in input_data.items():
        gmod.set_input(name, tvm.nd.array(data, dev))
    gmod.run()
    cd_output = gmod.get_output(0).numpy()

    # Step 4: reference (original, no column-disable)
    print("\nRunning CPU reference (no column-disable)...")
    ref_output = run_cpu_reference(model_name)

    # Step 5: compare
    print(f"\nReference shape: {ref_output.shape} dtype: {ref_output.dtype}")
    print(f"ColDis   shape: {cd_output.shape} dtype: {cd_output.dtype}")

    if ref_output.shape != cd_output.shape:
        print(f"SHAPE MISMATCH: {ref_output.shape} vs {cd_output.shape}")
        return False

    if np.issubdtype(ref_output.dtype, np.integer):
        match = np.array_equal(ref_output, cd_output)
        if not match:
            diff = np.abs(ref_output.astype(np.int32) - cd_output.astype(np.int32))
            n_diff = np.count_nonzero(diff)
            max_diff = diff.max()
            print(f"MISMATCH: {n_diff}/{diff.size} elements differ, max_diff={max_diff}")
            if max_diff <= 1:
                print("  (within 1 LSB — likely accumulation order difference)")
        else:
            print("EXACT MATCH")
    else:
        match = np.allclose(ref_output, cd_output, rtol=1e-5, atol=1e-5)
        if not match:
            diff = np.abs(ref_output - cd_output)
            print(f"MISMATCH: max_diff={diff.max():.6e}, mean_diff={diff.mean():.6e}")
        else:
            print("MATCH (within tolerance)")

    return match


# ── Phase C: Full simulation ──────────────────────────────────────────

def build_host_binary(eval_dir):
    """Build host binary using the template from test.py pattern."""
    import shutil
    import subprocess

    HOST_ISA = os.environ.get("IMCFLOW_HOST_ISA", "x86")

    test_host_binary_dir = f"{eval_dir}/host_binary_make"
    shutil.copytree("./host_binary_make.template", test_host_binary_dir, dirs_exist_ok=True)

    # Copy generated program_scan_reg_kernel.cc if it exists
    scan_kernel_src = f"{eval_dir}/build/program_scan_reg/program_scan_reg_kernel.cc"
    if os.path.exists(scan_kernel_src):
        scan_kernel_dst = f"{test_host_binary_dir}/src/program_scan_reg_kernel.cc"
        shutil.copy2(scan_kernel_src, scan_kernel_dst)

    host_build_dir = f"{test_host_binary_dir}/build"
    os.makedirs(host_build_dir, exist_ok=True)

    debug_exe = os.getenv("DEBUG_EXE", "0") == "1"
    main_script = "debug_execute_graph.c" if debug_exe else "execute_graph.c"
    binary_name = "debug_execute_graph" if debug_exe else "execute_graph"

    build_command = ["direnv", "exec", ".", "../build.sh", main_script, ".", HOST_ISA]
    log_dir = f"{eval_dir}/logs"
    os.makedirs(log_dir, exist_ok=True)
    build_log_path = os.path.join(log_dir, "build.log")

    print(f"Building host binary: {binary_name}")
    with open(build_log_path, "w") as log_file:
        process = subprocess.Popen(
            build_command, cwd=host_build_dir,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in process.stdout:
            log_file.write(line)
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, build_command)

    print(f"Host binary built. Log: {build_log_path}")
    return binary_name


def run_full_simulation(model_name, config_path, eval_dir, num_disable_columns=8,
                        noise_csv=None, noise_layout_json=None, noise_mode=None):
    """Full pipeline: compile + codegen + host binary build + py_runner simulation."""
    from tvm.driver.tvmc.imcflow_compiler_driver_v2 import compile_for_imcflow_v2
    from runners.imcflow_runner import get_runner
    from runners.input_generator import InputGenerator

    mod, param_dict, input_pattern = load_model(model_name)

    print(f"\n{'='*60}")
    print(f"Phase C: full simulation  model={model_name}")
    print(f"{'='*60}")

    mod_out, param_dict_out, tar_path = compile_for_imcflow_v2(
        mod, param_dict, eval_dir,
        column_disable_config_path=config_path,
        num_disable_columns=num_disable_columns,
        skip_codegen=False,
        save_intermediate=True,
        random_seed=42,
        noise_layout_json_path=noise_layout_json,
    )
    print(f"\nCompilation + codegen complete. tar={tar_path}")

    # Generate test inputs using InputGenerator (same as test.py)
    input_dir = os.path.join(eval_dir, "test_inputs")
    known_keys = list(param_dict.keys()) if param_dict else []
    gen = InputGenerator(mod=mod_out, known_keys=known_keys, seed=42)
    inputs = gen.generate_input(pattern=input_pattern)
    gen.save_to_files(inputs, input_dir)
    gen.save_to_files(param_dict_out, input_dir)
    print(f"Test inputs saved to {input_dir}")

    # Build host binary
    binary_name = build_host_binary(eval_dir)

    # Run py_runner simulation
    runner = get_runner("py")
    runner.setup()
    runner.run(
        binary_name=binary_name,
        gdb_mode="no",
        test_name=eval_dir,
        eval_dir=eval_dir,
        noise_csv=noise_csv,
        noise_layout_json=noise_layout_json,
        noise_mode=noise_mode,
    )

    output_path = runner.get_output_path(test_name=eval_dir)
    if os.path.exists(output_path):
        sim_output = np.load(output_path)
        print(f"\nSimulation output: shape={sim_output.shape} dtype={sim_output.dtype}")
        print(f"First 16 values: {sim_output.flat[:16]}")
    else:
        print(f"\nSimulation output not found at {output_path}")
        return False

    return True


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Column-disable verification")
    parser.add_argument("--model", required=True, help="Model name from registry")
    parser.add_argument("--phase", choices=["compile_only", "cpu_validate", "full"],
                        default="compile_only")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="Column disable config JSON")
    parser.add_argument("--num-disable", type=int, default=8,
                        help="Number of disabled columns per IMCE")
    parser.add_argument("--eval-dir", default=None,
                        help="Output directory (default: eval_dir/{model}_coldis_evl)")
    parser.add_argument("--noise-csv", default=None,
                        help="Path to ADC noise CSV (forwarded to py_runner). Phase 'full' only.")
    parser.add_argument("--noise-layout-json", default=None,
                        help="Path to imce_map noise layout JSON (concat_per_core.json). "
                             "Required when --noise-csv has n_cores*n_per_core channels. "
                             "Phase 'full' only.")
    parser.add_argument("--noise-mode", choices=["sample", "greedy"], default=None,
                        help="ADC noise sampling mode. 'sample' (default) or 'greedy' "
                             "(argmax over diff_bin). Phase 'full' only.")
    args = parser.parse_args()

    if args.model not in MODEL_REGISTRY:
        print(f"Unknown model: {args.model}")
        print(f"Available: {sorted(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    eval_dir = args.eval_dir or os.path.join(
        "eval_dir", f"{args.model}_coldis_evl"
    )
    os.makedirs(eval_dir, exist_ok=True)

    noise_layout_json = os.path.abspath(args.noise_layout_json) if args.noise_layout_json else None

    if args.phase == "compile_only":
        run_compile_only(args.model, args.config, eval_dir, args.num_disable,
                         noise_layout_json=noise_layout_json)
    elif args.phase == "cpu_validate":
        ok = run_cpu_validate(args.model, args.config, eval_dir, args.num_disable,
                              noise_layout_json=noise_layout_json)
        sys.exit(0 if ok else 1)
    elif args.phase == "full":
        noise_csv = os.path.abspath(args.noise_csv) if args.noise_csv else None
        ok = run_full_simulation(args.model, args.config, eval_dir, args.num_disable,
                                 noise_csv=noise_csv,
                                 noise_layout_json=noise_layout_json,
                                 noise_mode=args.noise_mode)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
