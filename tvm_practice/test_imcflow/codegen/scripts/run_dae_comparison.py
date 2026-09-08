"""Run the fixed 10-input/two-repeat PyRunner or RTL DAE comparison, preserving dumps.

Reuses the compiled driver-v2 model; never recompiles or cleans the eval tree.
Stops on the first failed run/boundary/repetition and retains its full artifacts.
This verifies one simulator condition, not the RTL/board/AUC integration goal.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from compare_dae_runner import compare_sample, logical_boundaries, read_dump, read_graph, sha256


def save_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def run(args):
    codegen = Path(__file__).resolve().parents[1]
    checkpoint = Path(args.checkpoint).resolve()
    capture = Path(args.deploy_capture).resolve()
    output = Path(args.output).resolve()
    disabled = Path(args.disabled_json).resolve()
    layout = Path(args.layout_json).resolve()
    model = "dae_toycar_full_pretrained"
    backend = args.backend
    runner_name = f"{backend}_runner"
    if backend == "rtl" and args.noise_csv:
        raise ValueError("RTL comparison is noise-off only; use PyRunner for greedy noise")
    evaluation = codegen / "eval_dir" / f"{model}_evl.baremetal"
    dataset = codegen / "dataset/toyadmos"
    mlf = evaluation / "lib_graph_system-lib.tar"
    build = json.loads((evaluation / "build_metadata.json").read_text())
    reference = json.loads((capture / "manifest.json").read_text())
    expected_build = dict(model_name=model, board="B1", driver_v2=True, num_disable_columns=32,
                          random_seed=42, vmode="HALF")
    if any(build.get(k) != v for k, v in expected_build.items()):
        raise ValueError("Compiled metadata differs from the DAE driver-v2/B1/HALF/N32/seed42 contract")
    if Path(build["checkpoint_path"]).resolve() != checkpoint:
        raise ValueError("Compiled checkpoint path differs")
    if sha256(build["column_disable_config"]) != sha256(disabled):
        raise ValueError("Disable configuration differs from compiled metadata")
    expected_reference = dict(checkpoint_sha256=sha256(checkpoint), input_sha256=sha256(dataset / "images.npy"),
                              metadata_sha256=sha256(dataset / "metadata.json"),
                              trace_sha256=sha256(capture / "traces.npz"), samples=10, repeats=2, deterministic=True)
    if any(reference.get(k) != v for k, v in expected_reference.items()):
        raise ValueError("Reference provenance/determinism differs from the fixed test contract")
    if reference["psum_config"]["acc_mask"] != 1 or reference["psum_config"]["prange"] != 2:
        raise ValueError("Expected ACC_MASK=1 and HALF ADC")
    inputs = np.load(dataset / "images.npy", allow_pickle=False)
    labels = np.load(dataset / "labels.npy", allow_pickle=False)
    metadata = json.loads((dataset / "metadata.json").read_text())
    if inputs.shape != (10, 640) or inputs.dtype != np.float32 or labels.shape != (10,):
        raise ValueError("Expected ten fixed float32 DAE inputs")
    if np.count_nonzero(labels == 0) != 5 or np.count_nonzero(labels == 1) != 5:
        raise ValueError("Expected five normal and five anomalous inputs")
    if metadata["mode"] != "verification" or len(metadata["files"]) != 10:
        raise ValueError("Expected verification manifest")
    noise_mode = "greedy" if args.noise_csv else "off"
    if reference["noise_mode"] != noise_mode:
        raise ValueError("Deploy and runner noise modes differ")
    tracked = dict(checkpoint=checkpoint, mlf=mlf, disabled=disabled, layout=layout,
                   psum_map=evaluation / "psum_imcu_column_map.npz", images=dataset / "images.npy",
                   labels=dataset / "labels.npy", metadata=dataset / "metadata.json", traces=capture / "traces.npz")
    if args.noise_csv:
        tracked["noise_csv"] = Path(args.noise_csv).resolve()
        config = reference["noise_config"]
        if (config.get("chip_noise_table_format") != args.noise_table_format or
                config.get("chip_noise_granularity") != args.noise_granularity):
            raise ValueError("Noise table format/granularity differs from deploy")
        for key in ("noise_csv", "psum_map", "layout"):
            if reference["noise_artifacts"][key]["sha256"] != sha256(tracked[key]):
                raise ValueError("Noise artifact differs from deploy")
    hashes = {key: sha256(path) for key, path in tracked.items()}
    graph = read_graph(mlf)
    boundaries = logical_boundaries(graph)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "manifest.json", dict(backend=runner_name, noise_mode=noise_mode,
              samples=10, repeats=2, integer_atol=1, integer_rtol=0, fp_atol=1e-3, fp_rtol=1e-3,
              repeat_comparison="exact", artifacts={k: dict(path=str(p), sha256=hashes[k]) for k, p in tracked.items()},
              compile_metadata=build, deploy_manifest=reference))
    env = {key: value for key, value in os.environ.items() if not key.startswith("IMCFLOW_NOISE_")}
    env.update(BOARD="B1", ACC_MASK="1", IMCFLOW_HOST_OS="baremetal", IMCFLOW_HOST_ISA="x86",
               IMCFLOW_RUNNER=backend, DEBUG_EXE="1", TVM_LOG_DEBUG="0", CKPT_PATH=str(checkpoint))
    tvm_root = codegen.parents[2]
    env["PYTHONPATH"] = os.pathsep.join([str(tvm_root / "tvm_practice"), str(tvm_root / "python"),
                                        str(tvm_root / "vta/python"), env.get("PYTHONPATH", "")])
    summaries = []
    for repeat in range(2):
        for sample in range(10):
            for key, path in tracked.items():
                if sha256(path) != hashes[key]:
                    raise ValueError(f"Artifact changed during validation: {key}")
            destination = output / f"repeat{repeat}" / f"sample_{sample}"
            destination.mkdir(parents=True)
            active_dump = evaluation / "test_outputs" / runner_name / f"sample_{sample}"
            if active_dump.exists():
                archive = output / "previous_outputs" / f"before_repeat{repeat}_sample{sample}"
                archive.parent.mkdir(exist_ok=True)
                shutil.move(str(active_dump), archive)
            command = [sys.executable, "main.py", "--model", model, "--driver-v2", "--ref-models", "transformed",
                       "--num-disable-columns", "32", "--column-disable-config", str(disabled),
                       "--random-seed", "42", "--start-at", "simulate", "--stop-at", "simulate",
                       "--dataset", "toyadmos", "--sample", str(sample)]
            if args.noise_csv:
                command += ["--noise-csv", str(tracked["noise_csv"]), "--noise-layout-json", str(layout),
                            "--noise-mode", "greedy", "--noise-table-format", args.noise_table_format,
                            "--noise-granularity", args.noise_granularity, "--noise-seed", "42"]
            save_json(destination / "command.json", dict(argv=command, environment={k: env[k] for k in
                      ("BOARD", "ACC_MASK", "IMCFLOW_RUNNER", "IMCFLOW_HOST_OS", "IMCFLOW_HOST_ISA", "DEBUG_EXE", "CKPT_PATH")}))
            print(f"START {noise_mode} repeat={repeat} sample={sample}", flush=True)
            start = time.monotonic()
            with (destination / "main.log").open("x") as log:
                process = subprocess.run(command, cwd=codegen, env=env, stdout=log, stderr=subprocess.STDOUT)
            # Preserve failed outputs/logs too; never mistake an old output for a new run.
            if active_dump.exists():
                shutil.move(str(active_dump), destination / "dumps")
            runner_logs = evaluation / "logs" / runner_name
            if runner_logs.exists():
                shutil.copytree(runner_logs, destination / "runner_logs")
            if process.returncode:
                raise RuntimeError(f"Runner failed with {process.returncode}: {destination / 'main.log'}")
            actual_input = evaluation / "test_inputs" / f"sample_{sample}" / "model_input.npy"
            if not np.array_equal(np.load(actual_input, allow_pickle=False), inputs[sample:sample + 1]):
                raise ValueError("Runner consumed a different dataset input")
            shutil.copy2(actual_input, destination / "model_input.npy")
            report = compare_sample(mlf, destination / "dumps", capture, sample, repeat)
            report["elapsed_seconds"] = time.monotonic() - start
            report["repeat_exact"] = None
            if repeat:
                first = output / "repeat0" / f"sample_{sample}" / "dumps"
                report["repeat_exact"] = all(np.array_equal(read_dump(graph, first, index),
                    read_dump(graph, destination / "dumps", index)) for index in boundaries.values())
                report["passed"] = report["passed"] and report["repeat_exact"]
            save_json(destination / "comparison.json", report)
            summary = {k: report[k] for k in ("sample", "repeat", "passed", "first_mismatch", "repeat_exact", "elapsed_seconds")}
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
            if not report["passed"]:
                raise AssertionError(f"DAE comparison failed: {destination / 'comparison.json'}")
    save_json(output / "result.json", dict(numeric_condition_complete=True, integration_complete=False,
              backend=runner_name, noise_mode=noise_mode, samples=10, repeats=2, repeat_exact=True, runs=summaries,
              note="This condition only; remaining simulator/RTL/board/AUC plan gates are separate."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("py", "rtl"), default="py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--deploy-capture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--disabled-json", required=True)
    parser.add_argument("--layout-json", required=True)
    parser.add_argument("--noise-csv")
    parser.add_argument("--noise-table-format", choices=("wpattern_ref", "ref"), default="wpattern_ref")
    parser.add_argument("--noise-granularity", choices=("weight_bitplane", "input_bitplane"), default="weight_bitplane")
    run(parser.parse_args())
