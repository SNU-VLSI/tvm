import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


CODEGEN_DIR = Path(__file__).resolve().parents[1]
TVM_ROOT = CODEGEN_DIR.parents[2]
REQUEST_TOOL = CODEGEN_DIR / "scripts" / "power_request.py"


def run_tool(*arguments, check=True):
    return subprocess.run(
        [sys.executable, str(REQUEST_TOOL), *map(str, arguments)],
        check=check,
        capture_output=True,
        text=True,
    )


def test_default_config_prepares_now_request(tmp_path):
    config = CODEGEN_DIR / "power_configs" / "default.json"
    request = tmp_path / "request.json"
    result = run_tool(
        "prepare",
        "--config",
        config,
        "--output",
        request,
        "--session-id",
        "unit_power_request",
        "--metadata",
        "model=resnet8",
    )
    assert result.returncode == 0
    value = json.loads(request.read_text(encoding="utf-8"))
    assert value["session_id"] == "unit_power_request"
    assert value["mode"] == "now"
    assert value["rails"][0]["name"] == "DMM_GPIB3"
    assert value["metadata"]["model"] == "resnet8"


def test_disabled_config_does_not_prepare_session(tmp_path):
    config = tmp_path / "disabled.json"
    config.write_text(
        json.dumps({"schema_version": 1, "enabled": False}), encoding="utf-8"
    )
    result = run_tool("config-status", config, check=False)
    assert result.returncode == 10
    assert result.stdout.strip() == "disabled"


def test_result_validation_and_tag_filter(tmp_path):
    result_dir = tmp_path / "result_validation"
    rails_dir = result_dir / "rails"
    rails_dir.mkdir(parents=True)
    for name in ("request.json", "resolved_config.json"):
        (result_dir / name).write_text("{}\n", encoding="utf-8")
    summary = {
        "session_id": result_dir.name,
        "status": "complete",
        "tag_event_count": 2,
        "rails": {
            "DMM_GPIB3": {
                "sample_count": 3,
                "energy_J": 0.003,
                "tag_states": [
                    {
                        "state": {"phase": "graph_execute"},
                        "sample_count": 2,
                        "average_current_A": 0.1,
                        "average_power_W": 0.1,
                        "energy_J": 0.002,
                    }
                ],
            }
        },
    }
    (result_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (result_dir / "session.json").write_text(json.dumps(summary), encoding="utf-8")
    (result_dir / "tags.jsonl").write_text("", encoding="utf-8")
    np.savez_compressed(
        rails_dir / "DMM_GPIB3.npz",
        current_A=np.asarray([0.1, 0.1, 0.1]),
        time_from_trigger_s=np.asarray([0.0, 0.01, 0.02]),
        power_W=np.asarray([0.1, 0.1, 0.1]),
        tag_state_id=np.asarray([0, 1, 1]),
    )

    validated = run_tool("validate-result", result_dir)
    assert '"status": "complete"' in validated.stdout
    filtered = run_tool("summarize", result_dir, "--tag", "phase=graph_execute")
    assert '"phase": "graph_execute"' in filtered.stdout


def test_shell_scripts_parse_and_power_runtime_compiles(tmp_path):
    subprocess.run(
        [
            "bash",
            "-n",
            str(CODEGEN_DIR / "power_steps.sh"),
            str(CODEGEN_DIR / "run_chiptest.sh"),
            str(CODEGEN_DIR / "run_dataset_eval.sh"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "cc",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{CODEGEN_DIR / 'power_runtime'}",
            f"-I{TVM_ROOT / '3rdparty/measurement_utils/capi'}",
            "-c",
            str(CODEGEN_DIR / "power_runtime/power_measure_runtime.c"),
            "-o",
            str(tmp_path / "power_runtime.o"),
        ],
        check=True,
    )


def _load_ext_codegen(monkeypatch):
    monkeypatch.setenv("IMCFLOW_HOST_OS", "linux")
    monkeypatch.setenv("IMCFLOW_ADDR", "0")
    monkeypatch.setenv("IMCFLOW_LEN", "0")
    monkeypatch.setenv("INT_ACK_GEN_ADDR", "0")
    monkeypatch.setenv("INT_ACK_GEN_LEN", "0")
    module_name = "tvm.relay.backend.contrib.imcflow.ext_codegen"
    if module_name in sys.modules:
        return sys.modules[module_name]
    return importlib.import_module(module_name)


def test_codegen_helpers_are_linux_only(monkeypatch):
    module = _load_ext_codegen(monkeypatch)

    generator = module.KernelCodeGenerator.__new__(module.KernelCodeGenerator)
    generator.os = "linux"
    assert generator.emit_power_tag_set("kernel", 'quoted"name') == (
        'dmm_tag_set("kernel", "quoted\\\"name");\n'
    )
    assert "dmm_measure.h" in generator.generateHeader()

    generator.os = "baremetal"
    assert generator.emit_power_tag_set("kernel", "x") == ""
    assert generator.emit_power_tag_clear("kernel") == ""
    assert "dmm_measure.h" not in generator.generateHeader()


def test_generated_kernel_tags_stages_tiles_and_retry(monkeypatch):
    module = _load_ext_codegen(monkeypatch)
    monkeypatch.setattr(module, "makeConstArrayDecl", lambda *_args: "")

    def make_generator(target_os):
        generator = module.KernelCodeGenerator.__new__(module.KernelCodeGenerator)
        generator.os = target_os
        generator.func_name = 'kernel_with_"quote'
        generator.func = SimpleNamespace(params=[])
        generator.target_func = SimpleNamespace(params=[])
        generator.target_func_info = SimpleNamespace(tiling_factor=2)
        generator.output_node_types = []
        generator.input_node_types = []
        generator.compiled_blocks = []
        generator.compiled_per_tile_blocks = []
        generator.const_blocks = []
        generator.input_blocks = []
        generator.output_blocks = []
        generator.generateExternLink = lambda: ""
        generator.generatePollingUtilities = lambda: ""
        generator.generateInterruptUtilities = lambda: ""
        generator.generateDevicePointerSetup = lambda: "device_setup();\n"
        generator.emitReset = lambda: "reset_device();\n"
        generator.emitWarmup = lambda: "warmup_device();\n"
        generator.generateToNpuTransferCode = lambda *_args: "transfer();\n"
        generator.generatePolicyUpdateCode = lambda: "policy_update();\n"
        generator.generateInvokeCode = lambda: "invoke();\n"
        generator.generateFromNpuTransferCode = lambda *_args: "read_output();\n"
        generator.generateDevicePointerCleanup = lambda: "cleanup_device();\n"
        generator.generatePackedFuncWrapper = lambda: ""
        generator.generateBaseAddrMacros = lambda: module.CodeWriter()
        return generator

    linux = str(make_generator("linux").makeKernelDef())
    assert 'dmm_tag_set("kernel", "kernel_with_\\\"quote")' in linux
    assert 'dmm_tag_set("kernel_stage", "warmup")' in linux
    assert 'dmm_tag_set("tile", "0")' in linux
    assert 'dmm_tag_set("tile", "1")' in linux
    assert 'dmm_tag_event("retry")' in linux
    assert linux.index('dmm_tag_set("kernel_stage", "input_transfer")') < linux.index(
        'dmm_tag_set("kernel_stage", "output_transfer")'
    )

    baremetal = str(make_generator("baremetal").makeKernelDef())
    assert "dmm_tag_" not in baremetal
    assert "dmm_measure.h" not in baremetal


def test_host_templates_cover_single_and_dataset_phases():
    sources = [
        CODEGEN_DIR / "host_binary_make.template/src/execute_graph.c",
        CODEGEN_DIR / "host_binary_make.template/src/debug_execute_graph.c",
        CODEGEN_DIR / "host_binary_make.dataset/src/execute_graph_for_dataset.c",
        CODEGEN_DIR / "host_binary_make.dataset/src/debug_execute_graph_for_dataset.c",
    ]
    for source in sources:
        text = source.read_text(encoding="utf-8")
        assert "power_measure_runtime_start()" in text
        assert 'power_measure_runtime_phase("graph_execute")' in text
        assert "power_measure_runtime_finish()" in text
    for source in sources[2:]:
        text = source.read_text(encoding="utf-8")
        assert "power_measure_runtime_sample(sample_idx)" in text
        assert 'power_measure_runtime_event("sample_timeout")' in text
