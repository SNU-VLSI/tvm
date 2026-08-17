import importlib
import hashlib
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
    assert value["scope"] == "continuous"
    assert value["mode"] == "now"
    assert value["rails"][0]["name"] == "DMM_GPIB3"
    assert value["metadata"]["model"] == "resnet8"

    short_config = json.loads(
        (CODEGEN_DIR / "power_configs" / "short_run.json").read_text(
            encoding="utf-8"
        )
    )
    assert short_config["mode"] == "now"
    assert short_config["duration_budget_s"] == 5

    region_config = json.loads(
        (CODEGEN_DIR / "power_configs" / "region.json").read_text(
            encoding="utf-8"
        )
    )
    assert region_config["scope"] == "region"
    scope = run_tool("config-scope", CODEGEN_DIR / "power_configs" / "region.json")
    assert scope.stdout.strip() == "region"


def test_disabled_config_does_not_prepare_session(tmp_path):
    config = tmp_path / "disabled.json"
    config.write_text(
        json.dumps({"schema_version": 1, "enabled": False}), encoding="utf-8"
    )
    result = run_tool("config-status", config, check=False)
    assert result.returncode == 10
    assert result.stdout.strip() == "disabled"


def test_codegen_build_identity_requires_matching_clean_revisions(tmp_path):
    metadata = tmp_path / "build_metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "tvm_git_rev": "a" * 40,
                "measurement_utils_git_rev": "b" * 40,
                "build_tree_dirty": False,
            }
        ),
        encoding="utf-8",
    )
    accepted = run_tool(
        "validate-build-identity",
        "--metadata",
        metadata,
        "--tvm-rev",
        "a" * 40,
        "--measurement-rev",
        "b" * 40,
    )
    assert "dirty=0" in accepted.stdout

    value = json.loads(metadata.read_text(encoding="utf-8"))
    value["build_tree_dirty"] = True
    metadata.write_text(json.dumps(value), encoding="utf-8")
    rejected = run_tool(
        "validate-build-identity",
        "--metadata",
        metadata,
        "--tvm-rev",
        "a" * 40,
        "--measurement-rev",
        "b" * 40,
        check=False,
    )
    assert rejected.returncode != 0
    assert "dirty tracked tree" in rejected.stderr


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


def test_schema_v2_raw_checksum_and_ambiguity_validation(tmp_path):
    result_dir = tmp_path / "metadata_result"
    rails_dir = result_dir / "rails"
    raw_dir = result_dir / "raw"
    rails_dir.mkdir(parents=True)
    raw_dir.mkdir()
    for name in ("request.json", "resolved_config.json"):
        (result_dir / name).write_text("{}\n", encoding="utf-8")
    raw = (
        b"Start date:,08/15/2026,Start time:,17:12:06.310\r\n"
        b"Sample interval:,0.000100\r\n"
        b"Reading #,Reading\r\n1,+1.0E-02\r\n2,+2.0E-02\r\n"
    )
    raw_path = raw_dir / "DMM_GPIB3.csv"
    raw_path.write_bytes(raw)
    checksum = {
        "DMM_GPIB3": {
            "path": "raw/DMM_GPIB3.csv",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size": len(raw),
        }
    }
    (raw_dir / "checksums.json").write_text(
        json.dumps(checksum), encoding="utf-8"
    )
    summary = {
        "schema_version": 2,
        "session_id": result_dir.name,
        "status": "complete",
        "tag_event_count": 1,
        "rails": {
            "DMM_GPIB3": {
                "sample_count": 2,
                "actual_sample_interval_s": 0.0001,
                "timestamp_source": "dmm_reading_metadata",
                "ambiguous_sample_count": 1,
                "tag_states": [
                    {
                        "tag_state_id": 0,
                        "state": {},
                        "sample_count": 2,
                        "ambiguous_sample_count": 1,
                        "average_current_A": 0.015,
                        "average_power_W": 0.015,
                        "energy_J": 0.000003,
                    }
                ],
            }
        },
    }
    for name in ("summary.json", "session.json"):
        (result_dir / name).write_text(json.dumps(summary), encoding="utf-8")
    (result_dir / "time_alignment.json").write_text("{}\n", encoding="utf-8")
    (result_dir / "tags.jsonl").write_text("", encoding="utf-8")
    np.savez_compressed(
        rails_dir / "DMM_GPIB3.npz",
        reading_number=np.asarray([1, 2]),
        current_A=np.asarray([0.01, 0.02]),
        time_from_trigger_s=np.asarray([0.0, 0.0001]),
        time_from_first_reading_s=np.asarray([0.0, 0.0001]),
        server_wall_time_ns=np.asarray([1000, 101000]),
        server_monotonic_time_ns=np.asarray([2000, 102000]),
        power_W=np.asarray([0.01, 0.02]),
        tag_state_id=np.asarray([0, 0]),
        tag_boundary_ambiguous=np.asarray([True, False]),
        sample_time_uncertainty_ns=np.asarray(1_000_000),
    )

    validated = run_tool("validate-result", result_dir)
    assert '"status": "complete"' in validated.stdout
    filtered = run_tool("summarize", result_dir, "--exclude-ambiguous")
    assert "samples=1" in filtered.stdout

    raw_path.write_bytes(raw + b"corrupt")
    rejected = run_tool("validate-result", result_dir, check=False)
    assert rejected.returncode != 0
    assert "SHA-256 mismatch" in rejected.stderr


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


def test_chip_runner_supports_bugfix_on_and_off_folder_suffixes():
    source = (CODEGEN_DIR / "run_chiptest.sh").read_text(encoding="utf-8")
    assert 'TEST_NAME="${TEST_FOLDER%_evl.linux}"' in source
    assert 'TEST_NAME="${TEST_FOLDER%_evl.linux.bugfixoff}"' in source
    assert "--no-patch" in source
    assert 'source "$SCRIPT_DIR/imcflow-linux.sh"' in source


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
    assert "power_measure_runtime.h" in generator.generateHeader()

    generator.os = "baremetal"
    assert generator.emit_power_tag_set("kernel", "x") == ""
    assert generator.emit_power_tag_clear("kernel") == ""
    assert "dmm_measure.h" not in generator.generateHeader()


def test_generated_kernel_tags_stages_tiles_and_retry(monkeypatch):
    monkeypatch.delenv("IMCFLOW_NO_PERKERNEL_WARMUP", raising=False)
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
    assert 'power_measure_runtime_region_begin("kernel_with_\\"quote")' in linux
    assert "power_measure_runtime_region_end()" in linux
    assert linux.index('dmm_tag_set("kernel_stage", "input_transfer")') < linux.index(
        'dmm_tag_set("kernel_stage", "output_transfer")'
    )
    assert linux.index("power_measure_runtime_region_begin") < linux.index(
        'dmm_tag_set("kernel_stage", "compiled_transfer")'
    )

    baremetal = str(make_generator("baremetal").makeKernelDef())
    assert "dmm_tag_" not in baremetal
    assert "power_measure_runtime_region_" not in baremetal
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
        assert '"--power-build-info"' in text
        assert "power_measure_runtime_print_build_info(stdout)" in text
    for source in sources[2:]:
        text = source.read_text(encoding="utf-8")
        assert "power_measure_runtime_sample(sample_idx)" in text
        assert 'power_measure_runtime_event("sample_timeout")' in text


def test_build_and_runner_gate_embed_deployed_revisions():
    for cmake_file in (
        CODEGEN_DIR / "host_binary_make.template/CMakeLists.txt",
        CODEGEN_DIR / "host_binary_make.dataset/CMakeLists.txt",
    ):
        text = cmake_file.read_text(encoding="utf-8")
        assert "IMCFLOW_BUILD_TVM_GIT_REV" in text
        assert "IMCFLOW_BUILD_MEASUREMENT_UTILS_GIT_REV" in text
        assert "IMCFLOW_BUILD_TREE_DIRTY" in text

    runner = (CODEGEN_DIR / "power_steps.sh").read_text(encoding="utf-8")
    assert '"$POWER_REMOTE_BINARY --power-build-info"' in runner
    assert "deployed binary revision mismatch" in runner
    assert "binary_tvm_git_rev" in runner
    assert "validate-build-identity" in runner
    assert "codegen_tvm_git_rev" in runner
    assert "HELLO 4" in runner
    assert "IMCFLOW_POWER_SCOPE" in runner
    assert "tracked repository changes must be committed" in runner
    assert "$DEFAULT_RUNNER_NAME $REMOTE_BASE_PATH/$NPZ_FILE_PATH" not in runner
    assert "/home/root/.venv/bin/activate" not in runner

    scan_steps = (CODEGEN_DIR / "scan_steps.sh").read_text(encoding="utf-8")
    assert "/home/root/.venv/bin/activate" not in scan_steps

    pipeline = (CODEGEN_DIR / "test.py").read_text(encoding="utf-8")
    assert 'metadata["tvm_git_rev"]' in pipeline
    assert 'metadata["measurement_utils_git_rev"]' in pipeline
    assert 'metadata["build_tree_dirty"]' in pipeline
