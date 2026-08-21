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
    assert "scope" not in value
    assert value["mode"] == "now"
    assert "region_loop" not in value
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
    assert region_config["scope"] == "REGION"
    scope = run_tool("config-scope", CODEGEN_DIR / "power_configs" / "region.json")
    assert scope.stdout.strip() == "REGION"
    loop = json.loads(
        run_tool("config-loop", CODEGEN_DIR / "power_configs" / "region.json").stdout
    )
    assert loop == {"loop_enable": False, "min_samples": 0, "min_seconds": 0.0}

    tile_config = json.loads(
        (CODEGEN_DIR / "power_configs" / "tile.json").read_text(encoding="utf-8")
    )
    assert tile_config["scope"] == "TILE"
    assert tile_config["region_loop"] == {
        "loop_enable": False,
        "min_samples": 0,
        "min_seconds": 0.0,
    }

    wait_config_path = CODEGEN_DIR / "power_configs" / "tile_wait_min.json"
    wait_config = json.loads(wait_config_path.read_text(encoding="utf-8"))
    assert wait_config["mode"] == "wait"
    assert wait_config["defaults"]["sample_interval_s"] == "MIN"
    assert wait_config["defaults"]["sample_count"] == 50000
    assert run_tool("config-mode", wait_config_path).stdout.strip() == "wait"
    wait_request = tmp_path / "wait_request.json"
    run_tool(
        "prepare",
        "--config",
        wait_config_path,
        "--output",
        wait_request,
        "--session-id",
        "wait_request",
    )
    assert json.loads(wait_request.read_text(encoding="utf-8"))["mode"] == "wait"


def test_disabled_config_does_not_prepare_session(tmp_path):
    config = tmp_path / "disabled.json"
    config.write_text(
        json.dumps({"schema_version": 1, "enabled": False}), encoding="utf-8"
    )
    result = run_tool("config-status", config, check=False)
    assert result.returncode == 10
    assert result.stdout.strip() == "disabled"


def test_power_policy_rejects_legacy_scope_unknown_mode_and_impossible_minimum(tmp_path):
    base = json.loads(
        (CODEGEN_DIR / "power_configs/region.json").read_text(encoding="utf-8")
    )
    cases = [
        ("legacy_scope", {**base, "scope": "continuous"}),
        ("unknown_mode", {**base, "mode": "continuous"}),
        (
            "tile_scope_loop",
            {
                **base,
                "scope": "TILE",
                "region_loop": {
                    "loop_enable": True,
                    "min_samples": 100,
                    "min_seconds": 0.0,
                },
            },
        ),
        (
            "too_many_samples",
            {
                **base,
                "scope": "MODEL",
                "region_loop": {
                    "loop_enable": True,
                    "min_samples": 50_001,
                    "min_seconds": 0.0,
                },
            },
        ),
    ]
    for name, value in cases:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        result = run_tool("config-status", path, check=False)
        assert result.returncode != 0


def test_tvm_manifest_groups_scope_free_region_artifacts(tmp_path):
    result_dir = tmp_path / "session_a"
    (result_dir / "regions" / "r0002_tile_1").mkdir(parents=True)
    (result_dir / "regions" / "r0001_tile_0").mkdir()
    policy = {"loop_enable": True, "min_samples": 100, "min_seconds": 0.1}
    run_tool(
        "write-tvm-manifest",
        result_dir,
        "--scope",
        "TILE",
        "--mode",
        "wait",
        "--region-loop",
        json.dumps(policy),
    )
    manifest = json.loads(
        (result_dir / "tvm_power_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["scope"] == "TILE"
    assert manifest["mode"] == "wait"
    assert manifest["region_ids"] == ["r0001_tile_0", "r0002_tile_1"]
    assert manifest["region_loop"] == policy


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
    (result_dir / "tags.jsonl").write_text(
        json.dumps(
            {
                "kind": "event",
                "name": "tile_start",
                "client_converted_monotonic_ns": 2000,
            }
        )
        + "\n",
        encoding="utf-8",
    )
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

    summary["status"] = "truncated"
    for name in ("summary.json", "session.json"):
        (result_dir / name).write_text(json.dumps(summary), encoding="utf-8")
    truncated = run_tool("validate-result", result_dir)
    assert truncated.returncode == 0
    assert '"status": "truncated"' in truncated.stdout
    assert "captured artifact is valid" in truncated.stderr

    plot_path = result_dir / "power_trace.png"
    run_tool("plot", result_dir, "--output", plot_path)
    assert plot_path.is_file()
    assert plot_path.stat().st_size > 0

    summary["status"] = "partial"
    for name in ("summary.json", "session.json"):
        (result_dir / name).write_text(json.dumps(summary), encoding="utf-8")
    partial = run_tool("validate-result", result_dir, check=False)
    assert partial.returncode != 0


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
    (result_dir / "tags.jsonl").write_text(
        json.dumps(
            {
                "kind": "event",
                "name": "tile_start",
                "client_converted_monotonic_ns": 2000,
            }
        )
        + "\n",
        encoding="utf-8",
    )
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
    plot_path = result_dir / "power_trace.png"
    run_tool("plot", result_dir, "--output", plot_path)
    assert plot_path.stat().st_size > 0

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
    power_steps = (CODEGEN_DIR / "power_steps.sh").read_text(encoding="utf-8")
    assert 'validate-result "$region_dir"' in power_steps
    assert 'plot "$region_dir"' in power_steps
    assert '"$region_dir/power_trace.png"' in power_steps
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
    assert 'getenv("DEBUG_PRINT_INSTRUMENT")' not in generator.generateHeader()
    assert "IMCFLOW_DEBUG_PRINT" not in generator.generateHeader()

    generator.os = "baremetal"
    assert generator.emit_power_tag_set("kernel", "x") == ""
    assert generator.emit_power_tag_clear("kernel") == ""
    assert "dmm_measure.h" not in generator.generateHeader()


def test_normal_tensor_word_accesses_use_per_word_mmio_barriers(monkeypatch):
    monkeypatch.setenv("IMCFLOW_MMIO_BARRIER", "0")
    module = _load_ext_codegen(monkeypatch)
    generator = module.KernelCodeGenerator.__new__(module.KernelCodeGenerator)

    write = generator.emitTensorMmioWrite32(
        "npu_pointer", "base + i", "input[i]", "  "
    )
    read = generator.emitTensorMmioRead32Expr("npu_pointer", "base + i")

    assert "imcflow_mmio_write32(npu_pointer, base + i, input[i]);" in write
    assert "MMIO-BARRIER-EXPERIMENT: fence this individual write" in write
    assert read == "imcflow_mmio_read32(npu_pointer, base + i)"


def test_tensor_transfer_has_no_debug_instrumentation(monkeypatch):
    module = _load_ext_codegen(monkeypatch)
    monkeypatch.setattr(module, "makeBaseAddrName", lambda _block: "INPUT_BASE_ADDR")
    monkeypatch.setattr(module, "getCInputVarName", lambda *_args: "input_tensor")

    generator = module.KernelCodeGenerator.__new__(module.KernelCodeGenerator)
    generator.func_name = "debug_kernel"
    generator.base_address_macros = {}
    block = SimpleNamespace(
        id=0,
        base_address=0x1000,
        size=4096,
        tiling_info=None,
    )

    generated = str(generator.generateToNpuTransferCode([block], None, "input"))
    assert "IMCFLOW_DEBUG_PRINT" not in generated
    assert "fprintf" not in generated
    assert "for(int i=0; i<1024; i++)" in generated


def test_invoke_run_wait_and_finalize_use_conservative_barriers(monkeypatch):
    monkeypatch.setenv("IMCFLOW_MMIO_BARRIER", "0")
    module = _load_ext_codegen(monkeypatch)
    generator = module.KernelCodeGenerator.__new__(module.KernelCodeGenerator)
    generator.os = "linux"
    generator.func_name = "tight_tile_kernel"

    prepare = str(generator.generateInvokePrepareCode())
    run_wait = str(generator.generateInvokeStartWaitCode())
    finalize = str(generator.generateInvokeFinalizeCode())
    complete = str(generator.generateInvokeCode())

    assert "enable_imcflow_interrupt(npu_fd)" in prepare
    assert "invoke interrupt arm completes before RUN doorbell" in prepare
    assert run_wait == (
        "/* IMCFLOW-INVOKE: RUN doorbell intentionally has no post barrier. */\n"
        "npu_pointer[STATE_REG_IDX] = SET_RUN_CODE;\n"
        "_wait_rc = wait_imcflow_interrupt(npu_fd, npu_pointer);"
    )
    assert "imcflow_mmio_barrier(" not in run_wait
    assert "IMCFLOW_DEBUG_PRINT(" not in run_wait
    assert "dmm_tag_" not in run_wait
    assert "invoke completion observed before interrupt ACK" in finalize
    assert "generate_ack(int_ack_gen_pointer)" in finalize
    assert "imcflow_mmio_write32(npu_pointer, INTR_DONE_REG_IDX, 1)" in finalize
    assert complete.index("enable_imcflow_interrupt") < complete.index(
        "npu_pointer[STATE_REG_IDX] = SET_RUN_CODE"
    )
    assert complete.index("wait_imcflow_interrupt") < complete.index(
        "generate_ack(int_ack_gen_pointer)"
    )


def test_generated_kernel_uses_only_timing_events(monkeypatch):
    monkeypatch.delenv("IMCFLOW_NO_PERKERNEL_WARMUP", raising=False)
    monkeypatch.setenv("IMCFLOW_MMIO_BARRIER", "0")
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
    assert 'dmm_tag_set("kernel"' not in linux
    assert 'dmm_tag_set("kernel_stage"' not in linux
    assert 'dmm_tag_set("tile"' not in linux
    assert 'dmm_tag_set("retry_attempt"' not in linux
    assert 'power_measure_runtime_event("retry")' in linux
    assert (
        'TVM_POWER_REGION_BEGIN(IMCFLOW_POWER_SCOPE_REGION, '
        '"kernel_with_\\"quote")' in linux
    )
    assert "TVM_POWER_REGION_BEGIN(IMCFLOW_POWER_SCOPE_TILE" in linux
    assert "TVM_POWER_REGION_END()" in linux
    assert linux.count("TVM_POWER_REGION_BEGIN(") == linux.count(
        "TVM_POWER_REGION_END()"
    )
    assert "_power_retry_requested = 1" in linux
    assert "power_measure_runtime_model_start_after_first_warmup()" in linux
    assert linux.index("warmup_device();") < linux.index(
        "power_measure_runtime_model_start_after_first_warmup()"
    )
    model_start_status = linux.index(
        "power_measure_runtime_model_start_after_first_warmup()"
    )
    model_start_event = linux.index(
        'power_measure_runtime_event("model_start")', model_start_status
    )
    region_begin = linux.index(
        "TVM_POWER_REGION_BEGIN(IMCFLOW_POWER_SCOPE_REGION", model_start_event
    )
    assert model_start_status < model_start_event < region_begin
    region_event_start = linux.index(
        'power_measure_runtime_event("region_start")', region_begin
    )
    region_event_end = linux.index(
        'power_measure_runtime_event("region_end")', region_event_start
    )
    region_end = linux.index("TVM_POWER_REGION_END()", region_event_end)
    assert region_begin < region_event_start < region_event_end < region_end
    assert (
        "power_measure_runtime_scope_is(IMCFLOW_POWER_SCOPE_REGION)"
        in linux[region_begin:region_end]
    )
    tile_begin = linux.index("TVM_POWER_REGION_BEGIN(IMCFLOW_POWER_SCOPE_TILE")
    tile_event_start = linux.index(
        'power_measure_runtime_event("tile_start")', tile_begin
    )
    tile_run = linux.index("npu_pointer[STATE_REG_IDX] = SET_RUN_CODE", tile_event_start)
    tile_wait = linux.index("wait_imcflow_interrupt", tile_run)
    tile_ack = linux.index("generate_ack(int_ack_gen_pointer)", tile_wait)
    tile_intr_done = linux.index(
        "imcflow_mmio_write32(npu_pointer, INTR_DONE_REG_IDX, 1)", tile_ack
    )
    tile_event_end = linux.index(
        'power_measure_runtime_event("tile_end")', tile_intr_done
    )
    tile_end = linux.index("TVM_POWER_REGION_END()", tile_event_end)
    tight_tile_body = linux[tile_begin:tile_end]
    assert (
        tile_begin < tile_event_start < tile_run < tile_wait < tile_ack
        < tile_intr_done < tile_event_end < tile_end
    )
    assert "imcflow_mmio_barrier(" in tight_tile_body
    assert "imcflow_mmio_write32(" in tight_tile_body
    assert "dmm_tag_" not in tight_tile_body

    baremetal = str(make_generator("baremetal").makeKernelDef())
    assert "dmm_tag_" not in baremetal
    assert "TVM_POWER_REGION_" not in baremetal
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
        assert "power_measure_runtime_phase(" not in text
        assert "power_measure_runtime_sample(" not in text
        assert "power_measure_runtime_finish()" in text
        assert "TVM_POWER_REGION_BEGIN(IMCFLOW_POWER_SCOPE_MODEL" in text
        assert '"--power-build-info"' in text
        assert "power_measure_runtime_print_build_info(stdout)" in text
    runtime = (
        CODEGEN_DIR / "power_runtime/power_measure_runtime.c"
    ).read_text(encoding="utf-8")
    assert 'power_measure_runtime_event("model_end")' in runtime
    for source in sources[2:]:
        text = source.read_text(encoding="utf-8")
        assert 'power_measure_runtime_event("sample_timeout")' in text


def test_nested_scope_macro_retry_shape_is_valid_c():
    subprocess.run(
        [
            "cc",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fsyntax-only",
            f"-I{CODEGEN_DIR / 'power_runtime'}",
            f"-I{TVM_ROOT / '3rdparty/measurement_utils/capi'}",
            str(CODEGEN_DIR / "unittests/power_scope_macro_syntax.c"),
        ],
        check=True,
    )


def test_model_scope_begins_after_generated_warmup(tmp_path):
    executable = tmp_path / "power_measure_runtime_unit"
    subprocess.run(
        [
            "cc",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{CODEGEN_DIR / 'power_runtime'}",
            f"-I{TVM_ROOT / '3rdparty/measurement_utils/capi'}",
            str(CODEGEN_DIR / "unittests/power_measure_runtime_unit.c"),
            str(CODEGEN_DIR / "power_runtime/power_measure_runtime.c"),
            "-o",
            str(executable),
        ],
        check=True,
    )
    result = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True
    )
    assert "MODEL/REGION/TILE loop policy: OK" in result.stdout


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
    assert "HELLO 6" in runner
    assert "IMCFLOW_POWER_SCOPE" in runner
    assert "IMCFLOW_POWER_MIN_SAMPLES" in runner
    assert "tracked repository changes must be committed" in runner
    assert "$DEFAULT_RUNNER_NAME $REMOTE_BASE_PATH/$NPZ_FILE_PATH" not in runner
    assert "/home/root/.venv/bin/activate" not in runner

    scan_steps = (CODEGEN_DIR / "scan_steps.sh").read_text(encoding="utf-8")
    assert "/home/root/.venv/bin/activate" not in scan_steps

    pipeline = (CODEGEN_DIR / "test.py").read_text(encoding="utf-8")
    assert 'metadata["tvm_git_rev"]' in pipeline
    assert 'metadata["measurement_utils_git_rev"]' in pipeline
    assert 'metadata["build_tree_dirty"]' in pipeline


def test_model_wait_runner_warms_chip_before_execution_and_has_liveness_guards():
    dataset_runner = (CODEGEN_DIR / "run_dataset_eval.sh").read_text(
        encoding="utf-8"
    )
    warmup = "cd /home/root/imcflow/xilinx/petalinux-csrc && make warmup"
    execute = "taskset -c $CHIP_EVAL_CPU"
    assert "IMCFLOW_PRE_RUN_WARMUP" in dataset_runner
    assert warmup in dataset_runner
    assert dataset_runner.index(warmup) < dataset_runner.index(execute)

    scan_steps = (CODEGEN_DIR / "scan_steps.sh").read_text(encoding="utf-8")
    assert "ConnectTimeout=$SCAN_SSH_CONNECT_TIMEOUT_SECONDS" in scan_steps
    assert "ServerAliveInterval=$SCAN_SSH_SERVER_ALIVE_INTERVAL_SECONDS" in scan_steps
    assert "ServerAliveCountMax=$SCAN_SSH_SERVER_ALIVE_COUNT_MAX" in scan_steps

    model_runner = (
        CODEGEN_DIR / "scripts/run_resnet_model_wait_power.sh"
    ).read_text(encoding="utf-8")
    assert 'IMCFLOW_PRE_RUN_WARMUP:-1' in model_runner
    assert 'CHIP_RUN_TIMEOUT_SECONDS:-360' in model_runner
    assert "timeout --signal=TERM --kill-after=20s" in model_runner
    assert 'IMCFLOW_ADDR:-0xa0000000' in model_runner
    assert 'IMCFLOW_LEN:-0x100000' in model_runner
    assert 'INT_ACK_GEN_ADDR:-0xa0110000' in model_runner
    assert 'INT_ACK_GEN_LEN:-0x10000' in model_runner
    assert '$TVM_ROOT/tvm_practice:$SCRIPT_DIR/tools:/root/project/CIM' in model_runner
    assert 'LD_LIBRARY_PATH="$TVM_ROOT/build:${LD_LIBRARY_PATH:-}"' in model_runner
    assert 'BOARD="${BOARD:-B1}"' in model_runner
    assert 'BOARD="$BOARD"' in model_runner
