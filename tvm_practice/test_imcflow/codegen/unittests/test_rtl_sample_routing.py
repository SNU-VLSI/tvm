import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("imcflow_runner",
    Path(__file__).resolve().parents[1] / "runners/imcflow_runner.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_rtl_sample_path_and_legacy_path(tmp_path):
    runner = module.RTLRunner()
    assert runner.get_output_path(str(tmp_path), 7) == str(tmp_path / "test_outputs/rtl_runner/sample_7/output.npy")
    assert runner.get_output_path(str(tmp_path)) == str(tmp_path / "test_outputs/rtl_runner/output.npy")


def test_rtl_relative_output_uses_current_worktree():
    codegen = Path(__file__).resolve().parents[1]
    assert module.RTLRunner().get_output_path("eval_dir/dae", 7) == str(
        codegen / "eval_dir/dae/test_outputs/rtl_runner/sample_7/output.npy"
    )


def test_rtl_passes_current_worktree_to_shared_runner(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setenv("IMCFLOW_TVM_CODEGEN_DIR", "/stale/worktree")
    monkeypatch.setattr(module.PortAllocator, "get_port_for_test", lambda _: 10042)
    monkeypatch.setattr(module.PortAllocator, "release_port", lambda _: None)
    monkeypatch.setattr(module.RTLRunner, "_stream_command_output", lambda self, **kwargs: commands.append(kwargs))
    module.RTLRunner().run("execute_graph", "no", "test", str(tmp_path))
    assert commands[0]["env"]["IMCFLOW_TVM_CODEGEN_DIR"] == str(Path(__file__).resolve().parents[1])


@pytest.mark.parametrize("sample", [None, 0, 7])
def test_rtl_passes_sample_to_run_script_without_noise(tmp_path, monkeypatch, sample):
    runner = module.RTLRunner()
    commands, released = [], []
    monkeypatch.setattr(module.PortAllocator, "get_port_for_test", lambda _: 10042)
    monkeypatch.setattr(module.PortAllocator, "release_port", released.append)
    monkeypatch.setattr(runner, "_stream_command_output", lambda **kwargs: commands.append(kwargs["command"]))
    runner.run("debug_execute_graph", "no", "test", str(tmp_path), sample_idx=sample, noise_mode="greedy")
    command = commands[0]
    assert "greedy" not in command
    if sample is None:
        assert "--sample-idx" not in command
    else:
        assert command[-2:] == ["--sample-idx", str(sample)]
    assert released == [10042]


def test_rtl_rejects_negative_index(tmp_path):
    with pytest.raises(ValueError, match="nonnegative"):
        module.RTLRunner().run("debug_execute_graph", "no", "test", str(tmp_path), sample_idx=-1)
