"""Tests for the TVM side of RTL BUGFIX/manifest build selection."""

import importlib.util
from pathlib import Path

import pytest

from tvm.contrib.imcflow import bugfix_off_mode, get_imcflow_bugfix_mode


CODEGEN = Path(__file__).resolve().parents[1]
RUNNER_MODULE_PATH = CODEGEN / "runners" / "imcflow_runner.py"


def load_runner_module():
    spec = importlib.util.spec_from_file_location("imcflow_runner_for_test", RUNNER_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_bugfix_mode_defaults_to_off(monkeypatch):
    monkeypatch.delenv("IMCFLOW_BUGFIX", raising=False)
    assert get_imcflow_bugfix_mode() == "off"
    assert bugfix_off_mode()


@pytest.mark.parametrize(("raw", "expected"), [("ON", "on"), (" off ", "off")])
def test_bugfix_mode_normalizes_valid_values(monkeypatch, raw, expected):
    monkeypatch.setenv("IMCFLOW_BUGFIX", raw)
    assert get_imcflow_bugfix_mode() == expected


def test_bugfix_mode_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("IMCFLOW_BUGFIX", "maybe")
    with pytest.raises(ValueError, match="must be 'on' or 'off'"):
        get_imcflow_bugfix_mode()


def test_rtl_runner_uses_shared_imcflow_directory_for_both_modes(tmp_path, monkeypatch):
    runner_module = load_runner_module()
    monkeypatch.delenv("IMCFLOW_RTL_RUNNER_DIR", raising=False)
    monkeypatch.setenv("IMCFLOW_DIR", str(tmp_path))

    expected = tmp_path / "pmap" / "ISA_sim" / "gem5" / "tests" / "imcflow" / "rtl_runner"
    monkeypatch.setenv("IMCFLOW_BUGFIX", "off")
    assert Path(runner_module.RTLRunner().directory_path) == expected
    monkeypatch.setenv("IMCFLOW_BUGFIX", "on")
    assert Path(runner_module.RTLRunner().directory_path) == expected


def test_rtl_setup_delegates_rebuild_decision_to_make(tmp_path, monkeypatch):
    runner_module = load_runner_module()
    monkeypatch.setenv("IMCFLOW_RTL_RUNNER_DIR", str(tmp_path))
    monkeypatch.delenv("IMCFLOW_BUGFIX", raising=False)
    calls = []
    runner = runner_module.RTLRunner()
    runner._stream_command_output = lambda **kwargs: calls.append(kwargs)

    runner.setup()

    assert len(calls) == 1
    assert calls[0]["command"] == [
        "direnv", "exec", ".", "make", "ensure-compiled", "IMCFLOW_BUGFIX=off"
    ]
    assert calls[0]["cwd"] == str(tmp_path)
