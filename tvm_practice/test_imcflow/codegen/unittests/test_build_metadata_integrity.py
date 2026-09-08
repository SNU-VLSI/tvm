"""Exercise metadata writing without importing the complete compiler pipeline."""
import ast
from enum import Enum
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


class RuntimeAccMask(Enum):
    BM_0001 = 1


def test_compile_metadata_records_target_and_artifact_content(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "test.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "save_build_metadata")
    scope = dict(os=os, sys=sys)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)
    monkeypatch.setenv("IMCFLOW_HOST_OS", "linux")
    monkeypatch.setenv("IMCFLOW_HOST_ISA", "arm")
    monkeypatch.setenv("BOARD", "B1")
    monkeypatch.setitem(sys.modules, "tvm.relay.backend.contrib.imcflow.acim_util", SimpleNamespace(
        get_default_vmode=lambda: SimpleNamespace(name="HALF"),
        get_default_acc_mask=lambda: RuntimeAccMask.BM_0001))
    checkpoint, disabled, mlf = tmp_path / "checkpoint", tmp_path / "disabled.json", tmp_path / "lib_graph_system-lib.tar"
    for path in (checkpoint, disabled, mlf):
        path.write_bytes(path.name.encode())
    options = SimpleNamespace(use_v2=True, column_disable_config=str(disabled), num_disable_columns=32,
                              random_seed=42, single_qconv=False, retry_disable=True, max_retry_count=0, with_patch=False)
    scope["save_build_metadata"](str(tmp_path), False, "dae_toycar_full_pretrained", options, str(checkpoint))
    metadata = json.loads((tmp_path / "build_metadata.json").read_text())
    assert (metadata["host_os"], metadata["host_isa"], metadata["acc_mask"]) == ("linux", "arm", 1)
    assert metadata["vmode"] == "HALF"
    for name, path in (("checkpoint", checkpoint), ("column_disable_config", disabled), ("mlf", mlf)):
        assert metadata[name + "_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("model", ["dae_toycar_full_pretrained", "deep_autoencoder_imcflow"])
def test_dae_recompile_preserves_prior_artifacts(tmp_path, model):
    source = Path(__file__).resolve().parents[1] / "test.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "setup_dir")
    scope = dict(os=os, shutil=shutil)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)
    directory = tmp_path / f"{model}_evl.baremetal"
    prior = directory / "test_outputs/sample_0"
    prior.mkdir(parents=True)
    (prior / "output.npy").write_bytes(b"prior regression evidence")
    scope["setup_dir"](str(directory))
    archives = list(tmp_path.glob(directory.name + ".previous.*"))
    assert len(archives) == 1
    assert (archives[0] / "test_outputs/sample_0/output.npy").read_bytes() == b"prior regression evidence"
    assert (directory / "test_outputs").is_dir()
    assert not (directory / "test_outputs/sample_0").exists()
