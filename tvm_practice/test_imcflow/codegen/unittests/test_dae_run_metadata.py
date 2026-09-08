import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("save_dae_run_metadata",
    Path(__file__).resolve().parents[1] / "scripts/save_dae_run_metadata.py")
metadata_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metadata_tool)


def prepare(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    staged = tmp_path / "staged"
    staged.mkdir()
    np.save(source / "images.npy", np.zeros((4, 640), dtype=np.float32))
    np.save(source / "labels.npy", np.array([0, 0, 1, 1]))
    np.save(staged / "images.npy", np.zeros((2, 640), dtype=np.float32))
    np.save(staged / "labels.npy", np.array([1, 0]))
    (source / "metadata.json").write_text(json.dumps(dict(shape=[4, 640], input_domain="before_x_f_1",
        images_sha256=metadata_tool.sha256(source / "images.npy"), labels_sha256=metadata_tool.sha256(source / "labels.npy"))))
    (staged / "sample_map.json").write_text(json.dumps(dict(source_dir=str(source), source_num_samples=4,
        num_staged=2, staged_to_original=[3, 0])))
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fixture")
    build = tmp_path / "build.json"
    build.write_text(json.dumps(dict(model_name="dae_toycar_full_pretrained", board="B1", driver_v2=True,
        num_disable_columns=32, acc_mask=1, checkpoint_path=str(checkpoint))))
    run_id = "ab" * 16
    result = tmp_path / "result.txt"
    result.write_text("DAE_RUN " + run_id + "\n")
    return result, source, staged, build, run_id, "chip4"


def test_preserves_stage_map_and_hashes(tmp_path):
    args = prepare(tmp_path)
    report = metadata_tool.save_metadata(*args)
    saved = json.loads(Path(str(args[0]) + ".sample_map.json").read_text())
    assert saved["staged_to_original"] == [3, 0]
    assert report["result_sha256"] == metadata_tool.sha256(args[0])


def test_stale_run_result_is_rejected(tmp_path):
    args = list(prepare(tmp_path))
    args[4] = "cd" * 16
    with pytest.raises(ValueError, match="stale"):
        metadata_tool.save_metadata(*args)


def test_wrong_staged_data_is_rejected(tmp_path):
    args = prepare(tmp_path)
    np.save(args[2] / "labels.npy", np.array([0, 1]))
    with pytest.raises(ValueError, match="Staged labels"):
        metadata_tool.save_metadata(*args)


def test_unknown_acc_mask_is_rejected(tmp_path):
    args = prepare(tmp_path)
    data = json.loads(args[3].read_text())
    del data["acc_mask"]
    args[3].write_text(json.dumps(data))
    with pytest.raises(ValueError, match="ACC_MASK"):
        metadata_tool.save_metadata(*args)
