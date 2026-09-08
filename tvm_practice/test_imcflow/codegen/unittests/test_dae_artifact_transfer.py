"""Exercise board receiver locally; no SSH or board writes occur in these tests."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import subprocess
import struct
import sys
import tarfile

import pytest


def load(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


receiver = load("dae_artifact_receiver")
sender = load("transfer_dae_artifacts")


def manifest_and_payload(files):
    manifest = {name: dict(size=len(data), sha256=hashlib.sha256(data).hexdigest(), executable=False)
                for name, data in files.items()}
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    stream.seek(0)
    return manifest, stream


def test_preserves_old_directory_and_reuses_cached_versions(tmp_path):
    base = tmp_path / "board/workspace"
    target = base / "dataset/toyadmos_eval/_staged"
    target.mkdir(parents=True)
    (target / "old.txt").write_text("keep me")
    first, payload = manifest_and_payload({"images.npy": b"first", "labels.npy": b"labels"})
    assert receiver.receive("probe", base, str(target.relative_to(base)), first)["cached"] is False
    result = receiver.receive("install", base, str(target.relative_to(base)), first, payload)
    assert target.is_symlink()
    assert Path(result["archived"]).joinpath("old.txt").read_text() == "keep me"
    original_artifact = target.resolve()
    second, payload = manifest_and_payload({"images.npy": b"second", "labels.npy": b"labels"})
    receiver.receive("install", base, str(target.relative_to(base)), second, payload)
    assert (target / "images.npy").read_bytes() == b"second"
    assert original_artifact.joinpath("images.npy").read_bytes() == b"first"
    cache_count = len(list((base / ".dae_artifacts").iterdir()))
    result = receiver.receive("probe", base, str(target.relative_to(base)), first)
    assert result["cached"] is True and target.resolve() == original_artifact
    assert len(list((base / ".dae_artifacts").iterdir())) == cache_count


def test_corrupt_cache_is_not_activated(tmp_path):
    base = tmp_path / "board/workspace"
    manifest, payload = manifest_and_payload({"input.bin": b"good"})
    receiver.receive("install", base, "dataset/current", manifest, payload)
    target = base / "dataset/current"
    (target / "input.bin").write_bytes(b"evil")
    with pytest.raises(ValueError, match="content differs"):
        receiver.receive("probe", base, "dataset/current", manifest)


@pytest.mark.parametrize("name", ["../outside", "/absolute", "a/../outside", "manifest.json"])
def test_unsafe_archive_names_rejected(tmp_path, name):
    manifest, payload = manifest_and_payload({name: b"bad"})
    with pytest.raises(ValueError, match="Unsafe"):
        receiver.receive("install", tmp_path / "board/workspace", "dataset/current", manifest, payload)


@pytest.mark.parametrize("destination", ["../outside", "/absolute", "."])
def test_unsafe_destination_rejected(tmp_path, destination):
    manifest, payload = manifest_and_payload({"good": b"data"})
    with pytest.raises(ValueError, match="Unsafe"):
        receiver.receive("install", tmp_path / "board/workspace", destination, manifest, payload)


def test_incomplete_upload_preserves_active_target(tmp_path):
    base = tmp_path / "board/workspace"
    first, payload = manifest_and_payload({"a": b"old"})
    receiver.receive("install", base, "dataset/current", first, payload)
    expected, _ = manifest_and_payload({"a": b"new", "b": b"required"})
    _, incomplete = manifest_and_payload({"a": b"new"})
    with pytest.raises(ValueError, match="Incomplete"):
        receiver.receive("install", base, "dataset/current", expected, incomplete)
    assert (base / "dataset/current/a").read_bytes() == b"old"
    assert list((base / ".dae_artifacts").glob("pending-*"))


def test_sender_manifest_and_password_not_in_command(tmp_path, monkeypatch):
    (tmp_path / "file").write_bytes(b"abc")
    manifest = sender.describe(tmp_path, ["file"])
    assert manifest["file"]["sha256"] == hashlib.sha256(b"abc").hexdigest()
    monkeypatch.setenv("REMOTE_HOST", "board.example")
    monkeypatch.setenv("REMOTE_AUTH_METHOD", "password")
    monkeypatch.setenv("REMOTE_PASSWORD", "test-password-not-for-argv")
    command, env = sender.ssh_command()
    assert env["SSHPASS"] == "test-password-not-for-argv"
    assert "test-password-not-for-argv" not in json.dumps(command)


def test_sender_receiver_wire_protocol_without_ssh(tmp_path, monkeypatch):
    root, base = tmp_path / "source", tmp_path / "board/workspace"
    root.mkdir()
    (root / "file").write_bytes(b"wire protocol payload")
    monkeypatch.setenv("REMOTE_HOST", "not-contacted")
    monkeypatch.setenv("REMOTE_AUTH_METHOD", "key")
    monkeypatch.setenv("REMOTE_BASE_PATH", str(base))
    original_run = subprocess.run
    modes = []

    def local_receiver(command, **kwargs):
        args = shlex.split(command[-1])
        assert args[0:2] == ["python3", "-c"]
        modes.append(args[3])
        return original_run([sys.executable] + args[1:], **kwargs)

    monkeypatch.setattr(sender.subprocess, "run", local_receiver)
    first = sender.transfer(root, ["file"], "dataset/current")
    second = sender.transfer(root, ["file"], "dataset/current")
    assert first["sha256"] == second["sha256"]
    assert modes == ["probe", "install", "probe"]
    assert (base / "dataset/current/file").read_bytes() == b"wire protocol payload"


@pytest.mark.parametrize("function", ["scan_transfer_reg_files", "scan_transfer_executable"])
def test_dae_scan_routes_through_safe_transfer(function):
    codegen = Path(__file__).resolve().parents[1]
    script = ('source ' + shlex.quote(str(codegen / 'scan_steps.sh')) + '\n'
              'IMCFLOW_DAE_SAFE_TRANSFER=1\nNPZ_FILE_PATH=scan_reg_files\n'
              'python3() { printf "%s\\n" "$@"; }\n' + function + ' 3 false\n')
    result = subprocess.run(["bash", "-c", script], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "transfer_dae_artifacts.py" in result.stdout
    assert "transfer_evl.sh" not in result.stdout


def test_rejects_x86_executable_before_board_transfer(tmp_path):
    path = tmp_path / "executable"
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<H", header, 18, 62)
    path.write_bytes(header)
    with pytest.raises(ValueError, match="AArch64"):
        sender.require_aarch64(path)
    struct.pack_into("<H", header, 18, 183)
    path.write_bytes(header)
    sender.require_aarch64(path)
