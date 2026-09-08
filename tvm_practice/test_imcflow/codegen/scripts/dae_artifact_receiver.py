"""Board-side stdlib-only immutable artifact receiver (Python 3.8 compatible).

Used through SSH stdin by transfer_dae_artifacts.py. Never deletes directories.
Existing non-managed targets are renamed; managed symlink replacement preserves
every artifact version. An interrupted upload leaves a distinct pending directory.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
import uuid


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Empty artifact manifest")
    for name, info in manifest.items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or str(path) != name or name == "manifest.json":
            raise ValueError("Unsafe artifact member name")
        if info["size"] < 0 or len(info["sha256"]) != 64:
            raise ValueError("Invalid artifact size/hash")


def verify(directory, manifest):
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file() or p.is_symlink()}
    if actual != set(manifest) | {"manifest.json"}:
        raise ValueError("Cached artifact members differ")
    for name, info in manifest.items():
        path = directory / name
        if path.is_symlink() or directory.resolve() not in path.resolve().parents:
            raise ValueError("Symlink in cached artifact")
        if path.stat().st_size != info["size"] or digest_file(path) != info["sha256"]:
            raise ValueError("Cached artifact content differs: " + name)
    if json.loads((directory / "manifest.json").read_text()) != manifest:
        raise ValueError("Cached manifest differs")


def receive(mode, base, relative_target, manifest, stream=None):
    validate_manifest(manifest)
    base = Path(base).resolve()
    relative = PurePosixPath(relative_target)
    if (len(base.parts) < 5 or relative.is_absolute() or ".." in relative.parts
            or len(relative.parts) < 1 or str(relative) in (".", "")):
        raise ValueError("Unsafe artifact target")
    target = base / relative_target
    if target.parent.resolve() != base and base not in target.parent.resolve().parents:
        raise ValueError("Artifact target escapes the board workspace")
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    cache = base / ".dae_artifacts"
    if cache.is_symlink():
        raise ValueError("Artifact cache must not be a symlink")
    artifact = cache / key
    if artifact.is_symlink():
        raise ValueError("Cached artifact must not be a symlink")
    if mode == "probe" and not artifact.exists():
        return {"cached": False, "sha256": key}
    if mode == "install":
        if artifact.exists():
            raise FileExistsError("Artifact already installed; probe before upload")
        cache.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix="pending-", dir=str(cache)))
        found = set()
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            for member in archive:
                if member.name not in manifest or member.name in found or not member.isfile():
                    raise ValueError("Unexpected, duplicate or non-regular archive member")
                if member.size != manifest[member.name]["size"]:
                    raise ValueError("Archive member size differs")
                path = stage / member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, path.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
                path.chmod(0o755 if manifest[member.name].get("executable") else 0o644)
                found.add(member.name)
        if found != set(manifest):
            raise ValueError("Incomplete artifact upload")
        with (stage / "manifest.json").open("x") as output:
            json.dump(manifest, output, sort_keys=True)
        verify(stage, manifest)
        stage.rename(artifact)
    elif mode != "probe":
        raise ValueError("Unknown receiver mode")
    verify(artifact, manifest)
    target.parent.mkdir(parents=True, exist_ok=True)
    archived = None
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == artifact:
            return {"cached": True, "sha256": key, "target": str(target), "archived": None}
        managed = target.is_symlink() and target.resolve().parent == cache
        if not managed:
            archived = str(target) + ".previous." + uuid.uuid4().hex
            target.rename(archived)
    link = target.with_name(target.name + ".link." + uuid.uuid4().hex)
    link.symlink_to(artifact, target_is_directory=True)
    os.replace(str(link), str(target))
    return {"cached": True, "sha256": key, "target": str(target), "archived": archived}


if __name__ == "__main__":
    mode, base, target, payload = sys.argv[1:]
    print(json.dumps(receive(mode, base, target, json.loads(payload), sys.stdin.buffer)))
