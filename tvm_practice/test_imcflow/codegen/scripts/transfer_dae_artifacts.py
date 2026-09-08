"""Transfer only DAE runtime/staged inputs to a hash-addressed board cache.

No source or destination directory is deleted. Every remote payload is hashed
before activation; a cached payload is verified and reused across iterations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import tarfile
import tempfile


def describe(root, names):
    result = {}
    for name in names:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError("Expected a regular DAE artifact: " + str(path))
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        result[name] = dict(size=path.stat().st_size, sha256=h.hexdigest(), executable=bool(path.stat().st_mode & 0o111))
    return result


def require_aarch64(path):
    with Path(path).open("rb") as stream:
        header = stream.read(64)
    if (len(header) < 20 or header[:6] != b"\x7fELF\x02\x01"
            or struct.unpack_from("<H", header, 18)[0] != 183):
        raise ValueError("Board executable must be little-endian AArch64 ELF: " + str(path))


def ssh_command():
    host = os.environ.get("REMOTE_HOST")
    user = os.environ.get("REMOTE_USER", "root")
    if not host:
        raise ValueError("REMOTE_HOST is required")
    command = ["ssh", "-o", "ConnectTimeout=15", "-p", os.environ.get("REMOTE_PORT", "1326")]
    env = dict(os.environ)
    method = env.get("REMOTE_AUTH_METHOD", "key")
    if method == "password":
        if not env.get("REMOTE_PASSWORD"):
            raise ValueError("REMOTE_PASSWORD is required")
        env["SSHPASS"] = env["REMOTE_PASSWORD"]
        command = ["sshpass", "-e"] + command
    elif method == "key":
        command += ["-o", "BatchMode=yes"]
    else:
        raise ValueError("Unknown REMOTE_AUTH_METHOD")
    command += [host if "@" in host else user + "@" + host]
    return command, env


def transfer(root, names, target):
    manifest = describe(root, names)
    receiver = Path(__file__).with_name("dae_artifact_receiver.py").read_text()
    command, env = ssh_command()
    base = env.get("REMOTE_BASE_PATH", "/home/root/tvm/tvm_practice/test_imcflow/codegen")

    def remote(mode, stream=None):
        args = ["python3", "-c", receiver, mode, base, target, json.dumps(manifest, sort_keys=True)]
        result = subprocess.run(command + [shlex.join(args)], stdin=stream, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, env=env, check=False)
        if result.returncode:
            raise RuntimeError("DAE artifact transfer failed: " + result.stderr.decode(errors="replace"))
        return json.loads(result.stdout)

    result = remote("probe")
    if not result["cached"]:
        with tempfile.TemporaryFile() as payload:
            with tarfile.open(fileobj=payload, mode="w") as archive:
                for name, info in manifest.items():
                    member = tarfile.TarInfo(name)
                    member.size = info["size"]
                    with (root / name).open("rb") as source:
                        archive.addfile(member, source)
            payload.seek(0)
            result = remote("install", payload)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--binary-dir", type=Path)
    mode.add_argument("--scan-reg-dir", type=Path)
    mode.add_argument("--scan-program-dir", type=Path)
    parser.add_argument("--executable", choices=("execute_graph_for_dataset", "debug_execute_graph_for_dataset"))
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--dataset-target")
    args = parser.parse_args()
    if args.scan_reg_dir:
        names = sorted(str(p.relative_to(args.scan_reg_dir)) for p in args.scan_reg_dir.rglob("*") if p.is_file())
        if not names:
            parser.error("Scan register directory is empty")
        transfer(args.scan_reg_dir, names, "scan_gen/" + args.scan_reg_dir.name)
    elif args.scan_program_dir:
        require_aarch64(args.scan_program_dir / "build/program_scan_reg")
        transfer(args.scan_program_dir, ["build/program_scan_reg"], "scan_gen/" + args.scan_program_dir.name)
    else:
        if not args.executable or not args.dataset_dir or not args.dataset_target:
            parser.error("Binary transfer requires executable, dataset-dir and dataset-target")
        require_aarch64(args.binary_dir / "build" / args.executable)
        transfer(args.binary_dir, ["build/" + args.executable, "build/mlf/executor-config/graph/default.graph",
                                  "build/mlf/parameters/default.params"], args.binary_dir.name)
        metadata = "sample_map.json" if (args.dataset_dir / "sample_map.json").exists() else "metadata.json"
        transfer(args.dataset_dir, ["images.npy", "labels.npy", metadata], args.dataset_target)
