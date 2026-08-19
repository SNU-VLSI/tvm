#!/usr/bin/env python3
"""Prepare TVM power policy and scope-free measurement requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
MAX_SAMPLES = 50_000
REQUIRED_RESULT_FILES = (
    "request.json",
    "resolved_config.json",
    "session.json",
    "tags.jsonl",
    "summary.json",
)


class ConfigError(ValueError):
    pass


def load_object(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: JSON root must be an object")
    return value


def finite_number(value: Any, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or result < minimum:
        raise ConfigError(f"{label} must be finite and >= {minimum}")
    return result


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(config)
    if normalized.get("schema_version", 1) != 1:
        raise ConfigError("only schema_version=1 is supported")
    normalized["schema_version"] = 1
    enabled = normalized.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigError("enabled must be boolean")
    normalized["enabled"] = enabled
    if not enabled:
        return normalized

    scope = str(normalized.get("scope", "REGION")).upper()
    if scope not in ("MODEL", "REGION", "TILE"):
        raise ConfigError("scope must be MODEL, REGION, or TILE")
    normalized["scope"] = scope
    mode = str(normalized.get("mode", "now")).lower()
    if mode != "now":
        raise ConfigError("power-region measurement supports only mode=now")
    normalized["mode"] = mode
    normalized["duration_budget_s"] = finite_number(
        normalized.get("duration_budget_s", 300),
        "duration_budget_s",
        0.001,
    )

    defaults = normalized.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ConfigError("defaults must be an object")
    rails = normalized.get("rails")
    if not isinstance(rails, list) or not rails:
        raise ConfigError("rails must be a non-empty array")
    seen = set()
    for index, rail in enumerate(rails):
        if not isinstance(rail, dict):
            raise ConfigError(f"rails[{index}] must be an object")
        merged = dict(defaults)
        merged.update(rail)
        name = merged.get("name")
        if not isinstance(name, str) or not name:
            raise ConfigError(f"rails[{index}].name is required")
        if name.upper() in seen:
            raise ConfigError(f"duplicate rail name: {name}")
        seen.add(name.upper())
        count = merged.get("sample_count", MAX_SAMPLES)
        if isinstance(count, bool) or not isinstance(count, int):
            raise ConfigError(f"{name}.sample_count must be an integer")
        if count < 1 or count > MAX_SAMPLES:
            raise ConfigError(
                f"{name}.sample_count must be between 1 and {MAX_SAMPLES}"
            )
        interval = merged.get("sample_interval_s", "auto")
        if isinstance(interval, str):
            if interval.lower() not in ("auto", "min"):
                raise ConfigError(
                    f"{name}.sample_interval_s must be numeric, auto, or MIN"
                )
        else:
            finite_number(interval, f"{name}.sample_interval_s", 0.000001)
        finite_number(merged.get("nplc", 0.001), f"{name}.nplc", 0.00001)
        finite_number(merged.get("voltage_V", 0.0), f"{name}.voltage_V", 0.0)

    metadata = normalized.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ConfigError("metadata must be an object")
    loop = normalized.get("region_loop", {})
    if not isinstance(loop, dict):
        raise ConfigError("region_loop must be an object")
    loop_enable = loop.get("loop_enable", False)
    if not isinstance(loop_enable, bool):
        raise ConfigError("region_loop.loop_enable must be boolean")
    if loop_enable and scope == "TILE":
        raise ConfigError(
            "region_loop.loop_enable is supported only for scope=MODEL or REGION"
        )
    min_samples = loop.get("min_samples", 0)
    if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 0:
        raise ConfigError("region_loop.min_samples must be a non-negative integer")
    if min_samples > min(
        int(dict(defaults, **rail).get("sample_count", MAX_SAMPLES))
        for rail in rails
    ):
        raise ConfigError("region_loop.min_samples exceeds a rail sample_count")
    min_seconds = finite_number(
        loop.get("min_seconds", 0.0), "region_loop.min_seconds", 0.0
    )
    normalized["region_loop"] = {
        "loop_enable": loop_enable,
        "min_samples": min_samples,
        "min_seconds": min_seconds,
    }
    return normalized


def parse_metadata(items: Iterable[str]) -> Dict[str, str]:
    result = {}
    for item in items:
        if "=" not in item:
            raise ConfigError(f"metadata must be KEY=VALUE: {item}")
        key, value = item.split("=", 1)
        if not key:
            raise ConfigError("metadata key cannot be empty")
        result[key] = value
    return result


def atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def prepare(args: argparse.Namespace) -> int:
    config = validate_config(load_object(Path(args.config)))
    if not config["enabled"]:
        print("disabled")
        return 10
    if not SESSION_RE.fullmatch(args.session_id):
        raise ConfigError("unsafe session_id")
    measurement_request = {
        key: value
        for key, value in config.items()
        if key not in ("enabled", "scope", "mode", "region_loop")
    }
    measurement_request["session_id"] = args.session_id
    metadata = dict(measurement_request.get("metadata", {}))
    metadata.update(parse_metadata(args.metadata))
    measurement_request["metadata"] = metadata
    atomic_write_json(Path(args.output), measurement_request)
    print(Path(args.output).resolve())
    return 0


def config_status(args: argparse.Namespace) -> int:
    config = validate_config(load_object(Path(args.config)))
    print("enabled" if config["enabled"] else "disabled")
    return 0 if config["enabled"] else 10


def config_scope(args: argparse.Namespace) -> int:
    config = validate_config(load_object(Path(args.config)))
    print(config.get("scope", "REGION"))
    return 0


def config_loop(args: argparse.Namespace) -> int:
    config = validate_config(load_object(Path(args.config)))
    print(json.dumps(config["region_loop"], separators=(",", ":"), sort_keys=True))
    return 0


def write_tvm_manifest(args: argparse.Namespace) -> int:
    result_dir = Path(args.result_dir)
    regions_dir = result_dir / "regions"
    region_ids = sorted(
        path.name for path in regions_dir.iterdir() if path.is_dir()
    ) if regions_dir.is_dir() else []
    if not region_ids:
        raise ConfigError("cannot write TVM manifest without power regions")
    policy = json.loads(args.region_loop)
    if not isinstance(policy, dict):
        raise ConfigError("region loop policy must be an object")
    manifest = {
        "schema_version": 1,
        "session_id": result_dir.name,
        "scope": str(args.scope).upper(),
        "mode": "now",
        "region_loop": policy,
        "region_ids": region_ids,
    }
    atomic_write_json(result_dir / "tvm_power_manifest.json", manifest)
    print((result_dir / "tvm_power_manifest.json").resolve())
    return 0


def validate_build_identity(args: argparse.Namespace) -> int:
    metadata = load_object(Path(args.metadata))
    actual_tvm = metadata.get("tvm_git_rev")
    actual_measurement = metadata.get("measurement_utils_git_rev")
    dirty = metadata.get("build_tree_dirty")
    if actual_tvm != args.tvm_rev:
        raise ConfigError(
            f"codegen TVM revision mismatch: metadata={actual_tvm!r} "
            f"expected={args.tvm_rev!r}"
        )
    if actual_measurement != args.measurement_rev:
        raise ConfigError(
            "codegen measurement_utils revision mismatch: "
            f"metadata={actual_measurement!r} expected={args.measurement_rev!r}"
        )
    if dirty is not False:
        raise ConfigError("codegen metadata was produced from a dirty tracked tree")
    print(
        "IMCFLOW_POWER_CODEGEN_INFO "
        f"tvm={actual_tvm} measurement_utils={actual_measurement} dirty=0"
    )
    return 0


def load_result(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    missing = [name for name in REQUIRED_RESULT_FILES if not (path / name).is_file()]
    if missing:
        raise ConfigError("missing result files: " + ", ".join(missing))
    summary = load_object(path / "summary.json")
    session = load_object(path / "session.json")
    if summary.get("session_id") != path.name:
        raise ConfigError("summary session_id does not match directory")
    if session.get("session_id") != path.name:
        raise ConfigError("session manifest ID does not match directory")
    if int(summary.get("schema_version", 1)) >= 2:
        for name in ("time_alignment.json", "raw/checksums.json"):
            if not (path / name).is_file():
                raise ConfigError(f"missing result file: {name}")
    return summary, session


def validate_result(args: argparse.Namespace) -> int:
    import numpy as np

    path = Path(args.result_dir)
    summary, _session = load_result(path)
    schema_version = int(summary.get("schema_version", 1))
    checksums = (
        load_object(path / "raw" / "checksums.json")
        if schema_version >= 2
        else {}
    )
    rails = summary.get("rails")
    if not isinstance(rails, dict) or not rails:
        raise ConfigError("summary has no rail results")
    for name, rail_summary in rails.items():
        artifact_path = path / "rails" / f"{name}.npz"
        if not artifact_path.is_file():
            raise ConfigError(f"missing rail artifact: {artifact_path.name}")
        with np.load(artifact_path) as artifact:
            required = (
                "current_A",
                "time_from_trigger_s",
                "power_W",
                "tag_state_id",
            )
            if schema_version >= 2:
                required += (
                    "reading_number",
                    "time_from_first_reading_s",
                    "server_wall_time_ns",
                    "server_monotonic_time_ns",
                    "tag_boundary_ambiguous",
                )
            missing = [key for key in required if key not in artifact]
            if missing:
                raise ConfigError(f"{name}: missing arrays {missing}")
            lengths = {len(artifact[key]) for key in required}
            if len(lengths) != 1 or next(iter(lengths)) < 1:
                raise ConfigError(f"{name}: result arrays are empty or misaligned")
            if int(rail_summary.get("sample_count", -1)) != next(iter(lengths)):
                raise ConfigError(f"{name}: summary sample count mismatch")
            if schema_version >= 2:
                if rail_summary.get("timestamp_source") != "dmm_reading_metadata":
                    raise ConfigError(f"{name}: DMM metadata timestamp source is required")
                checksum = checksums.get(name)
                if not isinstance(checksum, dict):
                    raise ConfigError(f"{name}: raw checksum entry is missing")
                relative_path = checksum.get("path")
                if not isinstance(relative_path, str):
                    raise ConfigError(f"{name}: raw path is invalid")
                raw_path = (path / relative_path).resolve()
                try:
                    raw_path.relative_to(path.resolve())
                except ValueError as exc:
                    raise ConfigError(f"{name}: raw path escapes result directory") from exc
                if not raw_path.is_file():
                    raise ConfigError(f"{name}: raw file is missing")
                actual_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
                if actual_hash != checksum.get("sha256"):
                    raise ConfigError(f"{name}: raw SHA-256 mismatch")
                if raw_path.stat().st_size != int(checksum.get("size", -1)):
                    raise ConfigError(f"{name}: raw size mismatch")
    status = summary.get("status")
    print(
        json.dumps(
            {
                "session_id": summary.get("session_id"),
                "status": status,
                "tag_event_count": summary.get("tag_event_count"),
                "rails": {
                    name: value.get("sample_count") for name, value in rails.items()
                },
            },
            sort_keys=True,
        )
    )
    if status == "complete":
        return 0
    if status == "truncated":
        sample_counts = ", ".join(
            f"{name}={value.get('sample_count')}"
            for name, value in sorted(rails.items())
        )
        print(
            "Warning: power result is truncated; the captured artifact is valid "
            f"and remains usable ({sample_counts})",
            file=sys.stderr,
        )
        return 0
    return 2


def summarize(args: argparse.Namespace) -> int:
    import numpy as np

    summary, _session = load_result(Path(args.result_dir))
    key = None
    value = None
    if args.tag:
        if "=" not in args.tag:
            raise ConfigError("--tag must be KEY=VALUE")
        key, value = args.tag.split("=", 1)
    for rail_name, rail in summary.get("rails", {}).items():
        artifact = None
        if args.exclude_ambiguous:
            artifact_path = Path(args.result_dir) / "rails" / f"{rail_name}.npz"
            artifact = np.load(artifact_path)
            if "tag_boundary_ambiguous" not in artifact:
                artifact.close()
                raise ConfigError(
                    f"{rail_name}: artifact has no ambiguity information"
                )
        print(
            f"[{rail_name}] samples={rail.get('sample_count')} "
            f"ambiguous={rail.get('ambiguous_sample_count', 0)} "
            f"energy_J={rail.get('energy_J')}"
        )
        for state in rail.get("tag_states", []):
            tags = state.get("state", {})
            if key is not None and tags.get(key) != value:
                continue
            values = {
                "sample_count": state.get("sample_count"),
                "average_current_A": state.get("average_current_A"),
                "average_power_W": state.get("average_power_W"),
                "energy_J": state.get("energy_J"),
            }
            if artifact is not None:
                state_id = int(state["tag_state_id"])
                mask = (artifact["tag_state_id"] == state_id) & ~artifact[
                    "tag_boundary_ambiguous"
                ]
                current = artifact["current_A"][mask]
                power = artifact["power_W"][mask]
                interval_s = float(rail.get("actual_sample_interval_s", 0.0))
                values = {
                    "sample_count": int(mask.sum()),
                    "average_current_A": (
                        float(current.mean()) if len(current) else None
                    ),
                    "average_power_W": (
                        float(power.mean()) if len(power) else None
                    ),
                    "energy_J": float(power.sum() * interval_s),
                }
            print(
                "  "
                + json.dumps(tags, ensure_ascii=False, sort_keys=True)
                + f" samples={values['sample_count']}"
                + f" ambiguous={state.get('ambiguous_sample_count', 0)}"
                + f" avg_A={values['average_current_A']}"
                + f" avg_W={values['average_power_W']}"
                + f" energy_J={values['energy_J']}"
            )
        if artifact is not None:
            artifact.close()
    return 0


def plot_timeline(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    result_dir = Path(args.result_dir)
    summary, _session = load_result(result_dir)
    rail_name = args.rail or next(iter(summary.get("rails", {})), None)
    if not rail_name:
        raise ConfigError("result has no rails")
    artifact_path = result_dir / "rails" / f"{rail_name}.npz"
    if not artifact_path.is_file():
        raise ConfigError(f"unknown rail: {rail_name}")
    with np.load(artifact_path) as artifact:
        if "time_from_first_reading_s" in artifact:
            time_s = artifact["time_from_first_reading_s"]
            time_label = "time from first DMM reading (s)"
        else:
            time_s = artifact["time_from_trigger_s"]
            time_label = "time from trigger (s)"
        current = artifact["current_A"]
        power = artifact["power_W"]
        state = artifact["tag_state_id"]
        ambiguous = (
            artifact["tag_boundary_ambiguous"]
            if "tag_boundary_ambiguous" in artifact
            else np.zeros(len(time_s), dtype=np.bool_)
        )

    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    axes[0].plot(time_s, current, linewidth=0.7, label="current_A")
    axes[0].plot(time_s, power, linewidth=0.7, label="power_W", alpha=0.8)
    if ambiguous.any():
        axes[0].scatter(
            time_s[ambiguous],
            current[ambiguous],
            s=8,
            color="red",
            label="ambiguous tag boundary",
        )
    axes[0].set_ylabel("A / W")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)
    axes[1].step(time_s, state, where="post", linewidth=0.8)
    axes[1].set_ylabel("tag_state_id")
    axes[1].set_xlabel(time_label)
    axes[1].grid(alpha=0.25)
    figure.suptitle(f"{summary.get('session_id')} / {rail_name}")
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print(output.resolve())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--config", required=True)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--session-id", required=True)
    prepare_parser.add_argument("--metadata", action="append", default=[])
    prepare_parser.set_defaults(handler=prepare)

    status_parser = subparsers.add_parser("config-status")
    status_parser.add_argument("config")
    status_parser.set_defaults(handler=config_status)

    scope_parser = subparsers.add_parser("config-scope")
    scope_parser.add_argument("config")
    scope_parser.set_defaults(handler=config_scope)

    loop_parser = subparsers.add_parser("config-loop")
    loop_parser.add_argument("config")
    loop_parser.set_defaults(handler=config_loop)

    manifest_parser = subparsers.add_parser("write-tvm-manifest")
    manifest_parser.add_argument("result_dir")
    manifest_parser.add_argument("--scope", required=True)
    manifest_parser.add_argument("--region-loop", required=True)
    manifest_parser.set_defaults(handler=write_tvm_manifest)

    build_parser = subparsers.add_parser("validate-build-identity")
    build_parser.add_argument("--metadata", required=True)
    build_parser.add_argument("--tvm-rev", required=True)
    build_parser.add_argument("--measurement-rev", required=True)
    build_parser.set_defaults(handler=validate_build_identity)

    validate_parser = subparsers.add_parser("validate-result")
    validate_parser.add_argument("result_dir")
    validate_parser.set_defaults(handler=validate_result)

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("result_dir")
    summary_parser.add_argument("--tag")
    summary_parser.add_argument("--exclude-ambiguous", action="store_true")
    summary_parser.set_defaults(handler=summarize)

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("result_dir")
    plot_parser.add_argument("--rail")
    plot_parser.add_argument("--output", required=True)
    plot_parser.set_defaults(handler=plot_timeline)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.handler(args))
    except ConfigError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
