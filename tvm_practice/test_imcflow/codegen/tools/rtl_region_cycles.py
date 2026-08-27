#!/usr/bin/env python3
"""Extract IMCFlow region or tile timing from RTL co-simulation artifacts.

The canonical path reads FSDB value changes directly through ``fsdb_cli``.
``--method poll`` remains available for the historical region-level estimate,
but poll results cannot be emitted as canonical tile timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence


TOP_SCOPE = "testbench_imcflow_gem5"
BUSY_SIGNAL = f"/{TOP_SCOPE}/imcflow_state_o"
CLOCK_SIGNAL = f"/{TOP_SCOPE}/clk"
IMCU_INPUT_SIGNAL_RE = re.compile(
    r"/core_row\[(\d+)\]/core_col\[(\d+)\]/imce_node/imce/"
    r"u_imce_datapath/bshr/(valid|ready)$"
)

_RESUME = "Reset sequence complete, resuming normal operation"
_TS = re.compile(r"^\[(\d+)\]\s*(.*)")
_REGION_NAME = re.compile(r"imcflow_main_\d+_round_imcflow_(region\d+)")
_POLL_PREFIX = "[SV] Processing READ (REG"
_HOST_SUBSTR = (
    "Processing WRITE (DMEM", "Processing READ (DMEM", "Processing WRITE (IMEM",
    "Processing WRITE (REG", "Asserting reset", "RESET_GEN",
)
_HIGH = {"1", "1'b1", "1'h1"}
_LOW = {"0", "1'b0", "1'h0"}


class TimingInputError(ValueError):
    """Raised when timing artifacts are incomplete or inconsistent."""


def _resolve_log_dir(eval_dir: Path) -> Path:
    if (eval_dir / "vcs_sim.log").is_file():
        return eval_dir
    candidate = eval_dir / "logs" / "rtl_runner"
    if (candidate / "vcs_sim.log").is_file():
        return candidate
    raise FileNotFoundError(
        f"vcs_sim.log not found under {eval_dir} "
        f"(looked in {eval_dir} and {candidate})"
    )


def _resolve_eval_dir(input_path: Path, log_dir: Path) -> Path:
    if input_path.resolve() == log_dir.resolve() and log_dir.parent.name == "logs":
        return log_dir.parent.parent
    return input_path


def _find_fsdb(log_dir: Path) -> Path:
    paths = list(log_dir.glob("*.fsdb"))
    if not paths:
        raise FileNotFoundError(f"no .fsdb in {log_dir}")
    return max(paths, key=lambda path: path.stat().st_size)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _git_dirty(path: Path) -> bool | None:
    try:
        return bool(subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"], check=True,
            capture_output=True, text=True,
        ).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def parse_regions(log_dir: Path) -> tuple[list[int], int]:
    """Return region reset-marker timestamps and final log timestamp."""
    starts: list[int] = []
    final = 0
    with (log_dir / "vcs_sim.log").open(errors="ignore") as stream:
        for line in stream:
            match = _TS.match(line)
            if not match:
                continue
            final = int(match.group(1))
            if _RESUME in match.group(2):
                starts.append(final)
    starts.sort()
    return starts, final


def _diagnose_no_markers(log_dir: Path) -> str:
    hints: list[str] = []
    vcs_text = (log_dir / "vcs_sim.log").read_text(errors="ignore")
    if "Starting transaction processing" in vcs_text and "Processing " not in vcs_text:
        hints.append("co-simulation connected but issued no RTL transactions")
    gem5 = log_dir / "gem5_output.log"
    if gem5.is_file():
        text = gem5.read_text(errors="ignore")
        if "No inputs loaded" in text or "Failed to open metadata file" in text:
            hints.append("gem5 output reports missing model input")
    return ("; ".join(hints) + ". Re-run with complete test_inputs.") if hints else ""


def region_names(log_dir: Path, count: int) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    path = log_dir / "gem5_output.log"
    if path.is_file():
        with path.open(errors="ignore") as stream:
            for line in stream:
                match = _REGION_NAME.search(line)
                if match and match.group(1) not in seen:
                    seen.add(match.group(1))
                    names.append(match.group(1))
    while len(names) < count:
        names.append(f"region{len(names) + 1}")
    return names[:count]


def _load_fsdb_cli() -> tuple[Any, Path]:
    imcflow_dir = Path(os.environ.get("IMCFLOW_DIR", "/root/project/imcflow"))
    tools_dir = imcflow_dir / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    try:
        import fsdb_cli as fsdb  # type: ignore
    except ImportError as exc:
        raise TimingInputError(
            f"cannot import fsdb_cli from {tools_dir}; set IMCFLOW_DIR correctly"
        ) from exc
    return fsdb, tools_dir / "fsdb_cli"


def _walk_signal_paths(node: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    path = "" if getattr(node, "name", "") == "<root>" else f"{prefix}/{node.name}"
    for variable in getattr(node, "vars", []):
        yield f"{path}/{variable.name}", variable
    for child in getattr(node, "children", []):
        yield from _walk_signal_paths(child, path)


def validate_signal_paths(paths: Sequence[str]) -> tuple[str, list[str], list[str]]:
    """Return RUN plus the 16 IMCU-input valid/ready handshake pairs."""
    busy = sorted(path for path in paths if path == BUSY_SIGNAL)
    valid: list[str] = []
    ready: list[str] = []
    coordinates: dict[str, set[tuple[int, int]]] = {"valid": set(), "ready": set()}
    for path in paths:
        match = IMCU_INPUT_SIGNAL_RE.search(path)
        if match is None:
            continue
        kind = match.group(3)
        (valid if kind == "valid" else ready).append(path)
        assert match is not None
        coordinates[kind].add((int(match.group(1)), int(match.group(2))))
    expected = {(row, col) for row in range(4) for col in range(1, 5)}
    if (len(busy) != 1 or len(valid) != 16 or len(ready) != 16
            or coordinates["valid"] != expected or coordinates["ready"] != expected):
        raise TimingInputError(
            "FSDB signal discovery failed closed: expected RUN=1 and "
            "IMCU input valid/ready=16/16, found "
            f"RUN={len(busy)}, valid={len(valid)}, ready={len(ready)}; "
            f"RUN candidates={busy!r}; valid candidates={valid!r}; "
            f"ready candidates={ready!r}"
        )
    return busy[0], sorted(valid), sorted(ready)


def discover_signal_paths(fsdb_path: Path, fsdb: Any) -> tuple[str, list[str], list[str]]:
    """Resolve exact signal hierarchy with fsdb_cli and validate cardinality."""
    leaf_candidates = fsdb.find_signals(
        str(fsdb_path), r"^(imcflow_state_o|valid|ready)$"
    )
    if not leaf_candidates:
        raise TimingInputError("fsdb_cli.find_signals found no anchor candidates")
    hierarchy = fsdb.hierarchy(str(fsdb_path), scopes_only=False)
    paths = [path for path, _variable in _walk_signal_paths(hierarchy)]
    try:
        return validate_signal_paths(paths)
    except TimingInputError as exc:
        candidate_names = sorted({candidate.name for candidate in leaf_candidates})
        raise TimingInputError(f"{exc}; leaf candidates={candidate_names!r}") from exc


def _is_high(value: str | None) -> bool:
    return value is not None and value.strip().lower() in _HIGH


def _is_low(value: str | None) -> bool:
    return value is not None and value.strip().lower() in _LOW


def rising_edges(events: Sequence[Any], signal: str) -> list[int]:
    """Return valid 0-to-1 transition timestamps, ignoring X/Z transitions."""
    previous: str | None = None
    result: list[int] = []
    for event in events:
        value = event.values.get(signal)
        if _is_high(value) and _is_low(previous):
            result.append(int(event.time))
        if _is_high(value) or _is_low(value):
            previous = value
    return result


def high_intervals(events: Sequence[Any], signal: str) -> list[tuple[int, int]]:
    """Return closed-open high intervals, ignoring unknown transitions."""
    previous: str | None = None
    start: int | None = None
    intervals: list[tuple[int, int]] = []
    for event in events:
        value = event.values.get(signal)
        if _is_high(value) and _is_low(previous):
            start = int(event.time)
        elif _is_low(value) and _is_high(previous) and start is not None:
            intervals.append((start, int(event.time)))
            start = None
        if _is_high(value) or _is_low(value):
            previous = value
    if start is not None:
        raise TimingInputError(f"{signal} is high at the end of the FSDB report")
    return intervals


def _time_unit_seconds(unit: str) -> float:
    units = {
        "s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9,
        "ps": 1e-12, "fs": 1e-15,
    }
    normalized = unit.strip().lower()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)?\s*(fs|ps|ns|us|ms|s)", normalized)
    if match is None:
      raise TimingInputError(f"unsupported or missing FSDB report time unit {unit!r}")
    multiplier = float(match.group(1) or 1.0)
    return multiplier * units[match.group(2)]


def _clock_period_units(events: Sequence[Any], signal: str) -> float:
    rises = rising_edges(events, signal)
    if len(rises) < 3:
        raise TimingInputError(f"not enough {signal} edges to verify RTL clock")
    periods = [right - left for left, right in zip(rises, rises[1:])]
    result = float(median(periods))
    if result <= 0 or any(abs(period - result) > 1 for period in periods):
        raise TimingInputError(f"non-uniform RTL clock periods: {periods[:20]!r}")
    return result


def _coordinates(signal_path: str) -> list[int]:
    match = IMCU_INPUT_SIGNAL_RE.search(signal_path)
    if match is None:
        raise TimingInputError(f"cannot parse IMCU coordinate from {signal_path}")
    return [int(match.group(1)), int(match.group(2)) - 1]


def tile_intervals_from_events(
    events: Sequence[Any], busy_signal: str, valid_signals: Sequence[str],
    ready_signals: Sequence[str],
) -> list[dict[str, Any]]:
    """Attach the first accepted IMCU input, if any, to every RUN interval."""
    if len(valid_signals) != len(ready_signals):
        raise TimingInputError("IMCU input valid/ready signal counts differ")
    intervals = high_intervals(events, busy_signal)
    pairs: list[tuple[str, str]] = []
    for valid, ready in zip(valid_signals, ready_signals):
        if _coordinates(valid) != _coordinates(ready):
            raise TimingInputError("sorted IMCU input valid/ready coordinates differ")
        pairs.append((valid, ready))

    runs: list[dict[str, Any]] = []
    for start, end in intervals:
        first: int | None = None
        first_valids: list[str] = []
        for event in events:
            timestamp = int(event.time)
            if timestamp < start:
                continue
            if timestamp >= end:
                break
            accepted = [valid for valid, ready in pairs
                        if _is_high(event.values.get(valid))
                        and _is_high(event.values.get(ready))]
            if accepted:
                first = timestamp
                first_valids = accepted
                break
        runs.append({
            "run_state_start_time_units": start,
            "any_imcu_input_handshake_time_units": first,
            "first_imcu_coordinates": [_coordinates(path) for path in first_valids],
            "run_state_end_time_units": end,
        })
    return runs


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise TimingInputError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TimingInputError(f"{path} must contain a JSON object")
    return value


def _manifest_regions(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    regions = manifest.get("regions")
    if not isinstance(regions, list) or not regions:
        raise TimingInputError("tile manifest must contain a non-empty regions list")
    for expected_index, region in enumerate(regions, 1):
        if not isinstance(region, dict):
            raise TimingInputError("tile manifest region entries must be objects")
        if region.get("region_index") != expected_index:
            raise TimingInputError("tile manifest region indexes must be contiguous from 1")
        if not isinstance(region.get("tile_count"), int) or region["tile_count"] < 1:
            raise TimingInputError("tile manifest tile_count must be a positive integer")
    return regions


def _metadata(eval_dir: Path) -> dict[str, Any]:
    path = eval_dir / "build_metadata.json"
    return _load_json(path) if path.is_file() else {}


def _bool_bugfix(metadata: dict[str, Any], eval_dir: Path) -> bool | None:
    if "imcflow_bugfix" in metadata:
        return bool(metadata["imcflow_bugfix"])
    name = eval_dir.name.lower()
    if "bugfixoff" in name:
        return False
    if "bugfixon" in name:
        return True
    return None


def build_tile_document(
    *, eval_dir: Path, fsdb_path: Path, fsdb_cli_root: Path,
    busy_signal: str, imcu_valid_signals: Sequence[str],
    imcu_ready_signals: Sequence[str], report_time_unit: str,
    period_units: float, runs: Sequence[dict[str, Any]], region_starts: Sequence[int],
    region_final: int, names: Sequence[str], manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest_regions = _manifest_regions(manifest)
    if len(region_starts) != len(manifest_regions):
        raise TimingInputError(
            f"region count mismatch: log={len(region_starts)}, manifest={len(manifest_regions)}"
        )
    region_ends = list(region_starts[1:]) + [region_final]
    assigned_runs: list[list[dict[str, Any]]] = [[] for _ in manifest_regions]
    for run in runs:
        start = run["run_state_start_time_units"]
        matches = [index for index, (left, right) in enumerate(zip(region_starts, region_ends))
                   if left <= start < right]
        if len(matches) != 1:
            raise TimingInputError(f"RUN at {start} does not map to exactly one region")
        assigned_runs[matches[0]].append(dict(run))

    unit_s = _time_unit_seconds(report_time_unit)
    rtl_clock_hz = 1.0 / (period_units * unit_s)
    regions: list[dict[str, Any]] = []
    setup_intervals: list[tuple[int, int]] = []
    total_cycles = 0
    for index, (manifest_region, region_runs) in enumerate(
        zip(manifest_regions, assigned_runs), 1
    ):
        expected = manifest_region["tile_count"]
        if len(region_runs) < expected:
            raise TimingInputError(
                f"tile count mismatch in region {index}: FSDB RUN={len(region_runs)}, "
                f"manifest={expected}"
            )
        # Each region emits setup/policy RUN pulses first and tile execution RUNs
        # last.  IMCU input handshakes cannot be used to classify tiles because
        # depthwise-convolution tiles legitimately bypass the IMCU array.
        setup_runs = region_runs[:-expected]
        region_tiles = region_runs[-expected:]
        setup_intervals.extend(
            (run["run_state_start_time_units"], run["run_state_end_time_units"])
            for run in setup_runs
        )
        rendered_tiles = []
        for tile_index, tile in enumerate(region_tiles):
            start_cycle = round(tile["run_state_start_time_units"] / period_units)
            handshake_units = tile["any_imcu_input_handshake_time_units"]
            handshake_cycle = (round(handshake_units / period_units)
                               if handshake_units is not None else None)
            end_cycle = round(tile["run_state_end_time_units"] / period_units)
            run_cycles = end_cycle - start_cycle
            imcu_cycles = (end_cycle - handshake_cycle
                           if handshake_cycle is not None else None)
            if handshake_cycle is not None and not (start_cycle <= handshake_cycle < end_cycle):
                raise TimingInputError(
                    f"tile {index}/{tile_index} violates start <= IMCU input handshake < end"
                )
            total_cycles += run_cycles
            rendered_tiles.append({
                "tile_index": tile_index,
                "run_state_start_cycle": start_cycle,
                "any_imcu_input_handshake_cycle": handshake_cycle,
                "first_imcu_coordinates": tile["first_imcu_coordinates"],
                "run_state_end_cycle": end_cycle,
                "imcu_input_delay_cycles": (handshake_cycle - start_cycle
                                            if handshake_cycle is not None else None),
                "run_state_cycles": run_cycles,
                "imcu_to_run_end_cycles": imcu_cycles,
                "run_state_time_s": run_cycles / rtl_clock_hz,
                "imcu_to_run_end_time_s": (imcu_cycles / rtl_clock_hz
                                           if imcu_cycles is not None else None),
            })
        regions.append({
            "region_index": index,
            "region": names[index - 1],
            "function": manifest_region.get("function"),
            "tiles": rendered_tiles,
        })

    build = _metadata(eval_dir)
    imcflow_root = fsdb_cli_root.parent.parent
    tvm_root = Path(__file__).resolve().parents[4]
    return {
        "schema_version": 2,
        "model": manifest.get("model") or build.get("model_name"),
        "rtl_method": "fsdb_cli",
        "fsdb_path": str(fsdb_path.resolve()),
        "fsdb_sha256": _sha256(fsdb_path),
        "fsdb_time_unit": report_time_unit,
        "fsdb_cli_root": str(fsdb_cli_root.resolve()),
        "fsdb_cli_revision": _git_revision(imcflow_root),
        "busy_signal": busy_signal,
        "imcu_input_valid_signals": list(imcu_valid_signals),
        "imcu_input_ready_signals": list(imcu_ready_signals),
        "imcu_input_anchor_semantics": "first cycle with bshr.valid && bshr.ready",
        "rtl_clock_hz": rtl_clock_hz,
        "rtl_revision": _git_revision(imcflow_root),
        "tvm_revision": _git_revision(tvm_root),
        "measurement_utils_revision": _git_revision(tvm_root / "3rdparty/measurement_utils"),
        "rtl_dirty": _git_dirty(imcflow_root),
        "tvm_dirty": _git_dirty(tvm_root),
        "measurement_utils_dirty": _git_dirty(tvm_root / "3rdparty/measurement_utils"),
        "board": build.get("board"),
        # A current manifest may intentionally be paired with an older but
        # codegen-identical FSDB. Prefer its board-independent identity fields
        # over stale/missing build metadata from the RTL eval directory.
        "checkpoint_alias": manifest.get("checkpoint_alias", build.get("checkpoint_alias")),
        "dataset": manifest.get("dataset", build.get("dataset")),
        "sample_index": manifest.get("sample_index", build.get("sample_index")),
        "random_seed": manifest.get("random_seed", build.get("random_seed")),
        "imcflow_bugfix": manifest.get(
            "imcflow_bugfix", _bool_bugfix(build, eval_dir)),
        "codegen_fingerprint": manifest.get("codegen_fingerprint") or build.get("codegen_fingerprint"),
        "tile_manifest_sha256": hashlib.sha256(
            (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest(),
        "regions": regions,
        "total_run_state_cycles": total_cycles,
        "total_run_state_time_s": total_cycles / rtl_clock_hz,
        "excluded_setup_policy_run_intervals": [list(value) for value in setup_intervals],
    }


def extract_fsdb_tile_timing(eval_dir: Path, manifest_path: Path) -> dict[str, Any]:
    log_dir = _resolve_log_dir(eval_dir)
    resolved_eval_dir = _resolve_eval_dir(eval_dir, log_dir)
    region_starts, final = parse_regions(log_dir)
    if not region_starts:
        detail = _diagnose_no_markers(log_dir)
        raise TimingInputError(
            f"no region markers in {log_dir}/vcs_sim.log" + (f": {detail}" if detail else "")
        )
    names = region_names(log_dir, len(region_starts))
    fsdb_path = _find_fsdb(log_dir)
    fsdb, fsdb_cli_root = _load_fsdb_cli()
    busy_signal, imcu_valid_signals, imcu_ready_signals = discover_signal_paths(
        fsdb_path, fsdb)
    clock_report = fsdb.report(str(fsdb_path), [CLOCK_SIGNAL], et="200ns")
    period_units = _clock_period_units(clock_report.events(), CLOCK_SIGNAL)
    signal_report = fsdb.report(
        str(fsdb_path),
        [busy_signal, *imcu_valid_signals, *imcu_ready_signals],
    )
    if signal_report.time_unit != clock_report.time_unit:
        raise TimingInputError(
            "FSDB reports returned inconsistent time units: "
            f"{clock_report.time_unit!r} vs {signal_report.time_unit!r}"
        )
    runs = tile_intervals_from_events(
        signal_report.events(), busy_signal, imcu_valid_signals, imcu_ready_signals)
    return build_tile_document(
        eval_dir=resolved_eval_dir, fsdb_path=fsdb_path, fsdb_cli_root=fsdb_cli_root,
        busy_signal=busy_signal, imcu_valid_signals=imcu_valid_signals,
        imcu_ready_signals=imcu_ready_signals,
        report_time_unit=signal_report.time_unit, period_units=period_units,
        runs=runs, region_starts=region_starts,
        region_final=final, names=names, manifest=_load_json(manifest_path),
    )


def extract_fsdb_region_document(eval_dir: Path) -> dict[str, Any]:
    """Return the legacy all-RUN-pulse region aggregate through fsdb_cli."""
    log_dir = _resolve_log_dir(eval_dir)
    starts_ps, final_ps = parse_regions(log_dir)
    if not starts_ps:
        raise TimingInputError(f"no region markers in {log_dir}/vcs_sim.log")
    names = region_names(log_dir, len(starts_ps))
    fsdb_path = _find_fsdb(log_dir)
    fsdb, _fsdb_cli_root = _load_fsdb_cli()
    candidates = fsdb.find_signals(str(fsdb_path), r"^imcflow_state_o$")
    if not candidates:
        raise TimingInputError(
            "fsdb_cli.find_signals found no imcflow_state_o candidate"
        )
    clock_report = fsdb.report(str(fsdb_path), [CLOCK_SIGNAL], et="200ns")
    period_units = _clock_period_units(clock_report.events(), CLOCK_SIGNAL)
    report = fsdb.report(str(fsdb_path), [BUSY_SIGNAL])
    if report.time_unit != clock_report.time_unit:
        raise TimingInputError("clock and RUN reports use different time units")
    unit_s = _time_unit_seconds(report.time_unit)
    rtl_clock_hz = 1.0 / (period_units * unit_s)
    ps_to_units = 1e-12 / unit_s
    starts = [round(value * ps_to_units) for value in starts_ps]
    ends = starts[1:] + [round(final_ps * ps_to_units)]
    intervals = high_intervals(report.events(), BUSY_SIGNAL)
    rows = []
    for name, start, end in zip(names, starts, ends):
        active_units = sum(
            min(right, end) - max(left, start)
            for left, right in intervals if min(right, end) > max(left, start)
        )
        pulses = sum(
            1 for left, right in intervals if min(right, end) > max(left, start)
        )
        active_s = active_units * unit_s
        rows.append({
            "region": name,
            "busy_ns": active_s * 1e9,
            "busy_cycles": round(active_s * rtl_clock_hz),
            "pulses": pulses,
        })
    busy_ns = sum(row["busy_ns"] for row in rows)
    return {
        "method": "fsdb_cli",
        "source": fsdb_path.name,
        "clk_mhz": rtl_clock_hz / 1e6,
        "regions": rows,
        "total": {
            "busy_us": busy_ns / 1000.0,
            "busy_cycles": sum(row["busy_cycles"] for row in rows),
        },
    }


def _poll_class(rest: str) -> str:
    if rest.startswith(_POLL_PREFIX):
        return "POLL"
    if rest.startswith("[SRAM_DIRECT]") or rest.startswith("[SV] Read data"):
        return "HOST"
    if any(value in rest for value in _HOST_SUBSTR):
        return "HOST"
    return ""


def busy_from_poll(log_dir: Path, start: int, end: int) -> float:
    sequence: list[tuple[int, str]] = []
    with (log_dir / "vcs_sim.log").open(errors="ignore") as stream:
        for line in stream:
            match = _TS.match(line)
            if not match:
                continue
            timestamp = int(match.group(1))
            if start <= timestamp < end:
                category = _poll_class(match.group(2))
                if category:
                    sequence.append((timestamp, category))
    if not sequence:
        return float(end - start)
    compute = 0.0
    for (timestamp, category), (next_timestamp, _next) in zip(sequence, sequence[1:]):
        if category == "POLL":
            compute += next_timestamp - timestamp
    if sequence[0][1] == "POLL":
        compute += sequence[0][0] - start
    if sequence[-1][1] == "POLL":
        compute += end - sequence[-1][0]
    return compute


def poll_region_document(eval_dir: Path, clk_mhz: float) -> dict[str, Any]:
    log_dir = _resolve_log_dir(eval_dir)
    starts, final = parse_regions(log_dir)
    if not starts:
        raise TimingInputError(f"no region markers in {log_dir}/vcs_sim.log")
    names = region_names(log_dir, len(starts))
    ends = starts[1:] + [final]
    ps_per_cycle = 1e6 / clk_mhz
    rows = []
    for name, start, end in zip(names, starts, ends):
        busy = busy_from_poll(log_dir, start, end)
        rows.append({
            "region": name, "busy_ns": busy / 1000.0,
            "busy_cycles": round(busy / ps_per_cycle), "pulses": None,
        })
    busy_ns = sum(row["busy_ns"] for row in rows)
    return {
        "method": "poll", "source": "vcs_sim.log (poll estimate)",
        "clk_mhz": clk_mhz, "regions": rows,
        "total": {"busy_us": busy_ns / 1000.0,
                  "busy_cycles": round(busy_ns * 1000 / ps_per_cycle)},
    }


def _region_summary_from_tiles(document: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for region in document["regions"]:
        cycles = sum(tile["run_state_cycles"] for tile in region["tiles"])
        rows.append({
            "region": region["region"],
            "busy_ns": cycles / document["rtl_clock_hz"] * 1e9,
            "busy_cycles": cycles, "pulses": len(region["tiles"]),
        })
    return {
        "method": "fsdb_cli", "source": Path(document["fsdb_path"]).name,
        "clk_mhz": document["rtl_clock_hz"] / 1e6, "regions": rows,
        "total": {"busy_us": document["total_run_state_time_s"] * 1e6,
                  "busy_cycles": document["total_run_state_cycles"]},
    }


def _print_region_table(document: dict[str, Any]) -> None:
    print(f"Method: {document['method']}")
    print(f"Source: {document['source']}")
    print(f"Accelerator clock: {document['clk_mhz']:g} MHz\n")
    header = f"{'Region':<10}{'busy(us)':>11}{'busy cyc':>11}{'#pulses':>9}"
    print(header)
    print("-" * len(header))
    for row in document["regions"]:
        pulses = "-" if row["pulses"] is None else str(row["pulses"])
        print(f"{row['region']:<10}{row['busy_ns']/1000:>11.2f}"
              f"{row['busy_cycles']:>11d}{pulses:>9}")
    print("-" * len(header))
    total = document["total"]
    print(f"{'TOTAL':<10}{total['busy_us']:>11.2f}{total['busy_cycles']:>11d}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_dir", type=Path)
    parser.add_argument("--method", choices=("fsdb_cli", "fsdb", "poll"),
                        default="fsdb_cli",
                        help="fsdb/fsdb_cli use canonical fsdb_cli; poll is estimate-only")
    parser.add_argument("--granularity", choices=("region", "tile"), default="region")
    parser.add_argument("--manifest", type=Path, help="tile manifest JSON")
    parser.add_argument("--output", type=Path, help="write JSON to this path")
    parser.add_argument("--clk-mhz", type=float, default=100.0, help="poll method only")
    parser.add_argument("--json", action="store_true", help="emit JSON to stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.method == "poll":
            if args.granularity == "tile":
                raise TimingInputError("poll method cannot produce canonical tile timing")
            output_document = poll_region_document(args.eval_dir, args.clk_mhz)
        else:
            if args.granularity == "region":
                output_document = extract_fsdb_region_document(args.eval_dir)
            else:
                log_dir = _resolve_log_dir(args.eval_dir)
                eval_dir = _resolve_eval_dir(args.eval_dir, log_dir)
                manifest = args.manifest or eval_dir / "tile_manifest.json"
                if not manifest.is_file():
                    raise TimingInputError(
                        f"tile manifest not found: {manifest}; recompile with manifest support"
                    )
                output_document = extract_fsdb_tile_timing(args.eval_dir, manifest)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output_document, indent=2, sort_keys=True) + "\n")
        if args.json or args.granularity == "tile":
            print(json.dumps(output_document, indent=2, sort_keys=True))
        else:
            _print_region_table(output_document)
        return 0
    except (OSError, TimingInputError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
