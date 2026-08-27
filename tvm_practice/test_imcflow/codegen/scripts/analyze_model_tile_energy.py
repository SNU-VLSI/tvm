#!/usr/bin/env python3
"""Align MODEL-scope DMM traces to RTL tile windows and integrate energy."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


RAIL_ALIASES = {
    "VDD": ("VDD", "DMM_GPIB1"),
    "DDA": ("DDA", "DMM_GPIB2"),
    "DDC": ("DDC", "DMM_GPIB4"),
}
EXPECTED_ANCHORS = {
    "VDD": "run_state",
    "DDA": "any_imcu_input_handshake",
    "DDC": "run_state",
}
IDENTITY_FIELDS = (
    ("model_name", "model"),
    ("checkpoint_alias", "checkpoint_alias"),
    ("dataset", "dataset"),
    ("sample_index", "sample_index"),
    ("random_seed", "random_seed"),
    ("imcflow_bugfix", "imcflow_bugfix"),
    ("codegen_fingerprint", "codegen_fingerprint"),
    ("tvm_revision", "tvm_revision"),
    ("measurement_utils_revision", "measurement_utils_revision"),
    ("imcflow_revision", "rtl_revision"),
    ("tvm_dirty", "tvm_dirty"),
    ("measurement_utils_dirty", "measurement_utils_dirty"),
    ("imcflow_dirty", "rtl_dirty"),
)


class AnalysisInputError(ValueError):
    """Raised when analysis would be ambiguous or unreproducible."""


@dataclass(frozen=True)
class Candidate:
    start_sample: int
    end_sample: int
    width_samples: int
    score_a: float
    baseline_a: float
    high_threshold_a: float
    low_threshold_a: float
    priority: str = "primary"


def load_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise AnalysisInputError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise AnalysisInputError(f"{path} must contain a JSON object")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_single_capture(path: Path) -> np.ndarray:
    captures: list[np.ndarray] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = ast.literal_eval(line)
            except (SyntaxError, ValueError) as exc:
                raise AnalysisInputError(f"{path}:{line_number}: invalid sample list") from exc
            if not isinstance(record, (list, tuple)):
                raise AnalysisInputError(f"{path}:{line_number}: expected a sample list")
            values = np.asarray(record, dtype=np.float64)
            if values.ndim != 1 or values.size == 0:
                raise AnalysisInputError(f"{path}:{line_number}: empty/non-vector capture")
            if not np.all(np.isfinite(values)):
                raise AnalysisInputError(f"{path}:{line_number}: NaN/Inf sample")
            captures.append(values)
    if len(captures) != 1:
        raise AnalysisInputError(
            f"{path}: expected exactly one capture, found {len(captures)}"
        )
    return captures[0]


def moving_median(values: Sequence[float], window: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if window < 1 or window % 2 == 0:
        raise AnalysisInputError("median_window must be a positive odd integer")
    if window == 1:
        return array.copy()
    radius = window // 2
    padded = np.pad(array, (radius, radius), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(windows, axis=1)


def _rolling_prior_median(values: np.ndarray, window: int) -> np.ndarray:
    if window < 3:
        raise AnalysisInputError("baseline_window must be at least 3")
    result = np.empty_like(values)
    initial = float(np.median(values[: min(window, values.size)]))
    result[:window] = initial
    # Traces are small (~50k); the explicit loop avoids a large temporary array.
    for index in range(window, values.size):
        result[index] = np.median(values[index - window:index])
    return result


def detector_threshold(values: np.ndarray, settings: dict[str, Any]) -> tuple[float, float]:
    count = min(int(settings["baseline_samples"]), values.size)
    baseline_values = values[:count]
    center = float(np.median(baseline_values))
    mad = float(np.median(np.abs(baseline_values - center)))
    robust_sigma = 1.4826 * mad
    rise = max(
        float(settings["absolute_min_rise_a"]),
        float(settings["mad_multiplier"]) * robust_sigma,
    )
    return rise, robust_sigma


def detect_rising_candidates(
    values: Sequence[float], settings: dict[str, Any], threshold_scale: float = 1.0
) -> tuple[list[Candidate], dict[str, float]]:
    """Detect local low-to-high transitions with median/MAD hysteresis."""
    smoothed = moving_median(values, int(settings["median_window"]))
    baseline = _rolling_prior_median(smoothed, int(settings["baseline_window"]))
    rise, robust_sigma = detector_threshold(smoothed, settings)
    rise *= threshold_scale
    low_rise = rise * float(settings["hysteresis_low_ratio"])
    delta = smoothed - baseline
    minimum_width = int(settings["minimum_width_samples"])

    candidates: list[Candidate] = []
    start: int | None = None
    peak = -math.inf
    for index, value in enumerate(delta):
        if start is None and value >= rise:
            start = index
            peak = float(value)
        elif start is not None:
            peak = max(peak, float(value))
            if value <= low_rise:
                width = index - start
                priority = "primary" if width >= minimum_width else "short_spike"
                candidates.append(Candidate(
                    start_sample=start,
                    end_sample=index,
                    width_samples=width,
                    score_a=peak,
                    baseline_a=float(baseline[start]),
                    high_threshold_a=float(baseline[start] + rise),
                    low_threshold_a=float(baseline[start] + low_rise),
                    priority=priority,
                ))
                start = None
                peak = -math.inf
    if start is not None:
        width = len(delta) - start
        candidates.append(Candidate(
            start_sample=start,
            end_sample=len(delta),
            width_samples=width,
            score_a=peak,
            baseline_a=float(baseline[start]),
            high_threshold_a=float(baseline[start] + rise),
            low_threshold_a=float(baseline[start] + low_rise),
            priority="primary" if width >= minimum_width else "short_spike",
        ))

    merged: list[Candidate] = []
    merge_gap = int(settings["merge_gap_samples"])
    for candidate in candidates:
        if merged and candidate.start_sample - merged[-1].end_sample <= merge_gap:
            previous = merged[-1]
            winner = candidate if candidate.score_a > previous.score_a else previous
            merged[-1] = Candidate(
                start_sample=previous.start_sample,
                end_sample=candidate.end_sample,
                width_samples=candidate.end_sample - previous.start_sample,
                score_a=max(previous.score_a, candidate.score_a),
                baseline_a=previous.baseline_a,
                high_threshold_a=previous.high_threshold_a,
                low_threshold_a=previous.low_threshold_a,
                priority=winner.priority,
            )
        else:
            merged.append(candidate)
    return merged, {"rise_delta_a": rise, "robust_sigma_a": robust_sigma,
                    "low_rise_delta_a": low_rise}


def _tile_key(region_index: int, tile_index: int) -> str:
    return f"region{region_index:02d}_tile{tile_index:02d}"


def flatten_tiles(timing: dict[str, Any]) -> list[dict[str, Any]]:
    regions = timing.get("regions")
    if (timing.get("schema_version") != 2 or timing.get("rtl_method") != "fsdb_cli"
            or not isinstance(regions, list)):
        raise AnalysisInputError("RTL timing must be schema-v2 fsdb_cli tile timing")
    tiles: list[dict[str, Any]] = []
    for expected_region, region in enumerate(regions, 1):
        if region.get("region_index") != expected_region:
            raise AnalysisInputError("RTL timing region indexes are not contiguous")
        for expected_tile, tile in enumerate(region.get("tiles", [])):
            if tile.get("tile_index") != expected_tile:
                raise AnalysisInputError("RTL timing tile indexes are not contiguous")
            start = tile.get("run_state_start_cycle")
            imcu = tile.get("any_imcu_input_handshake_cycle")
            end = tile.get("run_state_end_cycle")
            if not isinstance(start, int) or not isinstance(end, int):
                raise AnalysisInputError("RTL RUN anchors must be integer cycles")
            if imcu is not None and not isinstance(imcu, int):
                raise AnalysisInputError("RTL IMCU input anchor must be integer or null")
            if not start < end or (imcu is not None and not start <= imcu < end):
                raise AnalysisInputError("RTL tile violates RUN/IMCU anchor ordering")
            tiles.append({
                "region_index": expected_region,
                "tile_index": expected_tile,
                "function": region.get("function"),
                **tile,
            })
    if not tiles:
        raise AnalysisInputError("RTL timing contains no tiles")
    return tiles


def validate_analysis_config(config: dict[str, Any], require_clock: bool = True) -> None:
    if config.get("schema_version") != 1:
        raise AnalysisInputError("analysis config schema_version must be 1")
    clock = config.get("chip_clock_hz")
    if require_clock and (isinstance(clock, bool) or not isinstance(clock, (int, float))
                          or not math.isfinite(clock) or clock <= 0):
        raise AnalysisInputError(
            "chip_clock_hz is unknown; detector-only mode is available, but RUN "
            "charge/energy/TOPS/W cannot be calculated"
        )
    rails = config.get("rails")
    if not isinstance(rails, dict) or set(rails) != set(EXPECTED_ANCHORS):
        raise AnalysisInputError("analysis config rails must be exactly VDD, DDA, DDC")
    for rail, expected_anchor in EXPECTED_ANCHORS.items():
        entry = rails[rail]
        if entry.get("rtl_anchor") != expected_anchor:
            raise AnalysisInputError(
                f"{rail}.rtl_anchor must be {expected_anchor!r}"
            )
        voltage = entry.get("voltage_v")
        if voltage is not None and (
            isinstance(voltage, bool) or not isinstance(voltage, (int, float))
            or not math.isfinite(voltage) or voltage <= 0
        ):
            raise AnalysisInputError(f"{rail}.voltage_v must be null or positive")
    dmm_config = config.get("dmm_config")
    if dmm_config is not None and (not isinstance(dmm_config, str) or not dmm_config.strip()):
        raise AnalysisInputError("dmm_config must be null or a non-empty path")
    if config.get("missing_peak_policy", "error") not in ("error", "zero_energy"):
        raise AnalysisInputError("missing_peak_policy must be error or zero_energy")
    if config.get("all_chip_power") not in (True, False, None):
        raise AnalysisInputError("all_chip_power must be boolean or null")
    detector = config.get("peak_detector")
    required = {
        "median_window", "baseline_window", "baseline_samples", "mad_multiplier",
        "hysteresis_low_ratio", "minimum_width_samples", "merge_gap_samples",
        "absolute_min_rise_a", "threshold_variation_fraction",
        "alignment_tolerance_samples",
    }
    if not isinstance(detector, dict) or not required.issubset(detector):
        raise AnalysisInputError(f"peak_detector is missing {sorted(required - set(detector or {}))}")


def resolve_rail_voltages(config_path: Path, config: dict[str, Any]) -> dict[str, float | None]:
    """Resolve voltages from rail overrides or the measurement DMM config."""
    dmm_power: dict[str, Any] = {}
    dmm_config = config.get("dmm_config")
    if dmm_config:
        path = Path(dmm_config)
        if not path.is_absolute():
            path = config_path.parent / path
        if not path.is_file():
            raise AnalysisInputError(f"DMM config does not exist: {path}")
        document = load_json(path)
        power = document.get("POWER")
        if not isinstance(power, dict):
            raise AnalysisInputError(f"DMM config POWER must be an object: {path}")
        dmm_power = power

    result: dict[str, float | None] = {}
    for rail in RAIL_ALIASES:
        configured = config["rails"][rail].get("voltage_v")
        if configured is None and rail in dmm_power:
            entry = dmm_power[rail]
            if not isinstance(entry, dict):
                raise AnalysisInputError(f"DMM config POWER.{rail} must be an object")
            configured = entry.get("VOLTAGE_V", entry.get("voltage_v"))
        if configured is not None and (
                isinstance(configured, bool) or not isinstance(configured, (int, float))
                or not math.isfinite(configured) or configured <= 0):
            raise AnalysisInputError(f"{rail} voltage must be null or positive")
        result[rail] = float(configured) if configured is not None else None
    return result


def mac_counting_from_environment() -> int:
    raw = os.getenv("IMCFLOW_MAC_COUNTING", "1").strip()
    if raw not in ("1", "2"):
        raise AnalysisInputError("IMCFLOW_MAC_COUNTING must be 1 or 2")
    return int(raw)


def validate_identity(build: dict[str, Any], timing: dict[str, Any]) -> None:
    errors = []
    for build_key, timing_key in IDENTITY_FIELDS:
        left = build.get(build_key)
        right = timing.get(timing_key)
        if left is None or right is None:
            errors.append(f"missing {build_key}/{timing_key}")
        elif left != right:
            errors.append(f"{build_key}: power={left!r}, RTL={right!r}")
    if errors:
        raise AnalysisInputError("power/RTL identity mismatch: " + "; ".join(errors))


def resolve_traces(power_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    build_path = power_dir / "build_metadata.json"
    metadata_path = power_dir / "power_metadata.json"
    if not build_path.is_file() or not metadata_path.is_file():
        raise AnalysisInputError("power run needs build_metadata.json and power_metadata.json")
    build = load_json(build_path)
    settings = build.get("power_measurement", {})
    if settings.get("scope") != "MODEL":
        raise AnalysisInputError("power run scope must be MODEL")
    metadata = load_json(metadata_path)
    configured_names = settings.get("dmm_names")
    observed_names = metadata.get("dmm_names")
    if not isinstance(configured_names, list) or observed_names != configured_names:
        raise AnalysisInputError(
            "build_metadata power DMM names and power_metadata DMM names differ"
        )
    trace_entries = metadata.get("traces")
    if not isinstance(trace_entries, list):
        raise AnalysisInputError("power_metadata.json traces must be a list")

    result: dict[str, dict[str, Any]] = {}
    used_paths: set[Path] = set()
    for rail, aliases in RAIL_ALIASES.items():
        matches = [entry for entry in trace_entries
                   if isinstance(entry, dict) and entry.get("dmm_name") in aliases]
        if len(matches) != 1:
            raise AnalysisInputError(
                f"{rail}: expected exactly one raw trace via aliases {aliases}, found {len(matches)}"
            )
        entry = matches[0]
        raw_path = power_dir / str(entry.get("raw_file"))
        if not raw_path.is_file() or raw_path in used_paths:
            raise AnalysisInputError(f"{rail}: raw path missing or duplicated: {raw_path}")
        used_paths.add(raw_path)
        tag_name = entry.get("tag_file")
        tag_path = power_dir / str(tag_name) if tag_name else Path(f"{raw_path}.tags.json")
        if not tag_path.is_file():
            raise AnalysisInputError(f"{rail}: tag sidecar missing: {tag_path}")
        tags = load_json(tag_path)
        interval_ns = tags.get("sample_interval_ns")
        if isinstance(interval_ns, bool) or not isinstance(interval_ns, int) or interval_ns <= 0:
            raise AnalysisInputError(f"{rail}: invalid sample_interval_ns")
        values = load_single_capture(raw_path)
        actual = tags.get("actual_sample_count")
        if actual != len(values):
            raise AnalysisInputError(
                f"{rail}: sidecar actual_sample_count={actual}, raw={len(values)}"
            )
        result[rail] = {
            "logical_name_in_capture": entry.get("dmm_name"),
            "raw_path": raw_path,
            "raw_sha256": sha256_file(raw_path),
            "tag_path": tag_path,
            "tag_sha256": sha256_file(tag_path),
            "tags": tags,
            "sample_interval_s": interval_ns * 1e-9,
            "values": values,
        }
    return result, build


def load_overrides(path: Path | None, tiles: Sequence[dict[str, Any]]) -> dict[str, dict[int, int]]:
    if path is None:
        return {rail: {} for rail in RAIL_ALIASES}
    document = load_json(path)
    tile_lookup = {
        _tile_key(tile["region_index"], tile["tile_index"]): index
        for index, tile in enumerate(tiles)
    }
    result = {rail: {} for rail in RAIL_ALIASES}
    for rail, entries in document.items():
        if rail not in result or not isinstance(entries, dict):
            raise AnalysisInputError(f"invalid override rail {rail!r}")
        for key, sample in entries.items():
            if key not in tile_lookup or isinstance(sample, bool) or not isinstance(sample, int) or sample < 0:
                raise AnalysisInputError(f"invalid override {rail}.{key}={sample!r}")
            result[rail][tile_lookup[key]] = sample
    return result


def map_starts(
    candidates: Sequence[Candidate], tile_count: int, overrides: dict[int, int]
) -> tuple[list[dict[str, Any]], list[Candidate]]:
    needed = tile_count - len(overrides)
    primaries = [candidate for candidate in candidates if candidate.priority == "primary"]
    if len(primaries) < needed:
        raise AnalysisInputError(
            f"only {len(primaries)} direct peaks for {needed} non-overridden tiles"
        )
    # Preserve every candidate, but rank by local rise so slow CPU/setup ramps
    # do not displace the expected accelerator pulses.
    selected = sorted(sorted(primaries, key=lambda item: item.score_a, reverse=True)[:needed],
                      key=lambda item: item.start_sample)
    manual_samples = set(overrides.values())
    if len(manual_samples) != len(overrides):
        raise AnalysisInputError("manual override samples must be unique within a rail")
    combined = [
        {"start_sample": candidate.start_sample, "method": "direct", "candidate": candidate}
        for candidate in selected
    ] + [
        {"start_sample": sample, "method": "manual", "tile_slot": slot}
        for slot, sample in overrides.items()
    ]
    combined.sort(key=lambda item: item["start_sample"])
    if len(combined) != tile_count:
        raise AnalysisInputError("peak mapping did not produce the RTL tile count")
    for slot, sample in overrides.items():
        if combined[slot].get("method") != "manual" or combined[slot]["start_sample"] != sample:
            raise AnalysisInputError(
                f"manual override for tile slot {slot} conflicts with chronological mapping"
            )
    selected_ids = {id(item) for item in selected}
    rejected = [item for item in candidates if id(item) not in selected_ids]
    return combined, rejected


def map_starts_with_zero_fill(
    candidates: Sequence[Candidate], tile_count: int, overrides: dict[int, int],
    reference_samples: Sequence[float], tolerance_samples: float,
) -> tuple[list[dict[str, Any]], list[Candidate]]:
    """Align a partial rail sequence to a complete rail; unmatched tiles are zero."""
    if len(reference_samples) != tile_count or tolerance_samples <= 0:
        raise AnalysisInputError("invalid reference peak sequence/tolerance")
    primaries = [candidate for candidate in candidates if candidate.priority == "primary"]
    selected = sorted(
        sorted(primaries, key=lambda item: item.score_a, reverse=True)[:tile_count],
        key=lambda item: item.start_sample,
    )
    mappings: list[dict[str, Any] | None] = [None] * tile_count
    for slot, sample in overrides.items():
        mappings[slot] = {"start_sample": sample, "method": "manual", "tile_slot": slot}

    free_slots = [slot for slot in range(tile_count) if mappings[slot] is None]
    if selected and free_slots:
        hypotheses = [candidate.start_sample - reference_samples[slot]
                      for candidate in selected for slot in free_slots]
        best: tuple[int, float, list[tuple[int, Candidate]]] | None = None
        for offset in hypotheses:
            unused = set(range(len(selected)))
            matches: list[tuple[int, Candidate]] = []
            residual = 0.0
            for slot in free_slots:
                expected = reference_samples[slot] + offset
                choices = [(abs(selected[index].start_sample - expected), index)
                           for index in unused]
                if not choices:
                    continue
                distance, index = min(choices)
                if distance <= tolerance_samples:
                    candidate = selected[index]
                    unused.remove(index)
                    matches.append((slot, candidate))
                    residual += distance
            score = (len(matches), -residual)
            if best is None or score > (best[0], -best[1]):
                best = (len(matches), residual, matches)
        if best is not None:
            for slot, candidate in best[2]:
                mappings[slot] = {
                    "start_sample": candidate.start_sample,
                    "method": "direct",
                    "candidate": candidate,
                }

    used_ids = {id(mapping["candidate"]) for mapping in mappings
                if mapping is not None and "candidate" in mapping}
    rejected = [item for item in candidates if id(item) not in used_ids]
    return [mapping if mapping is not None else {
        "start_sample": None, "method": "missing_peak_zero"
    } for mapping in mappings], rejected


def fractional_integral(
    values: Sequence[float], sample_interval_s: float, start_time_s: float,
    duration_s: float, baseline_a: float = 0.0,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Integrate piecewise-constant samples using exact boundary overlap."""
    if sample_interval_s <= 0 or duration_s <= 0 or start_time_s < 0:
        raise AnalysisInputError("integration interval must be positive and in range")
    end_time_s = start_time_s + duration_s
    if end_time_s > len(values) * sample_interval_s + 1e-18:
        raise AnalysisInputError("integration window exceeds raw trace")
    first = int(math.floor(start_time_s / sample_interval_s))
    last = int(math.ceil(end_time_s / sample_interval_s))
    gross = 0.0
    dynamic = 0.0
    records: list[dict[str, Any]] = []
    for index in range(first, min(last, len(values))):
        sample_start = index * sample_interval_s
        sample_end = sample_start + sample_interval_s
        overlap = max(0.0, min(sample_end, end_time_s) - max(sample_start, start_time_s))
        if overlap <= 0:
            continue
        current = float(values[index])
        gross += current * overlap
        dynamic += (current - baseline_a) * overlap
        records.append({
            "sample_index": index,
            "current_a": current,
            "overlap_s": overlap,
            "boundary_weight": overlap / sample_interval_s,
        })
    return gross, dynamic, records


def _baseline_before(values: np.ndarray, start: int, count: int) -> float:
    left = max(0, start - count)
    if left == start:
        return float(np.median(values[: min(count, len(values))]))
    return float(np.median(values[left:start]))


def _candidate_json(candidate: Candidate) -> dict[str, Any]:
    return {
        "start_sample": candidate.start_sample,
        "end_sample": candidate.end_sample,
        "width_samples": candidate.width_samples,
        "score_a": candidate.score_a,
        "baseline_a": candidate.baseline_a,
        "high_threshold_a": candidate.high_threshold_a,
        "low_threshold_a": candidate.low_threshold_a,
        "priority": candidate.priority,
    }


def detector_report(
    traces: dict[str, dict[str, Any]], tile_count: int, settings: dict[str, Any]
) -> dict[str, Any]:
    result = {}
    for rail, trace in traces.items():
        candidates, threshold = detect_rising_candidates(trace["values"], settings)
        result[rail] = {
            "sample_count": len(trace["values"]),
            "sample_interval_s": trace["sample_interval_s"],
            "expected_tile_count": tile_count,
            "candidate_count": len(candidates),
            "threshold": threshold,
            "candidates": [_candidate_json(candidate) for candidate in candidates],
            "enough_primary_candidates":
                len([candidate for candidate in candidates if candidate.priority == "primary"]) >= tile_count,
        }
    return result


def _sensitivity(
    values: np.ndarray, interval: float, start_sample: int, duration: float,
    baseline: float, voltage: float | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for offset in (-1, 1):
        shifted = start_sample + offset
        if shifted < 0 or (shifted * interval + duration) > len(values) * interval:
            result[str(offset)] = None
            continue
        gross, dynamic, _records = fractional_integral(
            values, interval, shifted * interval, duration, baseline
        )
        result[str(offset)] = {
            "start_sample": shifted,
            "gross_charge_c": gross,
            "dynamic_charge_c": dynamic,
            "gross_energy_j": gross * voltage if voltage is not None else None,
            "dynamic_energy_j": dynamic * voltage if voltage is not None else None,
        }
    return result


def _write_plot(
    path: Path, traces: dict[str, dict[str, Any]], rail_results: dict[str, Any]
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(15, 9), squeeze=False)
    colors = plt.get_cmap("tab10")
    for axis, rail in zip(axes[:, 0], RAIL_ALIASES):
        values = traces[rail]["values"]
        axis.plot(np.arange(len(values)), values, linewidth=0.65, color="black")
        for index, window in enumerate(rail_results[rail]["tiles"]):
            start = window["start_sample"]
            end = window["end_sample_float"]
            color = colors(index % 10)
            if start is None or end is None:
                continue
            axis.axvline(start, color=color, linewidth=0.9,
                         linestyle="--" if window["detection_method"] == "direct" else ":")
            axis.axvspan(start, end, color=color, alpha=0.18)
            axis.text(start, 0.96, window["tile_key"], transform=axis.get_xaxis_transform(),
                      rotation=90, va="top", ha="right", fontsize=7, color=color)
        for rejected in rail_results[rail]["rejected_candidates"]:
            axis.axvline(rejected["start_sample"], color="tab:red", linewidth=0.5, alpha=0.4)
        axis.set_title(rail)
        axis.set_ylabel("current (A)")
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("rail-local sample index")
    figure.suptitle("MODEL trace: independently detected RTL RUN windows")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def analyze(
    power_dir: Path, timing_path: Path, config_path: Path,
    override_path: Path | None = None,
) -> dict[str, Any]:
    timing = load_json(timing_path)
    config = load_json(config_path)
    validate_analysis_config(config, require_clock=True)
    tiles = flatten_tiles(timing)
    traces, build = resolve_traces(power_dir)
    validate_identity(build, timing)
    overrides = load_overrides(override_path, tiles)
    clock = float(config["chip_clock_hz"])
    settings = config["peak_detector"]
    voltages = resolve_rail_voltages(config_path, config)
    detected = {
        rail: detect_rising_candidates(traces[rail]["values"], settings)
        for rail in RAIL_ALIASES
    }
    reference_rail = None
    reference_mapping = None
    for rail in ("VDD", "DDC", "DDA"):
        try:
            reference_mapping, _unused = map_starts(
                detected[rail][0], len(tiles), overrides[rail])
            reference_rail = rail
            break
        except AnalysisInputError:
            continue
    if reference_rail is None or reference_mapping is None:
        raise AnalysisInputError(
            "at least one rail must detect every tile to locate missing peaks on other rails")
    reference_times_s = [
        float(mapping["start_sample"]) * traces[reference_rail]["sample_interval_s"]
        for mapping in reference_mapping
    ]
    missing_policy = config.get("missing_peak_policy", "error")
    tolerance = float(settings.get("alignment_tolerance_samples", 200))
    mappings: dict[str, tuple[list[dict[str, Any]], list[Candidate]]] = {}
    for rail in RAIL_ALIASES:
        if rail == reference_rail:
            mappings[rail] = map_starts(detected[rail][0], len(tiles), overrides[rail])
        elif missing_policy == "zero_energy":
            reference_samples = [time_s / traces[rail]["sample_interval_s"]
                                 for time_s in reference_times_s]
            mappings[rail] = map_starts_with_zero_fill(
                detected[rail][0], len(tiles), overrides[rail],
                reference_samples, tolerance)
        else:
            mappings[rail] = map_starts(
                detected[rail][0], len(tiles), overrides[rail])
    rail_results: dict[str, Any] = {}
    sample_documents: dict[str, dict[str, Any]] = {}

    total_gross_energy = 0.0
    total_dynamic_energy = 0.0
    total_gross_charge = 0.0
    total_dynamic_charge = 0.0
    total_energy_complete = True
    for rail in RAIL_ALIASES:
        trace = traces[rail]
        values = trace["values"]
        candidates, threshold = detected[rail]
        mapped, rejected = mappings[rail]
        voltage = voltages[rail]
        include = bool(config["rails"][rail].get("include_in_total"))
        rail_tiles = []
        for slot, (tile, mapping) in enumerate(zip(tiles, mapped)):
            anchor = config["rails"][rail]["rtl_anchor"]
            cycles = (tile["run_state_cycles"] if anchor == "run_state"
                      else tile["imcu_to_run_end_cycles"])
            rtl_start = (tile["run_state_start_cycle"] if anchor == "run_state"
                         else tile["any_imcu_input_handshake_cycle"])
            missing_reason = None
            if mapping["start_sample"] is None:
                missing_reason = "power_peak_not_detected"
            elif cycles is None or rtl_start is None:
                missing_reason = "rtl_imcu_input_handshake_absent"
            key = _tile_key(tile["region_index"], tile["tile_index"])
            if missing_reason is not None:
                entry = {
                    "tile_key": key,
                    "region_index": tile["region_index"],
                    "tile_index": tile["tile_index"],
                    "rtl_anchor": anchor,
                    "rtl_start_cycle": rtl_start,
                    "rtl_end_cycle": tile["run_state_end_cycle"],
                    "duration_cycles": cycles,
                    "chip_duration_s": (cycles / clock if cycles is not None else None),
                    "start_sample": None,
                    "end_sample_float": None,
                    "detection_method": "missing_peak_zero",
                    "zero_energy_reason": missing_reason,
                    "baseline_current_a": None,
                    "gross_charge_c": 0.0,
                    "dynamic_charge_c": 0.0,
                    "voltage_v": voltage,
                    "gross_energy_j": 0.0,
                    "dynamic_energy_j": 0.0,
                    "first_boundary_weight": None,
                    "last_boundary_weight": None,
                    "sample_artifact": None,
                    "start_sample_sensitivity": None,
                }
                rail_tiles.append(entry)
                continue

            assert cycles is not None
            duration = cycles / clock
            start_sample = int(mapping["start_sample"])
            baseline = _baseline_before(values, start_sample, int(settings["baseline_samples"]))
            gross_charge, dynamic_charge, records = fractional_integral(
                values, trace["sample_interval_s"],
                start_sample * trace["sample_interval_s"], duration, baseline,
            )
            gross_energy = gross_charge * voltage if voltage is not None else None
            dynamic_energy = dynamic_charge * voltage if voltage is not None else None
            sample_document = {
                "schema_version": 1,
                "rail": rail,
                "tile_key": key,
                "raw_file": trace["raw_path"].name,
                "raw_sha256": trace["raw_sha256"],
                "sample_interval_s": trace["sample_interval_s"],
                "start_sample": start_sample,
                "duration_s": duration,
                "samples": records,
            }
            sample_name = f"{key}_{rail}.json"
            sample_documents[sample_name] = sample_document
            entry = {
                "tile_key": key,
                "region_index": tile["region_index"],
                "tile_index": tile["tile_index"],
                "rtl_anchor": anchor,
                "rtl_start_cycle": rtl_start,
                "rtl_end_cycle": tile["run_state_end_cycle"],
                "duration_cycles": cycles,
                "chip_duration_s": duration,
                "start_sample": start_sample,
                "end_sample_float": start_sample + duration / trace["sample_interval_s"],
                "detection_method": mapping["method"],
                "baseline_current_a": baseline,
                "gross_charge_c": gross_charge,
                "dynamic_charge_c": dynamic_charge,
                "voltage_v": voltage,
                "gross_energy_j": gross_energy,
                "dynamic_energy_j": dynamic_energy,
                "first_boundary_weight": records[0]["boundary_weight"],
                "last_boundary_weight": records[-1]["boundary_weight"],
                "sample_artifact": f"run_only_samples/{sample_name}",
                "start_sample_sensitivity": _sensitivity(
                    values, trace["sample_interval_s"], start_sample, duration,
                    baseline, voltage,
                ),
            }
            rail_tiles.append(entry)
            if include:
                total_gross_charge += gross_charge
                total_dynamic_charge += dynamic_charge
                if gross_energy is None:
                    total_energy_complete = False
                else:
                    total_gross_energy += gross_energy
                    total_dynamic_energy += float(dynamic_energy)

        for left, right in zip(rail_tiles, rail_tiles[1:]):
            if (left["end_sample_float"] is not None and right["start_sample"] is not None
                    and left["end_sample_float"] > right["start_sample"]):
                raise AnalysisInputError(
                    f"{rail}: integration windows overlap: {left['tile_key']} ends at "
                    f"{left['end_sample_float']}, {right['tile_key']} starts at "
                    f"{right['start_sample']}"
                )

        variation = {}
        fraction = float(settings["threshold_variation_fraction"])
        for label, scale in (("lower", 1.0 - fraction), ("upper", 1.0 + fraction)):
            varied, varied_threshold = detect_rising_candidates(values, settings, scale)
            try:
                if rail != reference_rail and missing_policy == "zero_energy":
                    reference_samples = [time_s / trace["sample_interval_s"]
                                         for time_s in reference_times_s]
                    varied_map, _unused = map_starts_with_zero_fill(
                        varied, len(tiles), overrides[rail], reference_samples, tolerance)
                else:
                    varied_map, _unused = map_starts(
                        varied, len(tiles), overrides[rail])
                varied_tiles = []
                for tile, varied_mapping in zip(tiles, varied_map):
                    anchor = config["rails"][rail]["rtl_anchor"]
                    cycles = (tile["run_state_cycles"] if anchor == "run_state"
                              else tile["imcu_to_run_end_cycles"])
                    if varied_mapping["start_sample"] is None or cycles is None:
                        varied_tiles.append({
                            "tile_key": _tile_key(tile["region_index"], tile["tile_index"]),
                            "start_sample": None,
                            "gross_charge_c": 0.0,
                            "dynamic_charge_c": 0.0,
                            "gross_energy_j": 0.0,
                            "dynamic_energy_j": 0.0,
                            "method": "missing_peak_zero",
                        })
                        continue
                    duration = cycles / clock
                    varied_start = int(varied_mapping["start_sample"])
                    varied_baseline = _baseline_before(
                        values, varied_start, int(settings["baseline_samples"]))
                    gross, dynamic, _records = fractional_integral(
                        values, trace["sample_interval_s"],
                        varied_start * trace["sample_interval_s"], duration,
                        varied_baseline,
                    )
                    varied_tiles.append({
                        "tile_key": _tile_key(tile["region_index"], tile["tile_index"]),
                        "start_sample": varied_start,
                        "gross_charge_c": gross,
                        "dynamic_charge_c": dynamic,
                        "gross_energy_j": gross * voltage if voltage is not None else None,
                        "dynamic_energy_j": dynamic * voltage if voltage is not None else None,
                    })
                variation[label] = {
                    "threshold_scale": scale,
                    "threshold": varied_threshold,
                    "start_samples": [item.get("start_sample") for item in varied_map],
                    "tiles": varied_tiles,
                }
            except AnalysisInputError as exc:
                variation[label] = {"threshold_scale": scale, "error": str(exc)}

        rail_results[rail] = {
            "logical_name_in_capture": trace["logical_name_in_capture"],
            "raw_file": trace["raw_path"].name,
            "raw_sha256": trace["raw_sha256"],
            "tag_file": trace["tag_path"].name,
            "tag_sha256": trace["tag_sha256"],
            "actual_sample_count": len(values),
            "sample_interval_s": trace["sample_interval_s"],
            "independent_time_origin": True,
            "detector_threshold": threshold,
            "all_candidates": [_candidate_json(value) for value in candidates],
            "rejected_candidates": [_candidate_json(value) for value in rejected],
            "threshold_variation": variation,
            "tiles": rail_tiles,
        }

    mac_count = build.get("conv_mac_count")
    if mac_count is not None and (
            isinstance(mac_count, bool) or not isinstance(mac_count, int) or mac_count <= 0):
        raise AnalysisInputError("build_metadata conv_mac_count must be a positive integer")
    mac_counting = mac_counting_from_environment()
    operations = mac_count * mac_counting if mac_count is not None else None
    tops_per_w = None
    if operations is not None:
        if total_energy_complete and total_gross_energy > 0:
            tops_per_w = operations / total_gross_energy / 1e12

    all_chip_power = config.get("all_chip_power") is True
    total_name = "total_chip_energy_j" if all_chip_power else "measured_rail_energy_j"
    tile_totals = []
    for slot, tile in enumerate(tiles):
        included = [rail_results[rail]["tiles"][slot] for rail in RAIL_ALIASES
                    if config["rails"][rail].get("include_in_total")]
        complete = all(entry["gross_energy_j"] is not None for entry in included)
        tile_totals.append({
            "tile_key": _tile_key(tile["region_index"], tile["tile_index"]),
            "region_index": tile["region_index"],
            "tile_index": tile["tile_index"],
            "gross_charge_c_sum_across_rails": sum(entry["gross_charge_c"] for entry in included),
            "dynamic_charge_c_sum_across_rails": sum(entry["dynamic_charge_c"] for entry in included),
            total_name: (sum(entry["gross_energy_j"] for entry in included)
                         if complete else None),
            ("total_chip_dynamic_energy_j" if all_chip_power
             else "measured_rail_dynamic_energy_j"):
                (sum(entry["dynamic_energy_j"] for entry in included)
                 if complete else None),
        })
    region_totals = []
    for region_index in sorted({tile["region_index"] for tile in tiles}):
        members = [entry for entry in tile_totals if entry["region_index"] == region_index]
        complete = all(entry[total_name] is not None for entry in members)
        region_totals.append({
            "region_index": region_index,
            "gross_charge_c_sum_across_rails":
                sum(entry["gross_charge_c_sum_across_rails"] for entry in members),
            "dynamic_charge_c_sum_across_rails":
                sum(entry["dynamic_charge_c_sum_across_rails"] for entry in members),
            total_name: (sum(entry[total_name] for entry in members) if complete else None),
        })
    totals = {
        "gross_charge_c_sum_across_rails": total_gross_charge,
        "dynamic_charge_c_sum_across_rails": total_dynamic_charge,
        total_name: total_gross_energy if total_energy_complete else None,
        ("total_chip_dynamic_energy_j" if all_chip_power
         else "measured_rail_dynamic_energy_j"):
            total_dynamic_energy if total_energy_complete else None,
        ("tops_per_w" if all_chip_power else "measured_rails_tops_per_w"): tops_per_w,
        "conv_mac_count": mac_count,
        "operation_count": operations,
        "mac_counting": mac_counting,
        "mac_counting_source": "IMCFLOW_MAC_COUNTING (default 1)",
    }
    document = {
        "schema_version": 1,
        "status": "complete" if total_energy_complete else "charge_only",
        "warnings": ([] if total_energy_complete else
                     ["one or more rail voltages are unknown; Joule and TOPS/W omitted"]),
        "power_run_dir": str(power_dir.resolve()),
        "rtl_timing_file": str(timing_path.resolve()),
        "rtl_timing_sha256": sha256_file(timing_path),
        "analysis_config_file": str(config_path.resolve()),
        "analysis_config_sha256": sha256_file(config_path),
        "peak_overrides_file": str(override_path.resolve()) if override_path else None,
        "model": timing.get("model"),
        "board": build.get("board"),
        "fsdb_board_metadata": timing.get("board"),
        "fsdb_board_reuse": True,
        "checkpoint_alias": timing.get("checkpoint_alias"),
        "dataset": timing.get("dataset"),
        "sample_index": timing.get("sample_index"),
        "random_seed": timing.get("random_seed"),
        "imcflow_bugfix": timing.get("imcflow_bugfix"),
        "codegen_fingerprint": timing.get("codegen_fingerprint"),
        "chip_clock_hz": clock,
        "rtl_clock_hz": timing.get("rtl_clock_hz"),
        "revisions": {
            "tvm": {"revision": timing.get("tvm_revision"),
                    "dirty": timing.get("tvm_dirty")},
            "measurement_utils": {
                "revision": timing.get("measurement_utils_revision"),
                "dirty": timing.get("measurement_utils_dirty"),
            },
            "imcflow_rtl": {"revision": timing.get("rtl_revision"),
                            "dirty": timing.get("rtl_dirty")},
            "fsdb_cli": timing.get("fsdb_cli_revision"),
        },
        "time_axis_assumption": (
            "Each raw trace is uniformly sampled at its sidecar sample_interval_ns; "
            "rails have independent trigger origins. A constant rail-local offset is fitted "
            "only to assign detected peaks to tile slots; samples are never copied across rails."
        ),
        "peak_detector": settings,
        "missing_peak_policy": missing_policy,
        "peak_alignment_reference_rail": reference_rail,
        "peak_alignment_tolerance_samples": tolerance,
        "rails": rail_results,
        "tile_totals": tile_totals,
        "region_totals": region_totals,
        "totals": totals,
        "_sample_documents": sample_documents,
    }
    return document


def write_outputs(power_dir: Path, document: dict[str, Any]) -> None:
    json_path = power_dir / "tile_energy.json"
    csv_path = power_dir / "tile_energy.csv"
    plot_path = power_dir / "run_only_power_trace.png"
    persisted = dict(document)
    sample_documents = persisted.pop("_sample_documents", {})
    sample_dir = power_dir / "run_only_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for sample_name, sample_document in sample_documents.items():
        (sample_dir / sample_name).write_text(
            json.dumps(sample_document, indent=2, sort_keys=True) + "\n"
        )
    json_path.write_text(json.dumps(persisted, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as stream:
        fields = [
            "rail", "region_index", "tile_index", "tile_key", "rtl_anchor",
            "duration_cycles", "chip_duration_s", "start_sample", "end_sample_float",
            "detection_method", "baseline_current_a", "gross_charge_c",
            "dynamic_charge_c", "voltage_v", "gross_energy_j", "dynamic_energy_j",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rail, rail_result in persisted["rails"].items():
            for tile in rail_result["tiles"]:
                writer.writerow({"rail": rail, **{field: tile.get(field) for field in fields[1:]}})
        energy_key = ("total_chip_energy_j" if "total_chip_energy_j" in persisted["totals"]
                      else "measured_rail_energy_j")
        dynamic_key = ("total_chip_dynamic_energy_j"
                       if "total_chip_dynamic_energy_j" in persisted["totals"]
                       else "measured_rail_dynamic_energy_j")
        for tile in persisted["tile_totals"]:
            writer.writerow({
                "rail": "TOTAL",
                "region_index": tile["region_index"],
                "tile_index": tile["tile_index"],
                "tile_key": tile["tile_key"],
                "gross_charge_c": tile["gross_charge_c_sum_across_rails"],
                "dynamic_charge_c": tile["dynamic_charge_c_sum_across_rails"],
                "gross_energy_j": tile[energy_key],
                "dynamic_energy_j": tile[dynamic_key],
            })
    traces, _build = resolve_traces(power_dir)
    _write_plot(plot_path, traces, persisted["rails"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("power_run_dir", type=Path)
    parser.add_argument("--rtl-timing", type=Path)
    parser.add_argument("--analysis-config", type=Path, required=True)
    parser.add_argument("--peak-overrides", type=Path)
    parser.add_argument("--detect-only", action="store_true",
                        help="validate raw traces and print candidates without RTL/energy")
    parser.add_argument("--expected-tiles", type=int, default=6,
                        help="detector-only expected tile count")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_json(args.analysis_config)
        validate_analysis_config(config, require_clock=not args.detect_only)
        if args.detect_only:
            traces, _build = resolve_traces(args.power_run_dir)
            report = detector_report(
                traces, args.expected_tiles, config["peak_detector"]
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            enough = [value["enough_primary_candidates"] for value in report.values()]
            if config.get("missing_peak_policy") == "zero_energy":
                # One complete rail supplies the chronological tile reference;
                # absent peaks on the remaining rails are intentional 0 J.
                return 0 if any(enough) else 4
            return 0 if all(enough) else 4
        if args.rtl_timing is None:
            raise AnalysisInputError("--rtl-timing is required unless --detect-only is used")
        document = analyze(
            args.power_run_dir, args.rtl_timing, args.analysis_config,
            args.peak_overrides,
        )
        write_outputs(args.power_run_dir, document)
        print(json.dumps(document["totals"], indent=2, sort_keys=True))
        return 0
    except (OSError, AnalysisInputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
