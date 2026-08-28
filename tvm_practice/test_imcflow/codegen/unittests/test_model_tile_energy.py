import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_model_tile_energy.py"
SPEC = importlib.util.spec_from_file_location("analyze_model_tile_energy", SCRIPT)
energy = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = energy
SPEC.loader.exec_module(energy)


SETTINGS = {
    "median_window": 3,
    "baseline_window": 9,
    "baseline_samples": 20,
    "mad_multiplier": 6.0,
    "hysteresis_low_ratio": 0.5,
    "minimum_width_samples": 2,
    "merge_gap_samples": 2,
    "absolute_min_rise_a": 0.2,
    "threshold_variation_fraction": 0.2,
    "alignment_tolerance_samples": 10,
}


def test_detector_finds_ordered_pulses_in_each_independent_trace():
  starts_by_rail = ([30, 80], [35, 90], [25, 70])
  for starts in starts_by_rail:
    values = np.zeros(140)
    for start in starts:
      values[start:start + 8] = 1.0
    candidates, _threshold = energy.detect_rising_candidates(values, SETTINGS)
    assert [candidate.start_sample for candidate in candidates] == starts


def test_missing_peak_requires_manual_override():
  values = np.zeros(100)
  values[20:30] = 1
  candidates, _threshold = energy.detect_rising_candidates(values, SETTINGS)
  with pytest.raises(energy.AnalysisInputError, match="only 1 direct peaks"):
    energy.map_starts(candidates, 2, {})
  mapped, _rejected = energy.map_starts(candidates, 2, {1: 70})
  assert [(item["start_sample"], item["method"]) for item in mapped] == [
      (20, "direct"), (70, "manual")]


def test_fractional_integral_constant_current_and_idle_subtraction():
  gross, dynamic, records = energy.fractional_integral(
      [2.0, 2.0, 2.0], sample_interval_s=1.0,
      start_time_s=0.5, duration_s=1.25, baseline_a=0.5,
  )
  assert gross == pytest.approx(2.5)
  assert dynamic == pytest.approx(1.875)
  assert [record["boundary_weight"] for record in records] == pytest.approx([0.5, 0.75])


def test_flatten_tiles_preserves_two_duration_anchors():
  timing = {
      "schema_version": 2,
      "rtl_method": "fsdb_cli",
      "regions": [{
          "region_index": 1,
          "function": "f",
          "tiles": [{
              "tile_index": 0,
              "run_state_start_cycle": 10,
              "any_imcu_input_handshake_cycle": 13,
              "run_state_end_cycle": 20,
              "run_state_cycles": 10,
              "imcu_to_run_end_cycles": 7,
          }],
      }],
  }
  tile = energy.flatten_tiles(timing)[0]
  assert tile["run_state_cycles"] == 10
  assert tile["imcu_to_run_end_cycles"] == 7


def test_clock_and_voltage_validation_is_fail_closed():
  config = {
      "schema_version": 1,
      "chip_clock_hz": None,
      "rails": {
          rail: {"voltage_v": None, "include_in_total": True, "rtl_anchor": anchor}
          for rail, anchor in energy.EXPECTED_ANCHORS.items()
      },
      "peak_detector": dict(SETTINGS),
  }
  with pytest.raises(energy.AnalysisInputError, match="chip_clock_hz is unknown"):
    energy.validate_analysis_config(config)
  energy.validate_analysis_config(config, require_clock=False)
  config["chip_clock_hz"] = 100_000_000
  config["rails"]["VDD"]["voltage_v"] = -1
  with pytest.raises(energy.AnalysisInputError, match="VDD.voltage_v"):
    energy.validate_analysis_config(config)


def test_synthetic_end_to_end_produces_tile_region_and_model_totals(tmp_path):
  identity = {
      "board": "B2", "checkpoint_alias": "ckpt", "dataset": "cifar10",
      "sample_index": 0, "random_seed": 42, "imcflow_bugfix": False,
      "codegen_fingerprint": "fp", "tvm_revision": "tvm",
      "measurement_utils_revision": "mu", "tvm_dirty": False,
      "measurement_utils_dirty": False,
  }
  build = {
      "model_name": "model", **identity,
      "imcflow_revision": "rtl", "imcflow_dirty": False,
      "conv_mac_count": 1000,
      "power_measurement": {
          "scope": "MODEL", "dmm_names": ["VDD", "DDA", "DDC"],
      },
  }
  (tmp_path / "build_metadata.json").write_text(json.dumps(build))
  traces = []
  starts = {"VDD": (20, 60), "DDA": (25,), "DDC": (30,)}
  for rail in energy.RAIL_ALIASES:
    values = np.zeros(100)
    for start in starts[rail]:
      values[start:start + 6] = 1.0
    raw = tmp_path / f"trace_{rail}.txt"
    raw.write_text(repr(values.tolist()) + "\n")
    tags = tmp_path / f"trace_{rail}.txt.tags.json"
    tags.write_text(json.dumps({
        "sample_interval_ns": 100_000_000,
        "actual_sample_count": len(values),
    }))
    traces.append({"dmm_name": rail, "raw_file": raw.name, "tag_file": tags.name})
  (tmp_path / "power_metadata.json").write_text(json.dumps({
      "dmm_names": ["VDD", "DDA", "DDC"], "traces": traces,
  }))
  timing = {
      "schema_version": 2, "rtl_method": "fsdb_cli", "model": "model",
      **identity, "board": "B1", "rtl_revision": "rtl", "rtl_dirty": False,
      "rtl_clock_hz": 10,
      "regions": [{
          "region_index": 1, "function": "f", "tiles": [{
              "tile_index": index, "run_state_start_cycle": index * 10,
              "any_imcu_input_handshake_cycle": index * 10 + 1,
              "run_state_end_cycle": index * 10 + 2,
              "run_state_cycles": 2, "imcu_to_run_end_cycles": 1,
          } for index in range(2)],
      }],
  }
  timing_path = tmp_path / "timing.json"
  timing_path.write_text(json.dumps(timing))
  config = {
      "schema_version": 1, "chip_clock_hz": 10,
      "missing_peak_policy": "zero_energy", "all_chip_power": True,
      "rails": {
          "VDD": {"voltage_v": 1.0, "include_in_total": True,
                  "rtl_anchor": "run_state"},
          "DDA": {"voltage_v": 3.0, "include_in_total": True,
                  "rtl_anchor": "any_imcu_input_handshake"},
          "DDC": {"voltage_v": 2.0, "include_in_total": True,
                  "rtl_anchor": "run_state"},
      },
      "peak_detector": dict(SETTINGS),
  }
  config_path = tmp_path / "config.json"
  config_path.write_text(json.dumps(config))

  document = energy.analyze(tmp_path, timing_path, config_path)
  assert len(document["tile_totals"]) == 2
  assert len(document["region_totals"]) == 1
  assert document["totals"]["total_chip_energy_j"] == pytest.approx(1.1)
  assert document["rails"]["DDA"]["tiles"][1]["detection_method"] == "missing_peak_zero"
  assert document["rails"]["DDA"]["tiles"][1]["gross_energy_j"] == 0.0
  assert document["totals"]["conv_mac_count"] == 1000
  assert document["totals"]["operation_count"] == 1000
  assert not (tmp_path / "run_only_samples").exists()


def test_voltages_can_come_from_dmm_config(tmp_path):
  (tmp_path / "dmm.json").write_text(json.dumps({
      "POWER": {
          "VDD": {"VOLTAGE_V": 1.0},
          "DDA": {"VOLTAGE_V": 1.1},
          "DDC": {"VOLTAGE_V": 1.2},
      }
  }))
  config = {
      "dmm_config": "dmm.json",
      "rails": {rail: {"voltage_v": None} for rail in energy.RAIL_ALIASES},
  }
  assert energy.resolve_rail_voltages(tmp_path / "energy.json", config) == {
      "VDD": 1.0, "DDA": 1.1, "DDC": 1.2,
  }


def test_mac_counting_environment_defaults_to_two_and_accepts_one(monkeypatch):
  monkeypatch.delenv("IMCFLOW_MAC_COUNTING", raising=False)
  assert energy.mac_counting_from_environment() == 2
  monkeypatch.setenv("IMCFLOW_MAC_COUNTING", "1")
  assert energy.mac_counting_from_environment() == 1
  monkeypatch.setenv("IMCFLOW_MAC_COUNTING", "3")
  with pytest.raises(energy.AnalysisInputError, match="must be 1 or 2"):
    energy.mac_counting_from_environment()
