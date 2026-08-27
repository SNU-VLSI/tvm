import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "tools" / "rtl_region_cycles.py"
SPEC = importlib.util.spec_from_file_location("rtl_region_cycles", SCRIPT)
rtl = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(rtl)


def _paths():
  valid = [
      f"/testbench_imcflow_gem5/u/core_row[{row}]/core_col[{col}]/"
      "imce_node/imce/u_imce_datapath/bshr/valid"
      for row in range(4) for col in range(1, 5)
  ]
  ready = [path.removesuffix("valid") + "ready" for path in valid]
  return [rtl.BUSY_SIGNAL, *valid, *ready], valid, ready


def _event(time, busy, signals, high=()):
  values = {rtl.BUSY_SIGNAL: str(busy)}
  values.update({signal: ("1" if signal in high else "0") for signal in signals})
  return SimpleNamespace(time=time, values=values)


def test_validate_signal_discovery_exact_cardinality():
  paths, valid, ready = _paths()
  busy, discovered_valid, discovered_ready = rtl.validate_signal_paths(paths)
  assert busy == rtl.BUSY_SIGNAL
  assert discovered_valid == sorted(valid)
  assert discovered_ready == sorted(ready)
  with pytest.raises(rtl.TimingInputError, match="valid/ready=16/16"):
    rtl.validate_signal_paths(paths[:-1])


def test_tile_intervals_exclude_setup_and_find_simultaneous_first_imcus():
  _paths_all, valid, ready = _paths()
  signals = [*valid, *ready]
  events = [
      _event(0, 0, signals),
      _event(10, 1, signals),
      _event(20, 0, signals),
      _event(30, 1, signals),
      _event(35, 1, signals, (valid[0], ready[0], valid[5], ready[5])),
      _event(36, 1, signals),
      _event(50, 0, signals),
  ]
  runs = rtl.tile_intervals_from_events(events, rtl.BUSY_SIGNAL, valid, ready)
  assert len(runs) == 2
  assert runs[0]["any_imcu_input_handshake_time_units"] is None
  assert runs[1]["run_state_start_time_units"] == 30
  assert runs[1]["any_imcu_input_handshake_time_units"] == 35
  assert runs[1]["run_state_end_time_units"] == 50
  assert runs[1]["first_imcu_coordinates"] == [[0, 0], [1, 1]]


def test_unknown_transition_is_not_a_rise():
  signal = "s"
  events = [
      SimpleNamespace(time=0, values={signal: "0"}),
      SimpleNamespace(time=1, values={signal: "x"}),
      SimpleNamespace(time=2, values={signal: "1"}),
      SimpleNamespace(time=3, values={signal: "0"}),
      SimpleNamespace(time=4, values={signal: "1"}),
  ]
  assert rtl.rising_edges(events, signal) == [2, 4]


def test_manifest_tile_count_mismatch_fails_closed(tmp_path):
  manifest = {
      "schema_version": 1,
      "regions": [{"region_index": 1, "function": "region1", "tile_count": 2}],
  }
  with pytest.raises(rtl.TimingInputError, match="tile count mismatch"):
    rtl.build_tile_document(
        eval_dir=tmp_path,
        fsdb_path=tmp_path / "missing.fsdb",
        fsdb_cli_root=tmp_path,
        busy_signal=rtl.BUSY_SIGNAL,
        imcu_valid_signals=_paths()[1],
        imcu_ready_signals=_paths()[2],
        report_time_unit="ps",
        period_units=10_000,
        runs=[{
            "run_state_start_time_units": 10_000,
            "any_imcu_input_handshake_time_units": 20_000,
            "first_imcu_coordinates": [[0, 0]],
            "run_state_end_time_units": 30_000,
        }],
        region_starts=[0],
        region_final=100_000,
        names=["region1"],
        manifest=manifest,
    )


def test_manifest_identity_supports_board_independent_fsdb_reuse(tmp_path):
  fsdb = tmp_path / "fixture.fsdb"
  fsdb.write_bytes(b"fixture")
  manifest = {
      "schema_version": 2,
      "model": "model",
      "checkpoint_alias": "ckpt",
      "dataset": "cifar10",
      "sample_index": 0,
      "random_seed": 42,
      "imcflow_bugfix": False,
      "codegen_fingerprint": "fp",
      "regions": [{"region_index": 1, "function": "region1", "tile_count": 1}],
  }
  document = rtl.build_tile_document(
      eval_dir=tmp_path,
      fsdb_path=fsdb,
      fsdb_cli_root=tmp_path,
      busy_signal=rtl.BUSY_SIGNAL,
      imcu_valid_signals=_paths()[1],
      imcu_ready_signals=_paths()[2],
      report_time_unit="ps",
      period_units=10_000,
      runs=[{
          "run_state_start_time_units": 10_000,
          "any_imcu_input_handshake_time_units": None,
          "first_imcu_coordinates": [],
          "run_state_end_time_units": 30_000,
      }],
      region_starts=[0],
      region_final=100_000,
      names=["region1"],
      manifest=manifest,
  )
  assert document["checkpoint_alias"] == "ckpt"
  assert document["dataset"] == "cifar10"
  assert document["sample_index"] == 0
  assert document["random_seed"] == 42
  assert document["imcflow_bugfix"] is False
