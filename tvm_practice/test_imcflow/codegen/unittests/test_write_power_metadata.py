"""Tests for the multi-DMM power artifact manifest writer."""

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "write_power_metadata.py"
SPEC = importlib.util.spec_from_file_location("write_power_metadata", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_writes_one_entry_per_dmm_trace(tmp_path, monkeypatch):
  (tmp_path / "build_metadata.json").write_text(json.dumps({
      "power_measurement": {
          "dmm_names": ["DMM_GPIB1", "DMM_GPIB2"],
          "requested_interval_s": 0.00002,
          "sample_count": 50,
      }
  }))
  for name in ("DMM_GPIB1", "DMM_GPIB2"):
    raw = tmp_path / f"capture_model_{name}.txt"
    raw.write_text("[0.1, 0.2]\n")
    (tmp_path / f"capture_model_{name}.txt.tags.json").write_text("{}\n")
    plots = tmp_path / "plots"
    plots.mkdir(exist_ok=True)
    (plots / f"capture_model_{name}.png").write_bytes(b"png")
  (tmp_path / "power_trace.png").write_bytes(b"png")

  monkeypatch.setattr("sys.argv", ["write_power_metadata.py", str(tmp_path)])
  assert MODULE.main() == 0

  document = json.loads((tmp_path / "power_metadata.json").read_text())
  assert document["dmm_names"] == ["DMM_GPIB1", "DMM_GPIB2"]
  assert [entry["dmm_name"] for entry in document["traces"]] == document["dmm_names"]
  assert all(entry["plot_file"] for entry in document["traces"])
