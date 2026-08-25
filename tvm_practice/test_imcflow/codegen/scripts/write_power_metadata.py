#!/usr/bin/env python3
"""Create a compact manifest for collected legacy power traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("power_dir", type=Path)
  args = parser.parse_args()
  power_dir = args.power_dir
  build_path = power_dir / "build_metadata.json"
  build = json.loads(build_path.read_text()) if build_path.is_file() else {}
  settings = build.get("power_measurement", {})
  names = settings.get("dmm_names") or []
  if not names and settings.get("dmm_name"):
    names = [settings["dmm_name"]]

  raw_paths = sorted(
      path for path in power_dir.glob("*.txt")
      if path.is_file() and not path.name.endswith(".tags.json"))
  traces = []
  for name in names:
    # Plural mode writes _<logical-name>.txt.  Legacy one-DMM output has no
    # suffix, so associate its sole raw file below.
    matching = [path for path in raw_paths if path.stem.endswith(f"_{name}")]
    if not matching and len(names) == 1 and len(raw_paths) == 1:
      matching = raw_paths
    for raw_path in matching:
      trace = {
          "dmm_name": name,
          "raw_file": raw_path.name,
          "tag_file": f"{raw_path.name}.tags.json"
              if Path(f"{raw_path}.tags.json").is_file() else None,
          "plot_file": f"plots/{raw_path.stem}.png"
              if (power_dir / "plots" / f"{raw_path.stem}.png").is_file() else None,
        }
      traces.append(trace)

  document = {
      "dmm_names": names,
      "requested_interval_s": settings.get("requested_interval_s"),
      "sample_count": settings.get("sample_count"),
      "traces": traces,
      "combined_plot": "power_trace.png"
          if (power_dir / "power_trace.png").is_file() else None,
  }
  (power_dir / "power_metadata.json").write_text(
      json.dumps(document, indent=2, sort_keys=True) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
