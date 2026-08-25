import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "plot_legacy_power.py"
SPEC = importlib.util.spec_from_file_location("plot_legacy_power", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
plot_legacy_power = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot_legacy_power)


def test_load_captures_preserves_appended_record_boundaries(tmp_path):
    raw = tmp_path / "region1.txt"
    raw.write_text("[0.1, 0.2]\n[0.3, 0.4, 0.5]\n", encoding="utf-8")

    assert plot_legacy_power.load_captures(raw) == [
        [0.1, 0.2],
        [0.3, 0.4, 0.5],
    ]


def test_plot_directory_writes_combined_and_individual_plots(tmp_path):
    raw_dir = tmp_path / "run"
    raw_dir.mkdir()
    (raw_dir / "resnet_region1.txt").write_text("[0.1, 0.2]\n", encoding="utf-8")
    (raw_dir / "resnet_region2.txt").write_text("[0.3, 0.4]\n", encoding="utf-8")
    combined = raw_dir / "power_trace.png"
    individual = raw_dir / "plots"

    outputs = plot_legacy_power.plot_directory(raw_dir, combined, individual)

    assert outputs[0] == combined
    assert combined.stat().st_size > 0
    assert (individual / "resnet_region1.png").stat().st_size > 0
    assert (individual / "resnet_region2.png").stat().st_size > 0


def test_load_tags_reads_optional_sample_index_map(tmp_path):
    raw = tmp_path / "model.txt"
    raw.write_text("[0.1, 0.2, 0.3]\n", encoding="utf-8")
    tag_file = tmp_path / "model.txt.tags.json"
    tag_file.write_text('{"tags": {"2": 1, "3": 2}}\n', encoding="utf-8")

    assert plot_legacy_power.load_tags(raw) == {2: 1, 3: 2}


def test_plot_directory_draws_tag_lines(tmp_path, monkeypatch):
    raw_dir = tmp_path / "run"
    raw_dir.mkdir()
    raw = raw_dir / "model.txt"
    raw.write_text("[0.1, 0.2, 0.3]\n", encoding="utf-8")
    (raw_dir / "model.txt.tags.json").write_text(
        '{"tags": {"2": 1, "3": 2}}\n', encoding="utf-8"
    )
    observed_tags = []
    original_plot_axis = plot_legacy_power._plot_axis

    def capture_plot_axis(axis, captures, label, tags=None):
        observed_tags.append(tags)
        return original_plot_axis(axis, captures, label, tags)

    monkeypatch.setattr(plot_legacy_power, "_plot_axis", capture_plot_axis)
    plot_legacy_power.plot_directory(raw_dir, raw_dir / "power_trace.png")

    assert observed_tags == [{2: 1, 3: 2}]


def test_load_captures_rejects_non_numeric_samples(tmp_path):
    raw = tmp_path / "invalid.txt"
    raw.write_text("[0.1, 'bad']\n", encoding="utf-8")

    with pytest.raises(plot_legacy_power.PlotInputError, match="not numeric"):
        plot_legacy_power.load_captures(raw)
