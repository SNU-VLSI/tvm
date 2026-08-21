#!/usr/bin/env python3
"""Plot raw current lists produced by the legacy DMM measurement protocol."""

from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path
from typing import Iterable, Sequence


class PlotInputError(ValueError):
    """Raised when a legacy raw-current file cannot be interpreted safely."""


def load_captures(path: Path) -> list[list[float]]:
    """Load every list appended to a legacy raw-current text file."""

    captures: list[list[float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = ast.literal_eval(line)
        except (SyntaxError, ValueError) as exc:
            raise PlotInputError(f"{path}:{line_number}: invalid sample list") from exc
        if not isinstance(record, (list, tuple)):
            raise PlotInputError(f"{path}:{line_number}: expected a list of samples")
        values: list[float] = []
        for sample_index, value in enumerate(record):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PlotInputError(
                    f"{path}:{line_number}: sample {sample_index} is not numeric"
                )
            sample = float(value)
            if not math.isfinite(sample):
                raise PlotInputError(
                    f"{path}:{line_number}: sample {sample_index} is not finite"
                )
            values.append(sample)
        if values:
            captures.append(values)
    if not captures:
        raise PlotInputError(f"{path}: no current samples found")
    return captures


def _common_token_prefix(paths: Sequence[Path]) -> int:
    tokens = [path.stem.split("_") for path in paths]
    if len(tokens) < 2:
        return 0
    count = 0
    for columns in zip(*tokens):
        if len(set(columns)) != 1:
            break
        count += 1
    return count


def _trace_label(path: Path, common_tokens: int) -> str:
    tokens = path.stem.split("_")
    label = "_".join(tokens[common_tokens:]) if common_tokens else path.stem
    return label or path.stem


def _plot_axis(axis, captures: Sequence[Sequence[float]], label: str) -> None:
    import numpy as np

    offset = 0
    all_values: list[float] = []
    for capture_index, capture in enumerate(captures):
        values = np.asarray(capture, dtype=np.float64)
        samples = np.arange(offset, offset + len(values))
        axis.plot(samples, values, linewidth=0.7)
        if capture_index:
            axis.axvline(offset - 0.5, color="tab:gray", linewidth=0.7, alpha=0.7)
        offset += len(values)
        all_values.extend(capture)

    stats = np.asarray(all_values, dtype=np.float64)
    axis.set_title(label, fontsize=9)
    axis.set_ylabel("current (A)")
    axis.grid(alpha=0.25)
    axis.text(
        0.995,
        0.98,
        f"samples={len(stats)}  captures={len(captures)}  "
        f"mean={stats.mean():.6g} A  min={stats.min():.6g}  max={stats.max():.6g}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )


def plot_directory(
    input_dir: Path,
    output: Path,
    individual_dir: Path | None = None,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw_paths = sorted(
        path for path in input_dir.glob("*.txt") if path.is_file()
    )
    if not raw_paths:
        raise PlotInputError(f"{input_dir}: no legacy raw .txt files found")

    loaded = [(path, load_captures(path)) for path in raw_paths]
    common_tokens = _common_token_prefix(raw_paths)
    figure, axes = plt.subplots(
        len(loaded),
        1,
        squeeze=False,
        figsize=(12, max(3.2, 2.7 * len(loaded))),
    )
    for axis, (path, captures) in zip(axes[:, 0], loaded):
        _plot_axis(axis, captures, _trace_label(path, common_tokens))
    axes[-1, 0].set_xlabel("sample index (capture boundaries are gray lines)")
    figure.suptitle(input_dir.name)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)

    outputs = [output]
    if individual_dir is not None:
        individual_dir.mkdir(parents=True, exist_ok=True)
        for path, captures in loaded:
            trace_figure, trace_axis = plt.subplots(1, 1, figsize=(12, 3.5))
            _plot_axis(trace_axis, captures, path.stem)
            trace_axis.set_xlabel("sample index (capture boundaries are gray lines)")
            trace_figure.tight_layout()
            trace_output = individual_dir / f"{path.stem}.png"
            trace_figure.savefig(trace_output, dpi=150)
            plt.close(trace_figure)
            outputs.append(trace_output)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="directory containing legacy raw .txt files")
    parser.add_argument("--output", type=Path, required=True, help="combined PNG output")
    parser.add_argument(
        "--individual-dir",
        type=Path,
        help="optional directory for one PNG per raw trace",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outputs = plot_directory(args.input_dir, args.output, args.individual_dir)
    except (OSError, PlotInputError) as exc:
        print(f"Error: {exc}")
        return 1
    for output in outputs:
        print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
