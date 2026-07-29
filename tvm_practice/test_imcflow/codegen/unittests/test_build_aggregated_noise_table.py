import importlib.util
from pathlib import Path

import numpy as np


CODEGEN = Path(__file__).resolve().parents[1]
MODULE_PATH = CODEGEN / "scripts" / "build_aggregated_noise_table.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "build_aggregated_noise_table", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_sample_dirs_uses_staged_ordinal_dirs_when_exact_indices_are_absent(tmp_path):
    module = load_module()
    dump_dir = tmp_path / "debugging" / "fpga" / "vww_iter_000"
    for idx in range(3):
        (dump_dir / f"sample_{idx}").mkdir(parents=True)

    entries = list(module.resolve_sample_dirs([str(dump_dir)], [1147, 6158, 4318]))

    assert [(entry.sample_id, Path(entry.sample_dir).name) for entry in entries] == [
        (1147, "sample_0"),
        (6158, "sample_1"),
        (4318, "sample_2"),
    ]
    assert {entry.mode for entry in entries} == {"ordinal"}


def test_resolve_sample_dirs_always_uses_request_order_to_dump_postfix_mapping(tmp_path):
    module = load_module()
    dump_dir = tmp_path / "debugging" / "fpga" / "vww_iter_000"
    (dump_dir / "sample_0").mkdir(parents=True)
    (dump_dir / "sample_1147").mkdir(parents=True)

    entries = list(module.resolve_sample_dirs([str(dump_dir)], [1147]))

    assert len(entries) == 1
    assert entries[0].sample_id == 1147
    assert Path(entries[0].sample_dir).name == "sample_0"
    assert entries[0].mode == "ordinal"


def test_layered_accumulators_preserve_global_observation_count():
    module = load_module()
    ref_edges = np.array([-1.0, 1.0])
    noise_edges = np.array([-1.0, 0.0, 1.0])
    global_acc = module.AggregatedNoiseAccumulator(2, ref_edges, noise_edges)
    layer_accs = {
        "weight_pw_1": module.AggregatedNoiseAccumulator(2, ref_edges, noise_edges),
        "weight_pw_2": module.AggregatedNoiseAccumulator(2, ref_edges, noise_edges),
    }

    pch = np.array([0, 1])
    clean = np.zeros((2, 1, 2), dtype=np.int32)
    noise_a = np.array([[[0, 0]], [[0, 0]]], dtype=np.int32)
    noise_b = np.array([[[0, 0]], [[0, 0]]], dtype=np.int32)

    global_acc.add_batch(pch, clean, noise_a)
    layer_accs["weight_pw_1"].add_batch(pch, clean, noise_a)
    global_acc.add_batch(pch, clean, noise_b)
    layer_accs["weight_pw_2"].add_batch(pch, clean, noise_b)

    global_count = global_acc.results()["count"]
    layer_count = np.stack(
        [layer_accs[name].results()["count"] for name in sorted(layer_accs)],
        axis=0,
    )

    assert layer_count.shape == (2, 2, 1)
    assert int(global_count.sum()) == 8
    assert int(layer_count.sum()) == int(global_count.sum())
