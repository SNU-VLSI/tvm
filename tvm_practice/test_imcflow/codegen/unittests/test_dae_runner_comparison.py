import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "compare_dae_runner", Path(__file__).resolve().parents[1] / "scripts/compare_dae_runner.py")
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def make_graph():
    nodes, shapes, dtypes = [], [], []

    def node(name, inputs=(), shape=(1, 128, 1, 1), dtype="int16"):
        index = len(nodes)
        nodes.append(dict(name=name, op="tvm_op", inputs=[[i, 0, 0] for i in inputs]))
        shapes.append(list(shape))
        dtypes.append(dtype)
        return index

    x = node("model_input", shape=(1, 640), dtype="float32")
    x = node("tvmgen_default_fused_add", [x], shape=(1, 128), dtype="float32")
    x = node("tvmgen_default_fused_multiply", [x], shape=(1, 128), dtype="float32")
    x = node("tvmgen_default_fused_clip", [x], shape=(1, 128), dtype="float32")
    x = node("tvmgen_default_fused_cast", [x], shape=(1, 128))
    x = node("reshape_nop", [x])
    for block, (ic, oc) in enumerate(comparison.CHANNELS):
        q = node(f"tvmgen_default_fused_qnn_imcflow_min_max_quantize_{block}", [x],
                 shape=(1, ic, 1, 1), dtype="uint8")
        parts = []
        for part in range((oc + 31) // 32):
            packed = node("tvmgen_default_fused_nn_bitpack", [q])
            atomic = node(f"tvmgen_default_imcflow_main_{block}_{part}", [packed])
            layout = node("tvmgen_default_fused_layout_transform", [atomic])
            parts.append(node("tvmgen_default_fused_take", [layout], shape=(1, min(32, oc), 1, 1)))
        x = parts[0] if len(parts) == 1 else node("tvmgen_default_fused_concatenate", parts)
        x = node(f"tvmgen_default_fused_imcflow_fused_batch_norm_{block}", [x], shape=(1, oc, 1, 1))
    x = node("tvmgen_default_fused_cast", [x], dtype="float32")
    x = node("tvmgen_default_fused_multiply", [x], dtype="float32")
    x = node("reshape_nop", [x], shape=(1, 128), dtype="float32")
    x = node("tvmgen_default_fused_nn_relu", [x], shape=(1, 128), dtype="float32")
    x = node("tvmgen_default_fused_nn_dense", [x], shape=(1, 640), dtype="float32")
    x = node("tvmgen_default_fused_nn_bias_add", [x], shape=(1, 640), dtype="float32")
    return dict(nodes=nodes, node_row_ptr=list(range(len(nodes) + 1)), heads=[[x, 0, 0]],
                attrs=dict(shape=["list_shape", shapes], dltype=["list_str", dtypes]))


def test_logical_mapping_covers_eight_blocks_including_bottleneck_and_dec4():
    graph = make_graph()
    mapping = comparison.logical_boundaries(graph)
    assert len(mapping) == 36
    assert "take" in graph["nodes"][mapping["block3.linear"]]["name"]
    assert "concatenate" in graph["nodes"][mapping["block7.linear"]]["name"]


def test_missing_eighth_block_is_rejected():
    graph = make_graph()
    index = comparison.logical_boundaries(graph)["block7.act"]
    graph["nodes"][index]["name"] = "unknown_quantizer"
    with pytest.raises(ValueError, match="eight"):
        comparison.logical_boundaries(graph)


def test_wrong_atomic_activation_is_rejected():
    graph = make_graph()
    index = next(i for i, n in enumerate(graph["nodes"]) if "nn_bitpack" in n["name"])
    graph["nodes"][index]["inputs"][0][0] = 0
    with pytest.raises(ValueError, match="wrong logical activation"):
        comparison.logical_boundaries(graph)


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_integer_boundary_accepts_one_unit(delta):
    result = comparison.compare_tensor(np.array([8531], dtype=np.int32),
                                       np.array([8531 + delta], dtype=np.int16), integer=True)
    assert result["passed"] and result["mismatched_elements"] == 0
    assert result["differing_elements"] == int(delta != 0)


@pytest.mark.parametrize("delta", [-2, 2])
def test_integer_boundary_never_uses_relative_tolerance(delta):
    result = comparison.compare_tensor(np.array([8531], dtype=np.int32),
                                       np.array([8531 + delta], dtype=np.int16), integer=True)
    assert not result["passed"] and result["mismatched_elements"] == 1


def test_integer_comparison_does_not_overflow():
    result = comparison.compare_tensor(np.array([-32768], dtype=np.int16),
                                       np.array([32767], dtype=np.int16), integer=True)
    assert result["max_absolute_error"] == 65535


def test_float_tolerance_uses_reference_magnitude():
    assert comparison.compare_tensor(np.array([10.]), np.array([10.0109]), False)["passed"]
    assert not comparison.compare_tensor(np.array([10.]), np.array([10.0111]), False)["passed"]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_is_failure(value):
    with pytest.raises(ValueError, match="Nonfinite"):
        comparison.compare_tensor(np.array([0.]), np.array([value]), False)


def test_float_cannot_be_silently_rounded_at_integer_boundary():
    with pytest.raises(ValueError, match="Integer boundary"):
        comparison.compare_tensor(np.array([1], dtype=np.int32), np.array([1.]), True)


def test_reshape_nop_uses_its_actual_parent_dump(tmp_path):
    graph = make_graph()
    mapping = comparison.logical_boundaries(graph)
    index = mapping["front_int16"]
    parent = graph["nodes"][index]["inputs"][0][0]
    np.save(tmp_path / f"{parent:03d}_{graph['nodes'][parent]['name']}.npy", np.arange(128, dtype=np.int16))
    value = comparison.read_dump(graph, tmp_path, index)
    assert value.shape == (1, 128, 1, 1)
    np.testing.assert_array_equal(value.reshape(-1), np.arange(128))
