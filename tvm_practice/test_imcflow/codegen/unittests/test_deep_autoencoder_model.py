import numpy as np
import pytest
import torch
from tvm import relay
from tvm.relay.backend.contrib.imcflow.acim_util import get_default_acc_mask, set_default_acc_mask

from models import deep_autoencoder_imcflow


@pytest.fixture(autouse=True)
def acc_mask_one():
    previous = get_default_acc_mask()
    set_default_acc_mask(1)
    yield
    set_default_acc_mask(previous)


def test_synthetic_parameters_are_deterministic_and_valid():
    _, params_a = deep_autoencoder_imcflow.getModel(seed=7)
    _, params_b = deep_autoencoder_imcflow.getModel(seed=7)

    assert params_a.keys() == params_b.keys()
    for name in params_a:
        np.testing.assert_array_equal(params_a[name], params_b[name])

    assert np.all(params_a["bn_moving_var1"] > 0)
    for index in range(1, 9):
        assert params_a[f"quant_min{index}"] < params_a[f"quant_max{index}"]


def test_different_seed_changes_weights_not_numeric_configuration():
    _, params_a = deep_autoencoder_imcflow.getModel(seed=7)
    _, params_b = deep_autoencoder_imcflow.getModel(seed=8)

    assert not np.array_equal(params_a["weight2"], params_b["weight2"])
    np.testing.assert_array_equal(params_a["bn_moving_var1"], params_b["bn_moving_var1"])
    np.testing.assert_array_equal(params_a["quant_min1"], params_b["quant_min1"])
    np.testing.assert_array_equal(params_a["quant_max1"], params_b["quant_max1"])


def test_eight_block_topology_and_signed_front():
    mod, params = deep_autoencoder_imcflow.getModel()
    mod = relay.transform.InferType()(mod)
    assert tuple(int(d) for d in mod["main"].ret_type.shape) == (1, 640)
    calls = []
    relay.analysis.post_order_visit(mod["main"], lambda e: calls.append(e) if isinstance(e, relay.Call) else None)
    qconvs = [c for c in calls if getattr(c.op, "name", "") == "nn.imcflow_qconv"]
    assert [c.args[1].name_hint for c in qconvs] == [f"weight{i}" for i in range(2, 10)]
    for call in qconvs:
        words = call.args[2].data.numpy()
        config = int(words[0]) | (int(words[1]) << 32)
        assert (config >> 34) & 15 == 1
    assert params["weight6"].shape == (128, 8, 1, 1)
    assert params["weight9"].shape == (128, 128, 1, 1)
    assert params["dense_bias_final"].shape == (640,)
    relus = [c for c in calls if getattr(c.op, "name", "") == "nn.relu"]
    assert len(relus) == 1  # head only
    clips = [c for c in calls if getattr(c.op, "name", "") == "clip"]
    assert len(clips) == 1
    assert clips[0].attrs.a_min == -32768 and clips[0].attrs.a_max == 32767


@pytest.fixture
def exported_checkpoint(tmp_path, monkeypatch):
    _, params = deep_autoencoder_imcflow.getModel()
    state = {}
    mapping = {
        "weight1": "_fh.linear1.weight", "bn_gamma1": "_fh.bn1.weight",
        "bn_beta1": "_fh.bn1.bias", "bn_moving_mean1": "_fh.bn1.running_mean",
        "bn_moving_var1": "_fh.bn1.running_var",
        "dense_weight_final": "_fh.out.weight", "dense_bias_final": "_fh.out.bias",
    }
    for i in range(1, 9):
        p = f"blocks.{i - 1}.block_int16"
        mapping.update({f"weight{i + 1}": f"{p}.linear.weight",
                        f"quant_min{i}": f"{p}.act.min", f"quant_max{i}": f"{p}.act.max",
                        f"fused_scale{i}": f"{p}.bn.scale", f"fused_bias{i}": f"{p}.bn.bias"})
    for name, key in mapping.items():
        state[key] = torch.from_numpy(params[name].copy())
    # A nonzero final bias must survive export/load.
    state["_fh.out.bias"].fill_(0.25)
    factors = {f"{prefix}_{i}": 64.0 for i in range(1, 9) for prefix in ("x_f", "bn_f")}
    bundle = {"state_dict": state, "adjust_factors": factors}
    path = tmp_path / "checkpoint.pth.tar"
    torch.save(bundle, path)
    monkeypatch.setenv("CKPT_PATH", str(path))
    return bundle, path


def test_pretrained_mapping_and_subset(exported_checkpoint):
    _, path = exported_checkpoint
    mod, params = deep_autoencoder_imcflow.getModel_from_pretrained_weight()
    assert params["weight9"].shape == (128, 128, 1, 1)
    np.testing.assert_array_equal(params["dense_bias_final"], np.full(640, 0.25))
    assert deep_autoencoder_imcflow.get_last_checkpoint_path() == str(path)
    sub, subparams = deep_autoencoder_imcflow.getModel_from_pretrained_weight(until_relay=0)
    assert set(subparams) == {"weight1"}
    assert relay.transform.InferType()(sub)["main"].ret_type.dtype == "float32"


@pytest.mark.parametrize("value", [[64.0], [64.0, 64.0], 0, -1, float("nan"), float("inf")])
def test_pretrained_rejects_invalid_scalar(exported_checkpoint, value):
    bundle, path = exported_checkpoint
    bundle["adjust_factors"]["x_f_1"] = value
    torch.save(bundle, path)
    with pytest.raises(ValueError, match="positive finite scalar"):
        deep_autoencoder_imcflow.getModel_from_pretrained_weight()


def test_pretrained_rejects_broken_tie_and_int4_overflow(exported_checkpoint):
    bundle, path = exported_checkpoint
    bundle["adjust_factors"]["x_f_2"] = 32.0
    torch.save(bundle, path)
    with pytest.raises(ValueError, match="tie violated"):
        deep_autoencoder_imcflow.getModel_from_pretrained_weight()
    bundle["adjust_factors"]["x_f_2"] = 64.0
    bundle["state_dict"]["blocks.7.block_int16.linear.weight"].fill_(8)
    torch.save(bundle, path)
    with pytest.raises(ValueError, match="not representable"):
        deep_autoencoder_imcflow.getModel_from_pretrained_weight()
