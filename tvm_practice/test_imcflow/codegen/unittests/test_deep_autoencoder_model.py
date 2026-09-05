import numpy as np

from models import deep_autoencoder_imcflow


def test_synthetic_parameters_are_deterministic_and_valid():
    _, params_a = deep_autoencoder_imcflow.getModel(seed=7)
    _, params_b = deep_autoencoder_imcflow.getModel(seed=7)

    assert params_a.keys() == params_b.keys()
    for name in params_a:
        np.testing.assert_array_equal(params_a[name], params_b[name])

    assert np.all(params_a["bn_moving_var1"] > 0)
    for index in range(1, 8):
        assert params_a[f"quant_min{index}"] < params_a[f"quant_max{index}"]


def test_different_seed_changes_weights_not_numeric_configuration():
    _, params_a = deep_autoencoder_imcflow.getModel(seed=7)
    _, params_b = deep_autoencoder_imcflow.getModel(seed=8)

    assert not np.array_equal(params_a["weight2"], params_b["weight2"])
    np.testing.assert_array_equal(params_a["bn_moving_var1"], params_b["bn_moving_var1"])
    np.testing.assert_array_equal(params_a["quant_min1"], params_b["quant_min1"])
    np.testing.assert_array_equal(params_a["quant_max1"], params_b["quant_max1"])
