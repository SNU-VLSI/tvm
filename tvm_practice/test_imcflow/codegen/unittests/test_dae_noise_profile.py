"""DAE diagnosis uses all eight logical Relay weights, including dec1/dec4."""
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/diagnose_noise_per_qconv.py"
spec = importlib.util.spec_from_file_location("dae_noise_diagnosis", SCRIPT)
diagnosis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnosis)


@pytest.mark.parametrize("alias", ["ad_dae", "dae", "anomaly", "anomaly_detection", "dae_toycar_full_pretrained"])
def test_dae_profile(alias):
    params = diagnosis.get_conv_params(alias)
    assert list(params) == [f"weight{i}" for i in range(2, 10)]
    for i in range(8):
        assert params[f"weight{i + 2}"] == (1, 1, 0, f"blocks.{i}.block_int16.linear.weight")
    assert "dae_toycar_full_pretrained_evl.linux" in diagnosis.get_default_npz_path(alias)
    assert "dae_toycar_full_pretrained_evl.baremetal" in diagnosis.get_default_pysim_dir(alias)


def test_existing_profiles_unchanged():
    assert len(diagnosis.get_conv_params("resnet")) == 8
    assert len(diagnosis.get_conv_params("kws")) == 4
    assert len(diagnosis.get_conv_params("vww")) == 13
