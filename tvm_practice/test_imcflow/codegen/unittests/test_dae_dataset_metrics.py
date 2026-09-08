import json
from pathlib import Path
import subprocess

import numpy as np


def test_c_dae_mse_json_and_invalid_outputs(tmp_path):
    codegen = Path(__file__).resolve().parents[1]
    tvm = codegen.parents[2]
    binary = tmp_path / "test_dae_metrics"
    subprocess.run(["gcc", "-std=gnu11", "-I" + str(tvm / "3rdparty/dlpack/include"),
                    "-I" + str(codegen / "host_binary_make.dataset/src"),
                    str(codegen / "unittests/c/test_dae_dataset_metrics.c"), "-lm", "-o", str(binary)], check=True)
    result = subprocess.run([str(binary)], check=True, text=True, capture_output=True)
    lines = result.stdout.splitlines()
    assert len(lines) == 1 and lines[0].startswith("DAE_SAMPLE ")
    record = json.loads(lines[0].split(" ", 1)[1])
    assert record["sample_id"] == 7 and record["label"] == 1
    assert abs(record["mse"] - 4) < 1e-5
    values = np.asarray(record["reconstruction"], dtype=np.float32)
    assert values.shape == (640,)
    np.testing.assert_array_equal(values, np.arange(640, dtype=np.float32) / 10 + 2)
