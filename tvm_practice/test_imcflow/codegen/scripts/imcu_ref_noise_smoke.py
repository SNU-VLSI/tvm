#!/usr/bin/env python3
"""IMCU-level smoke test for signed-ref noise sampling.

This bypasses gem5 and calls IMCU.compute() directly.  It checks:
  * ref/input_bitplane CSV loading through the IMCU constructor
  * greedy signed-ref noise injection in the compute path
  * no broadcast: an IMCU not described by the layout errors on compute
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


DEFAULT_NOISE_CSV = (
    "/root/project/CIM/noise/noise_df/B2_signed_weight_ref_out/N0/"
    "B2_noise_matrix_per_ch_concat_signed_weight_ref.csv"
)
DEFAULT_LAYOUT_JSON = (
    "/root/project/CIM/noise/noise_df/B2_signed_weight_ref_out/N0/"
    "concat_per_core.json"
)
SIM_ROOT = "/root/project/imcflow/pmap/ISA_sim/multi_core"


def make_imcu(noise_csv: str | None, layout_json: str | None, imce_linear_id: int):
    sys.path.insert(0, SIM_ROOT)
    from imcflow_sim.imcflow.imce import IMCU

    return IMCU(
        name=f"smoke.imce{imce_linear_id}.imcu",
        parent=None,
        noise_csv=noise_csv,
        noise_layout_json=layout_json,
        imce_linear_id=imce_linear_id,
        noise_mode="greedy",
        noise_table_format="ref" if noise_csv else "auto",
        noise_granularity="input_bitplane" if noise_csv else "auto",
    )


def configure_lsb_only_weight(imcu):
    """Set every weight to +1 so each input bitplane has signed ref 126."""
    imcu.mem.fill(0)
    imcu.mem[:, :, 3] = 1


def expected_mode_noise(imcu, signed_ref: int, n_out: int):
    csv_ch = imcu._csv_channels_for_outputs(n_out)
    ref_idx = imcu._nearest_ref_indices(np.full(n_out, signed_ref, dtype=np.int16))
    mode_bin = imcu.noise_tables["mode_bin_idx"]
    levels = imcu.noise_tables["noise_levels"]
    return np.rint(levels[mode_bin[csv_ch, ref_idx]]).astype(np.int16)


def run(args):
    noise_csv = str(Path(args.noise_csv))
    layout_json = str(Path(args.layout_json))

    inputs = np.full(256, 15, dtype=np.uint8)
    adcmode = 0  # ADCMode.SIX
    vmode = 1    # VMode.HALF, adc_divider = 2
    acc_mask = 0

    clean = make_imcu(None, None, imce_linear_id=0)
    noisy = make_imcu(noise_csv, layout_json, imce_linear_id=args.valid_imce)
    configure_lsb_only_weight(clean)
    configure_lsb_only_weight(noisy)

    clean_out = clean.compute(inputs, adcmode, vmode, acc_mask).astype(np.int32)
    noisy_out = noisy.compute(inputs, adcmode, vmode, acc_mask).astype(np.int32)

    # With all inputs=15 and all weights=+1, each input bitplane reconstructs
    # signed_ref=126 after quantization. Greedy ref noise is sampled once per
    # input bitplane and accumulated with weights 1+2+4+8 = 15.
    mode_noise = expected_mode_noise(noisy, signed_ref=126, n_out=64).astype(np.int32)
    expected_delta = 15 * mode_noise
    actual_delta = noisy_out - clean_out

    if not np.array_equal(actual_delta, expected_delta):
        mismatch = np.flatnonzero(actual_delta != expected_delta)
        first = int(mismatch[0])
        raise AssertionError(
            f"delta mismatch at column {first}: "
            f"actual={actual_delta[first]}, expected={expected_delta[first]}, "
            f"mode_noise={mode_noise[first]}"
        )

    print("VALID_IMCU_INFERENCE")
    print(f"  clean_sum={int(clean_out.sum())}")
    print(f"  noisy_sum={int(noisy_out.sum())}")
    print(f"  delta_sum={int(actual_delta.sum())}")
    print(f"  mode_noise_min={int(mode_noise.min())}")
    print(f"  mode_noise_max={int(mode_noise.max())}")
    print(f"  mode_noise_sum={int(mode_noise.sum())}")
    print(f"  unique_mode_noise={np.unique(mode_noise).tolist()}")
    print(f"  first8_clean={clean_out[:8].tolist()}")
    print(f"  first8_noisy={noisy_out[:8].tolist()}")
    print(f"  first8_delta={actual_delta[:8].tolist()}")
    print(f"  first8_expected_delta={expected_delta[:8].tolist()}")

    missing = make_imcu(noise_csv, layout_json, imce_linear_id=args.missing_imce)
    configure_lsb_only_weight(missing)
    try:
        missing.compute(inputs, adcmode, vmode, acc_mask)
    except RuntimeError as err:
        first_line = str(err).splitlines()[0]
        print("MISSING_IMCU_ERROR")
        print(f"  imce_linear_id={args.missing_imce}")
        print(f"  message={first_line}")
    else:
        raise AssertionError("missing IMCU unexpectedly completed; broadcast may be active")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-csv", default=DEFAULT_NOISE_CSV)
    parser.add_argument("--layout-json", default=DEFAULT_LAYOUT_JSON)
    parser.add_argument("--valid-imce", type=int, default=0)
    parser.add_argument("--missing-imce", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
