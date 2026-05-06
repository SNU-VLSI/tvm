#!/usr/bin/env python3
"""Standalone CLI for the psum -> IMCU/column mapping dump.

Use this after a successful compile (single_qconv v1 or driver_v2) to
regenerate ``psum_imcu_column_map.npz`` without re-running the compiler.

Examples:

    # Default: write to <eval_dir>/psum_imcu_column_map.npz
    python scripts/dump_psum_mapping.py \\
        --eval-dir eval_dir/resnet8_subset31_pretrained_orig_evl.linux

    # Custom output path
    python scripts/dump_psum_mapping.py \\
        --eval-dir eval_dir/<model>_evl.linux \\
        --output /tmp/my_mapping.npz

The same dump can be triggered automatically at compile time by setting
the env var ``IMCFLOW_DUMP_PSUM_MAP=1`` before running ``main.py``; the
output lands at the same default path.
"""

import argparse
import sys

from tvm.relay.backend.contrib.imcflow.psum_mapping import dump_psum_mapping_offline


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--eval-dir", required=True,
        help="Path to a model's eval dir, e.g. eval_dir/<model>_evl.linux/")
    parser.add_argument(
        "--output", default=None,
        help="Output npz path (default: {eval_dir}/psum_imcu_column_map.npz)")
    args = parser.parse_args()

    out_path = dump_psum_mapping_offline(args.eval_dir, args.output)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
