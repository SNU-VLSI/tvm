#!/bin/bash

SRC_BASE=~/project/imcflow/xilinx/measurement/workspace

for row in 0 1 2 3; do
    for col in 1 2 3 4; do
        src="${SRC_BASE}/RatioScanPlanner_${row}_${col}/results/opt_scan.npz"
        dst="imce_${row}_${col}.npz"
        echo "cp ${src} ${dst}"
        cp "${src}" "${dst}"
    done
done
