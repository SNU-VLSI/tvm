"""Compare logical DAE boundaries in DEBUG_EXE runner dumps with deploy traces.

The compiled graph already restores disabled-column/atomic OC splitting through
take and concatenate. Read the input of each BN, not raw physical IMCU outputs.
Graph structure and shapes are checked; numeric node indices are never fixed.
A single-sample report is diagnostic, not the 10-input/two-repeat integration gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np


CHANNELS = ((128, 128),) * 3 + ((128, 8), (8, 128)) + ((128, 128),) * 3
FP_BOUNDARIES = {"front_fp", "head_input_fp", "reconstruction"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_graph(mlf):
    with tarfile.open(mlf) as archive:
        names = [n for n in archive.getnames() if n.endswith("/graph/default.graph")]
        if len(names) != 1:
            raise ValueError("Expected one default graph in MLF")
        return json.load(archive.extractfile(names[0]))


def logical_boundaries(graph):
    nodes = graph["nodes"]
    rows = graph["node_row_ptr"]
    if any(b - a != 1 for a, b in zip(rows, rows[1:])):
        raise ValueError("DAE comparison expects single-output graph nodes")

    def name(index):
        return nodes[index]["name"]

    def source(index):
        return nodes[index]["inputs"][0][0]

    def shape(index):
        return graph["attrs"]["shape"][1][rows[index]]

    def require(index, operation):
        if operation not in name(index):
            raise ValueError(f"Expected {operation} at node {index}: {name(index)}")
        return index

    def unreshape(index):
        while name(index) == "reshape_nop":
            index = source(index)
        return index

    quant = [i for i in range(len(nodes)) if "qnn_imcflow_min_max_quantize" in name(i)]
    bn = [i for i in range(len(nodes)) if "imcflow_fused_batch_norm" in name(i)]
    if len(quant) != 8 or len(bn) != 8:
        raise ValueError(f"Expected eight quantizers and BNs, got {len(quant)}, {len(bn)}")
    front_cast = require(unreshape(source(quant[0])), "fused_cast")
    clip = require(source(front_cast), "fused_clip")
    scale = require(source(clip), "fused_multiply")
    front_fp = require(source(scale), "fused_add")
    boundaries = {"front_fp": front_fp, "front_int16": source(quant[0])}
    covered_atomics = set()

    def restored_atomics(index, activation):
        if "fused_concatenate" in name(index):
            result = []
            for edge in nodes[index]["inputs"]:
                result.extend(restored_atomics(edge[0], activation))
            return result
        take = require(index, "fused_take")
        layout = require(source(take), "fused_layout_transform")
        atomic = require(source(layout), "tvmgen_default_imcflow_main_")
        bitpack = require(source(atomic), "fused_nn_bitpack")
        if source(bitpack) != activation:
            raise ValueError("Atomic qconv is connected to the wrong logical activation")
        return [atomic]

    for block, ((ic, oc), q, b) in enumerate(zip(CHANNELS, quant, bn)):
        if block and unreshape(source(q)) != bn[block - 1]:
            raise ValueError(f"Broken DAE chain before block {block}")
        linear = source(b)
        if shape(q) != [1, ic, 1, 1] or shape(linear) != [1, oc, 1, 1] or shape(b) != [1, oc, 1, 1]:
            raise ValueError(f"Unexpected logical shape at block {block}")
        atomics = restored_atomics(linear, q)
        if len(atomics) != (oc + 31) // 32 or len(set(atomics)) != len(atomics):
            raise ValueError(f"Expected N32 OC split at block {block}")
        if covered_atomics.intersection(atomics):
            raise ValueError("Atomic qconv belongs to multiple blocks")
        covered_atomics.update(atomics)
        boundaries.update({f"block{block}.act": q, f"block{block}.linear": linear,
                           f"block{block}.bn": b, f"block{block}.output": b})
    all_atomics = {i for i in range(len(nodes)) if name(i).startswith("tvmgen_default_imcflow_main_")}
    if covered_atomics != all_atomics:
        raise ValueError("Graph contains unmapped atomic qconvs")
    if len(graph["heads"]) != 1:
        raise ValueError("Expected one reconstruction output")
    reconstruction = require(graph["heads"][0][0], "fused_nn_bias_add")
    dense = require(source(reconstruction), "fused_nn_dense")
    relu = require(source(dense), "fused_nn_relu")
    head = source(relu)
    multiply = require(unreshape(head), "fused_multiply")
    cast = require(source(multiply), "fused_cast")
    if source(cast) != bn[-1] or shape(head) != [1, 128] or shape(reconstruction) != [1, 640]:
        raise ValueError("Unexpected dequantized DAE head")
    boundaries.update(head_input_fp=head, reconstruction=reconstruction)
    return boundaries


def read_dump(graph, directory, index):
    node = graph["nodes"][index]
    if node["name"] == "reshape_nop":
        value = read_dump(graph, directory, node["inputs"][0][0])
    else:
        value = np.load(Path(directory) / f"{index:03d}_{node['name']}.npy", allow_pickle=False)
    row = graph["node_row_ptr"][index]
    expected = tuple(graph["attrs"]["shape"][1][row])
    dtype = np.dtype(graph["attrs"]["dltype"][1][row])
    if value.dtype != dtype or value.size != int(np.prod(expected)):
        raise ValueError(f"Dump shape/dtype differs from graph at node {index}")
    return value.reshape(expected)


def compare_tensor(reference, actual, integer):
    if reference.shape != actual.shape:
        raise ValueError(f"Logical shapes differ: {reference.shape} vs {actual.shape}")
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("Nonfinite value at comparison boundary")
    if integer and (reference.dtype.kind not in "iu" or actual.dtype.kind not in "iu"):
        raise ValueError("Integer boundary requires integer tensors (no rounding allowed)")
    expected = reference.astype(np.float64)
    diff = np.abs(actual.astype(np.float64) - expected)
    mismatch = diff > 1 if integer else diff > (1e-3 + 1e-3 * np.abs(expected))
    relative = diff / np.maximum(np.abs(expected), 1e-12)
    examples = [dict(index=index.tolist(), reference=reference[tuple(index)].item(),
                     actual=actual[tuple(index)].item()) for index in np.argwhere(mismatch)[:5]]
    return dict(passed=not bool(mismatch.any()), mismatched_elements=int(mismatch.sum()),
                elements=reference.size, max_absolute_error=float(diff.max()),
                max_relative_error=float(relative.max()), integer_boundary=integer,
                integer_atol=1 if integer else None, differing_elements=int(np.count_nonzero(diff)),
                relative_denominator_floor=1e-12, mismatch_examples=examples)


def compare_sample(mlf, sample_directory, deploy_capture, sample=0, repeat=0):
    graph = read_graph(mlf)
    mapping = logical_boundaries(graph)
    results = {}
    with np.load(Path(deploy_capture) / "traces.npz", allow_pickle=False) as traces:
        for name, index in mapping.items():
            actual = read_dump(graph, sample_directory, index)
            reference = traces[f"repeat{repeat}/sample{sample}/{name}"]
            results[name] = dict(node=index, node_name=graph["nodes"][index]["name"],
                                 **compare_tensor(reference, actual, name not in FP_BOUNDARIES))
    return dict(schema_version=2, sample=sample, repeat=repeat,
                passed=all(r["passed"] for r in results.values()), complete_gate=False,
                note="Single-sample numeric diagnostic; provenance and 10x2 gate are separate.",
                first_mismatch=next((n for n, r in results.items() if not r["passed"]), None),
                atol=1e-3, rtol=1e-3, integer_atol=1, integer_rtol=0, mlf_sha256=sha256(mlf),
                deploy_trace_sha256=sha256(Path(deploy_capture) / "traces.npz"),
                boundaries=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlf", required=True)
    parser.add_argument("--sample-directory", required=True)
    parser.add_argument("--deploy-capture", required=True)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--repeat", type=int, choices=(0, 1), default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = compare_sample(args.mlf, args.sample_directory, args.deploy_capture, args.sample, args.repeat)
    with Path(args.output).open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k != "boundaries"}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
