"""Bind a fetched chip result to its run, staged rows and compile/dataset hashes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_metadata(result, dataset, staged, build_metadata, run_id, target):
    result, dataset, build_metadata = Path(result), Path(dataset), Path(build_metadata)
    with result.open() as stream:
        run_ids = [line.strip().split(" ", 1)[1] for line in stream if line.startswith("DAE_RUN ")]
    if run_ids != [run_id]:
        raise ValueError("Missing/mismatched DAE run ID; refusing stale or mixed chip results")
    metadata = json.loads((dataset / "metadata.json").read_text())
    if metadata.get("input_domain") != "before_x_f_1" or metadata["shape"][1:] != [640]:
        raise ValueError("Unexpected DAE input domain/shape")
    build = json.loads(build_metadata.read_text())
    if (build.get("model_name") != "dae_toycar_full_pretrained" or build.get("board") != "B1" or
            not build.get("driver_v2") or build.get("num_disable_columns") != 32 or build.get("acc_mask") != 1):
        raise ValueError("DAE chip result requires pretrained driver-v2/B1/N32/ACC_MASK=1 build provenance")
    source = Path(staged) if staged else dataset
    if staged:
        mapping = json.loads((source / "sample_map.json").read_text())
        ids = mapping["staged_to_original"]
        if (mapping["source_num_samples"] != metadata["shape"][0] or
                len(ids) != mapping["num_staged"] or len(set(ids)) != len(ids) or
                any(type(i) is not int or not 0 <= i < metadata["shape"][0] for i in ids)):
            raise ValueError("Invalid staged-to-original sample map")
        if Path(mapping["source_dir"]).resolve() != dataset.resolve():
            raise ValueError("Stage map refers to a different source dataset")
        for name in ("images", "labels"):
            original = np.load(dataset / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            selected = np.load(source / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            if selected.dtype != original.dtype or not np.array_equal(selected, original[ids]):
                raise ValueError(f"Staged {name} does not match its original window IDs")
    else:
        mapping = dict(source_num_samples=metadata["shape"][0], identity=True)
    map_path = Path(str(result) + ".sample_map.json")
    with map_path.open("x") as stream:
        json.dump(mapping, stream, indent=2)
    provenance = dict(schema_version=1, backend="chip", run_id=run_id, target=target,
        result_sha256=sha256(result), sample_map_sha256=sha256(map_path),
        dataset_metadata_sha256=sha256(dataset / "metadata.json"),
        source_images_sha256=metadata["images_sha256"], source_labels_sha256=metadata["labels_sha256"],
        staged_images_sha256=sha256(source / "images.npy"), staged_labels_sha256=sha256(source / "labels.npy"),
        build_metadata_sha256=sha256(build_metadata), compile_metadata=build,
        checkpoint_sha256=sha256(build["checkpoint_path"]))
    with Path(str(result) + ".provenance.json").open("x") as stream:
        json.dump(provenance, stream, indent=2)
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--staged")
    parser.add_argument("--build-metadata", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    save_metadata(args.result, args.dataset, args.staged, args.build_metadata, args.run_id, args.target)
