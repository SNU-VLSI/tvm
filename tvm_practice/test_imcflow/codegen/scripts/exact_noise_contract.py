"""Versioned, opt-in exact atomic-psum noise contract (CPU portable).
Keep this file identical in CIM/models and TVM/codegen/scripts; parity is tested.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

SCHEMA = "cim.exact-atomic-noise.v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(config):
    if not isinstance(config, dict):
        raise ValueError("noise_model must be an object")
    if "noise_model" not in config:
        raise ValueError("Expected top-level noise_model section")
    model = config["noise_model"]
    if not isinstance(model, dict):
        raise ValueError("noise_model must be an object")
    mode = model.get("distribution_mode", "histogram")
    allowed = {"distribution_mode"}
    if mode == "exact_psum":
        allowed |= {"min_samples", "insufficient_samples", "pool_max_distance", "unresolved_policy", "layer_specific"}
    elif mode != "histogram":
        raise ValueError(f"Unknown noise_model.distribution_mode: {mode}")
    if set(model) - allowed:
        raise ValueError(f"Unsupported noise_model options for {mode}: {sorted(set(model) - allowed)}")
    result = {"distribution_mode": mode}
    if mode == "exact_psum":
        result.update(min_samples=100, insufficient_samples="nearest_pool",
                      pool_max_distance=32, unresolved_policy="error", layer_specific=False)
        result.update(model)
        for key, minimum in (("min_samples", 1), ("pool_max_distance", 0)):
            if type(result[key]) is not int or result[key] < minimum:
                raise ValueError(f"noise_model.{key} must be an integer >= {minimum}")
        if type(result["layer_specific"]) is not bool:
            raise ValueError("noise_model.layer_specific must be boolean")
        if result["insufficient_samples"] != "nearest_pool" or result["unresolved_policy"] not in (
                "error", "nearest_sufficient", "nearest_local_sufficient", "nearest_local_or_max_count"):
            raise ValueError("Unsupported local-pool fallback policy")
        result.update(ref_axis="exact_integer", noise_axis="exact_integer", granularity="atomic_final")
    return {"schema_version": 1, "noise_model": result}


def load_contract(path):
    with open(path) as stream:
        payload = json.load(stream)
    # Resolved contracts contain derived fields; verify instead of trusting them.
    if "schema_version" in payload:
        if payload["schema_version"] != 1 or set(payload) != {"schema_version", "noise_model"}:
            raise ValueError("Unknown resolved noise contract schema")
        raw = dict(payload["noise_model"])
        for key in ("ref_axis", "noise_axis", "granularity"):
            raw.pop(key, None)
        expected = resolve({"noise_model": raw})
        if payload != expected:
            raise ValueError("Invalid derived noise contract fields")
        return expected
    return resolve(payload)


def integer_array(values):
    values = np.asarray(values)
    if not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all():
        raise ValueError("Exact noise requires finite integer observations")
    if values.size and (values.min() < -32768 or values.max() > 32767):
        raise ValueError("Exact atomic observation outside signed int16 domain")
    return values.astype(np.int64)


class ExactTable:
    """Sparse raw counts with deterministic, same-channel raw-only pooling."""
    def __init__(self, contract, n_channels, provenance=None):
        if contract["noise_model"]["distribution_mode"] != "exact_psum":
            raise ValueError("ExactTable requires explicit exact_psum contract")
        if type(n_channels) is not int or n_channels < 1:
            raise ValueError("Invalid channel count")
        self.contract = contract
        self.n_channels = n_channels
        self.provenance = dict(provenance or {})
        self.rows = {}
        self._resolved = {}
        self._channel_rows = None
        self._local_donor_refs = {}
        self._max_local_refs = {}

    def add_batch(self, channels, refs, noises, layer=""):
        refs, noises = integer_array(refs), integer_array(noises)
        channels = np.asarray(channels)
        if refs.shape != noises.shape or refs.ndim < 1 or len(channels) != refs.shape[0]:
            raise ValueError("Channel/ref/noise shape mismatch")
        if not self.contract["noise_model"]["layer_specific"]:
            layer = ""
        elif not layer:
            raise ValueError("layer_specific exact table requires a layer name")
        for ch, rs, ns in zip(channels, refs, noises):
            if int(ch) != ch or not 0 <= int(ch) < self.n_channels:
                raise ValueError("Invalid pseudo-channel")
            triples, counts = np.unique(np.stack([rs.ravel(), ns.ravel()], 1), axis=0, return_counts=True)
            for (ref, noise), count in zip(triples, counts):
                self.rows.setdefault((str(layer), int(ch), int(ref)), Counter())[int(noise)] += int(count)
        self._resolved.clear()
        self._channel_rows = None
        self._local_donor_refs.clear()
        self._max_local_refs.clear()

    def _local_pool(self, channel, ref, layer):
        """Use only original observations, stopping after a complete distance shell."""
        model = self.contract["noise_model"]
        combined, donors = Counter(), []
        for distance in range(model["pool_max_distance"] + 1):
            neighbors = [int(ref)] if distance == 0 else [int(ref) - distance, int(ref) + distance]
            for neighbor in neighbors:
                counts = self.rows.get((layer, int(channel), neighbor))
                if counts:
                    combined.update(counts)
                    donors.append({"ref": neighbor, "count": sum(counts.values())})
            if sum(combined.values()) >= model["min_samples"]:
                break
        return combined, donors

    def _sufficient_local_centers(self, channel, layer):
        """Index eligible integer centers, including previously unobserved psums.

        This index never reads resolved/fallback rows. A donor qualifies only on
        original counts inside its own bounded local neighborhood.
        """
        key = (layer, int(channel))
        if key in self._local_donor_refs:
            return self._local_donor_refs[key]
        if self._channel_rows is None:
            self._channel_rows = {}
            for (l, ch, r), counts in self.rows.items():
                self._channel_rows.setdefault((l, ch), []).append((r, sum(counts.values())))
        entries = sorted(self._channel_rows.get(key, []))
        if not entries:
            candidates = np.array([], dtype=np.int64)
            self._max_local_refs[key] = candidates
        else:
            refs, counts = np.array(entries, dtype=np.int64).T
            radius = self.contract["noise_model"]["pool_max_distance"]
            centers = np.arange(max(-32768, int(refs[0])-radius),
                                min(32767, int(refs[-1])+radius)+1, dtype=np.int64)
            prefix = np.r_[0, counts.cumsum()]
            mass = (prefix[np.searchsorted(refs, centers+radius, side="right")]
                    - prefix[np.searchsorted(refs, centers-radius, side="left")])
            candidates = centers[mass >= self.contract["noise_model"]["min_samples"]]
            self._max_local_refs[key] = centers[mass == mass.max()]
        self._local_donor_refs[key] = candidates
        return candidates

    def distribution(self, channel, ref, layer=""):
        if not np.isfinite(ref) or int(ref) != ref or not -32768 <= ref <= 32767:
            raise ValueError("Exact lookup requires signed integer atomic psum")
        if not 0 <= channel < self.n_channels:
            raise ValueError("Invalid lookup channel")
        model = self.contract["noise_model"]
        layer = str(layer) if model["layer_specific"] else ""
        key = (layer, int(channel), int(ref))
        if key in self._resolved:
            return self._resolved[key]
        combined, donors = self._local_pool(channel, ref, layer)
        total = sum(combined.values())
        resolution = "exact" if len(donors) == 1 and donors[0]["ref"] == ref else "local_pool"
        local_count = total
        selected_ref = int(ref)
        if total < model["min_samples"] and model["unresolved_policy"] == "nearest_sufficient":
            # Borrow one original distribution verbatim, never extend the local pool.
            # Lower ref wins equal-distance ties; never combine the tied distributions.
            eligible = [(abs(r - ref), r, counts) for (l, ch, r), counts in self.rows.items()
                        if l == layer and ch == int(channel) and sum(counts.values()) >= model["min_samples"]]
            if eligible:
                _, donor_ref, combined = min(eligible, key=lambda entry: (entry[0], entry[1]))
                total = sum(combined.values())
                donors = [{"ref": donor_ref, "count": total}]
                resolution = "nearest_sufficient"
                selected_ref = donor_ref
        if total < model["min_samples"] and model["unresolved_policy"] in (
                "nearest_local_sufficient", "nearest_local_or_max_count"):
            candidates = self._sufficient_local_centers(channel, layer)
            if candidates.size:
                # Sorted order provides a deterministic lower-psum tie break.
                selected_ref = int(candidates[np.argmin(np.abs(candidates - int(ref)))])
                combined, donors = self._local_pool(channel, selected_ref, layer)
                total = sum(combined.values())
                resolution = "nearest_local_sufficient"
            elif model["unresolved_policy"] == "nearest_local_or_max_count":
                candidates = self._max_local_refs[(layer, int(channel))]
                if candidates.size:
                    selected_ref = int(candidates[np.argmin(np.abs(candidates - int(ref)))])
                    combined, donors = self._local_pool(channel, selected_ref, layer)
                    total = sum(combined.values())
                    resolution = "max_count_local"
        low_confidence = resolution == "max_count_local" and total > 0
        if total < model["min_samples"] and not low_confidence:
            raise ValueError(f"Unresolved exact noise layer={layer!r} channel={channel} psum={ref}: "
                             f"{total}/{model['min_samples']} samples within distance {model['pool_max_distance']}; "
                             f"policy={model['unresolved_policy']} found no sufficient distribution")
        levels = np.array(sorted(combined), dtype=np.int64)
        probabilities = np.array([combined[int(x)] / total for x in levels], dtype=np.float64)
        diagnostics = {"raw_count": sum(self.rows.get(key, {}).values()), "effective_count": total,
                       "donors": donors, "max_distance": max(abs(d["ref"] - ref) for d in donors),
                       "resolution": resolution, "local_count": local_count,
                       "low_confidence": low_confidence,
                       "distribution_ref": selected_ref, "borrow_distance": abs(selected_ref - int(ref)),
                       "pool_radius_used": max(abs(d["ref"] - selected_ref) for d in donors)}
        self._resolved[key] = (levels, probabilities, diagnostics)
        return self._resolved[key]

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        entries = [(layer, ch, ref, noise, count) for (layer, ch, ref), counts in sorted(self.rows.items())
                   for noise, count in sorted(counts.items())]
        if not entries:
            raise ValueError("Cannot save empty exact noise table")
        names = sorted({entry[0] for entry in entries})
        indexes = {name: i for i, name in enumerate(names)}
        records = np.array([(indexes[l], c, r, n, k) for l, c, r, n, k in entries], dtype=np.int64)
        metadata = {"format": SCHEMA, "contract": self.contract, "contract_hash": digest(self.contract),
                    "n_channels": self.n_channels, "provenance": self.provenance}
        metadata["records_hash"] = hashlib.sha256(records.tobytes() + json.dumps(names).encode()).hexdigest()
        with path.open("xb") as stream:
            np.savez_compressed(stream, metadata=json.dumps(metadata, sort_keys=True),
                                layers=np.array(names), records=records)

    @classmethod
    def load(cls, path, contract):
        with np.load(path, allow_pickle=False) as raw:
            if set(raw.files) != {"metadata", "layers", "records"}:
                raise ValueError("Expected versioned exact noise artifact, not legacy histogram/CSV")
            meta = json.loads(str(raw["metadata"]))
            records, names = raw["records"], raw["layers"].tolist()
        if meta.get("format") != SCHEMA or meta.get("contract") != contract or meta.get("contract_hash") != digest(contract):
            raise ValueError("Exact artifact/config contract mismatch")
        if records.dtype != np.int64 or records.ndim != 2 or records.shape[1] != 5:
            raise ValueError("Invalid exact records")
        if meta.get("records_hash") != hashlib.sha256(records.tobytes() + json.dumps(names).encode()).hexdigest():
            raise ValueError("Exact artifact content hash mismatch")
        table = cls(contract, meta["n_channels"], meta["provenance"])
        for l, c, r, n, count in records.tolist():
            if not 0 <= l < len(names) or not 0 <= c < table.n_channels or count <= 0:
                raise ValueError("Invalid exact record index/count")
            if not -32768 <= r <= 32767 or not -32768 <= n <= 32767:
                raise ValueError("Invalid exact record value")
            key = (names[l], c, r)
            row = table.rows.setdefault(key, Counter())
            if n in row:
                raise ValueError("Duplicate exact record")
            row[n] = count
        return table
