"""Comparison engine for pysim vs RTL OpRecords."""

import os
from collections import defaultdict

from .models import OpRecord, CompareResult
from .extractor import extract_records
from .rtl_compat import extract_rtl_records

# Fields to compare for each normalised event name.
# Both pysim and RTL OpRecords must store these field names in their
# ``fields`` dict.  The extractor and rtl_compat modules are responsible
# for producing matching keys.
COMPARE_FIELDS = {
    "MM_QUANT": ["output"],
    "IMCU_OUTPUT": ["result"],
    "MULTL": ["result"],
    "ADD": ["result"],
    "SUB": ["result"],
    "MULTH": ["result"],
    "DWCONV": ["result"],
    "LBUF_INPUT": ["data"],
}

# For IMCU_OUTPUT we only compare the first 16 elements.
_TRUNCATE = {
    "IMCU_OUTPUT": {"result": 16},
}


def _field_val(record, field_name, event):
    """Get a comparable field value from *record*, optionally truncated."""
    val = record.fields.get(field_name)
    trunc = _TRUNCATE.get(event, {}).get(field_name)
    if trunc and isinstance(val, list):
        return val[:trunc]
    return val


def compare_records(pysim_records, rtl_records, event, compare_fields=None):
    """Compare two aligned lists of :class:`OpRecord`.

    Parameters
    ----------
    pysim_records, rtl_records : list[OpRecord]
    event : str
        Normalised event name (e.g. ``"MM_QUANT"``).
    compare_fields : list[str] | None
        Fields to diff.  Defaults to :data:`COMPARE_FIELDS[event]`.

    Returns
    -------
    CompareResult
    """
    fields = compare_fields or COMPARE_FIELDS.get(event, [])
    if not fields:
        fields = list(
            (pysim_records[0].fields.keys() & rtl_records[0].fields.keys())
            if pysim_records and rtl_records
            else []
        )

    pysim_count = len(pysim_records)
    rtl_count = len(rtl_records)
    n = min(pysim_count, rtl_count)

    matches = 0
    mismatches = []

    for i in range(n):
        p, r = pysim_records[i], rtl_records[i]
        diff_fields = []
        for f in fields:
            pv = _field_val(p, f, event)
            rv = _field_val(r, f, event)
            if pv != rv:
                diff_fields.append(f)

        if diff_fields:
            mismatches.append({
                "index": i,
                "pysim": {f: _field_val(p, f, event) for f in diff_fields},
                "rtl": {f: _field_val(r, f, event) for f in diff_fields},
                "diff_fields": diff_fields,
            })
        else:
            matches += 1

    return CompareResult(
        event=event,
        imce_coord=pysim_records[0].imce_coord if pysim_records else rtl_records[0].imce_coord if rtl_records else (0, 0),
        total=max(pysim_count, rtl_count),
        matches=matches,
        mismatches=mismatches,
        pysim_count=pysim_count,
        rtl_count=rtl_count,
    )


def compare_all(pysim_path, rtl_dir, events=None, coord_filter=None, verbose=False):
    """Full comparison pipeline.

    1. Extract pysim records via structured-log parser.
    2. Extract RTL records via regex compat parsers.
    3. Match by ``(coord, event)`` and compare.

    Parameters
    ----------
    pysim_path : str
        Path to pysim structured log file.
    rtl_dir : str
        Test directory root (containing ``logs/rtl_runner/fsim_logs/``).
    events : set[str] | None
        Normalised event names to compare.  ``None`` → all available.
    coord_filter : tuple | None
        Restrict to a single IMCE coordinate.

    Returns
    -------
    list[CompareResult]
    """
    pysim_records = extract_records(pysim_path, source="pysim")
    rtl_records = extract_rtl_records(rtl_dir, coord_filter=coord_filter)

    # Filter DWCONV to only RESULT entries on RTL side for fair comparison
    for key in list(rtl_records):
        coord, ev = key
        if ev == "DWCONV":
            rtl_records[key] = [
                r for r in rtl_records[key]
                if r.fields.get("stage") == "result"
            ]

    # Collect all (coord, event) keys present in at least one source
    all_keys = set(pysim_records.keys()) | set(rtl_records.keys())

    results = []
    for key in sorted(all_keys):
        coord, ev = key
        if events and ev not in events:
            continue
        if coord_filter and coord != coord_filter:
            continue

        p = pysim_records.get(key, [])
        r = rtl_records.get(key, [])

        if not p and not r:
            continue

        res = compare_records(p, r, ev)
        results.append(res)

    return results


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def format_results(results, verbose=False, max_mismatches=5):
    """Format a list of :class:`CompareResult` for terminal output.

    Returns a list of lines (no trailing newlines).
    """
    lines = []

    # Group by event
    by_event = defaultdict(list)
    for r in results:
        by_event[r.event].append(r)

    for event in sorted(by_event):
        lines.append(f"{'=' * 70}")
        lines.append(f"  {event}")
        lines.append(f"{'=' * 70}")

        total_match = 0
        total_mismatch = 0

        for r in sorted(by_event[event], key=lambda x: x.imce_coord):
            coord = r.imce_coord
            tag = f"IMCE({coord[0]},{coord[1]})"
            n_mis = len(r.mismatches)
            total_match += r.matches
            total_mismatch += n_mis

            if r.pysim_count == 0:
                lines.append(f"  {tag}: RTL only ({r.rtl_count} records, no pysim data)")
                continue
            if r.rtl_count == 0:
                lines.append(f"  {tag}: pysim only ({r.pysim_count} records, no RTL data)")
                continue

            if n_mis == 0:
                lines.append(f"  {tag}: {r.matches}/{r.total} OK")
            else:
                lines.append(f"  {tag}: {r.matches} match, {n_mis} MISMATCH (of {r.total})")

            if r.pysim_count != r.rtl_count:
                lines.append(f"    count: pysim={r.pysim_count} rtl={r.rtl_count}")

            show = r.mismatches if verbose else r.mismatches[:max_mismatches]
            for mm in show:
                lines.append(f"    [{mm['index']}] diff fields: {mm['diff_fields']}")
                for f in mm["diff_fields"]:
                    pv = mm["pysim"].get(f)
                    rv = mm["rtl"].get(f)
                    lines.append(f"      pysim.{f} = {_abbrev(pv)}")
                    lines.append(f"      rtl.{f}   = {_abbrev(rv)}")
                    if isinstance(pv, list) and isinstance(rv, list):
                        diff = [a - b for a, b in zip(pv, rv)]
                        lines.append(f"      diff     = {_abbrev(diff)}")

            if not verbose and len(r.mismatches) > max_mismatches:
                lines.append(f"    ... and {len(r.mismatches) - max_mismatches} more")

        lines.append(f"  TOTAL {event}: {total_match} match, {total_mismatch} mismatch")
        lines.append("")

    return lines


def _abbrev(val, limit=80):
    """Abbreviate long lists for display."""
    s = str(val)
    if len(s) > limit:
        return s[: limit - 3] + "..."
    return s
