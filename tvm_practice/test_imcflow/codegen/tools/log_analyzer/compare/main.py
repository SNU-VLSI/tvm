"""CLI entry point for ``python -m log_analyzer.compare``."""

import argparse
import os
import sys

from .comparator import compare_all, compare_records, format_results, COMPARE_FIELDS
from .extractor import extract_records
from .rtl_compat import extract_rtl_records

# Normalised event names for the --compare-* flags.
_EVENT_GROUPS = {
    "mm_quant": {"MM_QUANT"},
    "imcu": {"IMCU_OUTPUT"},
    "vpu": {"MULTL", "ADD", "SUB", "MULTH"},
    "dwconv": {"DWCONV"},
    "linebuffer": {"LBUF_INPUT"},
}


def _pysim_log_path(test_dir):
    return os.path.join(test_dir, "logs", "py_runner", "now.debug.log")


def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m log_analyzer.compare",
        description="Compare pysim and RTL simulation outputs.",
    )
    p.add_argument("test_dir", help="Test output directory")

    grp = p.add_argument_group("comparison selectors")
    grp.add_argument("--compare-all", action="store_true", help="Compare all event types")
    grp.add_argument("--compare-mm-quant", action="store_true", help="Compare MM_QUANT outputs")
    grp.add_argument("--compare-imcu", action="store_true", help="Compare IMCU outputs")
    grp.add_argument("--compare-vpu", action="store_true", help="Compare VPU ops (MULTL, ADD, SUB)")
    grp.add_argument("--compare-dwconv", action="store_true", help="Compare DWCONV outputs")
    grp.add_argument("--compare-linebuffer", action="store_true", help="Compare linebuffer inputs")

    p.add_argument("--imce", type=str, metavar="R,C", help="Filter to IMCE coordinate (e.g. 3,2)")
    p.add_argument("-v", "--verbose", action="store_true", help="Show all mismatches")
    p.add_argument("--max-mismatches", type=int, default=5,
                   help="Max mismatches to display per (coord, event) (default: 5)")

    # Diagnostic
    p.add_argument("--show-pysim", action="store_true", help="Dump pysim records")
    p.add_argument("--show-rtl", action="store_true", help="Dump RTL records")

    return p


def _resolve_events(args):
    """Return the set of normalised event names the user wants to compare."""
    if args.compare_all:
        return None  # all
    events = set()
    for flag, evts in _EVENT_GROUPS.items():
        if getattr(args, f"compare_{flag}", False):
            events |= evts
    return events or None  # fall back to all if nothing specified


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    test_dir = args.test_dir
    pysim_path = _pysim_log_path(test_dir)

    if not os.path.isfile(pysim_path):
        print(f"Pysim log not found: {pysim_path}", file=sys.stderr)
        sys.exit(1)

    coord_filter = None
    if args.imce:
        r, c = args.imce.split(",")
        coord_filter = (int(r), int(c))

    events = _resolve_events(args)

    # Diagnostic dumps
    if args.show_pysim:
        records = extract_records(pysim_path, source="pysim")
        _dump_records("pysim", records, coord_filter)
        if not args.show_rtl and not events:
            return

    if args.show_rtl:
        records = extract_rtl_records(test_dir, coord_filter=coord_filter)
        _dump_records("rtl", records, coord_filter)
        if not events:
            return

    # Run comparison
    results = compare_all(
        pysim_path, test_dir,
        events=events,
        coord_filter=coord_filter,
        verbose=args.verbose,
    )

    if not results:
        print("No matching events found for comparison.")
        sys.exit(0)

    lines = format_results(results, verbose=args.verbose, max_mismatches=args.max_mismatches)
    for line in lines:
        print(line)

    # Exit code: 1 if any mismatches
    has_mismatch = any(r.mismatches for r in results)
    sys.exit(1 if has_mismatch else 0)


def _dump_records(label, grouped, coord_filter):
    """Print records grouped by (coord, event) for debugging."""
    print(f"\n{'=' * 60}")
    print(f"  {label.upper()} records")
    print(f"{'=' * 60}")
    for key in sorted(grouped):
        coord, event = key
        if coord_filter and coord != coord_filter:
            continue
        recs = grouped[key]
        print(f"\n  IMCE({coord[0]},{coord[1]}) {event}: {len(recs)} records")
        for rec in recs[:5]:
            print(f"    [{rec.index}] @{rec.timestamp} {rec.fields}")
        if len(recs) > 5:
            print(f"    ... {len(recs) - 5} more")
