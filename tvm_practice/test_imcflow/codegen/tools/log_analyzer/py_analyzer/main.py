"""CLI entry point for Python Simulator Log Analyzer."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Python Simulator Log Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # (placeholder — analysis commands will be added here)
  %(prog)s --help
""",
    )

    parser.add_argument(
        "--log-file",
        "-f",
        default=None,
        help="Path to the pysim log file",
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
