#!/usr/bin/env python3
"""
Command-line interface for running IMCFLOW model tests manually.

This script provides a user-friendly CLI for testing IMCFLOW models with
different input patterns, with options to skip expensive setup steps when
experimenting with different inputs on an already-compiled model.
"""

import argparse
import sys
import subprocess
import os
from test import MODEL_REGISTRY, INPUT_PATTERNS, run_test_pipeline
from pipeline_options import PipelineOptions, PipelineStage, parse_stop_at


def main():
  """Command-line interface for running tests manually"""
  parser = argparse.ArgumentParser(
    description="Run IMCFLOW model tests with various input patterns",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Run with default input pattern (full setup)
  python main.py --model one_relu

  # Run with specific input pattern
  python main.py --model one_relu --pattern random

  # Stop at specific pipeline stage
  python main.py --model one_relu --stop-at transform   # Frontend only
  python main.py --model one_relu --stop-at compile     # Compile only

  # Rebuild modified C++ files only (skip TVM transform)
  python main.py --model one_relu --rebuild-cpp-only

  # Stop at codegen to observe memlayout changes
  python main.py --model one_relu --stop-at codegen --rebuild-cpp-only

  # Skip setup (reuse existing compiled model)
  python main.py --model one_relu --pattern zeros --skip-setup

  # List available models
  python main.py --list-models
    """
  )

  parser.add_argument(
    "--model", "-m",
    type=str,
    help=f"Model to test. Available: {', '.join(MODEL_REGISTRY.keys())}"
  )

  parser.add_argument(
    "--pattern", "-p",
    type=str,
    choices=INPUT_PATTERNS,
    help=f"Input pattern to use. Choices: {', '.join(INPUT_PATTERNS)}. Default: model's default pattern"
  )

  parser.add_argument(
    "--stop-at",
    type=str,
    choices=["transform", "codegen", "compile", "full"],
    default="full",
    help="Pipeline stage to stop at: transform (frontend only), codegen, compile (skip simulation), full (default)"
  )

  parser.add_argument(
    "--skip-setup", "-s",
    action="store_true",
    help="Skip model transformation, codegen, and graph generation (reuse existing compiled model)"
  )

  parser.add_argument(
    "--rebuild-cpp-only", "-r",
    action="store_true",
    help="Rebuild modified C++ files only (skip TVM transform, use saved DevConfig state)"
  )

  parser.add_argument(
    "--list-models", "-l",
    action="store_true",
    help="List available models and their default input patterns"
  )

  args = parser.parse_args()

  # Handle list models
  if args.list_models:
    print("\n" + "="*60)
    print("Available Models")
    print("="*60)
    for model_name, (_, default_pattern) in MODEL_REGISTRY.items():
      print(f"  {model_name:<40} (default: {default_pattern})")
    print(f"\nAvailable input patterns: {', '.join(INPUT_PATTERNS)}")
    return 0

  # Require model if not listing
  if not args.model:
    parser.print_help()
    print("\n❌ Error: --model is required (or use --list-models)")
    return 1

  # Validate model
  if args.model not in MODEL_REGISTRY:
    print(f"❌ Error: Unknown model '{args.model}'")
    print(f"Available models: {', '.join(MODEL_REGISTRY.keys())}")
    print("Use --list-models for more details")
    return 1

  # Build PipelineOptions from CLI arguments
  try:
    stop_at = parse_stop_at(args.stop_at)
    skip_stages = set()

    if args.skip_setup:
      # Skip transform, codegen, and graph executor stages
      skip_stages = {
        PipelineStage.TRANSFORM,
        PipelineStage.CODEGEN,
        PipelineStage.GRAPH_EXECUTOR,
      }

    options = PipelineOptions(
      stop_at=stop_at,
      skip_stages=skip_stages,
      rebuild_cpp_only=args.rebuild_cpp_only,
      input_pattern=args.pattern if args.pattern else "default",
    )
  except ValueError as e:
    parser.print_help()
    print(f"\n❌ Error: {e}")
    return 1

  # If rebuild_cpp_only, copy modified C++ files from handcraft to evl
  if options.rebuild_cpp_only:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    handcraft_dir = os.path.join(script_dir, "handcraft")
    copy_cpp_cmd = [
      "python", "copy_cpp.py",
      "--model", f"{args.model}_evl",
      "--to_evl"
    ]
    print(f"Copying modified C++ files by running: {' '.join(copy_cpp_cmd)} (in {handcraft_dir})")
    result = subprocess.run(copy_cpp_cmd, cwd=handcraft_dir)
    if result.returncode != 0:
      print("❌ Error: Failed to copy modified C++ files from handcraft to evl")
      return result.returncode

  # Run the test
  run_test_pipeline(test_name=args.model, options=options)
  return 0


if __name__ == "__main__":
  sys.exit(main())
