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
from runners.pipeline_options import PipelineOptions, PipelineStage, parse_stop_at, parse_start_at


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
  python main.py --model one_relu --stop-at compile     # Compile only (no simulation)

  # Start at specific pipeline stage (skip earlier stages)
  python main.py --model one_relu --start-at codegen    # Skip transform
  python main.py --model one_relu --start-at simulate   # Skip transform, codegen, compile

  # Use a specific dataset sample as input (e.g., CIFAR-10 sample #2)
  python main.py --model resnet8_subset31_pretrained_orig --dataset cifar10 --sample 2

  # Run with handcraft patches (copy C++ from handcraft, codegen, patch inode, rebuild)
  python main.py --model resnet8_subset31_pretrained_orig --with-patch

  # Start at codegen and stop at codegen to observe memlayout changes
  python main.py --model one_relu --start-at codegen --stop-at codegen

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
    choices=["transform", "codegen", "compile", "validate", "simulate", "compare"],
    default="compare",
    help="Pipeline stage to stop at: transform, codegen, compile, validate, simulate, compare (default)"
  )

  parser.add_argument(
    "--start-at",
    type=str,
    choices=["transform", "codegen", "compile", "validate", "simulate"],
    default=None,
    help="Pipeline stage to start at (skips earlier stages): transform, codegen, compile, validate, simulate"
  )

  parser.add_argument(
    "--with-patch",
    action="store_true",
    help="Apply handcraft patches: copy C++ from handcraft, run codegen, patch inode.cpp, rebuild"
  )

  parser.add_argument(
    "--dataset",
    type=str,
    default=None,
    help="Dataset to use for input (e.g., 'cifar10'). Overrides --pattern when specified."
  )

  parser.add_argument(
    "--sample",
    type=int,
    default=None,
    help="0-based sample index within the dataset (requires --dataset)"
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
    start_at = PipelineStage.TRANSFORM  # default: start from beginning

    if args.start_at:
      start_at = parse_start_at(args.start_at)

    # Validate --dataset/--sample requirements
    if args.dataset and args.sample is None:
      print("Error: --sample is required when --dataset is specified")
      return 1
    if args.sample is not None and not args.dataset:
      print("Error: --dataset is required when --sample is specified")
      return 1

    # Validate --with-patch requirements
    if args.with_patch:
      script_dir = os.path.dirname(os.path.abspath(__file__))
      handcraft_model_dir = os.path.join(script_dir, "handcraft", f"{args.model}_evl")
      if not os.path.exists(handcraft_model_dir):
        print(f"Error: --with-patch requires handcraft directory: {handcraft_model_dir}")
        print(f"Run without --with-patch first to generate the model, then copy to handcraft.")
        return 1
      # --with-patch requires CODEGEN stage to be in the execution range
      # (to generate mem_layout.txt and compile patched files)
      if start_at > PipelineStage.CODEGEN:
        print(f"Error: --with-patch requires CODEGEN stage to run (--start-at must be <= codegen)")
        print(f"Current --start-at: {args.start_at}")
        return 1
      if stop_at < PipelineStage.CODEGEN:
        print(f"Error: --with-patch requires CODEGEN stage to run (--stop-at must be >= codegen)")
        print(f"Current --stop-at: {args.stop_at}")
        return 1

    options = PipelineOptions(
      stop_at=stop_at,
      start_at=start_at,
      with_patch=args.with_patch,
      input_pattern=args.pattern if args.pattern else "default",
      dataset=args.dataset,
      sample=args.sample,
    )
  except ValueError as e:
    parser.print_help()
    print(f"\nError: {e}")
    return 1

  # Run the test
  run_test_pipeline(test_name=args.model, options=options)
  return 0


if __name__ == "__main__":
  sys.exit(main())
