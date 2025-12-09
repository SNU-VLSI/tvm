#!/usr/bin/env python3
"""
Command-line interface for running IMCFLOW model tests manually.

This script provides a user-friendly CLI for testing IMCFLOW models with
different input patterns, with options to skip expensive setup steps when
experimenting with different inputs on an already-compiled model.
"""

import argparse
import sys
from test import MODEL_REGISTRY, INPUT_PATTERNS, run_test_pipeline


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

  # Skip setup (reuse existing compiled model)
  python main.py --model one_relu --pattern zeros --skip-setup

  # List available models
  python main.py --list-models

  # Test multiple patterns quickly (shell script)
  # 1. First run with full setup:
  python main.py --model resnet8_small_pretrained --pattern ones
  # 2. Then test other patterns with --skip-setup:
  python main.py --model resnet8_small_pretrained --pattern random --skip-setup
  python main.py --model resnet8_small_pretrained --pattern zeros --skip-setup
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
    "--skip-setup", "-s",
    action="store_true",
    help="Skip model transformation, codegen, and graph generation (reuse existing compiled model)"
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

  # Run the test
  try:
    run_test_pipeline(
      test_name=args.model,
      input_pattern=args.pattern,
      skip_setup=args.skip_setup
    )
    print("\n✅ Test completed successfully")
    return 0
  except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    return 1


if __name__ == "__main__":
  sys.exit(main())
