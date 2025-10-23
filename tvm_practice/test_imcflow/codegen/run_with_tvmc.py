#!/usr/bin/env python3
"""
Script to run microTVM models using TVMC.
"""

import argparse
import subprocess
import sys
import pathlib


def run_with_tvmc(dir_name, fill_mode="random", print_top=5):
  """Run the model using TVMC micro command."""

  project_dir = pathlib.Path(f"{dir_name}/micro/project")

  print(f"Running model with TVMC from directory: {dir_name}")
  print(f"Project directory: {project_dir}")

  # TVMC micro run command
  cmd = [
      "python", "-m", "tvm.driver.tvmc", "run",
      "--device", "micro",
      str(project_dir),
      "--fill-mode", fill_mode,
      "--print-top", str(print_top)
  ]

  print(f"Executing: {' '.join(cmd)}")

  try:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print("TVMC output:")
    print(result.stdout)
    if result.stderr:
      print("TVMC stderr:")
      print(result.stderr)
  except subprocess.CalledProcessError as e:
    print(f"TVMC failed with error code {e.returncode}")
    print("STDOUT:")
    print(e.stdout)
    print("STDERR:")
    print(e.stderr)
    return False

  return True


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run microTVM models using TVMC")
  parser.add_argument(
      "dir_name",
      help="Directory name containing the microTVM project (e.g., small_ref, small_evl)"
  )
  parser.add_argument(
      "--fill-mode",
      default="random",
      choices=["random", "zeros", "ones"],
      help="Input fill mode (default: random)"
  )
  parser.add_argument(
      "--print-top",
      type=int,
      default=5,
      help="Number of top outputs to print (default: 5)"
  )

  args = parser.parse_args()

  success = run_with_tvmc(args.dir_name, args.fill_mode, args.print_top)
  if success:
    print("Model execution completed successfully!")
  else:
    print("Model execution failed!")
    sys.exit(1)
