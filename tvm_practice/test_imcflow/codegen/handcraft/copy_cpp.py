#!python
import argparse

# Argument parser

def main():
  parser = argparse.ArgumentParser(
    description="Preserve/modify/restore build directories between evl and handcraft",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Save evl build directory to handcraft (creates build/ and build.orig/)
  python copy_cpp.py --model one_relu --from_evl

  # Restore build directory from handcraft back to evl
  python copy_cpp.py --model one_relu --to_evl

Note:
  --from_evl creates build.orig/ only once (first run) to preserve the original.
  Subsequent runs will refresh build/ from build.orig/, never overwriting the original.
  This ensures the original is always safe, even if you run --from_evl multiple times.
    """
  )

  parser.add_argument(
    "--model", "-m",
    type=str,
    help="Model to parse C++ code for."
  )

  parser.add_argument(
    "--from_evl", "-f",
    action='store_true',
    help="Copy from evl to handcraft. Creates build.orig/ once (preserves original), always refreshes build/"
  )

  parser.add_argument(
    "--to_evl", "-t",
    action='store_true',
    help="Restore build directory from handcraft/<model>/build/ back to evl"
  )

  args = parser.parse_args()

  # just one of --from_evl and --to_evl must be set
  if not args.from_evl and not args.to_evl:
    print("Either --from_evl or --to_evl must be specified.")
    return
  if args.from_evl and args.to_evl:
    print("Only one of --from_evl and --to_evl can be set.")
    return

  model_name = args.model
  build_dir = f"../{model_name}/build/"

  # Copy from evl to handcraft with original preservation
  if args.from_evl:
    import os
    import shutil

    handcraft_model_dir = f"./{model_name}/"
    handcraft_build_dir = os.path.join(handcraft_model_dir, "build")
    handcraft_build_orig_dir = os.path.join(handcraft_model_dir, "build.orig")

    if not os.path.exists(build_dir):
      print(f"Error: Build directory {build_dir} does not exist")
      return

    # Create handcraft model directory if it doesn't exist
    os.makedirs(handcraft_model_dir, exist_ok=True)

    # Function to ignore all files except .cpp and .h (but keep all directories)
    def ignore_non_cpp_h(dir, files):
      ignored = []
      for f in files:
        full_path = os.path.join(dir, f)
        # Keep directories and .cpp/.h files, ignore everything else
        if os.path.isfile(full_path) and not (f.endswith('.cpp') or f.endswith('.h')):
          ignored.append(f)
      return ignored

    # Create build.orig only if it doesn't exist (preserve original)
    if not os.path.exists(handcraft_build_orig_dir):
      print(f"Creating original backup: {handcraft_build_orig_dir}")
      print(f"Copying *.cpp and *.h files from {build_dir}")
      shutil.copytree(build_dir, handcraft_build_orig_dir, ignore=ignore_non_cpp_h)
      shutil.copytree(build_dir, handcraft_build_dir, ignore=ignore_non_cpp_h)
      print(f"Original backup created successfully")
    else:
      print(f"Original backup already exists: {handcraft_build_orig_dir}")
      print(f"Skipping backup creation to preserve the original")

  elif args.to_evl:
    import os
    import shutil

    # Use the modified build directory from handcraft/<model>/build
    handcraft_model_dir = f"./{model_name}/"
    handcraft_build_dir = os.path.join(handcraft_model_dir, "build")

    # Check if the modified build directory exists
    if not os.path.exists(handcraft_build_dir):
      print(f"Error: Modified build directory {handcraft_build_dir} does not exist.")
      print(f"Please run with --from_evl first to save the build directory.")
      return

    # Check if target build directory exists
    if not os.path.exists(build_dir):
      print(f"Error: Target build directory {build_dir} does not exist.")
      return

    # Function to ignore all files except .cpp and .h (but keep all directories)
    def ignore_non_cpp_h(dir, files):
      ignored = []
      for f in files:
        full_path = os.path.join(dir, f)
        # Keep directories and .cpp/.h files, ignore everything else
        if os.path.isfile(full_path) and not (f.endswith('.cpp') or f.endswith('.h')):
          ignored.append(f)
      return ignored

    # Copy only *.cpp and *.h files, preserving other files in the target directory
    print(f"Copying modified *.cpp and *.h files from {handcraft_build_dir} to {build_dir}")
    shutil.copytree(handcraft_build_dir, build_dir,
                    dirs_exist_ok=True,  # Don't remove existing directory
                    ignore=ignore_non_cpp_h)
    print(f"Successfully updated *.cpp and *.h files in {build_dir}")
    print(f"Other files (Makefiles, object files, etc.) were preserved")
    
if __name__ == "__main__":
    main()