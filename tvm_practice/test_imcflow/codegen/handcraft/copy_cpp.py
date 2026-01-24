#!python
import argparse

# Argument parser

def main():
  parser = argparse.ArgumentParser(
    description="Copy C++ code between evl build directory and handcraft directory",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Copy from evl build directory to handcraft directory
  python copy_cpp.py --model one_relu --from_evl

  # Copy from handcraft directory to evl build directory
  python copy_cpp.py --model one_relu --to_evl

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
    help="When specified, copy from the evl build directory to handcraft directory."
  )

  parser.add_argument(
    "--to_evl", "-t",
    action='store_true',
    help="When specified, copy the handcraft code to the evl handcraft directory."
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

  # Copy the generated C++ files in the build directory recursively to the handcraft directory
  # e.g. copy *.cpp from ../big_conv_evl/build/*/*.cpp to ./big_conv_evl/*/*.cpp
  # If the destination files already exist, abort the copy and print a message
  if args.from_evl:
    import os
    import shutil
    for root, dirs, files in os.walk(build_dir):
      for file in files:
        if file.endswith(".cpp") or file.endswith(".h"):
          src_file = os.path.join(root, file)
          relative_path = os.path.relpath(root, build_dir)
          dest_dir = os.path.join(f"./{model_name}/", relative_path)
          os.makedirs(dest_dir, exist_ok=True)
          dest_file = os.path.join(dest_dir, file)
          if os.path.exists(dest_file):
            print(f"File {dest_file} already exists. Aborting copy.")
            return
          shutil.copy2(src_file, dest_file)
          print(f"Copied {src_file} to {dest_file}")
  elif args.to_evl:
    import os
    import shutil
    for root, dirs, files in os.walk(f"./{model_name}/"):
      for file in files:
        if file.endswith(".cpp") or file.endswith(".h"):
          src_file = os.path.join(root, file)
          relative_path = os.path.relpath(root, f"./{model_name}/")
          dest_dir = os.path.join(build_dir, relative_path)
          os.makedirs(dest_dir, exist_ok=True)
          dest_file = os.path.join(dest_dir, file)
          shutil.copy2(src_file, dest_file)
          print(f"Copied {src_file} to {dest_file}")
    
if __name__ == "__main__":
    main()