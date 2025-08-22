from tvm.relay.backend.contrib.imcflow.device_codegen import DeviceCodegen
import os
import argparse

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--func_dir", type=str, default=None)
  args = parser.parse_args()

  func_dir = args.func_dir
  if func_dir is None:
    func_dir = os.path.dirname(os.path.abspath(__file__))

  for file in ["inode.cpp", "imce.cpp"]:
    target = "inode" if file == "inode.cpp" else "imce"
    device_codegen = DeviceCodegen(target=target, build_dir=".")
    device_codegen.func_dir = func_dir
    device_codegen.compile_target_code(file)

if __name__ == "__main__":
  main()