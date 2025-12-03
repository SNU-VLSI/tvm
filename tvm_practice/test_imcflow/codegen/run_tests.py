import subprocess
import os, sys, argparse
from colorama import init, Fore, Back, Style
init()

ALL_TESTS = [
  "test_one_relu_evl",
  "test_one_conv_evl",
  "test_residual_model",
  "test_resnet8",
  "ds_cnn",
  "autoencoder",
  "mobilenet",
]

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--target", type=str, required=True, help="target model")
  parser.add_argument("--dry-run", action="store_true", help="perform a dry run without executing tests")
  args = parser.parse_args()

  if args.target not in ALL_TESTS and args.target != "all":
    print(f"Unknown target {args.target}. Available targets: {ALL_TESTS} or all")
    sys.exit(1)
  
  targets_to_run = ALL_TESTS if args.target == "all" else [args.target]

  if args.dry_run:
    print("Performing dry run...")

  for target in targets_to_run:
    name_max_len = max(len(str(t)) for t in ALL_TESTS)
    print(f"Run {target:<{name_max_len}}: ", end="", flush=True)
    cmds = [
      sys.executable, 
      "test.py",
      "-k", target,
      "-s"
    ]

    if args.dry_run:
      cmds[0] = "python_dbg"
      print(" ".join(cmds))
      continue
      
    os.makedirs("./logs", exist_ok=True)
    with open(f"./logs/{target}_test_log.txt", "w") as log_file:
      ret = subprocess.run(
        cmds,
        stdout=log_file,
        stderr=log_file,
      )
    if ret.returncode != 0:
      print(Fore.RED + Style.BRIGHT + f"{'Failed':10}" + Style.RESET_ALL)
    else:
      print(Fore.GREEN + Style.BRIGHT + f"{'Passed':10}" + Style.RESET_ALL)