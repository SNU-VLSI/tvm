import argparse
import subprocess

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--inst", type=int, required=True)
  parser.add_argument("--triple", type=str, choices=["INODE", "IMCE"], default="INODE")
  args = parser.parse_args()
  print(args)

  inst = args.inst
  hex_str = (hex(inst)[2:])
  hex_str = hex_str.zfill(8)

  hex_split_str = (f"0x{hex_str[0:2]} 0x{hex_str[2:4]} 0x{hex_str[4:6]} 0x{hex_str[6:8]}")

  subprocess.run(f"echo '{hex_split_str}' | llvm-mc -disassemble -triple {args.triple}", shell=True)


if __name__ == "__main__":
  main()
