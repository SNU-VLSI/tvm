# TVM CRT Build System (Unified)

Build TVM CRT + model for Cortex-M3 from an MLF tar.

This unified CMakeLists.txt supports both ARM and x86 builds with a single configuration file.

## Prerequisites

- Generate MLF with build_m3.py in parent dir (Graph Executor, system-lib):
  ```bash
  python3 ../build_m3.py -e graph -s
  ```
- ARM toolchain installed: `arm-none-eabi-gcc/g++`
- Host object files (`.host.o`) required

### For x86 builds:

- Standard x86_64 toolchain (`gcc/g++`)

### For ARM builds:

- ARM cross-compiler (`aarch64-xilinx-linux-gcc` or `aarch64-linux-gnu-gcc`)
  ```bash
  sudo apt update
  sudo apt install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu binutils-aarch64-linux-gnu
  ```
- Host object files (`.host.o`) optional

### For scan register write:

- zlib.so for ARM cross-compiler is needed
  ```bash
  dpkg --add-architecture arm64
  echo "deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports jammy main universe" >> /etc/apt/sources.list.d/ubuntu-arm64-ports.list
  echo "deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports jammy-updates main universe" >> /etc/apt/sources.list.d/ubuntu-arm64-ports.list
  echo "deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports jammy-security main universe" >> /etc/apt/sources.list.d/ubuntu-arm64-ports.list
  echo "deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports jammy-backports main universe" >> /etc/apt/sources.list.d/ubuntu-arm64-ports.list
  ```
- Open `/etc/apt/sources.list` and append `[arch=amd64]` to every deb entries, e.g.:
  ```
  deb [arch=amd64] http://security.ubuntu.com/ubuntu/ jammy-security multiverse
  ```
- Then install:
  ```bash
  apt update
  apt install zlib1g-dev:arm64
  ```
- If you get an error like:
  > The following packages have unmet dependencies: zlib1g:arm64 : Depends: libc6:arm64 (>= 2.17) but it is not going to be installed zlib1g-dev:arm64 : Depends: libc6-dev:arm64 but it is not going to be installed or libc-dev:arm64 E: Unable to correct problems, you have held broken packages.

  This is due to a version mismatch between amd64 and arm64 libc6. Fix by upgrading:
  ```bash
  sudo apt install --only-upgrade libc6 libc6-dev
  ```

## Quick Start

```bash
mkdir -p build
cd build

# x86 build
../build.sh execute_graph.c . x86

# ARM build
../build.sh execute_graph.c . arm
```

## Outputs

- Static libraries: `libtvm_model.a`, `libtvm_m3.a` and CRT component libs in build tree.
