# TVM CRT Build System (Unified)Build TVM CRT + model for Cortex-M3 from an MLF tar.



This unified CMakeLists.txt supports both ARM and x86 builds with a single configuration file.Prerequisites:

- Generate MLF with build_m3.py in parent dir (Graph Executor, system-lib):

## Prerequisites  python3 ../build_m3.py -e graph -s

- ARM toolchain installed: arm-none-eabi-gcc/g++

### For x86 builds:

- Standard x86_64 toolchain (gcc/g++)Configure & build:

- Host object files (`.host.o`) required  mkdir -p build && cd build

  cmake -DCMAKE_TOOLCHAIN_FILE=../../test_imcflow/codegen/arm-cortex-m3.cmake ..

### For ARM builds:  cmake --build . -j

- ARM cross-compiler (`aarch64-xilinx-linux-gcc` or `aarch64-linux-gnu-gcc`)

- Host object files (`.host.o`) optionalOutputs:

- Static libraries: libtvm_model.a, libtvm_m3.a and CRT component libs in build tree.

### For scan register write:

- zlib.so for ARM cross-compiler is needed
```bash
dpkg --add-architecture arm64
echo "deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports jammy main universe" >> /etc/apt/sources.list
echo "deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports jammy-updates main universe" >> /etc/apt/sources.list
apt update
apt install zlib1g-dev:arm64
```

## Quick Start

```bash
mkdir -p build
cd build
../build.sh execute_graph.c x86

or

../build.sh execute_graph.c arm
```