#!/bin/bash
if [ "$(basename "$PWD")" != "build" ]; then
	echo "Error: this script should be run in build dir"
	exit 1
fi

export PATH="/root/tools/llvm_backend_test/bin:${PATH}"

rm -rf *
cp ../cmake/config.cmake .

# conda include path addition
echo "include_directories("${CONDA_PREFIX}/include")" >> config.cmake

# controls default compilation flags (Candidates: Release, Debug, RelWithDebInfo)
echo "set(CMAKE_BUILD_TYPE Debug)" >> config.cmake

# LLVM is a must dependency for compiler end
# echo "set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")" >> config.cmake
echo "set(USE_LLVM ON)" >> config.cmake
# echo "set(LLVM_INCLUDE_DIRS $HOME/project/llvm-project/llvm/include $HOME/project/llvm-project/builddir/include)" >> config.cmake
# echo "set(LLVM_LIBS $(llvm-config --libfiles --system-libs))" >> config.cmake
# echo "set(LLVM_VERSION_MAJOR 19)" >> config.cmake
# echo "set(LLVM_VERSION_MINOR 1)" >> config.cmake
# echo "set(LLVM_VERSION_PATCH 1)" >> config.cmake
echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake

# GPU SDKs, turn on if needed
echo "set(USE_CUDA   OFF)" >> config.cmake
echo "set(USE_METAL  OFF)" >> config.cmake
echo "set(USE_VULKAN OFF)" >> config.cmake
echo "set(USE_OPENCL OFF)" >> config.cmake

# cuBLAS, cuDNN, cutlass support, turn on if needed
echo "set(USE_CUBLAS OFF)" >> config.cmake
echo "set(USE_CUDNN  OFF)" >> config.cmake
echo "set(USE_CUTLASS OFF)" >> config.cmake

# set GTEST off
echo "set(USE_GTEST OFF)" >> config.cmake

# DEBUG ON
echo "set(USE_RELAY_DEBUG ON)" >> config.cmake

# VTA ON
echo 'set(USE_VTA_FSIM ON)' >> config.cmake

# DNNL ON
# echo "set(USE_DNNL $(echo $CONDA_PREFIX))" >> config.cmake
echo "set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${CONDA_PREFIX}/lib)" >> config.cmake
echo "set(USE_DNNL C_SRC)" >> config.cmake

# IMCFLOW ON
echo "set(USE_IMCFLOW ON)" >> config.cmake