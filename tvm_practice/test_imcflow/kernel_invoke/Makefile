TVM_ROOT=$(TVM_HOME)
DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core

PKG_CFLAGS = -std=c++17 -O2 \
	-I${TVM_ROOT}/include\
	-I${DMLC_CORE}/include\
	-I${TVM_ROOT}/3rdparty/dlpack/include

PKG_LDFLAGS = -L${TVM_ROOT}/build -ldl -pthread -ltvm_runtime

all:
	g++  -c $(PKG_CFLAGS) -o tvmgen_default_imcflow_main_4.o  tvmgen_default_imcflow_main_4.cc $(PKG_LDFLAGS) 

cpp_graph/main: cpp_graph_deploy.cpp cpp_graph/lib.so
	g++  $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS) -ltvm_runtime

cpp_aot/main: cpp_aot_deploy.cpp cpp_aot/lib.so
	g++  $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS)