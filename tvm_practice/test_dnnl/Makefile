TVM_ROOT=$(TVM_HOME)
DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core

PKG_CFLAGS = -std=c++17 -O2 -fPIC\
	-I${TVM_ROOT}/include\
	-I${DMLC_CORE}/include\
	-I${TVM_ROOT}/3rdparty/dlpack/include\
	-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>

PKG_LDFLAGS = -L${TVM_ROOT}/build -ldl -pthread

cpp_graph/main: cpp_graph_deploy.cpp cpp_graph/lib.so
	g++  $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS) -ltvm_runtime

cpp_aot/main: cpp_aot_deploy.cpp cpp_aot/lib.so
	g++  $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS)