cpp_graph/main: cpp_graph_deploy.cpp cpp_graph/lib.so
	g++ -O2 -std=c++17 -fPIC \
		-I/root/project/tvm/include \
		-I/root/project/tvm/3rdparty/dlpack/include \
		-I/root/project/tvm/3rdparty/dmlc-core/include \
		-o $@ \
		$^ \
		-L/root/project/tvm/build \
		-ltvm_runtime -ldl -pthread

cpp_aot/main: cpp_aot_deploy.cpp cpp_aot/lib.so
	g++ -O2 -std=c++17 -fPIC \
		-I/root/project/tvm/include \
		-I/root/project/tvm/3rdparty/dlpack/include \
		-I/root/project/tvm/3rdparty/dmlc-core/include \
		-o $@ \
		$^ \
		-L/root/project/tvm/build \
		-ltvm_runtime -ldl -pthread