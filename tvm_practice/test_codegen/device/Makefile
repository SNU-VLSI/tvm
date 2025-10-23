.PHONY: help clean build

help:
	@echo "make help - show this help"
	@echo "make clean - remove all build artifacts"
	@echo "make build - build the project"

clean:
	rm -rf main *.o *.ll *.so *.cpp *.bin *.out

build: clean
	g++ -std=c++17 -o main main.cc device_codegen.cc inode_codegen.cc \
		-ltvm_runtime -L/root/project/tvm/build \
		-I/root/project/tvm/include/ \
		-I/root/project/tvm/3rdparty/dmlc-core/include/ \
		-I/root/project/tvm/3rdparty/dlpack/include \
		-I/root/project/tvm/3rdparty/nlohmann-json

test: main
	./main