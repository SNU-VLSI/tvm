.PHONY: help clean build

JSON = imcflow.json
FILE = relu.cpp
HID = 0

help:
	@echo "make help - show this help"
	@echo "make clean - remove all build artifacts"
	@echo "make build - build the project"

clean:
	rm -rf main *.cpp *.o *.ll

build: clean
	g++ -std=c++17 -o main main.cc inode_codegen.cc \
		-ltvm_runtime -L/root/project/tvm/build \
		-I/root/project/tvm/include/ \
		-I/root/project/tvm/3rdparty/dmlc-core/include/ \
		-I/root/project/tvm/3rdparty/dlpack/include \
		-I/root/project/tvm/3rdparty/nlohmann-json

test: main ${JSON}
	./main ${JSON} > ${FILE}

clang: main ${FILE}
	clang -O1 --target=INODE -c \
		-mllvm="-INODE_hid=${HID}" -mllvm="-INODE_wid=0" -mllvm="-force-hardware-loops" \
		-mllvm="-force-nested-hardware-loop" ${FILE}
