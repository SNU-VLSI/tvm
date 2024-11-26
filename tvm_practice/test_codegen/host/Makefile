.PHONY: help clean build

DEVICE_FILE=relu_inode.host.o

help:
	@echo "make help - show this help"
	@echo "make clean - remove all build artifacts"
	@echo "make build - build the project"

clean:
	rm -rf main *.o *.ll *.so *.cpp *.bin *.out

build: clean
	ln -s ../device/${DEVICE_FILE} .
	g++ -std=c++17 -fPIC -o main main.cc ${DEVICE_FILE}

test: main
	./main