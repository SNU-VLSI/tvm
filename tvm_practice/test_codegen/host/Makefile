.PHONY: help clean build

help:
	@echo "make help - show this help"
	@echo "make clean - remove all build artifacts"
	@echo "make build - build the project"

clean:
	rm -rf main *.o *.ll *.so *.cpp *.bin *.out

build: clean
	g++ -std=c++17 -fPIC -o main main.cc ../device/*.host.o

test: main
	./main