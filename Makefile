PYTHON ?= python3

all:
	${PYTHON} setup.py build_ext -if
	gcc -shared -o DFS.so -fPIC DFS.c
	rm -rf build

clean:
	rm -rf build pytrt.cpp *.so
