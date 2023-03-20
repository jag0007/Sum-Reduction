CC=gcc
CXX=g++
NVCC=nvcc

CUDAPATH=/opt/asn/apps/cuda_11.7.0

CCFLAGS=-std=c11
CXXFLAGS=-std=c++11 -O4
NVCCFLAGS=-std=c++11

NVCCARCHS=-gencode arch=compute_80,code=sm_80 -gencode arch=compute_70,code=sm_70

TIMERINCPATH=-I$(CUDAPATH)/include -ITimer/include
INCPATH=-Isaxpy/include -I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc -I$(GTESTPATH)/include
LIBPATH=-L$(CUDAPATH)/lib64 -L$(GTESTPATH)/lib64
RPATH=-Wl,-rpath=`pwd`/build/lib -Wl,-rpath=`pwd`/$(GTESTPATH)/lib64 -Wl,-rpath=`pwd`/$(CUDAPATH)/lib64
LIBS=-lcudart

.PHONY: clean modules

build: build/lib/libsumReduce.so build/bin/reduce_test

build/lib/libsumReduce.so: modules sumReduce/src/sumReduce.cu
	@mkdir -p build/.objects/sumReduce
	$(NVCC) -pg $(NVCCFLAGS) $(NVCCARCHS) -Xcompiler -fPIC \
		-IsumReduce/src -I$(CUDAPATH)/samples/common/inc \
		-dc -o build/.objects/sumReduce/sumReduce.o \
		sumReduce/src/sumReduce.cu
	$(NVCC) -pg $(NVCCFLAGS) $(NVCCARCHS) -Xcompiler -fPIC \
		-dlink -o build/.objects/sumReduce/sumReduce-dlink.o build/.objects/sumReduce/sumReduce.o 
	@mkdir -p build/lib
	$(CC) -shared -o build/lib/libsumReduce.so build/.objects/sumReduce/* \
		-Wl,-rpath=$(CUDAPATH)/lib64 -L$(CUDAPATH)/lib64 -lcudart
	@mkdir -p build/include
	@ln -sf ../../sumReduce/include/sumReduce.h build/include/sumReduce.h

build/bin/reduce_test: build/lib/libsumReduce.so sumReduce/test/src/test.cpp
	@mkdir -p build/bin
	$(CXX) -Ibuild/include -I$(CUDAPATH)/samples/common/inc \
		-IsumReduce/include \
		-o build/bin/reduce_test sumReduce/test/src/test.cpp \
		-Wl,-rpath=$(PWD)/build/lib -Lbuild/lib -L$(CUDAPATH)/lib64 \
		-lsumReduce -lcudart

run: #build/bin/reduce_test
	@rm -f *nsys-rep reduce_test.i* reduce_test.o* core.*
	@echo "Running"
	@echo -ne "class\n1\n\n10gb\n1\nampere\nreduce_test\n" | \
	run_gpu .runTests.sh > out.txt
#	@sleep 5
#	@tail -f runTestsshGPU.o*

clean:
	rm -rf build
	rm -f *nsys-rep
	rm -f reduce_test.*


