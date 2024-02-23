CXX :=g++
#CXX :=clang++

# -DUSE_OPENBLAS 
# -DUSE_MLKBLAS
CXX_FLAGS +=-Wextra -Wall -Wpedantic -O3 -std=c++20 -pthread -fopenmp -DUSE_MKLBLAS
# -g -O0 -fprofile-arcs -ftest-coverage -fno-inline 


MKL_ROOT_DIR=/opt/intel/oneapi
MKL_BLAS_DIR=${MKL_ROOT_DIR}/mkl/latest
MKL_COMP_DIR="${MKL_ROOT_DIR}/compiler/2023.2.0/linux/compiler"
#MKL_COMP_DIR=${MKL_ROOT_DIR}/compiler/2024.0
MKL_BLAS_INC=-I${MKL_BLAS_DIR}/include
MKL_BLAS_LIB=-Wl,--start-group ${MKL_BLAS_DIR}/lib/libmkl_intel_ilp64.a ${MKL_BLAS_DIR}/lib/libmkl_intel_thread.a ${MKL_BLAS_DIR}/lib/libmkl_core.a ${MKL_COMP_DIR}/lib/intel64_lin/libiomp5.a -Wl,--end-group
MKL_BLAS_LIB+=-lpthread -lm -ldl -m64 #-L${MKL_COMP_DIR}/lib -liomp5
MKL_BLAS_FLAGS=-DMKL_ILP64 -m64

OPENBLAS_INC :=-I/usr/include/x86_64-linux-gnu/openblas64-pthread
OPENBLAS_LIB :=-lopenblas

GTEST_LIB :=-lgtest -lpthread

TLIB_INC :=-I../include

#$(OPENBLAS_INC) 
#$(OPENBLAS_LIB) 

INCS +=$(TLIB_INC) -Iinclude $(MKL_BLAS_INC)
LIBS +=$(GTEST_LIB) $(MKL_BLAS_LIB) -lgomp -lpthread -lm 

CFILES2 = $(wildcard src/*.cpp)
OBJS5   = $(notdir $(CFILES2))
OBJS6   = $(patsubst %.cpp,%.o,$(OBJS5))
OBJSC   = $(addprefix build/,$(OBJS6))

CXX_FLAGS += $(MKL_BLAS_FLAGS)

TARGET := bin/main
$(TARGET): $(OBJSC)
	$(CXX) $(CXX_FLAGS) $(OBJSC) $(LIBS) -o $(TARGET)
#	$(CXX) --version

build/%.o : src/%.cpp
	$(CXX) $(CXX_FLAGS) -c $(INCS)  $< -o $@

.PHONY : clean

clean:
	rm -f $(TARGET) build/*.* 

