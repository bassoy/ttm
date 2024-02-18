CXX :=g++
#CXX :=clang++

# -DUSE_OPENBLAS 
# -DUSE_MLKBLAS
CXX_FLAGS +=-Wextra -Wall -Wpedantic -O3 -std=c++20 -pthread -fopenmp -DUSE_MKLBLAS
# -g -O0 -fprofile-arcs -ftest-coverage -fno-inline 

MLKBLAS_DIR :=/usr/lib/x86_64-linux-gnu
MLKBLAS_LIB :=-Wl,--start-group ${MLKBLAS_DIR}/libmkl_intel_ilp64.a ${MLKBLAS_DIR}/libmkl_intel_thread.a ${MLKBLAS_DIR}/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl -m64
MLKBLAS_FLAGS := -DMKL_ILP64 -m64

OPENBLAS_INC :=-I/usr/include/x86_64-linux-gnu/openblas64-pthread
OPENBLAS_LIB :=-lopenblas

GTEST_LIB :=-lgtest -lpthread

TLIB_INC :=-I../include



INCS +=$(TLIB_INC) -Iinclude $(OPENBLAS_INC)
LIBS +=$(GTEST_LIB) $(OPENBLAS_LIB) $(MLKBLAS_LIB) -lgomp -lpthread -lm 

CFILES2 = $(wildcard src/*.cpp)
OBJS5   = $(notdir $(CFILES2))
OBJS6   = $(patsubst %.cpp,%.o,$(OBJS5))
OBJSC   = $(addprefix build/,$(OBJS6))

CXX_FLAGS += $(MLKBLAS_FLAGS)

TARGET := bin/main
$(TARGET): $(OBJSC)
	$(CXX) $(CXX_FLAGS) $(OBJSC) $(LIBS) -o $(TARGET)
#	$(CXX) --version

build/%.o : src/%.cpp
	$(CXX) $(CXX_FLAGS) -c $(INCS)  $< -o $@

.PHONY : clean

clean:
	rm -f $(TARGET) build/*.* 

