CXX :=g++
#CXX :=clang++

CXX_FLAGS:=-Wextra -Wall -O3 -std=c++17 -pthread -fopenmp

ifeq ($(BLAS_FLAG), OPENBLAS)
  include ../openblas.mk
else ifeq ($(BLAS_FLAG), MKL)
  include ../mkl.mk
else ifeq ($(BLAS_FLAG), BLIS)
  include ../blis.mk
endif

TLIB_INC :=-I../include

CXX_FLAGS += $(BLAS_FLAGS)
INCS += $(TLIB_INC) $(BLAS_INC)
LIBS += $(BLAS_LIB) -lgomp -lpthread

# Source files
SRC1 := interface1.cpp
SRC2 := interface2.cpp
SRC3 := interface3.cpp
SRC4 := test.cpp

# Object files
OBJ1 := $(SRC1:.cpp=.o)
OBJ2 := $(SRC2:.cpp=.o)
OBJ3 := $(SRC3:.cpp=.o)
OBJ4 := $(SRC4:.cpp=.o)

# Targets
all: interface1 interface2 interface3 test

interface1: $(OBJ1)
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(LIBS)

interface2: $(OBJ2)
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(LIBS)

interface3: $(OBJ3)
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(LIBS) 

test: $(OBJ4)
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(LIBS) 


# Generic rule to build object files
%.o: %.cpp
	$(CXX) $(CXX_FLAGS) $(INCS) -c $< -o $@

# Clean rule
clean:
	rm -f interface1 interface2 interface3 $(OBJ1) $(OBJ2) $(OBJ3)

