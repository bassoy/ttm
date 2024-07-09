CXX :=g++
#CXX :=clang++-12

CXX_FLAGS +=-Wextra -Wall -Wpedantic -O3 -std=c++2a -pthread -fopenmp

ifeq ($(BLAS_FLAG), OPENBLAS)
  include ../openblas.mk
else ifeq ($(BLAS_FLAG), MKL)
  include ../mkl.mk
endif


include gtest.mk

TLIB_INC :=-I../include -Iinclude

CXX_FLAGS += $(BLAS_FLAGS)
INCS      += $(GTEST_INC) $(TLIB_INC)  $(BLAS_INC)
LIBS      += $(GTEST_LIB) $(BLAS_LIB) -lgomp -lpthread

CFILES2 = $(wildcard src/*.cpp)
OBJS5   = $(notdir $(CFILES2))
OBJS6   = $(patsubst %.cpp,%.o,$(OBJS5))
OBJSC   = $(addprefix build/,$(OBJS6))

TARGET := bin/main
$(TARGET): $(OBJSC)
	$(CXX) $(CXX_FLAGS) $(OBJSC) $(LIBS) -o $(TARGET)
#	$(CXX) --version

build/%.o : src/%.cpp
	$(CXX) $(CXX_FLAGS) -c $(INCS)  $< -o $@

.PHONY : clean

clean:
	rm -f $(TARGET) build/*.* 

