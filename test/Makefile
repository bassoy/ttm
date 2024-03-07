CXX :=g++
#CXX :=clang++

CXX_FLAGS +=-Wextra -Wall -Wpedantic -O3 -std=c++20 -pthread -fopenmp

ifeq ($(BLAS_FLAG), OPENBLAS)
  include ../openblas.mk
else ifeq ($(BLAS_FLAG), MKL)
  include ../mkl.mk
else 
  $(error BLAS_FLAG should be either OPENBLAS or MLK)
endif


include gtest.mk

TLIB_INC :=-I../include -Iinclude

CXX_FLAGS += $(BLAS_FLAGS)
INCS      += $(TLIB_INC)  $(BLAS_INC)
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

