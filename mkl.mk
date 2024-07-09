#MKL_ROOT_DIR=/opt/intel/oneapi
MKL_ROOT_DIR=/fs5/rzpool/intel/2022
MKL_BLAS_DIR=${MKL_ROOT_DIR}/mkl/latest
#MKL_COMP_DIR="${MKL_ROOT_DIR}/compiler/2023.2.0/linux/compiler"
MKL_COMP_DIR=${MKL_ROOT_DIR}/compiler/latest/linux/compiler
BLAS_INC=-I${MKL_BLAS_DIR}/include
BLAS_LIB=-Wl,--start-group ${MKL_BLAS_DIR}/lib/intel64/libmkl_intel_ilp64.a ${MKL_BLAS_DIR}/lib/intel64/libmkl_intel_thread.a ${MKL_BLAS_DIR}/lib/intel64/libmkl_core.a ${MKL_COMP_DIR}/lib/intel64_lin/libiomp5.a -Wl,--end-group
BLAS_LIB+=-lpthread -lm -ldl -m64 #-L${MKL_COMP_DIR}/lib -liomp5
BLAS_FLAGS=-DMKL_ILP64 -m64 -DUSE_MKLBLAS


#MKL_BLAS_DIR=/usr/lib/x86_64-linux-gnu
#BLAS_INC=-I/usr/include/mkl
#BLAS_LIB=-Wl,--start-group ${MKL_BLAS_DIR}/libmkl_intel_ilp64.a ${MKL_BLAS_DIR}/libmkl_intel_thread.a ${MKL_BLAS_DIR}/libmkl_core.a -Wl,--end-group -liomp5 -lm -ldl -m64
#BLAS_FLAGS=-DMKL_ILP64 -m64 -lpthread -DUSE_MKLBLAS
