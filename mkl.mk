#MKL_ROOT_DIR=/opt/intel/oneapi
#MKL_BLAS_DIR=${MKL_ROOT_DIR}/mkl/latest
#MKL_COMP_DIR="${MKL_ROOT_DIR}/compiler/2023.2.0/linux/compiler"
##MKL_COMP_DIR=${MKL_ROOT_DIR}/compiler/2024.0
#MKL_BLAS_INC=-I${MKL_BLAS_DIR}/include
#MKL_BLAS_LIB=-Wl,--start-group ${MKL_BLAS_DIR}/lib/libmkl_intel_ilp64.a ${MKL_BLAS_DIR}/lib/libmkl_intel_thread.a ${MKL_BLAS_DIR}/lib/libmkl_core.a ${MKL_COMP_DIR}/lib/intel64_lin/libiomp5.a -Wl,--end-group
#MKL_BLAS_LIB+=-lpthread -lm -ldl -m64 #-L${MKL_COMP_DIR}/lib -liomp5
#MKL_BLAS_FLAGS=-DMKL_ILP64 -m64


MKL_BLAS_DIR=/usr/lib/x86_64-linux-gnu
BLAS_INC=-I/usr/include/mkl
BLAS_LIB=-Wl,--start-group ${MKL_BLAS_DIR}/libmkl_intel_ilp64.a ${MKL_BLAS_DIR}/libmkl_intel_thread.a ${MKL_BLAS_DIR}/libmkl_core.a -Wl,--end-group -liomp5 -lm -ldl -m64
BLAS_FLAGS=-DMKL_ILP64 -m64 -lpthread -DUSE_MKLBLAS
