/*
TLIB_DIR=..
TLIB_INC="-I${TLIB_DIR}/include"

INCS="${TLIB_INC} ${MKL_BLAS_INC}"
LIBS="${MKL_BLAS_LIB}"
FLAGS="${MKL_BLAS_FLAGS} -DUSE_MKLBLAS"


g++ ${INCS} -std=c++17 -Ofast -fopenmp interface1.cpp ${LIBS} ${FLAGS} -o interface1 && ./interface1
*/

#include <tlib/ttm.h>

#include <vector>
#include <numeric>
#include <iostream>
#include <string>
#include <chrono> // for high precision timing

static const auto gdims = std::string("abcdefghij");

inline 
std::string 
get_dims(unsigned order)
{
  assert(order > 0u && order <= 10u);
  return gdims.substr(0,order);
}

inline 
std::string 
get_dims(unsigned mode, unsigned order)
{
  assert(mode > 0u && mode <= order);
  auto dims = get_dims(order);
  dims.at(mode-1) = 'm';
  return dims;
}

inline std::string 
get_index_a(std::size_t /*mode*/, std::size_t order)
{
  return get_dims(order);
}

inline 
std::string 
get_index_b(std::size_t mode, std::size_t /*order*/)
{
    assert(mode > 0u && mode <= 10u);
    return std::string("m") + gdims.at(mode-1);
}

inline 
std::string 
get_index_c(std::size_t mode, std::size_t order)
{
  return get_dims(mode,order);
}

inline
std::string
add_comma(std::string original)
{
    if (original.length() < 2u)
        return original;
        
    auto modified = std::string(1,original.at(0));
    auto comma = std::string(",");
    
    for(auto c = ++original.begin(); c != original.end(); ++c)
        modified += comma + *c;

    return modified;
}

inline
double 
get_gflops(double nn, double cdimc, double cdima)
{
  return cdimc * (nn/cdima) * (2.0*cdima-1.0) * 1e-9;
}

template<class value, class parallel_policy, class slicing_policy, class fusion_policy>
inline void measure(unsigned q, 
                    tlib::tensor<value> const& A, 
                    tlib::tensor<value> const& B, 
                    tlib::tensor<value>& C,
                    parallel_policy pp,
                    slicing_policy sp,
                    fusion_policy fp)
{   

    auto cache = std::vector<char>(1024ul*1024ul*128ul);
    const auto iters = 50u;
    
    auto time = double(0);
    for(auto i = 0u; i < iters; ++i){
        std::fill(cache.begin(), cache.end(),char{});
        auto start = std::chrono::high_resolution_clock::now();
        tlib::ttm(
            pp, sp, fp,
            q, A.order(),
            A.data().data(), A.shape().data(), A.strides().data(), A.layout().data(),
            B.data().data(), B.shape().data(),                     B.layout().data(),
            C.data().data(), C.shape().data(), C.strides().data());
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start; 
        time += duration.count();
    }
    
    

    const auto na = A.shape();
    const auto nb = B.shape();
    const auto num_elems = std::accumulate(na.begin(), na.end(), 1.0, std::multiplies<double>());
    const auto avg_time_s = time/double(iters);
    const auto gflops = get_gflops(num_elems,nb[0],nb[1]);
    const auto gbyte = sizeof(double) * double(A.data().size()+B.data().size()+C.data().size()) * 1e-9;


    if(q > 1 && q < A.order()){
        const auto inner = std::accumulate(na.begin(),   na.begin()+q-1, 1.0, std::multiplies<double>());
        const auto outer = std::accumulate(na.begin()+q, na.end(),     1.0, std::multiplies<double>());
        if(std::is_same_v<slicing_policy,tlib::slicing_policy::subtensor_t>){
            const auto rowsA = inner;
            const auto colsA = na[q-1];
            const auto rowsB = nb[1];
            const auto rowsC = rowsA;
            const auto colsC = nb[0];
            std::cout << "Par Dim: l = " << outer << std::endl; 
            std::cout << "Gemm Dims: m = " << rowsC << ", n = " << colsC << ", k = " << colsA << ", ";
            std::cout << "m*k = " << rowsC*colsA << ", m*k/l = " << double(rowsC*colsA)/double(outer) << ", m/k = " << double(rowsC)/double(colsA) << std::endl;

        }

        if(std::is_same_v<slicing_policy,tlib::slicing_policy::slice_t>){
            const auto rowsA = na[0];
            const auto colsA = na[q-1];
            const auto rowsB = nb[1];
            const auto rowsC = rowsA;
            const auto colsC = nb[0];
            std::cout << "Gemm Dims: m = " << rowsC << ", n = " << colsC << ", k = " << colsA << ", m*n*k = " << rowsC*colsC*colsA << std::endl;
            std::cout << "Par Dim: l = " << inner*outer << std::endl; 
        }
    }


    std::cout << "Memory: " << gbyte << " [GB]" << std::endl;
    std::cout << "Time : " << avg_time_s << " [s]" << std::endl;
    std::cout << "Gflops : " <<  gflops << " [gflops]" << std::endl;
    std::cout << "Performance : " <<  gflops/avg_time_s << " [gflops/s]" << std::endl;   
}

int main(int argc, char* argv[]) 
{

    using value    = float;
    using tensor   = tlib::tensor<value>;     // or std::array<value_t,N>
    using shape    = typename tensor::shape_t;

    assert(argc > 3);
    const auto p = std::stoi(argv[1]);
    assert(p >= 2 && p <= 10);  

    const auto q = std::stoi(argv[2]);
    assert(q >= 1 && q <= int(p));
    
    const auto astring = std::string("A[") + add_comma(get_index_a(q,p)) + "]";
    const auto bstring = std::string("B[") + add_comma(get_index_b(q,p)) + "]";
    const auto cstring = std::string("C[") + add_comma(get_index_c(q,p)) + "]";
    const auto mstring = std::string("x(") + std::to_string(q) + ")";
    
    std::cout << "ttm : " << cstring << " = " << astring << " " << mstring << " " << bstring << ";" << std::endl;

    auto na = shape(p,0);
    auto nb = shape(2u,0);
    auto nc = shape(p,0);
    for(auto i = 0; i < p; ++i){
        na[i] = std::stoi(argv[i+3]);
        nc[i] = std::stoi(argv[i+3]);
        assert(na[i] >= 1);
    }
    nb[0] = na[q-1];
    nb[1] = na[q-1];

    const auto pa = std::size_t(p);
    const auto pb = std::size_t(2);
    const auto pc = pa;

    // layout tuple for A and C
    const auto pia = tlib::detail::generate_k_order_layout(pa,1ul);
    const auto pib = tlib::detail::generate_k_order_layout(pb,1ul);
    const auto pic = tlib::detail::generate_k_order_layout(pc,1ul);

    auto A = tensor( na, pia );
    auto B = tensor( nb, pib );
    auto C = tensor( nc, pic );

    std::iota(A.begin(),A.end(),1);
    std::fill(B.begin(),B.end(),1);
    
    //std::cout << "Measure: <par-loop | slice-2d>" << std::endl;
    //measure(q, A, B, C, tlib::parallel_policy::omp_forloop,   tlib::slicing_policy::slice,     tlib::fusion_policy::all  );
    //std::cout << "---------" << std::endl << std::endl;
    
    std::cout << "Measure: <par-loop | slice-qd>" << std::endl;
    measure(q, A, B, C, tlib::parallel_policy::omp_forloop,   tlib::slicing_policy::subtensor, tlib::fusion_policy::outer);
    std::cout << "---------" << std::endl << std::endl;
    
    //std::cout << "Measure: <par-gemm | slice-2d>" << std::endl;
    //measure(q, A, B, C, tlib::parallel_policy::threaded_gemm, tlib::slicing_policy::slice,     tlib::fusion_policy::none );
    //std::cout << "---------" << std::endl << std::endl;
    
    std::cout << "Measure: <par-gemm | slice-qd>" << std::endl;
    measure(q, A, B, C, tlib::parallel_policy::threaded_gemm, tlib::slicing_policy::subtensor, tlib::fusion_policy::none );
    std::cout << "---------" << std::endl << std::endl;
     
    
}
