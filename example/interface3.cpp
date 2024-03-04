/*
# include either -DUSE_OPENBLAS or -DUSE_INTELBLAS for fast execution
g++ -I../include/ -std=c++17 -Ofast -fopenmp interface3.cpp -o interface3 && ./interface3
*/

#include <tlib/ttm.h>

#include <vector>
#include <numeric>
#include <iostream>


int main()
{
    using value_t    = float;
    using size_t     = std::size_t;
    using tensor_t   = std::vector<value_t>;     // or std::array<value_t,N>
    using shape_t   = std::vector<size_t>;
    using iterator_t = std::ostream_iterator<value_t>;

    auto na = shape_t{4,3,2};   // input shape tuple
    auto p = na.size();         // order of input tensor, i.e. number of dimensions - here 3
    auto k = 1ul;               // k-order of input tensor
    auto q = 2ul;

    auto pia = tlib::detail::generate_k_order_layout(p,k);  //  layout tuple of input tensor - here {1,2,3};
    auto wa  = tlib::detail::generate_strides(na,pia);      //  stride tuple of input tensor - here {1,4,12};
    auto nna  = std::accumulate(na.begin(),na.end(),1ul,std::multiplies<>()); // number of elements of input tensor

    auto pib = shape_t{1,2};
    auto nb  = shape_t{na[q-1]+1,na[q-1]};
    auto nnb  = std::accumulate(nb.begin(),nb.end(),1ul,std::multiplies<>()); // number of elements of input tensor

    auto nc = na;
    nc[q-1] = nb[0];
    auto pic = pia;
    auto wc  = tlib::detail::generate_strides(nc,pic);
    auto nnc  = std::accumulate(nc.begin(),nc.end(),1ul,std::multiplies<>()); // number of elements of input tensor


    auto A  = tensor_t(nna , 0.0f);
    auto B  = tensor_t(nnb , 1.0f);
    auto C1 = tensor_t(nnc , 0.0f);
    auto C2 = tensor_t(nnc , 0.0f);

    std::iota(A.begin(),A.end(),value_t{1});

    std::cout << "A = [ "; std::copy(A.begin(), A.end(), iterator_t(std::cout, " ")); std::cout << " ];" << std::endl;
    std::cout << "B = [ "; std::copy(B.begin(), B.end(), iterator_t(std::cout, " ")); std::cout << " ];" << std::endl;

    tlib::ttm(
        tlib::parallel_policy::threaded_gemm_t{} , tlib::slicing_policy::slice_t{},  tlib::fusion_policy::none_t{},
        q, p,
        A.data(), na.data(), wa.data(), pia.data(),
        B.data(), nb.data(),            pib.data(),
        C1.data(), nc.data(), wc.data());

    tlib::ttm(
        tlib::parallel_policy::omp_forloop_t{} , tlib::slicing_policy::subtensor_t{}, tlib::fusion_policy::outer_t{},
        q, p,
        A.data(), na.data(), wa.data(), pia.data(),
        B.data(), nb.data(),            pib.data(),
        C2.data(), nc.data(), wc.data());

    std::cout << "C1 = [ "; std::copy(C1.begin(), C1.end(), iterator_t(std::cout, " ")); std::cout << " ];" << std::endl;
    std::cout << "C2 = [ "; std::copy(C2.begin(), C2.end(), iterator_t(std::cout, " ")); std::cout << " ];" << std::endl;

}
