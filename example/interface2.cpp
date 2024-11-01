#include <tlib/ttm.h>

#include <vector>
#include <numeric>
#include <iostream>

using namespace tlib::ttm;

int main()
{
    using value_t    = float;
    using tensor_t   = tensor<value_t>;     // or std::array<value_t,N>
    using shape_t    = typename tensor_t::shape_t;

    // shape tuple for A
    auto na = shape_t{4,3,2};

    // order of A and C
    auto pa = na.size();

    // contraction mode
    auto q = 1ul;

    // shape tuple for B
    auto nb = shape_t{na[q-1]+1,na[q-1]};
    auto pb = nb.size();

    // layout tuple for A and C
    auto pia = detail::generate_k_order_layout(pa,1ul);
    auto pib = detail::generate_k_order_layout(pb,1ul);

    auto A = tensor_t( na, pia );
    auto B = tensor_t( nb, pib );

    // strides are automatically computed. shape and layout verified.

    std::iota(A.begin(),A.end(),1);
    std::fill(B.begin(),B.end(),1);

    std::cout << "A = " << A << std::endl;
    std::cout << "B = " << B << std::endl;

    /*
      A =
      { 1  5  9  | 13 17 21
        2  6 10  | 14 18 22
        3  7 11  | 15 19 23
        4  8 12  | 16 20 24 };

      B =
      { 1  1  1  1  1
        1  1  1  1  1
        1  1  1  1  1
        1  1  1  1  1};
    */


  // correct shape, layout and strides of the output tensors C1,C2 are automatically computed and returned by the functions.  
    auto C1 = ttm(q, A,B, parallel_policy::parallel_blas , slicing_policy::slice,     fusion_policy::none );
    auto C2 = ttm(q, A,B, parallel_policy::parallel_loop , slicing_policy::slice,     fusion_policy::all );
    auto C3 = ttm(q, A,B, parallel_policy::parallel_loop , slicing_policy::subtensor, fusion_policy::all );
    auto C4 = ttm(q, A,B, parallel_policy::batched_gemm  , slicing_policy::subtensor, fusion_policy::all );


    std::cout << "C1 = " << C1 << std::endl;
    std::cout << "C2 = " << C2 << std::endl;
    std::cout << "C3 = " << C3 << std::endl;
    std::cout << "C4 = " << C4 << std::endl;

  

    /* for q=1
      C =
      { 1+..+4 5+..+8 9+..+12 | 13+..+16 17+..+20 21+..+24
          ..     ..     ..    |    ..       ..       ..
        1+..+4 5+..+8 9+..+12 | 13+..+16 17+..+20 21+..+24 };
    */
}
