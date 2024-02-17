/*
 *   Copyright (C) 2024 Cem Bassoy (cem.bassoy@gmail.com)
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU Lesser General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <tlib/ttm.h>
#include <gtest/gtest.h>
#include "gtest_aux.h"

#include <vector>
#include <numeric>


template<class value_type, class size_type>
inline void 
ttm_init_recursive(
        const size_type r,
        const size_type q1,
        size_type j,
        size_type& k,
        std::vector<value_type> & a,
        std::vector<size_type> const& na,
        std::vector<size_type> const& wa,
        std::vector<size_type> const& pia)
{
    if(r==q1)
        ttm_init_recursive(r-1,q1,j,k, a, na, wa, pia);
    else if(r>0 && r!=q1)
        for(auto i = 0ul; i < na[pia[r-1]-1]; ++i, j+=wa[pia[r-1]-1])
            ttm_init_recursive(r-1,q1,j,k, a, na, wa, pia);
    else
        for(auto i = 0ul; i < na[pia[q1-1]-1]; ++i, j+=wa[pia[q1-1]-1])
            a[j] = ++k;
}

template<class value_type, class size_type>
inline void ttm_init(
        size_type const q,
        std::vector<value_type> & a,
        std::vector<size_type> const& na,
        std::vector<size_type> const& wa,
        std::vector<size_type> const& pia)
{
    assert(na.size() == wa .size());
    assert(wa.size() == pia.size());

    const size_type p  = na.size();
    assert(p>=2);
    assert(1<=q && q <= p);

    const size_type q1 = tlib::detail::inverse_mode(pia.begin(), pia.end(), q );
    assert(1<=q1 && q1 <= p);

    size_type k = 0ul;

    ttm_init_recursive(p,q1,0ul,k,a,na,wa,pia);
}



/* \brief creates a reference value
 *
 *
 * \param a input cube
 * \param j relative memory index except iq*wq
 * \param q contraction mode
*/
template<class value_type>
inline value_type refc(std::vector<value_type> const& a, std::size_t j, std::size_t jq) // unsigned q
{
    static auto sum = [](auto n) { return (n*(n+1))/2u; };

    // j = i(1)*w(1) + ... + i(q-1)*w(q-1) + i(q+1)*w(q+1) + ... + i(p)*w(p)

    return sum(a[jq]) - sum(a[j]-1.0);
};


template<class value_type, class size_type>
inline bool
ttm_check(
        const unsigned r,    // 1<=r<=p, initially p
        const unsigned q,    // 1<=q<=p
        const unsigned qhat, // 1<=qh<=p
        std::size_t ja,      // initially 0
        std::size_t jc,      // initially 0
        std::vector<value_type> const& a,
        std::vector<size_type> const& na,
        std::vector<size_type> const& wa,
        std::vector<size_type> const& pia,
        std::vector<value_type> const& c,
        std::vector<size_type> const& nc,
        std::vector<size_type> const& wc,
        std::vector<size_type> const& pic)
{


    if(r>0){
        if(r == qhat){
            return ttm_check(r-1,q,qhat, ja,jc, a,na,wa,pia, c,nc,wc, pic);
        }
        else{
            auto piar = pia[r-1]-1;
            auto picr = pic[r-1]-1;
            for(auto i = 0ul; i < nc[picr]; ++i, ja+=wa[piar], jc+=wc[picr])
                if(!ttm_check(r-1,q,qhat, ja,jc, a,na,wa,pia, c,nc,wc,pic))
                    return false;
            return true;
        }
    }
    else{
        auto q1 = q-1;
        auto jq = ja + (na[q1]-1)*wa[q1];
        auto ref = refc(a,ja,jq);
        for(auto i = 0ul; i < nc[q1]; ++i, jc+=wc[q1]){
            // EXPECT_FLOAT_EQ(c[jc],ref);

            testing::internal::FloatingPoint<value_type> lhs(c[jc]), rhs(ref);

            if (!lhs.AlmostEquals(rhs)) {
              return false;
            }
        }
        return true;
    }
}




//template<class value_type, class size_type, class execution_policy, class slicing_policy, class fusion_policy>
//inline void  ttm_check(execution_policy ep, slicing_policy sp, fusion_policy fp,
//                       unsigned const q, // mode
//                       std::vector<value_type> const& a,
//                       std::vector<value_type> const& b,
//                       std::vector<size_type> const& na, // shape tuple for a
//                       std::vector<size_type> const& nb,
//                       std::vector<size_type> const& wa, // stride tuple for a
//                       std::vector<size_type> const& pia) // layout tuple for a
//{
//    assert(na.size() == wa .size());
//    assert(wa.size() == pia.size());

//    const size_type p  = na.size();

//    assert(1u < p);
//    assert(1u <= q && q <= p);

//    auto const nq = na.at(q-1);
//    auto const m  = nb.at(0);
//    assert(nb.at(1) == nq);


//    auto pic = pia; // tlib::detail::generate_output_layout(pia,q);
//    auto nc  = na; // tlib::detail::generate_output_shape (na ,q);
//    nc.at(q-1) = m;
//    auto wc  = tlib::detail::generate_strides(nc ,pic );
//    auto nnc = std::accumulate(nc.begin(),nc.end(), 1ul, std::multiplies<size_type>());
//    auto c = std::vector(nnc,value_type{});

//    tlib::tensor_times_matrix(ep,sp,fp,  q,p,  a.data(), na.data(), wa.data(), pia.data(), b.data(), nb.data(), c.data(), nc.data(), wc.data());


//    auto A = tlib::gtest::matrix(na,a,pia);
//    auto B = tlib::gtest::matrix(nb,b,{2,1});
//    auto C = tlib::gtest::matrix(nc,c,pic);

//    auto qhat = tlib::detail::inverse_mode(pia.begin(),pia.end(),q);
//    assert(2u <= p);
//    assert(1u <= qhat && qhat <= p);

//    std::cout << "q=" << q << ", qhat=" << qhat << std::endl;

//    std::cout << "na = [ "; std::copy(na.begin(), na.end(), std::ostream_iterator<value_type>(std::cout, " ")); std::cout <<"];" << std::endl;
//    std::cout << "nc = [ "; std::copy(nc.begin(), nc.end(), std::ostream_iterator<value_type>(std::cout, " ")); std::cout <<"];" << std::endl;

//    std::cout << "A = " << A << std::endl;
//    std::cout << "B = " << B << std::endl;
//    std::cout << "C = " << C << std::endl;

//    auto compute_element = [nq](size_type i){return (nq*nq*(2ul*i-1ul)+nq)/2ul;};



//    ttm_check(p-1, qhat, 0ul, c, nc, wc, pic, compute_element) ;

//}





template<class value_type, class size_type, class execution_policy, class slicing_policy, class fusion_policy, unsigned rank>
inline void check_tensor_times_matrix(const size_type init, const size_type steps)
{

    auto init_v  = std::vector<size_type>(rank,init );
    auto steps_v = std::vector<size_type>(rank,steps);

    auto shapes  = tlib::gtest::generate_shapes      <size_type,rank> (init_v,steps_v);
    auto layouts = tlib::gtest::generate_permutations<size_type,rank> ();

    auto ep = execution_policy();
    auto sp = slicing_policy();
    auto fp = fusion_policy();


    for(auto const& na : shapes)
    {

        assert(tlib::detail::is_valid_shape(na.begin(), na.end()));

        auto nna = std::accumulate(na.begin(),na.end(), 1ul, std::multiplies<size_type>());
        auto a   = std::vector<value_type>(nna,value_type{});

        for(auto const& pia : layouts)
        {
            assert(tlib::detail::is_valid_layout(pia.begin(), pia.end()));

            auto wa = tlib::detail::generate_strides (na ,pia );

            assert(tlib::detail::is_valid_strides(pia.begin(), pia.end(),wa.begin()));

//            std::cout <<"pia = [ "; std::copy(pia.begin(), pia.end(), std::ostream_iterator<value_type>(std::cout, " ")); std::cout <<"];" << std::endl;

            for(auto q = 1ul; q <= rank; ++q)
            {
//                 if(q != 2u)
//                    continue;

                ttm_init(q,a,na,wa,pia);

                auto naq = na[q-1];

                auto nb = std::vector{naq*2,naq}; // mxnq
                auto b = std::vector(nb[0]*nb[1],value_type{1});


                const size_type p  = na.size();

                assert(1u < p);
                assert(1u <= q && q <= p);

                auto const nq = na.at(q-1);
                auto const m  = nb.at(0);
                assert(nb.at(1) == nq);


                auto pic = pia; // tlib::detail::generate_output_layout(pia,q);
                auto nc  = na; // tlib::detail::generate_output_shape (na ,q);
                nc.at(q-1) = m;
                auto wc  = tlib::detail::generate_strides(nc,pic);
                auto nnc = std::accumulate(nc.begin(),nc.end(), 1ul, std::multiplies<size_type>());
                auto c = std::vector(nnc,value_type{});

                auto A = tlib::gtest::tensor(na,a,pia);
                auto B = tlib::gtest::matrix(nb,b,{2,1});


                auto qh = tlib::detail::compute_qhat(pia.data(),p,q);


//                auto wa = A.w();
//                std::cout << "q=" << q << ", qh=" << qh << std::endl;
//                std::cout << "na = [ "; std::copy(na.begin(), na.end(), std::ostream_iterator<value_type>(std::cout, " ")); std::cout <<"];" << std::endl;
//                std::cout << "nc = [ "; std::copy(nc.begin(), nc.end(), std::ostream_iterator<value_type>(std::cout, " ")); std::cout <<"];" << std::endl;
//                std::cout << "wa = [ "; std::copy(wa.begin(), wa.end(), std::ostream_iterator<value_type>(std::cout, " ")); std::cout <<"];" << std::endl;
//                std::cout << "wc = [ "; std::copy(wc.begin(), wc.end(), std::ostream_iterator<value_type>(std::cout, " ")); std::cout <<"];" << std::endl;
//                std::cout << "A = " << A << std::endl;
//                std::cout << "B = " << B << std::endl;

                tlib::tensor_times_matrix(ep,sp,fp,  q,p,
                                          a.data(), na.data(), wa.data(), pia.data(),
                                          b.data(), nb.data(),
                                          c.data(), nc.data(), wc.data());

//                auto C = tlib::gtest::tensor(nc,c,pic);
//                std::cout << "C = " << C << std::endl;

                bool test = ttm_check(p,q,qh, 0u,0u,
                          a, na, wa, pia,
                          c, nc, wc, pic);

                EXPECT_TRUE(test);

            } // 1<=q<=p
            // break;
        } // layouts
        // break;
    } // shapes
}


TEST(TensorTimesMatrix, ThreadedGemmSlicesNoLoopFusion)
{
    using value_type       = double;
    using size_type        = std::size_t;
    using execution_policy = tlib::parallel_policy::threaded_gemm_t;
    using slicing_policy   = tlib::slicing_policy::slice_t;
    using fusion_policy    = tlib::fusion_policy::none_t;

    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,2u>(2u,3);
    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,3u>(2u,3);
    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,4u>(2u,3);
}


TEST(TensorTimesMatrix, ThreadedGemmSubtensorNoLoopFusion)
{
    using value_type       = double;
    using size_type        = std::size_t;
    using execution_policy = tlib::parallel_policy::threaded_gemm_t;
    using slicing_policy   = tlib::slicing_policy::subtensor_t;
    using fusion_policy    = tlib::fusion_policy::none_t;

    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,2u>(2u,3);
    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,3u>(2u,3);
    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,4u>(2u,3);
}




TEST(TensorTimesMatrix, OmpForLoopSlicesOuterFusion)
{
    using value_type       = double;
    using size_type        = std::size_t;
    using execution_policy = tlib::parallel_policy::omp_forloop_t;
    using slicing_policy   = tlib::slicing_policy::slice_t;
    using fusion_policy    = tlib::fusion_policy::outer_t;

    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,2u>(2u,3);
    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,3u>(2u,3);
    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,4u>(2u,3);
}



#if 0

TEST(TensorTimesVector, SequentialLargeSlicesNoLoopFusion)
{
    using value_type       = double;
    using size_type        = std::size_t;
    using execution_policy = tlib::execution::sequential_policy;
    using slicing_policy   = tlib::slicing::large_policy;
    using fusion_policy    = tlib::loop_fusion::none_policy;

    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,2u>(2u,3);
    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,3u>(2u,3);
    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,4u>(2u,3);
}



TEST(TensorTimesVector, ParallelLargeSlicesNoLoopFusion)
{
    using value_type       = double;
    using size_type        = std::size_t;
    using execution_policy = tlib::execution::parallel_policy;
    using slicing_policy   = tlib::slicing::large_policy;
    using fusion_policy    = tlib::loop_fusion::none_policy;

    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,2u>(2,3);
    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,3u>(2,3);
    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,4u>(2,3);
}


TEST(TensorTimesVector, ParallelBlasLargeSlicesNoLoopFusion)
{
    using value_type       = double;
    using size_type        = std::size_t;
    using execution_policy = tlib::execution::parallel_blas_policy;
    using slicing_policy   = tlib::slicing::large_policy;
    using fusion_policy    = tlib::loop_fusion::none_policy;

    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,2u>(2,3);
    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,3u>(2,3);
    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,4u>(2,3);
}


TEST(TensorTimesVector, ParallelBlasLargeSlicesCompleteLoopFusion)
{
    using value_type       = double;
    using size_type        = std::size_t;
    using execution_policy = tlib::execution::parallel_blas_policy;
    using slicing_policy   = tlib::slicing::large_policy;
    using fusion_policy    = tlib::loop_fusion::all_policy;

    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,2u>(2,3);
    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,3u>(2,3);
    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,4u>(2,3);
}





TEST(TensorTimesVector, SequentialSmallSlicesNoLoopFusion)
{
    using value_type       = double;
    using size_type        = std::size_t;
    using execution_policy = tlib::execution::sequential_policy;
    using slicing_policy   = tlib::slicing::small_policy;
    using fusion_policy    = tlib::loop_fusion::none_policy;

    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,2u>(2u,3u);
    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,3u>(2u,3u);
    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,4u>(2u,3u);
}

TEST(TensorTimesVector, ParallelSmallSlicesNoLoopFusion)
{
    using value_type       = double;
    using size_type        = std::size_t;
    using execution_policy = tlib::execution::parallel_policy;
    using slicing_policy   = tlib::slicing::small_policy;
    using fusion_policy    = tlib::loop_fusion::none_policy;

    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,2u>(2,3);
    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,3u>(2,3);
    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,4u>(2,3);
}


TEST(TensorTimesVector, ParallelBlasSmallSlicesNoLoopFusion)
{
    using value_type       = double;
    using size_type        = std::size_t;
    using execution_policy = tlib::execution::parallel_blas_policy;
    using slicing_policy   = tlib::slicing::small_policy;
    using fusion_policy    = tlib::loop_fusion::none_policy;

    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,2u>(2,3);
    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,3u>(2,3);
    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,4u>(2,3);
}


TEST(TensorTimesVector, ParallelBlasSmallSlicesOuterLoopFusion)
{
    using value_type       = double;
    using size_type        = std::size_t;
    using execution_policy = tlib::execution::parallel_blas_policy;
    using slicing_policy   = tlib::slicing::small_policy;
    using fusion_policy    = tlib::loop_fusion::outer_policy;

    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,2u>(2,3);
    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,3u>(2,3);
    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,4u>(2,3);
}


TEST(TensorTimesVector, ParallelBlasSmallSlicesCompleteLoopFusion)
{
    using value_type       = double;
    using size_type        = std::size_t;
    using execution_policy = tlib::execution::parallel_blas_policy;
    using slicing_policy   = tlib::slicing::small_policy;
    using fusion_policy    = tlib::loop_fusion::all_policy;

    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,2u>(2,3);
    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,3u>(2,3);
    check_tensor_times_matrix<value_type,size_type,execution_policy,slicing_policy,fusion_policy,4u>(2,3);
}

#endif

