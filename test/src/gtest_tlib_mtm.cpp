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

#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <functional>
#include <ostream>

#include "../include/tlib/detail/mtm.h"
#include "../include/tlib/detail/layout.h"

#include "gtest_aux.h"


template<class matrix_type>
[[nodiscard]] matrix_type mtm(matrix_type  const& a, matrix_type  const& b)
{

    auto M = a.n()[0];
    auto K = a.n()[1];
    auto N = b.n()[1];
    assert(K == b.n()[0]);

    auto c = matrix_type ({M,N},0.0,a.pi());
    auto cmajor = a.is_cm();


    auto ar = std::cref(a);
    auto br = std::cref(b);
    auto cr = std::ref(c);



    if (cmajor)
    {
#pragma omp parallel for firstprivate(M,N,K, ar,br,cr)
        for(auto j = 0ul; j < N; ++j){
            for(auto k = 0ul; k < K; ++k){
                auto bb = br(k,j);
                for(auto i = 0ul; i < M; ++i){
                    cr(i,j) += ar(i,k) * bb;
                }
            }
        }
    }
    else // row
    {
#pragma omp parallel for firstprivate(M,N,K, ar,br,cr)
        for(auto i = 0ul; i < M; ++i){
            for(auto k = 0ul; k < K; ++k){
                auto aa = a(i,k);
                for(auto j = 0ul; j < N; ++j){
                    c(i,j) += aa * b(k,j);
                }
            }
        }
    }
    return c;
}


template<class matrix_type>
[[nodiscard]] matrix_type mtv(matrix_type  const& a, matrix_type  const& b)
{
    auto M = a.n()[0];
    auto N = a.n()[1];
    auto c = matrix_type ({M,1},0.0,a.pi());
    auto cmajor = a.is_cm();
    assert(b.n()[0] == N && b.n()[1] == 1);

    if (cmajor)
    {
        //#pragma omp parallel for collapse(2) firstprivate(M,N, ar,br,cr)
        for(auto j = 0ul; j < N; ++j){
            for(auto i = 0ul; i < M; ++i){
                c[i] += a(i,j) * b[j];
            }
        }
    }
    else // row
    {
        //#pragma omp parallel for firstprivate(M,N, ar,br,cr)
        for(auto i = 0ul; i < M; ++i){
            auto cc = c[i];
            //#pragma omp simd reduction(+:cc)
            for(auto j = 0ul; j < N; ++j){
                cc += a(i,j) * b[j];
            }
            c[i] = cc;
        }
    }
    return c;
}

template<class matrix_type>
[[nodiscard]] matrix_type vtm(matrix_type  const& a, matrix_type  const& b)
{
    auto M = a.n()[0];
    auto N = a.n()[1];
    auto c = matrix_type ({1,N},0.0,a.pi());
    auto cmajor = a.is_cm();

    assert(b.n()[0] == 1 && b.n()[1] == M);

    //auto ar = std::cref(a);
    //auto br = std::cref(b);
    //auto cr = std::ref(c);

    if (cmajor)
    {
        for(auto j = 0ul; j < N; ++j){
            auto cc = 0;
            for(auto i = 0ul; i < M; ++i){
                cc += b[i] * a(i,j);
            }
            c[j] = cc;
        }
    }
    else // row
    {
        for(auto i = 0ul; i < M; ++i){
            auto bb = b[i];
            for(auto j = 0ul; j < N; ++j){
                c[j] += bb * a(i,j);
            }
        }
    }
    return c;
}



/* \brief creates a reference value
 *
 *
 * \param a input matrix
 * \param i row number for q=1 and col number for q=2
 * \param q contraction mode
*/
template<class matrix_type>
inline typename matrix_type::value_type
refc(matrix_type const& a, std::size_t i, std::size_t q)
{
    auto M = a.n().at(0);
    auto N = a.n().at(1);

    auto sum = [](auto n) { return (n*(n+1))/2u; };

    assert(q == 1 || q == 2);

    if(q == 2) return sum(a(i,N-1)) - sum(a(i,0)-1.0);
    else       return sum(a(M-1,i)) - sum(a(0,i)-1.0);
}


/* \brief creates a reference value
 *
 *
 * \param a input cube
 * \param i index 1 (not q)
 * \param j index 2 (not q)
 * \param q contraction mode
*/
template<class cube_type>
inline typename cube_type::value_type
refc(cube_type const& a, std::size_t i, std::size_t j, std::size_t q)
{
    auto M = a.n().at(0);
    auto N = a.n().at(1);
    auto K = a.n().at(2);

    auto sum = [](auto n) { return (n*(n+1))/2u; };

    assert(q == 1 || q == 2 || q == 3);

    if(q == 3) return sum(a(i,j,K-1)) - sum(a(i,j,0)-1.0);
    if(q == 2) return sum(a(i,N-1,j)) - sum(a(i,0,j)-1.0);
    else       return sum(a(M-1,i,j)) - sum(a(0,i,j)-1.0);
}


/* \brief creates a reference value
 *
 *
 * \param a input cube
 * \param j relative memory index except iq*wq
 * \param q contraction mode
*/
template<class value_type>
inline value_type refc_general(std::vector<value_type> const& a, std::size_t j, std::size_t jq) // unsigned q
{
    static auto sum = [](auto n) { return (n*(n+1))/2u; };

    // j = i(1)*w(1) + ... + i(q-1)*w(q-1) + i(q+1)*w(q+1) + ... + i(p)*w(p)

    return sum(a[jq]) - sum(a[j]-1.0);
}


template<class value_type, class size_type>
inline void
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
            ttm_check(r-1,q,qhat, ja,jc, a,na,wa,pia, c,nc,wc, pic);
        }
        else{
            auto piar = pia[r-1]-1;
            auto picr = pic[r-1]-1;
            for(auto i = 0ul; i < nc[picr]; ++i, ja+=wa[piar], jc+=wc[picr])
                ttm_check(r-1,q,qhat, ja,jc, a,na,wa,pia, c,nc,wc,pic);
        }
    }
    else{
        auto q1 = q-1;
        auto jq = ja + (na[q1]-1)*wa[q1];
        auto ref = refc_general(a,ja,jq);
        for(auto i = 0ul; i < nc[q1]; ++i, jc+=wc[q1]){
            EXPECT_FLOAT_EQ(c[jc],ref);

        }
    }
}








TEST(MatrixTimesVector, Ref)
{
    using matrix      = tlib::gtest::matrix<double>;
    using indices     = typename matrix::shape_type;
    using permutation = typename matrix::layout_type;

    auto start = indices(2u,2u);
    auto steps = indices(2u,5u);

    auto shapes   = tlib::gtest::generate_shapes<std::size_t,2u>(start,steps);
    auto formats  = std::array<permutation,2>{permutation{1,2}, permutation{2,1} };

    for(auto const& n : shapes)
    {
        auto M = n[0];
        auto N = n[1];

        for(auto f : formats)
        {
            {
                auto q = 2;

                auto a = matrix({M,N}, 0.0, f);
                auto b = matrix({N,1}, 1.0, f);


                tlib::gtest::init(a,q);

                auto c = mtv(a,b);

                auto qh = tlib::detail::inverse_mode(a.pi().begin(),a.pi().end(),q);
                ttm_check(a.p(),q,qh, 0ul,0ul, a.container(),a.n(),a.w(),a.pi(), c.container(),c.n(),c.w(),c.pi());

            }

            {
                auto q = 1;

                auto a = matrix({M,N}, 0.0, f);
                auto b = matrix({1,M}, 1.0, f);

                tlib::gtest::init(a,q);

                auto c = vtm(a,b);

                auto qh = tlib::detail::inverse_mode(a.pi().begin(),a.pi().end(),q);
                ttm_check(a.p(),q,qh, 0ul,0ul, a.container(),a.n(),a.w(),a.pi(), c.container(),c.n(),c.w(),c.pi());

            }

        }
    }
}



TEST(MatrixTimesMatrix, Ref)
{
    using matrix      = tlib::gtest::matrix<double>;
    using indices     = typename matrix::shape_type;
    using permutation = typename matrix::layout_type;

    auto start = indices(2u,2u);
    auto steps = indices(2u,1u);

    auto shapes  = tlib::gtest::generate_shapes<std::size_t,2u>(start,steps);
    auto formats = std::array<permutation,2>{permutation{1,2}, permutation{2,1} };

    auto p = 2u;

    for(auto const& n : shapes)
    {
        auto M = n[0];
        auto N = n[1];
        auto K = n[1]*2;

        for(auto f : formats)
        {
            auto q = 2;

            auto a = matrix({M,K}, 0.0, f);
            auto b = matrix({K,N}, 1.0, f);


            init(a,q);

            auto c = mtm(a,b);

            auto qh = tlib::detail::inverse_mode(f.begin(),f.end(),q);
            ttm_check(p,q,qh, 0ul,0ul, a.container(),a.n(),a.w(),a.pi(), c.container(),c.n(),c.w(),c.pi());

        }
    }
}

// q=1 | A(n,1),C(m,1), B(m,n) = RM       | C = A x1 B => c = B *(rm) a
TEST(MatrixTimesMatrix, Case1)
{
    using matrix      = tlib::gtest::matrix<double>;
    using indices     = typename matrix::shape_type;
    using permutation = typename matrix::layout_type;

    auto start = indices(2u,2u);
    auto steps = indices(2u,4u);

    auto shapes = tlib::gtest::generate_shapes<std::size_t,2u>(start,steps);
    auto cm = permutation{1,2};
    auto rm = permutation{2,1};

    auto q = 1ul;
    auto p = 1ul;

    // C is row-major
    for(auto const& nb : shapes)
    {

        auto n = nb[1];
        auto m = nb[0];


        auto a = matrix({n,1}, 1.0, cm);
        auto b = matrix({m,n}, 1.0, rm);
        auto c = matrix({m,1}, 0.0, cm);

        const auto na = a.n();
        const auto nc = c.n();
        const auto pia = a.pi();

        tlib::gtest::init(b,2u);


        tlib::detail::mtm_rm(
                    q,p,
                    a.data(), na.data(), pia.data(),
                    b.data(), nb.data(),
                    c.data(), nc.data());

        for(auto i = 0ul; i < m; ++i)
            EXPECT_FLOAT_EQ(c[i], refc(b,i,2u) );

    }


    // B is column-major
    for(auto const& nb : shapes)
    {

        auto n = nb[1];
        auto m = nb[0];


        auto a = matrix({n,1}, 1.0, cm);
        auto b = matrix({m,n}, 1.0, cm);
        auto c = matrix({m,1}, 0.0, cm);

        const auto na = a.n();
        const auto nc = c.n();
        const auto pia = a.pi();

        tlib::gtest::init(b,2u);


        tlib::detail::mtm_cm(
                    q,p,
                    a.data(), na.data(), pia.data(),
                    b.data(), nb.data(),
                    c.data(), nc.data());

        for(auto i = 0ul; i < m; ++i)
            EXPECT_FLOAT_EQ(c[i], refc(b,i,2u) );

    }
}


// q=1 | A(m,n),C(u,n) = CM | B(u,m) = RM | C = A x1 B => C = A *(rm) B' 
TEST(MatrixTimesMatrix, Case2)
{
    using matrix      = tlib::gtest::matrix<double>;
    using indices     = typename matrix::shape_type;
    using permutation = typename matrix::layout_type;

    auto start = indices(2u,2u);
    auto steps = indices(2u,5u);

    constexpr auto shape_size = 2u;

    auto shapes = tlib::gtest::generate_shapes<std::size_t,shape_size>(start,steps);
    auto cm = permutation{1,2};
    auto rm = permutation{2,1};

    auto q = 1ul;
    auto p = shape_size;

    for(auto const& na : shapes)
    {

        auto m = na[0];
        auto n = na[1];
        auto u = m*2;

        ASSERT_TRUE(tlib::detail::is_case<2>(p,q,cm.data()));


        auto a = matrix({m,n}, 1.0, cm);
        auto b = matrix({u,m}, 1.0, rm);
        auto c = matrix({u,n}, 0.0, cm);
        const auto nb = b.n();
        const auto nc = c.n();
        const auto pia = a.pi();
        
        tlib::gtest::init(a,q);

        tlib::detail::mtm_rm(
                    q,p,
                    a.data(), na.data(), pia.data(),
                    b.data(), nb.data(),
                    c.data(), nc.data());

        auto qh = tlib::detail::inverse_mode(pia.begin(),pia.end(),q);
        ttm_check(p,q,qh, 0ul,0ul, a.container(),a.n(),a.w(),a.pi(), c.container(),c.n(),c.w(),c.pi());

    }

    for(auto const& na : shapes)
    {

        auto m = na[0];
        auto n = na[1];
        auto u = m*2;

        ASSERT_TRUE(tlib::detail::is_case<2>(p,q,cm.data()));


        auto a = matrix({m,n}, 1.0, cm);
        auto b = matrix({u,m}, 1.0, cm);
        auto c = matrix({u,n}, 0.0, cm);
        const auto nb = b.n();
        const auto nc = c.n();
        const auto pia = a.pi();

        tlib::gtest::init(a,q);

        tlib::detail::mtm_cm(
                    q,p,
                    a.data(), na.data(), pia.data(),
                    b.data(), nb.data(),
                    c.data(), nc.data());

        auto qh = tlib::detail::inverse_mode(pia.begin(),pia.end(),q);
        ttm_check(p,q,qh, 0ul,0ul, a.container(),a.n(),a.w(),a.pi(), c.container(),c.n(),c.w(),c.pi());

    }
}


// q=2 | A(m,n),C(m,u) = CM | B(u,n) = RM | C = A x2 B => C = B *(rm) A
TEST(MatrixTimesMatrix, Case3)
{
    using matrix      = tlib::gtest::matrix<double>;
    using indices     = typename matrix::shape_type;
    using permutation = typename matrix::layout_type;

    auto start = indices(2u,2u);
    auto steps = indices(2u,5u);

    constexpr auto shape_size = 2u;

    auto shapes = tlib::gtest::generate_shapes<std::size_t,shape_size>(start,steps);
    auto cm = permutation{1,2};
    auto rm = permutation{2,1};

    auto q = 2ul;
    auto p = shape_size;

    for(auto const& na : shapes)
    {

        auto m = na[0];
        auto n = na[1];
        auto u = m*2;

        ASSERT_TRUE(tlib::detail::is_case<3>(p,q,cm.data()));


        auto a = matrix({m,n}, 1.0, cm);
        auto b = matrix({u,n}, 1.0, rm);
        auto c = matrix({m,u}, 0.0, cm);
        const auto nb = b.n();
        const auto nc = c.n();
        const auto pia = a.pi();
        
        tlib::gtest::init(a,q);

        tlib::detail::mtm_rm(
                    q,p,
                    a.data(), na.data(), pia.data(),
                    b.data(), nb.data(),
                    c.data(), nc.data());

        auto qh = tlib::detail::inverse_mode(a.pi().begin(),a.pi().end(),q);
        ttm_check(p,q,qh, 0ul,0ul, a.container(),a.n(),a.w(),a.pi(), c.container(),c.n(),c.w(),c.pi());
    }


    for(auto const& na : shapes)
    {

        auto m = na[0];
        auto n = na[1];
        auto u = m*2;

        ASSERT_TRUE(tlib::detail::is_case<3>(p,q,cm.data()));


        auto a = matrix({m,n}, 1.0, cm);
        auto b = matrix({u,n}, 1.0, cm);
        auto c = matrix({m,u}, 0.0, cm);
        const auto nb = b.n();
        const auto nc = c.n();
        const auto pia = a.pi();

        tlib::gtest::init(a,q);

        tlib::detail::mtm_cm(
                    q,p,
                    a.data(), na.data(), pia.data(),
                    b.data(), nb.data(),
                    c.data(), nc.data());

        auto qh = tlib::detail::inverse_mode(a.pi().begin(),a.pi().end(),q);
        ttm_check(p,q,qh, 0ul,0ul, a.container(),a.n(),a.w(),a.pi(), c.container(),c.n(),c.w(),c.pi());
    }
}



// q=1 | A(m,n),C(u,n) = RM | B(u,m) = RM | C = A x1 B => C = B *(rm) A
TEST(MatrixTimesMatrix, Case4)
{
    using matrix      = tlib::gtest::matrix<double>;
    using indices     = typename matrix::shape_type;
    using permutation = typename matrix::layout_type;

    auto start = indices(2u,2u);
    auto steps = indices(2u,5u);

    constexpr auto shape_size = 2u;

    auto shapes = tlib::gtest::generate_shapes<std::size_t,shape_size>(start,steps);
    auto cm = permutation{1,2};
    auto rm = permutation{2,1};

    auto q = 1ul;
    auto p = shape_size;

    for(auto const& na : shapes)
    {

        auto m = na[0];
        auto n = na[1];
        auto u = m*2;

        ASSERT_TRUE(tlib::detail::is_case<4>(p,q,rm.data()));


        auto a = matrix({m,n}, 1.0, rm);
        auto b = matrix({u,m}, 1.0, rm);
        auto c = matrix({u,n}, 0.0, rm);
        const auto nb = b.n();
        const auto nc = c.n();
        const auto pia = a.pi();

        tlib::gtest::init(a,q);

        tlib::detail::mtm_rm(
                    q,p,
                    a.data(), na.data(), pia.data(),
                    b.data(), nb.data(),
                    c.data(), nc.data());

        auto qh = tlib::detail::inverse_mode(a.pi().begin(),a.pi().end(),q);
        ttm_check(p,q,qh, 0ul,0ul, a.container(),a.n(),a.w(),a.pi(), c.container(),c.n(),c.w(),c.pi());
    }


    for(auto const& na : shapes)
    {

        auto m = na[0];
        auto n = na[1];
        auto u = m*2;

        ASSERT_TRUE(tlib::detail::is_case<4>(p,q,rm.data()));


        auto a = matrix({m,n}, 1.0, rm);
        auto b = matrix({u,m}, 1.0, cm);
        auto c = matrix({u,n}, 0.0, rm);
        const auto nb = b.n();
        const auto nc = c.n();
        const auto pia = a.pi();

        tlib::gtest::init(a,q);

        tlib::detail::mtm_cm(
                    q,p,
                    a.data(), na.data(), pia.data(),
                    b.data(), nb.data(),
                    c.data(), nc.data());

        auto qh = tlib::detail::inverse_mode(a.pi().begin(),a.pi().end(),q);
        ttm_check(p,q,qh, 0ul,0ul, a.container(),a.n(),a.w(),a.pi(), c.container(),c.n(),c.w(),c.pi());
    }
}


// q=2 | A(m,n),C(m,u) = RM | B(u,n) = RM | C = A x2 B => C = A *(rm) B'
TEST(MatrixTimesMatrix, Case5)
{
    using matrix      = tlib::gtest::matrix<double>;
    using indices     = typename matrix::shape_type;
    using permutation = typename matrix::layout_type;

    auto start = indices(2u,2u);
    auto steps = indices(2u,5u);

    constexpr auto shape_size = 2u;

    auto shapes = tlib::gtest::generate_shapes<std::size_t,shape_size>(start,steps);
    auto cm = permutation{1,2};
    auto rm = permutation{2,1};

    auto q = 2ul;
    auto p = shape_size;

    for(auto const& na : shapes)
    {

        auto m = na[0];
        auto n = na[1];
        auto u = m*2;

        ASSERT_TRUE(tlib::detail::is_case<5>(p,q,rm.data()));


        auto a = matrix({m,n}, 1.0, rm);
        auto b = matrix({u,n}, 1.0, rm);
        auto c = matrix({m,u}, 0.0, rm);
        const auto nb = b.n();
        const auto nc = c.n();
        const auto pia = a.pi();

        tlib::gtest::init(a,q);

        tlib::detail::mtm_rm(
                    q,p,
                    a.data(), na.data(), pia.data(),
                    b.data(), nb.data(),
                    c.data(), nc.data());

        auto qh = tlib::detail::inverse_mode(a.pi().begin(),a.pi().end(),q);
        ttm_check(p,q,qh, 0ul,0ul, a.container(),a.n(),a.w(),a.pi(), c.container(),c.n(),c.w(),c.pi());

    }

    for(auto const& na : shapes)
    {

        auto m = na[0];
        auto n = na[1];
        auto u = m*2;

        ASSERT_TRUE(tlib::detail::is_case<5>(p,q,rm.data()));


        auto a = matrix({m,n}, 1.0, rm);
        auto b = matrix({u,n}, 1.0, cm);
        auto c = matrix({m,u}, 0.0, rm);
        const auto nb = b.n();
        const auto nc = c.n();
        const auto pia = a.pi();

        tlib::gtest::init(a,q);

        tlib::detail::mtm_rm(
                    q,p,
                    a.data(), na.data(), pia.data(),
                    b.data(), nb.data(),
                    c.data(), nc.data());

        auto qh = tlib::detail::inverse_mode(a.pi().begin(),a.pi().end(),q);
        ttm_check(p,q,qh, 0ul,0ul, a.container(),a.n(),a.w(),a.pi(), c.container(),c.n(),c.w(),c.pi());

    }
}



// q=pi(1) | A(nn,nq),C(nn,u)   , B(u,nq) = RM | C = A xq B => C = A *(rm) B'
TEST(MatrixTimesMatrix, Case6)
{
    using matrix      = tlib::gtest::matrix<double>;
    using cube        = tlib::gtest::cube<double>;
    using indices     = typename matrix::shape_type;
    using permutation = typename matrix::layout_type;

    auto start = indices(3u,2u);
    auto steps = indices(3u,4u);

    constexpr auto shape_size = 3u;

    auto shapes = tlib::gtest::generate_shapes<std::size_t,shape_size>(start,steps);
    auto rm = permutation{2,1};
    auto cm = permutation{1,2};
    auto p = shape_size;

    for(auto const& na : shapes)
    {

        auto q = 1ul;
        auto layouts = std::vector<permutation>{ {1,2,3}, {1,3,2} } ;

        for(auto const& layout : layouts)
        {

            auto M = na[0];
            auto N = na[1];
            auto K = na[2];
            auto U = na[q-1]*2;

            auto a = cube  ({M,N,K}, 1.0, layout);
            auto b = matrix({U,M},   1.0, rm);
            auto c = cube  ({U,N,K}, 0.0, layout);
            const auto nb = b.n();
            const auto nc = c.n();
            const auto pia = a.pi();

            ASSERT_TRUE(tlib::detail::is_case<6>(p,q,pia.data()));

            tlib::gtest::init(a,q);

            tlib::detail::mtm_rm( q,p,   a.data(), na.data(), pia.data(),  b.data(), nb.data(),  c.data(), nc.data());

            auto qh = tlib::detail::inverse_mode(a.pi().begin(),a.pi().end(),q);
            ttm_check(p,q,qh, 0ul,0ul, a.container(),a.n(),a.w(),a.pi(), c.container(),c.n(),c.w(),c.pi());

        }	// layouts


        for(auto const& layout : layouts)
        {

            auto M = na[0];
            auto N = na[1];
            auto K = na[2];
            auto U = na[q-1]*2;

            auto a = cube  ({M,N,K}, 1.0, layout);
            auto b = matrix({U,M},   1.0, cm);
            auto c = cube  ({U,N,K}, 0.0, layout);
            const auto nb = b.n();
            const auto nc = c.n();
            const auto pia = a.pi();

            ASSERT_TRUE(tlib::detail::is_case<6>(p,q,pia.data()));

            tlib::gtest::init(a,q);

            tlib::detail::mtm_cm( q,p,   a.data(), na.data(), pia.data(),  b.data(), nb.data(),  c.data(), nc.data());

            auto qh = tlib::detail::inverse_mode(a.pi().begin(),a.pi().end(),q);
            ttm_check(p,q,qh, 0ul,0ul, a.container(),a.n(),a.w(),a.pi(), c.container(),c.n(),c.w(),c.pi());

        }	// layouts

    } // shapes

    for(auto const& na : shapes)
    {

        auto q = 2ul;
        auto layouts = std::vector<permutation>{ {2,1,3}, {2,3,1} } ;

        for(auto const& layout : layouts)
        {

            auto M = na[0];
            auto N = na[1];
            auto K = na[2];
            auto U = na[q-1]*2;

            auto a = cube  ({M,N,K}, 1.0, layout);
            auto b = matrix({U,N},   1.0, rm);
            auto c = cube  ({M,U,K}, 0.0, layout);
            const auto nb = b.n();
            const auto nc = c.n();
            const auto pia = a.pi();

            ASSERT_TRUE(tlib::detail::is_case<6>(p,q,pia.data()));

            tlib::gtest::init(a, q);

            tlib::detail::mtm_rm( q,p,   a.data(), na.data(), pia.data(),  b.data(), nb.data(),  c.data(), nc.data());

            auto qh = tlib::detail::inverse_mode(a.pi().begin(),a.pi().end(),q);
            ttm_check(p,q,qh, 0ul,0ul, a.container(),a.n(),a.w(),a.pi(), c.container(),c.n(),c.w(),c.pi());


        }	// layouts

        for(auto const& layout : layouts)
        {

            auto M = na[0];
            auto N = na[1];
            auto K = na[2];
            auto U = na[q-1]*2;

            auto a = cube  ({M,N,K}, 1.0, layout);
            auto b = matrix({U,N},   1.0, cm);
            auto c = cube  ({M,U,K}, 0.0, layout);
            const auto nb = b.n();
            const auto nc = c.n();
            const auto pia = a.pi();

            ASSERT_TRUE(tlib::detail::is_case<6>(p,q,pia.data()));

            tlib::gtest::init(a, q);

            tlib::detail::mtm_cm( q,p,   a.data(), na.data(), pia.data(),  b.data(), nb.data(),  c.data(), nc.data());

            auto qh = tlib::detail::inverse_mode(a.pi().begin(),a.pi().end(),q);
            ttm_check(p,q,qh, 0ul,0ul, a.container(),a.n(),a.w(),a.pi(), c.container(),c.n(),c.w(),c.pi());


        }	// layouts

    } // shapes


    for(auto const& na : shapes)
    {

        auto q = 3ul;
        auto layouts = std::vector<permutation>{ {3,1,2}, {3,2,1} } ;

        for(auto const& layout : layouts)
        {

            auto M = na[0];
            auto N = na[1];
            auto K = na[2];
            auto U = na[q-1]*2;

            auto a = cube  ({M,N,K}, 1.0, layout);
            auto b = matrix({U,K},   1.0, rm);
            auto c = cube  ({M,N,U}, 0.0, layout);
            const auto nb = b.n();
            const auto nc = c.n();
            const auto pia = a.pi();

            ASSERT_TRUE(tlib::detail::is_case<6>(p,q,pia.data()));

            init(a, q);

            tlib::detail::mtm_rm( q,p,   a.data(), na.data(), pia.data(),  b.data(), nb.data(),  c.data(), nc.data());

            auto qh = tlib::detail::inverse_mode(a.pi().begin(),a.pi().end(),q);
            ttm_check(p,q,qh, 0ul,0ul, a.container(),a.n(),a.w(),a.pi(), c.container(),c.n(),c.w(),c.pi());

        }	// layouts


        for(auto const& layout : layouts)
        {

            auto M = na[0];
            auto N = na[1];
            auto K = na[2];
            auto U = na[q-1]*2;

            auto a = cube  ({M,N,K}, 1.0, layout);
            auto b = matrix({U,K},   1.0, cm);
            auto c = cube  ({M,N,U}, 0.0, layout);
            const auto nb = b.n();
            const auto nc = c.n();
            const auto pia = a.pi();

            ASSERT_TRUE(tlib::detail::is_case<6>(p,q,pia.data()));

            init(a, q);

            tlib::detail::mtm_cm( q,p,   a.data(), na.data(), pia.data(),  b.data(), nb.data(),  c.data(), nc.data());

            auto qh = tlib::detail::inverse_mode(a.pi().begin(),a.pi().end(),q);
            ttm_check(p,q,qh, 0ul,0ul, a.container(),a.n(),a.w(),a.pi(), c.container(),c.n(),c.w(),c.pi());

        }	// layouts

    } // shapes

}



// q=pi(p) | A(nn,nq),C(nn,u)   , B(u,nq) = RM | C = A xq B => C = A *(rm) B'
TEST(MatrixTimesMatrix, Case7)
{
    using cube        = tlib::gtest::cube<double>;
    using matrix      = tlib::gtest::matrix<double>;
    using indices     = typename matrix::shape_type;
    using permutation = typename matrix::layout_type;

    auto start = indices(3u,2u);
    auto steps = indices(3u,4u);

    constexpr auto shape_size = 3u;

    auto shapes = tlib::gtest::generate_shapes<std::size_t,shape_size>(start,steps);
    auto rm = permutation{2,1};
    auto cm = permutation{1,2};
    auto p = shape_size;

    for(auto const& na : shapes)
    {

        auto q = 1ul;
        auto layouts = std::vector<permutation>{ {3,2,1}, {2,3,1} } ;

        for(auto const& layout : layouts)
        {

            auto M = na[0];
            auto N = na[1];
            auto K = na[2];
            auto U = na[q-1]*2;

            auto a = cube  ({M,N,K}, 1.0, layout);
            auto b = matrix({U,M},   1.0, rm);
            auto c = cube  ({U,N,K}, 0.0, layout);
            const auto nb = b.n();
            const auto nc = c.n();
            const auto pia = a.pi();

            ASSERT_TRUE(tlib::detail::is_case<7>(p,q,pia.data()));

            tlib::gtest::init(a,q);

            tlib::detail::mtm_rm( q,p,   a.data(), na.data(), pia.data(),  b.data(), nb.data(),  c.data(), nc.data());

            auto qh = tlib::detail::inverse_mode(a.pi().begin(),a.pi().end(),q);
            ttm_check(p,q,qh, 0ul,0ul, a.container(),a.n(),a.w(),a.pi(), c.container(),c.n(),c.w(),c.pi());

        }	// layouts


        for(auto const& layout : layouts)
        {

            auto M = na[0];
            auto N = na[1];
            auto K = na[2];
            auto U = na[q-1]*2;

            auto a = cube  ({M,N,K}, 1.0, layout);
            auto b = matrix({U,M},   1.0, cm);
            auto c = cube  ({U,N,K}, 0.0, layout);
            const auto nb = b.n();
            const auto nc = c.n();
            const auto pia = a.pi();

            ASSERT_TRUE(tlib::detail::is_case<7>(p,q,pia.data()));

            tlib::gtest::init(a,q);

            tlib::detail::mtm_cm( q,p,   a.data(), na.data(), pia.data(),  b.data(), nb.data(),  c.data(), nc.data());

            auto qh = tlib::detail::inverse_mode(a.pi().begin(),a.pi().end(),q);
            ttm_check(p,q,qh, 0ul,0ul, a.container(),a.n(),a.w(),a.pi(), c.container(),c.n(),c.w(),c.pi());

        }	// layouts

    } // shapes

    for(auto const& na : shapes)
    {

        auto q = 2ul;
        auto layouts = std::vector<permutation>{ {1,3,2}, {3,1,2} } ;

        for(auto const& layout : layouts)
        {

            auto M = na[0];
            auto N = na[1];
            auto K = na[2];
            auto U = na[q-1]*2;

            auto a = cube  ({M,N,K}, 1.0, layout);
            auto b = matrix({U,N},   1.0, rm);
            auto c = cube  ({M,U,K}, 0.0, layout);
            const auto nb = b.n();
            const auto nc = c.n();
            const auto pia = a.pi();

            ASSERT_TRUE(tlib::detail::is_case<7>(p,q,pia.data()));

            tlib::gtest::init(a, q);

            tlib::detail::mtm_rm( q,p,   a.data(), na.data(), pia.data(),  b.data(), nb.data(),  c.data(), nc.data());

            auto qh = tlib::detail::inverse_mode(a.pi().begin(),a.pi().end(),q);
            ttm_check(p,q,qh, 0ul,0ul, a.container(),a.n(),a.w(),a.pi(), c.container(),c.n(),c.w(),c.pi());
        }	// layouts

        for(auto const& layout : layouts)
        {

            auto M = na[0];
            auto N = na[1];
            auto K = na[2];
            auto U = na[q-1]*2;

            auto a = cube  ({M,N,K}, 1.0, layout);
            auto b = matrix({U,N},   1.0, cm);
            auto c = cube  ({M,U,K}, 0.0, layout);
            const auto nb = b.n();
            const auto nc = c.n();
            const auto pia = a.pi();

            ASSERT_TRUE(tlib::detail::is_case<7>(p,q,pia.data()));

            tlib::gtest::init(a, q);

            tlib::detail::mtm_cm( q,p,   a.data(), na.data(), pia.data(),  b.data(), nb.data(),  c.data(), nc.data());

            auto qh = tlib::detail::inverse_mode(a.pi().begin(),a.pi().end(),q);
            ttm_check(p,q,qh, 0ul,0ul, a.container(),a.n(),a.w(),a.pi(), c.container(),c.n(),c.w(),c.pi());
        }	// layouts

    } // shapes


    for(auto const& na : shapes)
    {

        auto q = 3ul;
        auto layouts = std::vector<permutation>{ {1,2,3}, {2,1,3} } ;

        for(auto const& layout : layouts)
        {

            auto M = na[0];
            auto N = na[1];
            auto K = na[2];
            auto U = na[q-1]*2;

            auto a = cube  ({M,N,K}, 1.0, layout);
            auto b = matrix({U,K},   1.0, rm);
            auto c = cube  ({M,N,U}, 0.0, layout);
            const auto nb = b.n();
            const auto nc = c.n();
            const auto pia = a.pi();

            ASSERT_TRUE(tlib::detail::is_case<7>(p,q,pia.data()));

            tlib::gtest::init(a, q);

            tlib::detail::mtm_rm( q,p,   a.data(), na.data(), pia.data(),  b.data(), nb.data(),  c.data(), nc.data());

            auto qh = tlib::detail::inverse_mode(a.pi().begin(),a.pi().end(),q);
            ttm_check(p,q,qh, 0ul,0ul, a.container(),a.n(),a.w(),a.pi(), c.container(),c.n(),c.w(),c.pi());

        }	// layouts


        for(auto const& layout : layouts)
        {

            auto M = na[0];
            auto N = na[1];
            auto K = na[2];
            auto U = na[q-1]*2;

            auto a = cube  ({M,N,K}, 1.0, layout);
            auto b = matrix({U,K},   1.0, cm);
            auto c = cube  ({M,N,U}, 0.0, layout);
            const auto nb = b.n();
            const auto nc = c.n();
            const auto pia = a.pi();

            ASSERT_TRUE(tlib::detail::is_case<7>(p,q,pia.data()));

            tlib::gtest::init(a, q);

            tlib::detail::mtm_cm( q,p,   a.data(), na.data(), pia.data(),  b.data(), nb.data(),  c.data(), nc.data());

            auto qh = tlib::detail::inverse_mode(a.pi().begin(),a.pi().end(),q);
            ttm_check(p,q,qh, 0ul,0ul, a.container(),a.n(),a.w(),a.pi(), c.container(),c.n(),c.w(),c.pi());

        }	// layouts


    } // shapes
}








// q=pi(p) | A(nn,nq),C(nn,u)   , B(u,nq) = RM | C = A xq B => C = A *(rm) B'
TEST(MatrixTimesMatrix, Case8)
{
    using cube        = tlib::gtest::cube<double>;
    using matrix      = tlib::gtest::matrix<double>;
    using indices     = typename matrix::shape_type;
    using permutation = typename matrix::layout_type;

    auto start = indices(3u,2u);
    auto steps = indices(3u,1u);

    constexpr auto shape_size = 3u;

    auto shapes = tlib::gtest::generate_shapes<std::size_t,shape_size>(start,steps);
    auto rm = permutation{2,1};
    auto p = shape_size;

    for(auto const& na : shapes)
    {

        auto q = 1ul;
        auto layouts = std::vector<permutation>{ {3,2,1}, {2,3,1} } ;

        for(auto const& layout : layouts)
        {

            auto M = na[0];
            auto N = na[1];
            auto K = na[2];
            auto U = na[q-1]*2;

            auto a = cube  ({M,N,K}, 1.0, layout);
            auto b = matrix({U,M},   1.0, rm);
            auto c = cube  ({U,N,K}, 0.0, layout);
            const auto nb = b.n();
            const auto nc = c.n();
            const auto pia = a.pi();

            ASSERT_TRUE(tlib::detail::is_case<7>(p,q,pia.data()));

            init(a,q);

            tlib::detail::mtm_rm( q,p,   a.data(), na.data(), pia.data(),  b.data(), nb.data(),  c.data(), nc.data());

            //std::cout << "q = " << q << std::endl;
            //std::cout << "a = " << a << std::endl;
            //std::cout << "b = " << b << std::endl;
            //std::cout << "c = " << c << std::endl;

            for(auto k = 0u; k < K; ++k){
                for(auto j = 0u; j < N; ++j){
                    for(auto i = 0u; i < U; ++i){
                        EXPECT_FLOAT_EQ(c(i,j,k), refc(a,j,k,q) );
                    }
                }
            }
        }	// layouts
    } // shapes

    for(auto const& na : shapes)
    {

        auto q = 2ul;
        auto layouts = std::vector<permutation>{ {1,3,2}, {3,1,2} } ;

        for(auto const& layout : layouts)
        {

            auto M = na[0];
            auto N = na[1];
            auto K = na[2];
            auto U = na[q-1]*2;

            auto a = cube  ({M,N,K}, 1.0, layout);
            auto b = matrix({U,N},   1.0, rm);
            auto c = cube  ({M,U,K}, 0.0, layout);
            const auto nb = b.n();
            const auto nc = c.n();
            const auto pia = a.pi();

            ASSERT_TRUE(tlib::detail::is_case<7>(p,q,pia.data()));

            init(a, q);

            tlib::detail::mtm_rm( q,p,   a.data(), na.data(), pia.data(),  b.data(), nb.data(),  c.data(), nc.data());

            //std::cout << "q = " << q << std::endl;
            //std::cout << "a = " << a << std::endl;
            //std::cout << "b = " << b << std::endl;
            //std::cout << "c = " << c << std::endl;

            for(auto k = 0u; k < K; ++k){
                for(auto j = 0u; j < U; ++j){
                    for(auto i = 0u; i < M; ++i){
                        EXPECT_FLOAT_EQ(c(i,j,k), refc(a,i,k,q) );
                    }
                }
            }


        }	// layouts
    } // shapes


    for(auto const& na : shapes)
    {

        auto q = 3ul;
        auto layouts = std::vector<permutation>{ {1,2,3}, {2,1,3} } ;

        for(auto const& layout : layouts)
        {

            auto M = na[0];
            auto N = na[1];
            auto K = na[2];
            auto U = na[q-1]*2;

            auto a = cube  ({M,N,K}, 1.0, layout);
            auto b = matrix({U,K},   1.0, rm);
            auto c = cube  ({M,N,U}, 0.0, layout);
            const auto nb = b.n();
            const auto nc = c.n();
            const auto pia = a.pi();

            ASSERT_TRUE(tlib::detail::is_case<7>(p,q,pia.data()));

            init(a, q);

            tlib::detail::mtm_rm( q,p,   a.data(), na.data(), pia.data(),  b.data(), nb.data(),  c.data(), nc.data());

            //std::cout << "q = " << q << std::endl;
            //std::cout << "a = " << a << std::endl;
            //std::cout << "b = " << b << std::endl;
            //std::cout << "c = " << c << std::endl;

            for(auto k = 0u; k < U; ++k){
                for(auto j = 0u; j < N; ++j){
                    for(auto i = 0u; i < M; ++i){
                        EXPECT_FLOAT_EQ(c(i,j,k), refc(a,i,j,q) );
                    }
                }
            }


        }	// layouts
    } // shapes
}
