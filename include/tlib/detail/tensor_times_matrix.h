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



#pragma once



#include <cstddef>
#include <stdexcept>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <thread>

#include "matrix_times_matrix.h"
#include "workload_computation.h"
#include "tags.h"
#include "cases.h"
#include "strides.h"
#include "index.h"


#ifdef USE_OPENBLAS
#include <cblas.h>
#endif

#ifdef USE_INTELBLAS
#include <mkl.h>
#endif




namespace tlib::detail{


template<class size_t>
inline void set_blas_threads(size_t num)
{
#ifdef USE_OPENBLAS
	openblas_set_num_threads(num);
#elif defined USE_INTELBLAS
	mkl_set_num_threads(num);
#endif
}

template<class size_t>
inline size_t compute_inverse_pi_q(size_t const*const pi,  size_t const p, size_t const q)
{
    size_t k = 0;
    for(; k<p; ++k)
        if(pi[k] == q)
            break;
    assert(k != p);
    auto const inv_pi_q = k+1; // pia^{-1}(m)
    assert(pi[inv_pi_q-1]==q);

    return inv_pi_q;
}



template<class size_t>
inline auto compute_ninvpia(size_t const*const na, size_t const*const pi, size_t inv_pi_q)
{
    assert(inv_pi_q>0);
	size_t nn = 1;
    for(size_t r = 0; r<(inv_pi_q-1); ++r){
        nn *= na[pi[r]-1];
    }

	return nn;
}



/* @brief Recursively executes gemm with tensor slices
 * 
 * @note is applied in tensor-times-matrix which uses tensor slices
 * @note gemm_t should be a general matrix-times-matrix function for matrices with row-major format
*/
template<class value_t, class size_t, class gemm_t>
inline void multiple_gemm_with_slices (
        gemm_t && gemm,
		size_t const r, // starts with p
//		size_t const q, // starts with p-1
		size_t const na_pia_1,
		size_t const na_m,
		size_t const wa_m,
        size_t const inv_pi_q, // one-based.
        value_t const*const __restrict a, size_t const*const na, size_t const*const wa, size_t const*const pia,
		value_t const*const __restrict b,
        value_t      *const __restrict c, size_t const*const nc, size_t const*const wc
		)
{    
	if(r>1){
        if (inv_pi_q == r) { // m == pia[p]
            //const auto qq = inv_pi_q == r ? q : q-1;
            multiple_gemm_with_slices(gemm, r-1,na_pia_1,na_m,wa_m,inv_pi_q,   a,na,wa,pia,  b,  c,nc,wc);
		}
        else{ //  inv_pi_q < r  --- m < pia[r]
            auto pia_r = pia[r-1]-1;
            for(unsigned i = 0; i < na[pia_r]; ++i){ // , a+=wa[pia[r-1]-1], c+=wc[pic[q-1]-1]
                multiple_gemm_with_slices(gemm, r-1,na_pia_1,na_m,wa_m,inv_pi_q,  a+i*wa[pia_r],na,wa,pia,  b,  c+i*wc[pia_r],nc,wc);
            }
		}
	}
	else {
        gemm( a,b,c, na_pia_1, na_m, wa_m  );
	}
}



/* @brief Recursively executes gemv over large tensor slices
 * 
 * @note is applied in tensor-times-vector which uses large tensor slices
 * @note gemv_t should be a general matrix-times-vector function for matrices of column-major format
 * @note pia_1[m]!=1 i.e. pia[1]!=m must hold!
*/
template<class value_t, class size_t, class gemm_t>
inline void multiple_gemm_with_subtensors (
        gemm_t && gemm, // should be gemv_col type
		size_t const r, // starts with p-1
//		size_t const q, // starts with p-1
		size_t const nn, // number of column elements of the matrix
        size_t const na_q, // number of row elements of the matrix
        size_t const wa_q,
        size_t const inv_pia_q, // one-based.
		value_t const*const __restrict a, size_t const*const na, size_t const*const wa, size_t const*const pia,
		value_t const*const __restrict b,
        value_t      *const __restrict c, size_t const*const nc, size_t const*const wc
		)
{
	assert(nn > 1);
    assert(inv_pia_q != 1);
	if(r>0){
        if (inv_pia_q >= r) {
            multiple_gemm_with_subtensors  (gemm, r-1,nn,  na_q,wa_q,inv_pia_q,   a,na,wa,pia,  b,  c,nc,wc);
		}
        else if (inv_pia_q < r){
            auto pia_r = pia[r-1]-1u;
            for(size_t i = 0; i < na[pia_r]; ++i){
                multiple_gemm_with_subtensors (gemm, r-1, nn,  na_q,wa_q,inv_pia_q,  a+i*wa[pia_r],na,wa,pia,  b,  c+i*wc[pia_r],nc,wc);
            }
		}
	}
	else {
        gemm(  a,b,c, nn, na_q, wa_q  );
	}
}




/**
 * \brief Implements a tensor-times-matrix-multiplication
 *
 * Performs a slice-times-vector operation in the most inner recursion level with subtensors of A and C
 *
 * It is a more sophisticated 2d-slice-times-vector implementation.
 *
 * @tparam value_t          type of the elements, usually float or double
 * @tparam size_t size      type of the extents, strides and layout elements, usually std::size_t
 * @tparam slicing_policy   type of the slicing method, i.e. small or large
 * @tparam loop_fusion      type of the loop fusion method, i.e. fusing none, all outer or even all free fusible loops
 * @tparam parallelization  type of the loop parallelization method, i.e. sequential, parallel or parallel with blas.
*/
template<class value_t, class size_t, class execution_policy, class slicing_policy, class fusion_policy>
inline void ttm(
	execution_policy, slicing_policy, fusion_policy,
	size_t const m, size_t const p,
	value_t const*const a, size_t const*const na, size_t const*const wa, size_t const*const pia,
	value_t const*const b, size_t const*const nb,
    value_t      *const c, size_t const*const nc, size_t const*const wc
	);



/*
 *
 *
*/
template<class value_t, class size_t>
inline void ttm(
			execution::sequential_policy, slicing::small_policy, loop_fusion::none_policy,
            size_t const q, size_t const p,
			value_t const*const a, size_t const*const na, size_t const*const wa, size_t const*const pia,
			value_t const*const b, size_t const*const nb,
            value_t      *const c, size_t const*const nc, size_t const*const wc
			)
{
    if(!is_case<8>(p,q,pia)){
         // mtm_rm(execution::seq, q, p,  a, na, wa, pia,  b, nb,  c, nc, wc);
	}
	else {
        auto const inv_pi_q = compute_inverse_pi_q( pia, p, q );
		size_t const na_pia_1 = na[pia[0]-1];
        auto gemm = tlib::detail::gemm_row::run<value_t>;

        multiple_gemm_with_slices(
           gemm, p, na_pia_1, na[q-1], wa[q-1], inv_pi_q, a, na, wa, pia, b,  c, nc, wc);
	}
}



/*
 *
 *
*/
template<class value_t, class size_t>
inline void ttm(
			execution::parallel_policy, slicing::small_policy, loop_fusion::none_policy,		
			size_t const m, size_t const p,
			value_t const*const a, size_t const*const na, size_t const*const wa, size_t const*const pia,
			value_t const*const b, size_t const*const nb,
            value_t      *const c, size_t const*const nc, size_t const*const wc
			)
{
	if(!is_case<8>(p,m,pia)) {
		set_blas_threads(std::thread::hardware_concurrency());
        mtm(execution::par,m, p,  a, na, wa, pia,  b, nb,  c, nc, wc);
	}
	else {
		assert(is_case<8>(p,m,pia));
	
        auto const inv_pi_q = compute_inverse_pi_q( pia, p, m );
		size_t const na_pia_1 = na[pia[0]-1];

		set_blas_threads(1);

		// m != pia[p]
		size_t pia_p = pia[p-1];
		assert(m > 0);
		assert(p>2);
        if(inv_pi_q == p) // m == pia[p]
            pia_p = pia[p-2];

		assert(pia[0]!=pia_p );
		assert(pia[p-1]!=m );

		const auto wa_pia_p = wa[pia_p-1];
        const auto wc_pic_p = wc[pia_p-1];

        auto gemm = tlib::detail::gemm_row::run<value_t>;

        #pragma omp parallel for schedule(dynamic) firstprivate(pia_p,p,m,   na_pia_1,inv_pi_q,   a,na,wa,pia,  b,c,nc,wc)
		for(size_t i = 0; i < na[pia_p-1]; ++i)
            multiple_gemm_with_slices(
                gemm, p-1, p-2, na_pia_1, na[m-1], wa[m-1], inv_pi_q, a+i*wa_pia_p, na, wa, pia, b,  c+i*wc_pic_p, nc, wc);
	}
}



/*
 *
 *
*/
//template<class value_t>
//struct TensorTimesVector<value_t,small_slices_tag,blas_tag,outer_tag>

template<class value_t, class size_t>
inline void ttm(
			execution::parallel_blas_policy, slicing::small_policy, loop_fusion::none_policy,	
            size_t const q,
			size_t const p,
			value_t const*const a, size_t const*const na, size_t const*const wa, size_t const*const pia,
			value_t const*const b, size_t const*const nb,
            value_t      *const c, size_t const*const nc, size_t const*const wc
			)
{

    if(!is_case<8>(p,q,pia)){
		set_blas_threads(std::thread::hardware_concurrency());
        // mtm(execution::blas, q, p,  a, na, wa, pia,  b, nb,  c, nc, wc);
	}
	else {
        assert(is_case<8>(p,q,pia));
		set_blas_threads(1);	
		
        auto const inv_pi_q = compute_inverse_pi_q( pia, p, q );
		size_t const na_pia_1 = na[pia[0]-1];

		// m != pia[p]
		size_t pia_p = pia[p-1];
        assert(q>0u);
        assert(p>2u);

        if(inv_pi_q == p) // m == pia[p]
            pia_p = pia[p-2];

		assert(pia[0]!=pia_p );
        assert(inv_pi_q != p);

		const auto wa_pia_p = wa[pia_p-1];
        const auto wc_pic_p = wc[pia_p-1];

        auto gemm = tlib::detail::gemm_row::run<value_t>;

        #pragma omp parallel for schedule(dynamic) firstprivate(pia_p,p,q,   na_pia_1,inv_pi_q,   a,na,wa,pia,  b,c,nc,wc)
		for(size_t i = 0; i < na[pia_p-1]; ++i)
            multiple_gemm_with_slices(
                gemm, p-1, na_pia_1, na[q-1], wa[q-1], inv_pi_q, a+i*wa_pia_p, na, wa, pia, b,  c+i*wc_pic_p, nc, wc);

	}
}



// parallel execution with blas using all free outer dimensions
//template<class value_t>
//struct TensorTimesVector<value_t,small_slices_tag,parallel_blas_tag,all_outer_tag> // _parallel_blas_3

template<class value_t, class size_t>
inline void ttm(
			execution::parallel_blas_policy, slicing::small_policy, loop_fusion::outer_policy,	
            size_t const q,
			size_t const p,
			value_t const*const a, size_t const*const na, size_t const*const wa, size_t const*const pia,
			value_t const*const b, size_t const*const nb,
            value_t      *const c, size_t const*const nc, size_t const*const wc
			)
{

    if(!is_case<8>(p,q,pia)){
		set_blas_threads(std::thread::hardware_concurrency());
        // mtm(execution::blas, m, p,  a, na, wa, pia,  b, nb,  c, nc, wc, pic);
	}
	else {
        assert(is_case<8>(p,q,pia));
		set_blas_threads(1);
		
        auto const inv_pi_q = compute_inverse_pi_q( pia, p, q );

        assert(q>0);
		assert(p>2);

		assert(pia[0]!=pia[p-1] );
        assert(inv_pi_q != p);
        assert(q != pia[p-1]);
        assert(p>inv_pi_q);
        assert(inv_pi_q>0);

		auto const na_pia_1 = na[pia[0]-1];

        auto const na_m = na[q-1];
        auto const wa_m = wa[q-1];

		auto num = 1u;
        for(auto i = inv_pi_q; i < p; ++i)
			num *= na[pia[i]-1];

        auto const wa_m1 = wa[pia[inv_pi_q]-1];
        auto const wc_m1 = wc[pia[inv_pi_q]-1];

        auto gemm = tlib::detail::gemm_row::run<value_t>;


        #pragma omp parallel for schedule(dynamic) firstprivate(p, q, num, wa_m1,wc_m1,inv_pi_q,  na_m,wa_m,na_pia_1, a,b,c)
		for(size_t k = 0u; k < num; ++k)
            multiple_gemm_with_slices ( gemm, inv_pi_q-1, inv_pi_q-1, na_pia_1, na[q-1], wa[q-1], inv_pi_q,  a+k*wa_m1 ,na,wa,pia,  b,  c+k*wc_m1,nc,wc );
	}
}





// parallel execution with blas using all free outer dimensions
//template<class value_t>
//struct TensorTimesVector<value_t,small_slices_tag,blas_tag,all_outer_inner_tag>// _parallel_blas_4

template<class value_t, class size_t>
inline void ttm(
			execution::parallel_blas_policy, slicing::small_policy, loop_fusion::all_policy,	
			size_t const m,
			size_t const p,
			value_t const*const a, size_t const*const na, size_t const*const wa, size_t const*const pia,
			value_t const*const b, size_t const*const nb,
            value_t      *const c, size_t const*const nc, size_t const*const wc
			)
{

	if(!is_case<8>(p,m,pia)){
		set_blas_threads(std::thread::hardware_concurrency());
        // mtm(execution::blas, m, p,  a, na, wa, pia,  b, nb,  c, nc, wc, pic);
	}
	else {
		assert(is_case<8>(p,m,pia));
        //auto const inv_pi_q = compute_inverse_pi_q( pia, pic, p, m );

		set_blas_threads(1);

		assert(m > 0);
		assert(p>2);

		assert(pia[0]!=pia[p-1] );
        //assert(inv_pi_q != p);
		assert(m != pia[p-1]);

		auto const na_pia_1 = na[pia[0]-1];

		auto const na_m = na[m-1];
		auto const wa_m = wa[m-1];

		auto const pia_pair = divide_layout(pia, p, m);
		auto const pia2 = pia_pair.second; // same for a and c
		assert(pia_pair.first.size() == 2);
		assert(pia2.size() > 0);

		auto const wa_pair = divide(wa, pia, p, m);
		auto const wa2 = wa_pair.second; // NOT same for a and c
		assert(wa_pair.first.size() == 2);
		assert(wa2.size() > 0);

        auto const wc_pair = divide(wc, pia, p-1);
		auto const wc2 = wc_pair.second; // NOT same for a and c
        assert(wc_pair.first.size() == 2);
		assert(wc2.size() > 0);

		assert(wc2.size() == wa2.size());

		auto const na_pair = divide(na, pia, p, m);
		auto const na2 = na_pair.second; // same for a and c
		assert(na2.size() > 0);
		
		auto const nn = std::accumulate(na2.begin(),na2.end(),1ul,std::multiplies<>());
		//auto const nn = na2.product();
		auto va2 = generate_strides(na2,pia2); // same for a and c


        auto gemm = tlib::detail::gemm_row::run<value_t>;



		#pragma omp parallel for schedule(dynamic) firstprivate(p, wc2, wa2,va2,pia2,  na_m,wa_m,na_pia_1, a,b,c)
        for(size_t k = 0ull; k < nn; ++k){
			auto ka = at_at_1(k, va2, wa2, pia2);
			auto kc = at_at_1(k, va2, wc2, pia2);
			auto const*const ap = a + ka;
			auto      *const cp = c + kc;
            // gemv_col_blas( ap,b,cp, na_pia_1, na_m, wa_m  );

            gemm( ap,b,cp, na_pia_1, na_m, wa_m  );
		}
	}
}




#if 0

/** \brief Implements a tensor-times-vector-multiplication with large tensor slices
 *
 * Performs a matrix-times-vector in the most inner recursion level where the matrix has the dimensions na_m x nn.
 * Squeezed matrix is a subtensor where we assume large memory
 * It is a very simply matrix-times-vector implementation.
 *
 *
*/
//template<class value_t>
//struct TensorTimesVector<value_t,large_slices_tag,sequential_tag,none_tag>

template<class value_t, class size_t>
inline void ttv(
			execution::sequential_policy, slicing::large_policy, loop_fusion::none_policy,
			size_t const m,
			size_t const p,
			value_t const*const __restrict a, size_t const*const na, size_t const*const wa, size_t const*const pia,
			value_t const*const __restrict b, size_t const*const nb,
			value_t      *const __restrict c, size_t const*const nc, size_t const*const wc, size_t const*const pic
			)
{

	if(!is_case<8>(p,m,pia)){
		mtv(execution::seq, m, p,  a, na, wa, pia,  b, nb,  c, nc, wc, pic);
	}
	else {
		assert(is_case<8>(p,m,pia));
        auto const inv_pi_q = compute_inverse_pi_q( pia, pic, p, m );

		assert(m>0);

		auto const na_m = na[m-1];
		auto const wa_m = wa[m-1];

        auto const n = compute_ninvpia( na, pia, inv_pi_q );
		assert(n == wa_m);
        multiple_gemm_with_subtensors( gemv_col<value_t,size_t>, p, p-1, n, na_m, wa_m, inv_pi_q, a, na, wa, pia, b,  c, nc, wc, pic);
	}
}
	
	

	// uses for case 8 the outer-most dimension for parallelization without BLAS.
	
//template<class value_t>
//struct TensorTimesVector<value_t,large_slices_tag,parallel_tag,outer_tag>
//{
//	static void run(
template<class value_t, class size_t>
inline void ttv(
			execution::parallel_policy, slicing::large_policy, loop_fusion::none_policy,	
			size_t const m,
			size_t const p,
			value_t const*const __restrict a, size_t const*const na, size_t const*const wa, size_t const*const pia,
			value_t const*const __restrict b, size_t const*const nb,
			value_t      *const __restrict c, size_t const*const nc, size_t const*const wc, size_t const*const pic
			)
{
	
	if(!is_case<8>(p,m,pia)){
		set_blas_threads(std::thread::hardware_concurrency());
		mtv(execution::par, m, p,  a, na, wa, pia,  b, nb,  c, nc, wc, pic);
	}
	else {
		assert(is_case<8>(p,m,pia));
		assert(m>0);
				
        auto const inv_pi_q = compute_inverse_pi_q( pia, pic, p, m );

		auto const na_m = na[m-1];
		auto const wa_m = wa[m-1];

		set_blas_threads(1);

		// m != pia[0] && m != pia[p-1]
		assert(p>2);
        assert(inv_pi_q != p);

		auto maxp = size_t{};
        for(auto k = inv_pi_q; k <= p; ++k)
            if(maxp < k) // inv_pi_q < maxp &&
				maxp = k;
		assert(maxp >= 2);
        assert(inv_pi_q != maxp);

        auto n = compute_ninvpia( na, pia, inv_pi_q ); // this is for the most inner computation
		assert(n == wa_m);
        #pragma omp parallel for schedule(dynamic) firstprivate(p,m,n,inv_pi_q,maxp,   a,na,wa,pia,  b,c,nc,wc,pic)
		for(size_t i = 0; i < na[pia[maxp-1]-1]; ++i)
            multiple_gemm_with_subtensors
                ( gemv_col<value_t,size_t>, p-1,p-2,n,  na_m,wa_m,inv_pi_q,  a+i*wa[pia[maxp-1]-1],na,wa,pia,  b,  c+i*wc[pic[maxp-2]-1],nc,wc,pic );
	}
}


// uses for case-8 the outer-most dimension for parallelization with BLAS.
//	static void run(
//template<class value_t>
//struct TensorTimesVector<value_t,large_slices_tag,parallel_blas_tag,outer_tag>  // _parallel_blas

template<class value_t, class size_t>
inline void ttv(
			execution::parallel_blas_policy, slicing::large_policy, loop_fusion::none_policy,
			size_t const m,
			size_t const p,
			value_t const*const __restrict a, size_t const*const na, size_t const*const wa, size_t const*const pia,
			value_t const*const __restrict b, size_t const*const nb,
			value_t      *const __restrict c, size_t const*const nc, size_t const*const wc, size_t const*const pic
			)
{
	
	if(!is_case<8>(p,m,pia)){
		set_blas_threads(std::thread::hardware_concurrency());
		mtv(execution::blas, m, p,  a, na, wa, pia,  b, nb,  c, nc, wc, pic);
	}
	else {					
		assert(m>0);			
		assert(p>2);
		assert(is_case<8>(p,m,pia));
					
        auto const inv_pi_q = compute_inverse_pi_q( pia, pic, p, m );
        assert(inv_pi_q!=p);

		auto const na_m = na[m-1];
		auto const wa_m = wa[m-1];

		set_blas_threads(1);

		auto maxp = size_t{};
        for(auto k = inv_pi_q; k <= p; ++k)
            if(maxp < k) // inv_pi_q < maxp &&
				maxp = k;

		assert(maxp >= 2);
        assert(inv_pi_q != maxp);

        auto n = compute_ninvpia( na, pia, inv_pi_q ); // this is for the most inner computation
		assert(n == wa_m);

        #pragma omp parallel for schedule(dynamic) firstprivate(p,m,n,inv_pi_q,maxp,   a,na,wa,pia,  b,c,nc,wc,pic)
		for(size_t i = 0; i < na[pia[maxp-1]-1]; ++i)
            multiple_gemm_with_subtensors
                (gemv_col_blas<value_t,size_t>, p-1,p-2,n,  na_m,wa_m,inv_pi_q,  a+i*wa[pia[maxp-1]-1],na,wa,pia,  b,  c+i*wc[pic[maxp-2]-1],nc,wc,pic);
	}
}

	// uses all available free dimensions for parallelization with BLAS.
	
//template<class value_t>
//struct TensorTimesVector<value_t,large_slices_tag,parallel_blas_tag,all_outer_tag> // _parallel_blas_3


template<class value_t, class size_t>
inline void ttv(
			execution::parallel_blas_policy, slicing::large_policy, loop_fusion::all_policy,
			size_t const m,
			size_t const p,
			value_t const*const __restrict a, size_t const*const na, size_t const*const wa, size_t const*const pia,
			value_t const*const __restrict b, size_t const*const nb,
			value_t      *const __restrict c, size_t const*const nc, size_t const*const wc, size_t const*const pic
			)
{	
	if(!is_case<8>(p,m,pia)){
		set_blas_threads(std::thread::hardware_concurrency());
		mtv(execution::blas, m, p,  a, na, wa, pia,  b, nb,  c, nc, wc, pic);
	}
	else {		
		assert(is_case<8>(p,m,pia));
		assert(m>0);
		
        auto const inv_pi_q = compute_inverse_pi_q( pia, pic, p, m );

		assert(p>2);
        assert(inv_pi_q!=p);

		set_blas_threads(1);

        assert(p>inv_pi_q);
        assert(inv_pi_q>0);

		auto const na_m = na[m-1];
		auto const wa_m = wa[m-1];

		auto num = 1u;
        for(auto i = inv_pi_q; i < p; ++i)
			num *= na[pia[i]-1];

        auto const wa_m1 = wa[pia[inv_pi_q  ]-1];
        auto const wc_m1 = wc[pic[inv_pi_q-1]-1];

        #pragma omp parallel for schedule(dynamic) firstprivate(p,m,inv_pi_q,wa_m1,wc_m1,   a,na,wa,pia,  b,c,nc,wc,pic)
		for(size_t i = 0; i < num; ++i)
            multiple_gemm_with_subtensors
                (gemv_col_blas<value_t,size_t>,  inv_pi_q-1, inv_pi_q-1, wa_m, na_m, wa_m, inv_pi_q, a+i*wa_m1,na,wa,pia,    b,   c+i*wc_m1,nc,wc,pic );
	}
}

#endif

} // namespace tlib::detail
