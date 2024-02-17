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
#include "cases_ttm.h"
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


/* @brief Computes \hat{q} = \pi^{-1}(q)
 *
 *
 * \hat{q} = \pi^{-1}(q) <=> q = \pi_{\hat{q}}
*/
template<class size_t>
inline size_t compute_qhat(size_t const*const pi,  unsigned const p, unsigned const q)
{
    unsigned k = 0;
    for(; k<p; ++k)
        if(pi[k] == q)
            break;
    assert(k != p);
    auto const qh = k+1; // pia^{-1}(m)
    assert(pi[qh-1]==q);

    return qh;
}




/* @brief Computes number of elements for the first pi_{1}...pi_{\hat{q}-1} modes
 *
 * \hat{q} = pi^{-1}(q)
 *
 * nn = prod_{k=1}^{\hat{q}-1} n_{\pi_{k}}
*/

template<class size_t>
inline auto product_qhat(size_t const*const n, size_t const*const pi, unsigned qh)
{
    assert(qh>0u);

	size_t nn = 1;
    for(unsigned r = 0; r<(qh-1); ++r){
        nn *= n[pi[r]-1];
    }

	return nn;
}


/* @brief Computes number of elements between modes start and finish
 *
 *
 * nn = n_{\pi_{start}} * n_{\pi_{start+1}} * ... * n_{\pi_{finish}}
 *
 * \param n dimension tuple
 * \param pi layout tuple
 * \param start starting mode (one-based)
 * \param finish end mode (one-based)
*/

template<class size_t>
inline auto product(size_t const*const n, size_t const*const pi, unsigned start, unsigned finish)
{

    size_t nn = 1;
    for(unsigned r = start-1; r<(finish-1); ++r){
        nn *= n[pi[r]-1];
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
        unsigned const r, // starts with p
        unsigned const q, // 1 <= q <= p
        unsigned const qh, // 1 <= qh <= p with \hat{q} = pi^{-1}(q)
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b,
              value_t *c, size_t const*const nc, size_t const*const wc
		)
{    
	if(r>1){
        if (r == qh) { // q == pia[r]
            multiple_gemm_with_slices(gemm, r-1, q, qh,   a,na,wa,pia,  b,  c,nc,wc);
		}
        else{ //  r>1 && r != qh
            auto pia_r = pia[r-1]-1;
            for(unsigned i = 0; i < na[pia_r]; ++i){ // , a+=wa[pia[r-1]-1], c+=wc[pic[q-1]-1]
                multiple_gemm_with_slices(gemm, r-1, q, qh,  a+i*wa[pia_r],na,wa,pia,  b,  c+i*wc[pia_r],nc,wc);
            }
		}
	}
	else {
        auto n1     = na[pia[0]-1];
        auto m      = nc[q-1];
        auto nq     = na[q-1];
        auto wq     = wa[q-1];

        gemm( b,a,c, m,n1,nq,  nq,wq,wq);  // ... m,n1,nq,   nq, n1, n1
	}
}



/* @brief Recursively executes gemv with subtensors
 * 
 * @note is applied in tensor-times-matrix with subtensors
 * @note gemm_t should be a general matrix-times-matrix function for matrices of row-major format
 * @note pia_1[q]!=1 i.e. pia[1]!=q must hold!
*/
template<class value_t, class size_t, class gemm_t>
inline void multiple_gemm_with_subtensors (
        gemm_t && gemm,
        unsigned const r, // starts with p
        unsigned const q,
        unsigned const qh, // qhat one-based
        unsigned const nnq, // prod_{k=1}^{qh-1} n_{pi_k}
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b,
              value_t *c, size_t const*const nc, size_t const*const wc
		)
{
    if(r>1){
        if (r <= qh) {
            multiple_gemm_with_subtensors  (gemm, r-1, q,qh,nnq,   a,na,wa,pia,  b,  c,nc,wc);
        }
        else if (r > qh){
            auto pia_r = pia[r-1]-1u;
            for(size_t i = 0; i < na[pia_r]; ++i){
                multiple_gemm_with_subtensors (gemm, r-1, q,qh,nnq,  a+i*wa[pia_r],na,wa,pia,  b,  c+i*wc[pia_r],nc,wc);
            }
        }
    }
    else {
        auto m      = nc[q-1];
        auto nq     = na[q-1];

        gemm( b,a,c, m,nnq,nq,  nq,nnq,nnq);
	}
}




/**
 * \brief Implements a tensor-times-matrix multiplication
 *
 *
 *
 * @tparam value_t          type of the elements, usually float or double
 * @tparam size_t size      type of the extents, strides and layout elements, usually std::size_t
 * @tparam slicing_policy   type of the slicing method, e.g. slice or subtensor
 * @tparam parallel_policy  type of the parallelization method, e.g. threaded_gemm, omp_taskloop, omp_forloop, batched_gemm
*/
template<class value_t, class size_t, class parallel_policy, class slicing_policy, class fusion_policy>
inline void ttm(
    parallel_policy, slicing_policy, fusion_policy,
    unsigned const q, unsigned const p,
    const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
    const value_t *b, size_t const*const nb,
          value_t *c, size_t const*const nc, size_t const*const wc
	);



/*
 *
 *
*/
template<class value_t, class size_t>
inline void ttm(
            parallel_policy::threaded_gemm_t, slicing_policy::slice_t, fusion_policy::none_t,
            unsigned const q, unsigned const p,
            const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
            const value_t *b, size_t const*const nb,
                  value_t *c, size_t const*const nc, size_t const*const wc
			)
{
    set_blas_threads(std::thread::hardware_concurrency());

    if(!is_case_rm<8>(p,q,pia)){
        mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
	}
	else {
        auto const qh = compute_qhat( pia, p, q );
        auto gemm = tlib::detail::gemm_row::run<value_t>;

        multiple_gemm_with_slices(gemm, p, q, qh,  a,na,wa,pia,   b,  c,nc,wc);
	}
}


// only parallelize the outer dimensions
template<class value_t, class size_t>
inline void ttm(
        parallel_policy::omp_forloop_t, slicing_policy::slice_t, fusion_policy::outer_t,
        unsigned const q, unsigned const p,
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b, size_t const*const nb,
              value_t *c, size_t const*const nc, size_t const*const wc
        )
{

    if(!is_case_rm<8>(p,q,pia)){
		set_blas_threads(std::thread::hardware_concurrency());
        mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
	}
	else {
        assert(is_case_rm<8>(p,q,pia));
        assert(p>2);
        assert(q>0);

		set_blas_threads(1);

        auto const qh = compute_qhat( pia, p, q );

        // num = n[pi[qh+1]] * n[pi[qh+2]] * ... * n[pi[p]]
        auto const num = product(na, pia, qh+1,p+1);

        // waq  = 1 * n[pi[1]] * n[pi[2]] * ... * n[pi[qh]] * n[pi[qh+1]]
        auto const waq = wa[pia[qh]-1];
        auto const wcq = wc[pia[qh]-1];

        auto gemm = tlib::detail::gemm_row::run<value_t>;

        #pragma omp parallel for schedule(dynamic) firstprivate(p,q,qh,num,a,b,c)
        for(size_t k = 0u; k < num; ++k){
            auto aa = a+k*waq;
            auto cc = c+k*wcq;

            multiple_gemm_with_slices ( gemm, qh, q, qh,  aa,na,wa,pia,  b,  cc,nc,wc );
        }
	}
}



// only parallelize the outer dimensions
template<class value_t, class size_t>
inline void ttm(
        parallel_policy::omp_forloop_t, slicing_policy::slice_t, fusion_policy::all_t,
        unsigned const q, unsigned const p,
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b, size_t const*const nb,
              value_t *c, size_t const*const nc, size_t const*const wc
        )
{

    if(!is_case_rm<8>(p,q,pia)){
        set_blas_threads(std::thread::hardware_concurrency());
        mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
    }
    else {
        assert(is_case_rm<8>(p,q,pia));
        assert(p>2);
        assert(q>0);

        set_blas_threads(1);

        auto const qh = compute_qhat( pia, p, q );

        // outer = n[pi[qh+1]] * n[pi[qh+2]] * ... * n[pi[p]]
        auto const outer = product(na, pia, qh+1,p+1);

        // w[pi[q]]  = 1 * n[pi[1]] * n[pi[2]] * ... * n[pi[qh-1]]
        auto const wao = wa[pia[qh]-1];
        auto const wco = wc[pia[qh]-1];

        // inner = n[pi[2]] * ... * n[pi[qh-1]] with pi[qh] = q
        auto const inner = product(na, pia, 2, qh);

        auto const wai = na[pia[0]-1];
        auto const wci = nc[pia[0]-1];


        auto m      = nc[q-1];
        auto nq     = na[q-1];
        auto n1     = na[pia[0]-1];
        auto wq     = wa[q-1];



        auto gemm = tlib::detail::gemm_row::run<value_t>;

        #pragma omp parallel for schedule(dynamic) collapse(2) firstprivate(p,q,qh,outer,inner,wai,wci,wao,wco,wq,n1,a,b,c)
        for(size_t k = 0u; k < outer; ++k){
            for(size_t j = 0u; j < inner; ++j){
                auto aa = a+k*wao + j*wai;
                auto cc = c+k*wco + j*wci;

                tlib::detail::gemm_row::run( b,aa,cc, m,n1,nq,  nq,wq,wq);
            }
        }
    }
}



template<class value_t, class size_t>
inline void ttm(
            parallel_policy::threaded_gemm_t, slicing_policy::subtensor_t, fusion_policy::none_t,
            unsigned const q, unsigned const p,
            const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
            const value_t *b, size_t const*const nb,
                  value_t *c, size_t const*const nc, size_t const*const wc
            )
{
    set_blas_threads(std::thread::hardware_concurrency());

    if(!is_case_rm<8>(p,q,pia)){
        mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
    }
    else {
        auto const qh  = compute_qhat( pia, p, q );
        // nnq = na[pi[1]] * na[pi[2]] * ... * na[pi[qh-1]]
        auto const nnq = product(na, pia, 1, qh); //product_qhat(na, pia, qh);
        auto gemm = tlib::detail::gemm_row::run<value_t>;

        multiple_gemm_with_subtensors(gemm, p, q, qh, nnq, a,na,wa,pia,   b,  c,nc,wc);
    }
}

template<class value_t, class size_t>
inline void ttm(
        parallel_policy::omp_forloop_t, slicing_policy::subtensor_t, fusion_policy::outer_t,
        unsigned const q, unsigned const p,
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b, size_t const*const nb,
              value_t *c, size_t const*const nc, size_t const*const wc
        )
{

    if(!is_case_rm<8>(p,q,pia)){
        set_blas_threads(std::thread::hardware_concurrency());
        mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
    }
    else {
        assert(is_case_rm<8>(p,q,pia));
        assert(q>0);
        assert(p>2);

        set_blas_threads(1);

        auto const qh = compute_qhat( pia, p, q );

        // num = n[pi[qh+1]] * n[pi[qh+2]] * ... * n[pi[p]]
        auto const num = product(na, pia, qh+1,p+1);

        // w[pi[q]]  = 1 * n[pi[1]] * n[pi[2]] * ... * n[pi[qh-1]]
        auto const waq = wa[pia[qh]-1];
        auto const wcq = wc[pia[qh]-1];

        // num = n[pi[1]] * n[pi[2]] * ... * n[pi[qh-1]] with pi[qh] = q
        auto const nnq = product(na, pia, 1, qh);

        auto m      = nc[q-1];
        auto nq     = na[q-1];

        #pragma omp parallel for schedule(dynamic) firstprivate(p, q, qh, m,nnq, num, waq,wcq, a,b,c)
        for(size_t k = 0u; k < num; ++k){

            auto aa = a+k*waq;
            auto cc = c+k*wcq;

            tlib::detail::gemm_row::run( b,aa,cc, m,nnq,nq,  nq,nnq,nnq);
        }
    }
}


template<class value_t, class size_t>
inline void ttm(
        parallel_policy::omp_forloop_t, slicing_policy::subtensor_t, fusion_policy::outer_t,
        unsigned const q, unsigned const p,
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b, size_t const*const nb,
              value_t *c, size_t const*const nc, size_t const*const wc
        )
{

    if(!is_case_rm<8>(p,q,pia)){
        set_blas_threads(std::thread::hardware_concurrency());
        mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
    }
    else {
        assert(is_case_rm<8>(p,q,pia));
        assert(q>0);
        assert(p>2);

        set_blas_threads(1);

        auto const qh = compute_qhat( pia, p, q );

        // num = n[pi[qh+1]] * n[pi[qh+2]] * ... * n[pi[p]]
        auto const num = product(na, pia, qh+1,p+1);

        // w[pi[q]]  = 1 * n[pi[1]] * n[pi[2]] * ... * n[pi[qh-1]]
        auto const waq = wa[pia[qh]-1];
        auto const wcq = wc[pia[qh]-1];

        // num = n[pi[1]] * n[pi[2]] * ... * n[pi[qh-1]] with pi[qh] = q
        auto const nnq = product(na, pia, 1, qh);

        auto m      = nc[q-1];
        auto nq     = na[q-1];

        #pragma omp parallel for schedule(dynamic) firstprivate(p, q, qh, m,nnq, num, waq,wcq, a,b,c)
        for(size_t k = 0u; k < num; ++k){

            auto aa = a+k*waq;
            auto cc = c+k*wcq;

            tlib::detail::gemm_row::run( b,aa,cc, m,nnq,nq,  nq,nnq,nnq);
        }
    }
}




//template<class value_t, class size_t>
//inline void ttm(
//            parallel_policy::omp_forloop, slicing_policy::slice_all,
//            size_t const q,
//			size_t const p,
//			value_t const*const a, size_t const*const na, size_t const*const wa, size_t const*const pia,
//			value_t const*const b, size_t const*const nb,
//            value_t      *const c, size_t const*const nc, size_t const*const wc
//			)
//{

//    if(!is_case<8>(p,q,pia)){
//		set_blas_threads(std::thread::hardware_concurrency());
//        mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
//	}
//	else {
//        assert(is_case<8>(p,q,pia));
//        //auto const qh = compute_qhat( pia, pic, p, m );

//		set_blas_threads(1);

//        assert(q>0u);
//        assert(p>2u);



//		assert(pia[0]!=pia[p-1] );
//        //assert(qh != p);
//        assert(q != pia[p-1]);


//        auto const n_pi_1 = na[pia[0]-1];

//        auto const m  = nc[q-1];
//        auto const nq = na[q-1];
//        auto const wq = wa[q-1];

//        assert(nb[0] == m);
//        assert(nb[1] == nq);


//        auto const pia_pair = divide_layout(pia, p, q);
//		auto const pia2 = pia_pair.second; // same for a and c
//		assert(pia_pair.first.size() == 2);
//		assert(pia2.size() > 0);

//        auto const wa_pair = divide(wa, pia, p, q);
//		auto const wa2 = wa_pair.second; // NOT same for a and c
//		assert(wa_pair.first.size() == 2);
//		assert(wa2.size() > 0);

//        auto const wc_pair = divide(wc, pia, p, q);
//		auto const wc2 = wc_pair.second; // NOT same for a and c
//        assert(wc_pair.first.size() == 2);
//		assert(wc2.size() > 0);

//		assert(wc2.size() == wa2.size());

//        auto const na_pair = divide(na, pia, p, q);
//		auto const na2 = na_pair.second; // same for a and c
//		assert(na2.size() > 0);
		
//		auto const nn = std::accumulate(na2.begin(),na2.end(),1ul,std::multiplies<>());
//		//auto const nn = na2.product();
//		auto va2 = generate_strides(na2,pia2); // same for a and c


//        auto gemm = tlib::detail::gemm_row::run<value_t>;



//        #pragma omp parallel for schedule(dynamic) firstprivate(p, wc2, wa2,va2,pia2,  m,nq,wq,n_pi_1, a,b,c)
//        for(size_t k = 0ull; k < nn; ++k){
//			auto ka = at_at_1(k, va2, wa2, pia2);
//			auto kc = at_at_1(k, va2, wc2, pia2);
//            auto const*const aa = a + ka;
//            auto      *const cc = c + kc;

//            //    A   B   C  M  N       K    LDA  LDB     LDC
//            gemm( b, aa, cc, m, n_pi_1, nq,  nq,  n_pi_1, n_pi_1);
//		}
//	}
//}




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
        auto const qh = compute_qhat( pia, pic, p, m );

		assert(m>0);

		auto const na_m = na[m-1];
		auto const wa_m = wa[m-1];

        auto const n = product_qhat( na, pia, qh );
		assert(n == wa_m);
        multiple_gemm_with_subtensors( gemv_col<value_t,size_t>, p, p-1, n, na_m, wa_m, qh, a, na, wa, pia, b,  c, nc, wc, pic);
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
				
        auto const qh = compute_qhat( pia, pic, p, m );

		auto const na_m = na[m-1];
		auto const wa_m = wa[m-1];

		set_blas_threads(1);

		// m != pia[0] && m != pia[p-1]
		assert(p>2);
        assert(qh != p);

		auto maxp = size_t{};
        for(auto k = qh; k <= p; ++k)
            if(maxp < k) // qh < maxp &&
				maxp = k;
		assert(maxp >= 2);
        assert(qh != maxp);

        auto n = product_qhat( na, pia, qh ); // this is for the most inner computation
		assert(n == wa_m);
        #pragma omp parallel for schedule(dynamic) firstprivate(p,m,n,qh,maxp,   a,na,wa,pia,  b,c,nc,wc,pic)
		for(size_t i = 0; i < na[pia[maxp-1]-1]; ++i)
            multiple_gemm_with_subtensors
                ( gemv_col<value_t,size_t>, p-1,p-2,n,  na_m,wa_m,qh,  a+i*wa[pia[maxp-1]-1],na,wa,pia,  b,  c+i*wc[pic[maxp-2]-1],nc,wc,pic );
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
					
        auto const qh = compute_qhat( pia, pic, p, m );
        assert(qh!=p);

		auto const na_m = na[m-1];
		auto const wa_m = wa[m-1];

		set_blas_threads(1);

		auto maxp = size_t{};
        for(auto k = qh; k <= p; ++k)
            if(maxp < k) // qh < maxp &&
				maxp = k;

		assert(maxp >= 2);
        assert(qh != maxp);

        auto n = product_qhat( na, pia, qh ); // this is for the most inner computation
		assert(n == wa_m);

        #pragma omp parallel for schedule(dynamic) firstprivate(p,m,n,qh,maxp,   a,na,wa,pia,  b,c,nc,wc,pic)
		for(size_t i = 0; i < na[pia[maxp-1]-1]; ++i)
            multiple_gemm_with_subtensors
                (gemv_col_blas<value_t,size_t>, p-1,p-2,n,  na_m,wa_m,qh,  a+i*wa[pia[maxp-1]-1],na,wa,pia,  b,  c+i*wc[pic[maxp-2]-1],nc,wc,pic);
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
		
        auto const qh = compute_qhat( pia, pic, p, m );

		assert(p>2);
        assert(qh!=p);

		set_blas_threads(1);

        assert(p>qh);
        assert(qh>0);

		auto const na_m = na[m-1];
		auto const wa_m = wa[m-1];

		auto num = 1u;
        for(auto i = qh; i < p; ++i)
			num *= na[pia[i]-1];

        auto const wa_m1 = wa[pia[qh  ]-1];
        auto const wc_m1 = wc[pic[qh-1]-1];

        #pragma omp parallel for schedule(dynamic) firstprivate(p,m,qh,wa_m1,wc_m1,   a,na,wa,pia,  b,c,nc,wc,pic)
		for(size_t i = 0; i < num; ++i)
            multiple_gemm_with_subtensors
                (gemv_col_blas<value_t,size_t>,  qh-1, qh-1, wa_m, na_m, wa_m, qh, a+i*wa_m1,na,wa,pia,    b,   c+i*wc_m1,nc,wc,pic );
	}
}

#endif

} // namespace tlib::detail
