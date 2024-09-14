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
#include <functional>

#include "mtm.h"
#include "tags.h"
#include "cases.h"
#include "strides.h"
#include "index.h"


#ifdef USE_OPENBLAS
#include <cblas.h>
#endif

#ifdef USE_MKL
#include <mkl.h>
#include <mkl_cblas.h>
#endif

#ifdef USE_BLIS
#include <blis.h>
#include <cblas.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif


namespace tlib::detail{

template<class size_t>
inline void set_blas_threads(size_t num)
{
#ifdef USE_OPENBLAS
	openblas_set_num_threads(num);
#elif defined USE_MKL
	mkl_set_num_threads(num);
#elif defined USE_BLIS
    //auto& rntm = get_blis_rntm();
    //bli_rntm_set_num_threads_only(num,&rntm);
    bli_thread_set_num_threads(num);
#endif
}

static inline unsigned get_blas_threads()
{
#ifdef USE_OPENBLAS
    return openblas_get_num_threads();
#elif defined USE_MKL
    return mkl_get_max_threads();
#elif defined USE_BLIS
    return bli_thread_get_num_threads();
    //auto& rntm = get_blis_rntm();
    //return bli_rntm_num_threads(&rntm);
#endif
}




static const unsigned hwthreads = omp_get_num_procs();

inline void set_blas_threads_max()
{
#ifdef USE_BLIS
    //auto& rntm = get_blis_rntm();
    //bli_rntm_set_thread_impl( BLIS_OPENMP, &rntm );
    //bli_rntm_set_num_threads(hwthreads, &rntm );
    set_blas_threads(hwthreads);
#else
    set_blas_threads(hwthreads); //hwthreads
#endif    
}

inline void set_blas_threads_min()
{
#ifdef USE_BLIS
    //auto& rntm = get_blis_rntm();
    //bli_rntm_set_thread_impl( BLIS_SINGLE, &rntm );
    //bli_rntm_set_num_threads( 1, &rntm );
#else
    set_blas_threads(1);
#endif
}


template<class size_t>
inline void set_omp_threads(unsigned num)
{
#ifdef _OPENMP
    omp_set_num_threads(num);
#endif
}


inline void set_omp_threads_max()
{
#ifdef _OPENMP
    omp_set_num_threads(hwthreads);
#else
    return 1;  
#endif
}

inline unsigned get_omp_threads()
{
#ifdef _OPENMP
    return omp_get_num_threads();
#else
    return 1u;
#endif
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
//        unsigned const q, // 1 <= q <= p
        unsigned const qh, // 1 <= qh <= p with \hat{q} = pi^{-1}(q)
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b,
              value_t *c, size_t const*const nc, size_t const*const wc
		)
{    
	if(r>1){
        if (r == qh) { // q == pia[r]
            multiple_gemm_with_slices(std::forward<gemm_t>(gemm), r-1, qh,   a,na,wa,pia,  b,  c,nc,wc);
		}
        else{ //  r>1 && r != qh
            auto pia_r = pia[r-1]-1;
            for(unsigned i = 0; i < na[pia_r]; ++i, a+=wa[pia_r], c+=wc[pia_r]){
                multiple_gemm_with_slices(std::forward<gemm_t>(gemm), r-1, qh,  a,na,wa,pia,  b,  c,nc,wc);
            }
		}
	}
	else {
        gemm( a, b, c );
	}
}



/* @brief Recursively executes gemm with subtensors
 * 
 * @note is applied in tensor-times-matrix with subtensors
 * @note gemm_t should be a general matrix-times-matrix function for matrices of row-major format
 * @note pia_1[q]!=1 i.e. pia[1]!=q must hold!
*/
template<class value_t, class size_t, class gemm_t>
inline void multiple_gemm_with_subtensors (
        gemm_t && gemm,
        unsigned const r, // starts with p
        unsigned const qh, // qhat one-based
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b,
              value_t *c, size_t const*const nc, size_t const*const wc
		)
{
    if(r>1){
        if (r <= qh) {
            multiple_gemm_with_subtensors  (std::forward<gemm_t>(gemm), r-1, qh,   a,na,wa,pia,  b,  c,nc,wc);
        }
        else if (r > qh){
            auto pia_r = pia[r-1]-1u;
            for(size_t i = 0; i < na[pia_r]; ++i, a+=wa[pia_r], c+=wc[pia_r]){
                multiple_gemm_with_subtensors (std::forward<gemm_t>(gemm), r-1, qh,  a,na,wa,pia,  b,  c,nc,wc);
            }
        }
    }
    else {
        gemm(a,b,c);
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
    const value_t *b, size_t const*const nb, size_t const*const pib,
          value_t *c, size_t const*const nc, size_t const*const wc
	);



template<class value_t, class size_t>
inline void ttm(
            parallel_policy::sequential_t, slicing_policy::slice_t, fusion_policy::none_t,
            unsigned const q, unsigned const p,
            const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
            const value_t *b, size_t const*const nb, size_t const*const pib,
                  value_t *c, size_t const*const nc, size_t const*const wc
			)
{
    set_blas_threads_min();
    assert(get_blas_threads() == 1);

    auto is_cm = pib[0] == 1;

    if(!is_case<8>(p,q,pia)){
        if(is_cm)
            mtm_cm(q, p,  a, na, pia, b, nb, c, nc );
        else
            mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
	}
	else {
        auto const qh = tlib::detail::inverse_mode(pia, pia+p, q);

        using namespace std::placeholders;

        auto n1     = na[pia[0]-1];
        auto m      = nc[q-1];
        auto nq     = na[q-1];
        auto wq     = wa[q-1];

        auto gemm_col = std::bind(tlib::detail::gemm_col_tr2::run<value_t>,_1,_2,_3,   n1,m,nq,   wq, m,wq); // a,b,c
        auto gemm_row = std::bind(tlib::detail::gemm_row::    run<value_t>,_2,_1,_3,   m,n1,nq,   nq,wq,wq); // b,a,c

        if(is_cm) multiple_gemm_with_slices(gemm_col, p, qh,  a,na,wa,pia,   b,  c,nc,wc);
        else      multiple_gemm_with_slices(gemm_row, p, qh,  a,na,wa,pia,   b,  c,nc,wc);
	}
}



template<class value_t, class size_t>
inline void ttm(
            parallel_policy::threaded_gemm_t, slicing_policy::slice_t, fusion_policy::none_t,
            unsigned const q, unsigned const p,
            const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
            const value_t *b, size_t const*const nb, size_t const*const pib,
                  value_t *c, size_t const*const nc, size_t const*const wc
			)
{
    set_blas_threads_max();
    assert(get_blas_threads() > 1u || get_blas_threads() <= hwthreads);

    auto is_cm = pib[0] == 1;

    if(!is_case<8>(p,q,pia)){
        if(is_cm)
            mtm_cm(q, p,  a, na, pia, b, nb, c, nc );
        else
            mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
	}
	else {
        auto const qh = tlib::detail::inverse_mode(pia, pia+p, q);

        using namespace std::placeholders;

        auto n1     = na[pia[0]-1];
        auto m      = nc[q-1];
        auto nq     = na[q-1];
        auto wq     = wa[q-1];

        auto gemm_col = std::bind(tlib::detail::gemm_col_tr2::run<value_t>,_1,_2,_3,   n1,m,nq,   wq, m,wq); // a,b,c
        auto gemm_row = std::bind(tlib::detail::gemm_row::    run<value_t>,_2,_1,_3,   m,n1,nq,   nq,wq,wq); // b,a,c

        if(is_cm) multiple_gemm_with_slices(gemm_col, p, qh,  a,na,wa,pia,   b,  c,nc,wc);
        else      multiple_gemm_with_slices(gemm_row, p, qh,  a,na,wa,pia,   b,  c,nc,wc);
	}
}


// only parallelize the outer dimensions
template<class value_t, class size_t>
inline void ttm(
        parallel_policy::omp_forloop_t, slicing_policy::slice_t, fusion_policy::outer_t,
        unsigned const q, unsigned const p,
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b, size_t const*const nb, size_t const*const pib,
              value_t *c, size_t const*const nc, size_t const*const wc
        )
{ 
    auto is_cm = pib[0] == 1;

    if(!is_case<8>(p,q,pia)){
        set_blas_threads_max();
        assert(get_blas_threads() > 1 || get_blas_threads() <= hwthreads);
        if(is_cm)
            mtm_cm(q, p,  a, na, pia, b, nb, c, nc );
        else
            mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
	}
	else {

        assert(is_case<8>(p,q,pia));
        assert(p>2);
        assert(q>0);

        auto const qh = tlib::detail::inverse_mode(pia, pia+p, q);

        // num = n[pi[qh+1]] * n[pi[qh+2]] * ... * n[pi[p]]
        auto const num = product(na, pia, qh+1,p+1);

        // waq  = 1 * n[pi[1]] * n[pi[2]] * ... * n[pi[qh]] * n[pi[qh+1]]
        auto const waq = wa[pia[qh]-1];
        auto const wcq = wc[pia[qh]-1];

        using namespace std::placeholders;

        auto n1     = na[pia[0]-1];
        auto m      = nc[q-1];
        auto nq     = na[q-1];
        auto wq     = wa[q-1];

        auto gemm_col = std::bind(tlib::detail::gemm_col_tr2::run<value_t>,_1,_2,_3,   n1,m,nq,   wq,m,wq); // a,b,c
        auto gemm_row = std::bind(tlib::detail::gemm_row::    run<value_t>,_2,_1,_3,   m,n1,nq,   nq,wq,wq); // b,a,c


        //unsigned incr = 0;

#pragma omp parallel for num_threads(hwthreads) firstprivate(qh,num,na,wa,pia,nc,wc,a,b,c,gemm_col, gemm_row, is_cm)
        for(size_t k = 0u; k < num; ++k){
            auto aa = a+k*waq;
            auto cc = c+k*wcq;          
            
            //if(incr++==0) std::cout << "OpenMP-Threads: " << get_omp_threads() << std::endl;

#ifndef USE_BLIS
            set_blas_threads_min();
            assert(get_blas_threads()==1);
#endif

            if(is_cm) multiple_gemm_with_slices(gemm_col, qh, qh,  aa,na,wa,pia,   b,  cc,nc,wc);
            else      multiple_gemm_with_slices(gemm_row, qh, qh,  aa,na,wa,pia,   b,  cc,nc,wc);

        }        
	}
}



// only parallelize the outer dimensions
template<class value_t, class size_t>
inline void ttm(
        parallel_policy::omp_forloop_t, slicing_policy::slice_t, fusion_policy::all_t,
        unsigned const q, unsigned const p,
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b, size_t const*const nb, size_t const*const pib,
              value_t *c, size_t const*const nc, size_t const*const wc
        )
{
    auto is_cm = pib[0] == 1;

    if(!is_case<8>(p,q,pia)){      
        set_blas_threads_max();
        assert(get_blas_threads() > 1 || get_blas_threads() <= hwthreads);

        if(is_cm)
            mtm_cm(q, p,  a, na, pia, b, nb, c, nc );
        else
            mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
    }
    else {
    
        assert(is_case<8>(p,q,pia));
        assert(p>2);
        assert(q>0);

        auto const qh = tlib::detail::inverse_mode(pia, pia+p, q);

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

        using namespace std::placeholders;
        auto gemm_col = std::bind(tlib::detail::gemm_col_tr2::run<value_t>,_1,_2,_3,   n1,m,nq,   wq, m,wq); // a,b,c
        auto gemm_row = std::bind(tlib::detail::gemm_row::    run<value_t>,_2,_1,_3,   m,n1,nq,   nq,wq,wq); // b,a,c

#pragma omp parallel for schedule(dynamic) collapse(2) num_threads(hwthreads) firstprivate(outer,inner,wai,wci,wao,wco,a,b,c, gemm_col, gemm_row, is_cm)
        for(size_t k = 0u; k < outer; ++k){
            for(size_t j = 0u; j < inner; ++j){
                auto aa = a+k*wao + j*wai;
                auto cc = c+k*wco + j*wci;

#ifndef USE_BLIS
                set_blas_threads_min();
                assert(get_blas_threads()==1);
#endif 
                assert(get_omp_threads ()==hwthreads);

                if(is_cm) gemm_col(aa, b, cc);
                else      gemm_row(aa, b, cc);
            }

        }
        
    }
}





// only parallelize the outer dimensions
template<class value_t, class size_t>
inline void ttm(
        parallel_policy::omp_forloop_and_threaded_gemm_t, slicing_policy::slice_t, fusion_policy::all_t,
        unsigned const q, unsigned const p,
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b, size_t const*const nb, size_t const*const pib,
              value_t *c, size_t const*const nc, size_t const*const wc
        )
{
    auto is_cm = pib[0] == 1;
    if(!is_case<8>(p,q,pia)){
        set_blas_threads_max();
        assert(get_blas_threads() > 1 || get_blas_threads() <= hwthreads);

        if(is_cm)
            mtm_cm(q, p,  a, na, pia, b, nb, c, nc );
        else
            mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
    }
    else {
        assert(is_case<8>(p,q,pia));
        assert(p>2);
        assert(q>0);

        auto const qh = tlib::detail::inverse_mode(pia, pia+p, q);

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

        using namespace std::placeholders;
        auto gemm_col = std::bind(tlib::detail::gemm_col_tr2::run<value_t>,_1,_2,_3,   n1,m,nq,   wq, m,wq); // a,b,c
        auto gemm_row = std::bind(tlib::detail::gemm_row::    run<value_t>,_2,_1,_3,   m,n1,nq,   nq,wq,wq); // b,a,c

#pragma omp parallel for schedule(dynamic) collapse(2) num_threads(hwthreads) firstprivate(outer,inner,wai,wci,wao,wco,a,b,c, gemm_col, gemm_row, is_cm)
        for(size_t k = 0u; k < outer; ++k){
            for(size_t j = 0u; j < inner; ++j){
                auto aa = a+k*wao + j*wai;
                auto cc = c+k*wco + j*wci;

#ifndef USE_BLIS
                set_blas_threads_max();
                assert(get_blas_threads() > 1 || get_blas_threads() <= hwthreads);
#endif

                if(is_cm) gemm_col(aa, b, cc);
                else      gemm_row(aa, b, cc);
            }
        }
    }
}







// only parallelize the outer dimensions
template<class value_t, class size_t>
inline void ttm(
        parallel_policy::omp_forloop_and_threaded_gemm_t, slicing_policy::slice_t, fusion_policy::all_t,
        unsigned const q, unsigned const p, double ratio,
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b, size_t const*const nb, size_t const*const pib,
              value_t *c, size_t const*const nc, size_t const*const wc
        )
{
    auto is_cm = pib[0] == 1;
    if(!is_case<8>(p,q,pia)){
        set_blas_threads_max();
        assert(get_blas_threads() > 1 || get_blas_threads() <= hwthreads);

        if(is_cm)
            mtm_cm(q, p,  a, na, pia, b, nb, c, nc );
        else
            mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
    }
    else {
        assert(is_case<8>(p,q,pia));
        assert(p>2);
        assert(q>0);

        auto const qh = tlib::detail::inverse_mode(pia, pia+p, q);

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

        using namespace std::placeholders;
        auto gemm_col = std::bind(tlib::detail::gemm_col_tr2::run<value_t>,_1,_2,_3,   n1,m,nq,   wq, m,wq); // a,b,c
        auto gemm_row = std::bind(tlib::detail::gemm_row::    run<value_t>,_2,_1,_3,   m,n1,nq,   nq,wq,wq); // b,a,c
        
        
        const auto ompthreads = unsigned (double(hwthreads)*ratio);
        const auto blasthreads = unsigned (double(hwthreads)*(1.0-ratio));        

#pragma omp parallel for schedule(dynamic) collapse(2) num_threads(ompthreads) firstprivate(outer,inner,wai,wci,wao,wco,a,b,c, gemm_col, gemm_row, is_cm)
        for(size_t k = 0u; k < outer; ++k){
            for(size_t j = 0u; j < inner; ++j){
                auto aa = a+k*wao + j*wai;
                auto cc = c+k*wco + j*wci;

#ifndef USE_BLIS
                set_blas_threads(blasthreads);
                assert(get_blas_threads() > 1 || get_blas_threads() <= hwthreads);
#endif

                if(is_cm) gemm_col(aa, b, cc);
                else      gemm_row(aa, b, cc);
            }
        }
    }
}





template<class value_t, class size_t>
inline void ttm(
            parallel_policy::sequential_t, slicing_policy::subtensor_t, fusion_policy::none_t,
            unsigned const q, unsigned const p,
            const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
            const value_t *b, size_t const*const nb, size_t const*const pib,
                  value_t *c, size_t const*const nc, size_t const*const wc
            )
{
    set_blas_threads_min();

    auto is_cm = pib[0] == 1;
    if(!is_case<8>(p,q,pia)){
        if(is_cm)
            mtm_cm(q, p,  a, na, pia, b, nb, c, nc );
        else
            mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
    }
    else {
        auto const qh  = tlib::detail::inverse_mode(pia, pia+p, q);
        auto const nnq = product(na, pia, 1, qh);
        auto const m   = nc[q-1];
        auto const nq  = na[q-1];


        using namespace std::placeholders;
        auto gemm_col = std::bind(tlib::detail::gemm_col_tr2::run<value_t>,_1,_2,_3,   nnq,m,nq,   nnq, m,nnq); // a,b,c
        auto gemm_row = std::bind(tlib::detail::gemm_row::    run<value_t>,_2,_1,_3,   m,nnq,nq,   nq,nnq,nnq); // b,a,c

        if(is_cm) multiple_gemm_with_subtensors(gemm_col, p, qh,  a,na,wa,pia,   b,  c,nc,wc);
        else      multiple_gemm_with_subtensors(gemm_row, p, qh,  a,na,wa,pia,   b,  c,nc,wc);

    }
}



template<class value_t, class size_t>
inline void ttm(
            parallel_policy::threaded_gemm_t, slicing_policy::subtensor_t, fusion_policy::none_t,
            unsigned const q, unsigned const p,
            const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
            const value_t *b, size_t const*const nb, size_t const*const pib,
                  value_t *c, size_t const*const nc, size_t const*const wc
            )
{
    set_blas_threads_max();
    assert(get_blas_threads() > 1 || get_blas_threads() <= hwthreads);
    

    auto is_cm = pib[0] == 1;
    if(!is_case<8>(p,q,pia)){
        if(is_cm)
            mtm_cm(q, p,  a, na, pia, b, nb, c, nc );
        else
            mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
    }
    else {
        auto const qh  = tlib::detail::inverse_mode(pia, pia+p, q);
        auto const nnq = product(na, pia, 1, qh);
        auto const m   = nc[q-1];
        auto const nq  = na[q-1];


        using namespace std::placeholders;
        auto gemm_col = std::bind(tlib::detail::gemm_col_tr2::run<value_t>,_1,_2,_3,   nnq,m,nq,   nnq, m,nnq); // a,b,c
        auto gemm_row = std::bind(tlib::detail::gemm_row::    run<value_t>,_2,_1,_3,   m,nnq,nq,   nq,nnq,nnq); // b,a,c

        if(is_cm) multiple_gemm_with_subtensors(gemm_col, p, qh,  a,na,wa,pia,   b,  c,nc,wc);
        else      multiple_gemm_with_subtensors(gemm_row, p, qh,  a,na,wa,pia,   b,  c,nc,wc);

    }
}

template<class value_t, class size_t>
inline void ttm(
        parallel_policy::omp_forloop_t, slicing_policy::subtensor_t, fusion_policy::outer_t,
        unsigned const q, unsigned const p,
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b, size_t const*const nb, size_t const*const pib,
              value_t *c, size_t const*const nc, size_t const*const wc
        )
{
    // mkl_set_dynamic(1);
    auto is_cm = pib[0] == 1;

    if(!is_case<8>(p,q,pia)){
        set_blas_threads_max();
        assert(get_blas_threads() > 1 || get_blas_threads() <= hwthreads);
        if(is_cm)
            mtm_cm(q, p,  a, na, pia, b, nb, c, nc );
        else
            mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
    }
    else {
    

    
        assert(is_case<8>(p,q,pia));
        assert(q>0);
        assert(p>2);

        auto const qh = tlib::detail::inverse_mode(pia, pia+p, q);

        // num = n[pi[qh+1]] * n[pi[qh+2]] * ... * n[pi[p]]
        auto const num = product(na, pia, qh+1,p+1);

        // w[pi[q]]  = 1 * n[pi[1]] * n[pi[2]] * ... * n[pi[qh-1]]
        auto const waq = wa[pia[qh]-1];
        auto const wcq = wc[pia[qh]-1];

        // num = n[pi[1]] * n[pi[2]] * ... * n[pi[qh-1]] with pi[qh] = q
        auto const nnq = product(na, pia, 1, qh);

        auto m      = nc[q-1];
        auto nq     = na[q-1];

        using namespace std::placeholders;
        auto gemm_col = std::bind(tlib::detail::gemm_col_tr2::run<value_t>,_1,_2,_3,   nnq,m,nq,   nnq, m,nnq); // a,b,c
        auto gemm_row = std::bind(tlib::detail::gemm_row::    run<value_t>,_2,_1,_3,   m,nnq,nq,   nq,nnq,nnq); // b,a,c
        
#pragma omp parallel for schedule(dynamic) num_threads(hwthreads) firstprivate(num, waq,wcq, a,b,c, gemm_col, gemm_row, is_cm)
        for(size_t k = 0u; k < num; ++k){
            auto aa = a+k*waq;
            auto cc = c+k*wcq;

#ifndef USE_BLIS
            set_blas_threads_min();
            assert(get_blas_threads()==1);
#endif
            assert(get_omp_threads()==hwthreads);
            
            if(is_cm) gemm_col(aa, b, cc);
            else      gemm_row(aa, b, cc);

        }      
    }
}




template<class value_t, class size_t>
inline void ttm(
        parallel_policy::omp_forloop_and_threaded_gemm_t, slicing_policy::subtensor_t, fusion_policy::outer_t,
        unsigned const q, unsigned const p,
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b, size_t const*const nb, size_t const*const pib,
              value_t *c, size_t const*const nc, size_t const*const wc
        )
{
    // mkl_set_dynamic(1);
    auto is_cm = pib[0] == 1;

    if(!is_case<8>(p,q,pia)){
        set_blas_threads_max();
        assert(get_blas_threads() > 1 || get_blas_threads() <= hwthreads);
        if(is_cm)
            mtm_cm(q, p,  a, na, pia, b, nb, c, nc );
        else
            mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
    }
    else {
        assert(is_case<8>(p,q,pia));
        assert(q>0);
        assert(p>2);

        auto const qh = tlib::detail::inverse_mode(pia, pia+p, q);

        // num = n[pi[qh+1]] * n[pi[qh+2]] * ... * n[pi[p]]
        auto const num = product(na, pia, qh+1,p+1);

        // w[pi[q]]  = 1 * n[pi[1]] * n[pi[2]] * ... * n[pi[qh-1]]
        auto const waq = wa[pia[qh]-1];
        auto const wcq = wc[pia[qh]-1];

        // num = n[pi[1]] * n[pi[2]] * ... * n[pi[qh-1]] with pi[qh] = q
        auto const nnq = product(na, pia, 1, qh);

        auto m      = nc[q-1];
        auto nq     = na[q-1];

        using namespace std::placeholders;
        auto gemm_col = std::bind(tlib::detail::gemm_col_tr2::run<value_t>,_1,_2,_3,   nnq,m,nq,   nnq, m,nnq); // a,b,c
        auto gemm_row = std::bind(tlib::detail::gemm_row::    run<value_t>,_2,_1,_3,   m,nnq,nq,   nq,nnq,nnq); // b,a,c

#pragma omp parallel for schedule(dynamic) num_threads(hwthreads) firstprivate(num, waq,wcq, a,b,c, gemm_col, gemm_row, is_cm)
        for(size_t k = 0u; k < num; ++k){
            auto aa = a+k*waq;
            auto cc = c+k*wcq;

#ifndef USE_BLIS
            set_blas_threads_max();
            assert(get_blas_threads() > 1 || get_blas_threads() <= hwthreads);
#endif
            if(is_cm) gemm_col(aa, b, cc);
            else      gemm_row(aa, b, cc);
        }
    }
}


template<class value_t, class size_t>
inline void ttm(
        parallel_policy::omp_forloop_and_threaded_gemm_t, slicing_policy::subtensor_t, fusion_policy::outer_t,
        unsigned const q, unsigned const p, double ratio,
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b, size_t const*const nb, size_t const*const pib,
              value_t *c, size_t const*const nc, size_t const*const wc
        )
{
    // mkl_set_dynamic(1);
    auto is_cm = pib[0] == 1;

    if(!is_case<8>(p,q,pia)){
        set_blas_threads_max();
        assert(get_blas_threads() > 1 || get_blas_threads() <= hwthreads);
        if(is_cm)
            mtm_cm(q, p,  a, na, pia, b, nb, c, nc );
        else
            mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
    }
    else {
        assert(is_case<8>(p,q,pia));
        assert(q>0);
        assert(p>2);

        auto const qh = tlib::detail::inverse_mode(pia, pia+p, q);

        // num = n[pi[qh+1]] * n[pi[qh+2]] * ... * n[pi[p]]
        auto const num = product(na, pia, qh+1,p+1);

        // w[pi[q]]  = 1 * n[pi[1]] * n[pi[2]] * ... * n[pi[qh-1]]
        auto const waq = wa[pia[qh]-1];
        auto const wcq = wc[pia[qh]-1];

        // num = n[pi[1]] * n[pi[2]] * ... * n[pi[qh-1]] with pi[qh] = q
        auto const nnq = product(na, pia, 1, qh);

        auto m      = nc[q-1];
        auto nq     = na[q-1];

        using namespace std::placeholders;
        auto gemm_col = std::bind(tlib::detail::gemm_col_tr2::run<value_t>,_1,_2,_3,   nnq,m,nq,   nnq, m,nnq); // a,b,c
        auto gemm_row = std::bind(tlib::detail::gemm_row::    run<value_t>,_2,_1,_3,   m,nnq,nq,   nq,nnq,nnq); // b,a,c
        
        const auto ompthreads = unsigned (double(hwthreads)*ratio);
        const auto blasthreads = unsigned (double(hwthreads)*(1.0-ratio));
        

#pragma omp parallel for schedule(dynamic) num_threads(ompthreads) firstprivate(num, waq,wcq, a,b,c, gemm_col, gemm_row, is_cm)
        for(size_t k = 0u; k < num; ++k){
            auto aa = a+k*waq;
            auto cc = c+k*wcq;

#ifndef USE_BLIS
            set_blas_threads(blasthreads);
            assert(get_blas_threads() > 1 || get_blas_threads() <= hwthreads);
#endif 

            if(is_cm) gemm_col(aa, b, cc);
            else      gemm_row(aa, b, cc);
        }
    }
}




template<class value_t, class size_t>
inline void ttm(
        parallel_policy::batched_gemm_t, slicing_policy::subtensor_t, fusion_policy::outer_t,
        unsigned const q, unsigned const p,
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b, size_t const*const nb, size_t const*const pib,
        value_t *c, size_t const*const nc, size_t const*const wc
        )
{
    auto is_cm = pib[0] == 1;

    // mkl_set_dynamic(1);
    set_blas_threads_max();
    assert(get_blas_threads() > 1 || get_blas_threads() <= hwthreads);
    
    if(!is_case<8>(p,q,pia)){
        if(is_cm)
            mtm_cm(q, p,  a, na, pia, b, nb, c, nc );
        else
            mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
    }
    else {
        assert(is_case<8>(p,q,pia));
        assert(q>0);
        assert(p>2);

        auto const qh = tlib::detail::inverse_mode(pia, pia+p, q);

        // num = n[pi[qh+1]] * n[pi[qh+2]] * ... * n[pi[p]]
        auto pp = product(na, pia, qh+1,p+1);

        // w[pi[q]]  = 1 * n[pi[1]] * n[pi[2]] * ... * n[pi[qh-1]]
        auto const waq = wa[pia[qh]-1];
        auto const wcq = wc[pia[qh]-1];

        // num = n[pi[1]] * n[pi[2]] * ... * n[pi[qh-1]] with pi[qh] = q
        auto const nnq = product(na, pia, 1, qh);

        auto m      = nc[q-1];
        auto nq     = na[q-1];

#ifdef USE_MKL
        using index_t = MKL_INT;
        using vector  = std::vector<value_t>;
        using ivector = std::vector<index_t>;
        using vvector = std::vector<value_t*>;

        const auto L =  is_cm ? CblasColMajor : CblasRowMajor;
        const auto Ta = std::vector<CBLAS_TRANSPOSE>(pp,CblasNoTrans);
        const auto Tb = std::vector<CBLAS_TRANSPOSE>(pp,is_cm ? CblasTrans : CblasNoTrans);

//        auto gemm_col = std::bind(tlib::detail::gemm_col_tr2::run<value_t>,_1,_2,_3,   nnq,m,nq,   nnq, m,nnq); // a,b,c
//        auto gemm_row = std::bind(tlib::detail::gemm_row::    run<value_t>,_2,_1,_3,   m,nnq,nq,   nq,nnq,nnq); // b,a,c

        auto Ma = ivector(pp, is_cm ? nnq :   m);
        auto Na = ivector(pp, is_cm ?   m : nnq);
        auto Ka = ivector(pp, nq);

        auto ALPHAa = vector(pp,1.0);
        auto BETAa  = vector(pp,0.0);

        auto LDAa = ivector(pp, is_cm ? nnq :  nq);
        auto LDBa = ivector(pp, is_cm ?   m : nnq);
        auto LDCa = ivector(pp, is_cm ? nnq : nnq);

        auto Ba = vvector(pp,nullptr);
        auto Aa = vvector(pp,nullptr); // (value_t*)b
        auto Ca = vvector(pp,nullptr);

        for(size_t k = 0u; k < pp; ++k){
            Aa[k] = is_cm ? (value_t*)a+k*waq : (value_t*)b;
            Ba[k] = is_cm ? (value_t*)b       : (value_t*)a+k*waq;
            Ca[k] = c+k*wcq;
        }

        const auto gcount = index_t(pp);
        const auto gsize  = ivector(pp,1);

        if constexpr (std::is_same_v<value_t,double>){
            cblas_dgemm_batch (L,Ta.data(),Tb.data(), Ma.data(),Na.data(),Ka.data(), ALPHAa.data(), (const value_t**)Aa.data(),LDAa.data(), (const value_t**)Ba.data(),LDBa.data(), BETAa.data(), Ca.data(),LDCa.data(), gcount, gsize.data());
        }
        else{
            cblas_sgemm_batch (L,Ta.data(),Tb.data(), Ma.data(),Na.data(),Ka.data(), ALPHAa.data(), (const value_t**)Aa.data(),LDAa.data(), (const value_t**)Ba.data(),LDBa.data(), BETAa.data(), Ca.data(),LDCa.data(), gcount, gsize.data());
        }
#endif
    }
}

} // namespace tlib::detail
