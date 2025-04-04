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

#include <cstdlib>
#include <cstdio>

namespace tlib::ttm::detail{

static inline unsigned get_number_cores() 
{

    const char* command_sockets = "lscpu | grep 'Socket' | cut -d':' -f2 | tr -d ' '";
    const char* command_cores   = "lscpu | grep 'socket' | cut -d':' -f2 | tr -d ' '";

    unsigned num_sockets = 0;
    unsigned num_cores_per_socket = 0;
    
    char output[10];

    FILE* fp = nullptr;
    fp = popen(command_sockets, "r");
    if (fp) {
        if(std::fgets(output, sizeof(output), fp) != nullptr)
            num_sockets = std::atoi(output);
        pclose(fp);
    }
    fp = popen(command_cores, "r");
    if (fp) {
        if(std::fgets(output, sizeof(output), fp) != nullptr)
            num_cores_per_socket = std::atoi(output);
        pclose(fp);
    } 

    unsigned num_cores = num_sockets*num_cores_per_socket;
    unsigned num_logical_cores = std::thread::hardware_concurrency();
    
    if(0u == num_cores || num_cores > num_logical_cores)
        num_cores = num_logical_cores;

    return num_cores;
}



static const unsigned cores = get_number_cores();


template<class size_t>
inline void set_blas_threads(size_t num)
{
#ifdef USE_OPENBLAS
    openblas_set_num_threads(num);
#elif defined USE_MKL
    mkl_set_num_threads(num);
#elif defined USE_BLIS
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
#endif
}



inline void set_blas_threads_max()
{
    set_blas_threads(cores);
}

inline void set_blas_threads_min()
{
    set_blas_threads(1);
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
    omp_set_num_threads(cores);
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

inline void set_omp_nested()
{
#if defined _OPENMP 
#if defined USE_OPENBLAS || defined USE_BLIS
  omp_set_nested(true);
#endif
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
inline void loops_over_gemm_with_slices (
        gemm_t && gemm,
        unsigned const r, // starts with p
        unsigned const qh, // 1 <= qh <= p with \hat{q} = pi^{-1}(q)
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b,
              value_t *c, size_t const*const nc, size_t const*const wc)
{
  if(r>1){
    if (r == qh) { // q == pia[r]
      loops_over_gemm_with_slices(std::forward<gemm_t>(gemm), r-1, qh,   a,na,wa,pia,  b,  c,nc,wc);
    }
    else{ //  r>1 && r != qh
      auto pia_r = pia[r-1]-1;
      for(unsigned i = 0; i < na[pia_r]; ++i, a+=wa[pia_r], c+=wc[pia_r]){
        loops_over_gemm_with_slices(std::forward<gemm_t>(gemm), r-1, qh,  a,na,wa,pia,  b,  c,nc,wc);
      }
    }
  }
  else {
    gemm( a, b, c );
  }
}



template<class value_t, class size_t, class gemm_t>
inline void taskloops_over_gemm_with_slices (
        gemm_t && gemm,
        unsigned const r, // starts with p
        unsigned const qh, // 1 <= qh <= p with \hat{q} = pi^{-1}(q)
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b,
              value_t *c, size_t const*const nc, size_t const*const wc)
{
  if(r>1){
    if (r == qh) { // q == pia[r]
      taskloops_over_gemm_with_slices(std::forward<gemm_t>(gemm), r-1, qh,   a,na,wa,pia,  b,  c,nc,wc);
    }
    else{ //  r>1 && r != qh
      auto pia_r = pia[r-1]-1;
#pragma omp taskloop untied
      for(unsigned i = 0; i < na[pia_r]; ++i){
        auto aa=a+i*wa[pia_r];
        auto cc=c+i*wc[pia_r];
        taskloops_over_gemm_with_slices(std::forward<gemm_t>(gemm), r-1, qh,  aa,na,wa,pia,  b,  cc,nc,wc);
      }
    }
  }
  else {
    gemm( a, b, c );
  }
}


template<class value_t, class size_t, class gemm_t>
inline void tasks_over_gemm_with_slices (
        gemm_t && gemm,
        unsigned const r, // starts with p
        unsigned const qh, // 1 <= qh <= p with \hat{q} = pi^{-1}(q)
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b,
              value_t *c, size_t const*const nc, size_t const*const wc)
{
#pragma omp task untied
  if(r>1){
    if (r == qh) { // q == pia[r]
      tasks_over_gemm_with_slices(std::forward<gemm_t>(gemm), r-1, qh,   a,na,wa,pia,  b,  c,nc,wc);
    }
    else{ //  r>1 && r != qh
      auto pia_r = pia[r-1]-1;
      for(unsigned i = 0; i < na[pia_r]; ++i){
        auto aa=a+i*wa[pia_r];
        auto cc=c+i*wc[pia_r];
        tasks_over_gemm_with_slices(std::forward<gemm_t>(gemm), r-1, qh,  aa,na,wa,pia,  b,  cc,nc,wc);
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
inline void loops_over_gemm_with_subtensors (
        gemm_t && gemm,
        unsigned const r, // starts with p
        unsigned const qh, // qhat one-based
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b,
              value_t *c, size_t const*const nc, size_t const*const wc )
{
  if(r>1){
    if (r <= qh) {
      loops_over_gemm_with_subtensors  (std::forward<gemm_t>(gemm), r-1, qh,   a,na,wa,pia,  b,  c,nc,wc);
    }
    else if (r > qh){
      auto pia_r = pia[r-1]-1u;
      for(size_t i = 0; i < na[pia_r]; ++i, a+=wa[pia_r], c+=wc[pia_r]){
        loops_over_gemm_with_subtensors (std::forward<gemm_t>(gemm), r-1, qh,  a,na,wa,pia,  b,  c,nc,wc);
      }
    }
  }
  else {
    gemm(a,b,c);
  }
}

template<class value_t, class size_t, class gemm_t>
inline void taskloops_over_gemm_with_subtensors (
        gemm_t && gemm,
        unsigned const r, // starts with p
        unsigned const qh, // qhat one-based
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b,
              value_t *c, size_t const*const nc, size_t const*const wc )
{
//#pragma omp task untied
  if(r>1){
    if (r <= qh) {
      taskloops_over_gemm_with_subtensors  (std::forward<gemm_t>(gemm), r-1, qh,   a,na,wa,pia,  b,  c,nc,wc);
    }
    else if (r > qh){
      auto pia_r = pia[r-1]-1u;
#pragma omp taskloop untied
      for(size_t i = 0; i < na[pia_r]; ++i){
        auto aa=a+i*wa[pia_r];
        auto cc=c+i*wc[pia_r];      
        taskloops_over_gemm_with_subtensors (std::forward<gemm_t>(gemm), r-1, qh,  aa,na,wa,pia,  b,  cc,nc,wc);
      }
    }
  }
  else {
    gemm(a,b,c);
  }
}


template<class value_t, class size_t, class gemm_t>
inline void tasks_over_gemm_with_subtensors (
        gemm_t && gemm,
        unsigned const r, // starts with p
        unsigned const qh, // qhat one-based
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b,
              value_t *c, size_t const*const nc, size_t const*const wc )
{
#pragma omp task untied
  if(r>1){
    if (r <= qh) {
      tasks_over_gemm_with_subtensors  (std::forward<gemm_t>(gemm), r-1, qh,   a,na,wa,pia,  b,  c,nc,wc);
    }
    else if (r > qh){
      auto pia_r = pia[r-1]-1u;
      for(size_t i = 0; i < na[pia_r]; ++i){
        auto aa=a+i*wa[pia_r];
        auto cc=c+i*wc[pia_r];      
        tasks_over_gemm_with_subtensors (std::forward<gemm_t>(gemm), r-1, qh,  aa,na,wa,pia,  b,  cc,nc,wc);
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
          value_t *c, size_t const*const nc, size_t const*const wc )
{
    std::cout << "This type of ttm is not yet defined" << std::endl;
}
	



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
          auto const qh = inverse_mode(pia, pia+p, q);

          using namespace std::placeholders;

          auto n1     = na[pia[0]-1];
          auto m      = nc[q-1];
          auto nq     = na[q-1];
          auto wq     = wa[q-1];

          auto gemm_col = std::bind(gemm_col_tr2::run<value_t>,_1,_2,_3,   n1,m,nq,   wq, m,wq); // a,b,c
          auto gemm_row = std::bind(gemm_row::    run<value_t>,_2,_1,_3,   m,n1,nq,   nq,wq,wq); // b,a,c

          if(is_cm) loops_over_gemm_with_slices(gemm_col, p, qh,  a,na,wa,pia,   b,  c,nc,wc);
          else      loops_over_gemm_with_slices(gemm_row, p, qh,  a,na,wa,pia,   b,  c,nc,wc);
    }
}



template<class value_t, class size_t>
inline void ttm(parallel_policy::parallel_blas_t, slicing_policy::slice_t, fusion_policy::none_t,
                unsigned const q, unsigned const p,
                const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
                const value_t *b, size_t const*const nb,                        size_t const*const pib,
                      value_t *c, size_t const*const nc, size_t const*const wc )
{
    set_blas_threads_max();
    assert(get_blas_threads() > 1u || get_blas_threads() <= cores);

    auto is_cm = pib[0] == 1;

    if(!is_case<8>(p,q,pia)){
        if(is_cm)
            mtm_cm(q, p,  a, na, pia, b, nb, c, nc );
        else
            mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
    }
    else {
        auto const qh = inverse_mode(pia, pia+p, q);

        using namespace std::placeholders;

        auto n1     = na[pia[0]-1];
        auto m      = nc[q-1];
        auto nq     = na[q-1];
        auto wq     = wa[q-1];

        auto gemm_col = std::bind(gemm_col_tr2::run<value_t>,_1,_2,_3,   n1,m,nq,   wq, m,wq); // a,b,c
        auto gemm_row = std::bind(gemm_row::    run<value_t>,_2,_1,_3,   m,n1,nq,   nq,wq,wq); // b,a,c

        if(is_cm) loops_over_gemm_with_slices(gemm_col, p, qh,  a,na,wa,pia,   b,  c,nc,wc);
        else      loops_over_gemm_with_slices(gemm_row, p, qh,  a,na,wa,pia,   b,  c,nc,wc);
    }
}


template<class value_t, class size_t>
inline void ttm(parallel_policy::parallel_taskloop_t, slicing_policy::slice_t, fusion_policy::none_t,
                unsigned const q, unsigned const p,
                const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
                const value_t *b, size_t const*const nb,                        size_t const*const pib,
                      value_t *c, size_t const*const nc, size_t const*const wc )
{
    set_omp_nested();
    auto is_cm = pib[0] == 1;
    if(!is_case<8>(p,q,pia)){
        set_blas_threads_max();
        assert(get_blas_threads() > 1u || get_blas_threads() <= cores);    
        if(is_cm)
            mtm_cm(q, p,  a, na, pia, b, nb, c, nc );
        else
            mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
    }
    else {
        auto const qh = inverse_mode(pia, pia+p, q);

        using namespace std::placeholders;

        auto n1     = na[pia[0]-1];
        auto m      = nc[q-1];
        auto nq     = na[q-1];
        auto wq     = wa[q-1];

        auto gemm_col = std::bind(gemm_col_tr2::run<value_t>,_1,_2,_3,   n1,m,nq,   wq, m,wq); // a,b,c
        auto gemm_row = std::bind(gemm_row::    run<value_t>,_2,_1,_3,   m,n1,nq,   nq,wq,wq); // b,a,c
        
#pragma omp parallel num_threads(cores) 
        {
            set_blas_threads_min();
            assert(get_blas_threads() == 1u);        
#pragma omp single
            {
                  if(is_cm) taskloops_over_gemm_with_slices(gemm_col, p, qh,  a,na,wa,pia,   b,  c,nc,wc);
                  else      taskloops_over_gemm_with_slices(gemm_row, p, qh,  a,na,wa,pia,   b,  c,nc,wc);
            }
        }
    }
}


template<class value_t, class size_t>
inline void ttm(parallel_policy::parallel_task_t, slicing_policy::slice_t, fusion_policy::none_t,
                unsigned const q, unsigned const p,
                const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
                const value_t *b, size_t const*const nb,                        size_t const*const pib,
                      value_t *c, size_t const*const nc, size_t const*const wc )
{

    set_omp_nested();
    auto is_cm = pib[0] == 1;
    if(!is_case<8>(p,q,pia)){
        set_blas_threads_max();
        assert(get_blas_threads() > 1u || get_blas_threads() <= cores);    
        if(is_cm)
            mtm_cm(q, p,  a, na, pia, b, nb, c, nc );
        else
            mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
    }
    else {
        auto const qh = inverse_mode(pia, pia+p, q);

        using namespace std::placeholders;

        auto n1     = na[pia[0]-1];
        auto m      = nc[q-1];
        auto nq     = na[q-1];
        auto wq     = wa[q-1];
        auto gemm_col = std::bind(gemm_col_tr2::run<value_t>,_1,_2,_3,   n1,m,nq,   wq, m,wq); // a,b,c
        auto gemm_row = std::bind(gemm_row::    run<value_t>,_2,_1,_3,   m,n1,nq,   nq,wq,wq); // b,a,c
        
#pragma omp parallel num_threads(cores) 
        {
            set_blas_threads_min();
            assert(get_blas_threads() == 1u);        
#pragma omp single
            {
                  if(is_cm) tasks_over_gemm_with_slices(gemm_col, p, qh,  a,na,wa,pia,   b,  c,nc,wc);
                  else      tasks_over_gemm_with_slices(gemm_row, p, qh,  a,na,wa,pia,   b,  c,nc,wc);
            }
        }
    }
}


template<class value_t, class size_t>
inline void ttm(
            parallel_policy::parallel_blas_t, slicing_policy::slice_t, fusion_policy::all_t,
            unsigned const q, unsigned const p,
            const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
            const value_t *b, size_t const*const nb, size_t const*const pib,
            value_t *c, size_t const*const nc, size_t const*const wc)
{
    set_blas_threads_max();
    assert(get_blas_threads() > 1 || get_blas_threads() <= cores);
        
    const auto is_cm = pib[0] == 1;

    if(!is_case<8>(p,q,pia)){    
        if(is_cm)
            mtm_cm(q, p,  a, na, pia, b, nb, c, nc );
        else
            mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
    }
    else {
    
        assert(is_case<8>(p,q,pia));
        assert(p>2);
        assert(q>0);

        auto const qh = inverse_mode(pia, pia+p, q);

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
        
        //std::cout << "m=" << m << ", n=" << n1 << ", k=" << nq << std::endl;

        using namespace std::placeholders;
        auto gemm_col = std::bind(gemm_col_tr2::run<value_t>,_1,_2,_3,   n1,m,nq,   wq, m,wq); // a,b,c
        auto gemm_row = std::bind(gemm_row::    run<value_t>,_2,_1,_3,   m,n1,nq,   nq,wq,wq); // b,a,c
        
        //std::cout << "Get blas threads: " << get_blas_threads() << std::endl;

        for(size_t k = 0u; k < outer; ++k){
            for(size_t j = 0u; j < inner; ++j){
                auto aa = a+k*wao + j*wai;
                auto cc = c+k*wco + j*wci;

                if(is_cm) gemm_col(aa, b, cc);
                else      gemm_row(aa, b, cc);
            }
        }
    }
}



// only parallelize the outer dimensions
template<class value_t, class size_t>
inline void ttm(
        parallel_policy::parallel_loop_t, slicing_policy::slice_t, fusion_policy::outer_t,
        unsigned const q, unsigned const p,
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b, size_t const*const nb, size_t const*const pib,
              value_t *c, size_t const*const nc, size_t const*const wc
        )
{
    set_omp_nested(); 
    auto is_cm = pib[0] == 1;

    if(!is_case<8>(p,q,pia)){  
        set_blas_threads_max();
        assert(get_blas_threads() > 1 || get_blas_threads() <= cores);
        if(is_cm)
            mtm_cm(q, p,  a, na, pia, b, nb, c, nc );
        else
            mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
	}
	else {

        assert(is_case<8>(p,q,pia));
        assert(p>2);
        assert(q>0);

        auto const qh = inverse_mode(pia, pia+p, q);

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

        auto gemm_col = std::bind(gemm_col_tr2::run<value_t>,_1,_2,_3,   n1,m,nq,   wq,m,wq); // a,b,c
        auto gemm_row = std::bind(gemm_row::    run<value_t>,_2,_1,_3,   m,n1,nq,   nq,wq,wq); // b,a,c

        #pragma omp parallel for schedule(static) num_threads(cores) proc_bind(spread)
        for(size_t k = 0u; k < num; ++k){
            auto aa = a+k*waq;
            auto cc = c+k*wcq;          

            set_blas_threads_min();
            assert(get_blas_threads()==1);

            if(is_cm) loops_over_gemm_with_slices(gemm_col, qh, qh,  aa,na,wa,pia,   b,  cc,nc,wc);
            else      loops_over_gemm_with_slices(gemm_row, qh, qh,  aa,na,wa,pia,   b,  cc,nc,wc);

        }        
    }
}



template<class value_t, class size_t>
inline void ttm(
        parallel_policy::parallel_loop_t, slicing_policy::slice_t, fusion_policy::all_t,
        unsigned const q, unsigned const p,
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b, size_t const*const nb, size_t const*const pib,
              value_t *c, size_t const*const nc, size_t const*const wc
        )
{
    set_omp_nested();
    auto is_cm = pib[0] == 1;

    if(!is_case<8>(p,q,pia)){    
        set_blas_threads_max();
        assert(get_blas_threads() > 1 || get_blas_threads() <= cores);

        if(is_cm)
            mtm_cm(q, p,  a, na, pia, b, nb, c, nc );
        else
            mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
    }
    else {
    
        assert(is_case<8>(p,q,pia));
        assert(p>2);
        assert(q>0);

        auto const qh = inverse_mode(pia, pia+p, q);

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
        auto gemm_col = std::bind(gemm_col_tr2::run<value_t>,_1,_2,_3,   n1,m,nq,   wq, m,wq); // a,b,c
        auto gemm_row = std::bind(gemm_row::    run<value_t>,_2,_1,_3,   m,n1,nq,   nq,wq,wq); // b,a,c

        #pragma omp parallel for schedule(static) collapse(2) num_threads(cores) proc_bind(spread)
        for(size_t k = 0u; k < outer; ++k){
            for(size_t j = 0u; j < inner; ++j){
                auto aa = a+k*wao + j*wai;
                auto cc = c+k*wco + j*wci;
                
                set_blas_threads(1);
                
                assert(get_blas_threads()==1);
                assert(get_omp_threads ()==cores);

                if(is_cm) gemm_col(aa, b, cc);
                else      gemm_row(aa, b, cc);
            }
        }
    }
}





// only parallelize the outer dimensions
template<class value_t, class size_t>
inline void ttm(
        parallel_policy::parallel_loop_blas_t, slicing_policy::slice_t, fusion_policy::all_t,
        unsigned const q, unsigned const p,
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b, size_t const*const nb, size_t const*const pib,
              value_t *c, size_t const*const nc, size_t const*const wc
        )
{
    set_omp_nested();
    auto is_cm = pib[0] == 1;
    if(!is_case<8>(p,q,pia)){
        set_blas_threads_max();
        assert(get_blas_threads() > 1 || get_blas_threads() <= cores);

        if(is_cm)
            mtm_cm(q, p,  a, na, pia, b, nb, c, nc );
        else
            mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
    }
    else {
        assert(is_case<8>(p,q,pia));
        assert(p>2);
        assert(q>0);

        auto const qh = inverse_mode(pia, pia+p, q);

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
        auto gemm_col = std::bind(gemm_col_tr2::run<value_t>,_1,_2,_3,   n1,m,nq,   wq, m,wq); // a,b,c
        auto gemm_row = std::bind(gemm_row::    run<value_t>,_2,_1,_3,   m,n1,nq,   nq,wq,wq); // b,a,c

        #pragma omp parallel for schedule(static) collapse(2) num_threads(cores) proc_bind(spread)
        for(size_t k = 0u; k < outer; ++k){
            for(size_t j = 0u; j < inner; ++j){
                auto aa = a+k*wao + j*wai;
                auto cc = c+k*wco + j*wci;

                set_blas_threads_max();
                assert(get_blas_threads() > 1 || get_blas_threads() <= cores);

                if(is_cm) gemm_col(aa, b, cc);
                else      gemm_row(aa, b, cc);
            }
        }
    }
}

// only parallelize the outer dimensions
template<class value_t, class size_t>
inline void ttm(
        parallel_policy::parallel_loop_blas_t, slicing_policy::slice_t, fusion_policy::all_t,
        unsigned const q, unsigned const p, double ratio,
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b, size_t const*const nb, size_t const*const pib,
              value_t *c, size_t const*const nc, size_t const*const wc
        )
{
    set_omp_nested();
    auto is_cm = pib[0] == 1;
    if(!is_case<8>(p,q,pia)){
        set_blas_threads_max();
        assert(get_blas_threads() > 1 || get_blas_threads() <= cores);

        if(is_cm)
            mtm_cm(q, p,  a, na, pia, b, nb, c, nc );
        else
            mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
    }
    else {
        assert(is_case<8>(p,q,pia));
        assert(p>2);
        assert(q>0);

        auto const qh = inverse_mode(pia, pia+p, q);

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
        auto gemm_col = std::bind(gemm_col_tr2::run<value_t>,_1,_2,_3,   n1,m,nq,   wq, m,wq); // a,b,c
        auto gemm_row = std::bind(gemm_row::    run<value_t>,_2,_1,_3,   m,n1,nq,   nq,wq,wq); // b,a,c
        
        
        const auto ompthreads = unsigned (double(cores)*ratio);
        const auto blasthreads = unsigned (double(cores)*(1.0-ratio));        

        #pragma omp parallel for schedule(static) collapse(2) num_threads(ompthreads) proc_bind(spread)
        for(size_t k = 0u; k < outer; ++k){
            for(size_t j = 0u; j < inner; ++j){
                auto aa = a+k*wao + j*wai;
                auto cc = c+k*wco + j*wci;

                set_blas_threads(blasthreads);
                assert(get_blas_threads() > 1 || get_blas_threads() <= cores);

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
        auto const qh  = inverse_mode(pia, pia+p, q);
        auto const nnq = product(na, pia, 1, qh);
        auto const m   = nc[q-1];
        auto const nq  = na[q-1];


        using namespace std::placeholders;
        auto gemm_col = std::bind(gemm_col_tr2::run<value_t>,_1,_2,_3,   nnq,m,nq,   nnq, m,nnq); // a,b,c
        auto gemm_row = std::bind(gemm_row::    run<value_t>,_2,_1,_3,   m,nnq,nq,   nq,nnq,nnq); // b,a,c

        if(is_cm) loops_over_gemm_with_subtensors(gemm_col, p, qh,  a,na,wa,pia,   b,  c,nc,wc);
        else      loops_over_gemm_with_subtensors(gemm_row, p, qh,  a,na,wa,pia,   b,  c,nc,wc);

    }
}



template<class value_t, class size_t>
inline void ttm(
            parallel_policy::parallel_blas_t, slicing_policy::subtensor_t, fusion_policy::none_t,
            unsigned const q, unsigned const p,
            const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
            const value_t *b, size_t const*const nb, size_t const*const pib,
                  value_t *c, size_t const*const nc, size_t const*const wc
            )
{
    set_blas_threads_max();
    assert(get_blas_threads() > 1 || get_blas_threads() <= cores);
    

    auto is_cm = pib[0] == 1;
    if(!is_case<8>(p,q,pia)){
        if(is_cm)
            mtm_cm(q, p,  a, na, pia, b, nb, c, nc );
        else
            mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
    }
    else {
        auto const qh  = inverse_mode(pia, pia+p, q);
        auto const nnq = product(na, pia, 1, qh);
        auto const m   = nc[q-1];
        auto const nq  = na[q-1];


        using namespace std::placeholders;
        auto gemm_col = std::bind(gemm_col_tr2::run<value_t>,_1,_2,_3,   nnq,m,nq,   nnq, m,nnq); // a,b,c
        auto gemm_row = std::bind(gemm_row::    run<value_t>,_2,_1,_3,   m,nnq,nq,   nq,nnq,nnq); // b,a,c

        if(is_cm) loops_over_gemm_with_subtensors(gemm_col, p, qh,  a,na,wa,pia,   b,  c,nc,wc);
        else      loops_over_gemm_with_subtensors(gemm_row, p, qh,  a,na,wa,pia,   b,  c,nc,wc);

    }
}




template<class value_t, class size_t>
inline void ttm(
            parallel_policy::parallel_taskloop_t, slicing_policy::subtensor_t, fusion_policy::none_t,
            unsigned const q, unsigned const p,
            const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
            const value_t *b, size_t const*const nb, size_t const*const pib,
                  value_t *c, size_t const*const nc, size_t const*const wc
            )
{
  
    set_omp_nested();
    auto is_cm = pib[0] == 1;
    if(!is_case<8>(p,q,pia)){
        set_blas_threads_max();
        assert(get_blas_threads() > 1 || get_blas_threads() <= cores);    
        if(is_cm)
            mtm_cm(q, p,  a, na, pia, b, nb, c, nc );
        else
            mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
    }
    else {
        auto const qh  = inverse_mode(pia, pia+p, q);
        auto const nnq = product(na, pia, 1, qh);
        auto const m   = nc[q-1];
        auto const nq  = na[q-1];


        using namespace std::placeholders;
        auto gemm_col = std::bind(gemm_col_tr2::run<value_t>,_1,_2,_3,   nnq,m,nq,   nnq, m,nnq); // a,b,c
        auto gemm_row = std::bind(gemm_row::    run<value_t>,_2,_1,_3,   m,nnq,nq,   nq,nnq,nnq); // b,a,c

#pragma omp parallel num_threads(cores) 
        {
            set_blas_threads_min();
            assert(get_blas_threads() == 1);        
#pragma omp single
            {
                if(is_cm) taskloops_over_gemm_with_subtensors(gemm_col, p, qh,  a,na,wa,pia,   b,  c,nc,wc);
                else      taskloops_over_gemm_with_subtensors(gemm_row, p, qh,  a,na,wa,pia,   b,  c,nc,wc);
            }
        }
    }
}


template<class value_t, class size_t>
inline void ttm(
            parallel_policy::parallel_task_t, slicing_policy::subtensor_t, fusion_policy::none_t,
            unsigned const q, unsigned const p,
            const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
            const value_t *b, size_t const*const nb, size_t const*const pib,
                  value_t *c, size_t const*const nc, size_t const*const wc
            )
{
  
    set_omp_nested();
    auto is_cm = pib[0] == 1;
    if(!is_case<8>(p,q,pia)){
        set_blas_threads_max();
        assert(get_blas_threads() > 1 || get_blas_threads() <= cores);    
        if(is_cm)
            mtm_cm(q, p,  a, na, pia, b, nb, c, nc );
        else
            mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
    }
    else {
        auto const qh  = inverse_mode(pia, pia+p, q);
        auto const nnq = product(na, pia, 1, qh);
        auto const m   = nc[q-1];
        auto const nq  = na[q-1];

        using namespace std::placeholders;
        auto gemm_col = std::bind(gemm_col_tr2::run<value_t>,_1,_2,_3,   nnq,m,nq,   nnq, m,nnq); // a,b,c
        auto gemm_row = std::bind(gemm_row::    run<value_t>,_2,_1,_3,   m,nnq,nq,   nq,nnq,nnq); // b,a,c

#pragma omp parallel num_threads(cores) 
        {
            set_blas_threads_min();
            assert(get_blas_threads() == 1);        
#pragma omp single
            {
                if(is_cm) tasks_over_gemm_with_subtensors(gemm_col, p, qh,  a,na,wa,pia,   b,  c,nc,wc);
                else      tasks_over_gemm_with_subtensors(gemm_row, p, qh,  a,na,wa,pia,   b,  c,nc,wc);
            }
        }
    }
}



template<class value_t, class size_t>
inline void ttm(
        parallel_policy::parallel_blas_t, slicing_policy::subtensor_t, fusion_policy::all_t,
        unsigned const q, unsigned const p,
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b, size_t const*const nb, size_t const*const pib,
              value_t *c, size_t const*const nc, size_t const*const wc
        )
{
    auto is_cm = pib[0] == 1;

    if(!is_case<8>(p,q,pia)){
        set_blas_threads_max();
        assert(get_blas_threads() > 1 || get_blas_threads() <= cores);
        if(is_cm)
            mtm_cm(q, p,  a, na, pia, b, nb, c, nc );
        else
            mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
    }
    else {
    
        assert(is_case<8>(p,q,pia));
        assert(q>0);
        assert(p>2);

        auto const qh = inverse_mode(pia, pia+p, q);

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
        auto gemm_col = std::bind(gemm_col_tr2::run<value_t>,_1,_2,_3,   nnq,m,nq,   nnq, m,nnq); // a,b,c
        auto gemm_row = std::bind(gemm_row::    run<value_t>,_2,_1,_3,   m,nnq,nq,   nq,nnq,nnq); // b,a,c
        
        for(size_t k = 0u; k < num; ++k){
            auto aa = a+k*waq;
            auto cc = c+k*wcq;
            
            if(is_cm) gemm_col(aa, b, cc);
            else      gemm_row(aa, b, cc);

        }      
    }
}

template<class value_t, class size_t>
inline void ttm(
        parallel_policy::parallel_loop_t, slicing_policy::subtensor_t, fusion_policy::all_t,
        unsigned const q, unsigned const p,
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b, size_t const*const nb, size_t const*const pib,
              value_t *c, size_t const*const nc, size_t const*const wc
        )
{
    set_omp_nested();
    auto is_cm = pib[0] == 1;

    if(!is_case<8>(p,q,pia)){
        set_blas_threads_max();
        assert(get_blas_threads() > 1 || get_blas_threads() <= cores);
        if(is_cm)
            mtm_cm(q, p,  a, na, pia, b, nb, c, nc );
        else
            mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
    }
    else {
    
        assert(is_case<8>(p,q,pia));
        assert(q>0);
        assert(p>2);

        auto const qh = inverse_mode(pia, pia+p, q);

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
        auto gemm_col = std::bind(gemm_col_tr2::run<value_t>,_1,_2,_3,   nnq,m,nq,   nnq, m,nnq); // a,b,c
        auto gemm_row = std::bind(gemm_row::    run<value_t>,_2,_1,_3,   m,nnq,nq,   nq,nnq,nnq); // b,a,c
        
        #pragma omp parallel for schedule(static) num_threads(cores) proc_bind(spread)
        for(size_t k = 0u; k < num; ++k){
            auto aa = a+k*waq;
            auto cc = c+k*wcq;

            set_blas_threads_min();
            assert(get_blas_threads()==1);
            assert(get_omp_threads()==cores);
            
            if(is_cm) gemm_col(aa, b, cc);
            else      gemm_row(aa, b, cc);

        }      
    }
}




template<class value_t, class size_t>
inline void ttm(
        parallel_policy::parallel_loop_blas_t, slicing_policy::subtensor_t, fusion_policy::all_t,
        unsigned const q, unsigned const p,
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b, size_t const*const nb, size_t const*const pib,
              value_t *c, size_t const*const nc, size_t const*const wc
        )
{
    set_omp_nested();
    auto is_cm = pib[0] == 1;

    if(!is_case<8>(p,q,pia)){
        set_blas_threads_max();
        assert(get_blas_threads() > 1 || get_blas_threads() <= cores);
        if(is_cm)
            mtm_cm(q, p,  a, na, pia, b, nb, c, nc );
        else
            mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
    }
    else {
        assert(is_case<8>(p,q,pia));
        assert(q>0);
        assert(p>2);

        auto const qh = inverse_mode(pia, pia+p, q);

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
        auto gemm_col = std::bind(gemm_col_tr2::run<value_t>,_1,_2,_3,   nnq,m,nq,   nnq, m,nnq); // a,b,c
        auto gemm_row = std::bind(gemm_row::    run<value_t>,_2,_1,_3,   m,nnq,nq,   nq,nnq,nnq); // b,a,c

        #pragma omp parallel for schedule(dynamic) num_threads(cores) proc_bind(spread)
        for(size_t k = 0u; k < num; ++k){
            auto aa = a+k*waq;
            auto cc = c+k*wcq;

            set_blas_threads_max();
            assert(get_blas_threads() > 1 || get_blas_threads() <= cores);
            
            if(is_cm) gemm_col(aa, b, cc);
            else      gemm_row(aa, b, cc);
        }
    }
}


template<class value_t, class size_t>
inline void ttm(
        parallel_policy::parallel_loop_blas_t, slicing_policy::subtensor_t, fusion_policy::all_t,
        unsigned const q, unsigned const p, double ratio,
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b, size_t const*const nb, size_t const*const pib,
              value_t *c, size_t const*const nc, size_t const*const wc
        )
{
    set_omp_nested();
    auto is_cm = pib[0] == 1;

    if(!is_case<8>(p,q,pia)){
        set_blas_threads_max();
        assert(get_blas_threads() > 1 || get_blas_threads() <= cores);
        if(is_cm)
            mtm_cm(q, p,  a, na, pia, b, nb, c, nc );
        else
            mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
    }
    else {
        assert(is_case<8>(p,q,pia));
        assert(q>0);
        assert(p>2);

        auto const qh = inverse_mode(pia, pia+p, q);

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
        auto gemm_col = std::bind(gemm_col_tr2::run<value_t>,_1,_2,_3,   nnq,m,nq,   nnq, m,nnq); // a,b,c
        auto gemm_row = std::bind(gemm_row::    run<value_t>,_2,_1,_3,   m,nnq,nq,   nq,nnq,nnq); // b,a,c
        
        const auto ompthreads = unsigned (double(cores)*ratio);
        const auto blasthreads = unsigned (double(cores)*(1.0-ratio));

        #pragma omp parallel for schedule(static) num_threads(ompthreads) proc_bind(spread)
        for(size_t k = 0u; k < num; ++k){
            auto aa = a+k*waq;
            auto cc = c+k*wcq;

            set_blas_threads(blasthreads);
            assert(get_blas_threads() > 1 || get_blas_threads() <= cores);

            if(is_cm) gemm_col(aa, b, cc);
            else      gemm_row(aa, b, cc);
        }
    }
}



template<class value_t, class size_t>
inline void ttm(
        parallel_policy::batched_gemm_t, slicing_policy::subtensor_t, fusion_policy::all_t,
        unsigned const q, unsigned const p,
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b, size_t const*const nb, size_t const*const pib,
        value_t *c, size_t const*const nc, size_t const*const wc
        )
{
    auto is_cm = pib[0] == 1;

    set_blas_threads_max();
    assert(get_blas_threads() > 1 || get_blas_threads() <= cores);
    
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
#ifdef USE_MKL
        auto const qh = inverse_mode(pia, pia+p, q);

        // num = n[pi[qh+1]] * n[pi[qh+2]] * ... * n[pi[p]]
        auto pp = product(na, pia, qh+1,p+1);

        // w[pi[q]]  = 1 * n[pi[1]] * n[pi[2]] * ... * n[pi[qh-1]]
        auto const waq = wa[pia[qh]-1];
        auto const wcq = wc[pia[qh]-1];

        // num = n[pi[1]] * n[pi[2]] * ... * n[pi[qh-1]] with pi[qh] = q
        auto const nnq = product(na, pia, 1, qh);

        auto m      = nc[q-1];
        auto nq     = na[q-1];


        using index_t = MKL_INT;
        using vector  = std::vector<value_t>;
        using ivector = std::vector<index_t>;
        using vvector = std::vector<value_t*>;

        const auto L =  is_cm ? CblasColMajor : CblasRowMajor;
        const auto Ta = std::vector<CBLAS_TRANSPOSE>(pp,CblasNoTrans);
        const auto Tb = std::vector<CBLAS_TRANSPOSE>(pp,is_cm ? CblasTrans : CblasNoTrans);

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




template<class value_t, class size_t>
inline void ttm(
        parallel_policy::combined_t, slicing_policy::combined_t, fusion_policy::all_t,
        unsigned const q, unsigned const p,
        const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
        const value_t *b, size_t const*const nb, size_t const*const pib,
        value_t *c, size_t const*const nc, size_t const*const wc
        )
{

    
    if(!is_case<8>(p,q,pia)){
        auto is_cm = pib[0] == 1;
        set_blas_threads_max();
        if(is_cm)
            mtm_cm(q, p,  a, na, pia, b, nb, c, nc );
        else
            mtm_rm(q, p,  a, na, pia, b, nb, c, nc );
            
        return;
    }
    

//=> compute `par-loop-count` with `par-loop` with `qD` slices
//=> IF `par-loop-count` > `num-procs` THEN use `par-loop` with `qD` slices.
//=> ELSE compute `par-loop-count` with `par-loop` with `2D` slices 
//    => IF `par-loop-count` > `num-procs` THEN use `par-loop` with `2D` slices
//    => ELSE use `par-gemm` with `qD` slices (`par-gemm` with `2D` slices does perform for non-symmetric and symmetric tensor shapes)
  
    
    
    assert(is_case<8>(p,q,pia));
    assert(q>0);
    assert(p>2);
    
    auto const qh = inverse_mode(pia, pia+p, q);

    // inner = n[pi[2]] * ... * n[pi[qh-1]] with pi[qh] = q
    auto const inner = product(na, pia, 2, qh);    
      
    // outer = n[pi[qh+1]] * n[pi[qh+2]] * ... * n[pi[p]]
    auto const outer = product(na, pia, qh+1,p+1);
        
    if( outer >= cores){
        ttm(parallel_policy::parallel_loop, slicing_policy::subtensor, fusion_policy::all,
            q, p,
            a, na, wa, pia,
            b, nb,     pib,
            c, nc, wc );
    }
    else {
      if(inner*outer >= cores) {
          ttm(parallel_policy::parallel_loop, slicing_policy::slice, fusion_policy::all,
              q, p,
              a, na, wa, pia,
              b, nb,     pib,
              c, nc, wc );        
      }
      else{
          ttm(parallel_policy::parallel_blas, slicing_policy::subtensor, fusion_policy::none,
              q, p,
              a, na, wa, pia,
              b, nb,     pib,
              c, nc, wc );
      }
    }
}


} // namespace tlib::ttm::detail
