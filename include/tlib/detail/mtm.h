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
#include <iostream>


#include "tags.h"
#include "cases.h"


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




namespace tlib::detail {

struct cblas_layout {};

struct cblas_row : cblas_layout { 
#ifdef USE_BLIS
    static inline constexpr CBLAS_ORDER value = CblasRowMajor; 
#else
    static inline constexpr CBLAS_LAYOUT value = CblasRowMajor; 
#endif
   };
struct cblas_col : cblas_layout { 
#ifdef USE_BLIS
    static inline constexpr CBLAS_ORDER value = CblasColMajor; 
#else
    static inline constexpr CBLAS_LAYOUT value = CblasColMajor; 
#endif
};

struct cblas_trans { 
    static CBLAS_TRANSPOSE value; 
};

struct cblas_tr   : public cblas_trans { 
    static inline constexpr CBLAS_TRANSPOSE value = CblasTrans; 
};
struct cblas_notr : public cblas_trans { 
    static inline constexpr CBLAS_TRANSPOSE value = CblasNoTrans; 
};




template<class layout_t, class transA_t, class transB_t>
class gemm_blas
{
public:   
    template<class value_t>
    static inline void run(const value_t* A, const value_t*B, value_t * C,
                           std::size_t const m,      std::size_t const n,      std::size_t const k, 
                           std::size_t const lda,    std::size_t const ldb,    std::size_t const ldc)
    {
        auto alpha = value_t(1.0);
        auto beta  = value_t(0.0);

        if constexpr (std::is_same_v<value_t,float>)
            cblas_sgemm(layout, transA, transB, m,n,k, alpha, A,lda, B,ldb, beta, C, ldc);
        else
            cblas_dgemm(layout, transA, transB, m,n,k, alpha, A,lda, B,ldb, beta, C, ldc);            
    }
private:
    static constexpr inline auto layout = layout_t::value;
    static constexpr inline auto transA = transA_t::value;
    static constexpr inline auto transB = transB_t::value;

};


using gemm_row_tr1 = gemm_blas < cblas_row, cblas_tr,   cblas_notr >;
using gemm_row_tr2 = gemm_blas < cblas_row, cblas_notr, cblas_tr   >;
using gemm_row     = gemm_blas < cblas_row, cblas_notr, cblas_notr >;

using gemm_col_tr1 = gemm_blas < cblas_col, cblas_tr  , cblas_notr >;
using gemm_col_tr2 = gemm_blas < cblas_col, cblas_notr, cblas_tr   >;
using gemm_col     = gemm_blas < cblas_col, cblas_notr, cblas_notr >;



template<class layout_t>
class gemv_blas
{
private:
    static constexpr inline auto layout = layout_t::value;

public:
    template<class value_t>
    static inline void run(const value_t *A, const value_t *x, value_t* y, std::size_t m, std::size_t n, std::size_t lda)
    {
        // CblasColMajor CblasNoTrans      m         n     alpha  a   lda   x  incx  beta  y   incy
        auto noTrA = cblas_notr::value;
        auto alpha = value_t(1.0);
        auto beta  = value_t(0.0);
        auto incx  = 1;
        auto incy  = 1;
        if constexpr (std::is_same_v<value_t,float>)
            cblas_sgemv(layout, noTrA, m,n, alpha, A,lda, x,incx, beta, y,incy);
        else
            cblas_dgemv(layout, noTrA, m,n, alpha, A,lda, x,incx, beta, y,incy);
    }
};

using gemv_row = gemv_blas <cblas_row>;
using gemv_col = gemv_blas <cblas_col>;


// B is a row-major matrix
template<class value_t>
inline void mtm_rm(unsigned const q, unsigned const p,
                   const value_t *a, std::size_t const*const na, std::size_t const*const pia,
                   const value_t *b, std::size_t const*const nb, 
                   value_t *c,       std::size_t const*const nc )
{

	
    assert(q>0);
    assert(p>0);	
    assert(!is_case<8>(p,q,pia));

    auto m  = na[0];
    auto n  = na[1];
    auto nq = na[q-1];
    auto u  = nb[0];

    assert(nc[q-1] == u);
    assert(nb[1]  == nq);

    assert(q==0 || std::equal(na,     na+q-1, nc    ));
    assert(q==p || std::equal(na+q+1, na+p,   nc+q+1));

    auto nn = std::accumulate( na, na+p, 1ull, std::multiplies<>() ) / nq;
 
	                                                  // A,x,y, m, n, lda
         if(is_case<1>(p,q,pia)) gemv_row::run     (b,a,c, u, m, m  );            // q=1     | A(u,1),C(m,1), B(m,u) = RM       | C = A x1 B => c = B *(rm) a
                                                    // a,b,c  m, n, k,   lda,ldb,ldc
    else if(is_case<2>(p,q,pia)) gemm_row_tr2::run (a,b,c, n, u, m,   m, m, u );  // q=1     | A(m,n),C(u,n) = CM , B(u,m)  = RM | C = A x1 B => C = A *(rm) B'
    else if(is_case<3>(p,q,pia)) gemm_row::run     (b,a,c, u, m, n,   n, m, m );  // q=2     | A(m,n),C(m,u) = CM , B(u,n)  = RM | C = A x2 B => C = B *(rm) A
	
    else if(is_case<4>(p,q,pia)) gemm_row::run     (b,a,c, u, n, m,   m, n, n );  // q=1     | A(m,n),C(u,n) = RM , B(u,m)  = RM | C = A x1 B => C = B *(rm) A
    else if(is_case<5>(p,q,pia)) gemm_row_tr2::run (a,b,c, m, u, n,   n, n, u );  // q=2     | A(m,n),C(m,u) = RM , B(u,n)  = RM | C = A x2 B => C = A *(rm) B'
	
    else if(is_case<6>(p,q,pia)) gemm_row_tr2::run (a,b,c, nn,u,nq,   nq,nq, u);  // q=pi(1) | A(nn,nq),C(nn,u)   , B(u,nq) = RM | C = A xq B => C = A *(rm) B'
    else if(is_case<7>(p,q,pia)) gemm_row::run     (b,a,c, u,nn,nq,   nq,nn,nn);  // q=pi(p) | A(nq,nn),C(u,nn)   , B(u,nq) = RM | C = A xq B => C = B *(rm) A
	
}  


// B is a column-major matrix
template<class value_t>
inline void mtm_cm(unsigned const q, unsigned const p,
                   const value_t *a, std::size_t const*const na, std::size_t const*const pia,
                   const value_t *b, std::size_t const*const nb, 
                   value_t *c,       std::size_t const*const nc )
{

	
    assert(q>0);
    assert(p>0);	
    assert(!is_case<8>(p,q,pia));

    auto m  = na[0];
    auto n  = na[1];
    auto nq = na[q-1];
    auto u  = nb[0];

    assert(nc[q-1] == u);
    assert(nb[1]  == nq);

    assert(q==0 || std::equal(na,     na+q-1, nc    ));
    assert(q==p || std::equal(na+q+1, na+p,   nc+q+1));

    auto nn = std::accumulate( na, na+p, 1ull, std::multiplies<>() ) / nq;
 
    //                                                 A,x,y, m, n, lda
         if(is_case<1>(p,q,pia)) gemv_col::run     (b,a,c, u, m, u  );            // q=1     | A(u,1),C(m,1), B(m,u) = CM        | C = A x1 B => c = B *(cm) a
                                                    // a,b,c  m, n, k,   lda,ldb,ldc
    else if(is_case<2>(p,q,pia)) gemm_col::run     (b,a,c, u, n, m,   u, m, u );  // q=1     | A(m,n),C(u,n) = CM , B(u,m) = CM  | C = A x1 B => C = B *(cm) A
    else if(is_case<3>(p,q,pia)) gemm_col_tr2::run (a,b,c, m, u, n,   m, u, m );  // q=2     | A(m,n),C(m,u) = CM , B(u,n) = CM  | C = A x2 B => C = A *(cm) B'
	
    else if(is_case<4>(p,q,pia)) gemm_col_tr2::run (a,b,c, n, u, m,   n, u, n );  // q=1     | A(m,n),C(u,n) = RM , B(u,m) = CM  | C = A x1 B => C = A *(cm) B'
    else if(is_case<5>(p,q,pia)) gemm_col::run     (b,a,c, u, m, n,   u, n, u );  // q=2     | A(m,n),C(m,u) = RM , B(u,n) = CM  | C = A x2 B => C = B *(cm) A
	
    else if(is_case<6>(p,q,pia)) gemm_col::run     (b,a,c, u,nn,nq,    u,nq, u);  // q=pi(1) | A(nn,nq),C(nn,u)   , B(u,nq) = CM | C = A xq B => C = B *(cm) A
    else if(is_case<7>(p,q,pia)) gemm_col_tr2::run (a,b,c, nn,u,nq,   nn, u,nn);  // q=pi(p) | A(nq,nn),C(u,nn)   , B(u,nq) = CM | C = A xq B => C = A *(cm) B'
	
}  


} // namespace tlib::detail
