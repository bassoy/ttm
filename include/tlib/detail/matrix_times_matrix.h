#pragma once

#include <cstddef>
#include <stdexcept>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <iostream>

#include <omp.h>



#include "tags.h"
#include "cases_ttm.h"


// <cblas.h>
#ifdef USE_OPENBLAS
#include <cblas.h>
#endif

#ifdef USE_INTELBLAS
#include <mkl.h>
#endif


namespace tlib::detail {

/** \brief computes 2d-slice-times-matrix
 *
 * the same as above only using basic linear algebra subroutines
 *
 * \note performs this with blas library
 *
*/

//enum CBLAS_LAYOUT { CblasRowMajor=101,CblasColMajor=102};
//enum CBLAS_TRANSPOSE { CblasNoTrans=111, CblasTrans=112};


struct cblas_layout {};

struct cblas_row : cblas_layout { static inline constexpr CBLAS_LAYOUT value = CblasRowMajor; };
struct cblas_col : cblas_layout { static inline constexpr CBLAS_LAYOUT value = CblasColMajor; };

struct cblas_trans { static CBLAS_TRANSPOSE value; };

struct cblas_tr   : public cblas_trans { static inline constexpr CBLAS_TRANSPOSE value = CblasTrans; };
struct cblas_notr : public cblas_trans { static inline constexpr CBLAS_TRANSPOSE value = CblasNoTrans; };




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
        
// Layout, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, m, n, k, alpha, double *const a, lda, double *const b, ldb,  beta, double *c, ldc);        
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


inline std::size_t num_elements(std::size_t const*const na, unsigned p)
{
	return std::accumulate( na, na+p, 1ull, std::multiplies<>()  );
}


#if 0
template<class value_t>
inline void mtm(
			size_t const m, size_t const p,
			value_t const*const a, size_t const*const na,     size_t const*const /*wa*/, size_t const*const pia,
			value_t const*const b, size_t const*const /*nb*/,
			value_t      *const c, size_t const*const /*nc*/, size_t const*const /*wc*/, size_t const*const /*pic*/
			);
#endif 

template<class value_t>
inline void mtm_rm(
			unsigned const q, unsigned const p,
            const value_t *a, std::size_t const*const na, std::size_t const*const pia,
            const value_t *b, std::size_t const*const nb, // is a row-major dense matrix
                  value_t *c, std::size_t const*const nc )
{

	
	assert(q>0);
	assert(p>0);	
	assert(!is_case_rm<8>(p,q,pia));
	
	auto m  = na[0];
	auto n  = na[1];
	auto nq = na[q-1];
	auto u  = nb[0];

	assert(nc[q-1] == u);
	assert(nb[1]  == nq);
	
	//std::cout << "na0 = " << na[0] << std::endl;
	//std::cout << "na1 = " << na[1] << std::endl;
	//std::cout << "nc0 = " << nc[0] << std::endl;
	//std::cout << "nc1 = " << nc[1] << std::endl;
	//std::cout << "q = " << q << std::endl;
	
	assert(q==0 || std::equal(na,     na+q-1, nc    ));
	assert(q==p || std::equal(na+q+1, na+p,   nc+q+1));
	
  auto nn  = num_elements(na,p) / nq;
 
	                                               // A,x,y, m, n, lda
         if(is_case_rm<1>(p,q,pia)) gemv_row::run     (b,a,c, u, m, m  );            // q=1 | A(u,1),C(m,1), B(m,u) = RM       | C = A x1 B => c = B *(rm) a
                                                 // a,b,c  m, n, k,   lda,ldb,ldc    	     
    else if(is_case_rm<2>(p,q,pia)) gemm_row_tr2::run (a,b,c, n, u, m,   m, m, u );  // q=1     | A(m,n),C(u,n) = CM , B(u,m) = RM | C = A x1 B => C = A *(rm) B'
    else if(is_case_rm<3>(p,q,pia)) gemm_row::run     (b,a,c, u, m, n,   n, m, m );  // q=2     | A(m,n),C(m,u) = CM , B(u,n) = RM | C = A x2 B => C = B *(rm) A
	
    else if(is_case_rm<4>(p,q,pia)) gemm_row::run     (b,a,c, u, n, m,   m, n, n );  // q=1     | A(m,n),C(u,n) = RM , B(u,m) = RM | C = A x1 B => C = B *(rm) A
    else if(is_case_rm<5>(p,q,pia)) gemm_row_tr2::run (a,b,c, m, u, n,   n, n, u );  // q=2     | A(m,n),C(m,u) = RM , B(u,n) = RM | C = A x2 B => C = A *(rm) B'
	
    else if(is_case_rm<6>(p,q,pia)) gemm_row_tr2::run (a,b,c, nn,u,nq,   nq,nq, u);  // q=pi(1) | A(nn,nq),C(nn,u)   , B(u,nq) = RM | C = A xq B => C = A *(rm) B'
    else if(is_case_rm<7>(p,q,pia)) gemm_row::run     (b,a,c, u,nn,nq,   nq,nn,nn);  // q=pi(p) | A(nq,nn),C(u,nn)   , B(u,nq) = RM | C = A xq B => C = B *(rm) A
	
}  



} // namespace tlib::detail
