#pragma once

#include <cstddef>
#include <stdexcept>
#include <cassert>
#include <numeric>
#include <algorithm>
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
class MatrixTimesMatrix
{
public:
    template<class value_t, class size_t>
    static inline void run(value_t *const A, value_t *const B, value_t * C,
                           size_t m,      size_t n,      size_t k, 
                           size_t lda,    size_t ldb,    size_t ldc)
                           { impl(m,n,k, A,lda, B,ldb, C,ldc); }
private:
    static constexpr inline auto layout = layout_t::value;
    static constexpr inline auto transA = transA_t::value;
    static constexpr inline auto transB = transB_t::value;

    template<class value_t, class size_t>
    static inline void impl(size_t m, size_t n, size_t k, 
                            value_t *const A, size_t lda,
                            value_t *const B, size_t ldb,
                            value_t * C, size_t ldc)
    {
        auto alpha = value_t(1.0);
        auto beta  = value_t(0.0);
        if constexpr (std::is_same_v<value_t,float>)
            cblas_sgemm(layout, transA, transB, m,n,k, alpha, A,lda, B,ldb, beta, C, ldc);
        else
            cblas_dgemm(layout, transA, transB, m,n,k, alpha, A,lda, B,ldb, beta, C, ldc);
    }
};


using mtm_row_tr1 = MatrixTimesMatrix < cblas_row, cblas_tr,   cblas_notr >;
using mtm_row_tr2 = MatrixTimesMatrix < cblas_row, cblas_notr, cblas_tr   >;
using mtm_row     = MatrixTimesMatrix < cblas_row, cblas_notr, cblas_notr >;

using mtm_col_tr1 = MatrixTimesMatrix < cblas_col, cblas_tr  , cblas_notr >;
using mtm_col_tr2 = MatrixTimesMatrix < cblas_col, cblas_notr, cblas_tr   >;
using mtm_col     = MatrixTimesMatrix < cblas_col, cblas_notr, cblas_notr >;



template<class layout_t>
class MatrixTimesVector
{
public:
    template<class value_t, class size_t >
    static inline void run(value_t *const A, value_t *const x, value_t * y, size_t m, size_t n, size_t lda){ impl(m,n, A,lda, x, y); }
private:
    static constexpr inline auto layout = layout_t::value;

    template<class value_t, class size_t>
    static inline void impl(size_t m, size_t n, value_t *const A, size_t lda, value_t *const x, value_t * y)
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

using mtv_row = MatrixTimesVector <cblas_row>;
using mtv_col = MatrixTimesVector <cblas_col>;


template<class size_t>
inline auto num_elements(size_t const*const na, size_t p)
{
	return std::accumulate( na, na+p, 1ul, std::multiplies<>()  );
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

template<class value_t, class size_t>
inline void mtm_rm(
			size_t const q, size_t const p,
			value_t *const a, size_t const*const na, size_t const*const pia,
			value_t *const b, size_t const*const nb, // is a row-major dense matrix
			value_t * c )
{


	
	// B is always 
	
	auto n1 = na[0];
	auto n2 = na[1];
	auto nq = na[q-1];
	auto mq = nb[q-1];
  auto n  = num_elements(na,p) / nq;
	
	                                               // A,x,y, m,  n,  lda
	     if(is_case_rm<1>(p,q,pia)) mtv_row::run     (b,a,c, mq, n1, n1  );
                                                 // a,b,c  m,  n,  k,  lda,ldb,ldc    	     
	else if(is_case_rm<2>(p,q,pia)) mtm_row_tr1::run (a,b,c, n2, mq, n1,  n1, mq, mq); // q=1 | A,C = CM | B = RM
	else if(is_case_rm<3>(p,q,pia)) mtm_row::run     (b,a,c, mq, n1, n2,  n2, n1, n1); // q=2 | A,C = CM | B = RM
	
	else if(is_case_rm<4>(p,q,pia)) mtm_row::run     (b,a,c, mq, n2, n1,  n1, n2, n2); // q=1 | A,C = RM | B = RM
	else if(is_case_rm<5>(p,q,pia)) mtm_row_tr2::run (a,b,c, n1, mq, n2,  n2, mq, mq); // q=2 | A,C = RM | B = RM
	
	else if(is_case_rm<6>(p,q,pia)) mtm_row_tr2::run (a,b,c,  n, nq, nq,  nq, mq, mq); // q=pi(1) | A,C = RM | B = RM
	else if(is_case_rm<7>(p,q,pia)) mtm_row::run     (b,a,c, mq,  n, nq,  nq,  n,  n); // q=pi(p) | A,C = RM | B = RM
	
}



} // namespace tlib::detail
