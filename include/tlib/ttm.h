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

#include "detail/ttm.h"
#include "detail/tensor.h"
#include "detail/tags.h"

namespace tlib::ttm
{
		

/**
 * \brief Implements a mode-q tensor-times-matrix multiplication
 *
 *  C( [k1, j, k2] ) =  A ( [k1,iq,k2] ) * B (j,iq) with k1=(i{1},...,i{q-1}), k2=(i{q+1},...,i{p})
 *
 * @tparam value_t           type of the elements which is usually an floating point type, i.e. float, double or int
 * @tparam size_t size       type of the extents, strides and layout elements which is usually std::size_t
 * @tparam execution_policy  type of the execution policy which can be tlib::execution::seq, tlib::execution::par or tlib::execution::blas.
 * @tparam slicing_policy    type of the slicing policy which can be tlib::slicing::small or tlib::slicing::large
 * @tparam fusion_policy     type of the loop fusion policy which can be tlib::loop_fusion::none, tlib::loop_fusion::outer or tlib::loop_fusion::all
 *
 * \param q  mode of the contraction with 1 <= q <= p
 * \param p  rank of the array A with p > 0.
 * \param a  pointer to the array A.
 * \param na extents of the array A. Length of the tuple must be p.
 * \param wa strides of the array A. Length of the tuple must be p.
 * \param pia permutations of the indices of array A. Length of the tuple must be p.
 * \param b  pointer to the vector b.
 * \param nb extents of the vector b. Length of the tuple must be 1.
 * \param wb strides of the vector b. Length of the tuple must be 1.
 * \param pib permutations of the indices of array B. Length of the tuple must be 1.
 * \param c  pointer to the array C.
 * \param nc extents of the array C. Length of the tuple must be p-1.
 * \param wc strides of the array C. Length of the tuple must be p-1.
 *
*/
template<class value_t, class size_t, class parallel_policy, class slicing_policy, class fusion_policy>
inline void ttm(
    parallel_policy ep, slicing_policy sp, fusion_policy fp,
    unsigned const q, unsigned const p,
    const value_t *a, size_t const*const na, size_t const*const wa, size_t const*const pia,
    const value_t *b, size_t const*const nb ,                       size_t const*const pib,
    value_t       *c, size_t const*const nc, size_t const*const wc
	)
{
    using namespace tlib::ttm;

    if(p==0)        throw std::runtime_error("Error in tlib::tensor_times_matrix: input tensor order should be greater zero.");
    if(q==0 || q>p) throw std::runtime_error("Error in tlib::tensor_times_matrix: contraction mode should be greater zero or less than or equal to p.");
    if(a==nullptr) 	throw std::runtime_error("Error in tlib::tensor_times_matrix: pointer to input tensor A should not be zero.");
    if(b==nullptr) 	throw std::runtime_error("Error in tlib::tensor_times_matrix: pointer to input vector B should not be zero.");
    if(c==nullptr) 	throw std::runtime_error("Error in tlib::tensor_times_matrix: pointer to output tensor C should not be zero.");

    if(na==nullptr) throw std::runtime_error("Error in tlib::tensor_times_matrix: pointer to input tensor shape vector na should not be zero.");
    if(nb==nullptr) throw std::runtime_error("Error in tlib::tensor_times_matrix: pointer to input vector shape vector nb should not be zero.");
    if(nc==nullptr) throw std::runtime_error("Error in tlib::tensor_times_matrix: pointer to output tensor shape vector nc should not be zero.");

    if(wa==nullptr) throw std::runtime_error("Error in tlib::tensor_times_matrix: pointer to input tensor stride vector wa should not be zero.");
    if(wc==nullptr) throw std::runtime_error("Error in tlib::tensor_times_matrix: pointer to output tensor stride vector wc should not be zero.");

    if(pia==nullptr) throw std::runtime_error("Error in tlib::tensor_times_matrix: pointer to input tensor permutation vector pia should not be zero.");

    if(na[q-1] != nb[1]) throw std::runtime_error("Error in tlib::tensor_times_matrix: contraction dimension of A and B are not equal.");
    if(nc[q-1] != nb[0]) throw std::runtime_error("Error in tlib::tensor_times_matrix: free dimension of C and B are not equal.");

	
    if(!detail::is_valid_shape(na,na+p     )) throw std::runtime_error("Error in tlib::tensor_times_matrix: shape vector of A is not valid.");
    if(!detail::is_valid_shape(nc,nc+p     )) throw std::runtime_error("Error in tlib::tensor_times_matrix: shape vector of C is not valid.");

    if(!detail::is_valid_layout(pia,pia+p  )) throw std::runtime_error("Error in tlib::tensor_times_matrix: layout vector of A is not valid.");

    if(!detail::is_valid_strides(pia,pia+p, wa)) throw std::runtime_error("Error in tlib::tensor_times_matrix: stride vector of A is not valid.");
    if(!detail::is_valid_strides(pia,pia+p, wc)) throw std::runtime_error("Error in tlib::tensor_times_matrix: stride vector of C is not valid.");

    detail::ttm(ep,sp,fp, q,p,   a,na,wa,pia,   b,nb,pib,  c,nc,wc);
}


/**
 * \brief Implements a mode-q tensor-times-matrix multiplication
 *
 */
template<class value_t, class execution_policy, class slicing_policy, class fusion_policy>
inline auto ttm(std::size_t q,
                ttm::tensor<value_t> const& a,
                ttm::tensor<value_t> const& b,
                execution_policy ep, slicing_policy sp, fusion_policy fp)
{
    auto const p = a.order();

    if(p==0)        throw std::runtime_error("Error in tlib::tensor_times_matrix: input tensor order should be greater zero.");
    if(q==0 || q>p) throw std::runtime_error("Error in tlib::tensor_times_matrix: contraction mode should be greater zero or less than or equal to p.");

    auto nc  = a.shape();
    auto nb  = b.shape();

    nc[q-1]  = nb.at(0);

    auto c   = tensor<value_t>(nc,a.layout());

    ttm( ep, sp, fp,
         q, p,
         a.data().data(), a.shape().data(), a.strides().data(), a.layout().data(),
         b.data().data(), b.shape().data(),                     b.layout().data(),
         c.data().data(), c.shape().data(), c.strides().data());
		
	return c;
}

}

/**
 * \brief Implements a mode-q tensor-times-matrix multiplication
 *
 */
template<class value_t>
inline auto operator*(tlib::ttm::tensor_view<value_t> const& a,  tlib::ttm::tensor<value_t> const& b)
{
    return ttm(a.contraction_mode(), a.get_tensor(),  b,
               tlib::ttm::parallel_policy::combined, tlib::ttm::slicing_policy::combined, tlib::ttm::fusion_policy::all) ;
}
