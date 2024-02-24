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


namespace tlib::execution
{
struct sequential_policy    {};
struct parallel_policy      {};
struct parallel_blas_policy {};

inline constexpr sequential_policy    seq;
inline constexpr parallel_policy      par;
inline constexpr parallel_blas_policy blas;
}

namespace tlib::slicing
{
struct small_policy    {};
struct large_policy    {};

inline constexpr small_policy    small;
inline constexpr large_policy    large;
}



namespace tlib::loop_fusion
{
struct none_policy   {};
struct outer_policy  {};
struct all_policy    {};

inline constexpr none_policy    none;
inline constexpr outer_policy   outer;
inline constexpr all_policy     all;
}



// ttm

namespace tlib::parallel_policy
{
struct threaded_gemm_t  {}; // multithreaded gemm
struct omp_taskloop_t   {}; // omp_taskloops with single threaded gemm
struct omp_forloop_t    {}; // omp_for with single threaded gemm
struct omp_forloop_and_threaded_gemm_t  {}; // omp_for with multi-threaded gemm
struct batched_gemm_t   {}; // multithreaded batched gemm with collapsed loops

inline constexpr threaded_gemm_t  threaded_gemm;
inline constexpr omp_taskloop_t   omp_taskloop;
inline constexpr omp_forloop_t    omp_forloop;
inline constexpr batched_gemm_t   batched_gemm;
}


namespace tlib::slicing_policy
{
struct slice_t     {};
struct subtensor_t {};

inline constexpr slice_t     slice;
inline constexpr subtensor_t subtensor;

}


namespace tlib::fusion_policy
{
struct none_t   {};
struct outer_t  {};
struct all_t    {};

inline constexpr none_t    none;
inline constexpr outer_t   outer;
inline constexpr all_t     all;
}


