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

// ttm

namespace tlib::ttm::parallel_policy
{

struct sequential_t          {}; // sequential loops and sequential gemm
struct parallel_blas_t       {}; // multithreaded gemm
struct parallel_loop_t       {}; // omp_for with single threaded gemm
struct parallel_taskloop_t   {}; // omp_task for each loop with single threaded gemm 
struct parallel_loop_blas_t  {}; // omp_for with multi-threaded gemm
struct batched_gemm_t        {}; // multithreaded batched gemm with collapsed loops
struct combined_t            {};

inline constexpr sequential_t         sequential;
inline constexpr parallel_blas_t      parallel_blas;
inline constexpr parallel_loop_t      parallel_loop;
inline constexpr parallel_taskloop_t  parallel_taskloop;
inline constexpr parallel_loop_blas_t parallel_loop_blas;
inline constexpr batched_gemm_t       batched_gemm;
inline constexpr combined_t           combined;

}


namespace tlib::ttm::slicing_policy
{
struct slice_t     {};
struct subtensor_t {};
struct combined_t  {};


inline constexpr combined_t   combined;
inline constexpr slice_t      slice;
inline constexpr subtensor_t  subtensor;

}


namespace tlib::ttm::fusion_policy
{
struct none_t     {};
struct outer_t    {};
struct all_t      {};

inline constexpr none_t     none;
inline constexpr outer_t    outer;
inline constexpr all_t      all;
}


