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

namespace tlib::parallel_policy
{
struct sequential_t     {}; // multithreaded gemm
struct threaded_gemm_t  {}; // multithreaded gemm
struct omp_taskloop_t   {}; // omp_taskloops with single threaded gemm
struct omp_forloop_t    {}; // omp_for with single threaded gemm
struct omp_forloop_and_threaded_gemm_t  {}; // omp_for with multi-threaded gemm
struct batched_gemm_t   {}; // multithreaded batched gemm with collapsed loops
struct depends_t        {};

inline constexpr sequential_t     sequential;
inline constexpr threaded_gemm_t  threaded_gemm;
inline constexpr omp_taskloop_t   omp_taskloop;
inline constexpr omp_forloop_t    omp_forloop;
inline constexpr batched_gemm_t   batched_gemm;
inline constexpr depends_t        depends;

}


namespace tlib::slicing_policy
{
struct slice_t     {};
struct subtensor_t {};
struct depends_t      {};


inline constexpr depends_t   depends;
inline constexpr slice_t     slice;
inline constexpr subtensor_t subtensor;

}


namespace tlib::fusion_policy
{
struct depends_t {};
struct none_t    {};
struct outer_t   {};
struct all_t     {};

inline constexpr depends_t depends;
inline constexpr none_t    none;
inline constexpr outer_t   outer;
inline constexpr all_t     all;
}


