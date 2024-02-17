/*
 *   Copyright (C) 2019 Cem Bassoy (cem.bassoy@gmail.com)
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


#include <vector>
#include <numeric>
#include <exception>
#include <cassert>
#include <iostream>
#include <sstream>

namespace tlib::gtest
{



template<class value_t = double>
class matrix
{
public:
    using value_type = value_t;
    using size_type = std::size_t;
    using container_type = std::vector<value_type>;
    using shape_type = std::vector<size_type>;
    using layout_type = std::vector<size_type>;

    matrix() = delete;
    matrix(shape_type n, value_type v = 0.0, layout_type pi = {1,2})
        : container_(prod(n),v)
        , n_(n)
        , pi_(pi)
        , w_(strides(n,pi))
    {
    }

    matrix(shape_type n, container_type const& c, layout_type const& pi)
        : container_(c)
        , n_(n)
        , pi_(pi)
        , w_(strides(n,pi))
    {
    }

    matrix(matrix const& other) : container_(other.container_), n_(other.n_), pi_(other.pi_), w_(other.w_) {}

    virtual ~matrix() = default;

    inline const value_type* data() const { return this->container_.data(); }
    inline value_type* data() { return this->container_.data(); }
    inline container_type const&  container() const { return this->container_; }
    inline container_type &  container() { return this->container_; }
    inline shape_type  const& n()  const { return this->n_; }
    inline shape_type  const& w()  const { return this->w_; }
    inline layout_type const& pi() const { return this->pi_; }
    inline unsigned  p() const { return this->n_.size(); }

    void set(layout_type pi)
    {
        pi_ = pi;
        w_ = strides(n_,pi_);
    }

    inline bool is_cm()  const { return pi_[0] == 1; }


    inline value_type      & operator()(size_type i, size_type j)       { return container_[at(i,j)]; }
    inline value_type const& operator()(size_type i, size_type j) const { return container_[at(i,j)]; }

    inline value_type      & operator[](size_type j)       { return container_[j]; }
    inline value_type const& operator[](size_type j) const { return container_[j]; }


protected:
    container_type container_;
    shape_type n_;
    layout_type pi_;
    shape_type w_;


    inline auto at(size_type i, size_type j) const{
        return i*w_[0] + j*w_[1];
    }

    static inline auto prod(shape_type n){
        return std::accumulate(n.begin(), n.end(), 1ull, std::multiplies<size_type>());
    }

    static inline auto strides(shape_type n, layout_type pi){
        unsigned p = n.size();
        auto w = shape_type(p,1);
        for(auto r = 1u; r < p; ++r)
            w[pi[r]-1] = w[pi[r-1]-1] * n[pi[r-1]-1];

        return w;
    }
};

template<class value_t = double>
class cube : public matrix<value_t>
{
    using super_type = matrix<value_t>;

public:
    using value_type     = typename super_type::value_type;
    using size_type      = typename super_type::size_type;
    using container_type = typename super_type::container_type;
    using shape_type     = typename super_type::shape_type;
    using layout_type    = typename super_type::layout_type;

    cube() = delete;
    cube(shape_type n, value_type v = 0.0, layout_type pi = {1,2,3})
        : super_type(n,v,pi)
    {
        assert(n.size() == 3u);
        assert(pi.size() == 3u);
    }

    cube(shape_type n,  container_type const& c, layout_type const& pi)
        : super_type(n,c,pi)
    {
        assert(n.size() == 3u);
        assert(pi.size() == 3u);
    }

    cube(cube const& other) : super_type(other.super_type) {}

    virtual ~cube() = default;

    inline value_type      & operator()(size_type i, size_type j, size_type k)       { return this->container_[at(i,j,k)]; }
    inline value_type const& operator()(size_type i, size_type j, size_type k) const { return this->container_[at(i,j,k)]; }

    inline value_type      & operator[](size_type j)       { return this->container_[j]; }
    inline value_type const& operator[](size_type j) const { return this->container_[j]; }

protected:
    inline auto at(size_type i, size_type j, size_type k) const{
        return i*this->w_[0] + j*this->w_[1] +  k * this->w_[2];
    }
};




template<class value_t = double>
class tensor : public matrix<value_t>
{
    using super_type = matrix<value_t>;

public:
    using value_type     = typename super_type::value_type;
    using size_type      = typename super_type::size_type;
    using container_type = typename super_type::container_type;
    using shape_type     = typename super_type::shape_type;
    using layout_type    = typename super_type::layout_type;

    tensor() = delete;
    tensor(shape_type n, layout_type const& pi)
        : super_type(n,value_type(0),pi)
    {
    }

    tensor(shape_type n,  container_type const& c, layout_type const& pi)
        : super_type(n,c,pi)
    {
    }

    tensor(tensor const& other) : super_type(other.super_type) {}

    virtual ~tensor() = default;

//    inline value_type      & operator()(size_type i, size_type j, size_type k)       { return this->container_[at(i,j,k)]; }
//    inline value_type const& operator()(size_type i, size_type j, size_type k) const { return this->container_[at(i,j,k)]; }

    inline value_type      & operator[](size_type j)       { return this->container_[j]; }
    inline value_type const& operator[](size_type j) const { return this->container_[j]; }

protected:
//    inline auto at(size_type i, size_type j, size_type k) const{
//        return i*this->w_[0] + j*this->w_[1] +  k * this->w_[2];
//    }
};


template<class value_type>
void init( matrix<value_type>& a, unsigned q )
{

    auto m = a.n()[0];
    auto n = a.n()[1];

    assert(q == 1 || q == 2);

    if (q == 2){
        for(auto j = 0ul; j < n; ++j)
            for(auto i = 0ul; i < m; ++i)
                a(i,j) = j+1 + i*n;
    }
    else{
        for(auto i = 0ul; i < m; ++i)
            for(auto j = 0ul; j < n; ++j)
                a(i,j) = i+1 + j*m;
    }
}

template<class value_type>
void init( cube<value_type>& a, unsigned q )
{

    auto M = a.n()[0];
    auto N = a.n()[1];
    auto K = a.n()[2];

    assert(q == 1 || q == 2 || q == 3);

    auto element = [q,M,N,K](auto i, auto j, auto k)
    {
             if(q == 1u) return i+1 + j*M + k*M*N;
        else if(q == 2u) return j+1 + i*N + k*M*N;
        else if(q == 3u) return k+1 + j*M + i*N*K;

        return 0ull;
    };

    for(auto k = 0ull; k < K; ++k)
        for(auto j = 0ull; j < N; ++j)
            for(auto i = 0ull; i < M; ++i)
                a(i,j,k) = element(i,j,k);
}



template<class size_type, unsigned rank>
inline auto generate_shapes_help(
	std::vector<std::vector<size_type>>& shapes,
	std::vector<size_type> const& start,
	std::vector<size_type> shape,
	std::vector<size_type> const& dims)
{
	if constexpr ( rank > 0 ){
		for(auto j = size_type{0u}, c = start.at(rank); j < dims.at(rank); ++j, c*=2u){
			shape.at(rank) = c;
			generate_shapes_help<size_type,rank-1>(shapes, start, shape, dims);
		}
	}
	else
	{
		for(auto j = size_type{0u}, c = start.at(rank); j < dims.at(rank); ++j, c*=2u){
			shape.at(rank) = c;
			shapes.push_back(shape);
		}
	}
}

template<class size_type, unsigned rank>
inline auto generate_shapes(std::vector<size_type> const& start, std::vector<size_type> const& dims)
{
	std::vector<std::vector<size_type>> shapes;
	static_assert (rank!=0,"Static Error in fhg::gtest_transpose: Rank cannot be zero.");
	std::vector<size_type> shape(rank);
	if(start.size() != rank)
		throw std::runtime_error("Error in fhg::gtest_transpose: start shape must have length Rank.");
	if(dims.size() != rank)
		throw std::runtime_error("Error in fhg::gtest_transpose: dims must have length Rank.");

	generate_shapes_help<size_type,rank-1>(shapes, start, shape, dims);
	return shapes;
}

template<class size_type, unsigned rank>
inline auto generate_permutations()
{
	auto f = size_type{1u};
	for(auto i = unsigned{2u}; i <= rank; ++i)
		f*=i;
	std::vector<std::vector<size_type>> layouts ( f );
	std::vector<size_type> current(rank);
	std::iota(current.begin(), current.end(), size_type{1u});
	for(auto i = size_type{0u}; i < f; ++i){
		layouts.at(i) = current;
		std::next_permutation(current.begin(), current.end());
	}
	return layouts;
}

} // namespace tlib::gtest





template<class value_type>
std::ostream& operator<< (std::ostream& out, tlib::gtest::matrix<value_type> const& a)
{

    auto m = a.n()[0];
    auto n = a.n()[1];

    out << "[ ... " << std::endl;
    for(auto i = 0ul; i < m; ++i){
        for(auto j = 0ul; j < n; ++j){
            out << a(i,j) << ", ";
        }
        out << "..." << std::endl;
    }
    out << "];" << std::endl;
    return out;
}


template<class value_type>
std::ostream& operator<< (std::ostream& out, tlib::gtest::cube<value_type> const& a)
{

    auto M = a.n()[0];
    auto N = a.n()[1];
    auto K = a.n()[2];

    out << "cat(3,..." << std::endl;
    for(auto k = 0ull; k < K; ++k){
        out << "[ ... " << std::endl;
        for(auto i = 0ull; i < M; ++i){
            for(auto j = 0ull; j < N; ++j){
                out << a(i,j,k) << ", ";
            }
            out << "..." << std::endl;
        }
        out << "],..." << std::endl;
    }
    out << ");" << std::endl;
    return out;
}


template<class value_type>
void stream_out(std::ostringstream& out, tlib::gtest::tensor<value_type> const& a, std::size_t j, unsigned r)
{
//    if(r == a.p()){
//        out << "cat(" << r << ",..." << std::endl;
//        for(auto ir = 0ull; ir < a.n().at(r-1); ++ir, j += a.w().at(r-1)){
//            stream_out(out, a, j, r-1);
//        }
//        out << ");" << std::endl;
//    }
    if(r >= 3){
        out << "cat("<< r << ",..." << std::endl;
        for(auto ir = 0ull; ir < a.n().at(r-1); ++ir, j += a.w().at(r-1)){
            stream_out(out, a, j, r-1);
        }
        out << "), ..." << std::endl;
    }
    else if (r == 2){
        out << "[ ... " << std::endl;
        for(auto i1 = 0ull; i1 < a.n().at(0); ++i1, j += a.w().at(0)){
            auto j2 = 0ull;
            for(auto i2 = 0ull; i2 < a.n().at(1)-1; ++i2, j2 += a.w().at(1)){
                out << a[j+j2] << ", ";
            }
            out << a[j+j2] << "; ..." << std::endl;
        }
        out << "],..." << std::endl;
    }
}


template<class value_type>
std::ostream& operator<< (std::ostream& out, tlib::gtest::tensor<value_type> const& a)
{
    auto j = 0ull;
    if(a.p() == 1u){
        out << "[ " << std::endl;
        for(auto i1 = 0ull; i1 < a.n().at(0); ++i1, j += a.w().at(0)){
            out << a[j] << ", ";
        }
        out << "];" << std::endl;
    }
    else if (a.p() == 2u){
        out << "[ ... " << std::endl;
        for(auto i2 = 0ull; i2 < a.n().at(1); ++i2, j += a.w().at(1)){
            for(auto i1 = 0ull; i1 < a.n().at(0); ++i1, j += a.w().at(0)){
                out << a[j] << ", ";
            }
            out << "..." << std::endl;
        }
        out << "];" << std::endl;
    }
    else{
        std::ostringstream sout;
        stream_out(sout, a, j, a.p());
        out << sout.str();
    }
    return out;
}



