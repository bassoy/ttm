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

#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <functional>
#include <ostream>

#include <tlib/ttv.h>
#include "gtest_aux.h" 



template<class value_type = double>
class matrix
{
public:
  using size_type = std::size_t;
  using container_type = std::vector<value_type>;
  using shape_type = std::vector<size_type>;
  using layout_type = std::vector<unsigned>;

  matrix() = delete;
  matrix(shape_type n, value_type v = 0.0, layout_type pi = {1,2}) 
    : container_(elements_(n),v)
    , n_(n)
    , pi_(pi)
    , w_(strides_(n,pi)) 
    {
    }
    
  matrix(matrix const& other) : container_(other.container_), n_(other.n_), pi_(other.pi_), w_(other.w_) {}
  
  virtual ~matrix() = default;

  inline const value_type* data() const { return this->container_.data(); }
  inline value_type* data() { return this->container_.data(); }
  inline container_type const&  container() const { return this->container_; }
  inline container_type &  container() { return this->container_; }
  inline shape_type n()    const { return this->n_; }
  inline shape_type w()    const { return w_; }
  inline layout_type pi()   const { return pi_; }
  inline unsigned  p()    const { return n_.size(); }
  
  void set(layout_type pi)
  {
    pi_ = pi;
    w_ = strides_(n_,pi_);
  }
  
  inline bool is_cm()  const { return pi_[0] == 1; }
  

  inline value_type      & operator()(size_type i, size_type j)       { return container_[at_(i,j)]; }
  inline value_type const& operator()(size_type i, size_type j) const { return container_[at_(i,j)]; }
  
  inline value_type      & operator[](size_type j)       { return container_[j]; }
  inline value_type const& operator[](size_type j) const { return container_[j]; }
  

protected:
  container_type container_;
  shape_type n_;
  layout_type pi_;
  shape_type w_;

 
  inline auto at_(size_type i, size_type j) const{
    return i*w_[0] + j*w_[1];
  }

  static inline auto elements_(shape_type n){
    return std::accumulate(n.begin(), n.end(), 1ull, std::multiplies<size_type>());
  }
  
  static inline auto strides_(shape_type n, layout_type pi){    
    unsigned p = n.size();
    auto w = shape_type(p,1);
    for(auto r = 1u; r < p; ++r)
      w[pi[r]-1] = w[pi[r-1]-1] * n[pi[r-1]-1];
    
    return w;
  }
};

template<class value_type = double>
class cube : public matrix<value_type>
{
  using super_type = matrix<value_type>;

  using size_type      = typename super_type::size_type;
  using container_type = typename super_type::container_type;
  using shape_type     = typename super_type::shape_type;
  using layout_type    = typename super_type::layout_type;
  
public:
  cube() = delete;
  cube(shape_type n, value_type v = 0.0, layout_type pi = {1,2,3}) 
    : super_type(n,v,pi)
    {
      assert(n.size() == 3u);
      assert(pi.size() == 3u);
    }
    
  cube(cube const& other) : super_type(other.super_type) {}
  
  virtual ~cube() = default;  
  
  inline value_type      & operator()(size_type i, size_type j, size_type k)       { return this->container_[at_(i,j,k)]; }
  inline value_type const& operator()(size_type i, size_type j, size_type k) const { return this->container_[at_(i,j,k)]; }
  
  inline value_type      & operator[](size_type j)       { return this->container_[j]; }
  inline value_type const& operator[](size_type j) const { return this->container_[j]; }

protected:
  inline auto at_(size_type i, size_type j, size_type k) const{
    return i*this->w_[0] + j*this->w_[1] +  k * this->w_[2];
  }   
  
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


template<class value_type>
std::ostream& operator<< (std::ostream& out, matrix<value_type> const& a)
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
std::ostream& operator<< (std::ostream& out, cube<value_type> const& a)
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

template<class matrix_type>
[[nodiscard]] matrix_type mtm(matrix_type  const& a, matrix_type  const& b)
{
  
  auto M = a.n()[0];
  auto K = a.n()[1];
  auto N = b.n()[1];
  assert(K == b.n()[0]);
  
  auto c = matrix_type ({M,N});  
  auto cmajor = a.is_cm();
  
  
  auto ar = std::cref(a);
  auto br = std::cref(b);
  auto cr = std::ref(c);
  

  
  if (cmajor) 
  {
#pragma omp parallel for firstprivate(M,N,K, ar,br,cr)
    for(auto j = 0ul; j < N; ++j){
      for(auto k = 0ul; k < K; ++k){
        auto bb = br(k,j);  
        for(auto i = 0ul; i < M; ++i){    
          cr(i,j) += ar(i,k) * bb;
      	}
      }
    }
  }
  else // row
  {
#pragma omp parallel for firstprivate(M,N,K, ar,br,cr)
    for(auto i = 0ul; i < M; ++i){
      for(auto k = 0ul; k < K; ++k){
        auto aa = a(i,k);
      	for(auto j = 0ul; j < N; ++j){
      	  c(i,j) += aa * b(k,j);
      	}
      }
    }
  }
  return c;	
}


template<class matrix_type>
[[nodiscard]] matrix_type mtv(matrix_type  const& a, matrix_type  const& b)
{
  auto M = a.n()[0];
  auto N = a.n()[1];
  auto c = matrix_type ({M,1});  
  auto cmajor = a.is_cm();
  assert(b.n()[0] == N && b.n()[1] == 1);
  
  //auto ar = std::cref(a);
  //auto br = std::cref(b);
  //auto cr = std::ref(c);  
  
  if (cmajor) 
  {
//#pragma omp parallel for collapse(2) firstprivate(M,N, ar,br,cr)
    for(auto j = 0ul; j < N; ++j){
        for(auto i = 0ul; i < M; ++i){
          //cr.get()[i] += ar.get()(i,j) * br.get()[j];
          c[i] += a(i,j) * b[j];
      	}
      }
  }
  else // row
  {
//#pragma omp parallel for firstprivate(M,N, ar,br,cr)
    for(auto i = 0ul; i < M; ++i){
      auto cc = c[i];
//#pragma omp simd reduction(+:cc)
    	for(auto j = 0ul; j < N; ++j){
    	  cc += a(i,j) * b[j];
    	}
    	c[i] = cc;
    }
  }
  return c;	
}

template<class matrix_type>
[[nodiscard]] matrix_type vtm(matrix_type  const& a, matrix_type  const& b)
{
  auto M = a.n()[0];
  auto N = a.n()[1];
  auto c = matrix_type ({1,N});  
  auto cmajor = a.is_cm();
  
  assert(b.n()[0] == 1 && b.n()[1] == M);
  
  //auto ar = std::cref(a);
  //auto br = std::cref(b);
  //auto cr = std::ref(c);  
  
  if (cmajor) 
  {
    for(auto j = 0ul; j < N; ++j){
      auto cc = 0;
      for(auto i = 0ul; i < M; ++i){
        cc += b[i] * a(i,j);
    	}
    	c[j] = cc;
    }
  }
  else // row
  {
    for(auto i = 0ul; i < M; ++i){
      auto bb = b[i];
    	for(auto j = 0ul; j < N; ++j){
    	  c[j] += bb * a(i,j);
    	}
    }
  }
  return c;	
}



/* \brief creates a reference value
 * 
 * 
 * \param a input matrix 
 * \param i row number for q=1 and col number for q=2
 * \param q contraction mode
*/
template<class value_t>
value_t refc(matrix<value_t> const& a, std::size_t i, std::size_t q)
{
  auto M = a.n().at(0);
  auto N = a.n().at(1);
  
  auto sum = [](auto n) { return (n*(n+1))/2u; };
  
  assert(q == 1 || q == 2);
  
  if(q == 2) return sum(a(i,N-1)) - sum(a(i,0)-1.0);
  else       return sum(a(M-1,i)) - sum(a(0,i)-1.0);
};


/* \brief creates a reference value
 * 
 * 
 * \param a input matrix 
 * \param i row number for q=1 and col number for q=2
 * \param q contraction mode
*/
template<class value_t>
value_t refc(cube<value_t> const& a, std::size_t i, std::size_t j, std::size_t q)
{
  auto M = a.n().at(0);
  auto N = a.n().at(1);
  auto K = a.n().at(2);
  
  auto sum = [](auto n) { return (n*(n+1))/2u; };
  
  assert(q == 1 || q == 2 || q == 3);
  
  if(q == 3) return sum(a(i,j,K-1)) - sum(a(i,j,0)-1.0);
  if(q == 2) return sum(a(i,N-1,j)) - sum(a(i,0,j)-1.0);
  else       return sum(a(M-1,i,j)) - sum(a(0,i,j)-1.0);
};







TEST(MatrixTimesVector, Ref)
{
	using indices = std::vector<std::size_t>;
	using permuration = std::vector<unsigned>;
	
	auto start = indices(2u,2u);
	auto steps = indices(2u,5u);

	auto shapes   = tlib::gtest::generate_shapes<std::size_t,2u>(start,steps);	
	auto formats  = std::array<permuration,2>{permuration{1,2}, permuration{2,1} };

	for(auto const& n : shapes) 
	{ 
    auto M = n[0];
    auto N = n[1];
	
	  for(auto f : formats) 
	  {
	    {
	      auto q = 2;
	      
	      auto a = matrix({M,N}, 0.0, f);
        auto b = matrix({N,1}, 1.0, f);
        
	      init(a,q);

	      auto c = mtv(a,b);

	      // std::cout << "q = " << q << std::endl;
	      // std::cout << "a = " << a << std::endl;
	      // std::cout << "b = " << b << std::endl;	 	      
	      // std::cout << "c = " << c << std::endl;

	      for(auto i = 0ul; i < M; ++i)
      		EXPECT_FLOAT_EQ(c[i], refc(a,i,q) );
      }     
      
	    {
	      auto q = 1;
	      
	      auto a = matrix({M,N}, 0.0, f);
        auto b = matrix({1,M}, 1.0, f);
        
	      init(a,q);

	      auto c = vtm(a,b);

	      for(auto j = 0ul; j < N; ++j)
      		EXPECT_FLOAT_EQ(c[j], refc(a,j,q) );
      }
      
    }
	}
}


  
TEST(MatrixTimesMatrix, Ref)
{
	using indices = std::vector<std::size_t>;
	using permuration = std::vector<unsigned>;
	
	auto start = indices(2u,2u);
	auto steps = indices(2u,5u);

	auto shapes  = tlib::gtest::generate_shapes<std::size_t,2u>(start,steps);
	auto formats = std::array<permuration,2>{permuration{1,2}, permuration{2,1} };
	
	for(auto const& n : shapes) 
	{ 
	    auto M = n[0];
      auto N = n[1];
      auto K = n[1]*2;
	
	  for(auto f : formats) 
	  {
	    auto q = 2;

      auto a = matrix({M,K}, 0.0, f);
      auto b = matrix({K,N}, 1.0, f);
		  
		  init(a,q);

		  auto c = mtm(a,b);
		  
      // std::cout << "q = " << q << std::endl;
      // std::cout << "a = " << a << std::endl;
      // std::cout << "b = " << b << std::endl;	 	      
      // std::cout << "c = " << c << std::endl;		  

		  for(auto i = 0u; i < M; ++i){
    		for(auto j = 0u; j < N; ++j){
      		EXPECT_FLOAT_EQ(c(i,j), refc(a,i,q) );
        }
      }
    }
	}
}

// q=1 | A(n,1),C(m,1), B(m,n) = RM       | C = A x1 B => c = B *(rm) a
TEST(MatrixTimesMatrix, Case1)
{
	using indices = std::vector<std::size_t>;
	using permuration = std::vector<unsigned>;
	
	auto start = indices(2u,2u);
	auto steps = indices(2u,1u);

	auto shapes = tlib::gtest::generate_shapes<std::size_t,2u>(start,steps);
  auto cm = permuration{1,2};
  auto rm = permuration{2,1};
  	
	auto q = 1ul;
	auto p = 1ul;
	
	for(auto const& nb : shapes) 
	{
	  
	  auto n = nb[1];
	  auto m = nb[0];

    
    auto a = matrix({n,1}, 1.0, cm);
    auto b = matrix({m,n}, 1.0, rm);
    auto c = matrix({m,1}, 0.0, cm);
    
    const auto na = a.n();
    const auto nc = c.n();    
    const auto pia = a.pi();    
 
		init(b,2u);

		
    tlib::detail::mtm_rm(
		q,p,
		a.data(), na.data(), pia.data(),
		b.data(), nb.data(), 
		c.data(), nc.data());
		
	  for(auto i = 0ul; i < m; ++i)
  		EXPECT_FLOAT_EQ(c[i], refc(b,i,2u) );
	}
}


// q=1 | A(m,n),C(u,n) = CM | B(u,m) = RM | C = A x1 B => C = A *(rm) B' 
TEST(MatrixTimesMatrix, Case2)
{
	using indices     = std::vector<std::size_t>;
	using permutation = std::vector<unsigned>;
	
	auto start = indices(2u,2u);
	auto steps = indices(2u,5u); 
	
	constexpr auto shape_size = 2u;

	auto shapes = tlib::gtest::generate_shapes<std::size_t,shape_size>(start,steps);
  auto cm = permutation{1,2};
  auto rm = permutation{2,1};
  	
	auto q = 1ul;
	auto p = shape_size;
	
	for(auto const& na : shapes) 
	{
	  
	  auto m = na[0];
	  auto n = na[1];	  
	  auto u = m*2;
	  
	  ASSERT_TRUE(tlib::detail::is_case_rm<2>(p,q,cm.data()));

    
    auto a = matrix({m,n}, 1.0, cm); 
    auto b = matrix({u,m}, 1.0, rm); 
    auto c = matrix({u,n}, 0.0, cm);
    const auto nb = b.n();
    const auto nc = c.n();
    const auto pia = a.pi();
        
		init(a,q);

		
    tlib::detail::mtm_rm(
		q,p,
		a.data(), na.data(), pia.data(),
		b.data(), nb.data(), 
		c.data(), nc.data()); 
		
		// mtm(a,b)
		
		// std::cout << "a = " << a << std::endl;
		// std::cout << "b = " << b << std::endl;
		// std::cout << "c = " << c << std::endl;
		
	  for(auto i = 0u; i < u; ++i){
  		for(auto j = 0u; j < n; ++j){  		  
    		EXPECT_FLOAT_EQ(c(i,j), refc(a,j,1u) );
      }
    }
	}
}


// q=2 | A(m,n),C(m,u) = CM | B(u,n) = RM | C = A x2 B => C = B *(rm) A
TEST(MatrixTimesMatrix, Case3)
{
	using indices     = std::vector<std::size_t>;
	using permutation = std::vector<unsigned>;
	
	auto start = indices(2u,2u);
	auto steps = indices(2u,5u); 
	
	constexpr auto shape_size = 2u;

	auto shapes = tlib::gtest::generate_shapes<std::size_t,shape_size>(start,steps);
  auto cm = permutation{1,2};
  auto rm = permutation{2,1};
  	
	auto q = 2ul;
	auto p = shape_size;
	
	for(auto const& na : shapes) 
	{
	  
	  auto m = na[0];
	  auto n = na[1];	  
	  auto u = m*2;
	  
	  ASSERT_TRUE(tlib::detail::is_case_rm<3>(p,q,cm.data()));

    
    auto a = matrix({m,n}, 1.0, cm); 
    auto b = matrix({u,n}, 1.0, rm); 
    auto c = matrix({m,u}, 0.0, cm);
    const auto nb = b.n();
    const auto nc = c.n();
    const auto pia = a.pi();
        
		init(b,q); 

    tlib::detail::mtm_rm(
		q,p,
		a.data(), na.data(), pia.data(),
		b.data(), nb.data(), 
		c.data(), nc.data()); 
		
		// mtm(a,b)
		
		// std::cout << "a = " << a << std::endl;
		// std::cout << "b = " << b << std::endl;
		// std::cout << "c = " << c << std::endl;
		
	  for(auto i = 0u; i < m; ++i){
  	  for(auto j = 0u; j < u; ++j){
    		EXPECT_FLOAT_EQ(c(i,j), refc(b,j,q) );
      }
    }		
	}  
}



// q=1 | A(m,n),C(u,n) = RM | B(u,m) = RM | C = A x1 B => C = B *(rm) A
TEST(MatrixTimesMatrix, Case4)
{
	using indices     = std::vector<std::size_t>;
	using permutation = std::vector<unsigned>;
	
	auto start = indices(2u,2u);
	auto steps = indices(2u,4u); 
	
	constexpr auto shape_size = 2u;

	auto shapes = tlib::gtest::generate_shapes<std::size_t,shape_size>(start,steps);
  auto cm = permutation{1,2};
  auto rm = permutation{2,1};
  	
	auto q = 1ul;
	auto p = shape_size;
	
	for(auto const& na : shapes) 
	{
	  
	  auto m = na[0];
	  auto n = na[1];	  
	  auto u = m*2;
	  
	  ASSERT_TRUE(tlib::detail::is_case_rm<4>(p,q,rm.data()));

    
    auto a = matrix({m,n}, 1.0, rm); 
    auto b = matrix({u,m}, 1.0, rm); 
    auto c = matrix({u,n}, 0.0, rm);
    const auto nb = b.n();
    const auto nc = c.n();
    const auto pia = a.pi();
    
		init(b,2); 

    tlib::detail::mtm_rm(
		q,p,
		a.data(), na.data(), pia.data(),
		b.data(), nb.data(), 
		c.data(), nc.data()); 
		
		// mtm(a,b)
		
		// std::cout << "q = " << q << std::endl;
		// std::cout << "a = " << a << std::endl;
		// std::cout << "b = " << b << std::endl;
		// std::cout << "c = " << c << std::endl;
		
	  for(auto i = 0u; i < u; ++i){
  	  for(auto j = 0u; j < n; ++j){
    		EXPECT_FLOAT_EQ(c(i,j), refc(b,i,2) );
      }
    }		
	}  
}


// q=2 | A(m,n),C(m,u) = RM | B(u,n) = RM | C = A x2 B => C = A *(rm) B'
TEST(MatrixTimesMatrix, Case5)
{
	using indices     = std::vector<std::size_t>;
	using permutation = std::vector<unsigned>;
	
	auto start = indices(2u,2u);
	auto steps = indices(2u,5u); 
	
	constexpr auto shape_size = 2u;

	auto shapes = tlib::gtest::generate_shapes<std::size_t,shape_size>(start,steps);
  auto cm = permutation{1,2};
  auto rm = permutation{2,1};
  	
	auto q = 2ul;
	auto p = shape_size;
	
	for(auto const& na : shapes) 
	{
	  
	  auto m = na[0];
	  auto n = na[1];	  
	  auto u = m*2;
	  
	  ASSERT_TRUE(tlib::detail::is_case_rm<5>(p,q,rm.data()));

    
    auto a = matrix({m,n}, 1.0, rm); 
    auto b = matrix({u,n}, 1.0, rm); 
    auto c = matrix({m,u}, 0.0, rm);
    const auto nb = b.n();
    const auto nc = c.n();
    const auto pia = a.pi();
    
		init(a,q); 

    tlib::detail::mtm_rm(
		q,p,
		a.data(), na.data(), pia.data(),
		b.data(), nb.data(), 
		c.data(), nc.data()); 
		
		// mtm(a,b)
		
		// std::cout << "q = " << q << std::endl;
		// std::cout << "a = " << a << std::endl;
		// std::cout << "b = " << b << std::endl;
		// std::cout << "c = " << c << std::endl; 
		
	  for(auto i = 0u; i < m; ++i){
  	  for(auto j = 0u; j < u; ++j){
    		EXPECT_FLOAT_EQ(c(i,j), refc(a,i,q) );
      }
    }		
	}  
}



// q=2 | A(m,n),C(m,u) = RM | B(u,n) = RM | C = A x2 B => C = A *(rm) B'
TEST(MatrixTimesMatrix, Case6)
{
	using indices     = std::vector<std::size_t>;
	using permutation = std::vector<unsigned>;
	
	auto start = indices(3u,2u);
	auto steps = indices(3u,4u); 
	
	constexpr auto shape_size = 3u;

	auto shapes = tlib::gtest::generate_shapes<std::size_t,shape_size>(start,steps);
  auto rm = permutation{2,1};
 	auto p = shape_size;
	
	for(auto const& na : shapes) 
	{
	
		auto q = 1ul;
		auto layouts = std::vector<permutation>{ {1,2,3}, {1,3,2} } ;
		
		for(auto const& layout : layouts)
		{
	  
	    auto M = na[0];
	    auto N = na[1];
	    auto K = na[2];
	    auto U = na[q-1]*2;
      
      auto a = cube  ({M,N,K}, 1.0, layout); 
      auto b = matrix({U,M},   1.0, rm); 
      auto c = cube  ({U,N,K}, 0.0, layout);
      const auto nb = b.n();
      const auto nc = c.n();
      const auto pia = a.pi(); 
      
      ASSERT_TRUE(tlib::detail::is_case_rm<6>(p,q,pia.data()));
      
		  init(a,q); 

      tlib::detail::mtm_rm( q,p,   a.data(), na.data(), pia.data(),  b.data(), nb.data(),  c.data(), nc.data());   
		  
		  //std::cout << "q = " << q << std::endl;
		  //std::cout << "a = " << a << std::endl;
		  //std::cout << "b = " << b << std::endl;
		  //std::cout << "c = " << c << std::endl; 
		  
		  for(auto k = 0u; k < K; ++k){
    	  for(auto j = 0u; j < N; ++j){
      	  for(auto i = 0u; i < U; ++i){ 
        		EXPECT_FLOAT_EQ(c(i,j,k), refc(a,j,k,q) );
          }
        }
      }
      
      
	  }	// layouts
  } // shapes
	
	for(auto const& na : shapes) 
	{
	
		auto q = 2ul;
		auto layouts = std::vector<permutation>{ {2,1,3}, {2,3,1} } ;
		
		for(auto const& layout : layouts)
		{
	  
	    auto M = na[0];
	    auto N = na[1];
	    auto K = na[2];
	    auto U = na[q-1]*2;
      
      auto a = cube  ({M,N,K}, 1.0, layout); 
      auto b = matrix({U,N},   1.0, rm); 
      auto c = cube  ({M,U,K}, 0.0, layout);
      const auto nb = b.n();
      const auto nc = c.n();
      const auto pia = a.pi(); 
      
      ASSERT_TRUE(tlib::detail::is_case_rm<6>(p,q,pia.data()));
      
		  init(a, q); 

      tlib::detail::mtm_rm( q,p,   a.data(), na.data(), pia.data(),  b.data(), nb.data(),  c.data(), nc.data());   
		  
		  //std::cout << "q = " << q << std::endl;
		  //std::cout << "a = " << a << std::endl;
		  //std::cout << "b = " << b << std::endl;
		  //std::cout << "c = " << c << std::endl; 
		  
		  for(auto k = 0u; k < K; ++k){
    	  for(auto j = 0u; j < U; ++j){
      	  for(auto i = 0u; i < M; ++i){ 
        		EXPECT_FLOAT_EQ(c(i,j,k), refc(a,i,k,q) );
          }
        }
      }
      
      
	  }	// layouts
  } // shapes


	for(auto const& na : shapes) 
	{
	
		auto q = 3ul;
		auto layouts = std::vector<permutation>{ {3,1,2}, {3,2,1} } ;
		
		for(auto const& layout : layouts)
		{
	  
	    auto M = na[0];
	    auto N = na[1];
	    auto K = na[2];
	    auto U = na[q-1]*2;
      
      auto a = cube  ({M,N,K}, 1.0, layout); 
      auto b = matrix({U,K},   1.0, rm); 
      auto c = cube  ({M,N,U}, 0.0, layout);
      const auto nb = b.n();
      const auto nc = c.n();
      const auto pia = a.pi(); 
      
      ASSERT_TRUE(tlib::detail::is_case_rm<6>(p,q,pia.data()));
      
		  init(a, q); 

      tlib::detail::mtm_rm( q,p,   a.data(), na.data(), pia.data(),  b.data(), nb.data(),  c.data(), nc.data());   
		  
		  //std::cout << "q = " << q << std::endl;
		  //std::cout << "a = " << a << std::endl;
		  //std::cout << "b = " << b << std::endl;
		  //std::cout << "c = " << c << std::endl; 
		  
		  for(auto k = 0u; k < U; ++k){
    	  for(auto j = 0u; j < N; ++j){
      	  for(auto i = 0u; i < M; ++i){ 
        		EXPECT_FLOAT_EQ(c(i,j,k), refc(a,i,j,q) );
          }
        }
      }
      
      
	  }	// layouts
  } // shapes
	
}


#if 0
template<class value_type, class size_type, class blas_functor_type>
inline void check_mtm_help(
		blas_functor_type&& blas_function,
		std::vector<value_type> const& a,
		std::vector<value_type> const& b,
		std::vector<size_type> pia,
		std::vector<size_type> na)
{
	assert(pia.size() == 2u);
	assert(na .size() == 2u);
	
	auto const pic = pia;
	auto const nc  = na;
	
	auto const m = na[0];
	auto const n = na[1];
	
	assert(m > 0);
	assert(n > 0);	
	assert(n == b.size());
	
	auto c = std::vector<value_type>(m,0);
	
	auto const lda = pia[0]==1?m:n;
	
	blas_function(a.data(),b.data(),c.data(),m,n,lda);
	
/*	
	std::cout << "a = ";
	std::copy(a.begin(),a.end(),std::ostream_iterator<value_type>(std::cout," "));
	std::cout << std::endl;
	
	std::cout << "b = ";
	std::copy(b.begin(),b.end(),std::ostream_iterator<value_type>(std::cout," "));
	std::cout << std::endl;

	std::cout << "c = ";
	std::copy(c.begin(),c.end(),std::ostream_iterator<value_type>(std::cout," "));
	std::cout << std::endl;
*/	

	auto fn = [n](auto i){ return (i*n*(i*n+1))/2; };
	
	for(auto i = 1ul; i <= m; ++i){		
		const auto j = fn(i);				
		const auto k =  i>0ul ? fn(i-1) : 0ul;		
		const auto sum = j-k;
		EXPECT_EQ( c[i-1], sum );
	}


}


template<class value_type, class size_type>
inline void initialize_matrix( std::vector<value_type> & a, std::vector<size_type> const& na, std::vector<size_type> const& pia)
{
  assert(na. size() == 2u);
  assert(pia.size() == 2u);
  
  auto m = na[0];
  auto n = na[1];

/* a = [1 2 3
        4 5 6] */
	//row-major
	if(pia[1] == 1)
		for(auto i = 0ul; i < m; ++i) // 0,1
			for(auto j = 0ul; j < n; ++j) // 0,1,2
				a[j+i*n] = j+1 + i*n;
/* a = [1  2 5
        3 4 6] */
	//col-major
	else
		for(auto j = 0ul; j < n; ++j) // 0,1,2
			for(auto i = 0ul; i < m; ++i) // 0,1
				a[i+j*m] = j+1 + i*n;
	
}


TEST(MatrixTimesVector, Gemv)
{
	using value_type = double;
	using size_type = std::size_t;
	
	auto start = std::vector<size_type>(2u,2u);
	/* auto steps =std::vector<size_type>(2u,8u); */
	auto steps =std::vector<size_type>(2u,10u);
	
	auto shapes   = tlib::gtest::generate_shapes<size_type,2u>(start,steps);
	
	for(auto const& na : shapes) 
	{
		auto const nn = std::accumulate(na.begin(), na.end(), size_type{1} , std::multiplies<size_type>());
		auto a = std::vector<value_type>(nn);
		auto b = std::vector<value_type>(na.at(1),value_type{1});
		
		auto cm = std::vector<size_type>{1,2}; // column-major
		check_mtv_init(a,na,cm);		
		check_mtv_help(tlib::detail::gemv_col<value_type,size_type>,a,b,cm,na);		
		
		auto rm = std::vector<size_type>{2,1}; // row-major		
		check_mtv_init(a,na,rm);
		check_mtv_help(tlib::detail::gemv_row<value_type,size_type>,a,b,rm,na);
	}
}


TEST(MatrixTimesVector, GemvParallel)
{
	using value_type = double;
	using size_type = std::size_t;
	
	const auto start = std::vector<size_type>(2u,2u);
	/* const auto steps =std::vector<size_type>(2u,8u); */
	const auto steps =std::vector<size_type>(2u,10u);
	
	const auto shapes   = tlib::gtest::generate_shapes<size_type,2u>(start,steps);
	
	for(auto const& na : shapes) 
	{
		auto const nn = std::accumulate(na.begin(), na.end(), size_type{1} , std::multiplies<size_type>());
		auto a = std::vector<value_type>(nn);
		auto b = std::vector<value_type>(na.at(1),value_type{1});
		
		auto cm = std::vector<size_type>{1,2}; // column-major
		check_mtv_init(a,na,cm);		
		check_mtv_help(tlib::detail::gemv_col_parallel<value_type,size_type>,a,b,cm,na);		
		
		auto rm = std::vector<size_type>{2,1}; // row-major		
		check_mtv_init(a,na,rm);
		check_mtv_help(tlib::detail::gemv_row_parallel<value_type,size_type>,a,b,rm,na);
	}
}


TEST(MatrixTimesVector, GemvBLAS)
{
	using value_type = float;
	using size_type = std::size_t;
	
	const auto start = std::vector<size_type>(2u,2u);
	const auto steps =std::vector<size_type>(2u,8u);
	
	const auto shapes   = tlib::gtest::generate_shapes<size_type,2u>(start,steps);
	
	for(auto const& na : shapes) 
	{
		auto const nn = std::accumulate(na.begin(), na.end(), size_type{1} , std::multiplies<size_type>());
		auto a = std::vector<value_type>(nn);
		auto b = std::vector<value_type>(na.at(1),value_type{1});
		
		auto cm = std::vector<size_type>{1,2}; // column-major
		check_mtv_init(a,na,cm);		
		check_mtv_help(tlib::detail::gemv_col_blas<value_type,size_type>,a,b,cm,na);		
		
		auto rm = std::vector<size_type>{2,1}; // row-major		
		check_mtv_init(a,na,rm);
		check_mtv_help(tlib::detail::gemv_row_blas<value_type,size_type>,a,b,rm,na);
	}
}


#endif
