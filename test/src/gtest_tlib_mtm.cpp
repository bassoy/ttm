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
  using base_type = std::vector<value_type>;
  using shape_type = std::vector<size_type>;
  using permutation_type = std::vector<unsigned>;

  matrix() = delete;
  matrix(shape_type n, value_type v = 0.0, permutation_type pi = {1,2}) 
    : base_(elements_(n),v)
    , n_(n)
    , pi_(pi)
    , w_(strides_(n,pi)) 
    {
    }
    
  matrix(matrix const& other) : base_(other.base_), n_(other.n_), pi_(other.pi_), w_(other.w_) {}

  inline const value_type* data() const { return this->base_.data(); }
  inline value_type* data() { return this->base_.data(); }
  inline base_type const&  base() const { return this->base_; }
  inline base_type &  base() { return this->base_; }
  inline shape_type n()    const { return this->n_; }
  inline shape_type w()    const { return w_; }
  inline permutation_type pi()   const { return pi_; }
  inline unsigned  p()    const { return n_.size(); }
  
  void set(permutation_type pi)
  {
    pi_ = pi;
    w_ = strides_(n_,pi_);
  }
  
  inline bool is_cm()  const { return pi_[0] == 1; }
  

  inline value_type      & operator()(size_type i, size_type j)       { return base_[at_(i,j)]; }
  inline value_type const& operator()(size_type i, size_type j) const { return base_[at_(i,j)]; }
  
  inline value_type      & operator[](size_type j)       { return base_[j]; }
  inline value_type const& operator[](size_type j) const { return base_[j]; }
  

private:
  base_type base_;
  shape_type n_;
  permutation_type pi_;
  shape_type w_;

 
  inline auto at_(size_type i, size_type j) const{
    return i*w_[0] + j*w_[1];
  }

  static inline auto elements_(shape_type n){
    return std::accumulate(n.begin(), n.end(), 1ull, std::multiplies<size_type>());
  }
  
  static inline auto strides_(shape_type n, permutation_type pi){    
    unsigned p = n.size();
    auto w = shape_type(p,1);
    for(auto r = 1u; r < p; ++r)
      w[pi[r]-1] = w[pi[r-1]-1] * n[pi[r-1]-1];
    
    return w;
  }
  
};

template<class matrix_type>
void init( matrix_type& a )
{

  auto m = a.n()[0];
  auto n = a.n()[1];
    
  if (a.is_cm()){
    for(auto j = 0ul; j < n; ++j)
    	for(auto i = 0ul; i < m; ++i)
        a(i,j) = j+1 + i*n;
  }
	else{
  	for(auto i = 0ul; i < m; ++i)
	  	for(auto j = 0ul; j < n; ++j)
	  	  a(i,j) = j+1 + i*n;	  
  }
}

template<class value_type>
std::ostream& operator<< (std::ostream& out, matrix<value_type> const& a)
{
 
  auto m = a.n()[0];
  auto n = a.n()[1];
  auto is_col = a.is_cm();  

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
  
  auto ar = std::cref(a);
  auto br = std::cref(b);
  auto cr = std::ref(c);  
  
  if (cmajor) 
  {
//#pragma omp parallel for collapse(2) firstprivate(M,N, ar,br,cr)
    for(auto j = 0ul; j < N; ++j){
        for(auto i = 0ul; i < M; ++i){
          cr.get()[i] += ar.get()(i,j) * br.get()[j];
      	}
      }
  }
  else // row
  {
#pragma omp parallel for firstprivate(M,N, ar,br,cr)
    for(auto i = 0ul; i < M; ++i){
      auto cc = cr.get()[i];
#pragma omp simd reduction(+:cc)
    	for(auto j = 0ul; j < N; ++j){
    	  cc += ar(i,j) * br.get()[j];
    	}
    	cr.get()[i] = cc;
    }
  }
  return c;	
}


template<class value_t>
value_t refc(matrix<value_t> const& a, std::size_t N, std::size_t i)
{
  auto sum = [](auto n) { return (n*(n+1))/2u; };
  return sum(a(i,N-1)) - sum(a(i,0)-1.0);
};


template<class value_t>
value_t refc_rm(matrix<value_t> const& a, std::size_t M, std::size_t j)
{
  auto sum = [](auto n) { return (n*(n+1))/2u; };
  return sum(a(M-1,j)) - sum(a(0,j)-1.0);
};
		

TEST(MatrixTimesVector, Ref)
{
	using indices = std::vector<std::size_t>;
	using permuration = std::vector<unsigned>;
	
	auto start = indices(2u,2u);
	auto steps = indices(2u,8u);

	auto shapes   = tlib::gtest::generate_shapes<std::size_t,2u>(start,steps);	
	auto formats  = std::array<permuration,2>{permuration{1,2}, permuration{2,1} };

	for(auto const& n : shapes) 
	{ 
    auto M = n[0];
    auto N = n[1];
	
	  for(auto f : formats) 
	  {
    
      auto a = matrix({M,N}, 0.0, f);
      auto b = matrix({N,1}, 1.0, f);
		  
		  init(a);

		  auto cref = mtv(a,b);

		  for(auto i = 0ul; i < M; ++i)
    		EXPECT_FLOAT_EQ(cref[i], refc(a,N,i) );
    }
   
	}
}


  
TEST(MatrixTimesMatrix, Ref)
{
	using indices = std::vector<std::size_t>;
	using permuration = std::vector<unsigned>;
	
	auto start = indices(2u,2u);
	auto steps = indices(2u,6u);

	auto shapes  = tlib::gtest::generate_shapes<std::size_t,2u>(start,steps);
	auto formats = std::array<permuration,2>{permuration{1,2}, permuration{2,1} };
	
	for(auto const& n : shapes) 
	{ 
	    auto M = n[0];
      auto N = n[1];
      auto K = n[1]*2;
	
	  for(auto f : formats) 
	  {

      auto a = matrix({M,K}, 0.0, f);
      auto b = matrix({K,N}, 1.0, f);
		  
		  init(a);

		  auto cref = mtm(a,b);

		  for(auto i = 0u; i < M; ++i){
    		for(auto j = 0u; j < N; ++j){
      		EXPECT_FLOAT_EQ(cref(i,j), refc(a,K,i) );
        }
      }
    }
	}
}

// 
// A = nx1, B(rm) = mxn, C = mx1
// C = A x1 B => c = B *(rm) a
TEST(MatrixTimesMatrix, Case1)
{
	using indices = std::vector<std::size_t>;
	using permuration = std::vector<unsigned>;
	
	auto start = indices(2u,2u);
	auto steps = indices(2u,6u);

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
    
    auto na = a.n();
    auto nc = c.n();

		
		init(b);

		
    tlib::detail::mtm_rm(
		q,p,
		a.data(), na.data(), cm.data(),
		b.data(), nb.data(), 
		c.data(), nc.data());
		
	  for(auto i = 0ul; i < m; ++i)
  		EXPECT_FLOAT_EQ(c[i], refc(b,n,i) );
	}
}


// 
// A(cm) = mxn, B(rm) = qxm, C(cm) = nxq
// C = A x1 B ==> C = A *(rm) B' 
TEST(MatrixTimesMatrix, Case2)
{
	using indices     = std::vector<std::size_t>;
	using permutation = std::vector<unsigned>;
	
	auto start = indices(2u,2u);
	auto steps = indices(2u,6u); 
	
	constexpr auto shape_size = 2u;

	auto shapes = tlib::gtest::generate_shapes<std::size_t,shape_size>(start,steps);
  auto cm = permutation{1,2};
  auto rm = permutation{2,1};
  	
	auto q = 1ul;
	auto p = shape_size;
	
	for(auto const& na : shapes) 
	{
	  
	  auto n1 = na[0];
	  auto n2 = na[1];	  
	  auto m  = n1*2;
	  
	  ASSERT_TRUE(tlib::detail::is_case_rm<2>(p,q,cm.data()));

    
    auto a = matrix({n1,n2}, 1.0, rm); 
    auto b = matrix({m, n1}, 1.0, rm); 
    auto c = matrix({m, n2}, 0.0, rm);
    const auto nb = b.n();
    const auto nc = c.n();
    
		init(a); 
		
		a.set(cm);
		c.set(cm);

		
    tlib::detail::mtm_rm(
		q,p,
		a.data(), na.data(), cm.data(),
		b.data(), nb.data(), 
		c.data(), nc.data()); 
		
		// mtm(a,b)
		
		// std::cout << "a = " << a << std::endl;
		// std::cout << "b = " << b << std::endl;
		// std::cout << "c = " << c << std::endl;
		
	  for(auto i = 0u; i < m; ++i){
  		for(auto j = 0u; j < n2; ++j){  		  
    		EXPECT_FLOAT_EQ(c(i,j), refc_rm(a,n1,j) ); // check refc_rm. seems buggy. is not yet correctly implemented.
      }
    }
	}
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
