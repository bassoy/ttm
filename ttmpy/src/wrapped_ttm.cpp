#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <tlib/detail/layout.h>
#include <tlib/detail/shape.h>
#include <tlib/detail/strides.h>
#include <tlib/ttm.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <functional>

// g++ -Wall -shared -std=c++17 wrapped_ttv.cpp -o ttvpy.so $(python3 -m pybind11 --includes) -I../include -fPIC -fopenmp -DUSE_OPENBLAS -lm -lopenblas


namespace py = pybind11;


template<class T>
py::array_t<T> 
ttm(std::size_t const contraction_mode,
    py::array_t<T> const& a,
    py::array_t<T> const& b)
{

  auto const q = contraction_mode;
  
  auto const sizeofT = sizeof(T);
  
  auto const& ainfo = a.request(); // request a buffer descriptor from Python of type py::buffer_info
  auto const p = std::size_t(ainfo.ndim); //py::ssize_t  

	if(p==0)        throw std::invalid_argument("Error calling ttmpy::ttm: first input should be a tensor with order greater than zero.");
	if(q==0 || q>p) throw std::invalid_argument("Error calling ttmpy::ttm: contraction mode should be greater than zero and at most equal to p.");  
  
  auto const*const aptr = static_cast<T const*const>(ainfo.ptr);    // extract data an shape of input array  
  auto na = std::vector<std::size_t>(ainfo.shape  .begin(), ainfo.shape  .end());
  auto wa = std::vector<std::size_t>(ainfo.strides.begin(), ainfo.strides.end());
  std::for_each(wa.begin(), wa.end(), [sizeofT](auto& w){w/=sizeofT;});

  auto pia = tlib::detail::generate_k_order_layout(p, p);
  auto pib = tlib::detail::generate_k_order_layout(2ul, 2ul);
  
  auto const& binfo = b.request(); // request a buffer descriptor from Python of type py::buffer_info
  auto const*const bptr = static_cast<T const*const>(binfo.ptr);    // extract data an shape of input array
  auto const nb  = std::vector<std::size_t>(binfo.shape.begin(), binfo.shape.end());
  auto const pb  = binfo.ndim;  
  if(pb!=2)
    throw std::invalid_argument("Error calling ttmpy::ttm: second input should be a mtrix with order equal to 2.");

  
	auto nc  = na; // tlib::detail::generate_output_shape (na ,q);
	nc[q-1] = nb[0];
	auto const& pic = pia;
	//auto const pic = tlib::detail::generate_output_layout(pia,q);	
	auto wc  = tlib::detail::generate_strides(nc,pic);
	auto nc_ = std::vector<py::ssize_t>(nc.begin(),nc.end());
	auto wc_ = std::vector<py::ssize_t>(wc.begin(),wc.end());
  std::for_each(wc_.begin(), wc_.end(), [sizeofT](auto& w){w*=sizeofT;});
  
 	auto c            = py::array_t<T>(nc_,wc_);
  auto const& cinfo = c.request(); // request a buffer descriptor from Python of type py::buffer_info
  auto* cptr        = static_cast<T*>(cinfo.ptr);    // extract data an shape of input array  
  // auto nnc          = std::size_t(cinfo.size);

 
#if defined(USE_OPENBLAS) || defined(USE_MKL)
  tlib::ttm<T>(tlib::parallel_policy::omp_forloop_t{}, tlib::slicing_policy::slice_t{}, tlib::fusion_policy::all_t{}, 
                               q, p, 
                               aptr, na.data(), wa.data(), pia.data(),  
                               bptr, nb.data(),            pib.data(), 
                               cptr, nc.data(), wc.data());
#else 
  static_assert(0,"Static assert in ttmpy/ttm: missing specification of define USE_OPENBLAS or USE_MKL");
#endif

  return c;  
}




template<class T>
py::array_t<T> 
ttms(std::size_t const non_contraction_mode,
     py::array_t<T> const& apy,
     py::list const& bpy,
     std::string morder
    )
{

  if(morder!="optimal" && morder!="backward" && morder!="forward"){
    throw std::invalid_argument("Error calling ttmpy::ttms: multiplication order should be either 'optimal', 'backward' or 'forward'.");
  }

  auto const q = non_contraction_mode;
 
  auto const& ainfo = apy.request(); // request a buffer descriptor from Python of type py::buffer_info
  auto const p = std::size_t(ainfo.ndim); //py::ssize_t

	if(p==0)
	  throw std::invalid_argument("Error calling ttmpy::ttms: input tensor order should be greater than zero.");
	if(bpy.size() != p-1) 
	  throw std::invalid_argument("Error calling ttmpy::ttms: number of input matrices is not equal to the tensor order - 1.");	
	if(q==0 || q>p) 
	  throw std::invalid_argument("Error calling ttmpy::ttms: non-contraction mode should be greater than zero and at most equal to p.");  

  auto na = std::vector<std::size_t>(ainfo.shape.begin(), ainfo.shape.end());
  
  assert(na.size() == p);
  
  auto nb = std::vector<std::pair<std::size_t,std::size_t>>(p-1);
  auto bs = std::vector<py::array_t<T>>(p-1);
  // cast py::list of py::array_t to std::vector of py::array_t
  std::transform(bpy.begin(), bpy.end(), bs.begin(), [](auto const& bj){ return py::cast<py::array_t<T>>(bj); }  );
  
  // check if all array orders are equal to 2 (need to be matrices)
  auto all_dim_2 = std::all_of( bs.begin(), bs.end(), [](auto const& bj){ return bj.request().ndim == 2u; } ); 
  if(!all_dim_2)
    throw std::invalid_argument("Error calling ttmpy::ttms: some of the input matrices are not matrices.");  
    
  // copy vector dimensions to a separate container for convenience
  std::transform( bs.begin(), bs.end(), nb.begin(), [](auto const& b){ return std::make_pair(b.request().shape[0],b.request().shape[1]); } );

  // check if vector dimensions and corresponding tensor extents are the same
  bool na_equal_nb_q_1 = q==1 || std::equal ( nb.begin()  , nb.begin()+(q-1), na.begin()    , [](auto const& l, auto const& r){ return l.second == r; } );
  bool na_equal_nb_q_2 = q==p || std::equal ( nb.begin()+q, nb.end  ()      , na.begin()+q+1, [](auto const& l, auto const& r){ return l.second == r; } );  
  if(!na_equal_nb_q_1 || !na_equal_nb_q_2) 
    throw std::invalid_argument("Error calling ttmpy::ttms: vector dimension is not compatible with the dimension of a tensor mode.");  


  // B[0]...B[p-2]
  // r = 1,...,q-1,q+1,...,p <- contraction dimensions [1-based]
  
  
  auto c = py::array_t<T>{};
 
  // backward contractions  
  if( morder == "backward" ){


    auto r0 = q==p ? p-1 : p;
    c  = ttm(r0, apy, bs[p-2]);
    
    if(q==p){
      for(auto r = p-2; r > 0u ; --r)
        c = ttm(r,c,bs[r-1]);
    }
    else{
      for(auto r = p-1; r >  q; --r)
        c = ttm(r,c,bs[r-2]);
      for(auto r = q-1; r > 0; --r)
        c = ttm(r,c,bs[r-1]);
    }


    
  }
  else if( morder == "forward" ){
    auto r0 = q==1 ? 2u : 1u;
    c  = ttm(r0, apy, bs[0]);
    
    if(q==1u){
      for(auto r = 3u; r <= p; ++r)
        c = ttm(r,c,bs[r-2]);
    }
    else{
      for(auto r = 2u;  r <  q; ++r)
        c = ttm(r,c,bs[r-1]);
      for(auto r = q+1; r <= p; ++r)
        c = ttm(r,c,bs[r-2]);
    }
  }
  else /*if ( morder == "optimal" )*/ {  
    
    #if 0
    
    // copy references of all vectors and their contraction dimension.    
    auto bpairs = std::vector<std::pair<py::array_t<T>*,unsigned>>(p-1);   
    for(auto r = 1u; r < q; ++r) /* r = 1...q-1*/
      bpairs.at(r-1) = std::make_pair(std::addressof(bs.at(r-1)),r);     
    for(auto r = q+1; r <= p; ++r) /*r = q+1...p*/
      bpairs.at(r-2) = std::make_pair(std::addressof(bs.at(r-2)),r);
    
    // sort (ascending)  all vector references according to their dimension
    auto rhs_dim_is_larger = [](auto const& lhs, auto const& rhs){ return lhs.first->shape(0) < rhs.first->shape(0);};
    std::sort(bpairs.begin(), bpairs.end(), rhs_dim_is_larger);
    
    // check if vectors are well sorted.
    assert(std::is_sorted(bpairs.begin(), bpairs.end(), rhs_dim_is_larger));
          
    // update contraction modes for the remaining vectors after contraction
    auto update_contraction_modes = [&bpairs](auto const& ib){
      assert(ib != bpairs.rend());
      auto const r = ib->second;
      auto decrease_contraction = [r](auto &bpair){ if(bpair.second>r) --bpair.second; }; 
      std::for_each(ib, bpairs.rend(), decrease_contraction);   
    };
    
    auto ib = bpairs.rbegin();
    c = ttm(ib->second, apy, *(ib->first));  
    update_contraction_modes(ib);
    
    for(++ib; ib != bpairs.rend(); ++ib){
      c = ttm(ib->second, c, *(ib->first));
      update_contraction_modes(ib);
    } 
    
    #endif

  }

  return c;
  
}



PYBIND11_MODULE(ttmpy, m)
{
  m.doc() = "python plugin ttmpy for fast execution of the tensor-times-matrix multiply";
  m.def("ttm",  &ttm <double>, "computes the tensor-matrix product for the q-th mode",             py::return_value_policy::move, py::arg("q"), py::arg("A"), py::arg("B"));
  m.def("ttms", &ttms<double>, "computes multiple tensor-matrix product except for the q-th mode", py::return_value_policy::move, py::arg("q"), py::arg("A"),py::arg("Bs"), py::arg("order")="forward");    
}
