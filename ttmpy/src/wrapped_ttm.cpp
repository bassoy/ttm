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
	if(q==0 || q>p) throw std::invalid_argument("Error calling ttmpy::ttm: contraction mode should be greater than zero or less than or equal to p.");  
  
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
  assert(0);
#endif

  return c;  
}


PYBIND11_MODULE(ttmpy, m)
{
  m.doc() = "python plugin ttmpy for fast execution of the tensor-times-matrix multiply";
  m.def("ttm",  &ttm<double> , "computes the tensor-matrix product for the q-th mode", py::return_value_policy::move, py::arg("q"), py::arg("A"), py::arg("B"));  
}
