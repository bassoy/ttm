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

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <cassert>
#include <cstddef>
#include <vector>


namespace tlib::ttm::detail
{

template<class InputIt>
inline bool is_valid_shape(InputIt begin, InputIt end)
{	
	return begin!=end && std::none_of(begin, end, [](auto a){ return a==0u; });
}
	


template<class InputIt>
inline bool is_scalar(InputIt begin, InputIt end)
{
	if(!is_valid_shape(begin,end))
		return false;
		
	return std::all_of(begin, end, [](auto const& a){ return a == 1u;});
}


template<class InputIt>
inline bool is_vector(InputIt begin, InputIt end)
{
	if(!is_valid_shape(begin,end))
		return false;
		
	if(begin == end)
		return false;
		
	if(begin+1 == end)
		return *begin>1u;

	return  std::any_of(begin,    begin+2u, [](auto const& a){ return a >  1u;} ) &&
	        std::any_of(begin,    begin+2u, [](auto const& a){ return a == 1u;} ) &&
	        std::all_of(begin+2u, end,      [](auto const& a){ return a == 1u;} );
}

template<class InputIt>
inline bool is_matrix(InputIt begin, InputIt end)
{
	if(!is_valid_shape(begin,end))
		return false;
		
	if(std::distance(begin,end) < 2u)
		return false;

	return  std::all_of(begin,    begin+2u, [](auto const& a){ return a >   1u;} ) &&
	        std::all_of(begin+2u, end,      [](auto const& a){ return a ==  1u;} );
}


template<class InputIt>
inline bool is_tensor(InputIt begin, InputIt end)
{
	if(!is_valid_shape(begin,end))
		return false;
		
	if(std::distance(begin,end) < 3u)
		return false;

	return std::any_of(begin+2u, end, [](auto const& a){ return a > 1u;});
}





} // namespace tlib::ttm::detail
