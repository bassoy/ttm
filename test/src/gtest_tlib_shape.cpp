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

#include <algorithm>
#include <array>
#include <vector>


#include <tlib/detail/shape.h>
using namespace tlib::ttm;

class ShapeTest : public ::testing::Test {
protected:
	using shape = std::vector<unsigned>;

	void SetUp() override 
	{
		shapes = 
		{
			shape{},      // 0
			shape{0},     // 1
			shape{1},     // 2
			shape{2},     // 3
			shape{0,1},   // 4
			shape{1,0},   // 5
			shape{1,1},   // 6
			shape{2,1},   // 7
			shape{3,1},   // 8
			shape{1,2},   // 9
			shape{1,3},   //10
			shape{2,2},   //11
			shape{3,3},   //12
			shape{0,1,1}, //13
			shape{1,1,0}, //14
			shape{1,1,1}, //15
			shape{1,1,2}, //16
			shape{1,2,1}, //17
			shape{1,2,2}, //18
			shape{2,1,1}, //19
			shape{2,1,2}, //20
			shape{2,2,1}, //21
			shape{2,2,2}  //22
		};
  }
  std::vector<shape> shapes;  
};

TEST_F(ShapeTest, is_scalar)
{
  auto ints = std::vector<unsigned>{2,6,15};

  for(auto i : ints){
    EXPECT_TRUE (detail::is_scalar(shapes[i].begin(), shapes[i].end()));		
  }

  for(auto i = 0u; i < shapes.size(); ++i){
    if(std::find(ints.begin(), ints.end(),i)==ints.end()){
      EXPECT_FALSE(detail::is_scalar(shapes[i].begin(), shapes[i].end()));
    }
  }
}


TEST_F(ShapeTest, is_vector)
{
  auto ints = std::vector<unsigned>{3,7,8,9,10,17,19};

  for(auto i : ints){
    EXPECT_TRUE (detail::is_vector(shapes[i].begin(), shapes[i].end()));		
  }
  for(auto i = 0u; i < shapes.size(); ++i ){
    if(std::find(ints.begin(), ints.end(),i)==ints.end()){
      EXPECT_FALSE(detail::is_vector(shapes[i].begin(), shapes[i].end()));
    }
  }
}


TEST_F(ShapeTest, is_matrix)
{
  auto ints = std::vector<unsigned>{11,12,21};

  for(auto i : ints ){
    EXPECT_TRUE (detail::is_matrix(shapes[i].begin(), shapes[i].end()));		
  }
  for(auto i = 0u; i < shapes.size(); ++i ){
    if(std::find(ints.begin(), ints.end(),i)==ints.end()){
      EXPECT_FALSE(detail::is_matrix(shapes[i].begin(), shapes[i].end()));
    }
  }
}


TEST_F(ShapeTest, is_tensor)
{
  auto ints = std::vector<unsigned> {16,18,20,22};

  for(auto i : ints ){
    EXPECT_TRUE (detail::is_tensor(shapes[i].begin(), shapes[i].end()));		
  }
  for(auto i = 0u; i < shapes.size(); ++i ){
    if(std::find(ints.begin(), ints.end(),i)==ints.end()){
      EXPECT_FALSE(detail::is_tensor(shapes[i].begin(), shapes[i].end()));
    }
  }
}

TEST_F(ShapeTest, is_valid)
{
  auto ints = std::vector<unsigned> {0,1,4,5,13,14};

  for(auto i : ints ){
    EXPECT_FALSE(detail::is_valid_shape(shapes[i].begin(), shapes[i].end()));		
  }
  for(auto i = 0u; i < shapes.size(); ++i ){
    if(std::find(ints.begin(), ints.end(),i)==ints.end()){
      EXPECT_TRUE(detail::is_valid_shape(shapes[i].begin(), shapes[i].end()));
    }
  }
}


