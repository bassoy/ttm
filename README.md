High-Performance Tensor-Matrix Multiplication Library - TLIB(TTM)
=====
[![Language](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B#Standardization)
[![License](https://img.shields.io/badge/license-GPL-blue.svg)](https://github.com/bassoy/ttm/blob/master/LICENSE)
[![Wiki](https://img.shields.io/badge/ttm-wiki-blue.svg)](https://github.com/bassoy/ttm/wiki)
[![Discussions](https://img.shields.io/badge/ttm-discussions-blue.svg)](https://github.com/bassoy/ttm/discussions)
[![Build Status](https://github.com/bassoy/ttm/actions/workflows/test.yml/badge.svg)](https://github.com/bassoy/ttm/actions)

## Summary
**TLIB(TTM)** is C++ high-performance tensor-matrix multiplication **header-only library**.
It provides free C++ functions for parallel computing the **mode-`q` tensor-times-matrix product** of the general form

$$
\underline{\mathbf{C}} = \underline{\mathbf{A}} \times_q \mathbf{B} \quad :\Leftrightarrow \quad
\underline{\mathbf{C}} (i_1, \dots, i_{q-1}, j, i_{q+1}, \dots, i_p) = \sum_{i_q=1}^{n_q} \underline{\mathbf{A}}({i_1, \dots, i_q,  \dots, i_p}) \cdot \mathbf{B}({j,i_q}).
$$

where $q$ is the contraction mode, $\underline{\mathbf{A}}$ and $\underline{\mathbf{C}}$ are tensors of order $p$ with shapes $\mathbf{n}\_a= (n\_1,\dots n\_{q-1},n\_q ,n\_{q+1},\dots,n\_p)$ and $\mathbf{n}\_c = (n\_1,\dots,n\_{q-1},m,n\_{q+1},\dots,n\_p)$, respectively. The order $2$ tensor $\mathbf{B}$ is a matrix with shape $\mathbf{n}\_b = (m,n\_{q})$.

## Usage & Installation

Please have a look at the [wiki](https://github.com/bassoy/ttm/wiki) page for more informations about library **usage**, function **interfaces** and the **parameters** settings.

## Key Features

### Flexibility
* Contraction mode $q$, tensor order $p$, tensor extents $n$ and tensor layout $\mathbf{\pi}$ can be chosen at runtime
* Supports any linear tensor layout inlcuding the first-order and last-order storage layouts
* Offers two high-level and one C-like low-level interfaces for calling the tensor-times-matrix multiplication
* Implemented independent of a tensor data structure (can be used with `std::vector` and `std::array`)
* Currently supports float and double

### Performance
* Multi-threading support with OpenMP v4.5 or higher
* Currently must be used with a BLAS implementation
* Performs in-place operations without transposing the tensor - no extra memory needed
* For large tensors reaches peak matrix-times-matrix performance

### Requirements
* Requires the tensor elements to be contiguously stored in memory
* Element types must be either `float` or `double`

## Python Example
```python
import numpy as np
import ttmpy as tp

A = np.arange(4*3*2, dtype=np.float64).reshape(4,3,2)
B = np.arange(5*4,   dtype=np.float64).reshape(5,4)
C = tp.ttm(1,A,B)
D = np.einsum("ijk,xi->xjk", A, B)
np.all(np.equal(C,D))
```

## C++ Example 
```cpp
/*main.cpp*/
#include <tlib/ttm.h>

#include <vector>
#include <numeric>
#include <iostream>


int main()
{
    using value_t    = float;
    using tensor_t   = tlib::tensor<value_t>;

    auto A = tensor_t( {4,3,2} );
    auto B = tensor_t( {5,4}   );

    std::iota(A.begin(),A.end(),1);
    std::fill(B.begin(),B.end(),1);
    
    std::cout << "A = " << A << std::endl;
    std::cout << "B = " << B << std::endl;

/*
  A =
  { 1  5  9  | 13 17 21
    2  6 10  | 14 18 22
    3  7 11  | 15 19 23
    4  8 12  | 16 20 24 };

  B =
  { 1  1  1  1  1
    1  1  1  1  1
    1  1  1  1  1
    1  1  1  1  1};
*/

    auto C = A (1)* B;

    std::cout << "C = " << C << std::endl;


/* for q=1
  C =
  { 1+..+4 5+..+8 9+..+12 | 13+..+16 17+..+20 21+..+24
      ..     ..     ..    |    ..       ..       ..
    1+..+4 5+..+8 9+..+12 | 13+..+16 17+..+20 21+..+24 };
*/
}
```


## Citation
If you want to refer to TTM as part of a research paper, please cite the article [Design of a high-performance tensor–matrix multiplication with BLAS](https://doi.org/10.1016/j.jocs.2025.102568)

```
@article{ttm:bassoy:2025,
title = {Design of a high-performance tensor–matrix multiplication with BLAS},
journal = {Journal of Computational Science},
volume = {87},
pages = {102568},
year = {2025},
issn = {1877-7503},
doi = {https://doi.org/10.1016/j.jocs.2025.102568},
url = {https://www.sciencedirect.com/science/article/pii/S1877750325000456},
author = {Cem Savas Bassoy},
}
``` 
