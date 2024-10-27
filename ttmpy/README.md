## Installation

Clone the repo
```bash
git clone git@github.com:bassoy/ttm.git
cd ttm
```

Install OpenBLAS and MLK
```bash
sudo apt install libmkl-dev intel-mkl libopenblas-dev libopenblas64-pthread-dev libomp-dev python3-pip python3-virtualenv
```

Navigate to the python wrapper folder
```bash
cd ttmpy
```

Activate the virtual environment `env-ttm`
```bash
virtualenv ~/env-ttm
source ~/env-ttm/bin/activate
```

Install the pacakge in editable mode
```bash
pip install -e .
```

Test the package
```bash
cd tests
python3 -m unittest discover -v
```

Deactivate the virtual environment `env-ttm`
```bash
deactivate
```


## Python Interface

### q-mode Tensor-Matrix Product
```python
C = ttm(q, A, b)
```
* `q` is the contraction mode with `1<=q<=A.ndim`
* `A` is an input `numpy.ndarray` with typically `A.ndim>=2`
* `B` is an input `numpy.ndarray` with `b.ndim=2` and `b.shape[1]=A.shape[q-1]` and `b.shape[0]=C.shape[q-1]`
* `C` is the output `numpy.ndarray` with `A.ndim`

## Python Example

### 1-mode Tensor-Matrix Product
```python
import numpy as np
import ttmpy as tp

A = np.arange(4*3*2, dtype=np.float64).reshape(4,3,2)
B = np.arange(5*4,   dtype=np.float64).reshape(5,4)
C = tp.ttm(1,A,B)
D = np.einsum("ijk,xi->xjk", A, B)
np.all(np.equal(C,D))
```
