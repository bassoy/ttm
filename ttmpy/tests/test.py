import unittest
import numpy as np
import ttmpy as tp

class TestTTM(unittest.TestCase):
  
  def test_ttm_mode1(self):
    A = np.arange(3*2*4, dtype=np.float64).reshape(3,2,4)
    b = np.arange(5*3, dtype=np.float64).reshape(5,3)
    D = np.einsum("ijk,xi->xjk", A, b)
    C = tp.ttm(1, A, b)
    self.assertTrue(np.all(C==D))
 
  def test_ttm_mode2(self):
    A = np.arange(3*2*4, dtype=np.float64).reshape(3,2,4)
    b = np.arange(5*2, dtype=np.float64).reshape(5,2)
    D = np.einsum("ijk,xj->ixk", A, b)
    C = tp.ttm(2, A, b)
    self.assertTrue(np.all(C==D))


  def test_ttm_mode3(self):
    A = np.arange(3*2*4, dtype=np.float64).reshape(3,2,4)
    b = np.arange(5*4, dtype=np.float64).reshape(5,4)
    D = np.einsum("ijk,xk->ijx", A, b)
    C = tp.ttm(3, A, b)
    self.assertTrue(np.all(C==D)) 
    
    
if __name__ == '__main__':
    unittest.main()
    
    



