import unittest
import numpy as np
import ttmpy as tp

class TestTTM(unittest.TestCase):
  
  def test_ttm_mode1(self):
    A = np.arange(3*2*4, dtype=np.float64).reshape(3,2,4)
    B = np.arange(5*3, dtype=np.float64).reshape(5,3)
    D = np.einsum("ijk,xi->xjk", A, B)
    C = tp.ttm(1, A, B)
    self.assertTrue(np.all(C==D))
 
  def test_ttm_mode2(self):
    A = np.arange(3*2*4, dtype=np.float64).reshape(3,2,4)
    B = np.arange(5*2, dtype=np.float64).reshape(5,2)
    D = np.einsum("ijk,xj->ixk", A, B)
    C = tp.ttm(2, A, B)
    self.assertTrue(np.all(C==D))


  def test_ttm_mode3(self):
    A = np.arange(3*2*4, dtype=np.float64).reshape(3,2,4)
    B = np.arange(5*4, dtype=np.float64).reshape(5,4)
    D = np.einsum("ijk,xk->ijx", A, B)
    C = tp.ttm(3, A, B)
    self.assertTrue(np.all(C==D)) 



class TestTTMs(unittest.TestCase): 
  
  def test_ttms_mode1(self):
    A  = np.arange(3*2*4*5, dtype=np.float64).reshape(3, 2, 4, 5)
    B = [np.arange(2*6, dtype=np.float64).reshape(6,2), np.arange(4*7, dtype=np.float64).reshape(7,4), np.arange(5*9, dtype=np.float64).reshape(9,5)]
    
    #print(A.shape)
    #print(B[0].shape)
    #print(B[1].shape)
    #print(B[2].shape)
    
    D  = np.einsum("ijkl,yj->iykl", A, B[0])
    D  = np.einsum("ijkl,yk->ijyl", D, B[1])
    D  = np.einsum("ijkl,yl->ijky", D, B[2])   
    
    for order in ["forward", "backward", "optimal"] : #, "backward" , "optimal"
      C = tp.ttms(1, A, B, order)
      self.assertTrue(np.all(C==D))
      
  
 
  def test_ttms_mode2(self):
    A  = np.arange(3*2*4*5, dtype=np.float64).reshape(3, 2, 4, 5)
    B = [np.arange(3*6, dtype=np.float64).reshape(6,3), np.arange(4*7, dtype=np.float64).reshape(7,4), np.arange(5*9, dtype=np.float64).reshape(9,5)]

    D  = np.einsum("ijkl,yi->yjkl", A, B[0])
    D  = np.einsum("ijkl,yk->ijyl", D, B[1])
    D  = np.einsum("ijkl,yl->ijky", D, B[2])   

    for order in ["forward", "backward","optimal"] : # , "backward", 
      C = tp.ttms(2, A, B, order)
      self.assertTrue(np.all(C==D))  


  def test_ttms_mode3(self):
    A  = np.arange(3*2*4*5, dtype=np.float64).reshape(3, 2, 4, 5)
    B = [np.arange(3*6, dtype=np.float64).reshape(6,3), np.arange(2*7, dtype=np.float64).reshape(7,2), np.arange(5*9, dtype=np.float64).reshape(9,5)]
    
    D  = np.einsum("ijkl,yi->yjkl", A, B[0])
    D  = np.einsum("ijkl,yj->iykl", D, B[1])
    D  = np.einsum("ijkl,yl->ijky", D, B[2])

    for order in ["forward", "backward","optimal"] : # , "backward", "optimal"
      C = tp.ttms(3, A, B, order)
      self.assertTrue(np.all(C==D))  


  def test_ttms_mode4(self):
    A  = np.arange(3*2*4*5, dtype=np.float64).reshape(3, 2, 4, 5)
    B = [np.arange(3*6, dtype=np.float64).reshape(6,3), np.arange(2*7, dtype=np.float64).reshape(7,2), np.arange(4*9, dtype=np.float64).reshape(9,4)]
    
    D  = np.einsum("ijkl,yi->yjkl", A, B[0])
    D  = np.einsum("ijkl,yj->iykl", D, B[1])
    D  = np.einsum("ijkl,yk->ijyl", D, B[2])
    
    for order in ["forward", "backward","optimal"] : # , "backward", "optimal"
      C = tp.ttms(4, A, B, order)
      self.assertTrue(np.all(C==D)) 

    
if __name__ == '__main__':
    unittest.main()
    
    



