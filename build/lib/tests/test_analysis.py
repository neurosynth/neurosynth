import unittest
from neurosynth.base.dataset import Dataset
from neurosynth.analysis import *
import numpy as np

class TestAnalysis(unittest.TestCase):

  def setUp(self):
    """ Create a new Dataset and add features. """
    self.dataset = Dataset('data/test_dataset.txt')
    self.dataset.add_features('data/test_features.txt')
  
  def test_meta_analysis(self):
    """ Test full meta-analysis stream. """
    pass

  def test_decoder(self):
    pass

  def test_coactivation(self):
    """ Test seed-based coactivation. """ 
    pass

  def test_roi_averaging(self):
    pass

  def test_get_random_voxels(self):
    pass

    
suite = unittest.TestLoader().loadTestsFromTestCase(TestAnalysis)

if __name__ == '__main__':
    unittest.main()

    