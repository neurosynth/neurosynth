import unittest
import numpy as np

from neurosynth.analysis import *

from neurosynth.tests.utils import get_test_dataset


class TestAnalysis(unittest.TestCase):

    def setUp(self):
        """ Create a new Dataset and add features. """
        self.dataset = get_test_dataset()

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
