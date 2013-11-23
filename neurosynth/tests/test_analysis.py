import unittest
import numpy as np

from neurosynth.analysis import classify
from neurosynth.analysis import reduce
from neurosynth.tests.utils import get_test_dataset, get_test_data_path



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
        """ Test averaging within region labels in a mask. """
        filename = get_test_data_path() + 'sgacc_mask.nii.gz'
        avg_vox = reduce.average_within_regions(self.dataset, filename)
        n_studies = self.dataset.image_table.data.shape[1]
        self.assertEqual(n_studies, avg_vox.shape[1])
        self.assertGreater(avg_vox.sum(), 0.05)

    def test_get_random_voxels(self):
        """ Test random voxel retrieval. """
        n_vox = 100
        rand_vox = reduce.get_random_voxels(self.dataset, n_vox)
        n_studies = self.dataset.image_table.data.shape[1]
        self.assertEqual(rand_vox.shape, (n_vox, n_studies))

    def test_classify_regions(self):
        # score = classify.classify_regions(self.dataset,['data/regions/medial_motor.nii.gz', 'data/regions/vmPFC.nii.gz'], cross_val='4-Fold')['score']
        # self.assertEqual(score, 0.84600313479623823)

        # score = classify.classify_regions(self.dataset,['data/regions/medial_motor.nii.gz', 'data/regions/vmPFC.nii.gz'])['score']
        # self.assertEqual(score, 0.87813479623824453)

#     result = decode.classify_regions(self.dataset, masks=[get_test_data_path() + 'sgacc_mask.nii.gz'])
#     self.assertEquals(len(result['features']), 525)
#     self.assertEquals(result['scores'].shape = (3,525))
        pass

suite = unittest.TestLoader().loadTestsFromTestCase(TestAnalysis)

if __name__ == '__main__':
    unittest.main()
