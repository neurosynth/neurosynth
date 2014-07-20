import unittest
import numpy as np
import tempfile
import os
import shutil
from neurosynth.analysis import classify
from neurosynth.analysis import cluster
from neurosynth.analysis import reduce
from neurosynth.analysis import meta
from neurosynth.analysis import stats
from neurosynth.tests.utils import get_test_dataset, get_test_data_path
from numpy.testing import assert_array_almost_equal


class TestAnalysis(unittest.TestCase):

    def setUp(self):
        """ Create a new Dataset and add features. """
        self.dataset = get_test_dataset()

    def test_meta_analysis(self):
        """ Test full meta-analysis stream. """
        # run a meta-analysis
        ids = ['study1', 'study3']
        ma = meta.MetaAnalysis(self.dataset, ids)
        # save the results
        tempdir = tempfile.mkdtemp()
        ma.save_results(tempdir + os.path.sep, prefix='test')
        from glob import glob
        files = glob(tempdir + os.path.sep + "test_*.nii.gz")
        self.assertEquals(len(files), 9)
        shutil.rmtree(tempdir)

    def test_decoder(self):
        pass

    def test_coactivation(self):
        """ Test seed-based coactivation. """
        pass

    def test_roi_averaging(self):
        """ Test averaging within region labels in a mask. """
        filename = get_test_data_path() + 'sgacc_mask.nii.gz'
        regions = self.dataset.masker.mask(filename, in_global_mask=True)
        avg_vox = reduce.average_within_regions(self.dataset, regions)
        n_studies = self.dataset.image_table.data.shape[1]
        self.assertEqual(n_studies, avg_vox.shape[1])
        self.assertGreater(avg_vox.sum(), 0.05)

    def test_get_random_voxels(self):
        """ Test random voxel retrieval. """
        n_vox = 100
        rand_vox = reduce.get_random_voxels(self.dataset, n_vox)
        n_studies = self.dataset.image_table.data.shape[1]
        self.assertEqual(rand_vox.shape, (n_vox, n_studies))

    def test_apply_grid_to_image(self):
        data, grid = reduce.apply_grid(self.dataset, scale=6)
        self.assertEquals(data.shape, (1435, 5))
        sums = np.sum(data, 0)
        self.assertGreater(sums[2], sums[3])
        self.assertGreater(sums[4], sums[0])

    def test_two_way_chi_sq(self):
        p = stats.two_way(np.array([[42, 32], [60, 81]])[None,:,:])
        # Test value verified using several different packages
        assert_array_almost_equal(p, 0.04753082)

    # def test_clustering(self):
    #     clstr = cluster.Clusterer(self.dataset, grid_scale=20)
    #     clstr.cluster(algorithm='ward', n_clusters=3)
    #     t = 'ClusterImages/Cluster_k3.nii.gz'
    #     self.assertTrue(os.path.exists(t))
    #     os.unlink(t)
    #     os.rmdir('ClusterImages')


suite = unittest.TestLoader().loadTestsFromTestCase(TestAnalysis)

if __name__ == '__main__':
    unittest.main()
