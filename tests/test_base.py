import unittest
import numpy as np

import tempfile, os, shutil

from utils import get_test_dataset, get_test_data_path

from neurosynth.analysis import meta

class TestBase(unittest.TestCase):

  def setUp(self):
    """ Create a new Dataset and add features. """
    self.dataset = get_test_dataset()

  def test_dataset_initializes(self):
    """ Test whether dataset initializes properly. """
    self.assertIsNotNone(self.dataset.volume)
    self.assertIsNotNone(self.dataset.image_table)
    self.assertEqual(len(self.dataset.mappables), 5)
    self.assertIsNotNone(self.dataset.volume)
    self.assertIsNotNone(self.dataset.r)
    self.assertIsNotNone(self.dataset.mappables[0].data['extra_field'])

  def test_image_table_loads(self):
    """ Test ImageTable initialization. """
    self.assertIsNotNone(self.dataset.image_table)
    it = self.dataset.image_table
    self.assertEqual(len(it.ids), 5)
    self.assertIsNotNone(it.volume)
    self.assertIsNotNone(it.r)
    self.assertEqual(it.data.shape, (228453, 5))
    # Add tests for values in table

  def test_feature_table_loads(self):
    """ Test FeatureTable initialization. """
    tt = self.dataset.feature_table
    self.assertIsNotNone(tt)
    self.assertEqual(len(self.dataset.get_feature_names()), 5)
    self.assertEqual(tt.data.shape, (5,5))
    self.assertEqual(tt.feature_names[3], 'f4')
    self.assertEqual(tt.data[0,0], 0.0003)

  def test_feature_search(self):
    """ Test feature-based Mappable search. Tests both the FeatureTable method
    and the Dataset wrapper. """
    tt = self.dataset.feature_table
    features = tt.search_features(['f*'])
    self.assertEqual(len(features), 4)
    d = self.dataset
    ids = d.get_ids_by_features(['f*'], threshold=0.001)
    self.assertEqual(len(ids), 4)
    img_data = d.get_ids_by_features(['f1', 'f3', 'g1'], 0.001, func='max', get_image_data=True)
    self.assertEqual(img_data.shape, (228453, 5))

    # And some smoke-tests:
    # run a meta-analysis
    ma = meta.MetaAnalysis(d, ids)
    # save the results
    tempdir = tempfile.mkdtemp()
    ma.save_results(tempdir + os.path.sep)
    shutil.rmtree(tempdir)

  def test_selection_by_mask(self):
    """ Test mask-based Mappable selection.
    Only one peak in the test dataset (in study5) should be within the sgACC. """
    ids = self.dataset.get_ids_by_mask(
        get_test_data_path() + 'sgacc_mask.nii.gz')
    self.assertEquals(len(ids), 1)
    self.assertEquals('study5', ids[0])

  def test_selection_by_peaks(self):
    """ Test peak-based Mappable selection. """
    ids = self.dataset.get_ids_by_peaks(np.array([[3, 30, -9]]))
    self.assertEquals(len(ids), 1)
    self.assertEquals('study5', ids[0])

  # def test_invalid_coordinates_ignored(self):
    """ Test dataset contains 3 valid coordinates and one outside mask. But this won't work
    right now because invalid voxels are only excluded at the image mapping stage, and are
    not validated at the Mappable creation stage. Not sure whether it's a good idea to
    implement that early filter... """
    # ind = self.dataset.image_table.ids.index('study1')
    # self.assertEqual(len(self.dataset.mappables[ind].peaks), 3)


suite = unittest.TestLoader().loadTestsFromTestCase(TestBase)

if __name__ == '__main__':
    unittest.main()


