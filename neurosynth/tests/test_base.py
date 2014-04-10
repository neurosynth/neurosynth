import unittest
import numpy as np

import tempfile
import os
import shutil

from neurosynth.tests.utils import get_test_dataset, get_test_data_path
from neurosynth.analysis import meta
from neurosynth.base.dataset import Dataset, ImageTable
from neurosynth.base import imageutils
import json


class TestBase(unittest.TestCase):

    def setUp(self):
        """ Create a new Dataset and add features. """
        self.dataset = get_test_dataset()
        self.skipped_tests = []

    def test_dataset_save_and_load(self):
        # smoke test of saving and loading
        t = tempfile.mktemp()
        self.dataset.save(t, keep_mappables=True)
        self.assertTrue(os.path.exists(t))
        dataset = Dataset.load(t)
        self.assertIsNotNone(dataset)
        self.assertIsNotNone(dataset.mappables)
        self.assertEqual(len(dataset.mappables), 5)
        # Now with the mappables deleted
        dataset.save(t)
        self.assertTrue(os.path.exists(t))
        dataset = Dataset.load(t)
        self.assertEqual(len(dataset.mappables), 0)
        os.unlink(t)

    def test_export_to_json(self):
        js = self.dataset.to_json()
        self.assertTrue(isinstance(js, basestring))
        data = json.loads(js)
        self.assertEqual(len(data['mappables']), 5)

    def test_dataset_initializes(self):
        """ Test whether dataset initializes properly. """
        self.assertIsNotNone(self.dataset.masker)
        self.assertIsNotNone(self.dataset.image_table)
        self.assertEqual(len(self.dataset.mappables), 5)
        self.assertIsNotNone(self.dataset.masker)
        self.assertIsNotNone(self.dataset.r)
        self.assertIsNotNone(self.dataset.mappables[0].data['extra_field'])

    def test_get_mappables(self):
        mappables = self.dataset.get_mappables(['study2', 'study5'])
        self.assertEqual(len(mappables), 2)
        mappables = self.dataset.get_mappables(['study3', 'study4', 'study5'], get_image_data=True)
        self.assertEqual(mappables.shape, (228453, 3))

    def test_image_table_loads(self):
        """ Test ImageTable initialization. """
        self.assertIsNotNone(self.dataset.image_table)
        it = self.dataset.image_table
        self.assertEqual(len(it.ids), 5)
        self.assertIsNotNone(it.masker)
        self.assertIsNotNone(it.r)
        self.assertEqual(it.data.shape, (228453, 5))
        # Add tests for values in table

    def test_feature_table_loads(self):
        """ Test FeatureTable initialization. """
        tt = self.dataset.feature_table
        self.assertIsNotNone(tt)
        self.assertEqual(len(self.dataset.get_feature_names()), 5)
        self.assertEqual(tt.data.shape, (5, 5))
        self.assertEqual(tt.data.columns[3], 'f4')
        self.assertEqual(tt.data.to_dense().iloc[0,0], 0.0003)

    def test_feature_search(self):
        """ Test feature-based Mappable search. Tests both the FeatureTable method
        and the Dataset wrapper. """
        tt = self.dataset.feature_table
        features = tt.search_features(['f*'])
        self.assertEqual(len(features), 4)
        d = self.dataset
        ids = d.get_ids_by_features(['f*'], threshold=0.001)
        self.assertEqual(len(ids), 4)
        img_data = d.get_ids_by_features(
            ['f1', 'f3', 'g1'], 0.001, func=np.max, get_image_data=True)
        self.assertEqual(img_data.shape, (228453, 5))

        # And some smoke-tests:
        # run a meta-analysis
        ma = meta.MetaAnalysis(d, ids)
        # save the results
        tempdir = tempfile.mkdtemp()
        ma.save_results(tempdir + os.path.sep)
        shutil.rmtree(tempdir)

    def test_selection_by_expression(self):
        """ Tests the expression-based search using the lexer/parser. 
        Note that this functionality is optional, so only run the test if 
        ply is available. """
        try:
            import ply.lex as lex
            have_ply = True
        except:
            have_ply = False

        if have_ply:
            ids = self.dataset.get_ids_by_expression("* &~ (g*)", func=np.sum, threshold=0.003)
            self.assertEqual(list(ids), ['study3', 'study5'])
            ids = self.dataset.get_ids_by_expression("f* > 0.005", func=np.mean, threshold=0.0)
            self.assertEqual(list(ids), ['study3'])
            ids = self.dataset.get_ids_by_expression("f* < 0.05", func=np.sum, threshold=0.0)
            self.assertEqual(list(ids), ['study1', 'study2', 'study3', 'study4', 'study5'])
            ids = self.dataset.get_ids_by_expression("f* | g*", func=np.mean, threshold=0.003)
            self.assertEqual(list(ids), ['study1', 'study2', 'study3', 'study4'])
            ids = self.dataset.get_ids_by_expression("(f* & g*)", func=np.sum, threshold=0.001)
            self.assertEqual(list(ids), ['study1', 'study4'])
            try:
                os.unlink('lextab.py')
                os.unlink('parser.out')
                os.unlink('parsetab.py')
            except: pass

    def test_selection_by_mask(self):
        """ Test mask-based Mappable selection.
        Only one peak in the test dataset (in study5) should be within the sgACC. """
        ids = self.dataset.get_ids_by_mask(
            get_test_data_path() + 'sgacc_mask.nii.gz')
        self.assertEquals(len(ids), 1)
        self.assertEquals('study5', ids[0])

    def test_unmask(self):
        """ Test unmasking on 1d and 2d vectors (going back to 3d and 4d)

        TODO: test directly on Masker class and its functions, and on
        some smaller example data.  But then it should get into a separate
        TestCase to not 'reload' the same Dataset.
        So for now let's just reuse loaded Dataset and provide
        rudimentary testing
        """
        dataset = self.dataset
        ids = dataset.get_ids_by_mask(
            get_test_data_path() + 'sgacc_mask.nii.gz')
        nvoxels = dataset.masker.in_mask[0].shape[0]

        nvols = 2
        data2d = np.arange(nvoxels * nvols).reshape((nvoxels, -1))

        data2d_unmasked_separately = [
            dataset.masker.unmask(data2d[:, i]) for i in xrange(nvols)]
        data2d_unmasked = dataset.masker.unmask(data2d)
        self.assertEqual(data2d_unmasked.shape,
                         data2d_unmasked_separately[0].shape + (nvols,))
        # and check corresponding volumes
        for i in xrange(nvols):
            self.assertTrue(np.all(data2d_unmasked[
                            ..., i] == data2d_unmasked_separately[i]))

    def test_selection_by_peaks(self):
        """ Test peak-based Mappable selection. """
        ids = self.dataset.get_ids_by_peaks(np.array(
            [[3, 30, -9]]))  # Test with numpy array
        ids2 = self.dataset.get_ids_by_peaks(
            [[3, 30, -9]])  # Test with list of lists
        self.assertEquals(ids, ids2)
        self.assertEquals(len(ids), 1)
        self.assertEquals('study5', ids[0])
        # Repeat with list of lists

    def test_get_feature_counts(self):
        # If we set threshold too high -- nothing should get through and
        # all should be 0s
        feature_counts = self.dataset.get_feature_counts(threshold=1.)
        self.assertEqual(feature_counts,
                         dict((f, 0) for f in self.dataset.get_feature_names()))

        feature_counts = self.dataset.get_feature_counts()
        # all should have some non-0 loading with default threshold,
        # otherwise what is the point of having them?
        for f, c in feature_counts.items():
            self.assertGreater(c, 0, "feature %s has no hits" % f)
        # and should be equal to the ones computed directly (we do not do
        # any fancy queries atm), assumes default threshold of 0.001
        feature_counts_ = dict([(feature, np.sum(self.dataset.feature_table.data.to_dense().ix[:, col] > 0.001))
                               for col, feature in enumerate(self.dataset.feature_table.feature_names)])
        self.assertEqual(feature_counts, feature_counts_)

    def test_get_feature_data(self):
        """ Test retrieval of Mappable x feature data. """
        feature_data = self.dataset.get_feature_data(ids=['study3', 'study1'])
        self.assertEqual(feature_data.shape, (2,5))
        feature_data = self.dataset.get_feature_data(features=['f1', 'f4', 'g1'], dense=True)
        self.assertEqual(feature_data.shape, (5,3))

    def test_get_image_data(self):
        """ Test retrieval of voxel x Mappable data. """
        image_data = self.dataset.get_image_data(voxels=range(2000,3000))
        self.assertEquals(image_data.shape, (1000,5))
        image_data = self.dataset.get_image_data(ids=['study1','study2','study5'], dense=True)
        self.assertEquals(image_data.shape, (228453,3))
        # Or directly through the image table
        image_data = self.dataset.image_table.get_image_data(ids=['study1','study2','study5'], dense=False)
        self.assertEquals(image_data.shape, (228453,3))

    def test_trim_image_table(self):
        self.dataset.image_table.trim(['study1', 'study2', 'study3'])
        self.assertEqual(self.dataset.get_image_data().shape, (228453, 3))

    def test_either_dataset_or_manual_image_vars(self):
        with self.assertRaises(AssertionError):
            image_table = ImageTable()

    def test_img_to_json(self):
        path = get_test_data_path() + 'sgacc_mask.nii.gz'
        json_data = imageutils.img_to_json(path)
        data = json.loads(json_data)
        self.assertEqual(data['max'], 1)
        self.assertEqual(data['dims'], [91, 109, 91])
        self.assertEqual(data['values'], [1.0])
        self.assertEqual(len(data['indices'][0]), 1142)

    def test_get_features_by_ids(self):
        features = self.dataset.feature_table.get_features_by_ids(['study1','study3'], threshold=0.01)
        self.assertEquals(len(features), 3)
        features = self.dataset.feature_table.get_features_by_ids(['study2','study5'], func=np.sum, 
            threshold=0.0, get_weights=True)
        self.assertEquals(len(features), 5)
        self.assertEqual(features['f3'], 0.01)

    def test_get_feature_names(self):
        features = self.dataset.get_feature_names()
        self.assertEquals(len(features), 5)
        self.assertEquals(features[2], 'f3')
        features = self.dataset.get_feature_names(['g1', 'f4', 'f2'])
        self.assertEquals(features, ['f2', 'f4', 'g1'])

    def test_grid_creation(self):
        mask = self.dataset.masker.volume
        # Test with mask
        grid = imageutils.create_grid(image=mask, scale=4, mask=mask)
        self.assertEquals(grid.shape, (91, 109, 91))
        self.assertEquals(len(np.unique(grid.get_data())), 4359)
        # Test without mask
        grid = imageutils.create_grid(image=mask, scale=4)
        self.assertGreater(len(np.unique(grid.get_data())), 4359)

suite = unittest.TestLoader().loadTestsFromTestCase(TestBase)

if __name__ == '__main__':
    unittest.main()
