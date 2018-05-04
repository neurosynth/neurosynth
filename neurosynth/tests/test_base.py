import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import shutil
from glob import glob
from neurosynth.tests.utils import (get_test_dataset, get_test_data_path,
                                    get_resource_path)
from neurosynth.base.dataset import Dataset
from neurosynth.base import imageutils
from neurosynth.base.mask import Masker
import neurosynth as ns


class TestBase(unittest.TestCase):

    def setUp(self):
        """ Create a new Dataset and add features. """
        self.dataset = get_test_dataset()
        self.real_dataset = get_test_dataset(prefix='test_real')

    def test_dataset_download(self):
        tmpdir = tempfile.mkdtemp()
        url = 'https://raw.githubusercontent.com/neurosynth/neurosynth/master/neurosynth/tests/data/test_data.tar.gz'
        ns.base.dataset.download(tmpdir, url=url, unpack=True)
        files = glob(os.path.join(tmpdir, '*.txt'))
        self.assertEqual(len(files), 2)
        shutil.rmtree(tmpdir)

    def test_dataset_save_and_load(self):
        # smoke test of saving and loading
        t = tempfile.mktemp()
        self.dataset.save(t)
        self.assertTrue(os.path.exists(t))
        dataset = Dataset.load(t)
        self.assertIsNotNone(dataset)
        self.assertEqual(len(dataset.image_table.ids), 5)
        os.unlink(t)

    def test_dataset_initializes(self):
        """ Test whether dataset initializes properly. """
        Dataset(get_test_data_path() + 'test_dataset.txt',
                get_test_data_path() + 'test_features.txt')
        self.assertIsNotNone(self.dataset.masker)
        self.assertIsNotNone(self.dataset.image_table)
        self.assertEqual(len(self.dataset.image_table.ids), 5)
        self.assertIsNotNone(self.dataset.masker)
        self.assertIsNotNone(self.dataset.r)
        self.assertIsNotNone(
            self.dataset.activations['extra_field'].iloc[2], 'field')

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
        self.assertEqual(tt.data.to_dense().iloc[0, 0], 0.0003)

    def test_feature_addition(self):
        """ Add feature data from multiple sources to FeatureTable. """
        d = self.dataset
        new_data = pd.DataFrame(np.array([
            [0.001, 0.1, 0.003],
            [0.02, 0.0008, 0.0003],
            [0.001, 0.002, 0.05]]),
            index=['study1', 'study4', 'study6'], columns=['g1', 'g2', 'g3'])
        # Replace all features
        d.add_features(new_data, append=False)
        self.assertEqual(d.feature_table.data.shape, (3, 3))
        self.assertAlmostEqual(d.feature_table.data['g2']['study4'], 0.0008)
        # Outer merge and ignore duplicate features
        d = get_test_dataset()
        d.add_features(new_data)
        self.assertEqual(d.feature_table.data.shape, (6, 7))
        self.assertEqual(d.feature_table.data['g2']['study2'], 0.0)
        self.assertEqual(d.feature_table.data['g1']['study1'], 0.02)
        # Outer merge but overwrite old features with new ones
        d = get_test_dataset()
        d.add_features(new_data, duplicates='replace')
        self.assertEqual(d.feature_table.data.shape, (6, 7))
        self.assertEqual(d.feature_table.data['g1']['study1'], 0.001)
        # Left join (i.e., keep only old rows) and rename conflicting features
        d = get_test_dataset()
        d.add_features(new_data, merge='left', duplicates='merge')
        self.assertEqual(d.feature_table.data.shape, (5, 8))
        self.assertEqual(d.feature_table.data['g1_y']['study1'], 0.001)
        # Apply threshold
        d = get_test_dataset()
        d.add_features(new_data, min_studies=2, threshold=0.003)
        self.assertEqual(d.feature_table.data.shape, (6, 6))

    def test_selection_by_features(self):
        """ Test feature-based Mappable search. Tests both the FeatureTable method
        and the Dataset wrapper. """
        tt = self.dataset.feature_table
        features = tt.search_features(['f*'])
        self.assertEqual(len(features), 4)
        d = self.dataset
        ids = d.get_studies(features='f4', frequency_threshold=0.001)
        self.assertEqual(list(ids), ['study5'])
        ids = d.get_studies(features=['f*'], frequency_threshold=0.001)
        self.assertEqual(len(ids), 4)
        img_data = d.get_studies(
            features=['f1', 'f3', 'g1'], activation_threshold=0.001,
            func=np.max, return_type='data')
        self.assertEqual(img_data.shape, (228453, 5))
        d.feature_table.data.columns = [
            'f1', 'f2', 'my ngram', 'my ngram reprise', 'g1']
        ids = d.get_studies('my ngram reprise', frequency_threshold=0.001)
        self.assertEqual(list(ids), ['study5'])

    def test_selection_by_expression(self):
        """ Tests the expression-based search using the lexer/parser. This
        functionality is optional, so only run the test if ply is available.
        """
        try:
            import ply.lex as lex
            have_ply = True
        except:
            have_ply = False

        if have_ply:
            ids = self.dataset.get_studies(expression="* &~ (g*)", func=np.sum,
                                           frequency_threshold=0.003)
            self.assertEqual(sorted(ids), ['study3', 'study5'])
            ids = self.dataset.get_studies(
                expression="f* > 0.005", func=np.mean, frequency_threshold=0.0)
            self.assertEqual(ids, ['study3'])
            ids = self.dataset.get_studies(expression="f* < 0.05", func=np.sum,
                                           frequency_threshold=0.0)
            self.assertEqual(sorted(ids), ['study1', 'study2', 'study3',
                                           'study4', 'study5'])
            ids = self.dataset.get_studies(expression="f* | g*", func=np.mean,
                                           frequency_threshold=0.003)
            self.assertEqual(sorted(ids), ['study1', 'study2', 'study3',
                                           'study4'])
            ids = self.dataset.get_studies(expression="(f* & g*)", func=np.sum,
                                           frequency_threshold=0.001)
            self.assertEqual(sorted(ids), ['study1', 'study4'])
            # test N-gram feature handling
            self.dataset.feature_table.data.columns = ['f1', 'f2', 'my ngram',
                                                       'my ngram reprise',
                                                       'g1']
            ids = self.dataset.get_studies(expression="my ngram reprise")
            self.assertEqual(ids, ['study5'])
            ids = self.dataset.get_studies(expression="my ngram*",
                                           frequency_threshold=0.01)
            self.assertEqual(sorted(ids), ['study1', 'study4', 'study5'])
            try:
                os.unlink('lextab.py')
                os.unlink('parser.out')
                os.unlink('parsetab.py')
            except:
                pass

    def test_selection_by_mask(self):
        """ Test mask-based Mappable selection.
        Only one peak in the test dataset (in study5) should be within the
        sgACC. """
        ids = self.dataset.get_studies(mask=get_test_data_path() +
                                       'sgacc_mask.nii.gz')
        self.assertEquals(len(ids), 1)
        self.assertEquals('study5', ids[0])

    def test_selection_by_peaks(self):
        """ Test peak-based Mappable selection. """
        ids = self.real_dataset.get_studies(peaks=[[0, 20, 40]])
        self.assertEquals(len(ids), 1)
        self.assertTrue(9106283 in ids)
        peaks = np.array([[0, 20, 40], [-32, 22, 12]])
        ids = self.real_dataset.get_studies(peaks=peaks, r=8)
        self.assertEquals(len(ids), 11)

    def test_selection_by_multiple_criteria(self):
        ids = self.dataset.get_studies(
            peaks=[[3, 30, -9]], r=40, expression="f* < 0.013", func=np.sum,
            frequency_threshold=0.0)
        self.assertEquals(sorted(ids), ['study2', 'study5'])

    def test_unmask(self):
        """ Test unmasking on 1d and 2d vectors (going back to 3d and 4d)

        TODO: test directly on Masker class and its functions, and on
        some smaller example data.  But then it should get into a separate
        TestCase to not 'reload' the same Dataset.
        So for now let's just reuse loaded Dataset and provide
        rudimentary testing
        """
        dataset = self.dataset
        ids = dataset.get_studies(
            mask=get_test_data_path() + 'sgacc_mask.nii.gz')
        nvoxels = dataset.masker.n_vox_in_vol
        nvols = 2
        data2d = np.arange(nvoxels * nvols).reshape((nvoxels, -1))
        data2d_unmasked = dataset.masker.unmask(data2d, output='array')
        self.assertEqual(data2d_unmasked.shape, (91, 109, 91, 2))
        data2d_unmasked = dataset.masker.unmask(data2d, output='image')
        self.assertEqual(data2d_unmasked.shape, (91, 109, 91, 2))
        self.assertTrue(hasattr(data2d_unmasked, 'get_data'))

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
        # Retrieve dense
        feature_data = self.dataset.get_feature_data(
            features=['f1', 'f4', 'g1'])
        self.assertEqual(feature_data.shape, (5, 3))
        # Retrieve sparse
        feature_data = self.dataset.get_feature_data(
            ids=['study3', 'study1'], dense=False)
        self.assertEqual(feature_data.shape, (2, 5))
        # Skip this for now; behaves inconsistently across pandas versions
        # self.assertEqual(feature_data.iloc[0,1], 0.02)
        self.assertEqual(feature_data['f3']['study1'], 0.012)

    def test_get_image_data(self):
        """ Test retrieval of voxel x Mappable data. """
        image_data = self.dataset.get_image_data(voxels=range(2000, 3000))
        self.assertEquals(image_data.shape, (1000, 5))
        image_data = self.dataset.get_image_data(
            ids=['study1', 'study2', 'study5'], dense=True)
        self.assertEquals(image_data.shape, (228453, 3))
        # Or directly through the image table
        image_data = self.dataset.image_table.get_image_data(
            ids=['study1', 'study2', 'study5'], dense=False)
        self.assertEquals(image_data.shape, (228453, 3))

    def test_trim_image_table(self):
        self.dataset.image_table.trim(['study1', 'study2', 'study3'])
        self.assertEqual(self.dataset.get_image_data().shape, (228453, 3))

    def test_get_features_by_ids(self):
        features = self.dataset.feature_table.get_features_by_ids(
            ['study1', 'study3'], threshold=0.01)
        self.assertEquals(len(features), 3)
        features = self.dataset.feature_table.get_features_by_ids(
            ['study2', 'study5'], func=np.sum, threshold=0.0, get_weights=True)
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
        grid = imageutils.create_grid(image=mask, scale=4)
        self.assertEquals(grid.shape, (91, 109, 91))
        self.assertEquals(len(np.unique(grid.get_data())), 4359)
        # Test without mask
        grid = imageutils.create_grid(image=mask, scale=4, apply_mask=False)
        self.assertGreater(len(np.unique(grid.get_data())), 4359)


class TestMasker(unittest.TestCase):

    def setUp(self):
        """ Create a new Dataset and add features. """
        maskfile = get_resource_path() + 'MNI152_T1_2mm_brain.nii.gz'
        self.masker = Masker(maskfile)

    def test_add_and_remove_masks(self):
        self.masker.add(get_test_data_path() + 'sgacc_mask.nii.gz')
        self.masker.add(
            {'motor': get_test_data_path() + 'medial_motor.nii.gz'})
        self.assertEqual(len(self.masker.layers), 2)
        self.assertEqual(len(self.masker.stack), 2)
        self.assertEqual(
            set(self.masker.layers.keys()), set(['layer_0', 'motor']))
        self.assertEqual(np.sum(self.masker.layers['motor']), 1419)
        self.masker.remove('motor')
        self.assertEqual(len(self.masker.layers), 1)
        self.assertEqual(len(self.masker.stack), 1)
        self.masker.add(get_test_data_path() + 'medial_motor.nii.gz')
        self.masker.remove(-1)
        self.assertTrue('layer_0' in self.masker.layers.keys())
        self.assertEqual(len(self.masker.layers), 1)
