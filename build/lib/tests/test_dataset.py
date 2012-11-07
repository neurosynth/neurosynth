import unittest
import base.dataset

class TestDataset(unittest.TestCase):

	def setUp(self):
		""" Create a new Dataset """
		self.dataset = Dataset('fixme.txt')

	
	def test_empty_dataset_initializes(self):
		self.assertEqual(self.dataset.name, 'test Dataset')
		self.assertIsNotNone(self.dataset.volume)
		self.assertIsNone(self.dataset.image_table)
		self.assertIsNone(self.dataset.feature_table)

	def test_feature_table_initializes_from_txt(self):
		filename = 'data/test_feature_table.txt'
		tt = FeatureTable(self.dataset, filename, 'testFeatures')
		self.assertEqual(tt.name, 'testFeatures')
		self.assertEqual(tt.data.shape, (10,10))
		self.assertEqual(tt.Feature_names, ['bah','bo','bi'])
		self.assertEqual(tt.ids, [23, 31, 45])
		
	def test_feature_table_study_selection(self):
		filename = 'path.to.txt'
		tt = FeatureTable('testFeatures', filename)
		studies = tt.get_studies('Feature1', threshold=0.05, func='max')
		self.assertEqual(studies, [234,361,45])
		studies = tt.get_studies('Feature1', threshold=0.05, func='min')
		self.assertEqual(studies, [234,361,45])
		studies = tt.get_studies('Feature1', threshold=0.05, func='sum')
		self.assertEqual(studies, [234,361,45])
		
	
		
suite = unittest.TestLoader().loadTestsFromTestCase(TestDataset)