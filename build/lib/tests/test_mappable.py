import unittest
from neurosynth.base import mappable

class TestMappable(unittest.TestCase):
	
	def setUp():
		pass
		
	def test_article_is_mappable(self):
		article = mappable.Article()
		assert(self.article.name)


suite = unittest.TestLoader().loadTestsFromTestCase(TestMappable)