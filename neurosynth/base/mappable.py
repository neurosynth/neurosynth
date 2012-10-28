""" Contains base Mappable object and all objects that inherit from Mappable.
		A Mappable object is defined by the presence of one or more Peaks that can
		be meaningfully represented as a spatial image. """

import imageutils
import numpy as np
import transformations
import json

class Mappable(object):
	
	def __init__(self, data, transform=True):
		try:
			self.data = data
			self.id = data['id']
			self.space = data['space']
		except:
			print "Error: missing ID and/or space fields. Please check source."
			exit()
		
		# Loop through rows and set coordinates
		peaks = np.zeros((len(data['peaks']), 3), dtype=int)
		for i,f in enumerate(data['peaks']):
			peaks[i,] = [float(j) for j in f[0:3]]
		
		# Convert from talairach to MNI using Lancaster transform
		if transform and self.space == 'TAL':
			peaks = transformations.tal_to_mni(peaks)
			
		# Convert from XYZ coordinates to matrix indices, saving both
		self.xyz = peaks
		self.peaks = transformations.xyz_to_mat(peaks)
		# self.map_peaks
		
	def map_peaks(self):
		"""Map all Peaks to a new Nifti1Image."""
		# if len(self.peaks) == 0: return
		return imageutils.map_peaks_to_image(self.peaks)

	def to_json(self, filename=None):
		json_string = json.dumps({'id':self.id, 'space':self.space, 'peaks':self.xyz.tolist()})
		if filename is not None:
			open(filename, 'w').write(json_string)
		else: return json_string

	def to_s(self):
		s = "Mappable ID: %s\n" % self.id
		s += "Nominal space: %s\n" % self.space
		s += "Num. of peaks: %s\n\n" % str(self.peaks.shape[0])
		s += "Peaks:\n\n"
		for p in self.xyz.tolist():
			s += "\t%s\n" % str(p)
		return s

		
class Article(Mappable):
	
	def __init__(self, data, transform=True):
		super(Article, self).__init__(data, transform)

		
class Table(Mappable):
	
	def __init__(self, data, transform=True, article=None):
		self.article = article
		super(Table, self).__init__(data, transform)

		