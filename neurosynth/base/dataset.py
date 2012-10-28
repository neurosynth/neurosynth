""" A Neurosynth Dataset """
import numpy as np
import nibabel as nb
import mappable
import re
from neurosynth.base import mask, imageutils
import random
import os
from scipy import sparse

class Dataset:
	
	def __init__(self, filename, feature_filename=None, volume=None, r=6, name=''):

		# Instance properties
		self.name = name
		self.mappables = self._mappables_from_txt(filename)
		self.r = r

		# Load the volume into a new Mask
		try:
			if volume is None:
				resource_dir = os.path.join(os.path.dirname(__file__), '../resources')
				volume = os.path.join(resource_dir, 'MNI152_T1_2mm_brain.nii.gz')
			self.volume = mask.Mask(volume)
		except Exception, e:
			print "Error loading volume."
			print e

		# Create supporting tables for images and features
		self.create_image_table()
		if feature_filename is not None:
			self.feature_table = FeatureTable(self, feature_filename)

	
	def _mappables_from_txt(self, filename):
		""" Load mappables from a text file. 

		Args:
			filename: a string pointing to the location of the txt file to read from.
		"""
		print "Loading mappables from %s..." % filename
		data = {}
		c = re.split('[\r\n]+', open(filename).read())
		header = c.pop(0).lower().split('\t')
		# Get indices of mandatory columns
		mandatory_cols = ['x', 'y', 'z', 'id', 'space']
		mc_inds = {}
		try:
			for mc in mandatory_cols: mc_inds[mc] = header.index(mc)
		except Exception, e:
			print "Error: at least one of mandatory columns (x, y, z, id, and space) is missing."
			print e
			return

		for l in c:
			vals = l.split('\t')
			x, y, z, id, space = [vals[mc_inds[mc]] for mc in mandatory_cols]
			if not id in data:
				data[id] = {
					'id': id,
					'space': space,
					'peaks': []
				}
			data[id]['peaks'].append([x,y,z])

		# Initialize all mappables--for now, assume Articles are passed
		print "Converting text to mappables..."
		return [mappable.Article(m) for m in data.values()]


	def create_image_table(self, r=None):
		""" Create and store a new ImageTable instance based on the current Dataset.

		Will generally be called privately, but may be useful as a convenience 
		method in cases where the user wants to re-generate the table with a 
		new smoothing kernel of different radius.

		Args:
			r: An optional integer indicating the radius of the smoothing kernel. 
				By default, this is None, which will keep whatever value is currently
				set in the Dataset instance.
		"""
		print "Creating image table..."
		if r is not None: self.r = r
		self.image_table = ImageTable(self)


	def add_mappables(self, filename=None, mappables=None, remap=True):
		""" Append new Mappable objects to the end of the list. 

		Either a filename or a list of mappables must be passed.

		Args:
			filename: The location of the file to extract new mappables from.
			mappables: A list of Mappable instances to append to the current list.
			remap: Optional boolean indicating whether to regenerate the entire 
				ImageTable after appending the new Mappables.
		"""
		# TODO: (i) it would be more effiicent to only map the new Mappables into 
		# the ImageTable instead of redoing everything. (ii) we should check for 
		# duplicates and prompt whether to overwrite or update in cases where 
		# conflicts occur.
		if filename != None:
			self.mappables.extend(self._mappables_from_txt(filename))
		elif mappables != None:
			self.mappables.extend(mappables)
		if remap:
			self.image_table = create_image_table()


	def delete_mappables(self, ids, remap=True):
		""" Delete specific Mappables from the Dataset.

		Note that 'ids' is a list of unique identifiers of the Mappables (e.g., doi's), 
		and not indices in the current instance's mappables list. 

		Args:
			ids: A list of ids corresponding to the Mappables to delete.
			remap: Optional boolean indicating whether to regenerate the entire 
				ImageTable after deleting undesired Mappables.
		"""
		self.mappables = [m for m in self.mappables if m not in ids]
		if remap: self.image_table = create_image_table()


	def get_mappables(self, ids, get_image_data=False):
		""" Takes a list of unique ids and returns corresponding Mappables.

		Args:
			ids: A list of ids of the mappables to return.
			get_image_data: An optional boolean. When True, returns a voxel x mappable matrix 
				of image data rather than the Mappable instances themselves.
		
		Returns:
			If get_image_data is True, a 2D numpy array of voxels x Mappables. Otherwise, a 
			list of Mappables.
		"""
		if get_image_data:
			return self.get_image_data(ids)
		else:
			return [m for m in self.mappables if m.id in ids]

		
	def get_ids_by_features(self, features, threshold=None, func='sum', get_image_data=False):
		""" A wrapper for FeatureTable.get_ids(). 

		Args:
			features: A list of features to use when selecting Mappables.
			threshold: Optional float between 0 and 1. If passed, the threshold will be used as 
				a cut-off when selecting Mappables.
			func: The function to use when aggregating over the list of features. See 
				documentation in FeatureTable.get_ids() for a full explanation.
			get_image_data: An optional boolean. When True, returns a voxel x mappable matrix 
				of image data rather than the Mappable instances themselves.
		"""
		ids = self.feature_table.get_ids(features, threshold, func)
		return self.get_image_data(ids) if get_image_data else ids

		
	def get_ids_by_mask(self, mask, threshold=0.0, get_image_data=False):
		""" Return all mappable objects that activate within the bounds 
		defined by the mask image. Optional threshold parameter specifies 
		the proportion of voxels within the mask that must be active to 
		warrant inclusion. E.g., if threshold = 0.1, only mappables with 
		> 10% of voxels activated in mask will be returned. """
		mask = nb.load(mask).get_data().astype(bool).ravel()
		mask = self.volume.mask(mask)
		# mask = np.squeeze(mask)
		prop_mask_active = np.dot(self.image_table.T, mask).astype(float) / self.volume.num_vox_in_mask
		ids = np.where(prop_mask_active > threshold)
		return self.get_image_data(ids) if get_image_data else ids


	def get_ids_by_peaks(self, peaks, r=10, threshold=0.0, get_image_data=False):
		""" A wrapper for get_ids_by_mask. Takes a list of xyz 
		coordinates and generates a new Nifti1Image to use as a mask. """
		img = imageutils.map_peaks_to_image(peaks, r, vox_dims=self.volume.vox_dims,
				dims=self.volume.shape, header=self.volume.get_header())
		return self.get_ids_by_mask(img, r, threshold, get_image_data=get_image_data)


	# def get_features_by_mask(self, mask, feature_list=None, func='mean'):
	# 	""" Unimplemented. """
	# 	pass

	# def get_features_by_peaks(self, peaks, r=10, feature_list=None, func='mean'):
	# 	pass


	def add_features(self, filename, description='', validate=False):
		""" Construct a new FeatureTable from file. """
		self.feature_table = FeatureTable(self, filename, description, validate)

	
	def get_image_data(self, ids=None, dense=True):
		""" A convenience wrapper for ImageTable.get_image_data(). """
		return self.image_table.get_image_data(ids, dense=dense)

	def list_features(self):
		""" Returns a list of all current feature names. """
		return self.feature_table.feature_names


	def save(self, filename, keep_mappables=False):
		""" Pickle the Dataset instance to the provided file.

		If keep_mappables = False (default), will delete the Mappable objects 
		themselves before pickling. This will save a good deal of space and 
		is generally advisable once a stable Dataset is created, as the 
		Mappables are rarely used after the ImageTable is generated.
		"""
		if not keep_mappables:
			self.mappables = []
		import cPickle
		cPickle.dump(self, open(filename, 'wb'), -1)


	def to_json(self, filename=None):
		""" Save the Dataset to file in JSON format. 

		This is not recommended, as the resulting file will typically be several 
		GB in size. If no filename is provided, returns the JSON string.
		"""
		import json
		mappables = [m.to_json() for m in self.mappables]
		json_string = json.dumps({'mappables':mappables})
		if filename is not None:
			open(filename, 'w').write(json_string)
		else: return json_string



class ImageTable:
	
	def __init__(self, dataset=None, mappables=None, volume=None, r=6, use_sparse=True):
		""" Initialize a new ImageTable. 

		If a Dataset instance is passed, all inputs are taken from the Dataset.
		Alternatively, a user can manually pass the desired mappables 
		and volume (e.g., in cases where the ImageTable class is being used without a 
		Dataset). Can optionally specify the radius of the sphere used for smoothing (default: 
		6 mm), as well as whether or not to represent the data as a sparse array
		(generally this should be left to True, as these data are quite sparse and 
		computation can often be speeded up by an order of magnitude.)
		"""
		if dataset is not None:
			mappables, volume, r = dataset.mappables, dataset.volume, dataset.r
		self.ids = [m.id for m in mappables]
		self.volume = volume
		self.r = r
		self.data = np.zeros((self.volume.num_vox_in_mask, len(mappables)), dtype=int)
		for i, s in enumerate(mappables):
			print "%s/%s..." % (str(i+1), str(len(mappables)))
			img = imageutils.map_peaks_to_image(s.peaks, r=r, header=self.volume.get_header())
			self.data[:,i] = self.volume.mask(img.get_data())
		if use_sparse: self.data = sparse.csr_matrix(self.data)


	def get_image_data(self, ids=None, dense=True):
		""" Returns image data for specified mappable IDs.

		Selects columns from the voxel x mappable data matrix. Images will be returned in the 
		same format, i.e., each column in the array corresponds to a single mappable.
		If dense is True (default), convert the result to a dense array before returning.
		Note that toarray() can be quite slow when a large subset of data is requested. 
		If ids is None, return all data.
		"""
		if ids is None: return self.data
		idxs = [i for i in range(len(self.ids)) if self.ids[i] in ids]
		result = self.data[:,idxs]
		return result.toarray() if dense else result


	def trim(self, ids):
		""" Trim ImageTable to keep only the passed Mappables. This is a convenience 
		method, and should generally be avoided in favor of non-destructive alternatives 
		that don't require slicing (e.g., matrix multiplication). """
		self.data = self.get_image_data(ids, dense=False)#.tocoo()


	def save_images_to_file(self, ids, outroot='./'):
		""" Reconstructs vectorized images corresponding to the specified Mappable ids
		and saves them to file, prepending with the outroot (default: current directory). """
		pass


	def save(self, filename):
		import cPickle
		cPickle.dump(self, open(filename, 'wb'), -1)
		

	
class FeatureTable:

	""" A FeatureTable instance stores a matrix of mappables x features, along with 
	associated manipulation methods. """
	
	def __init__(self, dataset, filename, description=None, validate=False):
		""" Initialize a new FeatureTable. Takes as input a parent DataSet instance and 
		the name of a file containing feature data. Optionally, can provide a description 
		of the feature set. """
		self.dataset = dataset
		self.load(filename, validate=validate)
		self.description = description

		
	def load(self, filename, validate=False):
		""" Loads FeatureTable data from file. Input must be in 1 of 2 formats:
		(1) A sparse JSON representation (see _parse_json() for details)
		(2) A dense matrix stored as plaintext (see _parse_txt() for details)
		If validate == True, any mappable IDs in the input file that cannot be located 
		in the root Dataset's ImageTable will be silently culled. """
		try:
			self._features_from_json(filename, validate)
		except Exception, e:
			print e
			try: 
				self._features_from_txt(filename, validate)
			except Exception, e:
				print e
				print "Error: %s cannot be parsed." % filename


	def _features_from_json(self, filename, validate=False):
		""" Parses FeatureTable from a sparse JSON representation, where keys are feature 
		names and values are dictionaries of mappable id: weight mappings. E.g., 
			{'language': ['study1': 0.003, 'study2': 0.103]} """
		import json
		json_data = json.loads(open(filename))
		# Find all unique mappable IDs
		unique_ids = set()
		unique_ids = [unique_ids.update(d) for d in json_data.itervalues()]
		# Cull invalid IDs if validation is on
		if validate: unique_ids &= set(self.dataset.image_table.ids)
		# ...
		self.data = data


	def _features_from_txt(self, filename, validate=False):
		""" Parses FeatureTable from a plaintext file that represents a dense matrix, 
		with mappable objects in rows and features in columns. Values in cells reflect the 
		weight of the intersecting feature for the intersecting study. Feature names and 
		mappable IDs should be included as the first column and first row, respectively. """
		data = np.genfromtxt(filename, names=True, dtype=None)
		self.feature_names = list(data.dtype.names[1::])
		self.ids = data[data.dtype.names[0]]
		self.data = data[self.feature_names].view(np.float).reshape(len(data), -1)
		if validate:
			valid_ids = set(self.ids) & set(self.dataset.image_table.ids)
			if len(valid_ids) < len(self.dataset.image_table.ids):
				valid_id_inds = np.in1d(self.ids, list(valid_ids))
				self.data = self.data[valid_id_inds, :]
				self.ids = self.ids[valid_id_inds]
		self.data = sparse.csr_matrix(self.data)


	def get_ids(self, features, threshold=None, func='sum'):
		""" Returns a list of all Mappables in the table that meet the desired feature-based
		criteria. Will most commonly be used to retrieve Mappables that use one or more 
		features with some minimum frequency; e.g.,:
			get_ids(['fear', 'anxiety'], threshold=0.001)
		The func argument can be any numpy function (default: sum). The function will be 
		applied to the list of features and the result compared to the threshold. This can be 
		used to change the meaning of the query in powerful ways. E.g,:
			max: any of the features have to pass threshold (i.e., max > thresh)
			min: all features must each individually pass threshold (i.e., min > thresh)
			sum: the summed weight of all features must pass threshold (i.e., sum > thresh) 
		"""
		if type(features) == str : features = [features]
		features = self.search_features(features)  # Expand wild cards
		feature_indices = np.in1d(np.array(self.feature_names), np.array(features))
		data = self.data.toarray()
		feature_weights = data[:,feature_indices]
		weights = eval("np.%s(tw, 1)" % func, {}, {'np':np, 'tw':feature_weights}) # Safe eval
		return self.ids[weights >= threshold]

	def search_features(self, search):
		''' Returns all features that match any of the elements in the input list. '''
		search = [s.replace('*', '.*') for s in search]
		results = []
		for s in search:
			results.extend([f for f in self.feature_names if re.match(s, f)])
		return results


	def get_ids_by_expression(self, expression):
		""" Use a PEG to parse expression and return mappables. Currently unimplemented. """ 
		# Need to port the old Ruby code!
		pass
