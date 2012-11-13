import numpy as np
from neurosynth.base import imageutils

""" Decoding-related methods """

class Decoder:

  def __init__(self, dataset=None, method='pearson', features=None, images=None, mask=None):
    """ Initialize a new Decoder instance.

    Args:
      dataset: An optional Dataset instance containing features to use in decoding.
      method: The decoding method to use (optional). By default, Pearson correlation.
      features: Optional list of features to use in decoding. If None, use all 
        features found in dataset.
      images: Optional list of image names corresponding to features to use.
        When passed, dataset argument is ignored and all and only the 
        features passed as images are used in decoding.
    """

    if dataset is not None: self.dataset = dataset
    self.method = method.lower()

    # If no mask is passed, use the dataset's.
    if mask is None:
      self.mask = dataset.volume
    else:
      from neurosynth.base import mask
      self.mask = mask.Mask(mask)

    # Load feature data from 
    if images is None:
      self._load_features_from_dataset(features)
    else:
      if dataset is None: raise(Exception, "Error: No features found. You must provide either a Dataset instance or a list of image filenames.") # Fix
      self._load_features_from_images(images)


  def decode(self, filenames, method=None):
    """ Decodes a set of images.

    Args:
      files: A list of filenames of images to decode.
      method: Optional string indicating decoding method to use. If None, use 
        the method set when the Decoder instance was initialized.

    Returns:
      An n_files x n_features numpy array, where each feature is a row and 
      each image is a column. The meaning of the values depends on the 
      decoding method used. """

    if method is None: method = self.method
    imgs_to_decode = imageutils.load_imgs(filenames, self.mask)
    methods = {
      'pearson': self._pearson(imgs_to_decode)
      # 'nb': self._nb(imgs_to_decode),
      # 'pattern': self._pattern_expression(imgs_to_decode)
    }
    return methods[method]


  def set_method(self, method):
    """ Set decoding method. """
    self.method = method


  def load_features(features):
    """ Load features from current Dataset instance or a list of files. """
    from os import path
    # Check if the first element in list is a valid file; if yes, assume 
    # we're dealing with image files, otherwise treat as names of features in 
    # the current Dataset.
    if path.exists(features[0]):
      self._load_features_from_images(features)
    else:
      self._load_features_from_dataset(features)


  def _load_features_from_dataset(self, features=None):
    """ Load feature image data from the current Dataset instance.

    Args:
      features: Optional list of features to use. If None, all features in the 
        current Dataset instance will be used.
    """
    self.feature_names = self.dataset.feature_table.feature_names
    if features is not None:
      self.feature_names = list(set(self.feature_names) & set(features))
    # IMPLEMENT!
    self.feature_images = self.dataset.create_feature_images(self.feature_names) # TODO: check if already exists


  def _load_features_from_images(self, images, names=None):
    """ Load feature image data from image files.

    Args:
      images: A list of image filenames.
      names: An optional list of strings to use as the feature names. Must be
        in the same order as the images.
    """
    if names is not None and len(names) != len(images):
      raise Exception("Lists of feature names and image files must be of same length!")
    self.feature_names = names if names is not None else images
    self.feature_images = imageutils.load_imgs(images, self.mask)


  def _pearson(self, imgs_to_decode):
    """ Decode images using Pearson's r.

    Computes the correlation between each input image and each feature image across 
    voxels.
    """
    # Clunky; does numpy have a built-in way to correlate two matrices?
    nr, nc = imgs_to_decode.shape[1], len(self.feature_images)
    result = np.zeros((nr, nc))
    for i in range(nr):
      for j in range(nc):
        result[i,j] = np.corrcoef(imgs_to_decode[:,i], self.feature_images[:,j])
    return result


  def _nb(self, imgs_to_decode):
    """ Decode images using a Naive Bayes classifier. Unimplemented. """
    pass


  def _pattern_expression(self, imgs_to_decode):
    """ Decode images using pattern expression. """
    return np.dot(imgs_to_decode.T, self.feature_images)
