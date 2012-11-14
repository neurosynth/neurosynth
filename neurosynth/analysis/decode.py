import numpy as np
from neurosynth.base import imageutils

""" Decoding-related methods """

class Decoder:

  def __init__(self, dataset=None, method='pearson', features=None, mask=None, image_type=None):
    """ Initialize a new Decoder instance.

    Args:
      dataset: An optional Dataset instance containing features to use in decoding.
      method: The decoding method to use (optional). By default, Pearson correlation.
      features: Optional list of features to use in decoding. If None, use all 
        features found in dataset. If features is a list of strings, use only the 
        subset of features in the Dataset that are named in the list. If features 
        is a list of filenames, ignore the dataset entirely and use only the 
        features passed as image files in decoding.
      mask: An optional mask to apply to features and input images. If None, will use 
        the one in the current Dataset.
      image_type: An optional string indicating the type of image to use when constructing
        feature-based images. See meta.analyze_features() for details.

    """

    if dataset is not None: self.dataset = dataset
    self.method = method.lower()

    # If no mask is passed, use the dataset's.
    if mask is None:
      self.mask = dataset.volume
    else:
      from neurosynth.base import mask
      self.mask = mask.Mask(mask)

    self.load_features(features, image_type=image_type)


  def decode(self, filenames, method=None, save=None, fmt='%.3f'):
    """ Decodes a set of images.

    Args:
      files: A list of filenames of images to decode.
      method: Optional string indicating decoding method to use. If None, use 
        the method set when the Decoder instance was initialized.
      save: Optional filename to save results to. If None (default), returns 
        all results as an array.
      fmt: Optional format to pass to numpy.savetxt() if saving to file.

    Returns:
      An n_files x n_features numpy array, where each feature is a row and 
      each image is a column. The meaning of the values depends on the 
      decoding method used. """

    if method is None: method = self.method
    imgs_to_decode = imageutils.load_imgs(filenames, self.mask)
    methods = {
      'pearson': self._pearson_correlation(imgs_to_decode),
      # 'nb': self._naive_bayes(imgs_to_decode),
      'pattern': self._pattern_expression(imgs_to_decode)
    }

    result = methods[method]

    if save is not None:
      f = open(save, 'w')
      f.write('\t'.join(self.feature_names) + '\n')
      np.savetxt(f, result, fmt='%.3f', delimiter='\t')
    else:
      return methods[method]


  def set_method(self, method):
    """ Set decoding method. """
    self.method = method


  def load_features(self, features, image_type=None):
    """ Load features from current Dataset instance or a list of files. """
    from os import path
    # Check if the first element in list is a valid file; if yes, assume 
    # we're dealing with image files, otherwise treat as names of features in 
    # the current Dataset.
    if path.exists(features[0]):
      self._load_features_from_images(features)
    else:
      self._load_features_from_dataset(features, image_type=image_type)


  def _load_features_from_dataset(self, features=None, image_type=None):
    """ Load feature image data from the current Dataset instance.

    Args:
      features: Optional list of features to use. If None, all features in the 
        current Dataset instance will be used.
    """
    self.feature_names = self.dataset.feature_table.feature_names
    if features is not None:
      self.feature_names = filter(lambda x: x in self.feature_names, features)
    from neurosynth.analysis import meta
    self.feature_images = meta.analyze_features(self.dataset, self.feature_names, image_type=image_type)


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


  def _pearson_correlation(self, imgs_to_decode):
    """ Decode images using Pearson's r.

    Computes the correlation between each input image and each feature image across 
    voxels.
    
    Args:
      imgs_to_decode: An ndarray of images to decode, with voxels in rows and images
        in columns.

    Returns:
      An n_images x n_features 2D array, with each cell representing the pearson 
      correlation between the i'th image and the j'th feature across all voxels.
    """
    x, y = imgs_to_decode.astype(float), self.feature_images.astype(float)
    x, y = x - x.mean(0), y - y.mean(0)
    x, y = x/np.sqrt((x**2).sum(0)), y/np.sqrt((y**2).sum(0))
    return x.T.dot(y)


  def _naive_bayes(self, imgs_to_decode):
    """ Decode images using a Naive Bayes classifier. Unimplemented. """
    pass


  def _pattern_expression(self, imgs_to_decode):
    """ Decode images using pattern expression. For explanation, see:
    http://wagerlab.colorado.edu/wiki/doku.php/help/fmri_help/pattern_expression_and_connectivity
    """
    return np.dot(imgs_to_decode.T, self.feature_images)
