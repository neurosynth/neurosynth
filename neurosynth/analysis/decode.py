# emacs: -*- mode: python-mode; py-indent-offset: 2; tab-width: 2; indent-tabs-mode: nil -*-
# ex: set sts=2 ts=2 sw=2 et:
""" Decoding-related methods """

import numpy as np
from neurosynth.base import imageutils
from neurosynth.analysis import classify
from neurosynth.analysis import plotutils #import radar_factory


class Decoder:

    def __init__(self, dataset=None, method='pearson', features=None, mask=None, image_type='pFgA_z', threshold=0.001):
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
            feature-based images. See meta.analyze_features() for details. By default, uses 
            reverse inference z-score images.
          threshold: If decoding from a Dataset instance, this is the feature threshold to 
            use to generate the feature maps used in the decoding.


        """

        if dataset is not None:
            self.dataset = dataset
        self.method = method.lower()

        # If no mask is passed, use the dataset's.
        if mask is None:
            self.mask = dataset.masker
        else:
            from neurosynth.base import mask as m
            self.mask = m.Masker(mask)

        if features is None:
            features = dataset.get_feature_names()

        self.load_features(features, image_type=image_type, threshold=threshold)

    def decode(self, images, method=None, save=None, round=4, names=None):
        """ Decodes a set of images.

        Args:
          images: The images to decode. Can be:
            - A single String specifying the filename of the image to decode
            - A list of filenames
            - A single NumPy array containing the image data
          method: Optional string indicating decoding method to use. If None, use
            the method set when the Decoder instance was initialized.
          save: Optional filename to save results to. If None (default), returns
            all results as an array.
          round: Optional integer indicating number of decimals to round result
            to. Defaults to 4.
          names: Optional list of names corresponding to the images in filenames.
            If passed, must be of same length and in same order as filenames.
            By default, the columns in the output will be named using the image
            filenames.

        Returns:
          An n_features x n_files numpy array, where each feature is a row and
          each image is a column. The meaning of the values depends on the
          decoding method used. """

        if method is None:
            method = self.method

        if isinstance(images, basestring) or isinstance(images, list):
            imgs_to_decode = imageutils.load_imgs(images, self.mask)
        else:
            imgs_to_decode = images

        methods = {
            'pearson': self._pearson_correlation(imgs_to_decode),
            # 'nb': self._naive_bayes(imgs_to_decode),
            'pattern': self._pattern_expression(imgs_to_decode)
        }

        result = np.around(methods[method], round)

        if save is not None:

            if names is None:
                if type(images).__module__ == np.__name__:
                    names = ['image_%d' for i in range(images.shape[1])]
                else:
                    names = images

            rownames = np.array(
                self.feature_names, dtype='|S32')[:, np.newaxis]

            f = open(save, 'w')
            f.write('\t'.join(['Feature'] + names) + '\n')
            np.savetxt(f, np.hstack((
                rownames, result)), fmt='%s', delimiter='\t')
        else:
            return methods[method]

    def set_method(self, method):
        """ Set decoding method. """
        self.method = method

    def load_features(self, features, image_type=None, threshold=0.001):
        """ Load features from current Dataset instance or a list of files. """
        from os import path
        # If features is a string, assume it's a pointer to a numpy array on disk.
        # If it's a list and the first element is a valid filename, assume
        # we're dealing with image files. Otherwise treat as names of features in
        # the current Dataset.
        if isinstance(features, basestring):
            self._load_features_from_array(features)
        elif path.exists(features[0]):
            self._load_features_from_images(features)
        else:
            self._load_features_from_dataset(features, image_type=image_type, threshold=threshold)

    def _load_features_from_array(self, features):
        self.feature_images = np.load(features)
        self.feature_names = range(self.feature_images.shape[1])

    def _load_features_from_dataset(self, features=None, image_type=None, threshold=0.001):
        """ Load feature image data from the current Dataset instance.

        Args:
          features: Optional list of features to use. If None, all features in the
            current Dataset instance will be used.
        """
        self.feature_names = self.dataset.feature_table.feature_names
        if features is not None:
            self.feature_names = filter(
                lambda x: x in self.feature_names, features)
        from neurosynth.analysis import meta
        self.feature_images = meta.analyze_features(
            self.dataset, self.feature_names, image_type=image_type, threshold=threshold)

    def _load_features_from_images(self, images, names=None):
        """ Load feature image data from image files.

        Args:
          images: A list of image filenames.
          names: An optional list of strings to use as the feature names. Must be
            in the same order as the images.
        """
        if names is not None and len(names) != len(images):
            raise Exception(
                "Lists of feature names and image files must be of same length!")
        self.feature_names = names if names is not None else images
        self.feature_images = imageutils.load_imgs(images, self.mask)

    def train_classifiers(self, features=None):
        ''' Train a set of classifiers '''
        # for f in features:
        #     clf = Classifier(None)
        #     self.classifiers.append(clf)
        pass
        
    def _pearson_correlation(self, imgs_to_decode):
        """ Decode images using Pearson's r.

        Computes the correlation between each input image and each feature image across
        voxels.

        Args:
          imgs_to_decode: An ndarray of images to decode, with voxels in rows and images
            in columns.

        Returns:
          An n_features x n_images 2D array, with each cell representing the pearson
          correlation between the i'th feature and the j'th image across all voxels.
        """
        x, y = imgs_to_decode.astype(float), self.feature_images.astype(float)
        x, y = x - x.mean(0), y - y.mean(0)
        x, y = x / np.sqrt((x ** 2).sum(0)), y / np.sqrt((y ** 2).sum(0))
        return x.T.dot(y).T

    def _pattern_expression(self, imgs_to_decode):
        """ Decode images using pattern expression. For explanation, see:
        http://wagerlab.colorado.edu/wiki/doku.php/help/fmri_help/pattern_expression_and_connectivity
        """
        return np.dot(imgs_to_decode.T, self.feature_images).T


