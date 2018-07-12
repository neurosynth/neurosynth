""" Decoding tools"""

import numpy as np
from neurosynth.base.mask import Masker
from neurosynth.base import imageutils
from neurosynth.analysis.reduce import average_within_regions
from os import path
import pandas as pd
import matplotlib.pyplot as plt
from six import string_types


class Decoder:

    def __init__(self, dataset=None, method='pearson', features=None,
                 mask=None, image_type='association-test_z', threshold=0.001):
        """ Initialize a new Decoder instance.

        Args:
            dataset: An optional Dataset instance containing features to use in
                decoding.
            method: The decoding method to use (optional). By default, Pearson
                correlation.
            features: Optional list of features to use in decoding. If None,
                use all features found in dataset. If features is a list of
                strings, use only the subset of features in the Dataset that
                are named in the list. If features is a list of filenames,
                ignore the dataset entirely and use only the features passed as
                image files in decoding.
            mask: An optional mask to apply to features and input images. If
                None, will use the one in the current Dataset.
            image_type: An optional string indicating the type of image to use
                when constructing feature-based images. See
                meta.analyze_features() for details. By default, uses reverse
                inference z-score images.
            threshold: If decoding from a Dataset instance, this is the feature
                threshold to use to generate the feature maps used in the
                decoding.


        """

        self.dataset = dataset

        if dataset is not None:
            self.masker = self.dataset.masker
            if features is None:
                features = dataset.get_feature_names()
            if mask is not None:
                self.masker.add(mask)
        elif mask is not None:
            self.masker = Masker(mask)
        else:
            self.masker = None

        self.method = method.lower()

        if self.method == 'roi':
            self.feature_names = features
        else:
            self.load_features(features, image_type=image_type,
                               threshold=threshold)

    def decode(self, images, save=None, round=4, names=None, **kwargs):
        """ Decodes a set of images.

        Args:
          images: The images to decode. Can be:
            - A single String specifying the filename of the image to decode
            - A list of filenames
            - A single NumPy array containing the image data
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

        if isinstance(images, string_types):
            images = [images]

        if isinstance(images, list):
            imgs_to_decode = imageutils.load_imgs(images, self.masker)
        else:
            imgs_to_decode = images

        methods = {
            'pearson': self._pearson_correlation,
            'dot': self._dot_product,
            'roi': self._roi_association
        }

        result = np.around(
            methods[self.method](imgs_to_decode, **kwargs), round)

        # if save is not None:

        if names is None:
            if type(images).__module__ == np.__name__:
                names = ['image_%d' % i for i in range(images.shape[1])]
            elif self.method == 'roi':
                names = ['cluster_%d' % i for i in range(result.shape[1])]
            else:
                names = images

        result = pd.DataFrame(result, columns=names, index=self.feature_names)

        if save is not None:
            result.to_csv(save, index_label='Feature')
        return result

    def set_method(self, method):
        """ Set decoding method. """
        self.method = method

    def load_features(self, features, image_type=None, from_array=False,
                      threshold=0.001):
        """ Load features from current Dataset instance or a list of files.
        Args:
            features: List containing paths to, or names of, features to
                extract. Each element in the list must be a string containing
                either a path to an image, or the name of a feature (as named
                in the current Dataset). Mixing of paths and feature names
                within the list is not allowed.
            image_type: Optional suffix indicating which kind of image to use
                for analysis. Only used if features are taken from the Dataset;
                if features is a list of filenames, image_type is ignored.
            from_array: If True, the features argument is interpreted as a
                string pointing to the location of a 2D ndarray on disk
                containing feature data, where rows are voxels and columns are
                individual features.
            threshold: If features are taken from the dataset, this is the
                threshold passed to the meta-analysis module to generate fresh
                images.

        """
        if from_array:
            if isinstance(features, list):
                features = features[0]
            self._load_features_from_array(features)
        elif path.exists(features[0]):
            self._load_features_from_images(features)
        else:
            self._load_features_from_dataset(
                features, image_type=image_type, threshold=threshold)

    def _load_features_from_array(self, features):
        """ Load feature data from a 2D ndarray on disk. """
        self.feature_images = np.load(features)
        self.feature_names = range(self.feature_images.shape[1])

    def _load_features_from_dataset(self, features=None, image_type=None,
                                    threshold=0.001):
        """ Load feature image data from the current Dataset instance. See
        load_features() for documentation.
        """
        self.feature_names = self.dataset.feature_table.feature_names
        if features is not None:
            self.feature_names = [f for f in features
                                  if f in self.feature_names]
        from neurosynth.analysis import meta
        self.feature_images = meta.analyze_features(
            self.dataset, self.feature_names, image_type=image_type,
            threshold=threshold)
        # Apply a mask if one was originally passed
        if self.masker.layers:
            in_mask = self.masker.get_mask(in_global_mask=True)
            self.feature_images = self.feature_images[in_mask, :]

    def _load_features_from_images(self, images, names=None):
        """ Load feature image data from image files.

        Args:
          images: A list of image filenames.
          names: An optional list of strings to use as the feature names. Must
            be in the same order as the images.
        """
        if names is not None and len(names) != len(images):
            raise Exception(
                "Lists of feature names and images must be of same length!")
        self.feature_names = names if names is not None else images
        self.feature_images = imageutils.load_imgs(images, self.masker)

    def train_classifiers(self, features=None):
        ''' Train a set of classifiers '''
        # for f in features:
        #     clf = Classifier(None)
        #     self.classifiers.append(clf)
        pass

    def _pearson_correlation(self, imgs_to_decode):
        """ Decode images using Pearson's r.

        Computes the correlation between each input image and each feature
        image across voxels.

        Args:
            imgs_to_decode: An ndarray of images to decode, with voxels in rows
                and images in columns.

        Returns:
            An n_features x n_images 2D array, with each cell representing the
            pearson correlation between the i'th feature and the j'th image
            across all voxels.
        """
        x, y = imgs_to_decode.astype(float), self.feature_images.astype(float)
        return self._xy_corr(x, y)

    def _dot_product(self, imgs_to_decode):
        """ Decoding using the dot product.
        """
        return np.dot(imgs_to_decode.T, self.feature_images).T

    def _roi_association(self, imgs_to_decode, value='z', binarize=None):
        """ Computes the strength of association between activation in a mask
        and presence/absence of a semantic feature. This is essentially a
        generalization of the voxel-wise reverse inference z-score to the
        multivoxel case.
        """
        imgs_to_decode = imgs_to_decode.squeeze()
        x = average_within_regions(self.dataset, imgs_to_decode).astype(float)
        y = self.dataset.feature_table.data[self.feature_names].values
        if binarize is not None:
            y[y > binarize] = 1.
            y[y < 1.] = 0.
        r = self._xy_corr(x.T, y)
        if value == 'r':
            return r
        elif value == 'z':
            f_r = np.arctanh(r)
            return f_r * np.sqrt(y.shape[0] - 3)

    def _xy_corr(self, x, y):
        x, y = x - x.mean(0), y - y.mean(0)
        x, y = x / np.sqrt((x ** 2).sum(0)), y / np.sqrt((y ** 2).sum(0))
        return x.T.dot(y).T

    def plot_polar(self, data, n_top=3, overplot=False, labels=None,
                   palette='husl'):

        n_panels = data.shape[1]

        if labels is None:
            labels = []
            for i in range(n_panels):
                labels.extend(data.iloc[:, i].order(ascending=False)
                              .index[:n_top])
            labels = np.unique(labels)

        data = data.loc[labels, :]

        # Use hierarchical clustering to order
        from scipy.spatial.distance import pdist
        from scipy.cluster.hierarchy import linkage, leaves_list
        dists = pdist(data, metric='correlation')
        pairs = linkage(dists)
        order = leaves_list(pairs)
        data = data.iloc[order, :]
        labels = [labels[i] for i in order]

        theta = np.linspace(0.0, 2 * np.pi, len(labels), endpoint=False)
        if overplot:
            fig, ax = plt.subplots(1, 1, subplot_kw=dict(polar=True))
            fig.set_size_inches(10, 10)
        else:
            fig, axes = plt.subplots(1, n_panels, sharex=False, sharey=False,
                                     subplot_kw=dict(polar=True))
            fig.set_size_inches((6 * n_panels, 6))
        # A bit silly to import seaborn just for this...
        # should extract just the color_palette functionality.
        import seaborn as sns
        colors = sns.color_palette(palette, n_panels)
        for i in range(n_panels):
            if overplot:
                alpha = 0.2
            else:
                ax = axes[i]
                alpha = 0.8
            ax.set_ylim(data.values.min(), data.values.max())
            d = data.iloc[:, i].values
            ax.fill(theta, d, color=colors[i], alpha=alpha, ec='k',
                    linewidth=0)
            ax.fill(theta, d, alpha=1.0, ec=colors[i],
                    linewidth=2, fill=False)
            ax.set_xticks(theta)
            ax.set_xticklabels(labels, fontsize=18)
            [lab.set_fontsize(18) for lab in ax.get_yticklabels()]
            ax.set_title('Cluster %d' % i, fontsize=22, y=1.12)
        plt.tight_layout()
        return plt
