from copy import deepcopy
import numpy as np
from six import string_types
from sklearn import decomposition as sk_decomp
from sklearn import cluster as sk_cluster
from sklearn.metrics import pairwise_distances
from os.path import exists, join
from os import makedirs
from nibabel import nifti1
from neurosynth.analysis import meta


class Clusterable(object):

    '''
    Args:
        dataset: The Dataset instance to extract data from.
        mask: A mask defining the voxels to cluster. Can be a filename,
            nibabel image, or numpy array (see Masker.mask() for details).
        features (str or list): Optional string or list of strings specifying
            any feature names to use for study selection. E.g., passing
            ['emotion', 'reward'] would retain for analysis only those studies
            associated with the features emotion or reward at a frequency
            greater than feature_threshold.
        feature_threshold (float): The threshold to use when selecting studies
            on the basis of features.
        min_voxels (int): Minimum number of active voxels a study
            must report in order to be retained in the dataset. By default,
            all studies are used.
        min_studies (int): Minimum number of studies a voxel must be
            active in in order to be retained in analysis. By default, all
            voxels are used.
    '''

    def __init__(self, dataset, mask=None, features=None,
                 feature_threshold=None, min_voxels=None, min_studies=None):

        self.dataset = dataset
        self.masker = deepcopy(dataset.masker)

        # Condition study inclusion on specific features
        if features is not None:
            ids = dataset.get_studies(features=features,
                                      frequency_threshold=feature_threshold)
            data = dataset.get_image_data(ids, dense=False)
        else:
            data = dataset.image_table.data

        # Trim data based on minimum number of voxels or studies
        if min_studies is not None:
            av = self.masker.unmask(
                data.sum(1) >= min_studies, output='vector')
            self.masker.add({'voxels': av})

        if min_voxels is not None:
            data = data[:, np.array(data.sum(0) >= min_voxels).squeeze()]

        if mask is not None:
            self.masker.add({'roi': mask})

        self.data = data[self.masker.get_mask(['voxels', 'roi']), :].toarray()

    def transform(self, transformer, transpose=False):
        ''' Apply a transformation to the Clusterable instance. Accepts any
        scikit-learn-style class that implements a fit_transform() method. '''
        data = self.data.T if transpose else self.data
        data = transformer.fit_transform(data)
        self.data = data.T if transpose else data
        return self


def magic(dataset, method='coactivation', roi_mask=None,
          coactivation_mask=None, features=None, feature_threshold=0.05,
          min_voxels_per_study=None, min_studies_per_voxel=None,
          reduce_reference='pca', n_components=100,
          distance_metric='correlation', clustering_algorithm='kmeans',
          n_clusters=5, clustering_kwargs={}, output_dir=None, filename=None,
          coactivation_images=False, coactivation_threshold=0.1):
    ''' Execute a full clustering analysis pipeline.
    Args:
        dataset: a Dataset instance to extract all data from.
        method (str): the overall clustering approach to use. Valid options:
            'coactivation' (default): Clusters voxel within the ROI mask based
                on shared pattern of coactivation with the rest of the brain.
            'studies': Treat each study as a feature in an n-dimensional space.
                I.e., voxels will be assigned to the same cluster if they tend
                to be co-reported in similar studies.
        roi_mask: A string, nibabel image, or numpy array providing an
            inclusion mask of voxels to cluster. If None, the default mask
            in the Dataset instance is used (typically, all in-brain voxels).
        coactivation_mask: If method='coactivation', this mask defines the
            voxels to use when generating the pairwise distance matrix. For
            example, if a PFC mask is passed, all voxels in the roi_mask will
            be clustered based on how similar their patterns of coactivation
            with PFC voxels are. Can be a str, nibabel image, or numpy array.
        features (str or list): Optional string or list of strings specifying
            any feature names to use for study selection. E.g., passing
            ['emotion', 'reward'] would retain for analysis only those studies
            associated with the features emotion or reward at a frequency
            greater than feature_threshold.
        feature_threshold (float): The threshold to use when selecting studies
            on the basis of features.
        min_voxels_per_study (int): Minimum number of active voxels a study
            must report in order to be retained in the dataset. By default,
            all studies are used.
        min_studies_per_voxel (int): Minimum number of studies a voxel must be
            active in in order to be retained in analysis. By default, all
            voxels are used.
        reduce_reference (str, scikit-learn object or None): The dimensionality
            reduction algorithm to apply to the feature space prior to the
            computation of pairwise distances. If a string is passed (either
            'pca' or 'ica'), n_components must be specified. If None, no
            dimensionality reduction will be applied. Otherwise, must be a
            scikit-learn-style object that exposes a transform() method.
        n_components (int): Number of components to extract during the
            dimensionality reduction step. Only used if reduce_reference is
            a string.
        distance_metric (str): The distance metric to use when computing
            pairwise distances on the to-be-clustered voxels. Can be any of the
            metrics supported by sklearn.metrics.pairwise_distances.
        clustering_algorithm (str or scikit-learn object): the clustering
            algorithm to use. If a string, must be one of 'kmeans' or 'minik'.
            Otherwise, any sklearn class that exposes a fit_predict() method.
        n_clusters (int): If clustering_algorithm is a string, the number of
            clusters to extract.
        clustering_kwargs (dict): Additional keywords to pass to the clustering
            object.
        output_dir (str): The directory to write results to. If None (default),
            returns the cluster label image rather than saving to disk.
        filename (str): Name of cluster label image file. Defaults to
            cluster_labels_k{k}.nii.gz, where k is the number of clusters.
        coactivation_images (bool): If True, saves a meta-analytic coactivation
            map for every ROI in the resulting cluster map.
        coactivation_threshold (float or int): If coactivation_images is True,
            this is the threshold used to define whether or not a study is
            considered to activation within a cluster ROI. Integer values are
            interpreted as minimum number of voxels within the ROI; floats
            are interpreted as the proportion of voxels. Defaults to 0.1 (i.e.,
            10% of all voxels within ROI must be active).
    '''

    roi = Clusterable(dataset, roi_mask, min_voxels=min_voxels_per_study,
                      min_studies=min_studies_per_voxel, features=features,
                      feature_threshold=feature_threshold)

    if method == 'coactivation':
        reference = Clusterable(dataset, coactivation_mask,
                                min_voxels=min_voxels_per_study,
                                min_studies=min_studies_per_voxel,
                                features=features,
                                feature_threshold=feature_threshold)
    elif method == 'features':
        reference = deepcopy(roi)
        feature_data = dataset.feature_table.data
        n_studies = len(feature_data)
        reference.data = reference.data.dot(feature_data.values) / n_studies
    elif method == 'studies':
        reference = roi

    if reduce_reference is not None:
        if isinstance(reduce_reference, string_types):
            reduce_reference = {
                'pca': sk_decomp.RandomizedPCA,
                'ica': sk_decomp.FastICA
            }[reduce_reference](n_components)

        transpose = (method == 'coactivation')
        reference = reference.transform(reduce_reference, transpose=transpose)

    if method == 'coactivation':
        distances = pairwise_distances(roi.data, reference.data,
                                       metric=distance_metric)
    else:
        distances = reference.data

    # TODO: add additional clustering methods
    if isinstance(clustering_algorithm, string_types):
        clustering_algorithm = {
            'kmeans': sk_cluster.KMeans,
            'minik': sk_cluster.MiniBatchKMeans
        }[clustering_algorithm](n_clusters, **clustering_kwargs)

    labels = clustering_algorithm.fit_predict(distances) + 1.

    header = roi.masker.get_header()
    header['cal_max'] = labels.max()
    header['cal_min'] = labels.min()
    voxel_labels = roi.masker.unmask(labels)
    img = nifti1.Nifti1Image(voxel_labels, None, header)

    if output_dir is not None:
        if not exists(output_dir):
            makedirs(output_dir)
        if filename is None:
            filename = 'cluster_labels_k%d.nii.gz' % n_clusters
        outfile = join(output_dir, filename)
        img.to_filename(outfile)

        # Write coactivation images
        if coactivation_images:
            for l in np.unique(voxel_labels):
                roi_mask = np.copy(voxel_labels)
                roi_mask[roi_mask != l] = 0
                ids = dataset.get_studies(
                    mask=roi_mask, activation_threshold=coactivation_threshold)
                ma = meta.MetaAnalysis(dataset, ids)
                ma.save_results(output_dir=join(output_dir, 'coactivation'),
                                prefix='cluster_%d_coactivation' % l)
    else:
        return img
