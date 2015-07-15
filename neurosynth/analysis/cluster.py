# from abc import ABCMeta, abstractmethod
# from neurosynth import masker
from copy import deepcopy
import numpy as np
from six import string_types
from sklearn import decomposition as sk_decomp
from sklearn import cluster as sk_cluster
from sklearn.metrics import pairwise_distances
from os.path import exists, join
from os import makedirs
from nibabel import nifti1


class Clusterable(object):

    def __init__(self, dataset, mask=None, min_voxels=None, min_studies=None,
                 features=None, feature_threshold=None):

        self.dataset = dataset
        self.masker = deepcopy(dataset.masker)

        # Condition study inclusion on specific features
        if features is not None:
            ids = dataset.get_studies(features=features,
                                      feature_threshold=feature_threshold)
            data = dataset.get_image_data(ids, dense=True)
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
        data = self.data.T if transpose else self.data
        self.data = transformer.fit_transform(data)
        return self


def magic(dataset, n_clusters, method='coactivation', roi_mask=None,
          coactivation_mask=None, coactivation_features=None,
          min_voxels_per_study=None, min_studies_per_voxel=None,
          reduce_reference='pca', n_components=100, roi_sign=None,
          distance_metric='correlation',
          clustering_method='kmeans', output_dir=None, prefix=None,
          clustering_kwargs={}, features=None, feature_threshold=0.05,
          filename=None):

    roi = Clusterable(dataset, roi_mask, min_voxels=min_voxels_per_study,
                      min_studies=min_studies_per_voxel, features=features,
                      feature_threshold=feature_threshold)

    if method == 'coactivation':
        reference = Clusterable(dataset, coactivation_mask,
                                min_voxels=min_voxels_per_study,
                                min_studies=min_studies_per_voxel,
                                features=features,
                                feature_threshold=feature_threshold)
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
        if transpose:
            reference.data = reference.data.T

    distances = pairwise_distances(roi.data, reference.data,
                                   metric=distance_metric)

    if isinstance(clustering_method, string_types):
        clustering_method = {
            'kmeans': sk_cluster.KMeans
        }[clustering_method](n_clusters, **clustering_kwargs)

    labels = clustering_method.fit_predict(distances) + 1.

    header = roi.masker.get_header()
    header.set_data_dtype(float)
    header['cal_max'] = labels.max()
    header['cal_min'] = labels.min()
    img = nifti1.Nifti1Image(roi.masker.unmask(labels), None, header)

    if output_dir is not None:
        if not exists(output_dir):
            makedirs(output_dir)
        if filename is None:
            raise ValueError('If output_dir is provided, a valid filename '
                             'argument must also be passed.')
        outfile = join(output_dir, 'test_cluster_img.nii.gz')
        img.to_filename(outfile)
    else:
        return img
