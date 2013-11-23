# emacs: -*- mode: python-mode; py-indent-offset: 2; tab-width: 2; indent-tabs-mode: nil -*-
# ex: set sts=2 ts=2 sw=2 et:
""" Dimensional/data reduction methods. """

import numpy as np


def average_within_regions(dataset, img, threshold=None, remove_zero=True):
    """ Averages over all voxels within each ROI in the input image.

    Takes a Dataset and a Nifti image that defines distinct regions, and
    returns a numpy matrix of  ROIs x mappables, where the value at each ROI is
    the proportion of active voxels in that ROI. Each distinct ROI must have a
    unique value in the image; non-contiguous voxels with the same value will
    be assigned to the same ROI.

    Args:
      dataset: A Dataset instance
      img: A NIFTI or Analyze-format image that provides the ROI definitions
      threshold: An optional float in the range of 0 - 1. If passed, the array
        will be binarized, with ROI values above the threshold assigned to True
        and values below the threshold assigned to False. (E.g., if threshold =
        0.05, only ROIs in which more than 5% of voxels are active will be
        considered active.).
      remove_zero: An optional boolean; when True, assume that voxels with value
      of 0 should not be considered as a separate ROI, and will be ignored.

    Returns:
      If replace == True, nothing is returned (the Dataset is modified in-place).
      Otherwise, returns a 2D numpy array with ROIs in rows and mappables in columns.
    """
    regions = dataset.volume.mask(img)
    labels = np.unique(regions)
    if remove_zero:
        labels = labels[np.nonzero(labels)]
    n_regions = labels.size
    m = np.zeros((regions.size, n_regions))
    for i in range(n_regions):
        m[regions == labels[i], i] = 1.0 / np.sum(regions == labels[i])
    # produces roi x study matrix
    result = np.transpose(m) * dataset.get_image_data(ids=None, dense=False)
    if threshold is not None:
        result[result < threshold] = 0.0
        result = result.astype(bool)
    return result


def get_random_voxels(dataset, n_voxels):
    """ Returns mappable data for a random subset of voxels.

    May be useful as a baseline in predictive analyses--e.g., to compare performance
    of a more principled feature selection method with simple random selection.

    Args:
      dataset: A Dataset instance
      n_voxels: An integer specifying the number of random voxels to select.

    Returns:
      A 2D numpy array with (randomly-selected) voxels in rows and mappables in columns.
    """
    voxels = np.arange(dataset.volume.num_vox_in_mask)
    np.random.shuffle(voxels)
    selected = voxels[0:n_voxels]
    return dataset.get_image_data(voxels=selected)
