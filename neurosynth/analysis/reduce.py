# emacs: -*- mode: python-mode; py-indent-offset: 2; tab-width: 2; indent-tabs-mode: nil -*-
# ex: set sts=2 ts=2 sw=2 et:
""" Dimensional/data reduction methods. """

import numpy as np
from neurosynth.base.dataset import Dataset
from neurosynth.base import imageutils
import logging

logger = logging.getLogger('neurosynth.cluster')

def average_within_regions(dataset, regions, mask=None, threshold=None, remove_zero=True):
        """ Aggregates over all voxels within each ROI in the input image.

        Takes a Dataset and a Nifti image that defines distinct regions, and
        returns a numpy matrix of  ROIs x mappables, where the value at each ROI is
        the proportion of active voxels in that ROI. Each distinct ROI must have a
        unique value in the image; non-contiguous voxels with the same value will
        be assigned to the same ROI.

        Args:
            dataset: Either a Dataset instance from which image data are extracted, or a 
                Numpy array containing image data to use. If the latter, the array contains 
                voxels in rows and features/studies in columns. The number of voxels must 
                be equal to the length of the vectorized image mask in the 
            regions: An image defining the boundaries of the regions to use. Can be:
                1) A string name of the NIFTI or Analyze-format image
                2) A NiBabel SpatialImage
                3) A 1D numpy array of the same length as the mask vector in the Dataset's
                     current Masker.
            threshold: An optional float in the range of 0 - 1. If passed, the array
                will be binarized, with ROI values above the threshold assigned to True
                and values below the threshold assigned to False. (E.g., if threshold =
                0.05, only ROIs in which more than 5% of voxels are active will be
                considered active.).
            remove_zero: An optional boolean; when True, assume that voxels with value
            of 0 should not be considered as a separate ROI, and will be ignored.

        Returns:
            A 2D numpy array with ROIs in rows and mappables in columns.
        """
        if type(regions).__module__ != np.__name__:
            regions = dataset.volume.mask(regions)
            
        if isinstance(dataset, Dataset):
            dataset = dataset.get_image_data(dense=False)

        labels = np.unique(regions)

        if remove_zero:
                labels = labels[np.nonzero(labels)]

        n_regions = labels.size

        # Create the ROI-coding matrix
        m = np.zeros((regions.size, n_regions))
        for i in range(n_regions):
                m[regions == labels[i], i] = 1.0 / np.sum(regions == labels[i])

        # Call dot() on the array itself as this will use sparse matrix 
        # multiplication if possible.
        result = dataset.T.dot(m).T
        # try:
        #     result = dataset.T.dot(m).T
        # except:
        #     result = np.dot(np.transpose(m), dataset)

        if threshold is not None:
                result[result < threshold] = 0.0
                result = result.astype(bool)

        return result

def apply_grid(dataset, image=None, scale=4, threshold=None):
    """ Imposes a 3D grid on the brain volume and averages across all voxels that 
    fall within each cell.
    Args:
        dataset: Data to apply grid to. Either a Dataset instance, or a numpy array
            with voxels in rows and features in columns.
        image: Optional image to use to define the grid dimensions. If dataset is 
            a Dataset instance and image is None, the default mask in the Dataset
            wil be used. If dataset is a numpy array, image must be provided.
        threshold: Optional float to pass to reduce.average_within_regions().
    Returns:
        A tuple of length 2, where the first element is a numpy array of dimensions
        n_cubes x n_studies, and the second element is a numpy array, with the same 
        dimensions as the Mask instance in the current Dataset, that maps voxel 
        identities onto cell IDs in the grid.
    """
    if image is None:
        if isinstance(dataset, Dataset):
            image = dataset.volume.volume
        else:
            raise ValueError("If dataset is a numpy array, a Nifti image defining " + 
                "the grid dimensions must be provided.")

    grid = imageutils.create_grid(image, scale)
    data = average_within_regions(dataset, grid, threshold)
    return (data, grid)

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
