# emacs: -*- mode: python-mode; py-indent-offset: 2; tab-width: 2; indent-tabs-mode: nil -*-
# ex: set sts=2 ts=2 sw=2 et:

import numpy as np
import nibabel as nb
import os


class Mask(object):

    """ A lightweight wrapper around the NiBabel classes that
    handles vectorization/masking/unmasking of images. """

    def __init__(self, volume):
        """ Initialize a new ImageMask. The volume passed to the constructor indicates
        both the space in which all subsequent images are represented as well as the
        mask to use for all analyses. Any voxel in the mask with a non-zero valid is
        considered valid for analyses.

        TODO: implement additional masking to allow more efficient small-volume analyses.
        """
        self.volume = nb.load(volume)
        data = self.volume.get_data()
        self.dims = data.shape
        self.vox_dims = self.get_header().get_zooms()
        self.full = np.float64(data.ravel())
        self.in_mask = np.where(
            self.full)  # Indices of in-mask voxels within full volume
        self.num_vox_in_mask = np.shape(self.in_mask)[1]

    def mask(self, img, nan_to_num=True):
        """ Vectorize an image and mask out all invalid voxels.

        Args:
          img: The image to vectorize and mask. Can be either a filename or a SpatialImage
            previously loaded with NiBabel.
          nan_to_num: boolean indicating whether to convert NaNs to 0.

        Returns:
          A 1D NumPy array of in-mask voxels.
        """
        if isinstance(img, basestring):
            img = nb.load(img)
        masked_data = img.get_data().ravel()[self.in_mask]
        if nan_to_num:
            masked_data = np.nan_to_num(masked_data)
        return masked_data

    def unmask(self, data):
        """ Reconstruct a masked vector into the original 3D volume. """
        n_in_mask_voxels = len(self.in_mask[0])
        if data.ndim == 2:
            n_volumes = data.shape[1]
            # we got two dimensions, so take 2nd dimension as the temporal
            # dimension
            assert(len(data) == n_in_mask_voxels)
            assert(self.full.ndim == 1)
            # but we generate x,y,z,t volume
            img = np.zeros(self.full.shape + (n_volumes,))
            # for t in xrange(n_volumes):
            img[self.in_mask, :] = data
            return np.reshape(img, self.volume.shape + (n_volumes,))
        else:
            img = self.full.copy()
            img[self.in_mask] = data
            return np.reshape(img, self.volume.shape)

    def get_header(self):
        """ A wrapper for the NiBabel method. """
        return self.volume.get_header()
