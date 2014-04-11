# emacs: -*- mode: python-mode; py-indent-offset: 2; tab-width: 2; indent-tabs-mode: nil -*-
# ex: set sts=2 ts=2 sw=2 et:

import numpy as np
import pandas as pd
import nibabel as nb
import os


class Masker(object):

    """ Handles vectorization/masking/unmasking of images. """


    def __init__(self, volume, layers=None):
        """ Initialize a new Masker.
        Args:
            volume: A volume indicating the global space within which all subsequent 
                layers must reside. Any voxel in the mask with a non-zero valid is
                considered valid for analyses. Can be either an image filename or a
                NiBabel image.
            layers: Optional masking layers to add; see documentation for add().
        """

        self.layers = {}
        self.stack = []

        if layers is not None:
            self.add(layers)

        if isinstance(volume, basestring):
            volume = nb.load(volume)
        self.volume = volume
        data = self.volume.get_data()
        self.dims = data.shape
        self.vox_dims = self.get_header().get_zooms()
        self.full = np.float64(data.ravel())
        self._set_mask()
        self.num_vox_in_mask = len(np.where(self.current_mask)[0])


    def add(self, layers):
        """ Add one or more layers to the stack of masking layers. 
        Args:
            layers: A string, NiBabel image, list, or dict.
                If anything other than a dict is passed, assigns sequential layer names 
                based on the current position in stack; if a dict, uses key as the name 
                and value as the mask image.
        """
        if isinstance(layers, dict):
            for (name, image) in layers.items():
                self._add_named_layer(name, image)

        else:
            if not isinstance(layers, (list, dict)):
                layers = [layers]
            for image in layers:
                name = 'layer_%d' % len(self.stack)
                self._add_named_layer(name, image)


    def _add_named_layer(self, name, image):
        image = self.get_image(image, output='vector')
        self.layers[name] = image
        self.stack.append(name)


    def remove(self, layers):
        """ Remove one or more layers from the stack of masking layers.
        Args:
            layers: An int, string or list of strings and/or ints. Ints are interpreted 
                as indices in the stack to remove; strings are interpreted as names of 
                layers to remove.
        """
        if not isinstance(layers, list):
            layers = [layers]
            for l in layers:
                if isinstance(l, basestring):
                    self.stack.remove(l)
                else:
                    l = self.stack.pop(l)
                self.layers.pop(l)


    def get_image(self, image, output='vector'):
        """ A flexible method for transforming between different representations of 
        image data.
        Args:
            image: The input image. Can be a string (filename of image), NiBabel image,
                N-dimensional array (must have same shape as self.volume), or 
                vectorized image data (must have same length as current conjunction 
                    mask).
            output: The format of the returned image representation. Must be one of:
                'vector': A 1D vectorized array
                'array': An N-dimensional array, with shape = self.volume.shape
                'image': A NiBabel image
        Returns: An object containing image data; see output options above.
        """
        if isinstance(image, basestring):
            image = nb.load(image)
        
        if type(image).__module__.startswith('nibabel'):
            if output == 'image':
                return image
            image = image.get_data()

        if not type(image).__module__.startswith('numpy'):
            raise ValueError("Input image must be a string, a NiBabel image, or a " + 
                "numpy array.")

        if image.shape == self.volume.shape:
            if output == 'image':
                return nb.nifti1.Nifti1Image(image, None, self.get_header())
            elif output == 'array':
                return image
            else:
                image = image.ravel()
        
        if output == 'vector':
            return image
        
        image = np.reshape(image, self.volume.shape)

        if output == 'array':
            return image

        return nb.nifti1.Nifti1Image(image, None, self.get_header())


    def mask(self, image, layers=None, nan_to_num=True, compute_mask=True):
        """ Vectorize an image and mask out all invalid voxels.

        Args:
            images: The image to vectorize and mask. Input can be any object handled 
                by get_image().
            layers: Which mask layers to use (specified as int, string, or list of 
                ints and strings). When None, applies the conjunction of all layers.
            nan_to_num: boolean indicating whether to convert NaNs to 0.
            compute_mask: If True, will recompute the current mask from scratch. 
                When False, reuses the existing current_mask. Mainly useful when 
                iterating over input images and/or output formats.

        Returns:
          A 1D NumPy array of in-mask voxels.
        """
        image = self.get_image(image, output='vector')
        if compute_mask:
            self._set_mask(layers)
        masked_data = image[self.current_mask]
        if nan_to_num:
            masked_data = np.nan_to_num(masked_data)
        return masked_data


    def unmask(self, data, layers=None, output='array', compute_mask=True):
        """ Reconstruct a masked vector into the original 3D volume. 
        Args:
            data: The 1D vector to reconstruct. (Can also be a 2D vector where 
                the second dimension is time, but then output will always
                be set to 'array'--i.e., a 4D image will be returned.)
            layers: Which mask layers to use (specified as int, string, or list of 
                ints and strings). When None, applies the conjunction of all layers.
                Note that the layers specified here must exactly match the layers 
                used in the mask() operation, otherwise the shape of the mask will 
                be incorrect and bad things will happen.
            output: What kind of object to return. See options in get_image(). By 
                default, returns an N-dimensional array of reshaped data.
            compute_mask: If True, will recompute the current mask from scratch.
                When False, reuses the existing current_mask. Mainly useful when 
                iterating over input images and/or output formats.
        """
        if compute_mask:
            self._set_mask(layers)
        if data.ndim == 2:
            n_volumes = data.shape[1]
            # Assume 1st dimension is voxels, 2nd is time
            assert(len(data) == self.num_vox_in_mask)
            assert(self.full.ndim == 1)
            # but we generate x,y,z,t volume
            image = np.zeros(self.full.shape + (n_volumes,))
            image[self.current_mask, :] = data
            return np.reshape(image, self.volume.shape + (n_volumes,))
        else:
            img = self.full.copy()
            image = np.zeros(self.full.shape)
            image[self.current_mask] = data
            return self.get_image(image, output)


    def _set_mask(self, layers=None, include_global_mask=True):
        """ Set the current mask by taking the conjunction of all specified layers.
        Args:
            layers: Which layers to include. See documentation for add() for format.
            include_global_mask: Whether or not to automatically include the global 
                mask (i.e., self.volume) in the conjunction.
        """
        if layers is None:
            layers = self.layers.keys()
        elif not isinstance(layers, list):
            layers = [layers]
        
        layers = map(lambda x: x if isinstance(x, basestring) else self.stack[x], layers)

        layers = [self.layers[l] for l in layers]
        if include_global_mask:
            layers.append(self.full)  # Always include the original volume
        layers = np.vstack(layers).T.astype(bool)
        self.current_mask = layers.all(axis=1)


    def get_header(self):
        """ A wrapper for the NiBabel method. """
        return self.volume.get_header()
