""" Miscellaneous image-related utility functions. """

import logging
import nibabel as nb
from nibabel import nifti1
import numpy as np
from six import string_types

logger = logging.getLogger('neurosynth.imageutils')


def get_sphere(coords, r=4, vox_dims=(2, 2, 2), dims=(91, 109, 91)):
    """ # Return all points within r mm of coordinates. Generates a cube
    and then discards all points outside sphere. Only returns values that
    fall within the dimensions of the image."""
    r = float(r)
    xx, yy, zz = [slice(-r / vox_dims[i], r / vox_dims[
                        i] + 0.01, 1) for i in range(len(coords))]
    cube = np.vstack([row.ravel() for row in np.mgrid[xx, yy, zz]])
    sphere = cube[:, np.sum(np.dot(np.diag(
        vox_dims), cube) ** 2, 0) ** .5 <= r]
    sphere = np.round(sphere.T + coords)
    return sphere[(np.min(sphere, 1) >= 0) &
                  (np.max(np.subtract(sphere, dims), 1) <= -1), :].astype(int)


def map_peaks_to_image(peaks, r=4, vox_dims=(2, 2, 2), dims=(91, 109, 91),
                       header=None):
    """ Take a set of discrete foci (i.e., 2-D array of xyz coordinates)
    and generate a corresponding image, convolving each focus with a
    hard sphere of radius r."""
    data = np.zeros(dims)
    for p in peaks:
        valid = get_sphere(p, r, vox_dims, dims)
        valid = valid[:, ::-1]
        data[tuple(valid.T)] = 1
    return nifti1.Nifti1Image(data, None, header=header)


def load_imgs(filenames, masker, nan_to_num=True):
    """ Load multiple images from file into an ndarray.

    Args:
      filenames: A single filename or list of filenames pointing to valid
        images.
      masker: A Masker instance.
      nan_to_num: Optional boolean indicating whether to convert NaNs to zero.

    Returns:
      An m x n 2D numpy array, where m = number of voxels in mask and
      n = number of images passed.
    """
    if isinstance(filenames, string_types):
        filenames = [filenames]
    data = np.zeros((masker.n_vox_in_mask, len(filenames)))
    for i, f in enumerate(filenames):
        data[:, i] = masker.mask(f, nan_to_num)
    return data


def save_img(data, filename, masker, header=None):
    """ Save a vectorized image to file. """
    if not header:
        header = masker.get_header()
    header.set_data_dtype(data.dtype)  # Avoids loss of precision
    # Update min/max -- this should happen on save, but doesn't seem to
    header['cal_max'] = data.max()
    header['cal_min'] = data.min()
    img = nifti1.Nifti1Image(masker.unmask(data), None, header)
    img.to_filename(filename)


def threshold_img(data, threshold, mask=None, mask_out='below'):
    """ Threshold data, setting all values in the array above/below threshold
    to zero.
    Args:
        data (ndarray): The image data to threshold.
        threshold (float): Numeric threshold to apply to image.
        mask (ndarray): Optional 1D-array with the same length as the data. If
            passed, the threshold is first applied to the mask, and the
            resulting indices are used to threshold the data. This is primarily
            useful when, e.g., applying a statistical threshold to a z-value
            image based on a p-value threshold.
        mask_out (str): Thresholding direction. Can be 'below' the threshold
            (default) or 'above' the threshold. Note: use 'above' when masking
            based on p values.
    """
    if mask is not None:
        mask = threshold_img(mask, threshold, mask_out=mask_out)
        return data * mask.astype(bool)
    if mask_out.startswith('b'):
        data[data < threshold] = 0
    elif mask_out.startswith('a'):
        data[data > threshold] = 0
    return data


def create_grid(image, scale=4, apply_mask=True, save_file=None):
    """ Creates an image containing labeled cells in a 3D grid.
    Args:
        image: String or nibabel image. The image used to define the grid
            dimensions. Also used to define the mask to apply to the grid.
            Only voxels with non-zero values in the mask will be retained; all
            other voxels will be zeroed out in the returned image.
        scale: The scaling factor which controls the grid size. Value reflects
            diameter of cube in voxels.
        apply_mask: Boolean indicating whether or not to zero out voxels not in
            image.
        save_file: Optional string giving the path to save image to. Image
            written out is a standard Nifti image. If save_file is None, no
            image is written.
    Returns:
        A nibabel image with the same dimensions as the input image. All voxels
        in each cell in the 3D grid are assigned the same non-zero label.
    """
    if isinstance(image, string_types):
        image = nb.load(image)

    # create a list of cluster centers
    centers = []
    x_length, y_length, z_length = image.shape
    for x in range(0, x_length, scale):
        for y in range(0, y_length, scale):
            for z in range(0, z_length, scale):
                centers.append((x, y, z))

    # create a box around each center with the diameter equal to the scaling
    # factor
    grid = np.zeros(image.shape)
    for (i, (x, y, z)) in enumerate(centers):
        for mov_x in range((-scale + 1) // 2, (scale + 1) // 2):
            for mov_y in range((-scale + 1) // 2, (scale + 1) // 2):
                for mov_z in range((-scale + 1) // 2, (scale + 1) // 2):
                    try:  # Ignore voxels outside bounds of image
                        grid[x + mov_x, y + mov_y, z + mov_z] = i + 1
                    except:
                        pass

    if apply_mask:
        mask = image
        if isinstance(mask, string_types):
            mask = nb.load(mask)
        if type(mask).__module__ != np.__name__:
            mask = mask.get_data()
        grid[~mask.astype(bool)] = 0.0

    grid = nb.Nifti1Image(grid, image.get_affine(), image.get_header())

    if save_file is not None:
        nb.save(grid, save_file)

    return grid
