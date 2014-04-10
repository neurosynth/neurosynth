# emacs: -*- mode: python-mode; py-indent-offset: 2; tab-width: 2; indent-tabs-mode: nil -*-
# ex: set sts=2 ts=2 sw=2 et:
""" Miscellaneous image-related functions. """

import json
import logging

import nibabel as nb
from nibabel import nifti1
import numpy as np

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
    return sphere[(np.min(sphere, 1) >= 0) & (np.max(np.subtract(sphere, dims), 1) <= -1),:].astype(int)


def map_peaks_to_image(peaks, r=4, vox_dims=(2, 2, 2), dims=(91, 109, 91), header=None):
    """ Take a set of discrete foci (i.e., 2-D array of xyz coordinates)
    and generate a corresponding image, convolving each focus with a
    hard sphere of radius r."""
    data = np.zeros(dims)
    for p in peaks:
        valid = get_sphere(p, r, vox_dims, dims)
        valid = valid[:, ::-1]
        data[tuple(valid.T)] = 1
    return nifti1.Nifti1Image(data, None, header=header)


def convolve_image(img, r=4, header=None, method='mean', save=None):
    """ Take an image and multiples every non-zero value found by a hard
    sphere of radius r. Where multiple values overlap, use either sum or
    mean. """
    img = nb.load(img)
    data = img.get_data()
    dims = data.shape
    result = np.zeros(dims)
    counts = np.zeros(dims)
    nonzero = np.nonzero(data)
    for point in zip(*nonzero):
        fill = tuple(get_sphere(point, r, dims=dims).T)
        # fill = tuple(fill)
        result[fill] += data[point]
        counts[fill] += 1
    result = np.divide(result, counts)
    result = np.nan_to_num(result)
    if save is None:
        return result
    else:
        img = nifti1.Nifti1Image(result, None, img.get_header())
        img.to_filename(save)


def load_imgs(filenames, masker, nan_to_num=True):
    """ Load multiple images from file into an ndarray.

    Args:
      filenames: A single filename or list of filenames pointing to valid images.
      masker: A Masker instance.
      nan_to_num: Optional boolean indicating whether to convert NaNs to zero.

    Returns:
      An m x n 2D numpy array, where m = number of voxels in mask and
      n = number of images passed.
    """
    if isinstance(filenames, basestring):
        filenames = [filenames]
    data = np.zeros((masker.num_vox_in_mask, len(filenames)))
    for i, f in enumerate(filenames):
        data[:, i] = masker.mask(f, nan_to_num)
    return data


def save_img(data, filename, mask, header=None):
    """ Save a vectorized image to file. """
    if not header:
        header = mask.get_header()
    header.set_data_dtype(data.dtype)  # Avoids loss of precision
    img = nifti1.Nifti1Image(mask.unmask(data), None, header)
    img.to_filename(filename)


def threshold_img(data, threshold, mask=None, mask_out='below'):
    """ Threshold data, setting all values in the array above/below threshold to zero.
    Optionally, can provide a mask (a 1D array with the same length as data), in which
    case the threshold is first applied to the mask, and the resulting indices are used
    to threshold the data. This is primarily useful when, e.g., applying a statistical
    threshold to a z-value image based on a p-value threshold. The mask_out argument
    indicates whether to zero out values 'below' the threshold (default) or 'above' the
    threshold. Note: use 'above' when masking based on p values. """
    if mask is not None:
        mask = threshold_img(mask, threshold, mask_out=mask_out)
        return data * mask.astype(bool)
    if mask_out.startswith('b'):
        data[data < threshold] = 0
    elif mask_out.startswith('a'):
        data[data > threshold] = 0
    return data

def create_grid(image, scale=4, mask=None, save_file=None):
    """ Creates an image containing labeled cells in a 3D grid.
    Args:
        image: String or nibabel image. The image used to define the grid dimensions.
        scale: The scaling factor which controls the grid size. Value reflects diameter of cube
            in voxels.
        mask: Optional string or nibabel image; a mask to apply to the grid. Only voxels with 
            non-zero values in the mask will be retained; all other voxels will be zeroed out 
            in the returned image.
        save_file: Optional string giving the path to save image to. Image written out is
            a standard Nifti image. If save_file is None, no image is written.
    Returns:
        A nibabel image with the same dimensions as the input image. All voxels in each cell
        in the 3D grid are assigned the same non-zero label.
    """
    if isinstance(image, basestring):
        image = nb.load(image)

    #create a list of cluster centers 
    centers = []
    x_length, y_length, z_length = image.shape
    for x in range(0, x_length, scale):
        for y in range(0, y_length, scale):
            for z in range(0, z_length, scale):
                centers.append((x, y, z))

    #create a box around each center with the diameter equal to the scaling factor
    grid = np.zeros(image.shape)
    for (i, (x,y,z)) in enumerate(centers):
        for mov_x in range((-scale+1)/2,(scale+1)/2):
            for mov_y in range((-scale+1)/2,(scale+1)/2):
                for mov_z in range((-scale+1)/2,(scale+1)/2):
                    try:  # Ignore voxels outside bounds of image
                        grid[x+mov_x, y+mov_y, z+mov_z] = i+1
                    except: pass

    if mask is not None:
        if isinstance(mask, basestring):
            mask = nb.load(mask)
        grid[~mask.get_data().astype(bool)] = 0.0

    grid = nb.Nifti1Image(grid, image.get_affine(), image.get_header())

    if save_file is not None:
        nb.save(grid, save_file)

    return grid


def img_to_json(img, decimals=2, swap=False, save=None):
    """ Convert an image volume to web-ready JSON format suitable for import into
    the Neurosynth viewer.

    Args:
      img: An image filename.
      round: Optional integer giving number of decimals to round values to.
      swap: A temporary kludge to deal with some orientation problems. For some reason
        the switch from PyNifti to NiBabel seems to produce images that load in a
        different orientation given the same header. In practice this can be addressed
        by flipping the x and z axes (swap = True), but need to look into this and
        come up with a permanent solution.

    Returns:
      a JSON-formatted string.

    """
    try:
        data = nb.load(img).get_data()
    except Exception as e:
        raise Exception("Error loading %s: %s" % (img, str(e)))

    dims = list(data.shape)

    # Convenience method to package and output the converted data;
    # also handles cases where image is blank.
    def package_json(contents=None):
        if contents is None:
            contents = {
              'thresh': 0.0,
              'max': 0.0,
              'min': 0.0,
              'dims': dims,
              'values': [],
              'indices': []
            }
        # Write to file or return string
        if save is None:
            return json.dumps(contents)
        else:
            json.dump(contents, open(save, 'w'))

    # Skip empty images
    data = np.nan_to_num(data)
    if np.sum(data) == 0:
        return package_json()

    # Round values to save space. Note that in practice the resulting JSON file will
    # typically be larger than the original nifti unless the image is relatively
    # dense (even when compressed). More reason to switch from JSON to nifti reading
    # in the viewer!
    data = np.round_(data, decimals)

    # Temporary kludge to fix orientation issue
    if swap:
        data = np.swapaxes(data, 0, 2)

    # Identify threshold--minimum nonzero value
    thresh = np.min(np.abs(data[np.nonzero(data)]))

    # compress into 2 lists, one with values, the other with list of indices
    # for each value
    uniq = list(np.unique(data))
    # uniq = np.unique()
    uniq.remove(0)
    if len(uniq) == 0:
        return package_json()

    contents = {
      'thresh': round(thresh, decimals),
      'max': round(np.max(data), decimals),
      'min': round(np.min(data), decimals),
      'dims': dims,
      'values': [float('%.2f' % u) for u in uniq]
    }
    ds_flat = data.ravel()
    all_inds = []

    for val in uniq:
        if val == 0:
            continue
        ind = [int(x) for x in list(np.where(ds_flat == val)[0])]  # UGH
        all_inds.append(ind)
    contents['indices'] = all_inds

    return package_json(contents)
