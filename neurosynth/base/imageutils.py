from nibabel import nifti1
import numpy as np

""" Miscellaneous image-related functions. """

def get_sphere(coords, r=4, vox_dims=(2,2,2), dims=(91,109,91)):
  """ # Return all points within r mm of coordinates. Generates a cube 
  and then discards all points outside sphere. Only returns values that
  fall within the dimensions of the image."""
  r = float(r)
  xx, yy, zz = [slice(-r/vox_dims[i], r/vox_dims[i]+0.01, 1) for i in range(len(coords))]
  cube = np.vstack([row.ravel() for row in np.mgrid[xx,yy,zz]])
  sphere = cube[:, np.sum(np.dot(np.diag(vox_dims), cube)**2,0)**.5<=r]
  sphere = np.round(sphere.T + coords)
  return sphere[(np.min(sphere, 1)>=0) & (np.max(np.subtract(sphere, dims), 1)<=-1), :].astype(int)


def map_peaks_to_image(peaks, r=4, vox_dims=(2,2,2), dims=(91,109,91), header=None):
  """ Take a set of discrete foci (i.e., 2-D array of xyz coordinates)
  and generate a corresponding image, convolving each focus with a 
  hard sphere of radius r."""
  data = np.zeros(dims)
  for p in peaks:
    valid = get_sphere(p, r, vox_dims, dims)
    valid = valid[:, ::-1]
    data[tuple(valid.T)] = 1
  # affine = header.get_sform() if header else None
  return nifti1.Nifti1Image(data, None, header=header)
  

def disjunction(images):
  """ Returns a binary disjunction of all passed images, i.e., value=1
  at any voxel that's non-zero in at least one image."""
  pass


def conjunction(images):
  """ Returns a binary conjunction of all passed images, i.e., value=1
  at any voxel that's non-zero in at least one image."""
  pass

def load_imgs(filenames, mask):
  """ Load multiple images from file into an ndarray.

  Args:
    filenames: A list of filenames pointing to valid images.
    mask: A Mask instance.

  Returns:
    An m x n 2D numpy array, where m = number of voxels in mask and 
    n = number of images passed.
  """
  data = np.zeros((mask.num_vox_in_mask, len(filenames)))
  for i, f in enumerate(filenames):
    data[:,i] = mask.mask(f)
  return data


def save_img(data, filename, mask, header=None):
  """ Save a vectorized image to file. """
  # data = data.astype(dtype)
  if not header:
    header = mask.get_header()
  header.set_data_dtype(data.dtype) # Avoids loss of precision
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


