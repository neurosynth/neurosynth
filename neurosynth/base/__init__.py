# emacs: -*- mode: python-mode; py-indent-offset: 2; tab-width: 2; indent-tabs-mode: nil -*-
# ex: set sts=2 ts=2 sw=2 et:

__all__ = ['dataset', 'mask', 'imageutils',
           'mappable', 'transformations', 'lexparser']

# Verify nibabel version right here since nothing would work
import nibabel as nib
from distutils.version import LooseVersion
if LooseVersion(nib.__version__) < LooseVersion("1.2.0"):
    raise ImportError("Neurosynth requires nibabel >= 1.2.0")
