#emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 noet:
""" Network analysis-related methods """

from neurosynth.analysis import meta

def coactivation(dataset, seed, threshold=0.0, outroot=None):
    """ Compute and save coactivation map given input image as seed.

    This is essentially just a wrapper for a meta-analysis defined
    by the contrast between those studies that activate within the seed
    and those that don't.

    Args:
        dataset: a Dataset instance containing study and activation data.
        seed: a Nifti or Analyze image defining the boundaries of the seed
            region. Note that voxels do not need to be contiguous
        threshold: optional float indicating the threshold above which voxels
            are considered to be part of the seed ROI (default = 0)
        outroot: optional string to prepend to all coactivation images.
            If none, defaults to using the first part of the seed filename.
    """

    studies = dataset.get_ids_by_mask(seed, threshold=threshold)
    ma = meta.MetaAnalysis(dataset, studies)
    if outroot is None:
        outroot = seed.split('.')[0] + "_coact"
    ma.save_results(outroot)
