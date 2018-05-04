""" Dimensionality reduction methods"""

import os
import numpy as np
from neurosynth.base.dataset import Dataset
from neurosynth.base import imageutils
from neurosynth.tests.utils import get_resource_path
import logging
import subprocess
import pandas as pd
import shutil
from os.path import dirname, join

logger = logging.getLogger('neurosynth.cluster')


def average_within_regions(dataset, regions, masker=None, threshold=None,
                           remove_zero=True):
    """ Aggregates over all voxels within each ROI in the input image.

    Takes a Dataset and a Nifti image that defines distinct regions, and
    returns a numpy matrix of  ROIs x mappables, where the value at each
    ROI is the proportion of active voxels in that ROI. Each distinct ROI
    must have a unique value in the image; non-contiguous voxels with the
    same value will be assigned to the same ROI.

    Args:
        dataset: Either a Dataset instance from which image data are
            extracted, or a Numpy array containing image data to use. If
            the latter, the array contains voxels in rows and
            features/studies in columns. The number of voxels must be equal
            to the length of the vectorized image mask in the regions
            image.
        regions: An image defining the boundaries of the regions to use.
            Can be one of:
            1) A string name of the NIFTI or Analyze-format image
            2) A NiBabel SpatialImage
            3) A list of NiBabel images
            4) A 1D numpy array of the same length as the mask vector in
                the Dataset's current Masker.
        masker: Optional masker used to load image if regions is not a
            numpy array. Must be passed if dataset is a numpy array.
        threshold: An optional float in the range of 0 - 1 or integer. If
            passed, the array will be binarized, with ROI values above the
            threshold assigned to True and values below the threshold
            assigned to False. (E.g., if threshold = 0.05, only ROIs in
            which more than 5% of voxels are active will be considered
            active.) If threshold is integer, studies will only be
            considered active if they activate more than that number of
            voxels in the ROI.
        remove_zero: An optional boolean; when True, assume that voxels
            with value of 0 should not be considered as a separate ROI, and
            will be ignored.

    Returns:
        A 2D numpy array with ROIs in rows and mappables in columns.
    """

    if masker is not None:
        masker = masker
    else:
        if isinstance(dataset, Dataset):
            masker = dataset.masker
        else:
            if not type(regions).__module__.startswith('numpy'):
                raise ValueError(
                    "If dataset is a numpy array and regions is not a numpy "
                    "array, a masker must be provided.")

    if not type(regions).__module__.startswith('numpy'):
        regions = masker.mask(regions)

    if isinstance(dataset, Dataset):
        dataset = dataset.get_image_data(dense=False)

    # If multiple images are passed, give each one a unique value
    if regions.ndim == 2:
        m = regions
        for i in range(regions.shape[1]):
            _nz = np.nonzero(m[:, i])[0]
            if isinstance(threshold, int):
                m[_nz, i] = 1.0
            else:
                m[_nz, i] = 1.0 / np.count_nonzero(m[:, i])

    # Otherwise create an ROI-coding matrix
    else:
        labels = np.unique(regions)

        if remove_zero:
            labels = labels[np.nonzero(labels)]

        n_regions = labels.size

        m = np.zeros((regions.size, n_regions))
        for i in range(n_regions):
            if isinstance(threshold, int):
                m[regions == labels[i], i] = 1.0
            else:
                m[regions == labels[i], i] = 1.0 / \
                    np.sum(regions == labels[i])

    # Call dot() on the array itself as this will use sparse matrix
    # multiplication if possible.
    result = dataset.T.dot(m).T

    if threshold is not None:
        result[result < threshold] = 0.0
        result = result.astype(bool)

    return result


def apply_grid(dataset, masker=None, scale=5, threshold=None):
    """ Imposes a 3D grid on the brain volume and averages across all voxels
    that fall within each cell.
    Args:
        dataset: Data to apply grid to. Either a Dataset instance, or a numpy
            array with voxels in rows and features in columns.
        masker: Optional Masker instance used to map between the created grid
            and the dataset. This is only needed if dataset is a numpy array;
            if dataset is a Dataset instance, the Masker in the dataset will
            be used.
        scale: int; scaling factor (in mm) to pass onto create_grid().
        threshold: Optional float to pass to reduce.average_within_regions().
    Returns:
        A tuple of length 2, where the first element is a numpy array of
        dimensions n_cubes x n_studies, and the second element is a numpy
        array, with the same dimensions as the Masker instance in the current
        Dataset, that maps voxel identities onto cell IDs in the grid.
    """
    if masker is None:
        if isinstance(dataset, Dataset):
            masker = dataset.masker
        else:
            raise ValueError(
                "If dataset is a numpy array, a masker must be provided.")

    grid = imageutils.create_grid(masker.volume, scale)
    cm = masker.mask(grid, in_global_mask=True)
    data = average_within_regions(dataset, cm, threshold)
    return (data, grid)


def get_random_voxels(dataset, n_voxels):
    """ Returns mappable data for a random subset of voxels.

    May be useful as a baseline in predictive analyses--e.g., to compare
    performance of a more principled feature selection method with simple
    random selection.

    Args:
        dataset: A Dataset instance
        n_voxels: An integer specifying the number of random voxels to select.

    Returns:
        A 2D numpy array with (randomly-selected) voxels in rows and mappables
        in columns.
    """
    voxels = np.arange(dataset.masker.n_vox_in_vol)
    np.random.shuffle(voxels)
    selected = voxels[0:n_voxels]
    return dataset.get_image_data(voxels=selected)


def _get_top_words(model, feature_names, n_top_words=40):
    """ Return top forty words from each topic in trained topic model.
    """
    topic_words = []
    for topic in model.components_:
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]
        topic_words += [top_words]
    return topic_words


def run_lda(abstracts, n_topics=50, n_words=31, n_iters=1000, alpha=None,
            beta=0.001):
    """ Perform topic modeling using Latent Dirichlet Allocation with the
    Java toolbox MALLET.

    Args:
        abstracts:  A pandas DataFrame with two columns ('pmid' and 'abstract')
                    containing article abstracts.
        n_topics:   Number of topics to generate. Default=50.
        n_words:    Number of top words to return for each topic. Default=31,
                    based on Poldrack et al. (2012).
        n_iters:    Number of iterations to run in training topic model.
                    Default=1000.
        alpha:      The Dirichlet prior on the per-document topic
                    distributions.
                    Default: 50 / n_topics, based on Poldrack et al. (2012).
        beta:       The Dirichlet prior on the per-topic word distribution.
                    Default: 0.001, based on Poldrack et al. (2012).

    Returns:
        weights_df: A pandas DataFrame derived from the MALLET
                    output-doc-topics output file. Contains the weight assigned
                    to each article for each topic, which can be used to select
                    articles for topic-based meta-analyses (accepted threshold
                    from Poldrack article is 0.001). [n_topics]+1 columns:
                    'pmid' is the first column and the following columns are
                    the topic names. The names of the topics match the names
                    in df (e.g., topic_000).
        keys_df:    A pandas DataFrame derived from the MALLET
                    output-topic-keys output file. Contains the top [n_words]
                    words for each topic, which can act as a summary of the
                    topic's content. Two columns: 'topic' and 'terms'. The
                    names of the topics match the names in weights (e.g.,
                    topic_000).
    """
    if abstracts.index.name != 'pmid':
        abstracts.index = abstracts['pmid']

    resdir = os.path.abspath(get_resource_path())
    tempdir = os.path.join(resdir, 'topic_models')
    absdir = os.path.join(tempdir, 'abstracts')
    if not os.path.isdir(tempdir):
        os.mkdir(tempdir)

    if alpha is None:
        alpha = 50. / n_topics

    # Check for presence of abstract files and convert if necessary
    if not os.path.isdir(absdir):
        print('Abstracts folder not found. Creating abstract files...')
        os.mkdir(absdir)
        for pmid in abstracts.index.values:
            abstract = abstracts.loc[pmid]['abstract']
            with open(os.path.join(absdir, str(pmid) + '.txt'), 'w') as fo:
                fo.write(abstract)

    # Run MALLET topic modeling
    print('Generating topics...')
    mallet_bin = join(dirname(dirname(__file__)),
                      'resources/mallet/bin/mallet')
    import_str = ('{mallet} import-dir '
                  '--input {absdir} '
                  '--output {outdir}/topic-input.mallet '
                  '--keep-sequence '
                  '--remove-stopwords').format(mallet=mallet_bin,
                                               absdir=absdir,
                                               outdir=tempdir)

    train_str = ('{mallet} train-topics '
                 '--input {out}/topic-input.mallet '
                 '--num-topics {n_topics} '
                 '--num-top-words {n_words} '
                 '--output-topic-keys {out}/topic_keys.txt '
                 '--output-doc-topics {out}/doc_topics.txt '
                 '--num-iterations {n_iters} '
                 '--output-model {out}/saved_model.mallet '
                 '--random-seed 1 '
                 '--alpha {alpha} '
                 '--beta {beta}').format(mallet=mallet_bin, out=tempdir,
                                         n_topics=n_topics, n_words=n_words,
                                         n_iters=n_iters,
                                         alpha=alpha, beta=beta)

    subprocess.call(import_str, shell=True)
    subprocess.call(train_str, shell=True)

    # Read in and convert doc_topics and topic_keys.
    def clean_str(string):
        return os.path.basename(os.path.splitext(string)[0])

    def get_sort(lst):
        return [i[0] for i in sorted(enumerate(lst), key=lambda x:x[1])]

    topic_names = ['topic_{0:03d}'.format(i) for i in range(n_topics)]

    # doc_topics: Topic weights for each paper.
    # The conversion here is pretty ugly at the moment.
    # First row should be dropped. First column is row number and can be used
    # as the index.
    # Second column is 'file: /full/path/to/pmid.txt' <-- Parse to get pmid.
    # After that, odd columns are topic numbers and even columns are the
    # weights for the topics in the preceding column. These columns are sorted
    # on an individual pmid basis by the weights.
    n_cols = (2 * n_topics) + 1
    dt_df = pd.read_csv(os.path.join(tempdir, 'doc_topics.txt'),
                        delimiter='\t', skiprows=1, header=None, index_col=0)
    dt_df = dt_df[dt_df.columns[:n_cols]]

    # Get pmids from filenames
    dt_df[1] = dt_df[1].apply(clean_str)

    # Put weights (even cols) and topics (odd cols) into separate dfs.
    weights_df = dt_df[dt_df.columns[2::2]]
    weights_df.index = dt_df[1]
    weights_df.columns = range(n_topics)

    topics_df = dt_df[dt_df.columns[1::2]]
    topics_df.index = dt_df[1]
    topics_df.columns = range(n_topics)

    # Sort columns in weights_df separately for each row using topics_df.
    sorters_df = topics_df.apply(get_sort, axis=1)
    weights = weights_df.as_matrix()
    sorters = sorters_df.as_matrix()
    # there has to be a better way to do this.
    for i in range(sorters.shape[0]):
        weights[i, :] = weights[i, sorters[i, :]]

    # Define topic names (e.g., topic_000)
    index = dt_df[1]
    weights_df = pd.DataFrame(columns=topic_names, data=weights, index=index)
    weights_df.index.name = 'pmid'

    # topic_keys: Top [n_words] words for each topic.
    keys_df = pd.read_csv(os.path.join(tempdir, 'topic_keys.txt'),
                          delimiter='\t', header=None, index_col=0)

    # Second column is a list of the terms.
    keys_df = keys_df[[2]]
    keys_df.rename(columns={2: 'terms'}, inplace=True)
    keys_df.index = topic_names
    keys_df.index.name = 'topic'

    # Remove all temporary files (abstract files, model, and outputs).
    shutil.rmtree(tempdir)

    # Return article topic weights and topic keys.
    return weights_df, keys_df
