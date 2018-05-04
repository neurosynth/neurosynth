""" Base classes for representing and manipulating data. """

import logging
import re
import os
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from neurosynth.base import mask, imageutils, transformations
from neurosynth.base import lexparser as lp
from neurosynth.utils import deprecated

# For Python 2/3 compatibility
from six import string_types
from functools import reduce
try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen
try:
    import cPickle as pickle
except:
    import pickle

logger = logging.getLogger('neurosynth.dataset')


def download(path='.', url=None, unpack=False):
    """ Download the latest data files.
    Args:
        path (str): Location to save the retrieved data files. Defaults to
            current directory.
        unpack (bool): If True, unzips the data file post-download.
    """

    if url is None:
        url = 'https://github.com/neurosynth/neurosynth-data/blob/master/current_data.tar.gz?raw=true'
    if os.path.exists(path) and os.path.isdir(path):
        basename = os.path.basename(url).split('?')[0]
        filename = os.path.join(path, basename)
    else:
        filename = path

    f = open(filename, 'wb')

    u = urlopen(url)
    file_size = int(u.headers["Content-Length"][0])
    print("Downloading the latest Neurosynth files: {0} bytes: {1}".format(
        url, file_size))

    bytes_dl = 0
    block_size = 8192
    while True:
        buffer = u.read(block_size)
        if not buffer:
            break
        bytes_dl += len(buffer)
        f.write(buffer)
        p = float(bytes_dl) / file_size
        status = r"{0}  [{1:.2%}]".format(bytes_dl, p)
        status = status + chr(8) * (len(status) + 1)
        sys.stdout.write(status)

    f.close()

    if unpack:
        import tarfile
        tarfile.open(filename, 'r:gz').extractall(os.path.dirname(filename))


def download_abstracts(dataset, path='.', email=None, out_file=None):
    """ Download the abstracts for a dataset/list of pmids
    """
    try:
        from Bio import Entrez, Medline
    except:
        raise Exception(
            'Module biopython is required for downloading abstracts from PubMed.')

    if email is None:
        raise Exception('No email address provided.')
    Entrez.email = email

    if isinstance(dataset, Dataset):
        pmids = dataset.image_table.ids.astype(str).tolist()
    elif isinstance(dataset, list):
        pmids = [str(pmid) for pmid in dataset]
    else:
        raise Exception(
            'Dataset type not recognized: {0}'.format(type(dataset)))

    records = []
    # PubMed only allows you to search ~1000 at a time. I chose 900 to be safe.
    chunks = [pmids[x: x + 900] for x in range(0, len(pmids), 900)]
    for chunk in chunks:
        h = Entrez.efetch(db='pubmed', id=chunk, rettype='medline',
                          retmode='text')
        records += list(Medline.parse(h))

    # Pull data for studies with abstracts
    data = [[study['PMID'], study['AB']]
            for study in records if study.get('AB', None)]
    df = pd.DataFrame(columns=['pmid', 'abstract'], data=data)
    if out_file is not None:
        df.to_csv(os.path.join(os.path.abspath(path), out_file), index=False)
    return df


class Dataset(object):

    """ Base Dataset class.

    The core data-representing object in Neurosynth. Internally stores
    information about both reported activations and tagged features. Provides a
    variety of methods for manipulating and retrieving various kinds of data.

    The Dataset is typically initialized by passing in a database file as the
    first argument. At minimum, the input file must contain tab-delimited
    columns named x, y, z, id, and space (case-insensitive). The x/y/z columns
    indicate the coordinates of the activation center or peak, the id column is
    used to group multiple activations from a single study. Typically the id
    should be a uniquely identifying field accessible to others, e.g., a PubMed
    ID in the case of entire articles. The space column indicates the nominal
    atlas used to produce each activation. Currently all values except 'TAL'
    (Talairach) will be ignored. If space == TAL and the transform argument is
    True, all activations reported in Talairach space will be converted to MNI
    space using the Lancaster et al. transform.

    Args:
        filename (str): The name of a database file containing a list of
            activations.
        feature_filename (str): An optional filename to construct a
            FeatureTable from.
        masker (str): An optional Nifti/Analyze image name defining the space
            to use for all operations. If no image is passed, defaults to the
            MNI152 2 mm template packaged with FSL.
        r (int): An optional integer specifying the radius of the smoothing
            kernel, in mm. Defaults to 6 mm.
        transform (bool, dict): Optional argument specifying how to handle
            transformation between coordinates reported in different
            stereotactic spaces. When True (default), activations in Talairach
            (T88) space will be converted to MNI space using the Lancaster et
            al (2007) transform; no other transformations will be applied. When
            False, no transformation will be applied. Alternatively, the user
            can pass their own dictionary of named transformations to apply, in
            which case each activation will be checked against the dictionary
            as it is read in and the specified transformation will be applied
            if found (for further details, see transformations.Transformer).
        target (str): The name of the target space within which activation
            coordinates are represented. By default, MNI.
        kwargs (dict): Additional optional arguments passed to add_features().
    """

    def __init__(
            self, filename, feature_filename=None, masker=None, r=6,
            transform=True, target='MNI', **kwargs):

        # Instance properties
        self.r = r

        # Set up transformations between different image spaces
        if transform:
            if not isinstance(transform, dict):
                transform = {'T88': transformations.t88_to_mni(),
                             'TAL': transformations.t88_to_mni()
                             }
            self.transformer = transformations.Transformer(transform, target)
        else:
            self.transformer = None

        # Load and process activation data
        self.activations = self._load_activations(filename)

        # Load the volume into a new Masker
        if masker is None:
            resource_dir = os.path.join(os.path.dirname(__file__),
                                        os.path.pardir,
                                        'resources')
            masker = os.path.join(
                resource_dir, 'MNI152_T1_2mm_brain.nii.gz')
        self.masker = mask.Masker(masker)

        # Create supporting tables for images and features
        self.create_image_table()
        if feature_filename is not None:
            self.add_features(feature_filename, **kwargs)

    def _load_activations(self, filename):
        """ Load activation data from a text file.

        Args:
            filename (str): a string pointing to the location of the txt file
                to read from.
        """
        logger.info("Loading activation data from %s..." % filename)

        activations = pd.read_csv(filename, sep='\t')
        activations.columns = [col.lower()
                               for col in list(activations.columns)]

        # Make sure all mandatory columns exist
        mc = ['x', 'y', 'z', 'id', 'space']
        if (set(mc) - set(list(activations.columns))):
            logger.error(
                "At least one of mandatory columns (x, y, z, id, and space) "
                "is missing from input file.")
            return

        # Transform to target space where needed
        spaces = activations['space'].unique()
        xyz = activations[['x', 'y', 'z']].values
        for s in spaces:
            if s != self.transformer.target:
                inds = activations['space'] == s
                xyz[inds] = self.transformer.apply(s, xyz[inds])
        activations[['x', 'y', 'z']] = xyz

        # xyz --> ijk
        ijk = pd.DataFrame(
            transformations.xyz_to_mat(xyz), columns=['i', 'j', 'k'])
        activations = pd.concat([activations, ijk], axis=1)
        return activations

    def create_image_table(self, r=None):
        """ Create and store a new ImageTable instance based on the current
        Dataset. Will generally be called privately, but may be useful as a
        convenience method in cases where the user wants to re-generate the
        table with a new smoothing kernel of different radius.

        Args:
            r (int): An optional integer indicating the radius of the smoothing
                kernel. By default, this is None, which will keep whatever
                value is currently set in the Dataset instance.
        """
        logger.info("Creating image table...")
        if r is not None:
            self.r = r
        self.image_table = ImageTable(self)

    def get_studies(self, features=None, expression=None, mask=None,
                    peaks=None, frequency_threshold=0.001,
                    activation_threshold=0.0, func=np.sum, return_type='ids',
                    r=6
                    ):
        """ Get IDs or data for studies that meet specific criteria.

        If multiple criteria are passed, the set intersection is returned. For
        example, passing expression='emotion' and mask='my_mask.nii.gz' would
        return only those studies that are associated with emotion AND report
        activation within the voxels indicated in the passed image.

        Args:
            ids (list): A list of IDs of studies to retrieve.
            features (list or str): The name of a feature, or a list of
                features, to use for selecting studies.
            expression (str): A string expression to pass to the PEG for study
                retrieval.
            mask: the mask image (see Masker documentation for valid data
                types).
            peaks (ndarray or list): Either an n x 3 numpy array, or a list of
                lists or tuples (e.g., [(-10, 22, 14)]) specifying the world
                (x/y/z) coordinates of the target location(s).
            frequency_threshold (float): For feature-based or expression-based
                selection, the threshold for selecting studies--i.e., the
                cut-off for a study to be included. Must be a float in range
                [0, 1].
            activation_threshold (int or float): For mask-based selection,
                threshold for a study to be included based on amount of
                activation displayed. If an integer, represents the absolute
                number of voxels that must be active within the mask in order
                for a study to be selected. If a float, it represents the
                proportion of voxels that must be active.
            func (Callable): The function to use when aggregating over the list
                of features. See documentation in FeatureTable.get_ids() for a
                full explanation. Only used for feature- or expression-based
                selection.
            return_type (str): A string specifying what data to return. Valid
                options are:
                'ids': returns a list of IDs of selected studies.
                'images': returns a voxel x study matrix of data for all
                selected studies.
                'weights': returns a dict where the keys are study IDs and the
                values are the computed weights. Only valid when performing
                feature-based selection.
            r (int): For peak-based selection, the distance cut-off (in mm)
                for inclusion (i.e., only studies with one or more activations
                within r mm of one of the passed foci will be returned).

        Returns:
            When return_type is 'ids' (default), returns a list of IDs of the
            selected studies. When return_type is 'data', returns a 2D numpy
            array, with voxels in rows and studies in columns. When return_type
            is 'weights' (valid only for expression-based selection), returns
            a dict, where the keys are study IDs, and the values are the
            computed weights.

        Examples
        --------
        Select all studies tagged with the feature 'emotion':

            >>> ids = dataset.get_studies(features='emotion')

        Select all studies that activate at least 20% of voxels in an amygdala
        mask, and retrieve activation data rather than IDs:

            >>> data = dataset.get_studies(mask='amygdala_mask.nii.gz',
                threshold=0.2, return_type='images')

        Select studies that report at least one activation within 12 mm of at
        least one of three specific foci:

            >>> ids = dataset.get_studies(peaks=[[12, -20, 30], [-26, 22, 22],
                                                [0, 36, -20]], r=12)

        """
        results = []

        # Feature-based selection
        if features is not None:
            # Need to handle weights as a special case, because we can't
            # retrieve the weights later using just the IDs.
            if return_type == 'weights':
                if expression is not None or mask is not None or \
                        peaks is not None:
                    raise ValueError(
                        "return_type cannot be 'weights' when feature-based "
                        "search is used in conjunction with other search "
                        "modes.")
                return self.feature_table.get_ids(
                    features, frequency_threshold, func, get_weights=True)
            else:
                results.append(self.feature_table.get_ids(
                    features, frequency_threshold, func))

        # Logical expression-based selection
        if expression is not None:
            _ids = self.feature_table.get_ids_by_expression(
                expression, frequency_threshold, func)
            results.append(list(_ids))

        # Mask-based selection
        if mask is not None:
            mask = self.masker.mask(mask, in_global_mask=True).astype(bool)
            num_vox = np.sum(mask)
            prop_mask_active = self.image_table.data.T.dot(mask).astype(float)
            if isinstance(activation_threshold, float):
                prop_mask_active /= num_vox
            indices = np.where(prop_mask_active > activation_threshold)[0]
            results.append([self.image_table.ids[ind] for ind in indices])

        # Peak-based selection
        if peaks is not None:
            r = float(r)
            found = set()
            for p in peaks:
                xyz = np.array(p, dtype=float)
                x = self.activations['x']
                y = self.activations['y']
                z = self.activations['z']
                dists = np.sqrt(np.square(x - xyz[0]) + np.square(y - xyz[1]) +
                                np.square(z - xyz[2]))
                inds = np.where((dists > 5.5) & (dists < 6.5))[0]
                tmp = dists[inds]
                found |= set(self.activations[dists <= r]['id'].unique())
            results.append(found)

        # Get intersection of all sets
        ids = list(reduce(lambda x, y: set(x) & set(y), results))

        if return_type == 'ids':
            return ids
        elif return_type == 'data':
            return self.get_image_data(ids)

    def add_features(self, features, append=True, merge='outer',
                     duplicates='ignore', min_studies=0.0, threshold=0.001):
        """ Construct a new FeatureTable from file.

        Args:
            features: Feature data to add. Can be:
                (a) A text file containing the feature data, where each row is
                a study in the database, with features in columns. The first
                column must contain the IDs of the studies to match up with the
                image data.
                (b) A pandas DataFrame, where studies are in rows, features are
                in columns, and the index provides the study IDs.
            append (bool): If True, adds new features to existing ones
                incrementally. If False, replaces old features.
            merge, duplicates, min_studies, threshold: Additional arguments
                passed to FeatureTable.add_features().
         """
        if (not append) or not hasattr(self, 'feature_table'):
            self.feature_table = FeatureTable(self)

        self.feature_table.add_features(features, merge=merge,
                                        duplicates=duplicates,
                                        min_studies=min_studies,
                                        threshold=threshold)

    def get_image_data(self, ids=None, voxels=None, dense=True):
        """ A convenience wrapper for ImageTable.get_image_data().

        Args:
            ids (list, array): A list or 1D numpy array of study ids to
                return. If None, returns data for all studies.
            voxels (list, array): A list or 1D numpy array of voxel indices
                (i.e., rows) to return. If None, returns data for all voxels.
        """
        return self.image_table.get_image_data(ids, voxels=voxels, dense=dense)

    def get_feature_data(self, ids=None, **kwargs):
        """ A convenience wrapper for FeatureTable.get_image_data(). """
        return self.feature_table.get_feature_data(ids, **kwargs)

    def get_feature_names(self, features=None):
        """ Returns names of features. If features is None, returns all
        features. Otherwise assumes the user is trying to find the order of the
        features.  """
        if features:
            return self.feature_table.get_ordered_names(features)
        else:
            return self.feature_table.feature_names

    def get_feature_counts(self, threshold=0.001):
        """ Returns a dictionary, where the keys are the feature names
        and the values are the number of studies tagged with the feature. """
        counts = np.sum(self.get_feature_data() >= threshold, 0)
        return dict(zip(self.get_feature_names(), list(counts)))

    @classmethod
    def load(cls, filename):
        """ Load a pickled Dataset instance from file. """
        try:
            dataset = pickle.load(open(filename, 'rb'))
        except UnicodeDecodeError:
            # Need to try this for python3
            dataset = pickle.load(open(filename, 'rb'), encoding='latin')

        if hasattr(dataset, 'feature_table'):
            dataset.feature_table._csr_to_sdf()
        return dataset

    def save(self, filename):
        """ Pickle the Dataset instance to the provided file.
        """
        if hasattr(self, 'feature_table'):
            self.feature_table._sdf_to_csr()

        pickle.dump(self, open(filename, 'wb'), -1)

        if hasattr(self, 'feature_table'):
            self.feature_table._csr_to_sdf()


class ImageTable(object):

    """ Represents image data from multiple studies in an accessible form.

    Args:
        dataset (Dataset): Dataset instance to pull inputs from.
        r (int): The radius of the sphere used for smoothing (default = 6 mm).
        use_sparse (bool): Flag indicating whether or not to represent the data
            as a sparse array (generally this should be left to True, as these
            data are quite sparse and computation is often considerably slower
            in dense form.)
    """

    def __init__(self, dataset, r=6, use_sparse=True):
        activations, masker, r = dataset.activations, dataset.masker, dataset.r
        for var in [activations, masker, r]:
            assert var is not None
        self.ids = activations['id'].unique()
        self.masker = masker
        self.r = r

        n_studies = len(self.ids)
        data_shape = (self.masker.n_vox_in_vol, n_studies)
        if use_sparse:
            # Fancy indexing assignment is not supported for sparse matrices,
            # so let's keep lists of values and their indices (rows, cols) to
            # later construct the csr_matrix.
            vals, rows, cols = [], [], []
        else:
            self.data = np.zeros(data_shape, dtype=int)

        logger.info("Mapping %d studies into image space..." % (n_studies,))
        for i, (name, data) in enumerate(activations.groupby('id')):
            logger.debug("%s/%s..." % (str(i + 1), str(n_studies)))
            img = imageutils.map_peaks_to_image(
                data[['i', 'j', 'k']].values, r=r,
                header=self.masker.get_header())
            img_masked = self.masker.mask(img)
            if use_sparse:
                nz = np.nonzero(img_masked)
                assert(len(nz) == 1)
                vals += list(img_masked[nz])
                rows += list(nz[0])
                cols += [i] * len(nz[0])
            else:
                self.data[:, i] = img_masked

        if use_sparse:
            self.data = sparse.csr_matrix((
                vals, (rows, cols)), shape=data_shape)

    def get_image_data(self, ids=None, voxels=None, dense=True):
        """ Slices and returns a subset of image data.

        Args:
            ids (list, array): A list or 1D numpy array of study ids to
                return. If None, returns data for all studies.
            voxels (list, array): A list or 1D numpy array of voxel indices
                (i.e., rows) to return. If None, returns data for all voxels.
            dense (bool): Optional boolean. When True (default), convert the
                result to a dense array before returning. When False, keep as
                sparse matrix.

        Returns:
          A 2D numpy array with voxels in rows and studies in columns.
        """
        if dense and ids is None and voxels is None:
            logger.warning(
                "Warning: get_image_data() is being called without specifying "
                "a subset of studies or voxels to retrieve. This may result in"
                " a very large amount of data (several GB) being read into "
                "memory. If you experience any problems, consider returning a "
                "sparse matrix by passing dense=False, or pass in a list of "
                "ids of voxels to retrieve only a portion of the data.")

        result = self.data
        if ids is not None:
            idxs = np.where(np.in1d(np.array(self.ids), np.array(ids)))[0]
            result = result[:, idxs]
        if voxels is not None:
            result = result[voxels, :]
        return result.toarray() if dense else result

    def trim(self, ids):
        """ Trim ImageTable to keep only the passed studies. This is a
        convenience method, and should generally be avoided in favor of
        non-destructive alternatives that don't require slicing (e.g.,
            matrix multiplication). """
        self.data = self.get_image_data(ids, dense=False)  # .tocoo()
        idxs = np.where(np.in1d(np.array(self.ids), np.array(ids)))[0]
        self.ids = [self.ids[i] for i in idxs]

    def save_images_to_file(self, ids, outroot='./'):
        """ Reconstructs vectorized images corresponding to the specified
        study ids and saves them to file, prepending with the outroot
        (default: current directory). """
        pass

    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'), -1)


class FeatureTable(object):

    """ A FeatureTable instance stores a matrix of studies x features,
    along with associated manipulation methods. """

    def __init__(self, dataset):
        """ Initialize a new FeatureTable. Takes as input a parent DataSet
        instance and feature data (if provided). """
        self.dataset = dataset
        self.data = pd.DataFrame()

    def add_features(self, features, merge='outer', duplicates='ignore',
                     min_studies=0, threshold=0.0001):
        """ Add new features to FeatureTable.
        Args:
            features (str, DataFrame): A filename to load data from, or a
                pandas DataFrame. In either case, studies are in rows and
                features are in columns. Values in cells reflect the weight of
                the intersecting feature for the intersecting study. Feature
                names and study IDs should be included as the first column
                and first row, respectively.
            merge (str): The merge strategy to use when merging new features
                with old. This is passed to pandas.merge, so can be 'left',
                'right', 'outer', or 'inner'. Defaults to outer (i.e., all data
                in both new and old will be kept, and missing values will be
                assigned zeros.)
            duplicates (str): string indicating how to handle features whose
                name matches an existing feature. Valid options:
                    'ignore' (default): ignores the new feature, keeps old data
                    'replace': replace the old feature's data with the new data
                    'merge': keeps both features, renaming them so they're
                        different
            min_studies (int): minimum number of studies that pass threshold in
                order to add feature.
            threshold (float): minimum frequency threshold each study must
                exceed in order to count towards min_studies.
        """
        if isinstance(features, string_types):
            if not os.path.exists(features):
                raise ValueError("%s cannot be found." % features)
            try:
                features = pd.read_csv(features, sep='\t', index_col=0)
            except Exception as e:
                logger.error("%s cannot be parsed: %s" % (features, e))

        if min_studies:
            valid = np.where(
                (features.values >= threshold).sum(0) >= min_studies)[0]
            features = features.iloc[:, valid]

        # Warn user if no/few IDs match between the FeatureTable and the
        # Dataset. This most commonly happens because older database.txt files
        # used doi's as IDs whereas we now use PMIDs throughout.
        n_studies = len(features)
        n_common_ids = len(
            set(features.index) & set(self.dataset.image_table.ids))
        if float(n_common_ids) / n_studies < 0.01:  # Minimum 1% overlap
            msg = "Only %d" % n_common_ids if n_common_ids else "None of the"
            logger.warning(
                msg + " studies in the feature file matched studies currently "
                "the Dataset. The most likely cause for this is that you're "
                "pairing a newer feature set with an older, incompatible "
                "database file. You may want to try regenerating the Dataset "
                "instance from a newer database file that uses PMIDs rather "
                "than doi's as the study identifiers in the first column.")

        old_data = self.data.to_dense()
        # Handle features with duplicate names
        common_features = list(set(old_data.columns) & set(features.columns))
        if duplicates == 'ignore':
            features = features.drop(common_features, axis=1)
        elif duplicates == 'replace':
            old_data = old_data.drop(common_features, axis=1)

        data = old_data.merge(
            features, how=merge, left_index=True, right_index=True)
        self.data = data.fillna(0.0).to_sparse()

    @property
    def feature_names(self):
        return list(self.data.columns)

    def get_feature_data(self, ids=None, features=None, dense=True):
        """ Slices and returns a subset of feature data.

        Args:
            ids (list, array): A list or 1D numpy array of study ids to
                return rows for. If None, returns data for all studies
                (i.e., all rows in array).
            features (list, array): A list or 1D numpy array of named features
                to return. If None, returns data for all features (i.e., all
                columns in array).
            dense (bool): Optional boolean. When True (default), convert the
                result to a dense array before returning. When False, keep as
                sparse matrix. Note that if ids is not None, the returned array
                will always be dense.
        Returns:
          A pandas DataFrame with study IDs in rows and features incolumns.
        """
        result = self.data

        if ids is not None:
            result = result.ix[ids]

        if features is not None:
            result = result.ix[:, features]

        return result.to_dense() if dense else result

    def get_ordered_names(self, features):
        """ Given a list of features, returns features in order that they
        appear in database.

        Args:
            features (list): A list or 1D numpy array of named features to
            return.

        Returns:
            A list of features in order they appear in database.
        """

        idxs = np.where(
            np.in1d(self.data.columns.values, np.array(features)))[0]
        return list(self.data.columns[idxs].values)

    def get_ids(self, features, threshold=0.0, func=np.sum, get_weights=False):
        """ Returns a list of all studies in the table that meet the desired
        feature-based criteria.

        Will most commonly be used to retrieve studies that use one or more
        features with some minimum frequency; e.g.,:

            get_ids(['fear', 'anxiety'], threshold=0.001)

        Args:
            features (lists): a list of feature names to search on.
            threshold (float): optional float indicating threshold features
                must pass to be included.
            func (Callable): any numpy function to use for thresholding
                (default: sum). The function will be applied to the list of
                features and the result compared to the threshold. This can be
                used to change the meaning of the query in powerful ways. E.g,:
                    max: any of the features have to pass threshold
                        (i.e., max > thresh)
                    min: all features must each individually pass threshold
                        (i.e., min > thresh)
                    sum: the summed weight of all features must pass threshold
                        (i.e., sum > thresh)
            get_weights (bool): if True, returns a dict with ids => weights.

        Returns:
            When get_weights is false (default), returns a list of study
                names. When true, returns a dict, with study names as keys
                and feature weights as values.
        """
        if isinstance(features, str):
            features = [features]
        features = self.search_features(features)  # Expand wild cards
        feature_weights = self.data.ix[:, features]
        weights = feature_weights.apply(func, 1)
        above_thresh = weights[weights >= threshold]
        # ids_to_keep = self.ids[above_thresh]
        return above_thresh if get_weights else list(above_thresh.index)

    def search_features(self, search):
        ''' Returns all features that match any of the elements in the input
        list.

        Args:
            search (str, list): A string or list of strings defining the query.

        Returns:
            A list of matching feature names.
        '''
        if isinstance(search, string_types):
            search = [search]
        search = [s.replace('*', '.*') for s in search]
        cols = list(self.data.columns)
        results = []
        for s in search:
            results.extend([f for f in cols if re.match(s + '$', f)])
        return list(set(results))

    def get_ids_by_expression(self, expression, threshold=0.001, func=np.sum):
        """ Use a PEG to parse expression and return study IDs."""
        lexer = lp.Lexer()
        lexer.build()
        parser = lp.Parser(
            lexer, self.dataset, threshold=threshold, func=func)
        parser.build()
        return parser.parse(expression).keys().values

    def get_features_by_ids(self, ids=None, threshold=0.0001, func=np.mean,
                            get_weights=False):
        ''' Returns features for which the mean loading across all specified
        studies (in ids) is >= threshold. '''
        weights = self.data.ix[ids].apply(func, 0)
        above_thresh = weights[weights >= threshold]
        return above_thresh if get_weights else list(above_thresh.index)

    def _sdf_to_csr(self):
        """ Convert FeatureTable to SciPy CSR matrix. """
        data = self.data.to_dense()
        self.data = {
            'columns': list(data.columns),
            'index': list(data.index),
            'values': sparse.csr_matrix(data.values)
        }

    def _csr_to_sdf(self):
        """ Inverse of _sdf_to_csr(). """
        self.data = pd.DataFrame(self.data['values'].todense(),
                                 index=self.data['index'],
                                 columns=self.data['columns']).to_sparse()
