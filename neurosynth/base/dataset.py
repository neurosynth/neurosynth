# emacs: -*- mode: python-mode; py-indent-offset: 2; tab-width: 2; indent-tabs-mode: nil -*-
# ex: set sts=2 ts=2 sw=2 et:
""" A Neurosynth Dataset """

import logging
import re
import random
import os

import numpy as np
import pandas as pd
from scipy import sparse

import mappable

from neurosynth.base import mask, imageutils, transformations

logger = logging.getLogger('neurosynth.dataset')


class Dataset(object):

    def __init__(
        self, filename, feature_filename=None, masker=None, r=6, transform=True,
                  target='MNI'):
        """ Initialize a new Dataset instance.

        Creates a new Dataset instance from a text file containing activation data.
        At minimum, the input file must contain tab-delimited columns named x, y, z,
        id, and space (case-insensitive). The x/y/z columns indicate the coordinates
        of the activation center or peak, the id column is used to group multiple
        activations from a single Mappable (e.g. an article). Typically the id should
        be a uniquely identifying field accessible to others, e.g., a doi in the case
        of entire articles. The space column indicates the nominal atlas used to
        produce each activation. Currently all values except 'TAL' (Talairach) will
        be ignored. If space == TAL and the transform argument is True, all activations
        reported in Talairach space will be converted to MNI space using the
        Lancaster et al transform.

        Args:
          filename: The name of a database file containing a list of activations.
          feature_filename: An optional filename to construct a FeatureTable from.
          masker: An optional Nifti/Analyze image name defining the space to use for
            all operations. If no image is passed, defaults to the MNI152 2 mm
            template packaged with FSL.
          r: An optional integer specifying the radius of the smoothing kernel, in mm.
            Defaults to 6 mm.
          transform: Optional argument specifying how to handle transformation between
            coordinates reported in different stereotactic spaces. When True (default),
            activations in Talairach (T88) space will be converted to MNI space using
            the Lancaster et al (2007) transform; no other transformations will be
            applied. When False, no transformation will be applied. Alternatively,
            the user can pass their own dictionary of named transformations to apply,
            in which case each activation will be checked against the dictionary
            as it is read in and the specified transformation will be applied if
            found (for further explanation, see transformations.Transformer).
          target: The name of the target space within which activation coordinates
            are represented. By default, MNI.

        Returns:
          A Dataset instance.

        """

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

        # Load mappables
        self.mappables = self._load_mappables_from_txt(filename)

        # Load the volume into a new Masker
        try:
            if masker is None:
                resource_dir = os.path.join(os.path.dirname(__file__),
                                            os.path.pardir,
                                            'resources')
                masker = os.path.join(
                    resource_dir, 'MNI152_T1_2mm_brain.nii.gz')
            self.masker = mask.Masker(masker)
        except Exception as e:
            logger.error("Error loading masker %s: %s" % (masker, e))
            # yoh: TODO -- IMHO should re-raise or not even swallow the exception here
            # raise e

        # Create supporting tables for images and features
        self.create_image_table()
        if feature_filename is not None:
            self.feature_table = FeatureTable(self, feature_filename)

    def _load_mappables_from_txt(self, filename):
        """ Load mappables from a text file.

        Args:
          filename: a string pointing to the location of the txt file to read from.
        """
        logger.info("Loading mappables from %s..." % filename)
        
        # Read in with pandas
        contents = pd.read_csv(filename, sep='\t')
        contents.columns = [col.lower() for col in list(contents.columns)]

        # Make sure all mandatory columns exist
        mc = ['x', 'y', 'z', 'id', 'space']
        if (set(mc) - set(list(contents.columns))):
            logger.error(
                "At least one of mandatory columns (x, y, z, id, and space) is missing from input file.")
            return

        extra_fields = list(set(list(contents.columns)) - set(mc))  # save non-standard fields

        def get_mappable_data(grp):
            first = grp.iloc[0]
            d = { 'peaks': grp[['x','y','z']].values }
            for f in ['id', 'space'] + extra_fields:
                d[f] = first[f]
            return d

        data = list(contents.groupby('id').apply(get_mappable_data))

        # Initialize all mappables--for now, assume Articles are passed
        logger.info("Converting text to mappables...")
        return [mappable.Article(m, self.transformer) for m in data]

    def create_image_table(self, r=None):
        """ Create and store a new ImageTable instance based on the current Dataset.

        Will generally be called privately, but may be useful as a convenience
        method in cases where the user wants to re-generate the table with a
        new smoothing kernel of different radius.

        Args:
          r: An optional integer indicating the radius of the smoothing kernel.
            By default, this is None, which will keep whatever value is currently
            set in the Dataset instance.
        """
        logger.info("Creating image table...")
        if r is not None:
            self.r = r
        self.image_table = ImageTable(self)

    def add_mappables(self, filename=None, mappables=None, remap=True):
        """ Append new Mappable objects to the end of the list.

        Either a filename or a list of mappables must be passed.

        Args:
          filename: The location of the file to extract new mappables from.
          mappables: A list of Mappable instances to append to the current list.
          remap: Optional boolean indicating whether to regenerate the entire
            ImageTable after appending the new Mappables.
        """
        # TODO: (i) it would be more effiicent to only map the new Mappables into
        # the ImageTable instead of redoing everything. (ii) we should check for
        # duplicates and prompt whether to overwrite or update in cases where
        # conflicts occur.
        if filename != None:
            self.mappables.extend(self._load_mappables_from_txt(filename))
        elif mappables != None:
            self.mappables.extend(mappables)
        if remap:
            self.image_table = create_image_table()

    def delete_mappables(self, ids, remap=True):
        """ Delete specific Mappables from the Dataset.

        Note that 'ids' is a list of unique identifiers of the Mappables (e.g., doi's),
        and not indices in the current instance's mappables list.

        Args:
          ids: A list of ids corresponding to the Mappables to delete.
          remap: Optional boolean indicating whether to regenerate the entire
            ImageTable after deleting undesired Mappables.
        """
        self.mappables = [m for m in self.mappables if m not in ids]
        if remap:
            self.image_table = create_image_table()

    def get_mappables(self, ids, get_image_data=False):
        """ Takes a list of unique ids and returns corresponding Mappables.

        Args:
          ids: A list of ids of the mappables to return.
          get_image_data: An optional boolean. When True, returns a voxel x mappable matrix
            of image data rather than the Mappable instances themselves.

        Returns:
          If get_image_data is True, a 2D numpy array of voxels x Mappables. Otherwise, a
          list of Mappables.
        """
        if get_image_data:
            return self.get_image_data(ids)
        else:
            return [m for m in self.mappables if m.id in ids]

    def get_ids_by_features(self, features, threshold=None, func=np.sum, get_image_data=False, get_weights=False):
        """ A wrapper for FeatureTable.get_ids().

        Args:
          features: A list of features to use when selecting Mappables.
          threshold: Optional float between 0 and 1. If passed, the threshold will be used as
            a cut-off when selecting Mappables.
          func: The function to use when aggregating over the list of features. See
            documentation in FeatureTable.get_ids() for a full explanation.
          get_image_data: An optional boolean. When True, returns a voxel x mappable matrix
            of image data rather than the Mappable instances themselves.
        """
        ids = self.feature_table.get_ids(features, threshold, func, get_weights)
        return self.get_image_data(ids) if get_image_data else ids

    def get_ids_by_expression(self, expression, threshold=0.001, func=np.sum, get_image_data=False):
        ids = self.feature_table.get_ids_by_expression(expression, threshold, func)
        return self.get_image_data(ids) if get_image_data else ids

    def get_ids_by_mask(self, mask, threshold=0.0, get_image_data=False):
        """ Return all mappable objects that activate within the bounds
        defined by the mask image. Optional threshold parameter specifies
        the proportion of voxels within the mask that must be active to
        warrant inclusion. E.g., if threshold = 0.1, only mappables with
        > 10% of voxels activated in mask will be returned. """
        mask = self.masker.mask(mask).astype(bool)
        num_vox = np.sum(mask)
        prop_mask_active = self.image_table.data.T.dot(
            mask).astype(float) / num_vox
        indices = np.where(prop_mask_active > threshold)[0]
        return self.get_image_data(indices) if get_image_data else [self.image_table.ids[ind] for ind in indices]


    def get_ids_by_peaks(self, peaks, r=10, threshold=0.0, get_image_data=False):
        """ A wrapper for get_ids_by_mask. Takes a set of xyz coordinates and generates
        a new Nifti1Image to use as a mask.

        Args:
          peaks: Either an n x 3 numpy array, or a list of lists (e.g., [[-10, 22, 14]])
            specifying the world (x/y/z) coordinates of the target location(s).
          r: Radius in millimeters of the sphere to grow around each location.
          threshold: Optional float indicating the proportion of voxels that must be
            active in order for a Mappable to be considered active.
          get_image_data: If true, returns the image data for all activated Mappables in
            a voxel x Mappable numpy array. Otherwise, returns just the IDs of Mappables.

        Returns:
          Either a list of ids (if get_image_data = False) or a numpy array of image data.

        """
        peaks = np.array(peaks)  # Make sure we have a numpy array
        peaks = transformations.xyz_to_mat(peaks)
        img = imageutils.map_peaks_to_image(
            peaks, r, vox_dims=self.masker.vox_dims,
            dims=self.masker.dims, header=self.masker.get_header())
        return self.get_ids_by_mask(img, threshold, get_image_data=get_image_data)

    def add_features(self, filename, description=''):
        """ Construct a new FeatureTable from file. Note: this is destructive, and will
        overwrite existing FeatureTable. Need to add merging operations that gracefully
        handle missing studies and conflicting feature names. """
        self.feature_table = FeatureTable(self, filename, description)

    def get_image_data(self, ids=None, voxels=None, dense=True):
        """ A convenience wrapper for ImageTable.get_image_data(). """
        return self.image_table.get_image_data(ids, voxels=voxels, dense=dense)

    def get_feature_data(self, ids=None, **kwargs):
        """ A convenience wrapper for FeatureTable.get_image_data(). """
        return self.feature_table.get_feature_data(ids, **kwargs)

    def get_feature_names(self, features=None):
        """ Returns names of features. If features is None, returns all features.
        Otherwise assumes the user is trying to find the order of the features.  """
        if features:
            return self.feature_table.get_ordered_names(features)
        else:
            return self.feature_table.feature_names

    def get_feature_counts(self, func=np.sum, threshold=0.001):
        """ Returns a dictionary, where the keys are the feature names
        and the values are the number of studies tagged with the feature. """
        result = {}
        for f in self.get_feature_names():
            result[f] = len(self.get_ids_by_features([f], func=func, threshold=threshold))
        return result

    @classmethod
    def load(cls, filename):
        """ Load a pickled Dataset instance from file. """
        import cPickle
        dataset = cPickle.load(open(filename, 'rb'))
        dataset.feature_table._csr_to_sdf()
        return dataset

    def save(self, filename, keep_mappables=False):
        """ Pickle the Dataset instance to the provided file.

        If keep_mappables = False (default), will delete the Mappable objects
        themselves before pickling. This will save a good deal of space and
        is generally advisable once a stable Dataset is created, as the
        Mappables are rarely used after the ImageTable is generated.
        """
        if not keep_mappables:
            self.mappables = []

        self.feature_table._sdf_to_csr()

        import cPickle
        cPickle.dump(self, open(filename, 'wb'), -1)

        self.feature_table._csr_to_sdf()

    def to_json(self, filename=None):
        """ Save the Dataset to file in JSON format.

        This is not recommended, as the resulting file will typically be several
        GB in size. If no filename is provided, returns the JSON string.
        """
        import json
        mappables = [m.to_json() for m in self.mappables]
        json_string = json.dumps({'mappables': mappables})
        if filename is not None:
            open(filename, 'w').write(json_string)
        else:
            return json_string


class ImageTable(object):

    def __init__(self, dataset=None, mappables=None, masker=None, r=6, use_sparse=True):
        """ Initialize a new ImageTable.

        If a Dataset instance is passed, all inputs are taken from the Dataset.
        Alternatively, a user can manually pass the desired mappables
        and masker (e.g., in cases where the ImageTable class is being used without a
        Dataset). Can optionally specify the radius of the sphere used for smoothing (default:
        6 mm), as well as whether or not to represent the data as a sparse array
        (generally this should be left to True, as these data are quite sparse and
        computation can often be speeded up by an order of magnitude.)
        """
        if dataset is not None:
            mappables, masker, r = dataset.mappables, dataset.masker, dataset.r
        for var in [mappables, masker, r]:
            assert var is not None
        self.ids = [m.id for m in mappables]
        self.masker = masker
        self.r = r

        data_shape = (self.masker.num_vox_in_mask, len(mappables))
        if use_sparse:
            # Fancy indexing assignment is not supported for sparse matrices, so
            # let's keep lists of values and their indices (rows, cols) to later
            # construct the csr_matrix.
            vals, rows, cols = [], [], []
        else:
            self.data = np.zeros(data_shape, dtype=int)

        logger.info("Creating matrix of %d mappables..." % (len(mappables),))
        for i, s in enumerate(mappables):
            logger.debug("%s/%s..." % (str(i + 1), str(len(mappables))))
            img = imageutils.map_peaks_to_image(
                s.peaks, r=r, header=self.masker.get_header())
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
          ids: A list or 1D numpy array of Mappable ids to return. If None, returns
            data for all Mappables.
          voxels: A list or 1D numpy array of voxel indices (i.e., rows) to return.
            If None, returns data for all voxels.
          dense: Optional boolean. When True (default), convert the result to a dense
            array before returning. When False, keep as sparse matrix.

        Returns:
          A 2D numpy array, with voxels in rows and mappables in columns.
        """
        result = self.data
        if ids is not None:
            idxs = np.where(np.in1d(np.array(self.ids), np.array(ids)))[0]
            result = result[:, idxs]
        if voxels is not None:
            result = result[voxels,:]
        return result.toarray() if dense else result

    def trim(self, ids):
        """ Trim ImageTable to keep only the passed Mappables. This is a convenience
        method, and should generally be avoided in favor of non-destructive alternatives
        that don't require slicing (e.g., matrix multiplication). """
        self.data = self.get_image_data(ids, dense=False)  # .tocoo()
        idxs = np.where(np.in1d(np.array(self.ids), np.array(ids)))[0]
        self.ids = [self.ids[i] for i in idxs]

    def save_images_to_file(self, ids, outroot='./'):
        """ Reconstructs vectorized images corresponding to the specified Mappable ids
        and saves them to file, prepending with the outroot (default: current directory). """
        pass

    def save(self, filename):
        import cPickle
        cPickle.dump(self, open(filename, 'wb'), -1)


class FeatureTable(object):

    """ A FeatureTable instance stores a matrix of mappables x features, along with
    associated manipulation methods. """

    def __init__(self, dataset, filename, description=None, validate=False):
        """ Initialize a new FeatureTable. Takes as input a parent DataSet instance and
        the name of a file containing feature data. Optionally, can provide a description
        of the feature set. """
        self.dataset = dataset
        self.load(filename)
        self.description = description

    def load(self, filename):
        """ Loads FeatureTable data from file. Input must be a dense matrix stored as 
        plaintext (see _parse_txt() for details).
        """
        try:
            self._features_from_txt(filename)
        except Exception as e:
            logger.error("%s cannot be parsed: %s" % (filename, e))


    def _features_from_txt(self, filename):
        """ Parses FeatureTable from a plaintext file that represents a dense matrix,
        with mappable objects in rows and features in columns. Values in cells reflect the
        weight of the intersecting feature for the intersecting study. Feature names and
        mappable IDs should be included as the first column and first row, respectively. """

        # Use pandas to read in data
        self.data = pd.read_csv(filename, delim_whitespace=True, index_col=0).to_sparse()

        # Remove mappables without any features
        # if validate:
        #     valid_ids = set(self.ids) & set(self.dataset.image_table.ids)
        #     if len(valid_ids) < len(self.dataset.image_table.ids):
        #         valid_id_inds = np.in1d(self.ids, np.array(valid_ids))
        #         self.data = self.data[valid_id_inds,:]
        #         self.ids = self.ids[valid_id_inds]
        # self.data = sparse.csr_matrix(self.data)

    @property
    def feature_names(self):
        return list(self.data.columns)
    
    def get_feature_data(self, ids=None, features=None, dense=True):
        """ Slices and returns a subset of feature data.

        Args:
            ids: A list or 1D numpy array of Mappable ids to return rows for. 
                If None, returns data for all Mappables (i.e., all rows in array). 
            features: A list or 1D numpy array of named features to return. 
                If None, returns data for all features (i.e., all columns in array).
            dense: Optional boolean. When True (default), convert the result to a dense
                array before returning. When False, keep as sparse matrix. Note that if 
                ids is not None, the returned array will always be dense.
        Returns:
          A 2D numpy array, with mappable IDs in rows and features in columns.
        """
        result = self.data

        if ids is not None:
            result = result.ix[ids]

        if features is not None:
            result = result.ix[:,features]

        return result#.to_dense() if dense else result

    def get_ordered_names(self, features):
        """ Given a list of features, returns features in order that they appear in database
        Args:
            features: A list or 1D numpy array of named features to return. 

        Returns:
            A list of features in order they appear in database
        """

        idxs = np.where(np.in1d(self.data.columns.values, np.array(features)))[0]
        return list(self.data.columns[idxs].values)

    def get_ids(self, features, threshold=None, func=np.sum, get_weights=False):
        """ Returns a list of all Mappables in the table that meet the desired feature-based
        criteria.

        Will most commonly be used to retrieve Mappables that use one or more
        features with some minimum frequency; e.g.,: get_ids(['fear', 'anxiety'], threshold=0.001)

        Args:
          features: a list of feature names to search on
          threshold: optional float indicating threshold features must pass to be included
          func: any numpy function to use for thresholding (default: sum). The function will be
            applied to the list of features and the result compared to the threshold. This can be
            used to change the meaning of the query in powerful ways. E.g,:
              max: any of the features have to pass threshold (i.e., max > thresh)
              min: all features must each individually pass threshold (i.e., min > thresh)
              sum: the summed weight of all features must pass threshold (i.e., sum > thresh)
          get_weights: boolean indicating whether or not to return weights.

        Returns:
          When get_weights is false (default), returns a list of Mappable names. When true,
          returns a dict, with mappable names as keys and feature weights as values.
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
        ''' Returns all features that match any of the elements in the input list. '''
        search = [s.replace('*', '.*') for s in search]
        cols = list(self.data.columns)
        results = []
        for s in search:
            results.extend([f for f in cols if re.match(s + '$', f)])
        return list(set(results))

    def get_ids_by_expression(self, expression, threshold=0.001, func=np.sum):
        """ Use a PEG to parse expression and return mappables. """
        from neurosynth.base import lexparser as lp
        lexer = lp.Lexer()
        lexer.build()
        parser = lp.Parser(
            lexer, self.dataset, threshold=threshold, func=func)
        parser.build()
        return parser.parse(expression).keys()

    def get_features_by_ids(self, ids=None, threshold=0.0001, func=np.mean, get_weights=False):
        ''' Returns features for which the mean loading across all specified studies (in ids)
        is >= threshold. '''
        weights = self.data.ix[ids].apply(func, 0)
        above_thresh = weights[weights >= threshold]
        return above_thresh if get_weights else list(above_thresh.index)

    def _sdf_to_csr(self):
        """ Convert FeatureTable to SciPy CSR matrix because pandas has a weird bug that 
        crashes de-serializing when data are in SparseDataFrame. (Bonus: takes up 
        less space.) Really need to fix this! """
        data = self.data.to_dense()
        self.data = {
            'columns': list(data.columns),
            'index': list(data.index),
            'values': sparse.csr_matrix(data.values)
        }

    def _csr_to_sdf(self):
        """ Inverse of _sdf_to_csr(). """
        self.data = pd.DataFrame(self.data['values'].todense(), index=self.data['index'], 
                            columns=self.data['columns']).to_sparse()


