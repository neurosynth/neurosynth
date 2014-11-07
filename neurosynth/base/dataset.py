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
                  target='MNI', **kwargs):
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
            kwargs: Additional optional arguments passed to add_features().


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

        # Initialize all mappables--for now, assume Articles are passed
        logger.info("Loading study data from database file...")
        return list(contents.groupby('id', as_index=False).apply(lambda x: 
                mappable.Article(x, self.transformer)))


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
        defined by the mask image. 
        Args:
            mask: the mask image (see Masker documentation for valid data types).
            threshold: an integer or float. If an integer, the absolute number of 
                voxels that must be active within the mask for a study to be retained.
                When a float, proportion of voxels that must be active.
            get_image_data: if True, returns the image data rather than the study IDs.
        """
        mask = self.masker.mask(mask).astype(bool)
        num_vox = np.sum(mask)
        prop_mask_active = self.image_table.data.T.dot(mask).astype(float)
        if isinstance(threshold, float):
            prop_mask_active /= num_vox
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

    def add_features(self, features, append=True, merge='outer', duplicates='ignore',
            min_studies=0.0, threshold=0.001):
        """ Construct a new FeatureTable from file.
        Args:
            features: Feature data to add. Can be:
                (a) A text file containing the feature data, where each row is a 
                study in the database, with features in columns. The first column 
                must contain the IDs of the studies to match up with the image data.
                (b) A pandas DataFrame, where studies are in rows, features are 
                in columns, and the index provides the study IDs.
            append: If True, adds new features to existing ones incrementally.
                If False, replaces old features.
            merge, duplicates, min_studies, threshold: Additional arguments passed to 
                FeatureTable.add_features().
         """
        if (not append) or not hasattr(self, 'feature_table'):
            self.feature_table = FeatureTable(self)

        self.feature_table.add_features(features, merge=merge, duplicates=duplicates,
            min_studies=min_studies, threshold=threshold)

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
        if hasattr(dataset, 'feature_table'):
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

        if hasattr(self, 'feature_table'):
            self.feature_table._sdf_to_csr()

        import cPickle
        cPickle.dump(self, open(filename, 'wb'), -1)

        if hasattr(self, 'feature_table'):
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

        data_shape = (self.masker.n_vox_in_vol, len(mappables))
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
        if dense and ids is None and voxels is None:
            logger.warning("Warning: get_image_data() is being called without specifying a " +
                "subset of studies or voxels to retrieve. This may result in a very large " +
                "amount of data (several GB) being read into memory. If you experience any " +
                "problems, consider returning a sparse matrix by passing dense=False, or " +
                "pass in a list of ids of voxels to retrieve only a portion of the data.")

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

    def __init__(self, dataset, **kwargs):
        """ Initialize a new FeatureTable. Takes as input a parent DataSet instance and
        feature data (if provided). """
        self.dataset = dataset
        self.data = pd.DataFrame()
        if kwargs:
            self.add_features(features, **kwargs)

    def add_features(self, features, merge='outer', duplicates='ignore', min_studies=0,
                    threshold=0.0001):
        """ Add new features to FeatureTable.
        Args:
            features: A filename to load data from, or a pandas DataFrame. In either case, 
                studies are in rows and features are in columns. Values in cells reflect the
                weight of the intersecting feature for the intersecting study. Feature names and
                mappable IDs should be included as the first column and first row, respectively.
            merge: The merge strategy to use when merging new features with old. This is passed 
                to pandas.merge, so can be 'left', 'right', 'outer', or 'inner'. Defaults to 
                outer (i.e., all data in both new and old will be kept, and missing values 
                will be assigned zeros.)
            duplicates: string indicating how to handle features whose name matches an existing
                feature. Valid options:
                'ignore' (default): ignores the new feature, keeps old data
                'replace': replace the old feature's data with the new data
                'merge': keeps both features, renaming them so they're different
            min_studies: minimum number of studies that pass threshold in order to add feature
            threshold: minimum threshold to use for applying min_studies
        """
        if isinstance(features, basestring):
            if not os.path.exists(features):
                raise ValueError("%s cannot be found." % features)
            try:
                features = pd.read_csv(features, sep='\t', index_col=0)
            except Exception as e:
                logger.error("%s cannot be parsed: %s" % (features, e))

        if min_studies:
            valid = np.where((features.values>=threshold).sum(0) >= min_studies)[0]
            features = features.iloc[:,valid]

        # Warn user if no/few IDs match between the FeatureTable and the Dataset.
        # This most commonly happens because older database.txt files used doi's as 
        # IDs whereas we now use PMIDs throughout.
        n_studies = len(features)
        n_common_ids = len(set(features.index) & set(self.dataset.image_table.ids))
        if float(n_common_ids)/n_studies < 0.01: # Minimum 1% overlap
            msg = "Only %d" % n_common_ids if n_common_ids else "None of the" 
            logger.warning(msg + " studies in the feature file matched studies currently in " + 
                "the Dataset. The most likely cause for this is that you're pairing a newer " +
                "feature set with an older, incompatible database file. You may want to try " +
                "regenerating the Dataset instance from a newer database file that uses PMIDs " +
                "rather than doi's as the study identifiers in the first column.")

        old_data = self.data.to_dense()
        # Handle features with duplicate names
        common_features = list(set(old_data.columns) & set(features.columns))
        if duplicates == 'ignore':
            features = features.drop(common_features, axis=1)
        elif duplicates == 'replace':
            old_data = old_data.drop(common_features, axis=1)

        data = old_data.merge(features, how=merge, left_index=True, right_index=True)
        self.data = data.fillna(0.0).to_sparse()

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
          A pandas DataFrame, with mappable IDs in rows and features in columns.
        """
        result = self.data

        if ids is not None:
            result = result.ix[ids]

        if features is not None:
            result = result.ix[:,features]

        return result.to_dense() if dense else result

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


