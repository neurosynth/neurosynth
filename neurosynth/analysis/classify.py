""" Classification and decoding related tools """

import numpy as np
from functools import reduce


def classify_by_features(dataset, features, studies=None, method='SVM', scikit_classifier=None):
    pass


def classify_regions(dataset, masks, remove_overlap=True, features=None, threshold=0.001,
                     output='summary', studies=None, method='SVM', classifier=None,
                     regularization="Alejandro's expert judgment", class_weight=None,
                     cross_val=None):
    '''
        Args:
            ...
            features: An optional list of feature names used to constrain the set used in
                classification. If None, will use all features in the dataset.
    '''

    import nibabel as nib
    import os

    # Get base file names for masks
    mask_names = [os.path.basename(file).split('.')[0] for file in masks]

    # Load masks using NiBabel
    try:
        loaded_masks = [nib.load(os.path.relpath(m)) for m in masks]
    except OSError:
        print "Error loading masks. Check the path"

    # Get a list of studies that activate for each mask file--i.e.,  a list of
    # lists
    all_ids = [dataset.get_ids_by_mask(m, threshold=threshold) for m in loaded_masks]

    # Flattened ids
    flat_ids = reduce(lambda a, b: a + b, all_ids)

    # Remove duplicates
    if remove_overlap:
        import collections
        unique_ids = [id for (id, count) in collections.Counter(flat_ids).items() if count == 1]
        all_ids = [[x for x in m if x in unique_ids] for m in all_ids]  # Remove

    # Loop over list of masksids and get features and create masklabel vector
    y = [[mask_names[idx]] * len(ids) for idx, ids in enumerate(all_ids)]
    y = reduce(lambda a, b: a + b, y)  # Flatten
    y = np.array(y)

    # Extract feature set for only relevant ids
    X = dataset.get_feature_data(ids=unique_ids, features=features)

    return classify(X, y, method, classifier, output, cross_val, class_weight,
                    regularization=regularization)


def classify(X, y, method='SVM', classifier=None, output='summary', cross_val=None,
             class_weight=None, regularization="Alejandro's expert judgment"):

    # Build classifier
    clf = Classifier(method, classifier, class_weight)

    # Regularize
    X = clf.regularize(X)

    # Fit model with or without cross-validation
    if cross_val is not None:
        clf.cross_val_fit(X, y)
    else:
        clf.fit(X, y)

    # Return some stuff...
    if output == 'summary':
        pass
    elif output == 'scikit':
        pass
    else:
        pass

    return


class Classifier:

    def __init__(self, method='SVM', classifier=None, class_weight=None):
        """ Initialize a new classifier instance """
        if classifier:
            self.sk_classifier = classifier
        else:
            if method == 'SVM':
                from sklearn import svm
                self.sk_classifier = svm.SVC(class_weight=class_weight)
            else:
                # Error handling?
                self.sk_classifier = None

    def fit(self, X, y):
        """ Fits X to outcomes y, using sk_classifier """
        # Incorporate error checking such as :
        # if isinstance(self.classifier, ScikitClassifier):
        #     do one thingNone
        # otherwiseNone.

        self.X = X
        self.y = y
        self.sk_classifier.fit(X, y)

    def cross_val_fit(self, X, y):
        pass

    def fit_dataset(self, dataset, y, features=None, feature_type='features'):
        """ Given a dataset, fits either features or voxels to y """

        # Get data from dataset
        if feature_type == 'features':
            X = np.rot90(dataset.feature_table.data.toarray())
        elif feature_type == 'voxels':
            X = np.rot90(dataset.image_table.data.toarray())

        self.sk_classifier.fit(X, y)

    def regularize(self, X, method='None'):
        # Nonewhat to give scikitNone or do it yourselfNone
        return X

