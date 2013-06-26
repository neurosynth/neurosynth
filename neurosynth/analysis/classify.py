#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Classification and decoding related tools """

import numpy as np
from functools import reduce


def classify_by_features(
    dataset,
    features,
    studies=None,
    method='SVM',
    scikit_classifier=None,
    ):
    pass


def classify_regions(
    dataset,
    masks,
    method='ERF',
    threshold=0.08,
    remove_overlap=True,
    regularization='scale',
    output='summary',
    studies=None,
    features=None,
    class_weight='auto',
    classifier=None,
    cross_val='4-Fold',
    param_grid=None,
    ):
    '''
        Args:
            ...
            features: An optional list of feature names used to constrain the set used in
                classification. If None, will use all features in the dataset.
    '''

    import nibabel as nib
    import os

    # Load masks using NiBabel

    try:
        loaded_masks = [nib.load(os.path.relpath(m)) for m in masks]
    except OSError:
        print 'Error loading masks. Check the path'

    # Get a list of studies that activate for each mask file--i.e.,  a list of
    # lists

    grouped_ids = [dataset.get_ids_by_mask(m, threshold=threshold)
                   for m in loaded_masks]

    # Flattened ids

    flat_ids = reduce(lambda a, b: a + b, grouped_ids)

    # Remove duplicates

    if remove_overlap:
        import collections
        flat_ids = [id for (id, count) in
                    collections.Counter(flat_ids).items() if count == 1]
        grouped_ids = [[x for x in m if x in flat_ids] for m in
                       grouped_ids]  # Remove

    # Create class label(y)

    y = [[idx] * len(ids) for (idx, ids) in enumerate(grouped_ids)]
    y = reduce(lambda a, b: a + b, y)  # Flatten
    y = np.array(y)

    # Extract feature set for only relevant ids

    X = dataset.get_feature_data(ids=flat_ids, features=features)

    return classify(
        X,
        y,
        method,
        classifier,
        output,
        cross_val,
        class_weight,
        regularization=regularization,
        param_grid=param_grid,
        )


def classify(
    X,
    y,
    method='SVM',
    classifier=None,
    output='summary',
    cross_val=None,
    class_weight=None,
    regularization=None,
    param_grid=None,
    ):

    # Build classifier

    clf = Classifier(method, classifier, class_weight, param_grid)

    # Regularize

    if regularization:
        X = clf.regularize(X, method=regularization)

    # Fit & test model with or without cross-validation

    if cross_val:
        score = clf.cross_val_fit(X, y, cross_val)
    else:
        score = clf.fit(X, y).score(X, y)

    # Return some stuff...

    if output == 'summary':
        from collections import Counter
        return {'score': score, 'n': dict(Counter(y))}
    elif output == 'summary_clf':
        from collections import Counter
        return {'score': score, 'n': dict(Counter(y)), 'clf': clf}
    elif output == 'clf':
        return clf
    else:
        pass


class Classifier:

    def __init__(
        self,
        clf_method='ERF',
        classifier=None,
        class_weight=None,
        param_grid=None,
        ):
        """ Initialize a new classifier instance """

        # Set classifier

        if classifier:
            self.clf = classifier
        else:
            if clf_method == 'SVM':
                from sklearn import svm
                self.clf = svm.SVC(class_weight=class_weight)
            elif clf_method == 'ERF':
                from sklearn.ensemble import ExtraTreesClassifier
                self.clf = ExtraTreesClassifier(
                    n_estimators=10,
                    max_depth=None,
                    min_samples_split=1,
                    random_state=0,
                    n_jobs=-1,
                    compute_importances=True,
                    )
            elif clf_method == 'Dummy':
                from sklearn.dummy import DummyClassifier
                self.clf = DummyClassifier(strategy='stratified')
            else:
                raise Exception('Unrecognized classification method')

        if isinstance(param_grid, dict):
            from sklearn.grid_search import GridSearchCV
            self.clf = GridSearchCV(estimator=self.clf,
                                    param_grid=param_grid, n_jobs=-1)

    def fit(self, X, y):
        """ Fits X to outcomes y, using clf """

        # Incorporate error checking such as :
        # if isinstance(self.classifier, ScikitClassifier):
        #     do one thingNone
        # otherwiseNone.

        self.X = X
        self.y = y
        self.clf = self.clf.fit(X, y)

        return self.clf

    def cross_val_fit(
        self,
        X,
        y,
        cross_val='4-Fold',
        ):
        """ Fits X to outcomes y, using clf and cv_method """

        from sklearn import cross_validation

        self.X = X
        self.y = y

        # Set cross validator

        if isinstance(cross_val, basestring):
            if cross_val == '4-Fold':
                self.cver = cross_validation.KFold(len(self.y), 4,
                        indices=False, shuffle=True)
            else:
                raise Exception('Unrecognized cross validation method')
        else:
            self.cver = cross_val

        from sklearn.grid_search import GridSearchCV
        if isinstance(self.clf, GridSearchCV):
            self.clf.set_params(cv=self.cver)

            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=UserWarning)
                self.clf = self.clf.fit(X, y)

            self.cvs = self.clf.best_score_
        else:
            self.cvs = cross_validation.cross_val_score(self.clf,
                    self.X, self.y, cv=self.cver, n_jobs=-1)

        return self.cvs.mean()

    def fit_dataset(
        self,
        dataset,
        y,
        features=None,
        feature_type='features',
        ):
        """ Given a dataset, fits either features or voxels to y """

        # Get data from dataset

        if feature_type == 'features':
            X = np.rot90(dataset.feature_table.data.toarray())
        elif feature_type == 'voxels':
            X = np.rot90(dataset.image_table.data.toarray())

        self.sk_classifier.fit(X, y)

    def regularize(self, X, method='scale'):

        if method == 'scale':
            from sklearn import preprocessing
            return preprocessing.scale(X, with_mean=False)
        else:
            raise Exception('Unrecognized regularization method')
