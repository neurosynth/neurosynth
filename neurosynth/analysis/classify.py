""" Classification and decoding related tools """

import numpy as np
from functools import reduce
from sklearn.feature_selection.univariate_selection import SelectKBest
import re
from six import string_types


def feature_selection(feat_select, X, y):
    """" Implements various kinds of feature selection """
    # K-best
    if re.match('.*-best', feat_select) is not None:
        n = int(feat_select.split('-')[0])

        selector = SelectKBest(k=n)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            features_selected = np.where(
                selector.fit(X, y).get_support() is True)[0]

    elif re.match('.*-randombest', feat_select) is not None:
        n = int(feat_select.split('-')[0])

        from random import shuffle
        features = range(0, X.shape[1])
        shuffle(features)

        features_selected = features[:n]

    return features_selected


def get_score(X, y, clf, scoring='accuracy'):
    prediction = clf.predict(X)

    if scoring == 'accuracy':
        from sklearn.metrics import accuracy_score
        score = accuracy_score(y, prediction)
    elif scoring == 'f1':
        from sklearn.metrics import f1_score
        score = f1_score(y, prediction)
    else:
        score = scoring(y, prediction.squeeze())

    return prediction, score


def classify_by_features(dataset, features, studies=None, method='SVM',
                         scikit_classifier=None):
    pass


def regularize(X, method='scale'):
    if method == 'scale':
        from sklearn import preprocessing
        return preprocessing.scale(X, with_mean=False)
    else:
        raise Exception('Unrecognized regularization method')


def get_studies_by_regions(dataset, masks, threshold=0.08, remove_overlap=True,
                           studies=None, features=None,
                           regularization="scale"):
    """ Set up data for a classification task given a set of masks

        Given a set of masks, this function retrieves studies associated with
        each mask at the specified threshold, optionally removes overlap and
        filters by studies and features, and returns studies by feature matrix
        (X) and class labels (y)

        Args:
            dataset: a Neurosynth dataset
            maks: a list of paths to Nifti masks
            threshold: percentage of voxels active within the mask for study
                to be included
            remove_overlap: A boolean indicating if studies studies that
                appear in more than one mask should be excluded
            studies: An optional list of study names used to constrain the set
                used in classification. If None, will use all features in the
                dataset.
            features: An optional list of feature names used to constrain the
                set used in classification. If None, will use all features in
                the dataset.
            regularize: Optional boolean indicating if X should be regularized

        Returns:
            A tuple (X, y) of np arrays.
            X is a feature by studies matrix and y is a vector of class labels
    """

    import nibabel as nib
    import os

    # Load masks using NiBabel

    try:
        loaded_masks = [nib.load(os.path.relpath(m)) for m in masks]
    except OSError:
        print('Error loading masks. Check the path')

    # Get a list of studies that activate for each mask file--i.e.,  a list of
    # lists

    grouped_ids = [dataset.get_studies(mask=m, activation_threshold=threshold)
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

    # Extract feature set for each class separately
    X = [dataset.get_feature_data(ids=group_ids, features=features)
         for group_ids in grouped_ids]

    X = np.vstack(tuple(X))

    if regularization:
        X = regularize(X, method=regularization)

    return (X, y)


def get_feature_order(dataset, features):
    """ Returns a list with the order that features requested appear in
    dataset """
    all_features = dataset.get_feature_names()

    i = [all_features.index(f) for f in features]

    return i


def classify_regions(dataset, masks, method='ERF', threshold=0.08,
                     remove_overlap=True, regularization='scale',
                     output='summary', studies=None, features=None,
                     class_weight='auto', classifier=None,
                     cross_val='4-Fold', param_grid=None, scoring='accuracy'):
    """ Perform classification on specified regions

        Given a set of masks, this function retrieves studies associated with
        each mask at the specified threshold, optionally removes overlap and
        filters by studies and features. Then it trains an algorithm to
        classify studies based on features and tests performance.

        Args:
            dataset: a Neurosynth dataset
            maks: a list of paths to Nifti masks
            method: a string indicating which method to used.
                'SVM': Support Vector Classifier with rbf kernel
                'ERF': Extremely Randomized Forest classifier
                'Dummy': A dummy classifier using stratified classes as
                    predictor
            threshold: percentage of voxels active within the mask for study
                to be included
            remove_overlap: A boolean indicating if studies studies that
                appear in more than one mask should be excluded
            regularization: A string indicating type of regularization to use.
                If None, performs no regularization.
                'scale': Unit scale without demeaning
            output: A string indicating output type
                'summary': Dictionary with summary statistics including score
                    and n
                'summary_clf': Same as above but also includes classifier
                'clf': Only returns classifier
                Warning: using cv without grid will return an untrained
                classifier
            studies: An optional list of study names used to constrain the set
                used in classification. If None, will use all features in the
                dataset.
            features: An optional list of feature names used to constrain the
                set used in classification. If None, will use all features in
                the dataset.
            class_weight: Parameter to pass to classifier determining how to
                weight classes
            classifier: An optional sci-kit learn classifier to use instead of
                pre-set up classifiers set up using 'method'
            cross_val: A string indicating type of cross validation to use.
                Can also pass a scikit_classifier
            param_grid: A dictionary indicating which parameters to optimize
                using GridSearchCV. If None, no GridSearch will be used

        Returns:
            A tuple (X, y) of np arrays.
            X is a feature by studies matrix and y is a vector of class labels
    """

    (X, y) = get_studies_by_regions(dataset, masks, threshold, remove_overlap,
                                    studies, features,
                                    regularization=regularization)

    return classify(X, y, method, classifier, output, cross_val,
                    class_weight, scoring=scoring, param_grid=param_grid)


def classify(X, y, clf_method='ERF', classifier=None, output='summary_clf',
             cross_val=None, class_weight=None, regularization=None,
             param_grid=None, scoring='accuracy', refit_all=True,
             feat_select=None):
    """ Wrapper for scikit-learn classification functions
    Imlements various types of classification and cross validation """

    # Build classifier
    clf = Classifier(clf_method, classifier, param_grid)

    # Fit & test model with or without cross-validation
    if cross_val is not None:
        score = clf.cross_val_fit(X, y, cross_val, scoring=scoring,
                                  feat_select=feat_select,
                                  class_weight=class_weight)
    else:
        # Does not support scoring function
        score = clf.fit(X, y, class_weight=class_weight).score(X, y)

    # Return some stuff...
    from collections import Counter

    if output == 'clf':
        return clf
    else:
        if output == 'summary':
            output = {'score': score, 'n': dict(Counter(y))}
        elif output == 'summary_clf':
            output = {
                'score': score,
                'n': dict(Counter(y)),
                'clf': clf,
                'features_selected': clf.features_selected,
                'predictions': clf.predictions
            }

        return output


class Classifier:

    def __init__(self, clf_method='ERF', classifier=None, param_grid=None):
        """ Initialize a new classifier instance """

        # Set classifier
        self.features_selected = None
        self.predictions = None

        if classifier is not None:
            self.clf = classifier

            from sklearn.svm import LinearSVC
            import random
            if isinstance(self.clf, LinearSVC):
                self.clf.set_params().random_state = random.randint(0, 200)
        else:
            if clf_method == 'SVM':
                from sklearn import svm
                self.clf = svm.SVC()
            elif clf_method == 'ERF':
                from sklearn.ensemble import ExtraTreesClassifier
                self.clf = ExtraTreesClassifier(
                    n_estimators=100, max_depth=None, min_samples_split=1,
                    random_state=0)
            elif clf_method == 'GBC':
                from sklearn.ensemble import GradientBoostingClassifier
                self.clf = GradientBoostingClassifier(n_estimators=100,
                                                      max_depth=1)
            elif clf_method == 'Dummy':
                from sklearn.dummy import DummyClassifier
                self.clf = DummyClassifier(strategy='stratified')
            else:
                raise Exception('Unrecognized classification method')

        if isinstance(param_grid, dict):
            from sklearn.grid_search import GridSearchCV
            self.clf = GridSearchCV(estimator=self.clf,
                                    param_grid=param_grid)

    def fit(self, X, y, cv=None, class_weight='auto'):
        """ Fits X to outcomes y, using clf """

        # Incorporate error checking such as :
        # if isinstance(self.classifier, ScikitClassifier):
        #     do one thingNone
        # otherwiseNone.

        self.X = X
        self.y = y

        self.set_class_weight(class_weight=class_weight, y=y)

        self.clf = self.clf.fit(X, y)

        return self.clf

    def set_class_weight(self, class_weight='auto', y=None):
        """ Sets the class_weight of the classifier to match y """

        if class_weight is None:
            cw = None

            try:
                self.clf.set_params(class_weight=cw)
            except ValueError:
                pass

        elif class_weight == 'auto':
            c = np.bincount(y)
            ii = np.nonzero(c)[0]
            c = c / float(c.sum())
            cw = dict(zip(ii[::-1], c[ii]))

            try:
                self.clf.set_params(class_weight=cw)
            except ValueError:
                import warnings
                warnings.warn(
                    "Tried to set class_weight, but failed. The classifier "
                    "probably doesn't support it")

    def cross_val_fit(self, X, y, cross_val='4-Fold', scoring='accuracy',
                      feat_select=None, class_weight='auto'):
        """ Fits X to outcomes y, using clf and cv_method """

        from sklearn import cross_validation

        self.X = X
        self.y = y

        self.set_class_weight(class_weight=class_weight, y=y)

        # Set cross validator
        if isinstance(cross_val, string_types):
            if re.match('.*-Fold', cross_val) is not None:
                n = int(cross_val.split('-')[0])
                self.cver = cross_validation.StratifiedKFold(self.y, n)
            else:
                raise Exception('Unrecognized cross validation method')
        else:
            self.cver = cross_val

        if feat_select is not None:
            self.features_selected = []

        # Perform cross-validated classification
        from sklearn.grid_search import GridSearchCV
        if isinstance(self.clf, GridSearchCV):
            import warnings

            if feat_select is not None:
                warnings.warn(
                    "Cross-validated feature selection not supported with "
                    "GridSearchCV")
            self.clf.set_params(cv=self.cver, scoring=scoring)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=UserWarning)
                self.clf = self.clf.fit(X, y)

            self.cvs = self.clf.best_score_
        else:
            self.cvs = self.feat_select_cvs(
                feat_select=feat_select, scoring=scoring)

        if feat_select is not None:
            fs = feature_selection(
                feat_select, X, y)
            self.features_selected.append(fs)

            X = X[:, fs]

        self.clf.fit(X, y)

        return self.cvs.mean()

    def feat_select_cvs(self, scoring=None, feat_select=None):
        """ Returns cross validated scores (just like cross_val_score),
        but includes feature selection as part of the cross validation loop """

        scores = []
        self.predictions = []

        for train, test in self.cver:
            X_train, X_test, y_train, y_test = self.X[
                train], self.X[test], self.y[train], self.y[test]

            if feat_select is not None:
                # Get which features are kept
                fs = feature_selection(
                    feat_select, X_train, y_train)

                self.features_selected.append(fs)

                # Filter X to only keep selected features
                X_train, X_test = X_train[
                    :, fs], X_test[:, fs]

            # Set scoring (not implement as accuracy is default)

            # Train classifier
            self.clf.fit(X_train, y_train)

            # Test classifier
            predicition, s = get_score(
                X_test, y_test, self.clf, scoring=scoring)

            scores.append(s)
            self.predictions.append((y_test, predicition))

        return np.array(scores)

    def fit_dataset(self, dataset, y, features=None,
                    feature_type='features'):
        """ Given a dataset, fits either features or voxels to y """

        # Get data from dataset

        if feature_type == 'features':
            X = np.rot90(dataset.feature_table.data.toarray())
        elif feature_type == 'voxels':
            X = np.rot90(dataset.image_table.data.toarray())

        self.sk_classifier.fit(X, y)
