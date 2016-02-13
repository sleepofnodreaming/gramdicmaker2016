from grdicmaker import DataTransformer
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np


def get_feature_percentage(training_set, paradigm_lengths, category_description):
    transfomer = DataTransformer(training_set, paradigm_lengths, category_description)
    headlines, matrix, targets = transfomer.get_training_data_matrix(normalize=True)
    matrix = matrix.toarray()
    forest = ExtraTreesClassifier(n_estimators=10)
    forest.fit(matrix, targets)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    dict = {}

    for f in range(matrix.shape[1]):
        dict[headlines[indices[f]]] = importances[indices[f]]
    return dict


def estimate_random_forest():
    """
    Print a table with weights of features for random forest trained on Kazakh data.
    :return: None.
    """
    dict_full = get_feature_percentage(
        "training_sets/kz/training_data_full.json",
        "training_sets/kz/paradigm_lengths.json",
        "category_description/kazakh.json"
    )
    dict_thr = get_feature_percentage(
        "training_sets/kz/training_data_thresholded.json",
        "training_sets/kz/paradigm_lengths.json",
        "category_description/kazakh.json"
    )
    print 'Feature\tin full\tin thresholded'
    for feature_name in dict_full:
        print '%s\t%f\t%f' % (feature_name, dict_full[feature_name], dict_thr[feature_name])
