
from grdicmaker import DataTransformer
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np



transfomer = DataTransformer(
        "training_sets/kz/training_data_full.json",
        "training_sets/kz/paradigm_lengths.json",
        "category_description/kazakh.json"
    )
headlines, matrix_kz_full, targets_kz_full = transfomer.get_training_data_matrix(normalize=True)

transfomer = DataTransformer(
        "training_sets/kz/training_data_thresholded.json",
        "training_sets/kz/paradigm_lengths.json",
        "category_description/kazakh.json"
    )
headlines_thr, matrix_kz_thr, targets_kz_thr = transfomer.get_training_data_matrix(normalize=True)



forest = ExtraTreesClassifier(n_estimators=10)

matrix_kz_full = matrix_kz_full.toarray()

forest.fit(matrix_kz_full, targets_kz_full)
importances_full = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices_full = np.argsort(importances_full)[::-1]


forest = ExtraTreesClassifier(n_estimators=10)

matrix_kz_thr = matrix_kz_thr.toarray()

forest.fit(matrix_kz_thr, targets_kz_thr)
importances_thr = forest.feature_importances_
std_thr = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices_thr = np.argsort(importances_thr)[::-1]

dict_full = {}
dict_thresholded = {}

# # Print the feature ranking
# print("Feature ranking (full):")

for f in range(matrix_kz_full.shape[1]):
    # print("%d. feature '%s' (%d) (%f)" % (f + 1, headlines[indices_full[f]], indices_full[f], importances_full[indices_full[f]]))
    dict_full[headlines[indices_full[f]]] = importances_full[indices_full[f]]

# print("Feature ranking (thresholded):")

for f in range(matrix_kz_thr.shape[1]):
    # print("%d. feature '%s' (%d) (%f)" % (f + 1, headlines[indices_thr[f]], indices_thr[f], importances_thr[indices_thr[f]]))
    dict_thresholded[headlines_thr[indices_thr[f]]] = importances_thr[indices_thr[f]]

# Plot the feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(
#     range(matrix_kz_full.shape[1]),
#     importances_full[indices_full],
#        color="#aed6db",
#     align="center"
# )
# plt.xticks(range(matrix_kz_full.shape[1]), indices_full)
# plt.xlim([-1, matrix_kz_full.shape[1]])
# plt.show()

print 'Feature\tin full\tin thresholded'
for feature_name in dict_full:
    print '%s\t%f\t%f' % (feature_name, dict_full[feature_name], dict_thresholded[feature_name])