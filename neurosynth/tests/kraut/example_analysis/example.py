from neurosynth.base.dataset import Dataset
from neurosynth.analysis import meta
import os
dataset = Dataset('database.txt')
dataset.add_features('features.txt')
print dataset.get_feature_names()
ids = dataset.get_ids_by_features('emo*', threshold=0.001)
print len(ids)
ma = meta.MetaAnalysis(dataset, ids)
ma.save_results('emotion')
