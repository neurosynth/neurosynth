from neurosynth.base.dataset import Dataset
from neurosynth.analysis import meta

""" Create a new Dataset instance from a database file and load features. 
This is basically the example from the quickstart in the README. 
Assumes you have database.txt and features.txt files in the current dir.
"""

# Create Dataset instance from a database file.
dataset = Dataset('database.txt')

# Load features from file
dataset.add_features('features.txt')

# Pickle the Dataset to file so we can use Dataset.load() next time 
# instead of having to sit through the generation process again.
dataset.save('dataset.pkl')

# Get Mappable IDs for all features that start with 'emo'
ids = dataset.get_ids_by_features('emo*', threshold=0.001)

# Run a meta-analysis and save results
ma = meta.MetaAnalysis(dataset, ids)
ma.save_results('emotion')