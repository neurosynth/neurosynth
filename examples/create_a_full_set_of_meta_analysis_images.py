from neurosynth.base.dataset import Dataset
from neurosynth.analysis import meta

""" Load a Dataset and generate a full set of meta-analysis
images--i.e., run a meta-analysis on every single feature.
"""

# Load pickled Dataset--assumes you've previously saved it. If not,
# follow the create_a_new_dataset_and_load_features example.
dataset = Dataset.load('dataset.pkl')

# Get the full list of feature names
feature_list = dataset.get_feature_names()

# Run a meta-analysis on each feature, and save all the results to 
# a directory called results. Note that the directory will not be 
# created for you, so make sure it exists.
# Here we use the default frequency threshold of 0.001 (i.e., a 
# study is said to have a feature if more than 1 in every 1,000
# words is the target word), and an FDR correction level of 0.05.
meta.analyze_features(dataset, feature_list, threshold=0.001, q=0.05, save='results/')