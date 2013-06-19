from neurosynth.base.dataset import Dataset
from neurosynth.analysis import decode

# Load a saved Dataset file. This example will work with the 
# file saved in the create_a_new_dataset_and_load_features example.
dataset = Dataset.load('dataset.pkl')

# Initialize a new Decoder instance with a few features. Note that 
# if you don't specify a subset of features, ALL features in the 
# Dataset will be loaded, which will take a long time because 
# meta-analysis images for each feature need to be generated.
decoder = decode.Decoder(dataset, features=['emotion', 'pain', 'somatosensory', 'wm', 'inhibition'])

# Decode three images. The sample images here are coactivation 
# maps for ventral, dorsal, and posterior insula clusters, 
# respectively. Maps are drawn from data reported in 
# Chang, Yarkoni, Khaw, & Sanfey (2012); see paper for details.
# We save the output--an image x features matrix--to a file.
# By default, the decoder will use Pearson correlation, i.e., 
# each value in our results table indicates the correlation 
# between the input image and each feature's meta-analysis image.
result = decoder.decode(['vIns.nii.gz', 'dIns.nii.gz', 'pIns.nii.gz'], save='decoding_results.txt')