# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Overview
# 
# In this lab, we'll walk through some of the features of the Neurosynth core tools (http://github.com/neurosynth/neurosynth). By the end, you'll know how to:
# 
# * Import the modules you'll need to perform basic analyses
# * Create a new Neurosynth dataset object from provided text files
# * Run a simple term-based meta-analysis
# * Run a slightly more complicated term-based meta-analysis
# * Perform meta-analytic contrasts
# * Generate seed-based coactivation maps
# * "Decode" your own images
# 
# ## Installation
# 
# We're not going to cover installation here--sorry! See the quickstart guide in the [github repository](http://github.com/neurosynth/neurosynth) for that.
# 
# ## Running this code
# 
# To run the examples below, you have several options:
# 
# * *Run from within IPython Notebook*. This is the preferred approach if you have IPython Notebook installed. Put this file (neurosynth_demo.ipynb) in its own directory. Now launch the IPython dashboard from a command line prompt by typing "ipython notebook" (without quotes), then open the Neurosynth demo. You should now be able to run any cell in the notebook.
# 
# * *Run in IPython*. From a terminal prompt, launch IPython (just type ipython). Now paste the code blocks below.
# 
# * *Write a standalone script*. Save the code blocks below to a separate file and run it as a Python script. This is not recommended as it's non-interactive.
# 
# In all cases, you'll need to make sure that the data files called below (database.txt and features.txt) are accessible at the locations indicated (a subfolder called data/).
# 
# ## Importing modules
# 
# Let's start with the basics. In Python, most modules (i.e., organized chunks of code) are inaccessible by default. Other than a few very basic built-in functions, you'll need to explicitly include every piece of code you want to work with. This may seem cumbersome, but it has the nice effect of (a) making sure you always know exactly what dependencies your code has, and (b) minimizing the memory footprint of your app by only including functionality you know you'll need.
# 
# Like most Python packages, Neurosynth consists of several modules arranged into a semi-sensible tree structure. For this lab, we'll need functionality available in several modules, which we can include like so:

# <codecell>

# Core functionality for managing and accessing data
from neurosynth.base.dataset import Dataset
# Analysis tools for meta-analysis, image decoding, and coactivation analysis
from neurosynth.analysis import meta, decode, network
# The root-level module, included here just so we can set the logging level.
import neurosynth

# We set the logger to display everything at level INFO or above.
neurosynth.set_logging_level('info')

# <markdowncell>

# ## Creating a new dataset
# 
# Next, we create a Dataset, which is the core object most Neurosynth tools operate on. We initialize a Dataset by passing in a database file, which is essentially just a giant list of activation coordinates and associated study IDs. This file can be downloaded from the Neurosynth website or installed from the data submodule (see the Readme for instructions).
# 
# Creating the object will take a few minutes on most machines, as we need to process about 200,000 activations drawn from nearly 6,000 studies. Once that's done, we also need to add some features to the Dataset. Features are just variables associated with the studies in our dataset; literally any dimension a study could be coded on can constitute a feature that Neurosynth can use. In practice, the default set of features included in the data download includes 500 psychological terms (e.g., 'language', 'emotion', 'memory', etc.) that occur with some frequency in the dataset. So when we're talking about the "emotion" feature, we're really talking about how frequently each study in the Dataset uses the word 'emotion' in the full-text of the corresponding article.
# 
# Let's go ahead and create a dataset and add some features:

# <codecell>

# Create a new Dataset instance
dataset = Dataset('data/database.txt')

# Add some features
dataset.add_features('data/features.txt')

# <markdowncell>

# Because this takes a while, we'll save our Dataset object to disk. That way, the next time we want to use it, we won't have to sit through the whole creation operation again:

# <codecell>

dataset.save('dataset.pkl')

# <markdowncell>

# Now in future, instead of waiting, we could just load the dataset from file:

# <codecell>

dataset = Dataset.load('dataset.pkl')   # Note the capital D in the second Dataset--load() is a class method

# <markdowncell>

# ## Doing stuff with Neurosynth
# Now that our Dataset has both activation data and some features, we're ready to start doing some analyses! By design, Neurosynth focuses on facilitating simple, fast, and modestly useful analyses. This means you probably won't break any new ground using Neurosynth, but you should be able to supplement results you've generated using other approaches with a bunch of nifty analyses that take just 2 - 3 lines of code.
# 
# ### Simple feature-based meta-analyses
# The most straightforward thing you can do with Neurosynth is use the features we just loaded above to perform automated large-scale meta-analyses of the literature. Let's see what features we have:

# <codecell>

dataset.get_feature_names()

# <markdowncell>

# If the loading process went smoothly, this should return a list of about 500 terms. We can use these terms--either in isolation or in combination--to select articles for inclusion in a meta-analysis. For example, suppose we want to run a meta-analysis of emotion studies. We could operationally define a study of emotion as one in which the authors used words starting with 'emo' with high frequency:

# <codecell>

ids = dataset.get_ids_by_features('emo*', threshold=0.001)

# <markdowncell>

# Here we're asking for a list of IDs of all studies that use words starting with 'emo' (e.g.,'emotion', 'emotional', 'emotionally', etc.) at a frequency of 1 in 1,000 words or greater (in other words, if an article has 5,000 words of text, it will only be included in our set if it uses words starting with 'emo' at least 5 times). Let's find out how many studies are in our list:

# <codecell>

len(ids)

# <markdowncell>

# The resulting set includes 639 studies.
# 
# Once we've got a set of studies we're happy with, we can run a simple meta-analysis, prefixing all output files with the string 'emotion' to distinguish them from other analyses we might run:

# <codecell>

# Run a meta-analysis on emotion
ids = dataset.get_ids_by_features('emo*', threshold=0.001)
ma = meta.MetaAnalysis(dataset, ids)
ma.save_results('emotion')

# <markdowncell>

# You should now have a set of Nifti-format brain images on your drive that display various meta-analytic results. The image names are somewhat cryptic; see documentation elsewhere for details. It's important to note that the meta-analysis routines currently implemented in Neurosynth aren't very sophisticated; they're designed primarily for efficiency (most analyses should take just a few seconds), and take multiple shortcuts as compared to other packages like ALE or MKDA. But with that caveat in mind (and one that will hopefully be remedied in the near future), Neurosynth gives you a streamlined and quick way of running large-scale meta-analyses of fMRI data. Of course, all of the images you could generate using individual features are already available on the Neurosynth website, so there's probably not much point in doing this kind of thing yourself unless you've defined entirely new features.
# 
# ### More complex feature-based meta-analyses
# 
# Fortunately, we're not constrained to using single features in our meta-analyses. Neurosynth implements a parsing expression grammar, which is a fancy way of saying you can combine terms according to syntactic rules--in this case, basic logical operations.
# 
# For example, suppose we want to restrict our analysis to studies of emotion that do NOT use the terms 'reward' or 'pain', which we might construe as somewhat non-prototypical affective states. Then we could do the following:

# <codecell>

ids = dataset.get_ids_by_expression('emo* &~ (reward* | pain*)', threshold=0.001)
ma = meta.MetaAnalysis(dataset, ids)
ma.save_results('emotion_without_reward_or_pain')
print "Found %d studies." % len(ids)

# <markdowncell>

# This meta-analysis is somewhat more restrictive than the previous one (555 studies instead of 639), and the result should theoretically be at least somewhat more spatially specific.
# 
# There's no inherent restriction on how many terms you combine or how deeply you nest logical expressions within parentheses, but the cardinal of GIGO (garbage in, garbage out) always applies, so if your expression is very specific and the number of studies drops too far (in practice, sensible results are unlikely with fewer than 50 studies), don't expect to see much.
# 
# ### Meta-analytic contrasts
# 
# In addition to various logical operations, one handy thing you can do with Neurosynth is perform meta-analytic contrasts. Meaning, you can identify voxels in which the average likelihood of activation being reported differ for two different sets of studies. For example, let's say you want to meta-analytically contrast studies that use the term 'recollection' with studies that use the term 'recognition'. You can do this by defining both sets of studies separately, and then passing them both to the meta-analysis object:

# <codecell>

# Get the recognition studies and print some info...
recog_ids = dataset.get_ids_by_features('recognition', threshold=0.001)
print "We found %d studies of recognition" % len(recog_ids)

# Repeat for recollection studies
recoll_ids = dataset.get_ids_by_features('recollection', threshold=0.001)
print "We found %d studies of recollection" % len(recoll_ids)

# Run the meta-analysis
ma = meta.MetaAnalysis(dataset, recog_ids, recoll_ids)
ma.save_results('recognition_vs_recollection')

# <markdowncell>

# This produces the same set of maps we've seen before, except the images now represent a meta-analytic contrast between two specific sets of studies, rather than between one set of studies and all other studies in the database.
# 
# It's worth noting that meta-analytic contrasts generated using Neurosynth should be interpreted very cautiously. Remember that this is a meta-analytic contrast rather than a meta-analysis of contrasts. In the above example, we're comparing activation in all studies in which the term recognition shows up often to activation in all studies in which the term recollection shows up often (implicitly excluding studies that use both terms). We are NOT meta-analytically combining direct contrasts of recollection and recognition, which would be a much more sensible thing to do (but is something that can't be readily automated).
# 
# ### Seed-based coactivation maps
# 
# By now you're all familiar with seed-based functional connectivity. We can do something very similar at a meta-analytic level (e.g., Toro et al, 2008, Robinson et al, 2010, Chang et al, 2012) using the Neurosynth data. Specifically, we can define a seed region and then ask what other regions tend to be reported in studies that report activity in our seed region. The Neurosynth tools make this very easy to do. We can either pass in a mask image defining our ROI, or pass in a list of coordinates to use as the centroid of spheres. In this example, we'll do the latter:

# <codecell>

# Seed-based coactivation
network.coactivation(dataset, [[0, 20, 28]], threshold=0.1, outroot='coactivation_from_coords', r=10)

# <markdowncell>

# Here we're generating a coactivation map for a sphere with radius 10 mm centered on an anterior cingulate cortex (ACC) voxel. The threshold argument indicates what proportion of voxels within the ACC sphere have to be activated for a study to be considered 'active'.
# 
# In general, meta-analytic coactivation produces results quite similar--but substantially less spatially specific--than time series-based functional connectivity. Note that if you're only interested in individual points in the brain, you can find precomputed coactivation maps for spheres centered on every gray matter voxel in the brain on the Neurosynth website.
# 
# ### Decoding your own images
# 
# One of the most useful features of Neurosynth is the ability to 'decode' arbitrary images by assessing their similarity to the reverse inference meta-analysis maps generated for different terms. For example, you might wonder whether a group-level z-score map for some experimental contrast is more consistent with recollection or with recognition. You could even use Neurosynth as a simple (but often effective) classifier by running a series of individual subjects through the decoder and picking the class (i.e., term) with the highest similarity. Perhaps the most powerful--though somewhat more computationally intensive--use is to do open-ended decoding. That is, we can take the entire set of features included in the base Neurosynth data download and rank-order them by similarity to each of our input images.
# 
# In this example, we'll decode three insula-based coactivation networks drawn from Chang, Yarkoni, Khaw, & Sanfey (2012). You should substitute your own images into the list below. We assess the similarity of each map with respect to 9 different terms and save the results to a file. Note that if we left the features argument unspecified, the decoder would default to using the entire set of 500+ features (which will take a few minutes on most machines unless you've pregenerated the feature maps--but that's for a different tutorial).

# <codecell>

# Decode images
decoder = decode.Decoder(dataset, features=['taste', 'disgust', 'emotion', 'auditory', 'pain', 'somatosensory', 'conflict', 'switching', 'inhibition'])
data = decoder.decode(['pIns.nii.gz', 'vIns.nii.gz', 'dIns.nii.gz'], save='decoding_results.txt')

# <markdowncell>

# In decoding_results.txt, we have features in rows, and input images in columns. By default, each cell reflects the pearson correlation between the corresponding input image (i.e., the column) and reverse inference meta-analysis map (i.e., the row). Sort the columns in descending order and you've got a crude but potentially quite useful open-ended decoding of your images. Mind you, if you're lazy, you can achieve the same thing by uploading your images to [NeuroVault](http://neurovault.org) and then using the (currently experimental) [decode](http://neurosynth.org) function on the [Neurosynth website](http://neurosynth.org).

# <codecell>


