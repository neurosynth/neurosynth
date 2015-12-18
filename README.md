
# What is Neurosynth?

Neurosynth is a Python package for large-scale synthesis of functional neuroimaging data.

## Code status

* [![tests status](https://secure.travis-ci.org/neurosynth/neurosynth.png?branch=master)](https://travis-ci.org/neurosynth/neurosynth) travis-ci.org (master branch)

* [![Coverage Status](https://coveralls.io/repos/neurosynth/neurosynth/badge.png?branch=master)](https://coveralls.io/r/neurosynth/neurosynth?branch=master)

## Installation

Dependencies:

* NumPy/SciPy
* pandas
* NiBabel
* [ply](http://www.dabeaz.com/ply/)
* scikit-learn

We recommend installing the core scientific packages (NumPy, SciPy, pandas, sklearn) via a distribution like [https://store.continuum.io/cshop/anaconda/](Anaconda), which will keep clutter and conflicts to a minimum. The other packages can be installed with pip using the requirements file:

	> pip install -r requirements.txt

Or by name:

	> pip install nibabel ply

Assuming you have those packages in working order, the easiest way to install Neurosynth is from the command line with pip:

	> pip install neurosynth

Alternatively, if you want the latest development version, you can install directly from the github repo:

	> pip install git+https://github.com/neurosynth/neurosynth.git

Depending on your operating system, you may need superuser privileges (prefix the above line with 'sudo').

That's it! You should now be ready to roll.

## Documentation

Documentation, including a [full API reference](http://neurosynth.readthedocs.org/en/latest/reference.html), is available [here](http://neurosynth.readthedocs.org/en/latest/)(caution: work in progress).

## Usage

Running analyses in Neurosynth is pretty straightforward. We're working on a user manual; in the meantime, you can take a look at the code in the /examples directory for an illustration of some common uses cases (some of the examples are in IPython Notebook format; you can view these online by entering the URL of the raw example on github into the online [IPython Notebook Viewer](http://nbviewer.ipython.org)--for example [this tutorial](http://nbviewer.ipython.org/urls/raw.github.com/neurosynth/neurosynth/master/examples/neurosynth_demo.ipynb) provides a nice overview). The rest of this Quickstart guide just covers the bare minimum.

The NeuroSynth dataset resides in a separate submodule located [here](http://github.com/neurosynth/neurosynth-data). Probably the easiest way to get the most recent data though is from within the Neurosynth package itself:

	import neurosynth as ns
	ns.dataset.download(path='.', unpack=True)


...which should download the latest database files and save them to the current directory. Alternatively, you can manually download the data files from the [neurosynth-data repository](http://github.com/neurosynth/neurosynth-data). The latest dataset is always stored in current_data.tar.gz in the root folder. Older datasets are also available in the archive folder.

The dataset archive (current_data.tar.gz) contains 2 files: database.txt and features.txt. These contain the activations and meta-analysis tags for Neurosynth, respectively.

Once you have the data in place, you can generate a new Dataset instance from the database.txt file:

	> from neurosynth.base.dataset import Dataset
	> dataset = Dataset('data/database.txt')

This should take several minutes to process. Note that this is a memory-intensive operation, and may be very slow on machines with less than 8 GB of RAM.

Once initialized, the Dataset instance contains activation data from nearly 10,000 published neuroimaging articles. But it doesn't yet have any features attached to those data, so let's add some:

	> dataset.add_features('data/features.txt')

Now our Dataset has both activation data and some features we can use to manipulate the data with. In this case, the features are just term-based tags--i.e., words that occur in the abstracts of the articles from which the dataset is drawn (for details, see this [Nature Methods] paper, or the Neurosynth website).

We can now do various kinds of analyses with the data. For example, we can use the features we just added to perform automated large-scale meta-analyses. Let's see what features we have:

	> dataset.get_feature_names()
	['phonetic', 'associative', 'cues', 'visually', ... ]

We can use these features--either in isolation or in combination--to select articles for inclusion in a meta-analysis. For example, suppose we want to run a meta-analysis of emotion studies. We could operationally define a study of emotion as one in which the authors used words starting with 'emo' with high frequency:

	> ids = dataset.get_studies(features='emo*', frequency_threshold=0.001)

Here we're asking for a list of IDs of all studies that use words starting with 'emo' (e.g.,'emotion', 'emotional', 'emotionally', etc.) at a frequency of 1 in 1,000 words or greater (in other words, if an article has 5,000 words of text, it will only be included in our set if it uses words starting with 'emo' at least 5 times).

	> len(ids)
	639

The resulting set includes 639 studies.

Once we've got a set of studies we're happy with, we can run a simple meta-analysis, prefixing all output files with the string 'emotion' to distinguish them from other analyses we might run:

	> from neurosynth.analysis import meta
	> ma = meta.MetaAnalysis(dataset, ids)
	> ma.save_results('some_directory/emotion')

You should now have a set of Nifti-format brain images on your drive that display various meta-analytic results. The image names are somewhat cryptic; see the Documentation for details. It's important to note that the meta-analysis routines currently implemented in Neurosynth aren't very sophisticated; they're designed primarily for efficiency (most analyses should take just a few seconds), and take multiple shortcuts as compared to other packages like ALE or MKDA. But with that caveat in mind (and one that will hopefully be remedied in the near future), Neurosynth gives you a streamlined and quick way of running large-scale meta-analyses of fMRI data.


## Getting help

For a more comprehensive set of examples, see [this tutorial](http://nbviewer.ipython.org/urls/raw.github.com/neurosynth/neurosynth/master/examples/neurosynth_demo.ipynb)--also included in IPython Notebook form in the examples/ folder (along with several other simpler examples).

For bugs or feature requests, please [create a new issue](https://github.com/neurosynth/neurosynth/issues/new). If you run into problems installing or using the software, try posting to the [Neurosynth Google group](https://groups.google.com/forum/#!forum/neurosynthlist) or email [Tal Yarkoni](mailto:tyarkoni@gmail.com).
