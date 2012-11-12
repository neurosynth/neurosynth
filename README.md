test
# What is Neurosynth?

Neurosynth is a Python package for large-scale synthesis of functional neuroimaging data.

## Installation

Neurosynth depends on the NumPy/SciPy libraries. It also depends on NiBabel. Assuming you have those packages in working order, download this package, then install it from source:

	> python setup.py install

Depending on your operating system, you may need superuser privileges (prefix the above line with 'sudo').

That's it! You should now be ready to roll.


## Usage

Running analyses in Neurosynth is pretty straightforward. For a reasonably thorough walk-through, see the Getting Started page in the documentation (well, once it exists). This Quickstart guide just covers the bare minimum.

First, [download some data](http://neurosynth.org/data/current_data.tar.gz) from the Neurosynth website:

	> curl -O http://neurosynth.org/data/current_data.tar.gz

Unpack the archive, which should contain 2 files: database.txt and features.txt.

Now generate a new Dataset instance from the database.txt file:

	> from neurosynth.base.dataset import Dataset
	> dataset = Dataset('database.txt')

This should take several minutes to process.

Once initialized, the Dataset instance contains activation data from nearly 6,000 published neuroimaging articles. But it doesn't yet have any features attached to those data, so let's add some:

	> dataset.add_features('features.txt')

Now our Dataset has both activation data and some features we can use to manipulate the data with. In this case, the features are just term-based tags--i.e., words that occur frequently in the articles from which the dataset is drawn (for details, see this [Nature Methods] paper, or the Neurosynth website).

We can now do various kinds of analyses with the data. For example, we can use the features we just added to perform automated large-scale meta-analyses. Let's see what features we have:

	> dataset.list_features()
	['phonetic', 'associative', 'cues', 'visually', ... ]

We can use these features--either in isolation or in combination--to select articles for inclusion in a meta-analysis. For example, suppose we want to run a meta-analysis of emotion studies. We could operationally define a study of emotion as one in which the authors used words starting with 'emo' with high frequency:

	> ids = dataset.get_ids_by_features('emo*', threshold=0.001)

Here we're asking for a list of IDs of all studies that use words starting with 'emo' (e.g.,'emotion', 'emotional', 'emotionally', etc.) at a frequency of 1 in 1,000 words or greater (in other words, if an article has 5,000 words of text, it will only be included in our set if it uses words starting with 'emo' at least 5 times).

	> len(ids)
	639

The resulting set includes 639 studies.

Once we've got a set of studies we're happy with, we can run a simple meta-analysis, prefixing all output files with the string 'emotion' to distinguish them from other analyses we might run:

	> from neurosynth.analysis import meta
	> ma = meta.MetaAnalysis(dataset, ids)
	> ma.save_results('some_directory/emotion')

You should now have a set of Nifti-format brain images on your drive that display various meta-analytic results. The image names are somewhat cryptic; see the Documentation for details. It's important to note that the meta-analysis routines currently implemented in Neurosynth aren't very sophisticated; they're designed primarily for efficiency (most analyses should take just a few seconds), and take multiple shortcuts as compared to other packages like ALE or MKDA. But with that caveat in mind (and one that will hopefully be remedied in the near future), Neurosynth gives you a streamlined and quick way of running large-scale meta-analyses of fMRI data.
