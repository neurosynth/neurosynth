## 0.3.2 (November 7, 2014
- Dataset, FeatureTable, and ImageTable all now use pandas everywhere
- New datasets in data submodule
- Improved test coverage of Dataset
- Substantially improved Masker object that handles multiple masking layers
- Added functionality to downsample whole brain to isometric grid
- Initial stab at clustering pipeline (work in progress)
- Improved interface for adding new features to Dataset
- Bug fixes/minor improvements:
	- FeatureTable no longer breaks when reading in columns with N-gram names using pandas
	- get_feature_data() now takes reorder argument
	- convert pandas SparseDataFrame to CSR_matrix on data access as workaround for pandas bug
	- Fixed dataset loading/saving bug for datasets loaded/saved using previous versions
	- Fixed the above bug *again*, because it's a persistent little bugger
	- Updated dependency list
	- Fixed a variety of minor bugs caused by minor API changes in new pandas versions

## 0.3.1 (May 15, 2014)
- Data matrices now stored internally as pandas DataFrames
- Began implementation of classifier class (thanks to @zorro4)
- Better/faster data accessor
- Added several usage examples in /examples, including an IPython demo
- Bug fixes/minor tweaks:
	- Random voxel selection routine works properly now
	- All IDs are now strings to avoid type mismatches for numerical IDs
	- Patched README and travis/coveralls definitions to reflect current dependencies

## 0.3.0 (May 26, 2013)

First-ever *official* release following the [Neurosynth hackathon](http://hackathon.neurosynth.org)

- Enabled travis-ci.org and coveralls.io
- Most of the current functionality was developed pre-0.3.0, so, really, just about everything
- Added tests, thanks to @yarikoptic's heroic efforts
- [Testkraut](https://github.com/neurodebian/testkraut) integration

Neurosynth is a work in progress and we reserve the right to occasionally break the API.
