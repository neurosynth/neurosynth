.. neurosynth documentation master file, created by
   sphinx-quickstart on Tue Dec 30 10:29:31 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=============
API Reference
=============

This reference provides detailed documentation for all modules, classes, and 
methods in the current release of Neurosynth. The root Neurosynth package
is comprised of two modules: neurosynth.base and neurosynth.analysis. The 
base module contains functionality for representing, manipulating, and
retrieving data. The analysis module contains functionality for doing more 
interesting things with the data--meta-analysis, clustering, decoding, etc.

.. Contents:

.. .. toctree::
..    :maxdepth: 4

..    neurosynth


:mod:`neurosynth.base`: Data storage/manipulation functionality
===============================================================

Base modules
------------

.. currentmodule:: neurosynth

.. autosummary::
    :toctree: generated

	base.dataset
	base.imageutils
	base.lexparser
	base.mappable
	base.mask
	base.transformations

:mod:`neurosynth.analysis`: Analysis tools
==========================================

Analysis modules
----------------

.. autosummary::
    :toctree: generated

    analysis.classify
    analysis.cluster
    analysis.decode
    analysis.meta
    analysis.network
    analysis.reduce
    analysis.stats


Index
=====

* :ref:`genindex`
* :ref:`modindex`
.. * :ref:`search`

