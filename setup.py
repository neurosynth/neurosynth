import os
import sys

# Borrowing a trick from nibabel to enable some functionality coming
# from setuptools.
# For some commands, use setuptools
if len(set(('test', 'easy_install')).intersection(sys.argv)) > 0:
    import setuptools

from distutils.core import setup

extra_setuptools_args = {}
if 'setuptools' in sys.modules:
    extra_setuptools_args = dict(
        tests_require=['nose'],
        test_suite='nose.collector',
        extras_require=dict(
            test='nose>=0.10.1')
    )

# fetch version from within neurosynth module
with open(os.path.join('neurosynth', 'version.py')) as f:
    exec(f.read())

setup(name="neurosynth",
      version=__version__,
      description="Large-scale synthesis of functional neuroimaging data",
      maintainer='Tal Yarkoni',
      maintainer_email='tyarkoni@gmail.com',
      url='http://github.com/neurosynth/neurosynth',
      packages=["neurosynth",
                  "neurosynth.base",
                  "neurosynth.analysis",
                  "neurosynth.tests"],
      package_data={'neurosynth': ['resources/*'],
                    'neurosynth.tests': ['data/*']
                    },
      **extra_setuptools_args
      )
