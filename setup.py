import os
import sys
import setuptools
from setuptools import setup, find_packages

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
      download_url='https://github.com/neurosynth/neurosynth/tarball/%s' % __version__,
      install_requires=['numpy', 'scipy', 'pandas', 'ply', 'scikit-learn',
                        'nibabel', 'six', 'biopython'],
      packages=find_packages(),
      package_data={'neurosynth': ['resources/*'],
                    'neurosynth.tests': ['data/*']
                    },
      **extra_setuptools_args
      )
