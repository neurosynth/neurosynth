from distutils.core import setup
files = ['resources/*']
setup(name = "neurosynth", 
			version = "0.2", 
			maintainer='Tal Yarkoni', 
			maintainer_email='tyarkoni@gmail.com', 
			url='http://github.com/neurosynth/core',
      		packages = ["neurosynth", "neurosynth.base", "neurosynth.analysis", "tests"],
			package_data = {'neurosynth' : files})