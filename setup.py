#!/usr/bin/env python

from setuptools import setup
import numpy as np
from Cython.Build import cythonize

setup(name='gendis-subgroup',
      version='0.2',
      description='TODO',
      author='Samir Saliba Jr',
      author_email='samirtsj@gmail.com',
      url='TODO',
      packages=['gendis'],
      install_requires=[],
      test_suite='nose2.collector.collector',
      # ext_modules = cythonize(['gendis/pairwise_dist.pyx']),
      include_dirs=[np.get_include()]
     )
