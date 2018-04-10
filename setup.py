#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup

setup(name='miniframe',
      version='0.1',
      description='Implementation of the Gaussian process framework of Rajpaul et al. (2015)',
      author='João Camacho',
      author_email='joao.camacho@astro.up.pt',
      license='MIT',
      url='https://github.com/jdavidrcamacho/mini-frame',
      packages=['miniframe'],
      install_requires=[
        'numpy',
        'scipy',
        'matplotlib>=1.5.3',
        'corner',
        'emcee'
      ],
     )
