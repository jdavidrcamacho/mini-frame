# -*- coding: utf-8 -*-

##### Necessary scripts to everything work #####
"""
Package to analyze radial velocity measurements using Gaussian processes.

BIGgp: 
    It is where all the math of Rajpaul et al is made.
kernels:
    Contains all the developed kernels.
means:
    Contains all the developed mean functions.
SMALLgp: 
    It is where all the math of Jones et al in made (in development).
kernels_autograd:
    Outdated (to be removed).
"""

from miniframe import BIGgp

from miniframe import kernels

from miniframe import means

from miniframe import SMALLgp