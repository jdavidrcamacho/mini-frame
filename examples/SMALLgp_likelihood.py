#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from miniframe import kernels
from miniframe.SMALLgp import SMALLgp #importing Jones et al. framework
from miniframe.means import Constant, Linear, Keplerian

import numpy as np

#Importing the data
phase, flux, rv, bis = np.loadtxt("/home/joaocamacho/GitHub/mini-frame/miniframe/datasets/1spot_soap.rdb",
                                  skiprows=2, unpack=True, 
                                  usecols=(0, 1, 2, 3))
t = 25.05 * phase
log_rhk = flux**2
rv = 100*rv
bis = 100*bis
rhk = log_rhk
flux = flux

rms_rv = np.sqrt((1./rv.size*np.sum(rv**2)))
rms_bis = np.sqrt((1./bis.size*np.sum(bis**2)))
rms_rhk = np.sqrt((1./rhk.size*np.sum(rhk**2)))
rvyerr = 0.05*rms_rv * np.ones(rv.size)
bis_err = 0.10*rms_bis * np.ones(bis.size)
sig_rhk = 0.20*rms_rhk * np.ones(rhk.size)

#Making our GP object
gpObj = SMALLgp(kernels.QuasiPeriodic, None, [None, None, None], 
                t, rv, rvyerr, bis, bis_err, rhk, sig_rhk)

#framework parameters
a = [1, 2, 3, 4,
     2, 2, 2,
     2, 2, 2,
     1, 1, 1]
b = []
c = []

#Calculation of the log marginal likelihood
print(gpObj.log_likelihood(a, b, c))