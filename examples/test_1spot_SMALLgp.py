#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from miniframe import kernels
from miniframe.SMALLgp import SMALLgp
from miniframe.means import Constant, Linear, Keplerian
from time import time

import numpy as np
import emcee
import matplotlib.pyplot as plt
import _pickle as pickle

from matplotlib.ticker import MaxNLocator
from scipy import stats

phase, flux, rv, bis = np.loadtxt("/home/joaocamacho/GitHub/mini-frame/miniframe/datasets/1spot_soap.rdb",
                                  skiprows=2, unpack=True, 
                                  usecols=(0, 1, 2, 3))
t = 25.05 * phase
log_rhk = flux**2

rv = 100*rv
bis = 100*bis
rhk = log_rhk
flux = flux
#plt.plot(t,rv,'.')

rms_rv = np.sqrt((1./rv.size*np.sum(rv**2)))
rms_bis = np.sqrt((1./bis.size*np.sum(bis**2)))
rms_rhk = np.sqrt((1./rhk.size*np.sum(rhk**2)))
rvyerr = 0.05*rms_rv * np.ones(rv.size)
bis_err = 0.10*rms_bis * np.ones(bis.size)
sig_rhk = 0.20*rms_rhk * np.ones(rhk.size)

#rv, rvyerr = scale(rv, rvyerr)
#rhk, sig_rhk = scale(rhk,sig_rhk)
#bis, bis_err = scale(bis,bis_err)

#y = np.hstack((rv,rhk,bis))
#yerr = np.hstack((rvyerr,sig_rhk,bis_err))

gpObj = SMALLgp(kernels.QuasiPeriodic, None, [None, None, None], 
                t, rv, rvyerr, bis, bis_err, rhk, sig_rhk)

a = [1, 2, 3, 4,
     2, 2, 2,
     2, 2, 2,
     1, 1, 1]
b = []
c = []

print(gpObj.log_likelihood(a, b, c))