# -*- coding: utf-8 -*-
import kernels
from BIGgp import BIGgp

import numpy as np
import matplotlib.pylab as pl
import emcee

from matplotlib.ticker import MaxNLocator
from scipy import stats
from scipy.stats import multivariate_normal
from scipy.optimize import minimize


### Example for the squared exponential  ###
#data
t,rv,rvyerr, bis, rhk, sig_rhk = np.loadtxt("HD41248_harps.rdb",skiprows=2,unpack=True, usecols=(0,1,2,5,9,10))
t=t-t[0]
#t = np.linspace(1,160,160)

a = np.array([0.1, 10, 1, 1, 1, 1])

gpObj = BIGgp(kernels.SquaredExponential, t=t, rv=rv, rverr=rvyerr,
                    bis=bis, sig_bis=2*rvyerr, rhk=rhk, sig_rhk=sig_rhk)

#measurements
y=np.hstack((rv,rhk,bis))
#error in measurements
yerr=np.hstack((rvyerr,sig_rhk,2*rvyerr))

# #log-likelihood
print(gpObj.log_likelihood(a,y))

samples=gpObj.sample(a)
pl.plot(t,samples[0:228])

print(gpObj._kernel_matrix(kernels.SquaredExponential,a))