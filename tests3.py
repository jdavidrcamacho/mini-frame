# -*- coding: utf-8 -*-
from miniframe import kernels
from miniframe.BIGgp import BIGgp
from miniframe.BIGgp import isposdef
import sys
import numpy as np
import emcee

import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
from scipy import stats
from scipy.stats import multivariate_normal
from scipy.optimize import minimize

### a = [l, vc, vr, lc, bc, br] -> kernel parameters
a = np.array([1, 1,10,-1,10,10])
### b = [P, e, K, w] -> keplerian parameters
b = np.array([100, 0.5, 10, 10])

### Example for the squared exponential  #######################################
### 1st set pf data - zero mean
t,rv, rvyerr, bis, rhk,sig_rhk = np.loadtxt("HD41248_harps.rdb",skiprows=2,unpack=True, usecols=(0,1,2,5,9,10))
#t = np.linspace(1, 300, 228)
bis_err=2*rvyerr

gpObj = BIGgp(kernels.SquaredExponential, t=t, rv=rv, rverr=rvyerr,
                    bis=bis, sig_bis=bis_err, rhk=rhk, sig_rhk=sig_rhk)
matriz = gpObj.compute_matrix(a)
print(isposdef(matriz))

#measurements
y=np.hstack((rv,rhk,bis))
#error in measurements
yerr=np.hstack((rvyerr,sig_rhk,2*rvyerr))
# #log-likelihood
print('1st try ->', gpObj.log_likelihood(a, y))
print()

### 2nd set pf data - with keplerian
t,rv, rvyerr, bis, rhk,sig_rhk = np.loadtxt("HD41248_harps.rdb",skiprows=2,unpack=True, usecols=(0,1,2,5,9,10))
#t = np.linspace(1, 300, 228)
bis_err=2*rvyerr

gpObj = BIGgp(kernels.SquaredExponential, t=t, rv=rv, rverr=rvyerr,
                    bis=bis, sig_bis=bis_err, rhk=rhk, sig_rhk=sig_rhk)
matriz = gpObj.compute_matrix(a)
print(isposdef(matriz))

#measurements
y=np.hstack((rv,rhk,bis))
#error in measurements
yerr=np.hstack((rvyerr,sig_rhk,2*rvyerr))
# #log-likelihood
print('1st try ->', gpObj.kepler_likelihood(a, b, y))
print()
