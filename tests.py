# -*- coding: utf-8 -*-
import kernels
from BIGgp import BIGgp
from BIGgp import isposdef

import numpy as np
import emcee

import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
from scipy import stats
from scipy.stats import multivariate_normal
from scipy.optimize import minimize

### a = [l, vc, vr, lc, bc, br] -> kernel parameters
a = np.array([10, 10, 10, 10, 10, 10])


### Example for the squared exponential  #######################################
### 1st set pf data - original data
t,rv,bis, rhk, rvyerr,sig_rhk = np.loadtxt("HD41248_harps.rdb",skiprows=2,unpack=True, usecols=(0,1,2,5,9,10))
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
################################################################################
### 2nd set of data - evenly spaced time
t,rv,bis, rhk, rvyerr,sig_rhk = np.loadtxt("HD41248_harps.rdb",skiprows=2,unpack=True, usecols=(0,1,2,5,9,10))
bis_err=2*rvyerr
t = np.linspace(100, 500, 228)

gpObj = BIGgp(kernels.SquaredExponential, t=t, rv=rv, rverr=rvyerr,
                    bis=bis, sig_bis=bis_err, rhk=rhk, sig_rhk=sig_rhk)
matriz = gpObj.compute_matrix(a)
print(isposdef(matriz))

#measurements
y=np.hstack((rv,rhk,bis))
#error in measurements
yerr=np.hstack((rvyerr,sig_rhk,2*rvyerr))
# #log-likelihood
print('2nd try ->', gpObj.log_likelihood(a, y))
print()
################################################################################
### one of rajpaul's dataset from jones et al. 2017
t, rv, rhk, bis, rvyerr, sig_rhk,bis_err  = np.loadtxt("rajpaul_hd_fine_datasets.csv",delimiter=',',skiprows=1,unpack=True)
gpObj = BIGgp(kernels.SquaredExponential, t=t, rv=rv, rverr=rvyerr,
                    bis=bis, sig_bis=bis_err, rhk=rhk, sig_rhk=sig_rhk)
matriz = gpObj.compute_matrix(a)
print(isposdef(matriz))

#measurements
y=np.hstack((rv,rhk,bis))
#error in measurements
yerr=np.hstack((rvyerr,sig_rhk,2*rvyerr))
# #log-likelihood
print('3rd try ->', gpObj.log_likelihood(a, y))
print()
################################################################################
### if this works -.-
t,rv,bis, rhk, rvyerr,sig_rhk = np.loadtxt("HD41248_harps.rdb",skiprows=2,unpack=True, usecols=(0,1,2,5,9,10))

whynot = np.random.choice(228, 228, replace=False)
t1 = t[whynot]
bis1= bis[whynot]
rv1 = rv[whynot]
rhk1 = rhk[whynot]
rvyerr1 = rvyerr[whynot]
sig_rhk1 = sig_rhk[whynot]
bis_err1 = 2*rvyerr1

gpObj = BIGgp(kernels.SquaredExponential, t=t1, rv=rv1, rverr=rvyerr1,
                    bis=bis1, sig_bis=bis_err1, rhk=rhk1, sig_rhk=sig_rhk1)
matriz = gpObj.compute_matrix(a)
print(isposdef(matriz))

#measurements
y=np.hstack((rv1,rhk1,bis1))
#error in measurements
yerr=np.hstack((rvyerr1,sig_rhk1,bis_err1))
# #log-likelihood
print('1st try again ->', gpObj.log_likelihood(a, y))
print()