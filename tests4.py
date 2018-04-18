#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from miniframe import kernels
from miniframe.BIGgp import BIGgp
from miniframe.BIGgp import isposdef
from miniframe.BIGgp import scale


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats
from scipy.stats import multivariate_normal
from scipy.optimize import minimize

#data
t,rv, rvyerr, bis, rhk,sig_rhk = np.loadtxt("miniframe/datasets/HD41248_harps.rdb",skiprows=2,unpack=True, usecols=(0,1,2,5,9,10))
bis_err = 2*rvyerr

rv, rvyerr = scale(rv, rvyerr)
rhk, sig_rhk = scale(rhk,sig_rhk)
bis, bis_err = scale(bis,bis_err)
t = np.linspace(1,300,228)
rvyerr = 0.2*np.ones_like(t)
rv = np.sin(t) + rvyerr * np.random.randn(len(t))

y = np.hstack((rv,rhk,bis))
yerr = np.hstack((rvyerr,sig_rhk,bis_err))

#GP object
gpObj = BIGgp(kernels.QuasiPeriodic,[None,None, None] , t=t,
              rv=rv, rverr=rvyerr, bis=bis, sig_bis=bis_err, rhk=rhk, sig_rhk=sig_rhk)


lp, le, P = 1, 10, 10
vc1, vr1 = 0, 0
vc2, vr2 = 1000, 0
np.random.seed(seed=233423)
### Example #######################################
# a = [lp, le, P, vc, vr, lc, bc, br] -> kernel parameters
a = np.array([lp,le, P, vc1,vr1, 0,0,0])


print('Likelihood with no mean function =', gpObj.log_likelihood(a, [], y))
plt.figure()
for i in range(1):
    plt.plot(t, gpObj.sample_from_G(t,a))
plt.show()

np.random.seed(seed=233423)
# a = [lp, le, P, vc, vr, lc, bc, br] -> kernel parameters
a = np.array([lp,le, P, vc2,vr2, 0,0,0])

#GP object
gpObj = BIGgp(kernels.QuasiPeriodic,[None,None, None] , t=t,
              rv=rv, rverr=rvyerr, bis=bis, sig_bis=bis_err, rhk=rhk, sig_rhk=sig_rhk)
print('Likelihood with no mean function =', gpObj.log_likelihood(a, [], y))
plt.figure()
for j in range(1):
    plt.plot(t, gpObj.sample_from_G(t,a))
plt.show()