#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 09:58:26 2018

@author: joaocamacho
"""
import numpy as np
from miniframe import SMALLgp
from miniframe import kernels
from miniframe.means import Constant, Linear, Keplerian

t,rv, rvyerr, bis, rhk,sig_rhk = np.loadtxt("miniframe/datasets/HD41248_harps.rdb",skiprows=2,
                                            unpack=True, usecols=(0,1,2,5,9,10))
bis_err=2*rvyerr


# a = [l, a_01,a_02,a_03, a_11,a_12,a_13, ..., a_l1,a_l2,a_l3 ]
gpObj = SMALLgp.SMALLgp(kernels.SquaredExponential, [Constant, Constant], 2, t, rv, rvyerr, bis, bis_err)
a = np.array([1, 2,1,0.5, 1,2,3])
b = [10, 5]
print(gpObj.log_likelihood(a,b))

gpObj = SMALLgp.SMALLgp(kernels.SquaredExponential, [Constant, Constant, None], 3, t, rv, rvyerr, bis, bis_err, rhk, sig_rhk)
a = np.array([1, 2,1,0.5, 1,2,3, 1,0,0])
b = [10, 5]
print(gpObj.log_likelihood(a,b))

k11 = gpObj.kii(a = a, x = t, position = 1)
k22 = gpObj.kii(a = a, x = t, position = 2)
k33 = gpObj.kii(a = a, x = t, position = 3)
k12 = gpObj.kij(a = a, x = t, position1 = 1, position2 = 2)
k13 = gpObj.kij(a = a, x = t, position1 = 1, position2 = 3)
k23 = gpObj.kij(a = a, x = t, position1 = 2, position2 = 3)
K = gpObj.compute_matrix(a)

################################################################################