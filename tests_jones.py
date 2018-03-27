#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 09:58:26 2018

@author: joaocamacho
"""
import numpy as np
from BIGgp_jones import BIGgp
import kernels

t,rv, rvyerr, bis, rhk,sig_rhk = np.loadtxt("HD41248_harps.rdb",skiprows=2,
                                            unpack=True, usecols=(0,1,2,5,9,10))
bis_err=2*rvyerr

gpObj = BIGgp(kernels.SquaredExponential, 3, t, rv, rvyerr, bis, bis_err, rhk, sig_rhk)

# a = [l, a_01,a_02,a_03, a_11,a_12,a_13, ..., a_l1,a_l2,a_l3 ]
a = np.array([1, 1,1,0, 1,1,0, 1,1,0])

k11 = gpObj.kii(a = a, x = t, position = 1)
k22 = gpObj.kii(a = a, x = t, position = 2)
k33 = gpObj.kii(a = a, x = t, position = 3)

k12 = gpObj.kij(a = a, x = t, position1 = 1, position2 = 2)
k13 = gpObj.kij(a = a, x = t, position1 = 1, position2 = 3)
k23 = gpObj.kij(a = a, x = t, position1 = 2, position2 = 3)

K = gpObj.compute_matrix(a)
print(gpObj.log_likelihood(a))
