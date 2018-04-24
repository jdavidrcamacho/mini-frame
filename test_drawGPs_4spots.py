#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from miniframe import kernels
from miniframe.BIGgp import BIGgp
from miniframe.means import Constant, Linear, Keplerian

import numpy as np
import emcee
import sys
from time import time
from scipy import stats

import matplotlib.pyplot as plt

start_time = time()

phase, flux, rv, bis = np.loadtxt("miniframe/datasets/4spots_soap.rdb",skiprows=2,unpack=True, usecols=(0, 1, 2, 3))
log_rhk = flux**2

rv = 100*rv
bis = 100*bis
rhk = log_rhk
flux = flux
t = 25.05 * phase

rms_rv = np.sqrt((1./rv.size*np.sum(rv**2)))
rms_bis = np.sqrt((1./bis.size*np.sum(bis**2)))
rms_rhk = np.sqrt((1./rhk.size*np.sum(rhk**2)))
rvyerr = 0.05*rms_rv * np.ones(200)
bis_err = 0.10*rms_bis * np.ones(200)
sig_rhk = 0.20*rms_rhk * np.ones(200)


y = np.hstack((rv,bis))
yerr = np.hstack((rvyerr,bis_err))

gpObj = BIGgp(kernels.QuasiPeriodic,[None,None, None] , t=t,
                  rv=rv, rverr=rvyerr, bis=bis, sig_bis=bis_err, rhk=rhk, sig_rhk=sig_rhk)



time = np.linspace(0, 75.15, 500)
# a = [lp, le, p ,vc, vr, lc, bc, br]
a = [3947.411, 0.626, 23.584, 20096.503, 5.651e-5, 298.029, 4088.363, 16900.052]

mu1, std1 = gpObj.draw_from_gp(time, a,  model = 'rv')
mu2, std2 = gpObj.draw_from_gp(time, a,  model = 'bis')
mu3, std3 = gpObj.draw_from_gp(time, a,  model = 'rhk')

f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
ax1.set_title(' ')
ax1.fill_between(time, mu1+std1, mu1-std1, color="grey", alpha=0.5)
ax1.plot(time, mu1, "k-", alpha=1, lw=1.5)
ax1.plot(t,rv,"b*")
ax1.set_ylabel("RVs")

ax2.fill_between(time, mu2+std2, mu2-std2, color="grey", alpha=0.5)
ax2.plot(time, mu2, "k-", alpha=1, lw=1.5)
ax2.plot(t,bis,"b*")
ax2.set_ylabel("BIS")

ax3.fill_between(time, mu3+std3, mu3-std3, color="grey", alpha=0.5)
ax3.plot(time, mu3, "k-", alpha=1, lw=1.5)
ax3.plot(t,rhk,"b*")
ax3.set_ylabel("flux")
ax3.set_xlabel("time")
plt.show()
