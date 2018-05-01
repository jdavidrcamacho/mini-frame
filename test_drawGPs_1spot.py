#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from miniframe import kernels
from miniframe.BIGgp import BIGgp
from miniframe.BIGgp import isposdef
from miniframe.BIGgp import scale
from miniframe.means import Constant, Linear, Keplerian
from time import time as tempo

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats

#import _pickle as pickle
plt.close('all')

phase, flux, rv, bis = np.loadtxt("miniframe/datasets/1spot_soap.rdb",
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


y = np.hstack((rv,rhk,bis))
yerr = np.hstack((rvyerr,sig_rhk,bis_err))

gpObj = BIGgp(kernels.QuasiPeriodic,[Constant, Constant, Constant] , t=t,
                  rv=rv, rverr=rvyerr, bis=bis, sig_bis=bis_err, rhk=rhk, sig_rhk=sig_rhk)

time = np.linspace(0, 76, 1000)
start = tempo()

#a = [kernel le, kernel lp, kernel period, white noise, vc, vr, lc, bc, br]
a = [1083.1091125670669, 1.215034890646895, 25.07312864134963, 0.031950873139068185,
     6.064550597545819, 4.23391412490362, 
     0.3552833092394814, 
     12.807709071739335, 9.755026033334879]
b = [0, 0, 0]

#mu1, cov1, std1 = gpObj.predict_rv(time, a)
#mu2, cov2, std2 = gpObj.predict_bis(time, a)
#mu3, cov3, std3 = gpObj.predict_rhk(time, a)
#
#f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
#ax1.set_title(' ')
#ax1.fill_between(time, mu1+std1, mu1-std1, color="grey", alpha=0.5)
#ax1.plot(time, mu1, "k--", alpha=1, lw=1.5)
#ax1.plot(t,rv,"b.")
#ax1.set_ylabel("RVs")
#
#ax2.fill_between(time, mu3+std3, mu3-std3, color="grey", alpha=0.5)
#ax2.plot(time, mu3, "k--", alpha=1, lw=1.5)
#ax2.plot(t,rhk,"b.")
#ax2.set_ylabel("flux")
#
#ax3.fill_between(time, mu2+std2, mu2-std2, color="grey", alpha=0.5)
#ax3.plot(time, mu2, "k--", alpha=1, lw=1.5)
#ax3.plot(t,bis,"b.")
#ax3.set_ylabel("BIS")
#ax3.set_xlabel("time")
#plt.show()
#
#f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey = True, sharex = True)
#ax1.imshow(cov1)
#ax2.imshow(cov3)
#ax3.imshow(cov2)
#plt.show()
#end = tempo()
#print ("It took", end-start)


start = tempo()
mu11, cov11, std11  = gpObj.predict_gp(time, a, b, model = 'rv')
mu22, cov22, std22  = gpObj.predict_gp(time, a, b, model = 'bis')
mu33, cov33, std33  = gpObj.predict_gp(time, a, b, model = 'rhk')

f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
ax1.set_title(' ')
ax1.fill_between(time, mu11+std11, mu11-std11, color="grey", alpha=0.5)
ax1.plot(time, mu11, "k--", alpha=1, lw=1.5)
ax1.plot(t,rv,"b.")
ax1.set_ylabel("RVs")

ax2.fill_between(time, mu33+std33, mu33-std33, color="grey", alpha=0.5)
ax2.plot(time, mu33, "k--", alpha=1, lw=1.5)
ax2.plot(t,rhk,"b.")
ax2.set_ylabel("flux")

ax3.fill_between(time, mu22+std22, mu22-std22, color="grey", alpha=0.5)
ax3.plot(time, mu22, "k--", alpha=1, lw=1.5)
ax3.plot(t,bis,"b.")
ax3.set_ylabel("BIS")
ax3.set_xlabel("time")
plt.show()

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey = True, sharex = True)
ax1.imshow(cov11)
ax2.imshow(cov33)
ax3.imshow(cov22)
plt.show()
end = tempo()
print ("It took", end-start)