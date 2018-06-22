#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from miniframe import kernels
from miniframe.SMALLgp import SMALLgp
from miniframe.means import Constant, Linear, Keplerian
from time import time as tempo

import numpy as np

import matplotlib.pyplot as plt
plt.close('all')

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

#rv, rvyerr = scale(rv, rvyerr)
#rhk, sig_rhk = scale(rhk,sig_rhk)
#bis, bis_err = scale(bis,bis_err)

#y = np.hstack((rv,rhk,bis))
#yerr = np.hstack((rvyerr,sig_rhk,bis_err))

gpObj = SMALLgp(kernels.QuasiPeriodic, None, [None, None, None], 
                t, rv, rvyerr, rhk, sig_rhk, bis, bis_err)

time = np.linspace(0, 76, 1000)

#a = [13441.501457621176, 4.6483490473930615, 24.93669751950206, 0.0627314509883369,
#     1*47.97591574850945, 1*40.64966532128839, 0*46.59497137307535,
#     1*40.50786737356975, 1*61.789646806855565, 0*34.5895556069076,
#     1*38.95241617890942, 0*40.4588184754026, 0*43.47745123586486] 

a = [1083.1091125670669, 1.215034890646895, 25.07312864134963, 0.031950873139068185,
     6.064550597545819, 4.23391412490362, 0,
     0.3552833092394814, 0, 0,
     12.807709071739335, 9.755026033334879, 0]

b = []
c = [0, 0, 0,
      0, 0, 0]

i, trials = 0, 100
times = []
while i <= trials:
    start = tempo()
    mu11, cov11, std11  = gpObj.predict_gp(time, a, b, c, model = 1)
    mu22, cov22, std22  = gpObj.predict_gp(time, a, b, c, model = 2)
    mu33, cov33, std33  = gpObj.predict_gp(time, a, b, c, model = 3)
    end = tempo()
    times.append(end-start)
    print ("It took", end-start, 's')
    i += 1

print("Average time of the", trials, "trials =", 
      np.mean(np.array(times)), 'seconds')

#f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
#ax1.set_title(' ')
#ax1.fill_between(time, mu11+std11, mu11-std11, color="grey", alpha=0.5)
#ax1.plot(time, mu11, "k--", alpha=1, lw=1.5)
#ax1.plot(t,rv,"b.")
#ax1.set_ylabel("RVs")
#
#ax2.fill_between(time, mu22+std22, mu22-std22, color="grey", alpha=0.5)
#ax2.plot(time, mu22, "k--", alpha=1, lw=1.5)
#ax2.plot(t,rhk,"b.")
#ax2.set_ylabel("flux")
#
#ax3.fill_between(time, mu33+std33, mu33-std33, color="grey", alpha=0.5)
#ax3.plot(time, mu33, "k--", alpha=1, lw=1.5)
#ax3.plot(t,bis,"b.")
#ax3.set_ylabel("BIS")
#ax3.set_xlabel("time")
#plt.show()
#
#f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey = True, sharex = True)
#ax1.imshow(cov11)
#ax2.imshow(cov33)
#ax3.imshow(cov22)
#plt.show()
