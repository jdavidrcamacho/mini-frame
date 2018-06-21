#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from miniframe import kernels
from miniframe.BIGgp import BIGgp
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


y = np.hstack((rv,rhk,bis))
yerr = np.hstack((rvyerr,sig_rhk,bis_err))

gpObj = BIGgp(kernels.QuasiPeriodic,[Constant, Constant, Constant] , t=t,
                  rv=rv, rverr=rvyerr, bis=bis, sig_bis=bis_err, rhk=rhk, sig_rhk=sig_rhk)

time = np.linspace(0, 76, 1000)
#a = [kernel le, kernel lp, kernel period, white noise, vc, vr, lc, bc, br]

## with WN
a = [1083.1091125670669, 1.215034890646895, 25.07312864134963, 0.031950873139068185,
     6.064550597545819, 4.23391412490362, 
     0.3552833092394814, 
     12.807709071739335, 9.755026033334879]
#a = [7007.513548743503, 1.0226921706026995, 25.05465110454914, 0.5778395294007469,
#     4.9704569197133965, 4.219512871079798, 
#     0.3827768189302914, 
#     5.711096223144502, 11.427452171937151]
#a = [14986.7403965787, 1.8893002353788237, 25.05233528277538, 0.07207944027787755,
#     21.746034894186458, 0.010853033462958014,
#     0.0003587694052429295,
#     0.03553003103967403, 17.973951017585645]
#a = [6331.984038213745, 1.8033164648664974, 25.070884614456006, 0.06893529979875161,
#     17.694209878440283, 0.06384752322598014,
#     0.0009857564298580237, 
#     0.3493800719825983, 25.298207575936242]

#a =[ 9571.348317608801, 0.04020139027969265, 24.945951504582425, 0.09933982712895068,
#    0.11198019279705342, 3.419412541993913,
#    0.9911054746798689,
#    0.0024402208361650295, 0.7775385931642231]

## no WN
#a  = [ 12711.508704216278, 2.7492848130724012, 25.04909115774515, 0,
#      0.20880597394351352, 95.88291303705131, 
#      8.403754137892965e-05,
#      0.24608962459685757, 0.0018744919973775012]

## free le
#a = [955.0257888302123 , 2.0745451592623465 ,25.00497084749747 , 0.0006389729515563523,
#     0.3732013481460286,77.98062589194765,
#     9.527822500555753e-05,
#     0.36884484583630284, 0.009156187776153702]

# more noise
#a = [ 311.9696701377493, 0.042629709086808736, 24.94097149392546, 0.09877775788734904,
#     0.0012040807115627087, 3.2049962257655915,
#     0.9788521319509121,
#     0.4571461994576663, 0.7538996195916922]

#a = [1000, 1, 25, 0.05,
#     1*6, 0*1, 
#     1*0.1, 
#     1*5, 0*1]


b = [0, 0, 0]

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

#f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey = True, sharex = True)
#ax1.imshow(cov11)
#ax2.imshow(cov33)
#ax3.imshow(cov22)
#plt.show()
end = tempo()
print ("It took", end-start, 's')