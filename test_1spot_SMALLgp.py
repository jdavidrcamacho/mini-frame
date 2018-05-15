#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from miniframe import kernels
from miniframe.SMALLgp import SMALLgp
from miniframe.means import Constant, Linear, Keplerian
from time import time

import numpy as np
import emcee
import matplotlib.pyplot as plt
import _pickle as pickle

from matplotlib.ticker import MaxNLocator
from scipy import stats

phase, flux, rv, bis = np.loadtxt("/home/joaocamacho/GitHub/mini-frame/miniframe/datasets/1spot_soap.rdb",
                                  skiprows=2, unpack=True, 
                                  usecols=(0, 1, 2, 3))
t = 25.05 * phase
log_rhk = flux**2

rv = 100*rv
bis = 100*bis
rhk = log_rhk
flux = flux
#plt.plot(t,rv,'.')

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

gpObj = SMALLgp(kernels.QuasiPeriodic,[None,None, None] , t, 
                rv, rvyerr, bis, bis_err, rhk, sig_rhk)

#### simple sample and marginalization with emcee
runs, burns = 20000, 20000
#probabilistic model
def logprob(p):
    if any([p[0] < -10, p[0] > 10, 
            p[1] < -10, p[1] > np.log(10),
            p[2] < np.log(15), p[2] > np.log(35), 
            p[3] < -10, p[3] > np.log(0.1),

            p[4] < -10, p[4] > np.log(100),
            p[5] < -10, p[5] > np.log(100),

            p[6] < -10, p[6] > np.log(1),

            p[7] < -10, p[7] > np.log(100),
            p[8] < -10, p[8] > np.log(100),]):
        return -np.inf
    logprior = 0.0
    return logprior + gpObj.log_likelihood(np.exp(p), [])


#prior from exp(-10) to exp(10)
le_prior = stats.uniform(np.exp(-10), np.exp(10) -np.exp(-10)) #from exp(-10) to 1
lp_prior = stats.uniform(np.exp(-10), 10 -np.exp(-10)) #from exp(-10) to exp(10)
p_prior = stats.uniform(15, 35-15) #from 15 to 35
wn_prior = stats.uniform(np.exp(-10), 0.1 -np.exp(-10)) #from exp(-10) to exp(10)

vc_prior = stats.uniform(np.exp(-10), 100 -np.exp(-10)) #from exp(-10) to 100
vr_prior = stats.uniform(np.exp(-10), 100 -np.exp(-10)) #from exp(-10) to 100

lc_prior = stats.uniform(np.exp(-10), 1 -np.exp(-10)) #from exp(-10) to 100
bc_prior = stats.uniform(np.exp(-10), 100 -np.exp(-10)) #from exp(-10) to 100
br_prior = stats.uniform(np.exp(-10), 100 -np.exp(-10)) #from exp(-10) to 100


def from_prior():
    #[lp,le,p, vc,vr,lc,bc,br, P,k,e,w,t0, const,const]
    return np.array([ le_prior.rvs(), lp_prior.rvs(), p_prior.rvs(), wn_prior.rvs(),
                    vc_prior.rvs(), vr_prior.rvs(), lc_prior.rvs(), 
                    bc_prior.rvs(), br_prior.rvs()])

#Set up the sampler.
nwalkers, ndim = 2*9, 9
sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob, threads= 4)
#Initialize the walkers.
p0=[np.log(from_prior()) for i in range(nwalkers)]

print("Running burn-in")
p0, _, _ = sampler.run_mcmc(p0, burns)
print("Running production chain")
sampler.run_mcmc(p0, runs);

#Compute the quantiles
burnin = 0
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
samples[:, 0] = np.exp(samples[:, 0])   #le
samples[:, 1] = np.exp(samples[:, 1])   #lp
samples[:, 2] = np.exp(samples[:, 2])   #Period
samples[:, 3] = np.exp(samples[:, 3])   #wn
samples[:, 4] = np.exp(samples[:, 4])   #vb
samples[:, 5] = np.exp(samples[:, 5])   #vr
samples[:, 6] = np.exp(samples[:, 6])   #lc
samples[:, 7] = np.exp(samples[:, 7])   #bc
samples[:, 8] = np.exp(samples[:, 8])   #br
#save data
pickle.dump(sampler.chain[:, :, 0],open("lp_1spot.p", 'wb'),protocol=-1)
pickle.dump(sampler.chain[:, :, 1],open("le_1spot.p", 'wb'),protocol=-1)
pickle.dump(sampler.chain[:, :, 2],open("P_1spot.p", 'wb'),protocol=-1)
pickle.dump(sampler.chain[:, :, 3],open("wn_1spot.p", 'wb'),protocol=-1)
pickle.dump(sampler.chain[:, :, 4],open("vc_1spot.p", 'wb'),protocol=-1)
pickle.dump(sampler.chain[:, :, 5],open("vr_1spot.p", 'wb'),protocol=-1)
pickle.dump(sampler.chain[:, :, 6],open("lc_1spot.p", 'wb'),protocol=-1)
pickle.dump(sampler.chain[:, :, 7],open("bc_1spot.p", 'wb'),protocol=-1)
pickle.dump(sampler.chain[:, :, 8],open("br_1spot.p", 'wb'),protocol=-1)

ll1, ll2, pp, wnn, vcvc, vrvr, lclc, bcbc,brbr = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],axis=0)))

print()
print('Aperiodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(ll1))
print('Periodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(ll2))
print('Kernel period = {0[0]} +{0[1]} -{0[2]}'.format(pp))
print('Kernel wn = {0[0]} +{0[1]} -{0[2]}'.format(wnn))
print()
print('Vc = {0[0]} +{0[1]} -{0[2]}'.format(vcvc))
print('Vr = {0[0]} +{0[1]} -{0[2]}'.format(vrvr))
print('Lc = {0[0]} +{0[1]} -{0[2]}'.format(lclc))
print('Bc = {0[0]} +{0[1]} -{0[2]}'.format(bcbc))
print('Br = {0[0]} +{0[1]} -{0[2]}'.format(brbr))

print('graphics')
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 9))
axes[0].plot(np.exp(sampler.chain[:, burns:, 0]).T, color="k", alpha=0.4)
axes[0].yaxis.set_major_locator(MaxNLocator(5))
axes[0].set_ylabel("$Aperiodic length scale$")
axes[1].plot(np.exp(sampler.chain[:, burns:, 1]).T, color="k", alpha=0.4)
axes[1].yaxis.set_major_locator(MaxNLocator(5))
axes[1].set_ylabel("$Periodic length scale$")
axes[2].plot(np.exp(sampler.chain[:, burns:, 2]).T, color="k", alpha=0.4)
axes[2].yaxis.set_major_locator(MaxNLocator(5))
axes[2].set_ylabel("$period$")
axes[3].plot(np.exp(sampler.chain[:, burns:, 3]).T, color="k", alpha=0.4)
axes[3].yaxis.set_major_locator(MaxNLocator(5))
axes[3].set_ylabel("$WN$")
axes[3].set_xlabel("step number")
fig.tight_layout(h_pad=0.0)
plt.show()

fig, axes = plt.subplots(5, 1, sharex=True, figsize=(8, 9))
axes[0].plot(np.exp(sampler.chain[:, burns:, 4]).T, color="k", alpha=0.4)
axes[0].yaxis.set_major_locator(MaxNLocator(5))
axes[0].set_ylabel("$Vc$")
axes[1].plot(np.exp(sampler.chain[:, burns:, 5]).T, color="k", alpha=0.4)
axes[1].yaxis.set_major_locator(MaxNLocator(5))
axes[1].set_ylabel("$Vr$")
axes[2].plot(np.exp(sampler.chain[:, burns:, 6]).T, color="k", alpha=0.4)
axes[2].yaxis.set_major_locator(MaxNLocator(5))
axes[2].set_ylabel("$Lc$")
axes[3].plot(np.exp(sampler.chain[:, burns:, 7]).T, color="k", alpha=0.4)
axes[3].yaxis.set_major_locator(MaxNLocator(5))
axes[3].set_ylabel("$Bc$")
axes[4].plot(np.exp(sampler.chain[:, burns:, 8]).T, color="k", alpha=0.4)
axes[4].yaxis.set_major_locator(MaxNLocator(5))
axes[4].set_ylabel("$Br$")
axes[4].set_xlabel("step number")
fig.tight_layout(h_pad=0.0)
plt.show()

#time = np.linspace(0, 75, 500)
#a = [ll1[0], ll2[0], pp[0], vcvc[0], vrvr[0], lclc[0], bcbc[0], brbr[0]]
#mu1, std1 = gpObj.draw_from_gp(time, a,  model = 'rv')
#mu2, std2 = gpObj.draw_from_gp(time, a,  model = 'bis')
#mu3, std3 = gpObj.draw_from_gp(time, a,  model = 'rhk')
#
#f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
#ax1.set_title(' ')
#ax1.fill_between(time, mu1+std1, mu1-std1, color="grey", alpha=0.5)
#ax1.plot(time, mu1, "k-", alpha=1, lw=1.5)
#ax1.plot(t,rv,"b*")
#ax1.set_ylabel("RVs")
#
#ax2.fill_between(time, mu2+std2, mu2-std2, color="grey", alpha=0.5)
#ax2.plot(time, mu2, "k-", alpha=1, lw=1.5)
#ax2.plot(t,bis,"b*")
#ax2.set_ylabel("BIS")
#
#ax3.fill_between(time, mu3+std3, mu3-std3, color="grey", alpha=0.5)
#ax3.plot(time, mu3, "k-", alpha=1, lw=1.5)
#ax3.plot(t,rhk,"b*")
#ax3.set_ylabel("flux")
#ax3.set_xlabel("time")
#plt.show()
