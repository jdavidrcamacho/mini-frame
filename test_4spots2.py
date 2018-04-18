#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from miniframe import kernels
from miniframe.MEDIUMgp import MEDIUMgp
from miniframe.means import Constant, Linear, Keplerian

import numpy as np
import emcee
import sys
from time import time
from scipy import stats

start_time = time()

flux, rv, bis = np.loadtxt("miniframe/datasets/4spots_dataset.rdb",skiprows=2,unpack=True, usecols=(1, 2, 3))
log_rhk = flux**2

rv = np.tile(1000*rv, 3)
bis = np.tile(1000*bis, 3)
rhk = np.tile(log_rhk, 3)
flux = np.tile(flux, 3)
t = np.linspace(0, 75.15, 300)
#pl.plot(t,rv)

rms_rv = np.sqrt((1./rv.size*np.sum(rv**2)))
rms_bis = np.sqrt((1./bis.size*np.sum(bis**2)))
rms_rhk = np.sqrt((1./rhk.size*np.sum(rhk**2)))
rvyerr = 0.05*rms_rv * np.ones(300)
bis_err = 0.10*rms_bis * np.ones(300)
sig_rhk = 0.20*rms_rhk * np.ones(300)

#rv, rvyerr = scale(rv, rvyerr)
#rhk, sig_rhk = scale(rhk,sig_rhk)
#bis, bis_err = scale(bis,bis_err)

y = np.hstack((rv,bis))
yerr = np.hstack((rvyerr,bis_err))

gpObj = MEDIUMgp(kernels.QuasiPeriodic,[None,None] , t=t,
                  rv=rv, rverr=rvyerr, bis=bis, sig_bis=bis_err)

#### simple sample and marginalization with emcee
runs, burns = 10000, 10000
#probabilistic model
def logprob(p):
    if any([p[0] < -10, p[0] > np.log(1), 
            p[1] < -10, p[1] > 10,
            p[2] < np.log(10), p[2] > np.log(50), 
            p[3] < -10, p[3] > np.log(500),
            p[4] < -10, p[4] > np.log(500),
            p[5] < -10, p[5] > np.log(500),
            p[6] < -10, p[6] > np.log(500)]):
        return -np.inf
    logprior = 0.0
    return logprior + gpObj.log_likelihood(np.exp(p), [])


#prior from exp(-10) to exp(10)
lp_prior = stats.uniform(np.exp(-10), 1 -np.exp(-10)) #from exp(-10) to 1
le_prior = stats.uniform(np.exp(-10), np.exp(10) -np.exp(-10)) #from exp(-10) to exp(10)
p_prior = stats.uniform(10, 50-10) #from 15 to 35

vc_prior = stats.uniform(np.exp(-10), 500 -np.exp(-10)) #from exp(-10) to 100
vr_prior = stats.uniform(np.exp(-10), 500 -np.exp(-10)) #from exp(-10) to 100
bc_prior = stats.uniform(np.exp(-10), 500 -np.exp(-10)) #from exp(-10) to 100
br_prior = stats.uniform(np.exp(-10), 500 -np.exp(-10)) #from exp(-10) to 100


def from_prior():
    #[lp,le,p, vc,vr,lc,bc,br, P,k,e,w,t0, const,const]
    return np.array([ lp_prior.rvs(), le_prior.rvs(), p_prior.rvs(),
                     vc_prior.rvs(), vr_prior.rvs(),
                    bc_prior.rvs(), br_prior.rvs()])

#Set up the sampler.
nwalkers, ndim = 2*7, 7
sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob, threads=4)
#Initialize the walkers.
p0=[np.log(from_prior()) for i in range(nwalkers)]

print("Running burn-in")
p0, _, _ = sampler.run_mcmc(p0, burns)
print("Running production chain")
sampler.run_mcmc(p0, runs);

#Compute the quantiles
burnin = burns
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
samples[:, 0] = np.exp(samples[:, 0])   #lp
samples[:, 1] = np.exp(samples[:, 1])   #le
samples[:, 2] = np.exp(samples[:, 2])   #Period
samples[:, 3] = np.exp(samples[:, 3])   #vc
samples[:, 4] = np.exp(samples[:, 4])   #vr
samples[:, 5] = np.exp(samples[:, 5])   #bc
samples[:, 6] = np.exp(samples[:, 6])   #br


ll1, ll2, pp,vcvc, vrvr, bcbc,brbr = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],axis=0)))

print('periodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(ll1))
print('aperiodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(ll2))
print('kernel period = {0[0]} +{0[1]} -{0[2]}'.format(pp))
print('Vc = {0[0]} +{0[1]} -{0[2]}'.format(vcvc))
print('Vr = {0[0]} +{0[1]} -{0[2]}'.format(vrvr))
print('Bc = {0[0]} +{0[1]} -{0[2]}'.format(bcbc))
print('Br = {0[0]} +{0[1]} -{0[2]}'.format(brbr))


end_time = time()
print('It took ', end_time-start_time, 'seconds to run', runs*2, 'iterations.')

sys.exit(0)

import matplotlib.pylab as plt
from matplotlib.ticker import MaxNLocator
print('graphics')
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 9))
axes[0].plot(np.exp(sampler.chain[:, burns:, 0]).T, color="k", alpha=0.4)
axes[0].yaxis.set_major_locator(MaxNLocator(5))
axes[0].set_ylabel("$periodic length scale$")
axes[1].plot(np.exp(sampler.chain[:, burns:, 1]).T, color="k", alpha=0.4)
axes[1].yaxis.set_major_locator(MaxNLocator(5))
axes[1].set_ylabel("$aperiodic length scale$")
axes[2].plot(np.exp(sampler.chain[:, burns:, 2]).T, color="k", alpha=0.4)
axes[2].yaxis.set_major_locator(MaxNLocator(5))
axes[2].set_ylabel("$kernel period$")
axes[2].set_xlabel("step number")
fig.tight_layout(h_pad=0.0)
plt.show()

fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 9))
axes[0].plot(np.exp(sampler.chain[:, burns:, 3]).T, color="k", alpha=0.4)
axes[0].yaxis.set_major_locator(MaxNLocator(5))
axes[0].set_ylabel("$Vc$")
axes[1].plot(np.exp(sampler.chain[:, burns:, 4]).T, color="k", alpha=0.4)
axes[1].yaxis.set_major_locator(MaxNLocator(5))
axes[1].set_ylabel("$Vr$")
axes[2].plot(np.exp(sampler.chain[:, burns:, 5]).T, color="k", alpha=0.4)
axes[2].yaxis.set_major_locator(MaxNLocator(5))
axes[2].set_ylabel("$bc$")
axes[3].plot(np.exp(sampler.chain[:, burns:, 6]).T, color="k", alpha=0.4)
axes[3].yaxis.set_major_locator(MaxNLocator(5))
axes[3].set_ylabel("$Br$")
axes[3].set_xlabel("step number")
fig.tight_layout(h_pad=0.0)
plt.show()
