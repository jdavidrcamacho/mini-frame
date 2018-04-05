# -*- coding: utf-8 -*-
from miniframe import kernels
from miniframe.BIGgp import BIGgp
from miniframe.BIGgp import isposdef
from miniframe.BIGgp import scale
from miniframe.means import Constant, Linear, Keplerian
from time import time

import numpy as np
import emcee

import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
from scipy import stats

start_time = time()

flux, rv, bis = np.loadtxt("miniframe/datasets/1spot_dataset.rdb",skiprows=2,unpack=True, usecols=(1, 2, 3))
log_rhk = flux**2

rv = np.tile(1000*rv, 3)
bis = np.tile(1000*bis, 3)
rhk = np.tile(log_rhk, 3)
flux = np.tile(flux, 3)
t = np.linspace(0, 75.15, 300)

rms_rv = np.sqrt((1./rv.size*np.sum(rv**2)))
rms_bis = np.sqrt((1./bis.size*np.sum(bis**2)))
rms_rhk = np.sqrt((1./rhk.size*np.sum(rhk**2)))
rvyerr = 0.05*rms_rv * np.ones(300)
bis_err = 0.10*rms_bis * np.ones(300)
sig_rhk = 0.20*rms_rhk * np.ones(300)

rv, rvyerr = scale(rv, rvyerr)
rhk, sig_rhk = scale(rhk,sig_rhk)
bis, bis_err = scale(bis,bis_err)

y = np.hstack((rv,rhk,bis))
yerr = np.hstack((rvyerr,sig_rhk,bis_err))

#no mean funtion as 1st test
gpObj = BIGgp(kernels.QuasiPeriodic,[None,None, None] , t=t,
                  rv=rv, rverr=rvyerr, bis=bis, sig_bis=bis_err, rhk=rhk, sig_rhk=sig_rhk)

#### simple sample and marginalization with emcee
runs, burns = 2500, 2500
#probabilistic model
def logprob(p):
    if any([p[0] < -10, p[0] > np.log(100), 
            p[1] < -10, p[1] > np.log(30),
            p[2] < -10, p[2] > np.log(100), 
            p[3] < -10, p[3] > 10,
            p[4] < -10, p[4] > np.log(2500),
            p[5] < -10, p[5] > 10,
            p[6] < -10, p[6] > np.log(750),
            p[7] < -10, p[7] > 10]):
        return -np.inf
    logprior = 0.0
    return logprior + gpObj.log_likelihood(np.exp(p), [])


#prior from exp(-10) to exp(10)
lp_prior = stats.uniform(np.exp(-10), 100 -np.exp(-10)) #from exp(-10) to 100
le_prior = stats.uniform(np.exp(-10), 30 -np.exp(-10)) #from exp(-10) to 50
p_prior = stats.uniform(np.exp(-10), 100 -np.exp(-10)) #from exp(-10) to 100

vc_prior = stats.uniform(np.exp(-10), np.exp(10) -np.exp(-10))
vr_prior = stats.uniform(np.exp(-10), 2500 -np.exp(-10)) #from exp(-10) to 2500
lc_prior = stats.uniform(np.exp(-10), np.exp(10) -np.exp(-10))
bc_prior = stats.uniform(np.exp(-10), 750 -np.exp(-10)) #from exp(-10) to 750
br_prior = stats.uniform(np.exp(-10), np.exp(10) -np.exp(-10))


def from_prior():
    #[lp,le,p, vc,vr,lc,bc,br, P,k,e,w,t0, const,const]
    return np.array([ lp_prior.rvs(), le_prior.rvs(), p_prior.rvs(),
                     vc_prior.rvs(), vr_prior.rvs(), lc_prior.rvs(), 
                    bc_prior.rvs(), br_prior.rvs()])

#Set up the sampler.
nwalkers, ndim = 2*8, 8
sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob)
#Initialize the walkers.
p0=[np.log(from_prior()) for i in range(nwalkers)]

print("Running burn-in")
p0, _, _ = sampler.run_mcmc(p0, burns)
print("Running production chain")
sampler.run_mcmc(p0, runs);
burns=0

#Compute the quantiles
burnin = burns
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
samples[:, 0] = np.exp(samples[:, 0])   #lp
samples[:, 1] = np.exp(samples[:, 1])   #le
samples[:, 2] = np.exp(samples[:, 2])   #Period
samples[:, 3] = np.exp(samples[:, 3])   #vc
samples[:, 4] = np.exp(samples[:, 4])   #vr
samples[:, 5] = np.exp(samples[:, 5])   #lc
samples[:, 6] = np.exp(samples[:, 6])   #bc
samples[:, 7] = np.exp(samples[:, 7])   #br


ll1, ll2, pp,vcvc, vrvr, lclc, bcbc,brbr = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],axis=0)))

print('periodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(ll1))
print('aperiodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(ll2))
print('kernel period = {0[0]} +{0[1]} -{0[2]}'.format(pp))
print('Vc = {0[0]} +{0[1]} -{0[2]}'.format(vcvc))
print('Vr = {0[0]} +{0[1]} -{0[2]}'.format(vrvr))
print('Lc = {0[0]} +{0[1]} -{0[2]}'.format(lclc))
print('Bc = {0[0]} +{0[1]} -{0[2]}'.format(bcbc))
print('Br = {0[0]} +{0[1]} -{0[2]}'.format(brbr))

#print('graphics')
#fig, axes = pl.subplots(3, 1, sharex=True, figsize=(8, 9))
#axes[0].plot(np.exp(sampler.chain[:, burns:, 0]).T, color="k", alpha=0.4)
#axes[0].yaxis.set_major_locator(MaxNLocator(5))
#axes[0].set_ylabel("$periodic length scale$")
#axes[1].plot(np.exp(sampler.chain[:, burns:, 1]).T, color="k", alpha=0.4)
#axes[1].yaxis.set_major_locator(MaxNLocator(5))
#axes[1].set_ylabel("$aperiodic length scale$")
#axes[2].plot(np.exp(sampler.chain[:, burns:, 2]).T, color="k", alpha=0.4)
#axes[2].yaxis.set_major_locator(MaxNLocator(5))
#axes[2].set_ylabel("$kernel period$")
#axes[2].set_xlabel("step number")
#fig.tight_layout(h_pad=0.0)
#pl.show()
#
#fig, axes = pl.subplots(5, 1, sharex=True, figsize=(8, 9))
#axes[0].plot(np.exp(sampler.chain[:, burns:, 3]).T, color="k", alpha=0.4)
#axes[0].yaxis.set_major_locator(MaxNLocator(5))
#axes[0].set_ylabel("$Vc$")
#axes[1].plot(np.exp(sampler.chain[:, burns:, 4]).T, color="k", alpha=0.4)
#axes[1].yaxis.set_major_locator(MaxNLocator(5))
#axes[1].set_ylabel("$Vr$")
#axes[2].plot(np.exp(sampler.chain[:, burns:, 5]).T, color="k", alpha=0.4)
#axes[2].yaxis.set_major_locator(MaxNLocator(5))
#axes[2].set_ylabel("$Lc$")
#axes[3].plot(np.exp(sampler.chain[:, burns:, 6]).T, color="k", alpha=0.4)
#axes[3].yaxis.set_major_locator(MaxNLocator(5))
#axes[3].set_ylabel("$Bc$")
#axes[4].plot(np.exp(sampler.chain[:, burns:, 7]).T, color="k", alpha=0.4)
#axes[4].yaxis.set_major_locator(MaxNLocator(5))
#axes[4].set_ylabel("$Br$")
#axes[4].set_xlabel("step number")
#fig.tight_layout(h_pad=0.0)
#pl.show()

end_time = time()
print('It took ', end_time-start_time, 'seconds to run', runs+burnin, 'iterations.')