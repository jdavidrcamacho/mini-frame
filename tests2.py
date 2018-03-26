# -*- coding: utf-8 -*-
from miniframe import kernels
from miniframe.BIGgp import BIGgp
from miniframe.BIGgp import isposdef
import sys
import numpy as np
import emcee

import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
from scipy import stats
from scipy.stats import multivariate_normal
from scipy.optimize import minimize

### a = [lp,le,p, vc,vr, lc, bc,br] -> kernel parameters
b = np.exp(1)
c = np.exp(10)
#a = np.array([b,b,b, 1,0, 0, 0,0])
d = np.array([ np.random.uniform(np.exp(-100), 100),
                  np.random.uniform(np.exp(-100), 100), 
                  np.random.uniform(np.exp(-100), 50), 
                  np.random.uniform(np.exp(-100), 100), 
                  np.random.uniform(np.exp(-100), 100), 
                  np.random.uniform(np.exp(-100), 100), 
                  np.random.uniform(np.exp(-100), 100), 
                  np.random.uniform(np.exp(-100), 100) ])

#a = np.array([1, 1, 10, -1, 10, 1000])
a = d

### Example for the squared exponential  #######################################
### 1st set pf data - original data
t,rv, rvyerr, bis, rhk,sig_rhk = np.loadtxt("HD41248_harps.rdb",skiprows=2,unpack=True, usecols=(0,1,2,5,9,10))
#t = np.linspace(1, 300, 228)
bis_err=2*rvyerr

gpObj = BIGgp(kernels.QuasiPeriodic, t=t, rv=rv, rverr=rvyerr,
                    bis=bis, sig_bis=bis_err, rhk=rhk, sig_rhk=sig_rhk)
matriz = gpObj.compute_matrix(a)
print(isposdef(matriz))

#measurements
y=np.hstack((rv,rhk,bis))
#error in measurements
yerr=np.hstack((rvyerr,sig_rhk,2*rvyerr))
# #log-likelihood
print('1st try ->', gpObj.log_likelihood(a, y))
print()

k11 = matriz[0:228, 0:228]
k12 = matriz[0:228, 228:456]
k13 = matriz[0:228, 456:684]
k23 = matriz[228:456, 456:684]
k22 = matriz[228:456, 228:456]
k33 = matriz[456:684, 456:684]

k21 = matriz[228:456, 0:228]     #equal to k12.T
k31 = matriz[456:684, 0:228]     #equal to k13.T
k32 = matriz[456:684, 228:456]   #equal to k23.T

kernel = True #just in case I don't want things to run
#### simple sample and marginalization with emcee
runs, burns = 100, 100
if kernel:
    #probabilistic model
    def logprob(p):
        #print np.exp(p)
        if any([p[0] < -10, p[0] > 10, 
                p[1] < -10, p[1] > 10,
                p[2] < -10, p[2] > np.log(100),
                p[3] < -100, p[3] > np.log(30),
                p[4] < -100, p[4] > np.log(25),
                p[5] < -100, p[5] > np.log(25),
                p[6] < -100, p[6] > np.log(50),
                p[7] < -100, p[7] > np.log(25)]):
            return -np.inf
        logprior = 0.0
        return logprior + gpObj.log_likelihood(np.exp(p), y)

    lp_prior = stats.uniform(np.exp(-10), np.exp(10) -np.exp(-10))          #[exp(-10) to exp(10)]
    le_prior = stats.uniform(np.exp(-10), np.exp(10) -np.exp(-10))          #[exp(-10) to exp(10)]
    p_prior = stats.uniform(np.exp(-10), 100 -np.exp(-10))                  #[exp(-10) to 100]
    vc_prior = stats.uniform(np.exp(-100), 30 -np.exp(-100))                    #[exp(-100) to 30]
    vr_prior = stats.uniform(np.exp(-100), 25 -np.exp(-100))                    #[exp(-100) to 25]
    lc_prior = stats.uniform(np.exp(-100), 25 -np.exp(-100))                    #[exp(-100) to 25]
    bc_prior = stats.uniform(np.exp(-100), 50 -np.exp(-100))                    #[exp(-100) to 50]
    br_prior = stats.uniform(np.exp(-100), 25 -np.exp(-100))                     #[exp(-100) to 25]
    
    def from_prior():
        return np.array([ lp_prior.rvs(), le_prior.rvs(), p_prior.rvs(),vc_prior.rvs(),
                         vr_prior.rvs(),lc_prior.rvs(),bc_prior.rvs(),br_prior.rvs() ])

    # Set up the sampler.
    nwalkers, ndim = 2*8, 8
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob)
    # Initialize the walkers.
    #p0 = a + 1e-4 * np.random.randn(nwalkers, ndim)
    p0=[np.log(from_prior()) for i in range(nwalkers)]
    #assert not np.isinf(map(logprob, p0)).any()
    #assert not np.isnan(map(logprob, p0)).any()
    
    print("Running burn-in")
    p0, _, _ = sampler.run_mcmc(p0, burns)
    
    print("Running production chain")
    sampler.run_mcmc(p0, runs);

    #graphs
    print('graphics')
    fig, axes = pl.subplots(8, 1, sharex=True, figsize=(8, 9))
    axes[0].plot(np.exp(sampler.chain[:, burns:, 0]).T, color="k", alpha=0.4)
    axes[0].yaxis.set_major_locator(MaxNLocator(5))
    axes[0].set_ylabel("$lp$")
    axes[1].plot(np.exp(sampler.chain[:, burns:, 1]).T, color="k", alpha=0.4)
    axes[1].yaxis.set_major_locator(MaxNLocator(5))
    axes[1].set_ylabel("$le$")
    axes[2].plot(np.exp(sampler.chain[:, burns:, 2]).T, color="k", alpha=0.4)
    axes[2].yaxis.set_major_locator(MaxNLocator(5))
    axes[2].set_ylabel("$P$")
    axes[3].plot(np.exp(sampler.chain[:, burns:, 3]).T, color="k", alpha=0.4)
    axes[3].yaxis.set_major_locator(MaxNLocator(5))
    axes[3].set_ylabel("$Vc$")
    axes[4].plot(np.exp(sampler.chain[:, burns:, 4]).T, color="k", alpha=0.4)
    axes[4].yaxis.set_major_locator(MaxNLocator(5))
    axes[4].set_ylabel("$Vr$")
    axes[5].plot(np.exp(sampler.chain[:, burns:, 5]).T, color="k", alpha=0.4)
    axes[5].yaxis.set_major_locator(MaxNLocator(5))
    axes[5].set_ylabel("$Lc$")
    axes[6].plot(np.exp(sampler.chain[:, burns:, 6]).T, color="k", alpha=0.4)
    axes[6].yaxis.set_major_locator(MaxNLocator(5))
    axes[6].set_ylabel("$Bc$")
    axes[7].plot(np.exp(sampler.chain[:, burns:, 7]).T, color="k", alpha=0.4)
    axes[7].yaxis.set_major_locator(MaxNLocator(5))
    axes[7].set_ylabel("$Br$")
    axes[7].set_xlabel("step number")
    fig.tight_layout(h_pad=0.0)
    pl.show()

    # Compute the quantiles.
    burnin = burns
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    samples[:, 0] = np.exp(samples[:, 0])   #lp
    samples[:, 1] = np.exp(samples[:, 1])   #le
    samples[:, 2] = np.exp(samples[:, 2])   #Period
    samples[:, 3] = np.exp(samples[:, 3])   #Vc
    samples[:, 4] = np.exp(samples[:, 4])   #Vr
    samples[:, 5] = np.exp(samples[:, 5])   #Lc
    samples[:, 6] = np.exp(samples[:, 6])   #Bc
    samples[:, 7] = np.exp(samples[:, 7])   #Br

    lp_mcmc,le_mcmc,p_mcmc,vc_mcmc,vr_mcmc,lc_mcmc,bc_mcmc,br_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                 zip(*np.percentile(samples, [16, 50, 84],axis=0)))

    print('lp = {0[0]} +{0[1]} -{0[2]}'.format(lp_mcmc))
    print('le = {0[0]} +{0[1]} -{0[2]}'.format(le_mcmc))
    print('P = {0[0]} +{0[1]} -{0[2]}'.format(p_mcmc))
    print()
    print('Vc = {0[0]} +{0[1]} -{0[2]}'.format(vc_mcmc))
    print('Vr = {0[0]} +{0[1]} -{0[2]}'.format(vr_mcmc))
    print('Lc = {0[0]} +{0[1]} -{0[2]}'.format(lc_mcmc))
    print('Bc = {0[0]} +{0[1]} -{0[2]}'.format(bc_mcmc))
    print('Br = {0[0]} +{0[1]} -{0[2]}'.format(br_mcmc))

sys.exit(0)

################################################################################
### 2nd set of data - evenly spaced time
t,rv, rvyerr, bis, rhk,sig_rhk = np.loadtxt("HD41248_harps.rdb",skiprows=2,unpack=True, usecols=(0,1,2,5,9,10))
bis_err=2*rvyerr
t = np.linspace(100, 500, 228)

gpObj = BIGgp(kernels.QuasiPeriodic, t=t, rv=rv, rverr=rvyerr,
                    bis=bis, sig_bis=bis_err, rhk=rhk, sig_rhk=sig_rhk)
matriz = gpObj.compute_matrix(a)
print(isposdef(matriz))

#measurements
y=np.hstack((rv,rhk,bis))
#error in measurements
yerr=np.hstack((rvyerr,sig_rhk,2*rvyerr))
# #log-likelihood
print('2nd try ->', gpObj.log_likelihood(a, y))
print()
################################################################################
### one of rajpaul's dataset from jones et al. 2017

a = np.array([1,1,1, 1, 1, 1, 1, 1])

t, rv, rhk, bis, rvyerr, sig_rhk,bis_err  = np.loadtxt("rajpaul_hd_fine_datasets.csv",delimiter=',',skiprows=1,unpack=True)
gpObj = BIGgp(kernels.QuasiPeriodic, t=t, rv=rv, rverr=rvyerr,
                    bis=bis, sig_bis=bis_err, rhk=rhk, sig_rhk=sig_rhk)
matriz = gpObj.compute_matrix(a)
print(isposdef(matriz))

#measurements
y=np.hstack((rv,rhk,bis))
#error in measurements
yerr=np.hstack((rvyerr,sig_rhk,2*rvyerr))
# #log-likelihood
print('3rd try ->', gpObj.log_likelihood(a, y))
print()
################################################################################
### if this works -.-
t,rv, rvyerr, bis, rhk,sig_rhk = np.loadtxt("HD41248_harps.rdb",skiprows=2,unpack=True, usecols=(0,1,2,5,9,10))

whynot = np.random.choice(228, 228, replace=False)
t1 = t[whynot]
bis1= bis[whynot]
rv1 = rv[whynot]
rhk1 = rhk[whynot]
rvyerr1 = rvyerr[whynot]
sig_rhk1 = sig_rhk[whynot]
bis_err1 = 2*rvyerr1

gpObj = BIGgp(kernels.QuasiPeriodic, t=t1, rv=rv1, rverr=rvyerr1,
                    bis=bis1, sig_bis=bis_err1, rhk=rhk1, sig_rhk=sig_rhk1)
matriz = gpObj.compute_matrix(a)
print(isposdef(matriz))

#measurements
y=np.hstack((rv1,rhk1,bis1))
#error in measurements
yerr=np.hstack((rvyerr1,sig_rhk1,bis_err1))
# #log-likelihood
print('1st try again ->', gpObj.log_likelihood(a, y))
print()
