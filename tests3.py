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

### a = [l, vc, vr, lc, bc, br] -> kernel parameters
a = np.array([1, 1, 10, -1, 10, 10])
### b = [P, e, K, w] -> keplerian parameters
b = np.array([100, 0.5, 10, 10])
print(b[1])
### Example for the squared exponential  #######################################
### 1st set pf data - zero mean
t,rv, rvyerr, bis, rhk,sig_rhk = np.loadtxt("HD41248_harps.rdb",skiprows=2,
                                            unpack=True, usecols=(0,1,2,5,9,10))
#t = np.linspace(1, 300, 228)
bis_err=2*rvyerr

gpObj = BIGgp(kernels.SquaredExponential, t=t, rv=rv, rverr=rvyerr,
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

################################################################################
### 2nd set of data - with keplerian
t,rv, rvyerr, bis, rhk,sig_rhk = np.loadtxt("HD41248_harps.rdb",skiprows=2,
                                            unpack=True, usecols=(0,1,2,5,9,10))
#t = np.linspace(1, 300, 228)
bis_err=2*rvyerr

gpObj = BIGgp(kernels.SquaredExponential, t=t, rv=rv, rverr=rvyerr,
                    bis=bis, sig_bis=bis_err, rhk=rhk, sig_rhk=sig_rhk)
matriz = gpObj.compute_matrix(a)
print(isposdef(matriz))

#measurements
y=np.hstack((rv,rhk,bis))
#error in measurements
yerr=np.hstack((rvyerr,sig_rhk,2*rvyerr))
# #log-likelihood
print('2nd try ->', gpObj.kepler_likelihood(a, b, y))
print()

################################################################################
### 3rd set of data - with keplerian and mcmc
# a = [l, vc, vr, lc, bc, br] -> kernel parameters
a = np.array([ np.random.uniform(0, 1),
                  np.random.uniform(0, 100), 
                  np.random.uniform(0, 100), 
                  np.random.uniform(0, 100), 
                  np.random.uniform(0, 100), 
                  np.random.uniform(0, 100) ])
# b = [P, e, K, w] -> keplerian parameters
b = np.array([ np.random.uniform(0, 100),
                  np.random.uniform(0, 1), 
                  np.random.uniform(0, 100), 
                  np.random.uniform(0, 2*np.pi) ])


t,rv, rvyerr, bis, rhk,sig_rhk = np.loadtxt("HD41248_harps.rdb",skiprows=2,
                                            unpack=True, usecols=(0,1,2,5,9,10))
#t = np.linspace(1, 300, 228)
bis_err=2*rvyerr

gpObj = BIGgp(kernels.SquaredExponential, t=t, rv=rv, rverr=rvyerr,
                    bis=bis, sig_bis=bis_err, rhk=rhk, sig_rhk=sig_rhk)
matriz = gpObj.compute_matrix(a)
print(isposdef(matriz))

#measurements
y=np.hstack((rv,rhk,bis))
#error in measurements
yerr=np.hstack((rvyerr,sig_rhk,2*rvyerr))
# #log-likelihood
print('3rd try ->', gpObj.kepler_likelihood(a, b, y))
print()

kernel = True #just in case I don't want things to run
#### simple sample and marginalization with emcee
runs, burns = 100, 100
if kernel:
    #probabilistic model
    def logprob(p):
        if any([p[0] < -10, p[0] > np.log(5),
                p[1] < -10, p[1] > np.log(30),
                p[2] < -10, p[2] > np.log(25),
                p[3] < -10, p[3] > np.log(25),
                p[4] < -10, p[4] > np.log(50),
                p[5] < -10, p[5] > np.log(25),
                p[6] < -10, p[6] > np.log(500),
                p[7] < -10, p[7] > np.log(1),
                p[8] < -10, p[8] > np.log(100),
                p[9] < -10, p[9] > np.log(np.pi) ]):
            return -np.inf
        logprior = 0.0
        return logprior + gpObj.kepler_likelihood(np.exp(p[:-4]), np.exp(p[-4:]), y)

    l_prior = stats.uniform(np.exp(-10), 5 -np.exp(-10))                        #[exp(-10) to 5]
    vc_prior = stats.uniform(np.exp(-10), 30 -np.exp(-10))                      #[exp(-10) to 30]
    vr_prior = stats.uniform(np.exp(-10), 25 -np.exp(-10))                      #[exp(-10) to 25]
    lc_prior = stats.uniform(np.exp(-10), 25 -np.exp(-10))                      #[exp(-10) to 25]
    bc_prior = stats.uniform(np.exp(-10), 50 -np.exp(-10))                      #[exp(-10) to 50]
    br_prior = stats.uniform(np.exp(-10), 25 -np.exp(-10))                      #[exp(-10) to 25]

    P_prior = stats.uniform(np.exp(-10), 500 -np.exp(-10))                      #[exp(-10) to 500]
    e_prior = stats.uniform(np.exp(-10), 1-np.exp(-10))                         #[exp(-10) to 1]
    k_prior = stats.uniform(np.exp(-10), 100 -np.exp(-10))                      #[exp(-10) to 100]
    w_prior = stats.uniform(np.exp(-10), np.pi -np.exp(-10))                    #[exp(-10) to pi]

    def from_prior():
        return np.array([l_prior.rvs(),vc_prior.rvs(),
                         vr_prior.rvs(),lc_prior.rvs(),bc_prior.rvs(),br_prior.rvs(),
                         P_prior.rvs(),e_prior.rvs(),k_prior.rvs(),w_prior.rvs()])

    #Set up the sampler.
    nwalkers, ndim = 2*10, 10
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob)
    #Initialize the walkers.
    p0=[np.log(from_prior()) for i in range(nwalkers)]
#    assert not np.isinf(map(logprob, p0)).any()
#    assert not np.isnan(map(logprob, p0)).any()

    print("Running burn-in")
    p0, _, _ = sampler.run_mcmc(p0, burns)
    print("Running production chain")
    sampler.run_mcmc(p0, runs);

    #Compute the quantiles
    burnin = burns
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    samples[:, 0] = np.exp(samples[:, 0])   #Kernel length scale
    samples[:, 1] = np.exp(samples[:, 1])   #Vc
    samples[:, 2] = np.exp(samples[:, 2])   #Vr
    samples[:, 3] = np.exp(samples[:, 3])   #Lc
    samples[:, 4] = np.exp(samples[:, 4])   #Bc
    samples[:, 5] = np.exp(samples[:, 5])   #Br
    samples[:, 6] = np.exp(samples[:, 6])   #P
    samples[:, 7] = np.exp(samples[:, 7])   #e
    samples[:, 8] = np.exp(samples[:, 8])   #k
    samples[:, 9] = np.exp(samples[:, 9])   #w
    ll, vcvc, vrvr, lclc, bcbc,brbr, pp, ee, kk, ww = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                 zip(*np.percentile(samples, [16, 50, 84],axis=0)))

    print('kernel length scale = {0[0]} +{0[1]} -{0[2]}'.format(ll))
    print('Vc = {0[0]} +{0[1]} -{0[2]}'.format(vcvc))
    print('Vr = {0[0]} +{0[1]} -{0[2]}'.format(vrvr))
    print('Lc = {0[0]} +{0[1]} -{0[2]}'.format(lclc))
    print('Bc = {0[0]} +{0[1]} -{0[2]}'.format(bcbc))
    print('Br = {0[0]} +{0[1]} -{0[2]}'.format(brbr))
    print() 
    print('P = {0[0]} +{0[1]} -{0[2]}'.format(pp))
    print('e = {0[0]} +{0[1]} -{0[2]}'.format(ee))
    print('K = {0[0]} +{0[1]} -{0[2]}'.format(kk))
    print('w = {0[0]} +{0[1]} -{0[2]}'.format(ww))


sys.exit(0)
### END