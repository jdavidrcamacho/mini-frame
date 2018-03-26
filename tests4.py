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

### MCMC using the quasi periodic kernel

# a = [lp,le,p, vc,vr, lc, bc,br] -> kernel parameters
a = np.array([ np.random.uniform(np.exp(-100), 100),
                  np.random.uniform(np.exp(-100), 100), 
                  np.random.uniform(np.exp(-100), 50), 
                  np.random.uniform(np.exp(-100), 100), 
                  np.random.uniform(np.exp(-100), 100), 
                  np.random.uniform(np.exp(-100), 100), 
                  np.random.uniform(np.exp(-100), 100), 
                  np.random.uniform(np.exp(-100), 100) ])

# b = [P, e, K, w] -> keplerian parameters
b = np.array([ np.random.uniform(0, 100),
                  np.random.uniform(0, 1), 
                  np.random.uniform(0, 100), 
                  np.random.uniform(0, 2*np.pi) ])


t,rv, rvyerr, bis, rhk,sig_rhk = np.loadtxt("HD41248_harps.rdb",skiprows=2,
                                            unpack=True, usecols=(0,1,2,5,9,10))
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
print('3rd try ->', gpObj.kepler_likelihood(a, b, y))
print()

kernel = True #just in case I don't want things to run
#### simple sample and marginalization with emcee
runs, burns = 500, 500
if kernel:
    #probabilistic model
    def logprob(p):
        if any([p[0] < -10, p[0] > 10, 
                p[1] < -10, p[1] > 10,
                p[2] < -10, p[2] > np.log(100),
                p[3] < -100, p[3] > np.log(30),
                p[4] < -100, p[4] > np.log(25),
                p[5] < -100, p[5] > np.log(25),
                p[6] < -100, p[6] > np.log(50),
                p[7] < -100, p[7] > np.log(25),
                p[8] < -10, p[8] > np.log(500),
                p[9] < -10, p[9] > np.log(1),
                p[10] < -10, p[10] > np.log(100),
                p[11] < -10, p[11] > np.log(np.pi) ]):
            return -np.inf
        logprior = 0.0
        return logprior + gpObj.kepler_likelihood(np.exp(p[:-4]), np.exp(p[-4:]), y)

    lp_prior = stats.uniform(np.exp(-10), np.exp(10) -np.exp(-10))          #[exp(-10) to exp(10)]
    le_prior = stats.uniform(np.exp(-10), np.exp(10) -np.exp(-10))          #[exp(-10) to exp(10)]
    p_prior = stats.uniform(np.exp(-10), 100 -np.exp(-10))                  #[exp(-10) to 100]
    vc_prior = stats.uniform(np.exp(-100), 30 -np.exp(-100))                    #[exp(-100) to 30]
    vr_prior = stats.uniform(np.exp(-100), 25 -np.exp(-100))                    #[exp(-100) to 25]
    lc_prior = stats.uniform(np.exp(-100), 25 -np.exp(-100))                    #[exp(-100) to 25]
    bc_prior = stats.uniform(np.exp(-100), 50 -np.exp(-100))                    #[exp(-100) to 50]
    br_prior = stats.uniform(np.exp(-100), 25 -np.exp(-100))                     #[exp(-100) to 25]
    

    P_prior = stats.uniform(np.exp(-10), 500 -np.exp(-10))                      #[exp(-10) to 500]
    e_prior = stats.uniform(np.exp(-10), 1-np.exp(-10))                         #[exp(-10) to 1]
    k_prior = stats.uniform(np.exp(-10), 100 -np.exp(-10))                      #[exp(-10) to 100]
    w_prior = stats.uniform(np.exp(-10), np.pi -np.exp(-10))                    #[exp(-10) to pi]

    def from_prior():
        return np.array([lp_prior.rvs(), le_prior.rvs(), p_prior.rvs(),vc_prior.rvs(),
                         vr_prior.rvs(),lc_prior.rvs(),bc_prior.rvs(),br_prior.rvs(),
                         P_prior.rvs(),e_prior.rvs(),k_prior.rvs(),w_prior.rvs()])

    #Set up the sampler.
    nwalkers, ndim = 2*12, 12
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
    samples[:, 0] = np.exp(samples[:, 0])   #lp
    samples[:, 1] = np.exp(samples[:, 1])   #le
    samples[:, 2] = np.exp(samples[:, 2])   #Period
    samples[:, 3] = np.exp(samples[:, 3])   #Vc
    samples[:, 4] = np.exp(samples[:, 4])   #Vr
    samples[:, 5] = np.exp(samples[:, 5])   #Lc
    samples[:, 6] = np.exp(samples[:, 6])   #Bc
    samples[:, 7] = np.exp(samples[:, 7])   #Br
    samples[:, 8] = np.exp(samples[:, 8])   #P
    samples[:, 9] = np.exp(samples[:, 9])   #e
    samples[:, 10] = np.exp(samples[:, 10])   #k
    samples[:, 11] = np.exp(samples[:, 11])   #w
    llpp,llee,pp, vcvc, vrvr, lclc, bcbc,brbr, PP, ee, kk, ww = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                 zip(*np.percentile(samples, [16, 50, 84],axis=0)))

    print('periodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(llpp))
    print('aperiodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(llee))
    print('kernel period = {0[0]} +{0[1]} -{0[2]}'.format(pp))
    print('Vc = {0[0]} +{0[1]} -{0[2]}'.format(vcvc))
    print('Vr = {0[0]} +{0[1]} -{0[2]}'.format(vrvr))
    print('Lc = {0[0]} +{0[1]} -{0[2]}'.format(lclc))
    print('Bc = {0[0]} +{0[1]} -{0[2]}'.format(bcbc))
    print('Br = {0[0]} +{0[1]} -{0[2]}'.format(brbr))
    print() 
    print('P = {0[0]} +{0[1]} -{0[2]}'.format(PP))
    print('e = {0[0]} +{0[1]} -{0[2]}'.format(ee))
    print('K = {0[0]} +{0[1]} -{0[2]}'.format(kk))
    print('w = {0[0]} +{0[1]} -{0[2]}'.format(ww))


sys.exit(0)
### END