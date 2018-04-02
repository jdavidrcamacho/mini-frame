# -*- coding: utf-8 -*-
from miniframe import kernels
from miniframe.BIGgp import BIGgp
from miniframe.BIGgp import isposdef
from miniframe.BIGgp import scale
from miniframe.means import Constant, Linear, Keplerian

import sys
import numpy as np
import emcee

import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
from scipy import stats
from scipy.stats import multivariate_normal
from scipy.optimize import minimize

#setting things to True or False (Run or not)
LIKELIHOOD = True
MATRICES = True
MCMC_RUN = True

### Example for the squared exponential  #######################################
# a = [l, vc, vr, lc, bc, br] -> kernel parameters
a = np.array([ np.random.uniform(np.exp(-10), 1),
                  np.random.uniform(np.exp(-10), 100), 
                  np.random.uniform(np.exp(-10), 100), 
                  np.random.uniform(np.exp(-10), 100), 
                  np.random.uniform(np.exp(-10), 100), 
                  np.random.uniform(np.exp(-10), 100) ])
# B = [ ... ] -> mean funtions parameters
b = [10,15,0.5,80,0, 2, 1,2]


#data
t,rv, rvyerr, bis, rhk,sig_rhk = np.loadtxt("miniframe/datasets/HD41248_harps.rdb",skiprows=2,unpack=True, usecols=(0,1,2,5,9,10))
bis_err = 2*rvyerr

rv, rvyerr = scale(rv, rvyerr)
rhk, sig_rhk = scale(rhk,sig_rhk)
bis, bis_err = scale(bis,bis_err)

y = np.hstack((rv,rhk,bis))
yerr = np.hstack((rvyerr,sig_rhk,bis_err))


if LIKELIHOOD:
    #GP object
    gpObj = BIGgp(kernels.SquaredExponential,[None,None, None] , t=t,
                  rv=rv, rverr=rvyerr, bis=bis, sig_bis=bis_err, rhk=rhk, sig_rhk=sig_rhk)
    print('Likelihood with no mean function =', gpObj.log_likelihood(a, [], y))
    print()
    
    #GP object
    gpObj = BIGgp(kernels.SquaredExponential, [Keplerian, Constant, Linear], t=t,
                  rv=rv, rverr=rvyerr, bis=bis, sig_bis=bis_err, rhk=rhk, sig_rhk=sig_rhk)
    print('Likelihood with mean functions =', gpObj.log_likelihood(a, b, y))
    print()



#To check the created matrices
if MATRICES:
    main_matrix  = gpObj.compute_matrix(a)
    k11 = main_matrix[0:228, 0:228]
    k12 = main_matrix[0:228, 228:456]
    k13 = main_matrix[0:228, 456:684]
    k23 = main_matrix[228:456, 456:684]
    k22 = main_matrix[228:456, 228:456]
    k33 = main_matrix[456:684, 456:684]
    
    k21 = main_matrix[228:456, 0:228]     #equal to k12.T
    k31 = main_matrix[456:684, 0:228]     #equal to k13.T
    k32 = main_matrix[456:684, 228:456]   #equal to k23.T


#To run a mcmc
if MCMC_RUN:
    #### simple sample and marginalization with emcee
    runs, burns = 100, 100
    #probabilistic model
    def logprob(p):
        if any([p[0] < -10, p[0] > np.log(2), 
                p[1] < -10, p[1] > np.log(30),
                p[2] < -10, p[2] > np.log(30),
                p[3] < -10, p[3] > np.log(30),
                p[7] < -10, p[4] > np.log(30),
                p[5] < -10, p[5] > np.log(30),
                p[6] < -10, p[6] > np.log(50),
                p[7] < -10, p[7] > np.log(100),
                p[8] < -10, p[8] > np.log(1),
                p[9] < -10, p[9] > np.log(2*np.pi),
                p[10] < -10, p[10] > 10,
                p[11] < -10, p[11] > 10,
                p[12] < -10, p[12] > 10,
                p[13] < -10, p[13] > 10]):
            return -np.inf
        logprior = 0.0
        return logprior + gpObj.log_likelihood(np.exp(p[:-8]), np.exp(p[-8:]), y)


    prior = stats.uniform(np.exp(-10), np.exp(10) -np.exp(-10))     #prior from exp(-10) to exp(10)

    l_prior = stats.uniform(np.exp(-10), 2 -np.exp(-10))  #[exp(-10) to 2]
    vc_prior = stats.uniform(np.exp(-10), 30 -np.exp(-100)) #[exp(-10) to 30]
    vr_prior = stats.uniform(np.exp(-10), 30 -np.exp(-100)) #[exp(-10) to 30]
    lc_prior = stats.uniform(np.exp(-10), 30 -np.exp(-100)) #[exp(-10) to 30]
    bc_prior = stats.uniform(np.exp(-10), 30 -np.exp(-100)) #[exp(-10) to 30]
    br_prior = stats.uniform(np.exp(-10), 30 -np.exp(-100)) #[exp(-10) to 30]
    P_prior = stats.uniform(np.exp(-10), 50 -np.exp(-10)) #[exp(-10) to 50]
    k_prior = stats.uniform(np.exp(-10), 100 -np.exp(-10)) #[exp(-10) to 100]
    e_prior = stats.uniform(np.exp(-10), 1-np.exp(-10)) #[exp(-10) to 1]
    w_prior = stats.uniform(np.exp(-10), 2*np.pi -np.exp(-10)) #[exp(-10) to 2*pi]
    t0_prior = stats.uniform(np.exp(-10), np.exp(10) -np.exp(-10)) #prior
    const_prior = stats.uniform(np.exp(-10), np.exp(10) -np.exp(-10)) #prior
    slope_prior = stats.uniform(np.exp(-10), np.exp(10) -np.exp(-10)) #prior
    inter_prior = stats.uniform(np.exp(-10), np.exp(10) -np.exp(-10)) #prior

    def from_prior():
        #[l, vc,vr,lc,bc,br, p,k,e,w,t0, const, slope,intersect]
        return np.array([ l_prior.rvs(),
                         vc_prior.rvs(), vr_prior.rvs(), lc_prior.rvs(), bc_prior.rvs(), br_prior.rvs(),
                         P_prior.rvs(), k_prior.rvs(), e_prior.rvs(), w_prior.rvs(), t0_prior.rvs(), 
                         const_prior.rvs(), slope_prior.rvs(), inter_prior.rvs() ])

    #Set up the sampler.
    nwalkers, ndim = 2*14, 14
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
    samples[:, 0] = np.exp(samples[:, 0])   #l
    samples[:, 1] = np.exp(samples[:, 1])   #vc
    samples[:, 2] = np.exp(samples[:, 2])   #vr
    samples[:, 3] = np.exp(samples[:, 3])   #lc
    samples[:, 4] = np.exp(samples[:, 4])   #bc
    samples[:, 5] = np.exp(samples[:, 5])   #br
    samples[:, 6] = np.exp(samples[:, 6])   #P
    samples[:, 7] = np.exp(samples[:, 7])   #Krv
    samples[:, 8] = np.exp(samples[:, 8])   #e
    samples[:, 9] = np.exp(samples[:, 9])   #w
    samples[:, 10] = np.exp(samples[:, 10])   #t0
    samples[:, 11] = np.exp(samples[:, 11])   #constant
    samples[:, 12] = np.exp(samples[:, 12])   #Linear slope
    samples[:, 13] = np.exp(samples[:, 13])   #Liner intersect

    ll,vcvc, vrvr, lclc, bcbc,brbr, PP, ee, kk, ww, tt00, const, l1, l2 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                 zip(*np.percentile(samples, [16, 50, 84],axis=0)))

    print('length scale = {0[0]} +{0[1]} -{0[2]}'.format(ll))
    print('Vc = {0[0]} +{0[1]} -{0[2]}'.format(vcvc))
    print('Vr = {0[0]} +{0[1]} -{0[2]}'.format(vrvr))
    print('Lc = {0[0]} +{0[1]} -{0[2]}'.format(lclc))
    print('Bc = {0[0]} +{0[1]} -{0[2]}'.format(bcbc))
    print('Br = {0[0]} +{0[1]} -{0[2]}'.format(brbr))
    print() 
    print('P = {0[0]} +{0[1]} -{0[2]}'.format(PP))
    print('K = {0[0]} +{0[1]} -{0[2]}'.format(kk))
    print('e = {0[0]} +{0[1]} -{0[2]}'.format(ee))
    print('w = {0[0]} +{0[1]} -{0[2]}'.format(ww))
    print('T0 = {0[0]} +{0[1]} -{0[2]}'.format(tt00))
    print()
    print('constant = {0[0]} +{0[1]} -{0[2]}'.format(const))
    print('slope = {0[0]} +{0[1]} -{0[2]}'.format(l1))
    print('intersect = {0[0]} +{0[1]} -{0[2]}'.format(l2))
