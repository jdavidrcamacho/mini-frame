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
a = np.array([ np.random.uniform(np.exp(-10), 2),
                  np.random.uniform(20, 60), 
                  np.random.uniform(15, 35), 
                  np.random.uniform(np.exp(-10), 30), 
                  np.random.uniform(np.exp(-10), 30), 
                  np.random.uniform(np.exp(-10), 30), 
                  np.random.uniform(np.exp(-10), 30), 
                  np.random.uniform(np.exp(-10), 30) ])
# B = [ ... ] -> mean funtions parameters
b = [10,15,0.5,80,0, 2, 2]


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
    gpObj = BIGgp(kernels.QuasiPeriodic,[None,None, None] , t=t,
                  rv=rv, rverr=rvyerr, bis=bis, sig_bis=bis_err, rhk=rhk, sig_rhk=sig_rhk)
    print('Likelihood with no mean function =', gpObj.log_likelihood(a, [], y))
    print()
    
    #GP object
    gpObj = BIGgp(kernels.QuasiPeriodic, [Keplerian, Constant, Constant], t=t,
                  rv=rv, rverr=rvyerr, bis=bis, sig_bis=bis_err, rhk=rhk, sig_rhk=sig_rhk)
    print('Likelihood with mean functions =', gpObj.log_likelihood(a, b, y))
    print()



#To check the created matrices
if MATRICES:
    main_matrix = gpObj.compute_matrix(a)
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
    runs, burns = 500, 500
    #probabilistic model
    def logprob(p):
        if any([p[0] < -10, p[0] > np.log(2), 
                p[1] < np.log(20), p[1] > np.log(60),
                p[2] < np.log(15), p[2] > np.log(35), 
                p[3] < -10, p[3] > np.log(30),
                p[4] < -10, p[4] > np.log(30),
                p[5] < -10, p[5] > np.log(30),
                p[6] < -10, p[6] > np.log(30),
                p[7] < -10, p[7] > np.log(30),
                p[8] < -10, p[8] > np.log(50),
                p[9] < -10, p[9] > np.log(100),
                p[10] < -10, p[10] > np.log(1),
                p[11] < -10, p[11] > np.log(2*np.pi),
                p[12] < -10 , p[13] > np.log(10000),
                p[13] < -10 , p[13] > np.log(10000),
                p[14] < -10 , p[14] > np.log(10000)]):
            return -np.inf
        logprior = 0.0
        return logprior + gpObj.log_likelihood(np.exp(p[:-7]), np.exp(p[-7:]), y)


    prior = stats.uniform(np.exp(-10), np.exp(10) -np.exp(-10))     #prior from exp(-10) to exp(10)

    lp_prior = stats.uniform(np.exp(-10), 2 -np.exp(-10)) #[exp(-10) to 2]
    le_prior = stats.uniform(20, 60 - 20) #[exp(-10) to exp(10)]
    p_prior = stats.uniform(15, 35 - 15) #[15 to 35]

    vc_prior = stats.uniform(np.exp(-10), 30 -np.exp(-100)) #[exp(-10) to 30]
    vr_prior = stats.uniform(np.exp(-10), 30 -np.exp(-100)) #[exp(-10) to 30]
    lc_prior = stats.uniform(np.exp(-10), 30 -np.exp(-100)) #[exp(-10) to 30]
    bc_prior = stats.uniform(np.exp(-10), 30 -np.exp(-100)) #[exp(-10) to 30]
    br_prior = stats.uniform(np.exp(-10), 30 -np.exp(-100)) #[exp(-10) to 30]

    P_prior = stats.uniform(np.exp(-10), 50 -np.exp(-10)) #[exp(-10) to 50]
    k_prior = stats.uniform(np.exp(-10), 100 -np.exp(-10)) #[exp(-10) to 100]
    e_prior = stats.uniform(np.exp(-10), 1-np.exp(-10)) #[exp(-10) to 1]
    w_prior = stats.uniform(np.exp(-10), 2*np.pi -np.exp(-10)) #[exp(-10) to 2*pi]
    t0_prior = stats.uniform(np.exp(-10), 10000 -np.exp(-10)) #[1000 to 10000]

    const_prior = stats.uniform(np.exp(-10), 10000 -np.exp(-10)) #[1000 to 9000]
    const1_prior = stats.uniform(np.exp(-10), 10000 -np.exp(-10)) #[1000 to 9000]


    def from_prior():
        #[lp,le,p, vc,vr,lc,bc,br, P,k,e,w,t0, const,const]
        return np.array([ lp_prior.rvs(), le_prior.rvs(), p_prior.rvs(),
                         vc_prior.rvs(), vr_prior.rvs(), lc_prior.rvs(), bc_prior.rvs(), br_prior.rvs(),
                         P_prior.rvs(), k_prior.rvs(), e_prior.rvs(), w_prior.rvs(), t0_prior.rvs(), 
                         const_prior.rvs(), const1_prior.rvs() ])

    #Set up the sampler.
    nwalkers, ndim = 2*15, 15
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
    samples[:, 8] = np.exp(samples[:, 8])   #P
    samples[:, 9] = np.exp(samples[:, 9])   #Krv
    samples[:, 10] = np.exp(samples[:, 10])   #e
    samples[:, 11] = np.exp(samples[:, 11])   #w
    samples[:, 12] = np.exp(samples[:, 12])   #t0
    samples[:, 13] = np.exp(samples[:, 13])   #constant
    samples[:, 14] = np.exp(samples[:, 14])   #constant

    ll1, ll2, pp,vcvc, vrvr, lclc, bcbc,brbr, PP, kk, ee, ww, tt00, const, l1 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                 zip(*np.percentile(samples, [16, 50, 84],axis=0)))

    print('periodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(ll1))
    print('aperiodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(ll2))
    print('kernel period = {0[0]} +{0[1]} -{0[2]}'.format(pp))
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
    print('constant = {0[0]} +{0[1]} -{0[2]}'.format(l1))

    print('graphics')
    fig, axes = pl.subplots(3, 1, sharex=True, figsize=(8, 9))
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
    pl.show()

    fig, axes = pl.subplots(5, 1, sharex=True, figsize=(8, 9))
    axes[0].plot(np.exp(sampler.chain[:, burns:, 3]).T, color="k", alpha=0.4)
    axes[0].yaxis.set_major_locator(MaxNLocator(5))
    axes[0].set_ylabel("$Vc$")
    axes[1].plot(np.exp(sampler.chain[:, burns:, 4]).T, color="k", alpha=0.4)
    axes[1].yaxis.set_major_locator(MaxNLocator(5))
    axes[1].set_ylabel("$Vr$")
    axes[2].plot(np.exp(sampler.chain[:, burns:, 5]).T, color="k", alpha=0.4)
    axes[2].yaxis.set_major_locator(MaxNLocator(5))
    axes[2].set_ylabel("$Lc$")
    axes[3].plot(np.exp(sampler.chain[:, burns:, 6]).T, color="k", alpha=0.4)
    axes[3].yaxis.set_major_locator(MaxNLocator(5))
    axes[3].set_ylabel("$Bc$")
    axes[4].plot(np.exp(sampler.chain[:, burns:, 7]).T, color="k", alpha=0.4)
    axes[4].yaxis.set_major_locator(MaxNLocator(5))
    axes[4].set_ylabel("$Br$")
    axes[4].set_xlabel("step number")
    fig.tight_layout(h_pad=0.0)
    pl.show()

    fig, axes = pl.subplots(7, 1, sharex=True, figsize=(8, 9))
    axes[0].plot(np.exp(sampler.chain[:, burns:, 8]).T, color="k", alpha=0.4)
    axes[0].yaxis.set_major_locator(MaxNLocator(5))
    axes[0].set_ylabel("$P$")
    axes[1].plot(np.exp(sampler.chain[:, burns:, 9]).T, color="k", alpha=0.4)
    axes[1].yaxis.set_major_locator(MaxNLocator(5))
    axes[1].set_ylabel("$K$")
    axes[2].plot(np.exp(sampler.chain[:, burns:, 10]).T, color="k", alpha=0.4)
    axes[2].yaxis.set_major_locator(MaxNLocator(5))
    axes[2].set_ylabel("$e$")
    axes[3].plot(np.exp(sampler.chain[:, burns:, 11]).T, color="k", alpha=0.4)
    axes[3].yaxis.set_major_locator(MaxNLocator(5))
    axes[3].set_ylabel("$w$")
    axes[4].plot(np.exp(sampler.chain[:, burns:, 12]).T, color="k", alpha=0.4)
    axes[4].yaxis.set_major_locator(MaxNLocator(5))
    axes[4].set_ylabel("$T0$")
    axes[5].plot(np.exp(sampler.chain[:, burns:, 13]).T, color="k", alpha=0.4)
    axes[5].yaxis.set_major_locator(MaxNLocator(5))
    axes[5].set_ylabel("$const 1$")
    axes[6].plot(np.exp(sampler.chain[:, burns:, 14]).T, color="k", alpha=0.4)
    axes[6].yaxis.set_major_locator(MaxNLocator(5))
    axes[6].set_ylabel("$const 2$")
    axes[6].set_xlabel("step number")
    fig.tight_layout(h_pad=0.0)
    pl.show()