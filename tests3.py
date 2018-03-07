# -*- coding: utf-8 -*-
from log_ll import likelihood, inv_ll
from cov_matrix import build_bigmatrix

import numpy as np
import matplotlib.pylab as pl
import emcee

from matplotlib.ticker import MaxNLocator
from scipy import stats
from scipy.stats import multivariate_normal
from scipy.optimize import minimize

from time import time
start = time()
#defining the kernel; DO NOT TRY FOR THE QUASI PERIODIC, STILL WORKING ON IT
kernel = 2 #1 = squared exponential, 2 = quasi periodic, else = invalid kernel

#a = np.array([l,vc,vr,lc,bc,br]) <- THIS IS FOR THE SQUARED EXPONENTIAL
#a = np.array([0.1, 10, 0, 0, 0, 0])

#a = np.array([lp,le,p,vc,vr,lc,bc,br]) <- THIS IS FOR THE QUASI PERIODIC
#a = np.array([0.2, 10000, 100, 10, 10, 10, 10, 10])


### Example for the squared exponential  ###
#data
t,rv,rvyerr, bis, rhk, sig_rhk = np.loadtxt("dados.rdb",skiprows=2,unpack=True, usecols=(0,1,2,5,9,10))
t=t-t[0]

#time
#t = np.linspace(100,160,160)
#measurements
y=np.hstack((rv,rhk,bis))
#error in measurements
yerr=np.hstack((rvyerr,sig_rhk,np.random.uniform(2,2.5)*rvyerr))

##plot of covariance matrix
#k = build_bigmatrix(kernel,a,t,y,yerr)
#pl.figure()
#pl.imshow(k)
#pl.show()

##log-likelihood
#print('log marginal likelihood = ', likelihood(kernel,a,t,y,yerr))

##random draws of the GP
#yf=multivariate_normal(np.zeros(y.size), k).rvs()
#pl.figure()
#pl.plot(t,yf[0:160],'-')
#pl.show()

#pl.figure()
#pl.plot(np.tile(t,3),multivariate_normal(np.zeros(y.size), k).rvs(),'.')
#pl.show()

#### minimization
##res = minimize(inv_ll, x0=a, args=(t,y,yerr,kernel))
##print(res)
##print

#### simple sample and marginalization with emcee
runs, burns = 25000, 25000
if kernel == 2:
    #probabilistic model
    def logprob(p):
        #print np.exp(p)
        if any([p[0] < -10, p[0] > 1, 
                p[1] < -10, p[1] > 10,
                p[2] < -10, p[2] > np.log(100),
                p[3] < -10, p[3] > np.log(100),
                p[4] < -10, p[4] > 10,
                p[5] < -10, p[5] > 10,
                p[6] < -10, p[6] > np.log(20),
                p[7] < -10, p[7] > np.log(10)]):
            return -np.inf
        logprior = 0.0
        return logprior + likelihood(kernel,p,t,y,yerr)
        
    lp_prior = stats.uniform(np.exp(-10), np.exp(1) -np.exp(-10))          #[exp(-10) to exp(1)]
    le_prior = stats.uniform(np.exp(-10), np.exp(10) -np.exp(-10))          #[exp(-10) to exp(10)]
    p_prior = stats.uniform(np.exp(-10), 100 -np.exp(-10))                  #[exp(-10) to 100]
    vc_prior = stats.uniform(np.exp(-10), 100 -np.exp(-10))                 #[exp(-10) to 100]

    vr_prior = stats.uniform(np.exp(-10), np.exp(10) -np.exp(-10))         #[exp(-10) to 10]
    lc_prior = stats.uniform(np.exp(-10), np.exp(10)  -np.exp(-10))          #[exp(-10) to exp(10)]
    bc_prior = stats.uniform(np.exp(-10), 20 -np.exp(-10))          #[exp(-10) to 20]
    br_prior = stats.uniform(np.exp(-10), 10  -np.exp(-10))         #[exp(-10) to 10]
    def from_prior():           
        return np.array([ lp_prior.rvs(), le_prior.rvs(), p_prior.rvs(),vc_prior.rvs(),
                         vr_prior.rvs(),lc_prior.rvs(),bc_prior.rvs(),br_prior.rvs() ])
    
    # Set up the sampler.
    nwalkers, ndim = 2*8, 8
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob)
    
    # Initialize the walkers.
    #p0 = a + 1e-4 * np.random.randn(nwalkers, ndim)
    p0=[np.log(from_prior()) for i in range(nwalkers)]           
    assert not np.isinf(map(logprob, p0)).any()
    assert not np.isnan(map(logprob, p0)).any()
    
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
    axes[2].set_ylabel("$period$")
    axes[3].plot(np.exp(sampler.chain[:, burns:, 3]).T, color="k", alpha=0.4) #log
    axes[3].yaxis.set_major_locator(MaxNLocator(5))
    axes[3].set_ylabel("$Vc$")
    axes[4].plot(np.exp(sampler.chain[:, burns:, 4]).T, color="k", alpha=0.4) #log
    axes[4].yaxis.set_major_locator(MaxNLocator(5))
    axes[4].set_ylabel("$Vr$")
    axes[5].plot(np.exp(sampler.chain[:, burns:, 5]).T, color="k", alpha=0.4) #log 
    axes[5].yaxis.set_major_locator(MaxNLocator(5))
    axes[5].set_ylabel("$Lc$")
    axes[6].plot(np.exp(sampler.chain[:, burns:, 6]).T, color="k", alpha=0.4) #log
    axes[6].yaxis.set_major_locator(MaxNLocator(5))
    axes[6].set_ylabel("$Bc$")
    axes[7].plot(np.exp(sampler.chain[:, burns:, 7]).T, color="k", alpha=0.4) #log
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
    
    print 'lp = {0[0]} +{0[1]} -{0[2]}'.format(lp_mcmc)
    print 'le = {0[0]} +{0[1]} -{0[2]}'.format(le_mcmc)
    print 'P = {0[0]} +{0[1]} -{0[2]}'.format(p_mcmc)
    print
    print 'Vc = {0[0]} +{0[1]} -{0[2]}'.format(vc_mcmc)
    print 'Vr = {0[0]} +{0[1]} -{0[2]}'.format(vr_mcmc)
    print 'Lc = {0[0]} +{0[1]} -{0[2]}'.format(lc_mcmc)
    print 'Bc = {0[0]} +{0[1]} -{0[2]}'.format(bc_mcmc)
    print 'Br = {0[0]} +{0[1]} -{0[2]}'.format(br_mcmc)

end = time()

print 'running time = ', end-start, 'seconds'