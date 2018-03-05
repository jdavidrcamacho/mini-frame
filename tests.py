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


#defining the kernel; DO NOT TRY FOR THE QUASI PERIODIC, STILL WORKING ON IT
kernel = 2 #1 = squared exponential, 2 = quasi periodic, else = invalid kernel

#a = np.array([l,vc,vr,lc,bc,br]) <- THIS IS FOR THE SQUARED EXPONENTIAL
#a = np.array([0.1, 10, 0, 0, 0, 0])

#a = np.array([lp,le,p,vc,vr,lc,bc,br]) <- THIS IS FOR THE QUASI PERIODIC
a = np.array([0.2, 10000, 100, 20, 0, 0, 0, 0])


### Example for the squared exponential  ###
#data
t,rv,rvyerr, bis, rhk, sig_rhk = np.loadtxt("dados.rdb",skiprows=2,unpack=True, usecols=(0,1,2,5,9,10))
t=t-t[0]

#time
t = np.linspace(1,160,160)
#measurements
y=np.hstack((rv,rhk,bis))
#error in measurements
yerr=np.hstack((rvyerr,sig_rhk,2*rvyerr))

#plot of covariance matrix
k = build_bigmatrix(kernel,a,t,y,yerr)
pl.figure()
pl.imshow(k)
pl.show()

#log-likelihood
print('log marginal likelihood = ', likelihood(kernel,a,t,y,yerr))

##random draws of the GP
yf=multivariate_normal(np.zeros(y.size), k).rvs()
pl.figure()
pl.plot(t,yf[0:160],'-')
pl.show()

#pl.figure()
#pl.plot(np.tile(t,3),multivariate_normal(np.zeros(y.size), k).rvs(),'.')
#pl.show()

#### minimization
##res = minimize(inv_ll, x0=a, args=(t,y,yerr,kernel))
##print(res)
##print
#
##### simple sample and marginalization with emcee
#runs, burns = 10000, 10000
#if kernel == 1:
#    #probabilistic model
#    def logprob(p):
#        if np.any((-10 > p[1:]) + (p[1:] > 100)):
#            return -np.inf
#    #    if any([p[0] < -10, p[0] > np.log(10), 
#    #            p[1] < -10, p[1] > np.log(100),
#    #            p[2] < -10, p[2] > np.log(10),
#    #            p[3] < -10, p[3] > 10,
#    #            p[4] < -10, p[2] > np.log(100),
#    #            p[5] < -10, p[5] > np.log(100)]):
#    #        return -np.inf
#        logprior = 0.0
#        return logprior + likelihood(kernel,p,t,y,yerr)
#    
#    #l_prior  = stats.uniform(np.exp(-10), 10 -np.exp(-10))         #[exp(-10) to 10]
#    #vc_prior = stats.uniform(np.exp(-10), 100 -np.exp(-10))         #[exp(-10) to 100]
#    #vr_prior = stats.uniform(np.exp(-10), 10 -np.exp(-10))         #[exp(-10) to 10]
#    #lc_prior = stats.uniform(np.exp(-10), np.exp(10) -np.exp(-10))         #[exp(-10) to exp(10)]
#    #bc_prior = stats.uniform(np.exp(-10), 100 -np.exp(-10))         #[exp(-10) to 100]
#    #br_prior = stats.uniform(np.exp(-10), 100 -np.exp(-10))         #[exp(-10) to 100]
#    #def from_prior():           
#    #    return np.array([ l_prior.rvs(),vc_prior.rvs(),vr_prior.rvs(),
#    #                         lc_prior.rvs(),bc_prior.rvs(),br_prior.rvs() ])
#    
#    # Set up the sampler.
#    nwalkers, ndim = 2*len(a), len(a)
#    sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob)
#    
#    # Initialize the walkers.
#    p0 = a + 1e-4 * np.random.randn(nwalkers, ndim)
#    #p0=[np.log(from_prior()) for i in range(nwalkers)]           
#    #assert not np.isinf(map(logprob, p0)).any()
#    #assert not np.isnan(map(logprob, p0)).any()
#    
#    print("Running burn-in")
#    p0, _, _ = sampler.run_mcmc(p0, burns)
#    
#    print("Running production chain")
#    sampler.run_mcmc(p0, runs);
#
#    #graphs
#    print('graphics')
#    fig, axes = pl.subplots(6, 1, sharex=True, figsize=(8, 9))
#    axes[0].plot(np.exp(sampler.chain[:, burns:, 0]).T, color="k", alpha=0.4)
#    axes[0].yaxis.set_major_locator(MaxNLocator(5))
#    axes[0].set_ylabel("$kernel length scale$")
#    axes[1].plot(sampler.chain[:, burns:, 1].T, color="k", alpha=0.4) #log
#    axes[1].yaxis.set_major_locator(MaxNLocator(5))
#    axes[1].set_ylabel("$Vc$")
#    axes[2].plot(sampler.chain[:, burns:, 2].T, color="k", alpha=0.4) #log
#    axes[2].yaxis.set_major_locator(MaxNLocator(5))
#    axes[2].set_ylabel("$Vr$")
#    axes[3].plot(sampler.chain[:, burns:, 3].T, color="k", alpha=0.4) #log 
#    axes[3].yaxis.set_major_locator(MaxNLocator(5))
#    axes[3].set_ylabel("$Lc$")
#    axes[4].plot(sampler.chain[:, burns:, 4].T, color="k", alpha=0.4) #log
#    axes[4].yaxis.set_major_locator(MaxNLocator(5))
#    axes[4].set_ylabel("$Bc$")
#    axes[5].plot(sampler.chain[:, burns:, 5].T, color="k", alpha=0.4) #log
#    axes[5].yaxis.set_major_locator(MaxNLocator(5))
#    axes[5].set_ylabel("$Br$")
#    axes[5].set_xlabel("step number")
#    fig.tight_layout(h_pad=0.0)
#    pl.show()
#    
#    # Compute the quantiles.
#    burnin = burns
#    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
#    
#    samples[:, 0] = np.exp(samples[:, 0])   #Kernel length scale
#    samples[:, 1] = np.exp(samples[:, 1])   #Vc
#    samples[:, 2] = np.exp(samples[:, 2])   #Vr
#    samples[:, 3] = np.exp(samples[:, 3])   #Lc
#    samples[:, 4] = np.exp(samples[:, 4])   #Bc
#    samples[:, 5] = np.exp(samples[:, 5])   #Br
#    
#    l_mcmc,vc_mcmc,vr_mcmc,lc_mcmc,bc_mcmc,br_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
#                                 zip(*np.percentile(samples, [16, 50, 84],axis=0)))
#    
#    print 'kernel length scale = {0[0]} +{0[1]} -{0[2]}'.format(l_mcmc)
#    print 'Vc = {0[0]} +{0[1]} -{0[2]}'.format(vc_mcmc)
#    print 'Vr = {0[0]} +{0[1]} -{0[2]}'.format(vr_mcmc)
#    print 'Lc = {0[0]} +{0[1]} -{0[2]}'.format(lc_mcmc)
#    print 'Bc = {0[0]} +{0[1]} -{0[2]}'.format(bc_mcmc)
#    print 'Br = {0[0]} +{0[1]} -{0[2]}'.format(br_mcmc)
