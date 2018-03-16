# -*- coding: utf-8 -*-
import kernels
from BIGgp import BIGgp

import numpy as np
import emcee

import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
from scipy import stats
from scipy.stats import multivariate_normal
from scipy.optimize import minimize


### Example for the squared exponential  ###
#data
t,rv,rvyerr, bis, rhk, sig_rhk = np.loadtxt("HD41248_harps.rdb",skiprows=2,unpack=True, usecols=(0,1,2,5,9,10))
t=t-t[0]
t = np.linspace(1, 300, 228)

### a = [l, vc, vr, lc, bc, br] -> kernel parameters
#a = np.array([1, 10, 1, -1, -1, 0])

### R results for the quasi-periodic kernel
#kernel_params = [Period, l-periodic, l-aperiodic]
#kernel_params =[2.3, -0.9, 9.9]
#kernel_params = np.exp(kernel_params)   #because they work with the log
#model_params = [a1,a2,a3,a4, a1,a2,a3,a4, a1,a2,a3,a4]
#model_params = [0.03,0.3,0,0, -0.4,0,-0.07,0, 0,-0.3,0,0]
#model_params = np.exp(model_params)

### R results for the squared exponential kernel
#kernel_params = [beta, lambda]
kernel_params = [2, 0.95]
kernel_params = np.exp(kernel_params)   #because they work with the log
#model_params = [a1,a2,a3,a4, a1,a2,a3,a4, a1,a2,a3,a4]
model_params = [-0.001,-0.3,0,0, -0.04,0,-0.6,0, 0,0.3,0,0]
model_params = np.exp(model_params)

# a = [l, vc, vr, lc, bc, br]
a=np.array([kernel_params[1],
            kernel_params[0]*model_params[0],
            model_params[1],
            kernel_params[0]*model_params[4],
            kernel_params[0]*model_params[8],
            model_params[9]])


gpObj = BIGgp(kernels.SquaredExponential, t=t, rv=rv, rverr=rvyerr,
                    bis=bis, sig_bis=2*rvyerr, rhk=rhk, sig_rhk=sig_rhk)

#measurements
y=np.hstack((rv,rhk,bis))
#error in measurements
yerr=np.hstack((rvyerr,sig_rhk,2*rvyerr))
# #log-likelihood
print(gpObj.log_likelihood(a, y))


#### simple sample and marginalization with emcee
# a = [l, vc, vr, lc, bc, br]
a=np.array([kernel_params[1],
            kernel_params[0]*model_params[0],
            model_params[1],
            kernel_params[0]*model_params[4],
            kernel_params[0]*model_params[8],
            model_params[9]])


gpObj = BIGgp(kernels.SquaredExponential, t=t, rv=rv, rverr=rvyerr,
                    bis=bis, sig_bis=2*rvyerr, rhk=rhk, sig_rhk=sig_rhk)

#measurements
y=np.hstack((rv,rhk,bis))
#error in measurements
yerr=np.hstack((rvyerr,sig_rhk,2*rvyerr))
# #log-likelihood
print(gpObj.log_likelihood(a, y))


##random draws of the GP
#multivariate_normal(np.zeros(y.size), k).rvs()
#pl.figure()
#pl.plot(np.tile(t,3),multivariate_normal(np.zeros(y.size), k).rvs(),'.')
#pl.show()

#### simple sample and marginalization with emcee
runs, burns = 10000, 10000

def logprob(params):
    if np.any((-10 > params[1:]) + (params[1:] > 10)):
        return -np.inf
    
    logprior = 0.0
    return logprior + gpObj.log_likelihood(a, y)


#Set up the sampler.
nwalkers, ndim = 2*len(a), len(a)
sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob)

#Initialize the walkers.
#    p0 = a + 1e-4 * np.random.randn(nwalkers, ndim)
#p0[i,j] is the starting point for walk i along variable j.
p0 = np.empty((nwalkers, ndim))
p0[:,0] = np.random.uniform(np.exp(-10), np.exp(100), nwalkers)        # l
p0[:,1] = np.random.uniform(np.exp(-10), np.exp(100), nwalkers)        # vc
p0[:,2] = np.random.uniform(np.exp(-10), np.exp(100), nwalkers)        # vr
p0[:,3] = np.random.uniform(np.exp(-10), np.exp(100), nwalkers)        # lc
p0[:,4] = np.random.uniform(np.exp(-10), np.exp(100), nwalkers)        # bc
p0[:,5] = np.random.uniform(np.exp(-10), np.exp(100), nwalkers)        # vr
#Make sure we didn't by change pick an value that was too big
p0[:,0] = np.minimum(p0[:,0], np.exp(100))
p0[:,1] = np.minimum(p0[:,1], np.exp(100))
p0[:,2] = np.minimum(p0[:,2], np.exp(100))
p0[:,3] = np.minimum(p0[:,3], np.exp(100))
p0[:,4] = np.minimum(p0[:,4], np.exp(100))
p0[:,5] = np.minimum(p0[:,5], np.exp(100))


print("Running burn-in")
p0, _, _ = sampler.run_mcmc(p0, burns)

print("Running production chain")
sampler.run_mcmc(p0, runs);

#Chains graphics
print('graphics')
fig, axes = pl.subplots(6, 1, sharex=True, figsize=(8, 9))
axes[0].plot(np.exp(sampler.chain[:, burns:, 0]).T, color="k", alpha=0.4)
axes[0].yaxis.set_major_locator(MaxNLocator(5))
axes[0].set_ylabel("$kernel length scale$")
axes[1].plot(sampler.chain[:, burns:, 1].T, color="k", alpha=0.4) #log
axes[1].yaxis.set_major_locator(MaxNLocator(5))
axes[1].set_ylabel("$Vc$")
axes[2].plot(sampler.chain[:, burns:, 2].T, color="k", alpha=0.4) #log
axes[2].yaxis.set_major_locator(MaxNLocator(5))
axes[2].set_ylabel("$Vr$")
axes[3].plot(sampler.chain[:, burns:, 3].T, color="k", alpha=0.4) #log 
axes[3].yaxis.set_major_locator(MaxNLocator(5))
axes[3].set_ylabel("$Lc$")
axes[4].plot(sampler.chain[:, burns:, 4].T, color="k", alpha=0.4) #log
axes[4].yaxis.set_major_locator(MaxNLocator(5))
axes[4].set_ylabel("$Bc$")
axes[5].plot(sampler.chain[:, burns:, 5].T, color="k", alpha=0.4) #log
axes[5].yaxis.set_major_locator(MaxNLocator(5))
axes[5].set_ylabel("$Br$")
axes[5].set_xlabel("step number")
fig.tight_layout(h_pad=0.0)
pl.show()

#Compute the quantiles.
burnin = burns
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

samples[:, 0] = np.exp(samples[:, 0])   #Kernel length scale
samples[:, 1] = np.exp(samples[:, 1])   #Vc
samples[:, 2] = np.exp(samples[:, 2])   #Vr
samples[:, 3] = np.exp(samples[:, 3])   #Lc
samples[:, 4] = np.exp(samples[:, 4])   #Bc
samples[:, 5] = np.exp(samples[:, 5])   #Br


l_mcmc,vc_mcmc,vr_mcmc,lc_mcmc,bc_mcmc,br_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],axis=0)))

print('kernel length scale = {0[0]} +{0[1]} -{0[2]}'.format(l_mcmc))
print('Vc = {0[0]} +{0[1]} -{0[2]}'.format(vc_mcmc))
print('Vr = {0[0]} +{0[1]} -{0[2]}'.format(vr_mcmc))
print('Lc = {0[0]} +{0[1]} -{0[2]}'.format(lc_mcmc))
print('Bc = {0[0]} +{0[1]} -{0[2]}'.format(bc_mcmc))
print('Br = {0[0]} +{0[1]} -{0[2]}'.format(br_mcmc))