#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import emcee

from scipy.linalg import cho_factor, cho_solve, LinAlgError, eigh
from scipy import stats
from scipy.stats import multivariate_normal
from copy import copy

from miniframe.kernels import SquaredExponential
from miniframe.means import *

flatten = lambda l: [item for sublist in l for item in sublist]

class BIGgp(object):
    """ Big initial class to create our Gaussian process """
    def __init__(self, 
                 kernel, means, t, rv, rverr, bis, sig_bis, rhk, sig_rhk):
        self.kernel = kernel

        self.means = means
        self._mean_pars = []
        for i, m in enumerate(self.means):
            if m is None: 
                continue
            self.means[i] = m.initialize()
            self._mean_pars.append(self.means[i].pars)

        self._mean_pars = flatten(self._mean_pars)
        # self._mean_pars = np.concatenate(self._mean_pars)

        self.dKdt1, self.dKdt2, self.ddKdt2dt1, self.dddKdt2ddt1, \
            self.dddKddt2dt1, self.ddddKddt2ddt1 = self.kernel.__subclasses__()
        self.t = t
        self.rv = rv
        self.rverr = rverr
        self.bis = bis
        self.sig_bis = sig_bis
        self.rhk = rhk
        self.sig_rhk = sig_rhk
        self.L = 2

        self.y = np.concatenate([rv, bis, rhk])
        self.yerr = np.concatenate([rverr, sig_bis, sig_rhk])
        self.tt = np.tile(t, self.L+1)
        self.tplot = []
        for i in range(self.L + 1):
            self.tplot.append(self.t + 1.5*self.t.ptp()*i)
        self.tplot = np.array(self.tplot).flatten()


    @classmethod
    def from_rdb(cls, filename, kernel=SquaredExponential, **kwargs):
        """ 
        Create class from a .rdb file
        Arguments:
            filename
            kernel (optional)
            usecols = tuple with columns of the file to read into t, rv, rverr, bis, rhk, sig_rhk
                      (uncertainty in bis is 2*rverr)
            skiprows = number of rows to skip (default 2)
        """
        skip = kwargs.get('skiprows', 2)
        kwargs.get('unpack')
        t, rv, rverr, bis, rhk, sig_rhk = \
                    np.loadtxt(filename, skiprows=skip, unpack=True, **kwargs)

        # print ('removing t[0] from times: %f' % t[0])
        # print ('dividing times by time span: %f' % t.ptp())
        # t -= t[0]
        # t /= t.ptp()
        t = np.linspace(0, 1, t.size)
        biserr = 2*rverr # need to do this before changing rverr

        print ('removing mean from RVs: %f' % rv.mean())
        print ('dividing RVs by std: %f' % rv.std())
        rv, rverr = scale(rv, rverr)

        print ('removing mean from BIS: %f' % bis.mean())
        print ('dividing BIS by std: %f' % bis.std())
        bis, sig_bis = scale(bis, biserr)

        print ('removing mean from logRhk: %f' % rhk.mean())
        print ('dividing logRhk by std: %f' % rhk.std())
        rhk, sig_rhk = scale(rhk, sig_rhk)
        return cls(kernel, t, rv, rverr, bis, sig_bis, rhk, sig_rhk)


    def _kernel_matrix(self, kernel, x):
        """ returns the covariance matrix created by evaluating `kernel` at inputs x """
        r = x[:, None] - x[None, :]
        K = kernel(r)
        return K


    def _kernel_pars(self, a):
        if self.kernel.__name__ == 'SquaredExponential':
            l, vc, vr, lc, bc, br = a
            return [l]
        elif self.kernel.__name__ == 'QuasiPeriodic':
            lp, le, p, vc, vr, lc, bc, br = a
            return [lp, le, p]


    @property
    def mean_pars_size(self):
        return self._mean_pars_size


    @mean_pars_size.getter
    def mean_pars_size(self):
        self._mean_pars_size = 0
        for m in self.means:
            if m is None: self._mean_pars_size += 0
            else: self._mean_pars_size += m._parsize
        return self._mean_pars_size


    @property
    def mean_pars(self):
        return self._mean_pars


    # @mean_pars.getter
    # def mean_pars(self):
    #     tmp = []
    #     for m in self.means:
    #         if m is None: 
    #             continue
    #         else: 
    #             tmp.append(m.pars)
    #     return np.concatenate(tmp)


    @mean_pars.setter
    def mean_pars(self, pars):
        pars = list(pars)
        assert len(pars) == self.mean_pars_size
        self._mean_pars = copy(pars)
        for i, m in enumerate(self.means):
            if m is None: 
                continue
            j = 0
            for j in range(m._parsize):
                m.pars[j] = pars.pop(0)


    def mean(self):
        N = self.t.size
        m = np.zeros_like(self.tt)
        for i, meanfun in enumerate(self.means):
            if meanfun is None:
                continue
            else:
                m[i*N : (i+1)*N] = meanfun(self.t)
        return m


    def _mean_vector(self, MeanModel, x):
        """ returns the value of the mean function """
        return MeanModel(self.t)


    def _scaling_pars(self, a):
        return a[-5:]


    def k11(self, a, x):
        """ Equation 18 """
        kpars = self._kernel_pars(a)
        vc, vr, lc, bc, br = self._scaling_pars(a)

        gammagg  = self._kernel_matrix(self.kernel(*kpars), x)
        gammadgdg = self._kernel_matrix(self.ddKdt2dt1(*kpars), x)
        gammagdg = self._kernel_matrix(self.dKdt1(*kpars), x)
        gammadgg = self._kernel_matrix(self.dKdt2(*kpars), x)
        return vc**2 * gammagg + vr**2 * gammadgdg + vc*vr*(gammagdg + gammadgg)


    def k22(self, a, x):
        """ Equation 19 """
        kpars = self._kernel_pars(a)
        vc, vr, lc, bc, br = self._scaling_pars(a)

        gammagg  = self._kernel_matrix(self.kernel(*kpars), x)
        return lc**2 * gammagg


    def k33(self, a, x):
        """ Equation 20 """
        kpars = self._kernel_pars(a)
        vc, vr, lc, bc, br = self._scaling_pars(a)

        gammagg  = self._kernel_matrix(self.kernel(*kpars), x)
        gammadgdg = self._kernel_matrix(self.ddKdt2dt1(*kpars), x)
        gammagdg = self._kernel_matrix(self.dKdt1(*kpars), x)
        gammadgg = self._kernel_matrix(self.dKdt2(*kpars), x)
        return bc**2 * gammagg + br**2 * gammadgdg + bc*br*(gammagdg + gammadgg)


    def k12(self, a, x):
        """ Equation 21 """
        kpars = self._kernel_pars(a)
        vc, vr, lc, bc, br = self._scaling_pars(a)

        gammagg  = self._kernel_matrix(self.kernel(*kpars), x)
        gammagdg = self._kernel_matrix(self.dKdt1(*kpars), x)
        return vc*lc * gammagg + vr*lc * gammagdg


    def k13(self, a, x):
        """ Equation 22 """
        kpars = self._kernel_pars(a)
        vc, vr, lc, bc, br = self._scaling_pars(a)

        gammagg  = self._kernel_matrix(self.kernel(*kpars), x)
        gammadgdg = self._kernel_matrix(self.ddKdt2dt1(*kpars), x)
        gammagdg = self._kernel_matrix(self.dKdt2(*kpars), x)
        gammadgg = self._kernel_matrix(self.dKdt1(*kpars), x)
        return vc*bc*gammagg + vr*br* gammadgdg + vc*br*gammagdg + vr*bc*gammadgg


    def k23(self, a, x):
        """ Equation 23 """
        kpars = self._kernel_pars(a)
        vc, vr, lc, bc, br = self._scaling_pars(a)

        gammagg  = self._kernel_matrix(self.kernel(*kpars), x)
        gammagdg = self._kernel_matrix(self.dKdt2(*kpars), x)
        return lc*bc*gammagg + lc*br*gammagdg


    def compute_matrix(self, a, yerr=True, nugget=False):
        """ Creates the big covariance matrix, equations 24 in the paper """ 
        print ('Vc:%.2f  Vr:%.2f  Lc:%.2f  Bc:%.2f  Br:%.2f' % tuple(self._scaling_pars(a)))
        if yerr:
            diag1 = self.rverr**2 * np.identity(self.t.size)
            diag2 = self.sig_rhk**2 * np.identity(self.t.size)
            diag3 = self.sig_bis**2 * np.identity(self.t.size)
        else:
            diag1 = 1e-12 * np.identity(self.t.size)
            diag2 = diag3 = diag1

        K11 = self.k11(a, self.t) + diag1
        K22 = self.k22(a, self.t) + diag2
        K33 = self.k33(a, self.t) + diag3
        K12 = self.k12(a, self.t)
        K13 = self.k13(a, self.t)
        K23 = self.k23(a, self.t)
        K1 = np.hstack((K11, K12, K13))
        K2 = np.hstack((K12.T, K22, K23))
        K3 = np.hstack((K13.T, K23.T, K33))
        K = np.vstack((K1, K2, K3))

        if nugget:
            #To give more "weight" to the diagonal
            nugget_value = 0.01
            K = (1 - nugget_value)*K + nugget_value*np.diag(np.diag(K))
        return K


    def log_likelihood(self, a, b, nugget = True):
        """ Calculates the marginal log likelihood. 
        Parameters:
            a = array with the kernel parameters
            b = array with the mean functions parameters
            y = values of the dependent variable (the measurements)
        Returns:
            marginal log likelihood
        """
        #calculate covariance matrix with kernel parameters a
        K = self.compute_matrix(a)
        #calculate mean and residuals with mean parameters b
        self.mean_pars = b
        r = self.y - self.mean()

        try:
            L1 = cho_factor(K, overwrite_a=True, lower=False)
            log_like = - 0.5*np.dot(r.T, cho_solve(L1, r)) \
                       - np.sum(np.log(np.diag(L1[0]))) \
                       - 0.5*r.size*np.log(2*np.pi)
        except LinAlgError:
            return -np.inf
        return log_like


    def minus_log_likelihood(self, a, b, nugget = True):
        """ Equal to -log_likelihood(self, a, y, nugget = True) """
        return - self.log_likelihood(a, b, nugget = True)


    def sample(self, a):
        mean = np.zeros_like(self.tt)
        cov = self.compute_matrix(a)
        norm = multivariate_normal(mean, cov, allow_singular=True)
        return norm.rvs()


    def sample_from_gp(self, t, a):
        kpars = self._kernel_pars(a)
        cov = self._kernel_matrix(self.kernel(*kpars), t)
        norm = multivariate_normal(np.zeros_like(t), cov)
        return norm.rvs()

    #Do I want it here or outside?
    def draw_from_gp(self, t, a):
        kpars = self._kernel_pars(a)
        cov = self._kernel_matrix(self.kernel(*kpars), t)

        #drawing RVs
        L1 = cho_factor(cov)
        sol = cho_solve(L1, self.rv)

        #TO DO
        #First I want the normal covariance, not what is happening in cov,
        #needs redoing, 
        #then I calc for new yys in the np.dot bellow


        rv_mean = [] #mean = K*.K-1.y  
        for i, e in enumerate(t):
            rv_mean.append(np.dot(cov[i,:], sol))

        rv_var = [] #var=  K** - K*.K-1.K*.T
        diag = np.diagonal(cov)
        for i, e in enumerate(t):
            #K**=diag[i]; K*=new_lines[i]      
            a = diag[i]
            newsol = cho_solve(L1, cov[i])
            d = np.dot(cov[i,:], newsol)
            result = a - d      
            rv_var.append(result)
        
        rv_std = np.sqrt(rv_var) #standard deviation
        return [rv_mean,rv_std]

#    def run_mcmc(self, a=None, b=None, iter=20, burns=10):
#        """
#        A simple mcmc implementation using emcee
#        Parameters:
#            iter = number of iterarions
#            burns = numbber of burn-ins
#            p0 = parameters of the kernel and mean function (if exists)
#        """
#        if b == None:
#            #there is nothing to run... yet
#            pass
#        else:
#            #kernel parameters
#            params_size = a.size
#
#            #priors settings
#            prior = stats.uniform(np.exp(-10), np.exp(10) -np.exp(-10))
#            #sampler settings
#            nwalkers, ndim = 2*( len(a)+len(b) ), len(a)+len(b)
#
#            sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob)
#
#            #initializing the walkers
#            p0 = np.hstack((a,b))
#            p0 = [np.log(prior.rvs()) for i in range(nwalkers)]
#            #running burn-in
#            p0, _, _ = sampler.run_mcmc(p0, burns)
#            #running production chain
#            sampler.run_mcmc(p0, iter);
#
#            #quantiles
#            burnin = burns
#            samples = np.exp( sampler.chain[:, burnin:, :].reshape((-1, ndim)) )
#            mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
#                        zip(*np.percentile(samples, [16, 50, 84],axis=0)))
#            for i in range(len(mcmc)):
#                print('var{0} = {1[0]} +{1[1]} -{1[2]}'.format(i,mcmc[i]))
#
#
#def logprob(gp, p0):
#    params_size = gp._kernel_pars,size
#    if np.any((-10 > p0) + (p0 > 10)):
#        return -np.inf
#    logprior = 0.0
#    print(p0)
#    print('size of it', params_size)
#    a = p0[0 : params_size]
#    b = p0[params_size : -1]
#    #update the kernel and compute the log likelihood
#    return logprior + gp.log_likelihood(a, b, gp.y, nugget = True)


#Auxiliary functions
def scale(x, xerr):
    """ 
    to x: subtract mean, divide by std 
    to xerr: divide by std of x
    """
    m, s = x.mean(), x.std()
    return (x-m)/s, xerr/s


def isposdef(A, tol=1e-12):
    E = eigh(A, eigvals_only=True)
    return np.all(E > -tol)


def show_and_check(gp, a):
    K = gp.compute_matrix(a, yerr=True)
    print ('K is positive semi-definite:', isposdef(K))
    names = ('k11', 'k21', 'k31'), ('k12', 'k22', 'k32'), ('k13', 'k23', 'k33')
    (k11, k21, k31), (k12, k22, k32), (k13, k23, k33) = \
                    [np.vsplit(c, gp.L+1) for c in np.hsplit(K, gp.L+1)]
    mats = (k11, k21, k31), (k12, k22, k32), (k13, k23, k33)

    for i in range(gp.L+1):
        for j in range(gp.L+1):
            print ('%s is positive semi-definite: %s' 
                            % (names[i][j], isposdef(mats[i][j])) )

    plt.imshow(K)
    plt.show()
    return K, (k11, k21, k31), (k12, k22, k32), (k13, k23, k33)
