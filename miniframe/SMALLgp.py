#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.stats import multivariate_normal
from sys import exit
from copy import copy

flatten = lambda l: [item for sublist in l for item in sublist]

class SMALLgp(object):
    def __init__(self, kernel,  means, number_models, t, *args):
        self.kernel = kernel #kernel and its derivatives
        self.dKdt1, self.dKdt2, self.ddKdt2dt1, self.dddKdt2ddt1, \
            self.dddKddt2dt1, self.ddddKddt2ddt1 = self.kernel.__subclasses__()

        self.means = means
        self._mean_pars = []
        for i, m in enumerate(self.means):
            if m is None: 
                continue
            self.means[i] = m.initialize()
            self._mean_pars.append(self.means[i].pars)
        self._mean_pars = flatten(self._mean_pars)

        self.t = t #time
        self.number_models = number_models #number of models/equations
        self.tt = np.tile(t, self.number_models) #"extended" time

        self.args = args #the data, it should be given as [data1, data1_error, ...]
        self.y = [] 
        self.yerr = []
        for i,j  in enumerate(args):
            if i%2 == 0:
                self.y.append(j)
            else:
                self.yerr.append(j**2)
        self.y = np.array(self.y)
        self.yerr = np.concatenate(self.yerr)
        #if everything goes ok then numbers of y lists = number of models
        assert (i+1)/2 == number_models, 'Given data and number models dont match'


    def _kernel_matrix(self, kernel, x):
        """ returns the covariance matrix created by evaluating `kernel` at inputs x """
        r = x[:, None] - x[None, :]
        K = kernel(r)
        return K


    def _kernel_pars(self, a):
        """ returns the kernel parameters """
        if self.kernel.__name__ == 'SquaredExponential':
            l = a[0]
            return [l]
        elif self.kernel.__name__ == 'QuasiPeriodic':
            lp, le, p = a[:2]
            return [lp, le, p]


    def _scaling_pars(self, a, position):
        """ This returns the the constants of a given model/equation """
        if position == self.number_models:
            return a[-3*self.number_models+3*(position-1) :]
        return a[-3*self.number_models+3*(position-1) : -3*self.number_models+3*position]


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

    def kii(self, a, x, position):
        """ Creates the diagonal matrices used to form the final matrix """
        kpars = self._kernel_pars(a)
        a1, a2, a3 = self._scaling_pars(a, position)

        gammagg  = self._kernel_matrix(self.kernel(*kpars), x)
        gammadgdg = self._kernel_matrix(self.ddKdt2dt1(*kpars), x)
        gammaddgddg = self._kernel_matrix(self.ddddKddt2ddt1(*kpars), x)
        gammagdg = self._kernel_matrix(self.dKdt2(*kpars), x)
        gammadgg = self._kernel_matrix(self.dKdt1(*kpars), x)
        
        gammagddg = -self._kernel_matrix(self.ddKdt2dt1(*kpars), x)
        gammaddgg = -self._kernel_matrix(self.ddKdt2dt1(*kpars), x)
        gammadgddg = self._kernel_matrix(self.dddKddt2dt1(*kpars), x)
        gammaddgdg = self._kernel_matrix(self.dddKdt2ddt1(*kpars), x)

        f1 = a1**2*gammagg + a2**2*gammadgdg + a3**2*gammaddgddg
        f2 = (a1*a2)*(gammadgg + gammagdg)
        f3 = (a1*a3)*(gammagddg + gammaddgg)
        f4 = (a2*a3)*(gammadgddg + gammaddgdg)
        return f1 + f2 + f3 + f4


    def kij(self, a, x, position1, position2):
        """ Creates the remaining matrices used to form the final matrix """
        kpars = self._kernel_pars(a)
        a1, a2, a3 = self._scaling_pars(a, position1)
        b1, b2, b3 = self._scaling_pars(a, position2)

        gammagg  = self._kernel_matrix(self.kernel(*kpars), x)
        gammadgdg = self._kernel_matrix(self.ddKdt2dt1(*kpars), x)
        gammaddgddg = self._kernel_matrix(self.ddddKddt2ddt1(*kpars), x)
        gammagdg = self._kernel_matrix(self.dKdt2(*kpars), x)
        gammadgg = self._kernel_matrix(self.dKdt1(*kpars), x)
        
        gammagddg = -self._kernel_matrix(self.ddKdt2dt1(*kpars), x)
        gammaddgg = -self._kernel_matrix(self.ddKdt2dt1(*kpars), x)
        gammadgddg = self._kernel_matrix(self.dddKddt2dt1(*kpars), x)
        gammaddgdg = self._kernel_matrix(self.dddKdt2ddt1(*kpars), x)

        f1 = a1*b1*gammagg + a2*b2*gammadgdg + a3*b3*gammaddgddg
        f2 = a2*b1*gammadgg + a1*b2*gammagdg
        f3 = a1*b3*gammagddg + a3*b1*gammaddgg
        f4 = a2*b3*gammadgddg + a3*b2*gammaddgdg
        return f1 + f2 + f3 + f4


    def compute_matrix(self, a, yerr=True, nugget=False):
        """ Creates the final covariance matrix """
        if yerr:
            diag = self.yerr
        else:
            diag = 1e-12 * np.identity(self.t.size)

        K_size = self.t.size*self.number_models  #size of the matrix
        K_start = np.zeros((K_size, K_size))        #initial "empty" matrix
        if self.number_models == 1:
            K = self.kii(a, self.t, position = 1) + diag
        else:
            j=1
            while j <= self.number_models:
                for i in range(1, self.number_models+1):
                    if i == j:
                        k = self.kii(a, self.t, position = i)
                        K_start[(i-1)*self.t.size : i*self.t.size, (j-1)*self.t.size : j*self.t.size] = k
                        
                    else:
                        k = self.kij(a, self.t, position1 = i, position2 = j)
                        K_start[(i-1)*self.t.size : i*self.t.size, (j-1)*self.t.size : j*self.t.size] = k
                        K_start[(j-1)*self.t.size : j*self.t.size, (i-1)*self.t.size : i*self.t.size] = k.T
                j += 1
        K = K_start + np.diag(diag)

        if nugget:
            #To give more "weight" to the diagonal
            nugget_value = 0.01
            K = (1 - nugget_value)*K + nugget_value*np.diag(np.diag(K))
        return K


    def log_likelihood(self, a, b, nugget = True):
        """ Calculates the marginal log likelihood. 
        On it we consider the mean function to be zero.

        Parameters:
            a = array with the scaling parameters
            y = values of the dependent variable (the measurements)

        Returns:
            marginal log likelihood
        """
        #calculate covariance matrix with kernel parameters a
        K = self.compute_matrix(a)
        #calculate mean and residuals with mean parameters b
        yy = np.concatenate(self.y)
        self.mean_pars = b
        yy = yy - self.mean()

        try:
            L1 = cho_factor(K, overwrite_a=True, lower=False)
            print(- 0.5*np.dot(yy.T, cho_solve(L1, yy)), - np.sum(np.log(np.diag(L1[0]))), - 0.5*yy.size*np.log(2*np.pi))
            log_like = - 0.5*np.dot(yy.T, cho_solve(L1, yy)) \
                       - np.sum(np.log(np.diag(L1[0]))) \
                       - 0.5*yy.size*np.log(2*np.pi)
        except LinAlgError:
            return -np.inf
        return log_like


    def minus_log_likelihood(self, a, b, nugget = True):
        """ Equal to -log_likelihood(self, a, b, nugget = True) """
        return - self.log_likelihood(a, b, nugget = True)


    def check_symmetry(self, a, tol=1e-10):
        """ Checks if the covariance matrix is symmetric """
        K = self.compute_matrix(a)
        return np.allclose(K, K.T, atol=tol)


    def sample(self, a):
        mean = np.zeros_like(self.tt)
        cov = self.compute_matrix(a)
        norm = multivariate_normal(mean, cov, allow_singular=True)
        return norm.rvs()


    def sample_from_G(self, t, a):
        kpars = self._kernel_pars(a)
        cov = self._kernel_matrix(self.kernel(*kpars), t)

        norm = multivariate_normal(np.zeros_like(t), cov)
        return norm.rvs()
