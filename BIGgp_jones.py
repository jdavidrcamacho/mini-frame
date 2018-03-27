# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt 
from scipy.linalg import cho_factor, cho_solve, LinAlgError, eigh
from scipy.stats import multivariate_normal
from sys import exit

from kernels_jones import SquaredExponential


class BIGgp(object):
    def __init__(self, kernel, number_models, time, *args):
        self.kernel = kernel #kernel and its derivatives
        self.dKdt1, self.dKdt2, self.ddKdt2dt1, self.dddKdt2ddt1, self.dddKddt2dt1, self.ddddKddt2ddt1 = self.kernel.__subclasses__()

        self.time = time #time
        self.number_models = number_models #number of models/equations
        
        self.args = args #the data, it should be [data1, data1_err, ...]
        self.y = []
        self.yerr = []
        for i,j  in enumerate(args):
            if i%2 == 0:
                self.y.append(j)
            else:
                self.yerr.append(j**2)

        self.y = np.array(self.y)                    #if everything goes ok then
        self.yerr = np.concatenate(self.yerr)        #len(y) = num_models
        if len(self.y) != number_models:
            exit('ERROR! Number of models and data dont match')


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


    def kii(self, a, x, position):
        kpars = self._kernel_pars(a)
        a1, a2, a3 = self._scaling_pars(a, position)

        gammagg  = self._kernel_matrix(self.kernel(*kpars), x)
        gammadgdg = self._kernel_matrix(self.ddKdt2dt1(*kpars), x)
        gammaddgddg = self._kernel_matrix(self.ddddKddt2ddt1(*kpars), x)
        gammagdg = self._kernel_matrix(self.dKdt1(*kpars), x)
        gammadgg = self._kernel_matrix(self.dKdt2(*kpars), x)
        gammagddg = gammadgdg
        gammaddgg = gammadgdg
        gammadgddg = self._kernel_matrix(self.dddKdt2ddt1(*kpars), x)
        gammaddgdg = -gammadgddg

        f1 = a1**2*gammagg + a2**2*gammadgdg + a3**2*gammaddgddg
        f2 = (a1*a2)*(gammadgg + gammagdg)
        f3 = (a1*a3)*(gammagddg + gammaddgg)
        f4 = (a2*a3)*(gammadgddg + gammaddgdg)
        return f1 + f2 + f3 + f4


    def kij(self, a, x, position1, position2):
        kpars = self._kernel_pars(a)
        a1, a2, a3 = self._scaling_pars(a, position1)
        b1, b2, b3 = self._scaling_pars(a, position2)

        gammagg  = self._kernel_matrix(self.kernel(*kpars), x)
        gammadgdg = self._kernel_matrix(self.ddKdt2dt1(*kpars), x)
        gammaddgddg = self._kernel_matrix(self.ddddKddt2ddt1(*kpars), x)
        gammagdg = self._kernel_matrix(self.dKdt1(*kpars), x)
        gammadgg = self._kernel_matrix(self.dKdt2(*kpars), x)
        gammagddg = gammadgdg
        gammaddgg = gammadgdg
        gammadgddg = self._kernel_matrix(self.dddKdt2ddt1(*kpars), x)
        gammaddgdg = -gammadgddg

        f1 = a1*b1*gammagg + a2*b2*gammadgdg + a3*b3*gammaddgddg
        f2 = a2*b1*gammadgg + a1*b2*gammagdg
        f3 = a1*b3*gammagddg + a3*b1*gammaddgg
        f4 = a2*b3*gammadgddg + a3*b2*gammaddgdg
        return f1 + f2 + f3 + f4


    def compute_matrix(self, a, yerr=True, nugget=False):
        """ Creates the big covariance matrix, equations 24 in the paper """ 

        if yerr:
            diag = self.yerr
        else:
            diag = 1e-12 * np.identity(self.time.size)

        K_size = self.time.size*self.number_models  #size of the matrix
        K_start = np.zeros((K_size, K_size))        #initial "empty" matrix
        if self.number_models == 1:
            K = self.kii(a, self.t, position = 1) + diag
        else:
            j=1
            while j <= self.number_models:
                for i in range(1, self.number_models+1):
                    if i == j:
                        k = self.kii(a, self.time, position = i)
                        K_start[(i-1)*self.time.size : i*self.time.size, (j-1)*self.time.size : j*self.time.size] = k
                        
                    else:
                        k = self.kij(a, self.time, position1 = i, position2 = j)
                        K_start[(i-1)*self.time.size : i*self.time.size, (j-1)*self.time.size : j*self.time.size] = k
                        K_start[(j-1)*self.time.size : j*self.time.size, (i-1)*self.time.size : i*self.time.size] = k.T
                j += 1
        K = K_start + np.diag(diag)

        if nugget:
            #To give more "weight" to the diagonal
            nugget_value = 0.01
            K = (1 - nugget_value)*K + nugget_value*np.diag(np.diag(K))
        return K


    def log_likelihood(self, a, nugget = True):
        """ Calculates the marginal log likelihood. 
        On it we consider the mean function to be zero.

        Parameters:
            a = array with the scaling parameters
            y = values of the dependent variable (the measurements)

        Returns:
            marginal log likelihood
        """
        K = self.compute_matrix(a)
        yy = np.concatenate(self.y)
        print(type(yy))
        try:
            L1 = cho_factor(K, overwrite_a=True, lower=False)
            log_like = - 0.5*np.dot(yy.T, cho_solve(L1, yy)) \
                       - np.sum(np.log(np.diag(L1[0]))) \
                       - 0.5*yy.size*np.log(2*np.pi)
        except LinAlgError:
            return -np.inf
        return log_like


    def minus_log_likelihood(self, a, y, nugget = True):
        """ Equal to -log_likelihood(self, a, y, nugget = True) """
        return - self.log_likelihood(a, y, nugget = True)

