#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.stats import multivariate_normal
from copy import copy

flatten = lambda l: [item for sublist in l for item in sublist]

class SMALLgp(object):
    """ Initial class to create our Gaussian process.
    Parameters:
        kernel = kernel being used
        means = list of means being used, None if model doesn't use it
        number_models = number of datasets being fitted
        t = time array
        *args = datasets data, it should be given as data1, data1_error, ...
    """ 
    def __init__(self, kernel, extrakernel, means, t, *args):
        self.kernel = kernel #kernel and its derivatives
        self.extrakernel = extrakernel
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
        self.number_models = len(means) #number of models/equations
        self.tt = np.tile(t, self.number_models) #"extended" time

        self.args = args #the data, it should be given as data1, data1_error, ...
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
        assert (i+1)/2 == self.number_models, 'Given data and number models dont match'


    def _kernel_matrix(self, kernel, x):
        """ Returns the covariance matrix created by evaluating `kernel` at inputs x """
        r = x[:, None] - x[None, :]
        K = kernel(r)
        return K


    def _kernel_pars(self, a):
        """ Returns the kernel parameters, X(t) in Jones et al. (2017) """
        if self.kernel.__name__ == 'SquaredExponential':
            l, wn = a[:2]
            return [l]
        elif self.kernel.__name__ == 'QuasiPeriodic':
            lp, le, p, wn = a[:4]
            return [lp, le, p, wn]


    def _extrakernel_pars(self, c):
        """ Returns the extra kernel parameters, Z(t) in Jones et al. (2017) """
        if self.extrakernel is None:
            return []
        elif self.extrakernel.__name__ == 'SquaredExponential':
            l = c[:2]
            return [l]
        elif self.extrakernel.__name__ == 'QuasiPeriodic':
            lp, le, p, wn = c[:4]
            return [lp, le, p, wn]


    def _scaling_pars(self, a, position):
        """ Returns the constants of a given model/equation """
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
#        assert len(pars) == self.mean_pars_size
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
        """ Returns the value of the mean function """
        return MeanModel(self.t)


    def kii(self, a, c, x, position):
        """ Creates the diagonal matrices used to create the big final matrix
        Parameters:
            a = array with the kernel parameters
            x = time dataset
            position = position this kernel will have in the final matrix
        Return:
            matrix
        """ 
        kpars = self._kernel_pars(a)
        extrakpars = self._extrakernel_pars(c)
        a1, a2, a3 = self._scaling_pars(a, position)
        a4 = c[-self.number_models + position -1]

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
        if self.extrakernel is None:
            return f1 + f2 + f3 + f4
        #f5 = a4**2 * extracov
        f5 = a4**2 * self._kernel_matrix(self.extrakernel(*extrakpars), x)
        return f1 + f2 + f3 + f4 + f5


    def kij(self, a, x, position1, position2):
        """ Creates the remaining matrices used to create the big final matrix 
        Parameters:
            a = array with the kernel parameters
            x = time dataset
            position1, position2 = position this matrix will have in the final 
                                    matrix, think of it as its lines and columns
                                    position in the final matrix, see
                                    compute_matrix() to understand it better
        Return:
            matrix
        """ 
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


    def compute_matrix(self, a, c, yerr=True, nugget=False):
        """ Creates the big covariance matrix K
        Parameters:
            a = array with the kernel parameters
            yerr = True if measurements dataset has errors, False otherwise
            nugget = True if K is not positive definite, False otherwise
        Returns:
            Big final matrix 
        """
        if yerr:
            diag = self.yerr
        else:
            diag = 1e-12 * np.identity(self.t.size)

        K_size = self.t.size * self.number_models     #size of the matrix
        K_start = np.zeros((K_size, K_size))        #initial "empty" matrix
        if self.number_models == 1:
            K = self.kii(a, self.t, position = 1) + diag
        else:
            j=1
            while j <= self.number_models:
                for i in range(1, self.number_models+1):
                    if i == j:
                        k = self.kii(a, c, self.t, position = i)
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


    def log_likelihood(self, a, b, c=[], nugget = True):
        """ Calculates the marginal log likelihood. 
        Parameters:
            a = array with the kernel parameters
            b = array with the mean functions parameters
            y = values of the dependent variable (the measurements)
        Returns:
            Marginal log likelihood
        """
        #calculate covariance matrix with kernel parameters a
        K = self.compute_matrix(a, c)
        #calculate mean and residuals with mean parameters b
        yy = np.concatenate(self.y)
        self.mean_pars = b
        yy = yy - self.mean()

        try:
            L1 = cho_factor(K, overwrite_a=True, lower=False)
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


    def sample_from_G(self, t, a):
        """ Sample from the gaussian process G(t)
        Parameters:
            a = array with the kernel parameters
        Returns:
            Sample vector
        """
        kpars = self._kernel_pars(a)
        cov = self._kernel_matrix(self.kernel(*kpars), t)

        norm = multivariate_normal(np.zeros_like(t), cov)
        return norm.rvs()


    def show_matrix(self, x):
        """ Plot of the covariance matrix x 
        Parameters:
            x = matrix
        Returns:
            Matrix plot
        """
        plt.figure()
        plt.imshow(x)
        plt.show()


    def predict_gp(self, time, a, b, model = 1):
        """ Conditional predictive distribution of the Gaussian process
        Parameters:
            time = values where the predictive distribution will be calculated
            y = values of the dependent variable (the measurements)
            a = array with the kernel parameters
            b = array with the mean functions parameters
            model = 1,2,3,... accordingly to the data we are using, 1 represents
                    the first dataset, 2 the second data, etc...
        Returns:
            mean vector, covariance matrix, standard deviation vector
        """
        print('Working with model {0}'.format(model))
        kpars = self._kernel_pars(a)
        a1, a2, a3 = self._scaling_pars(a, model)

        self.mean_pars = b
        y = np.concatenate(self.y, axis=0)
        new_y = np.array_split(y - self.mean(), self.number_models)

        cov = self.kii(a, self.t, model)
        L1 = cho_factor(cov)
        sol = cho_solve(L1, new_y[model-1])
        tstar = time[:, None] - self.t[None, :]

        Kstar = a1*a1*self.kernel(*kpars)(tstar) \
                + a2*a2*self.ddKdt2dt1(*kpars)(tstar) \
                + a3*a3*self.ddddKddt2ddt1(*kpars)(tstar) \
                + a1*a2*(self.dKdt2(*kpars)(tstar) + self.dKdt1(*kpars)(tstar)) \
                + a1*a3*(self.ddKdt2dt1(*kpars)(tstar) + self.ddKdt2dt1(*kpars)(tstar)) \
                + a2*a3*(self.dddKddt2dt1(*kpars)(tstar) + self.dddKdt2ddt1(*kpars)(tstar))

        Kstarstar = self.kii(a, time, model)

        y_mean = np.dot(Kstar, sol)
        kstarT_k_kstar = []
        for i, e in enumerate(time):
            kstarT_k_kstar.append(np.dot(Kstar, cho_solve(L1, Kstar[i,:])))
        y_cov = Kstarstar - kstarT_k_kstar
        y_var = np.diag(y_cov) #variance
        y_std = np.sqrt(y_var) #standard deviation
        return y_mean, y_cov, y_std


### END