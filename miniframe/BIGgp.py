#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import cho_factor, cho_solve, LinAlgError, eigh
from scipy.stats import multivariate_normal
from copy import copy

from miniframe.kernels import SquaredExponential

flatten = lambda l: [item for sublist in l for item in sublist]

class BIGgp(object):
    """ Big initial class to create our Gaussian process.
    Parameters:
        kernel = kernel being used
        means = list of means being used, None if model doesn't use it
        t = time array
        rv, ..., sig_bis = data, datasets

    IMPORTANT DETAIL: Rajpaul et al. (2015) equations' order are RVs first,
                    then log(R_hk), and the BIS.
    """
    def __init__(self, kernel, means, t, 
                 rv, rverr, rhk, sig_rhk, bis, sig_bis):
        self.kernel = kernel

        self.means = means
        self._mean_pars = []
        for i, m in enumerate(self.means):
            if m is None: 
                continue
            self.means[i] = m.initialize()
            self._mean_pars.append(self.means[i].pars)

        self._mean_pars = flatten(self._mean_pars)

        self.dKdt1, self.dKdt2, self.ddKdt2dt1, self.dddKdt2ddt1, \
            self.dddKddt2dt1, self.ddddKddt2ddt1 = self.kernel.__subclasses__()
        self.t = t              #time
        self.rv = rv            #radial velocity
        self.rverr = rverr
        self.rhk = rhk          #activity index
        self.sig_rhk = sig_rhk
        self.bis = bis          #bisector inverse slope
        self.sig_bis = sig_bis
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
        """ Create class from a .rdb file
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
        """ Returns the covariance matrix created by evaluating `kernel` at inputs x """
        r = x[:, None] - x[None, :]
        K = kernel(r)
        return K


    def _kernel_pars(self, a):
        """ Returns the kernel parameters """
        if self.kernel.__name__ == 'SquaredExponential':
            l, wn, _, _, _, _, _ = a
            return [l, wn]
        elif self.kernel.__name__ == 'QuasiPeriodic':
            lp, le, p, wn, _, _, _, _, _ = a
            return [lp, le, p, wn]
#            lp, le, p, vc, vr, lc, bc, br = a
#            return [lp, le, p]


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


    def _scaling_pars(self, a):
        """ Returns Vc, Vr, Lc, Bc, and Br, see paper for more """
        return a[-5:]
 

    def k11(self, a, x):
        """ Equation 18, needed for compute_matrix() """
        kpars = self._kernel_pars(a)
        vc, vr, lc, bc, br = self._scaling_pars(a)

        gammagg  = self._kernel_matrix(self.kernel(*kpars), x)
        gammadgdg = self._kernel_matrix(self.ddKdt2dt1(*kpars), x)
        gammagdg = self._kernel_matrix(self.dKdt1(*kpars), x)
        gammadgg = self._kernel_matrix(self.dKdt2(*kpars), x)
        return vc**2 * gammagg + vr**2 * gammadgdg + vc*vr*(gammagdg + gammadgg)


    def k22(self, a, x):
        """ Equation 19, needed for compute_matrix() """
        kpars = self._kernel_pars(a)
        vc, vr, lc, bc, br = self._scaling_pars(a)

        gammagg  = self._kernel_matrix(self.kernel(*kpars), x)
        return lc**2 * gammagg


    def k33(self, a, x):
        """ Equation 20, needed for compute_matrix() """
        kpars = self._kernel_pars(a)
        vc, vr, lc, bc, br = self._scaling_pars(a)

        gammagg  = self._kernel_matrix(self.kernel(*kpars), x)
        gammadgdg = self._kernel_matrix(self.ddKdt2dt1(*kpars), x)
        gammagdg = self._kernel_matrix(self.dKdt1(*kpars), x)
        gammadgg = self._kernel_matrix(self.dKdt2(*kpars), x)
        return bc**2 * gammagg + br**2 * gammadgdg + bc*br*(gammagdg + gammadgg)


    def k12(self, a, x):
        """ Equation 21, needed for compute_matrix() """
        kpars = self._kernel_pars(a)
        vc, vr, lc, bc, br = self._scaling_pars(a)

        gammagg  = self._kernel_matrix(self.kernel(*kpars), x)
        gammagdg = self._kernel_matrix(self.dKdt1(*kpars), x)
        return vc*lc * gammagg + vr*lc * gammagdg


    def k13(self, a, x):
        """ Equation 22, needed for compute_matrix() """
        kpars = self._kernel_pars(a)
        vc, vr, lc, bc, br = self._scaling_pars(a)

        gammagg  = self._kernel_matrix(self.kernel(*kpars), x)
        gammadgdg = self._kernel_matrix(self.ddKdt2dt1(*kpars), x)
        gammagdg = self._kernel_matrix(self.dKdt2(*kpars), x)
        gammadgg = self._kernel_matrix(self.dKdt1(*kpars), x)
        return vc*bc*gammagg + vr*br* gammadgdg + vc*br*gammagdg + vr*bc*gammadgg


    def k23(self, a, x):
        """ Equation 23, needed for compute_matrix() """
        kpars = self._kernel_pars(a)
        vc, vr, lc, bc, br = self._scaling_pars(a)

        gammagg  = self._kernel_matrix(self.kernel(*kpars), x)
        gammagdg = self._kernel_matrix(self.dKdt2(*kpars), x)
        return lc*bc*gammagg + lc*br*gammagdg


    def compute_matrix(self, a, yerr=True, nugget=False):
        """ Creates the big covariance matrix K, equations 24 in the paper 
        Parameters:
            a = array with the kernel parameters
            yerr = True if measurements dataset has errors, False otherwise
            nugget = True if K is not positive definite, False otherwise
        Returns:
            Big final matrix 
        """
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
        """ Sample from final K, see equation (24) 
        Parameters:
            a = array with the kernel parameters
        Returns:
            Sample of K 
        """
        mean = np.zeros_like(self.tt)
        cov = self.compute_matrix(a)
        norm = multivariate_normal(mean, cov, allow_singular=True)
        return norm.rvs()


    def sample_from_G(self, a):
        """ Sample from the gaussian process G(t), 
        see equations (13), (14), and (15)
        Parameters:
            a = array with the kernel parameters
        Returns:
            Sample vector
        """
        kpars = self._kernel_pars(a)
        cov = self._kernel_matrix(self.kernel(*kpars), self.t)
        norm = multivariate_normal(np.zeros_like(self.t), cov, allow_singular=True)
        return norm.rvs()


    def sample_from_Gdot(self, a):
        """ Sample from the Gaussian process Gdot(t),
        see equations (13), (14), and (15)
        Parameters:
            a = array with the kernel parameters
        Returns:
            Sample vector
        """
        kpars = self._kernel_pars(a)
        cov = self._kernel_matrix(self.ddKdt2dt1(*kpars),self.t)
        norm = multivariate_normal(np.zeros_like(self.t), cov, allow_singular=True)
        return norm.rvs()


    def predict_G(self, time, y, a):
        """ Predition of the Gaussian process G(t)
        Parameters:
            time = values where the predictive distribution will be calculated
            y = values of the dependent variable (the measurements)
            a = array with the kernel parameters
        Returns:
            mean vector, covariance matrix
        """
        tstar = time[:, None] - self.t[None, :]
        kpars = self._kernel_pars(a)
        K =  self._kernel_matrix(self.kernel(*kpars), self.t)
        Kstar = self.kernel(*kpars)(tstar)

        try:
            L1 = cho_factor(K)
            print('Positive definite matrix')
        except LinAlgError:
            print('Not positive definite matrix')
            return np.zeros_like(time), 0

        sol = cho_solve(L1, y)
        y_mean = np.dot(Kstar, sol)

        Kstarstar = self._kernel_matrix(self.kernel(*kpars), time)
        kstarT_k_kstar = []
        for i, e in enumerate(time):
            kstarT_k_kstar.append(np.dot(Kstar, cho_solve(L1, Kstar[i,:])))
        y_cov = Kstarstar - kstarT_k_kstar
        return y_mean, y_cov

    def predict_Gdot(self, time, y, a):
        """ Predition of the Gaussian process Gdot(t), the derivative of G(t)
        Parameters:
            time = values where the predictive distribution will be calculated
            y = values of the dependent variable (the measurements)
            a = array with the kernel parameters
        Returns:
            mean vector, covariance matrix
        """
        tstar = time[:, None] - self.t[None, :]
        kpars = self._kernel_pars(a)
        K = self._kernel_matrix(self.ddKdt2dt1(*kpars), self.t)
        Kstar = self.ddKdt2dt1(*kpars)(tstar)

        try:
            L1 = cho_factor(K)
            print('Positive definite matrix')
        except LinAlgError:
            print('Not positive definite matrix')
            return np.zeros_like(time), 0

        sol = cho_solve(L1, y)
        y_mean = np.dot(Kstar, sol)

        Kstarstar = self._kernel_matrix(self.ddKdt2dt1(*kpars), time)
        kstarT_k_kstar = []
        for i, e in enumerate(time):
            kstarT_k_kstar.append(np.dot(Kstar, cho_solve(L1, Kstar[i,:])))
        y_cov = Kstarstar - kstarT_k_kstar
        return y_mean, y_cov


    def predict_rv(self, time, a):
        """ Conditional predictive distribution for the RVs
        Parameters:
            time = values where the predictive distribution will be calculated
            y = values of the dependent variable (the measurements)
            a = array with the kernel parameters
        Returns:
            mean vector, covariance matrix, standard deviation vector
        """
        mu, cov = self.predict_G(time, self.rv, a)
        mudot, covdot = self.predict_Gdot(time, self.rv, a)
        vc, vr, lc, bc, br = self._scaling_pars(a)
        mean = vc*mu + vr*mudot
        covariance = vc*cov + vr*covdot
        std = np.sqrt(np.diag(covariance))
        return mean, covariance, std


    def predict_rhk(self, time, a):
        """ Conditional predictive distribution for the log(R_hk)
        Parameters:
            time = values where the predictive distribution will be calculated
            y = values of the dependent variable (the measurements)
            a = array with the kernel parameters
        Returns:
            mean vector, covariance matrix, standard deviation vector
        """
        mu, cov = self.predict_G(time, self.rhk, a)
        vc, vr, lc, bc, br = self._scaling_pars(a)
        mean = lc*mu
        covariance = lc*cov 
        std = np.sqrt(np.diag(covariance))
        return mean, covariance, std


    def predict_bis(self, time, a):
        """ Conditional predictive distribution for the BIS
        Parameters:
            time = values where the predictive distribution will be calculated
            y = values of the dependent variable (the measurements)
            a = array with the kernel parameters
        Returns:
            mean vector, covariance matrix, standard deviation vector
        """
        mu, cov = self.predict_G(time, self.bis, a)
        mudot, covdot = self.predict_Gdot(time, self.bis, a)
        vc, vr, lc, bc, br = self._scaling_pars(a)
        mean = bc*mu + br*mudot
        covariance = bc*cov + br*covdot
        std = np.sqrt(np.diag(covariance))
        return mean, covariance, std


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


    def predict_gp(self, time, a, b, model = 'rv'):
        """ Conditional predictive distribution of the Gaussian process
        Parameters:
            time = values where the predictive distribution will be calculated
            y = values of the dependent variable (the measurements)
            a = array with the kernel parameters
            b = array with the means parameters
            model = 'rv' or 'bis' or 'rhk' accordingly to the data we are using
        Returns:
            mean vector, covariance matrix, standard deviation vector
        """
        kpars = self._kernel_pars(a)

        self.mean_pars = b
        r = self.y - self.mean()
        new_y = np.array_split(r, 3)

        if model == 'rv':
            print('Working with RVs')
            cov = self.k11(a, self.t)
            L1 = cho_factor(cov)
            sol = cho_solve(L1, new_y[0])
            tstar = time[:, None] - self.t[None, :]
            vc, vr, _, _, _ = self._scaling_pars(a)
            Kstar = vc*vc*self.kernel(*kpars)(tstar) + vr*vr*self.ddKdt2dt1(*kpars)(tstar) \
                    + vc*vr*(self.dKdt1(*kpars)(tstar) + self.dKdt2(*kpars)(tstar))
            Kstarstar = self.k11(a, time)
        if model == 'rhk':
            print('Working with log(Rhk)')
            cov = self.k22(a, self.t)
            L1 = cho_factor(cov)
            sol = cho_solve(L1, new_y[2])
            tstar = time[:, None] - self.t[None, :]
            _, _, lc, _, _ = self._scaling_pars(a)
            Kstar = lc*lc*self.kernel(*kpars)(tstar)
            Kstarstar = self.k22(a, time)
        if model == 'bis':
            print('Working with BIS')
            cov = self.k33(a, self.t)
            L1 = cho_factor(cov)
            sol = cho_solve(L1, new_y[1])
            tstar = time[:, None] - self.t[None, :]
            _, _, _, bc, br = self._scaling_pars(a)
            Kstar = bc*bc*self.kernel(*kpars)(tstar) + br*br*self.ddKdt2dt1(*kpars)(tstar) \
                    + bc*br*(self.dKdt1(*kpars)(tstar) + self.dKdt2(*kpars)(tstar))
            Kstarstar = self.k33(a, time)

        y_mean = np.dot(Kstar, sol)
        kstarT_k_kstar = []
        for i, e in enumerate(time):
            kstarT_k_kstar.append(np.dot(Kstar, cho_solve(L1, Kstar[i,:])))
        y_cov = Kstarstar - kstarT_k_kstar
        y_var = np.diag(y_cov) #variance
        y_std = np.sqrt(y_var) #standard deviation
        return y_mean, y_cov, y_std


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


### END