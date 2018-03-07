import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.stats import multivariate_normal

from kernels import SquaredExponential
import cov_matrix

class BIGgp(object):
    def __init__(self, kernel, t, rv, rverr, variables):
        
        self.kernel = kernel
        self.dKdt1, self.dKdt2, self.ddKdt2dt1 = self.kernel.__subclasses__()

        self.t = t
        self.rv = rv
        self.rverr = rverr
        # extra dependent variables, besides the RVs
        var = np.atleast_2d(variables)
        # how many more dependent variables
        self.L = var.shape[0]
        assert self.L == 2, 'for now I only work with RVs, Rhk and BIS... sorry'
        self.var = var

        self.yerr = np.tile(rverr, self.L+1)
        self.tt = np.tile(t, self.L+1)
        self.tplot = []
        for i in range(self.L + 1):
            self.tplot.append(self.t + 1.5*self.t.ptp()*i)
        self.tplot = np.array(self.tplot).flatten()


    @classmethod
    def from_rdb(cls, filename, kernel=SquaredExponential, **kwargs):
        skip = kwargs.get('skiprows', 2)
        kwargs.get('unpack')
        t, rv, rverr, *rest = np.loadtxt(filename, skiprows=skip, unpack=True, **kwargs)
        return cls(kernel, t, rv, rverr, rest)

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
        
        return vc**2 * gammagg + vr**2* gammadgdg + vc*vr*(gammagdg + gammadgg)


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

        return bc**2 * gammagg + br**2* gammadgdg + bc*br*(gammagdg + gammadgg)

    def k12(self, a, x):
        """ Equation 21 """
        kpars = self._kernel_pars(a)
        vc, vr, lc, bc, br = self._scaling_pars(a)

        gammagg  = self._kernel_matrix(self.kernel(*kpars), x) 
        gammagdg = self._kernel_matrix(self.dKdt1(*kpars), x)  

        return vc*lc * gammagg + vr*lc*gammagdg

    def k13(self, a, x):
        """ Equation 22 """
        kpars = self._kernel_pars(a)
        vc, vr, lc, bc, br = self._scaling_pars(a)

        gammagg  = self._kernel_matrix(self.kernel(*kpars), x) 
        gammadgdg = self._kernel_matrix(self.ddKdt2dt1(*kpars), x)  
        gammagdg = self._kernel_matrix(self.dKdt1(*kpars), x)  
        gammadgg = self._kernel_matrix(self.dKdt2(*kpars), x)  

        return vc*bc*gammagg + vr*br* gammadgdg + vc*br*gammagdg + vr*bc*gammadgg


    def k23(self, a, x):
        """ Equation 23 """
        kpars = self._kernel_pars(a)
        vc, vr, lc, bc, br = self._scaling_pars(a)

        gammagg  = self._kernel_matrix(self.kernel(*kpars), x) 
        gammagdg = self._kernel_matrix(self.dKdt1(*kpars), x)  

        return bc*lc * gammagg + br*lc*gammagdg


    def compute_matrix(self, a):
        """ Creates the big covariance matrix, equations 24 in the paper """ 
        
        K11 = self.k11(a, self.t)
        K22 = self.k22(a, self.t)
        K33 = self.k33(a, self.t)
        K12 = self.k12(a, self.t)
        K13 = self.k13(a, self.t)
        K23 = self.k23(a, self.t)
        
        K1 = np.hstack((K11, K12, K13))
        K2 = np.hstack((K12, K22, K23))
        K3 = np.hstack((K13, K23, K33))
        
        K = np.vstack((K1, K2, K3))
        # if yerr is not None:
        #     K = K + yerr**2 * np.identity(yerr.size)
        K = K + self.yerr**2 * np.identity(self.yerr.size)

        return K

    def log_likelihood(self, a, y):
        """ Calculates the marginal log likelihood

        Parameters:
            a = array with the scaling parameters
            y = values of the dependent variable (the measurements)

        Returns:
            marginal log likelihood
        """
            
        K = self.compute_matrix(a)

        try:
            L1 = cho_factor(K)
            sol = cho_solve(L1, y)
            n = y.size
            log_like = - 0.5*np.dot(y, sol) \
                       - np.sum(np.log(np.diag(L1[0]))) \
                       - n*0.5*np.log(2*np.pi)        
        except LinAlgError:
            #return -np.inf
            K2=np.linalg.inv(K)
            n = y.size
            log_like = -0.5* np.dot(np.dot(y.T,K2),y) \
                       -np.sum(np.log(np.diag(K))) \
                       -n*0.5*np.log(2*np.pi) 

        return log_like

    def minus_log_likelihood(self, a, y):
        return - self.log_likelihood(a, y)


    def sample(self, a):
        mean = np.zeros_like(self.tt)
        cov = self.compute_matrix(a)
        norm = multivariate_normal(mean, cov, allow_singular=True)
        return norm.rvs()



#a = np.array([l, vc,vr,lc,bc,br]) <- THIS IS FOR THE SQUARED EXPONENTIAL
#a = np.array([0.1, 10, 0, 0, 0, 0])

#a = np.array([lp, le, p, vc,vr,lc,bc,br]) <- THIS IS FOR THE QUASI PERIODIC
# a = np.array([0.2, 10000, 100, 20, 0, 0, 0, 0])


