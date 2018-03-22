import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError, eigh
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt 

from miniframe.kernels import SquaredExponential

def scale(x, xerr):
    """ 
    to x: subtract mean, divide by std 
    to xerr: divide by std of x
    """
    m, s = x.mean(), x.std()
    return (x-m)/s, xerr/s



class BIGgp(object):
    def __init__(self, kernel, t, rv, rverr, bis, sig_bis, rhk, sig_rhk):
        
        self.kernel = kernel
        self.dKdt1, self.dKdt2, self.ddKdt2dt1 = self.kernel.__subclasses__()

        self.t = t
        self.rv = rv
        self.rverr = rverr
        self.bis = bis
        self.sig_bis = sig_bis
        self.rhk = rhk
        self.sig_rhk = sig_rhk
        self.L = 2
        # assert self.L == 2, 'for now I only work with RVs, Rhk and BIS... sorry'

        self.yerr = np.array([rverr, sig_bis, sig_rhk])
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
        t, rv, rverr, bis, rhk, sig_rhk = np.loadtxt(filename, skiprows=skip, unpack=True, **kwargs)

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


    def compute_matrix(self, a, yerr=True):
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
        K = np.vstack((K1, K2, K3))     # equal to (2)
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
            L1 = cho_factor(K, overwrite_a=True, lower=False)
            log_like = - 0.5*np.dot(y.T, cho_solve(L1, y)) \
                       - np.sum(np.log(np.diag(L1[0]))) \
                       - 0.5*y.size*np.log(2*np.pi)
        except LinAlgError:
            return -np.inf
#            K2=np.linalg.inv(K)
#            n = y.size
#            log_like = -0.5* np.dot(np.dot(y.T,K2),y) \
#                       -0.5*np.log(np.linalg.det(K)) \
#                       -n*0.5*np.log(2*np.pi) 
        return log_like

    def minus_log_likelihood(self, a, y):
        return - self.log_likelihood(a, y)


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