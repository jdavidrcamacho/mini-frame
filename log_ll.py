# -*- coding: utf-8 -*-
import covariance_matrix
import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError
import matplotlib.pylab as pl


def likelihood(a, x, y, yerr):    
    """
        likelihood() calculates the marginal log likelihood.
    
        Parameters
    a = array with the important parameters
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments 

        Returns
    log_like = marginal log likelihood
    """
    K=covariance_matrix.build_bigmatrix(a, x, y, yerr)
    try:
        L1 = cho_factor(K)
        sol = cho_solve(L1, y)
        n = y.size
        log_like = -0.5*np.dot(y, sol) \
                  - np.sum(np.log(np.diag(L1[0]))) \
                  - n*0.5*np.log(2*np.pi)        
    except LinAlgError:
        return -np.inf
    
    return -log_like    
    



#np.random.seed(28022018)
#x=np.array([2,3,4,5,6])
#y=10*np.random.randn(15)
#yerr=1e-3*np.ones(15)

#a = np.array([l, Vc, Vr, Lc, Bc, Br])
a = np.array([0.1,1,0,0,0,0])
t,rv,rvyerr, bis, rhk, sig_rhk =np.loadtxt("dados.rdb",skiprows=2,unpack=True, usecols=(0,1,2,5,9,10))
t=t-t[0]
y=np.hstack((rv,rhk,bis))
yerr=np.hstack((rvyerr,sig_rhk,2*rvyerr))
k = covariance_matrix.build_bigmatrix(a,t,y,yerr)
#pl.imshow(k)
#pl.show()
print likelihood(a,t,y,yerr)