# -*- coding: utf-8 -*-
import cov_matrix
import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError

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
    K=cov_matrix.build_bigmatrix(a, x, y, yerr)
    try:
        L1 = cho_factor(K)
        sol = cho_solve(L1, y)
        n = y.size
        log_like = -0.5*np.dot(y, sol) \
                  - np.sum(np.log(np.diag(L1[0]))) \
                  - n*0.5*np.log(2*np.pi)        
    except LinAlgError:
        return -np.inf
    
    return log_like    
    


