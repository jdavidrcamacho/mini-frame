# -*- coding: utf-8 -*-
import cov_matrix
import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError

def likelihood(kern, a, x, y, yerr):    
    """
        likelihood() calculates the marginal log likelihood.
    
        Parameters
    kern = kernel being used; 
            1 for squared exponential
            2 for quasi periodic
    a = array with the important parameters
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments 

        Returns
    log_like = marginal log likelihood
    """
    listofkerns = [1, 2]
    if not kern in listofkerns:    
        print('Invalid kernel')
        raise SystemExit
        
    K=cov_matrix.build_bigmatrix(kern, a, x, y, yerr)
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

    
def inv_ll(a, x, y, yerr, kern):    
    """
        inv_ll() calculates the minus marginal log likelihood,
    to be used with scipy.optimize.minimize
    
        Parameters
    a = array with the important parameters
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments 
    kern = kernel being used; 
            1 for squared exponential
            2 for quasi periodic
    
        Returns
    log_like = marginal log likelihood
    """
    listofkerns = [1, 2]
    if not kern in listofkerns:    
        print('Invalid kernel value!')
        print('Choose: \n 1 for squared exponential \n 2 for quasi periodic')
        print
        raise SystemExit

    K=cov_matrix.build_bigmatrix(kern, a, x, y, yerr)
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

