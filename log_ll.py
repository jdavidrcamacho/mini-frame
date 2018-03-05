# -*- coding: utf-8 -*-
import cov_matrix
import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError

def likelihood(kern, a, x, y, yerr):    
    """
        likelihood() calculates the marginal log likelihood.
    On it we consider the mean function to be zero.
    
        Parameters
    kern = kernel being used: 
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
    kern = kernel being used:
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


def kepler_likelihood(kern, a, x, y, yerr):
    """
        kepler_likelihood() calculates the marginal log likelihood.
    On it we consider the mean function to be given by a 
    keplerian function.
    
        Parameters
    a = array with the important parameters ATTENTION that in this funcion
    this array will also contain the parameters P (period in days), 
    e (eccentricity), K (RV amplitude), and w (longitude of the periastron) 
    of the planet.
    kern = kernel being used: 
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
        print('Invalid kernel value!')
        print('Choose: \n 1 for squared exponential \n 2 for quasi periodic')
        print
        raise SystemExit

    lp, le, p, vc, vr, lc, bc, br, Pk, e, K, w = a
    T=0 #T = zero phase
    #w=np.pi #w = longitude of the periastron
    
    new_a = np.array([lp,le,p,vc,vr,lc,bc,br])
    
    #mean anomaly
    Mean_anom=[2*np.pi*(x1-T)/Pk  for x1 in x]
    
    #eccentric anomaly -> E0=M + e*sin(M) + 0.5*(e**2)*sin(2*M)
    E0=[x1 + e*np.sin(x1)  + 0.5*(e**2)*np.sin(2*x1) for x1 in Mean_anom]
    
    #mean anomaly -> M0=E0 - e*sin(E0)
    M0=[x1 - e*np.sin(x1) for x1 in E0]
    
    i=0
    while i<100:
        #[x + y for x, y in zip(first, second)]
        calc_aux=[x2-y2 for x2,y2 in zip(Mean_anom,M0)]    
        E1=[x3 + y3/(1-e*np.cos(x3)) for x3,y3 in zip(E0,calc_aux)]
        M1=[x4 - e*np.sin(x4) for x4 in E0]   
        i+=1
        E0=E1
        M0=M1
    nu=[2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(x5/2)) for x5 in E0]
    RV=[K*(e*np.cos(w)+np.cos(w+x6)) for x6 in nu]
    #RV=[x7 for x7 in RV] #in m/s 
 
    K=cov_matrix.build_bigmatrix(kern, new_a, x, y, yerr)
    K = K + yerr**2*np.identity(len(x))   
    L1 = cho_factor(K)
    new_y = np.array(y) - np.array(RV) #to include the keplerian function   
    sol = cho_solve(L1, new_y)
    n = new_y.size
    log_like = -0.5*np.dot(new_y, sol) \
              - np.sum(np.log(np.diag(L1[0]))) \
              - n*0.5*np.log(2*np.pi)        
    return log_like    
